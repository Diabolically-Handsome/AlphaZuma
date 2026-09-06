"""Supervised macro-decision distillation onto the park-settle interface.

Pivot rationale (verified 2026-08-17): the teacher through the macro
park-settle protocol wins 50/55 with 99.8% shot execution
(``probe_alphazuma_55_macro_teacher_v1``) while five macro-PPO runs
failed to discover or consolidate wins by exploration.  This trainer
supervises a MACRO-NATIVE student directly on the teacher's macro
decisions collected by
``tools/collect_alphazuma_55_macro_decisions_v1.py`` -- dense signal at
every decision point, no exploration lottery.

Model container (exactly the transplant tool's construction,
``build_alphazuma_55_macro_policy_bootstrap_v1``):

* action space ``MultiDiscrete([3, 180])`` -- the park-settle wrapper's
  own interface (wait_hold / fire / swap);
* observation space: the stacked ``Box`` of ``len(stack_lags)``
  feature-axis frames (``zuma_rl.observation_stack_wrapper``);
* feature extractor: ``StackedRevengeEntityFeatureExtractor`` imported
  from ``tools.distill_alphazuma_55_park_settle_v3`` -- the module path
  the saved zip's pickled ``policy_kwargs`` must reference so plain
  ``MaskablePPO.load`` resolves it (enforced at build time);
* aim head: v1's polar identity anchor
  (``initialize_polar_aim_head``), the proven fresh-student init;
* value head: FRESH -- SB3's own orthogonal initialization seeded by
  ``model_seed``.  The supervised loss never touches the value pathway
  (no value gradient exists), so the saved zip carries an untrained
  critic; any later PPO warm start must re-estimate it.

Losses (the winning-precedent shape, nothing exotic):

* verb: cross-entropy over logits masked by the wrapper's own macro
  mask (recorded per decision by the collector; the adapter guarantees
  every recorded verb is mask-legal, and the loader refuses a dataset
  where that does not hold);
* aim: PLAIN categorical cross-entropy over the 180 bins on ALL
  decisions -- wait_hold decisions carry the teacher's parked target, so
  every row supervises aim.  No smoothing: ``circular_smoothed`` losses
  were rejected by the preregistered offline screens, so only plain
  categorical CE exists here, matching that precedent.
* weights configurable, default 1.0 / 1.0;
* per-verb class weights (``--verb-class-weights WAIT_HOLD FIRE SWAP``,
  default None = unweighted): the park-settle v1 reweighting hook --
  the tuple becomes ``torch.nn.functional.cross_entropy``'s ``weight=``
  tensor inside the verb CE, the only reweighting (batches stay uniform
  shuffled samples, no verb rebalancing), recorded in the
  config.json/completion.json receipts.  Fire decisions are ~8% of the
  data and the unweighted argmax student under-fires in closed loop;
  this hook is the proven fix's shape.

Per-epoch HOLDOUT metrics (split by episode, default fraction 0.2):
verb accuracy; aim exact / within-3 overall AND on fire decisions
separately; fire-decision recall and precision; and ALWAYS (no flag) a
``fire_rate_calibration`` block -- the deterministic (masked argmax)
fire rate per macro decision: the student's predicted fire fraction
next to the teacher's fire fraction plus their ratio, so
verb-class-weight sweeps are gradeable offline without env evals.
Defaults: 20 epochs, batch 512, lr 3e-4 (fresh supervised training --
deliberately higher than the 3e-5/1e-5 PPO warm-start rates),
checkpoint every 4 epochs.

The saved ``final_model.zip`` loads and runs UNCHANGED in
``tools/probe_alphazuma_55_macro_native_eval_v1.py`` (its
``require_macro_native_model`` accepts the interface and the stacked
env factory matches the collection stack); the round trip is proven in
the test suite and re-checked at save time (reload + bitwise state-dict
comparison).

RAM discipline (the distill-v2 precedent on the per-episode layout):
one stacked decision observation is width x 4 bytes (68,712 floats =
274,848 bytes on the real engine), so the flagship full-collection load
(55 levels x 40 seeds, ~250-500 decisions/episode) projects far beyond
the ~94 GB WSL budget while per-level shards are only a few GB.
``load_decision_dataset`` therefore projects the observation bytes from
the manifest's own ``decision_count``/``observation_width`` receipts
BEFORE decompressing anything and picks a backing:

* ``ram`` (small loads): the original in-RAM ``np.concatenate`` path,
  REFUSED with a contract error (pointing at ``--levels`` sharding or
  memmap backing) when the projection exceeds the RAM budget;
* ``memmap`` (large loads): every episode's observations are
  decompressed ONE EPISODE AT A TIME into a single receipted raw
  float32 ``.npy`` (``ensure_decision_observations_extraction``, the
  ``distill_alphazuma_55_park_settle_v2.extract_observations`` reuse
  semantics keyed to the ordered per-episode sha256 receipts), and
  ``MacroDecisionDataset.observations`` becomes a read-only memmap
  view -- the training loop only ever touches it through per-batch row
  fancy-indexing plus ``obs_to_tensor``, which materializes one batch;
* ``auto`` (default): ``ram`` iff the projection fits
  ``ram_budget_bytes`` (default 16 GiB), else ``memmap``.

Actions, masks, and episode/tick indices always stay in RAM (small).

``formal_seed_consumption`` stays ``false`` throughout: this trainer
never steps an environment and carries no formal-selection authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
from gymnasium import spaces

from tools import collect_alphazuma_55_macro_decisions_v1 as collector
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from zuma_rl.observation_stack_wrapper import parse_stack_lags, stacked_box
from zuma_rl.park_settle_action_wrapper import (
    MACRO_FIRE,
    MACRO_VERB_NAMES,
)
from zuma_rl.revenge_features import initialize_polar_aim_head


SCRIPT_PATH = Path(__file__).resolve()
EXTRACTOR_MODULE = "tools.distill_alphazuma_55_park_settle_v3"
EXTRACTOR_CLASS_NAME = "StackedRevengeEntityFeatureExtractor"
POLICY_ARCHITECTURE = "entity_polar_stacked_macro_decisions"
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-macro-decisions-distillation"
MACRO_VERB_COUNT = len(MACRO_VERB_NAMES)
AIM_BINS = collector.AIM_BINS
MACRO_MASK_WIDTH = MACRO_VERB_COUNT + AIM_BINS
MACRO_NVEC = (MACRO_VERB_COUNT, AIM_BINS)
VERB_MASK_FILL = v1.VERB_MASK_FILL
AIM_LOSS_MODE = "categorical"
AIM_SMOOTHING = (
    "none: circular_smoothed losses were rejected by the preregistered "
    "offline screens; plain categorical CE only"
)
VALUE_HEAD_INIT = (
    "fresh: SB3's own orthogonal initialization seeded by model_seed; "
    "the supervised loss carries no value gradient, so the saved zip's "
    "critic is untrained"
)


@dataclass(frozen=True)
class MacroDecisionLossConfig:
    """Loss shape: masked verb CE plus plain categorical aim CE.

    ``verb_class_weights`` (default None = unweighted CE, the original
    behavior) is the park-settle v1 reweighting hook on the macro verbs
    (order: wait_hold, fire, swap): the tuple becomes the ``weight=``
    tensor of ``torch.nn.functional.cross_entropy``, the only,
    explicitly-reported reweighting -- batches stay uniform shuffled
    samples with no verb rebalancing.
    """

    verb_loss_weight: float = 1.0
    aim_loss_weight: float = 1.0
    verb_class_weights: tuple[float, ...] | None = None
    max_grad_norm: float = 0.5

    def validate(self) -> None:
        if float(self.verb_loss_weight) < 0.0:
            raise ValueError("verb_loss_weight must be non-negative")
        if float(self.aim_loss_weight) < 0.0:
            raise ValueError("aim_loss_weight must be non-negative")
        if float(self.verb_loss_weight) + float(self.aim_loss_weight) <= 0.0:
            raise ValueError("at least one loss weight must be positive")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.verb_class_weights is not None:
            weights = tuple(
                float(value) for value in self.verb_class_weights
            )
            if len(weights) != MACRO_VERB_COUNT:
                raise ValueError(
                    f"verb_class_weights must cover every macro verb "
                    f"({MACRO_VERB_COUNT}: {MACRO_VERB_NAMES})"
                )
            if any(value < 0.0 for value in weights):
                raise ValueError("verb_class_weights must be non-negative")
            if not any(value > 0.0 for value in weights):
                raise ValueError(
                    "at least one verb class weight must be positive"
                )


@dataclass(frozen=True)
class MacroDecisionModelConfig:
    """Stacked entity-polar macro student capacity and initialization."""

    learned_features_dim: int = 2048
    aim_head_identity_scale: float = 5.0
    model_seed: int = 99_081_690

    def validate(self) -> None:
        if int(self.learned_features_dim) < 1:
            raise ValueError("learned_features_dim must be positive")
        if float(self.aim_head_identity_scale) <= 0.0:
            raise ValueError("aim_head_identity_scale must be positive")


@dataclass(frozen=True)
class MacroDecisionTrainConfig:
    """Optimization schedule, splits, and device for one offline run."""

    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 3.0e-4
    holdout_fraction: float = 0.2
    shuffle_seed: int = 99_081_693
    device: str = "cpu"
    checkpoint_interval_epochs: int = 4
    loss: MacroDecisionLossConfig = field(
        default_factory=MacroDecisionLossConfig
    )

    def validate(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError("epochs must be positive")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= float(self.holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be within [0, 1)")
        if int(self.checkpoint_interval_epochs) < 0:
            raise ValueError("checkpoint_interval_epochs cannot be negative")
        self.loss.validate()


@dataclass
class MacroDecisionDataset:
    """Teacher macro decisions with the wrapper's own macro masks.

    ``actions[:, 1]`` is the teacher's target aim bin at EVERY decision
    (wait_hold carries the parked target), so every row supervises the
    aim head.  ``masks`` are the park-settle wrapper's ``action_masks()``
    (3 verbs + 180 always-valid aim bins); the adapter guarantees every
    recorded verb is legal under its mask, and ``validate`` enforces it.
    """

    observations: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    episode_indices: np.ndarray
    tick_indices: np.ndarray

    def validate(self) -> None:
        count = int(self.observations.shape[0])
        if count < 1:
            raise ValueError("macro decision dataset is empty")
        if self.observations.ndim != 2:
            raise ValueError("observations must be [rows, features]")
        if self.actions.shape != (count, 2):
            raise ValueError("actions must be [rows, 2]")
        if self.masks.shape != (count, MACRO_MASK_WIDTH):
            raise ValueError(
                f"masks must be [rows, {MACRO_MASK_WIDTH}] macro masks"
            )
        if self.episode_indices.shape != (count,):
            raise ValueError("episode_indices must align with rows")
        if self.tick_indices.shape != (count,):
            raise ValueError("tick_indices must align with rows")
        verbs = self.actions[:, 0]
        aims = self.actions[:, 1]
        if not (
            bool(np.all((verbs >= 0) & (verbs < MACRO_VERB_COUNT)))
            and bool(np.all((aims >= 0) & (aims < AIM_BINS)))
        ):
            raise ValueError("macro actions escape the macro interface")
        legal = self.masks[np.arange(count), verbs]
        illegal = int(np.count_nonzero(~legal))
        if illegal:
            raise ValueError(
                f"{illegal} recorded macro verbs are illegal under their "
                "own decision-point masks; the adapter guarantees "
                "legality, so this dataset is corrupt"
            )

    @property
    def sample_count(self) -> int:
        return int(self.observations.shape[0])


_DECISION_ARRAY_KEYS = (
    "observations",
    "macro_actions",
    "macro_masks",
    "tick_indices",
)

EXTRACTION_SCHEMA = f"{SCHEMA_PREFIX}-observation-extraction"
EXTRACTION_VERSION = 1
OBSERVATION_BACKING_MODES = ("auto", "ram", "memmap")
OBSERVATION_CACHE_DIRNAME = "observations_cache"
BYTES_PER_GIB = 1024**3
DEFAULT_RAM_BUDGET_GIB = 16.0
DEFAULT_RAM_BUDGET_BYTES = int(DEFAULT_RAM_BUDGET_GIB * BYTES_PER_GIB)
_OBSERVATION_ITEMSIZE = np.dtype(np.float32).itemsize


def _selection_digest(entry_infos: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic cache key over the ORDERED per-episode receipts.

    Covers each selected NPZ's sha256 and decision-row count, so any
    byte change, reorder, or different ``--levels`` filter yields a
    different cache identity -- a stale cache can never be reused
    silently.
    """

    canonical = json.dumps(
        {
            "schema": EXTRACTION_SCHEMA,
            "version": EXTRACTION_VERSION,
            "entries": [
                {"sha256": str(info["sha256"]), "rows": int(info["rows"])}
                for info in entry_infos
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def observation_cache_paths(
    cache_dir: Path, selection_digest: str
) -> tuple[Path, Path]:
    """(.npy path, receipt path) for one selection digest."""

    stem = f"decision_observations_{selection_digest[:16]}"
    return (
        Path(cache_dir) / f"{stem}.npy",
        Path(cache_dir) / f"{stem}.receipt.json",
    )


def _extract_observations_npy(
    *,
    temporary: Path,
    entry_infos: Sequence[Mapping[str, Any]],
) -> tuple[int, int, str]:
    """Stream every episode's observations into one raw float32 .npy.

    RAM is bounded by ONE episode's decompressed observations block
    (the distill-v2 extraction precedent scaled to the per-episode NPZ
    layout); the aggregate lands directly in a preallocated on-disk
    memmap.  Returns (rows, width, npy sha256) with the write handle
    CLOSED (drvfs refuses renames of open files).
    """

    total_rows = sum(int(info["rows"]) for info in entry_infos)
    if total_rows < 1:
        raise ValueError("observation extraction selects no decisions")
    temporary.unlink(missing_ok=True)
    writer: Any = None
    width: int | None = None
    offset = 0
    try:
        for info in entry_infos:
            path = Path(str(info["path"]))
            with np.load(path) as archive:
                block = np.asarray(
                    archive["observations"], dtype=np.float32
                )
            if block.ndim != 2:
                raise ValueError(
                    f"episode observations must be 2-D: {path}"
                )
            if int(block.shape[0]) != int(info["rows"]):
                raise ValueError(
                    f"episode observations rows ({int(block.shape[0])}) "
                    f"disagree with macro_actions rows "
                    f"({int(info['rows'])}): {path}"
                )
            if width is None:
                width = int(block.shape[1])
                writer = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total_rows, width),
                )
            elif int(block.shape[1]) != width:
                raise ValueError(
                    f"episode observation width {int(block.shape[1])} "
                    f"breaks the collection's width {width}: {path}"
                )
            writer[offset : offset + int(block.shape[0])] = block
            offset += int(block.shape[0])
        if writer is None or width is None or offset != total_rows:
            raise RuntimeError(
                f"observation extraction lost rows ({offset} != "
                f"{total_rows})"
            )
        writer.flush()
        writer = None  # drop the last reference: closes the mmap
        digest = legacy._sha256(temporary)
        return total_rows, int(width), digest
    except BaseException:
        writer = None
        temporary.unlink(missing_ok=True)
        raise


def ensure_decision_observations_extraction(
    *,
    manifest_path: Path,
    entry_infos: Sequence[Mapping[str, Any]],
    cache_dir: Path | None = None,
    verify_npy_sha256: bool = True,
) -> dict[str, Any]:
    """Ensure a receipted raw-``.npy`` twin of the selected observations.

    Distill-v2's honest reuse semantics
    (``distill_alphazuma_55_park_settle_v2.extract_observations``) on
    the per-episode layout: returns ``{"action": "extracted" |
    "reused", "npy_path": ..., "receipt_path": ..., "receipt": ...}``.
    An existing cache is REUSED only when its receipt's schema,
    selection digest, byte size, and (with ``verify_npy_sha256``) .npy
    sha256 all still verify; any mismatch, an unreceipted .npy, or an
    orphaned receipt is refused with the reason -- never silently
    re-extracted over.
    """

    manifest_path = Path(manifest_path)
    directory = (
        Path(cache_dir)
        if cache_dir is not None
        else manifest_path.parent / OBSERVATION_CACHE_DIRNAME
    )
    directory.mkdir(parents=True, exist_ok=True)
    digest = _selection_digest(entry_infos)
    npy_path, receipt_path = observation_cache_paths(directory, digest)

    if npy_path.exists():
        if not receipt_path.exists():
            raise ValueError(
                f"unreceipted observations extraction already exists; "
                f"refusing to trust or overwrite it (delete it to "
                f"re-extract): {npy_path}"
            )
        receipt = legacy._read_json(receipt_path)
        problems: list[str] = []
        if receipt.get("schema") != EXTRACTION_SCHEMA:
            problems.append(
                f"receipt schema is {receipt.get('schema')!r}, "
                f"expected {EXTRACTION_SCHEMA!r}"
            )
        selection = receipt.get("selection")
        extracted = receipt.get("observations_npy")
        selection = selection if isinstance(selection, dict) else {}
        extracted = extracted if isinstance(extracted, dict) else {}
        if selection.get("digest") != digest:
            problems.append(
                f"receipt selection digest {selection.get('digest')!r} "
                f"is not this selection's {digest!r}"
            )
        actual_bytes = int(npy_path.stat().st_size)
        if int(extracted.get("bytes", -1)) != actual_bytes:
            problems.append(
                f".npy size changed (receipt {extracted.get('bytes')}, "
                f"disk {actual_bytes})"
            )
        elif verify_npy_sha256:
            npy_digest = legacy._sha256(npy_path)
            if extracted.get("sha256") != npy_digest:
                problems.append(
                    f".npy sha256 changed (receipt "
                    f"{extracted.get('sha256')}, disk {npy_digest})"
                )
        if problems:
            listing = "\n  ".join(problems)
            raise ValueError(
                f"observation extraction receipt mismatch for "
                f"{npy_path}; refusing to reuse or overwrite:\n  "
                f"{listing}"
            )
        return {
            "action": "reused",
            "npy_path": str(npy_path),
            "receipt_path": str(receipt_path),
            "receipt": receipt,
        }

    if receipt_path.exists():
        raise ValueError(
            f"stale extraction receipt without its .npy; refusing to "
            f"overwrite it (delete it to re-extract): {receipt_path}"
        )

    temporary = npy_path.with_name(f".{npy_path.name}.{os.getpid()}.tmp")
    rows, width, npy_digest = _extract_observations_npy(
        temporary=temporary, entry_infos=entry_infos
    )
    os.replace(temporary, npy_path)
    receipt = {
        "schema": EXTRACTION_SCHEMA,
        "version": EXTRACTION_VERSION,
        "created_utc": legacy._utc_now(),
        "extractor": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "selection": {
            "digest": digest,
            "manifest_path": str(manifest_path),
            "episodes": len(entry_infos),
            "entries": [
                {
                    "path": str(info["path"]),
                    "sha256": str(info["sha256"]),
                    "rows": int(info["rows"]),
                    "row_offset": int(info["row_offset"]),
                }
                for info in entry_infos
            ],
        },
        "observations_npy": {
            "path": str(npy_path),
            "sha256": npy_digest,
            "bytes": int(npy_path.stat().st_size),
            "shape": [int(rows), int(width)],
            "dtype": "float32",
            "fortran_order": False,
        },
        "ram_bound": {
            "strategy": (
                "per_episode_decompression_into_preallocated_memmap"
            ),
        },
    }
    legacy._write_json_atomic(receipt_path, receipt)
    return {
        "action": "extracted",
        "npy_path": str(npy_path),
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def load_decision_dataset(
    manifest_path: Path,
    *,
    levels: Sequence[str] | None = None,
    verify_sha256: bool = True,
    observation_backing: str = "auto",
    ram_budget_bytes: int = DEFAULT_RAM_BUDGET_BYTES,
    observation_cache_dir: Path | None = None,
) -> tuple[MacroDecisionDataset, dict[str, Any]]:
    """Load a macro-decision collection manifest, verifying receipts.

    ``levels`` filters to a subset of the collection's levels (the
    per-level BC training path); an unknown or empty selection is
    refused rather than silently training on nothing.

    ``observation_backing`` governs observation residency (actions,
    masks, and indices always stay in RAM -- they are small):

    * ``"ram"``: the in-RAM concatenation path, REFUSED with a contract
      error when the manifest's own ``decision_count`` x
      ``observation_width`` projection exceeds ``ram_budget_bytes``
      (shard with ``levels``/``--levels``, switch to memmap, or raise
      the budget) -- never an OOM kill mid-run;
    * ``"memmap"``: observations are extracted once (one episode
      decompressed at a time) into a receipted raw float32 ``.npy``
      keyed to the ordered per-episode sha256 receipts
      (``ensure_decision_observations_extraction``) and the dataset's
      ``observations`` become a read-only memmap view.  Per-episode NPZ
      digests are always computed in this mode (the cache key needs
      them); ``verify_sha256`` still controls whether they are compared
      against the manifest and whether a reused cache's .npy sha256 is
      re-verified;
    * ``"auto"`` (default): ``ram`` iff the projection fits the budget,
      else ``memmap``.
    """

    requested_backing = str(observation_backing)
    if requested_backing not in OBSERVATION_BACKING_MODES:
        raise ValueError(
            f"observation_backing must be one of "
            f"{OBSERVATION_BACKING_MODES}, got {requested_backing!r}"
        )
    if int(ram_budget_bytes) < 1:
        raise ValueError("ram_budget_bytes must be positive")

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = legacy._read_json(manifest_path)
    if manifest.get("schema") != collector.MANIFEST_SCHEMA:
        raise ValueError(
            f"not a macro-decisions manifest ({collector.MANIFEST_SCHEMA}): "
            f"{manifest_path}"
        )
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest lists no episodes: {manifest_path}")
    manifest_levels = [str(level) for level in manifest.get("levels", [])]
    if levels is not None:
        requested = [str(level) for level in levels]
        if not requested:
            raise ValueError("the level filter selects nothing")
        unknown = [
            level for level in requested if level not in manifest_levels
        ]
        if unknown:
            raise ValueError(
                f"level filter {unknown} is not in this collection "
                f"(levels: {manifest_levels})"
            )
        selected_levels = [
            level for level in manifest_levels if level in set(requested)
        ]
        entries = [
            entry
            for entry in entries
            if str(entry.get("level_id")) in set(requested)
        ]
        if not entries:
            raise ValueError(
                f"the level filter {requested} matches no collected "
                f"episodes"
            )
    else:
        selected_levels = manifest_levels

    # Projection from the manifest's own receipts: nothing decompressed.
    projected_bytes: int | None = 0
    for entry in entries:
        decisions = entry.get("decision_count")
        width = entry.get("observation_width")
        if decisions is None or width is None:
            projected_bytes = None
            break
        projected_bytes += (
            int(decisions) * int(width) * _OBSERVATION_ITEMSIZE
        )
    if requested_backing == "auto":
        if projected_bytes is None:
            raise ValueError(
                "manifest entries lack decision_count/observation_width "
                "receipts, so the RAM projection is unknown; pass "
                "observation_backing='ram' or 'memmap' explicitly"
            )
        backing_mode = (
            "ram"
            if projected_bytes <= int(ram_budget_bytes)
            else "memmap"
        )
    elif requested_backing == "ram":
        if projected_bytes is not None and projected_bytes > int(
            ram_budget_bytes
        ):
            raise ValueError(
                f"projected observation bytes ({projected_bytes:,}) "
                f"exceed the RAM budget ({int(ram_budget_bytes):,}); "
                f"shard the run with --levels, use memmap backing "
                f"(--observation-backing memmap), or raise "
                f"--ram-budget-gib if this machine truly has the RAM"
            )
        backing_mode = "ram"
    else:
        backing_mode = "memmap"
    load_observations_in_ram = backing_mode == "ram"

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    tick_indices: list[np.ndarray] = []
    entry_receipts: list[dict[str, Any]] = []
    extraction_infos: list[dict[str, Any]] = []
    row_offset = 0
    for ordinal, entry in enumerate(entries):
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve(strict=True)
        digest: str | None = None
        if verify_sha256 or backing_mode == "memmap":
            digest = legacy._sha256(path)
        if verify_sha256 and digest != entry.get("sha256"):
            raise ValueError(
                f"episode bytes differ from the manifest receipt: {path}"
            )
        with np.load(path) as archive:
            missing = [
                key
                for key in _DECISION_ARRAY_KEYS
                if key not in archive.files
            ]
            if missing:
                raise ValueError(f"decision NPZ misses {missing}: {path}")
            # In memmap mode the observations member is deliberately
            # NOT indexed here: archive["observations"] decompresses
            # the whole block, and the point is to touch one episode's
            # block only once, inside the extraction step.
            if load_observations_in_ram:
                episode_observations = np.asarray(
                    archive["observations"], dtype=np.float32
                )
            episode_actions = np.asarray(
                archive["macro_actions"], dtype=np.int64
            )
            episode_masks = np.asarray(
                archive["macro_masks"], dtype=np.bool_
            )
            episode_ticks = np.asarray(
                archive["tick_indices"], dtype=np.int64
            )
        rows = int(episode_actions.shape[0])
        aligned: list[tuple[str, np.ndarray]] = [
            ("macro_masks", episode_masks),
            ("tick_indices", episode_ticks),
        ]
        if load_observations_in_ram:
            aligned.append(("observations", episode_observations))
        for name, value in aligned:
            if int(value.shape[0]) != rows:
                raise ValueError(f"episode array {name} misaligned: {path}")
        if load_observations_in_ram:
            observations.append(episode_observations)
        else:
            extraction_infos.append(
                {
                    "path": str(path),
                    "sha256": str(digest),
                    "rows": rows,
                    "row_offset": row_offset,
                }
            )
        actions.append(episode_actions)
        masks.append(episode_masks)
        episode_indices.append(np.full(rows, ordinal, dtype=np.int64))
        tick_indices.append(episode_ticks)
        row_offset += rows
        entry_receipts.append(
            {
                "path": str(path),
                "sha256": str(entry.get("sha256")),
                "level_id": str(entry.get("level_id")),
                "seed": int(entry.get("seed", -1)),
                "outcome": str(entry.get("outcome", "")),
                "teacher_won": bool(entry.get("teacher_won", False)),
                "decisions": rows,
            }
        )

    backing_receipt: dict[str, Any] = {
        "mode": backing_mode,
        "requested": requested_backing,
        "projected_observation_bytes": projected_bytes,
        "ram_budget_bytes": int(ram_budget_bytes),
    }
    if load_observations_in_ram:
        observation_matrix = np.concatenate(observations, axis=0)
    else:
        extraction = ensure_decision_observations_extraction(
            manifest_path=manifest_path,
            entry_infos=extraction_infos,
            cache_dir=observation_cache_dir,
            verify_npy_sha256=bool(verify_sha256),
        )
        npy_path = Path(extraction["npy_path"]).resolve(strict=True)
        memmap = np.load(npy_path, mmap_mode="r")
        if memmap.ndim != 2 or memmap.dtype != np.float32:
            raise ValueError(
                f"extracted observations must be 2-D float32: {npy_path}"
            )
        if int(memmap.shape[0]) != row_offset:
            raise ValueError(
                f"extracted observations rows ({int(memmap.shape[0])}) "
                f"disagree with the NPZ decision rows ({row_offset}): "
                f"{npy_path}"
            )
        observation_matrix = np.asarray(memmap)
        if (
            observation_matrix.base is not memmap
            and observation_matrix is not memmap
        ):
            raise RuntimeError(
                "np.asarray(memmap) stopped returning a zero-copy view; "
                "the memmap loading contract is broken"
            )
        backing_receipt.update(
            {
                "storage": "raw_npy_memmap_read_only",
                "observations_npy": dict(
                    extraction["receipt"]["observations_npy"]
                ),
                "extraction": {
                    "action": extraction["action"],
                    "receipt_path": extraction["receipt_path"],
                    "selection_digest": (
                        extraction["receipt"]["selection"]["digest"]
                    ),
                },
            }
        )

    dataset = MacroDecisionDataset(
        observations=observation_matrix,
        actions=np.concatenate(actions, axis=0),
        masks=np.concatenate(masks, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        tick_indices=np.concatenate(tick_indices, axis=0),
    )
    dataset.validate()
    fire_rows = int(np.count_nonzero(dataset.actions[:, 0] == MACRO_FIRE))
    receipt = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": legacy._sha256(manifest_path),
            "schema": manifest.get("schema"),
            "version": manifest.get("version"),
        },
        "sha256_verified": bool(verify_sha256),
        "observation_backing": backing_receipt,
        "collection_levels": manifest_levels,
        "levels": selected_levels,
        "level_filter": (
            [str(level) for level in levels] if levels is not None else None
        ),
        "stack_lags": [
            int(lag) for lag in manifest.get("stack_lags", [])
        ],
        "hold_ticks": int(manifest.get("hold_ticks", 0)),
        "teacher_policy_id": manifest.get("teacher_policy_id"),
        "entries": entry_receipts,
        "episodes": len(entry_receipts),
        "teacher_wins": sum(
            1 for entry in entry_receipts if entry["teacher_won"]
        ),
        "teacher_losses_flagged": sum(
            1 for entry in entry_receipts if not entry["teacher_won"]
        ),
        "sample_count": dataset.sample_count,
        "fire_decisions": fire_rows,
        "fire_decision_fraction": fire_rows / dataset.sample_count,
    }
    return dataset, receipt


def split_holdout_by_episode(
    dataset: MacroDecisionDataset,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Episode-level split (v1's split semantics on macro decisions)."""

    if not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be within [0, 1)")
    count = dataset.sample_count
    units = np.unique(dataset.episode_indices)
    if float(holdout_fraction) == 0.0 or units.size < 2:
        train = np.arange(count, dtype=np.int64)
        holdout = np.empty(0, dtype=np.int64)
        holdout_units = np.empty(0, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        permuted = rng.permutation(units)
        chosen = int(round(float(holdout_fraction) * units.size))
        chosen = min(max(chosen, 1), units.size - 1)
        holdout_units = permuted[:chosen]
        holdout_mask = np.isin(dataset.episode_indices, holdout_units)
        holdout = np.nonzero(holdout_mask)[0].astype(np.int64)
        train = np.nonzero(~holdout_mask)[0].astype(np.int64)
    return train, holdout, {
        "mode": "episode",
        "holdout_fraction": float(holdout_fraction),
        "seed": int(seed),
        "total_units": int(units.size),
        "holdout_units": int(holdout_units.size),
        "train_rows": int(train.size),
        "holdout_rows": int(holdout.size),
    }


def macro_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array(MACRO_NVEC, dtype=np.int64))


def build_macro_student_model(
    *,
    observation_space: spaces.Box,
    policy_kwargs: dict[str, Any],
    model_config: MacroDecisionModelConfig,
    train_config: MacroDecisionTrainConfig,
) -> Any:
    """Fresh macro-native stacked MaskablePPO container.

    Mirrors the transplant tool's construction: interface-only env over
    the stacked Box and the macro ``MultiDiscrete([3, 180])``, the
    distill-v3 stacked extractor (module-path enforced so the pickle
    resolves), the v1 polar aim-head identity anchor, and a FRESH value
    head (SB3's own init; the supervised loss never touches it).  The
    stored PPO fields (n_steps/gamma/...) are inert for this trainer.
    """

    from sb3_contrib import MaskablePPO

    model_config.validate()
    train_config.validate()
    extractor_class = policy_kwargs.get("features_extractor_class")
    if (
        getattr(extractor_class, "__module__", None) != EXTRACTOR_MODULE
        or getattr(extractor_class, "__name__", None) != EXTRACTOR_CLASS_NAME
    ):
        raise ValueError(
            "macro-native students must pickle-reference "
            f"{EXTRACTOR_MODULE}.{EXTRACTOR_CLASS_NAME} so plain "
            "MaskablePPO.load resolves the extractor; got "
            f"{extractor_class!r}"
        )
    environment = v1._SpaceOnlyMaskableEnv(
        observation_space, macro_action_space()
    )
    model = MaskablePPO(
        "MlpPolicy",
        environment,
        learning_rate=float(train_config.learning_rate),
        n_steps=512,
        batch_size=int(train_config.batch_size),
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        seed=int(model_config.model_seed),
        device=str(train_config.device),
        verbose=0,
    )
    initialize_polar_aim_head(
        model, scale=float(model_config.aim_head_identity_scale)
    )
    return model


def _train_macro_batch(
    *,
    model: Any,
    dataset: MacroDecisionDataset,
    rows: np.ndarray,
    loss: MacroDecisionLossConfig,
) -> dict[str, Any]:
    """One optimizer step of the macro-decision objective.

    Verb: CE over macro-mask-filled logits (every label is mask-legal
    by dataset validation), optionally per-verb class-weighted
    (``loss.verb_class_weights`` -> ``cross_entropy(weight=...)``, the
    park-settle v1 hook).  Aim: plain categorical CE on all rows.
    """

    import torch
    import torch.nn.functional as functional

    rows = np.asarray(rows, dtype=np.int64)
    observations = dataset.observations[rows]
    actions = dataset.actions[rows]
    masks = dataset.masks[rows]

    observation_tensor, _ = model.policy.obs_to_tensor(observations)
    action_tensor = torch.as_tensor(
        actions, dtype=torch.long, device=model.device
    )
    verb_mask_tensor = torch.as_tensor(
        masks[:, :MACRO_VERB_COUNT], dtype=torch.bool, device=model.device
    )

    model.policy.set_training_mode(True)
    verb_logits, aim_logits = v1._policy_logits(model, observation_tensor)
    masked_verb_logits = verb_logits.masked_fill(
        ~verb_mask_tensor, VERB_MASK_FILL
    )
    class_weight = None
    if loss.verb_class_weights is not None:
        class_weight = torch.as_tensor(
            loss.verb_class_weights,
            dtype=verb_logits.dtype,
            device=model.device,
        )
    verb_loss = functional.cross_entropy(
        masked_verb_logits, action_tensor[:, 0], weight=class_weight
    )
    aim_loss = functional.cross_entropy(aim_logits, action_tensor[:, 1])
    total_loss = (
        loss.verb_loss_weight * verb_loss + loss.aim_loss_weight * aim_loss
    )

    model.policy.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.policy.parameters(), float(loss.max_grad_norm)
    )
    model.policy.optimizer.step()
    model.policy.set_training_mode(False)

    with torch.no_grad():
        verb_after, aim_after = v1._policy_logits(model, observation_tensor)
        masked_after = verb_after.masked_fill(
            ~verb_mask_tensor, VERB_MASK_FILL
        )
        predicted_verbs = masked_after.argmax(dim=1).detach().cpu().numpy()
        predicted_aims = aim_after.argmax(dim=1).detach().cpu().numpy()
    distance = v1._circular_bin_distance(predicted_aims, actions[:, 1])
    return {
        "loss": float(total_loss.detach().cpu()),
        "verb_loss": float(verb_loss.detach().cpu()),
        "aim_loss": float(aim_loss.detach().cpu()),
        "gradient_norm": float(
            torch.as_tensor(gradient_norm).detach().cpu()
        ),
        "sample_count": int(len(rows)),
        "verb_accuracy": float(np.mean(predicted_verbs == actions[:, 0])),
        "aim_exact_accuracy": float(np.mean(distance == 0)),
        "aim_within_three_accuracy": float(np.mean(distance <= 3)),
        "masked_argmax_all_mask_legal": bool(
            np.all(masks[np.arange(len(rows)), predicted_verbs])
        ),
    }


def _aim_group(distance: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(group))
    if not count:
        return {
            "sample_count": 0,
            "aim_exact_accuracy": None,
            "aim_within_three_accuracy": None,
        }
    grouped = distance[group]
    return {
        "sample_count": count,
        "aim_exact_accuracy": float(np.mean(grouped == 0)),
        "aim_within_three_accuracy": float(np.mean(grouped <= 3)),
    }


def evaluate_macro_decisions(
    *,
    model: Any,
    dataset: MacroDecisionDataset,
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Deterministic-policy metrics on one slice.

    Masked verb argmax plus plain aim argmax: verb accuracy, aim exact /
    within-3 overall AND on fire decisions separately, and
    fire-decision recall/precision (None on empty denominators).

    ``fire_rate_calibration`` is ALWAYS reported (no flag): the
    deterministic fire rate per macro decision -- the student's
    predicted fire fraction next to the teacher's fire fraction on the
    same slice, plus their ratio (None on a fire-free slice) -- so
    verb-class-weight sweeps are gradeable offline without env evals.
    """

    import torch

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("evaluation slice is empty")
    predicted_verbs = np.empty(indices.size, dtype=np.int64)
    predicted_aims = np.empty(indices.size, dtype=np.int64)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, indices.size, int(batch_size)):
            rows = indices[start : start + int(batch_size)]
            observation_tensor, _ = model.policy.obs_to_tensor(
                dataset.observations[rows]
            )
            verb_logits, aim_logits = v1._policy_logits(
                model, observation_tensor
            )
            verb_mask_tensor = torch.as_tensor(
                dataset.masks[rows, :MACRO_VERB_COUNT],
                dtype=torch.bool,
                device=model.device,
            )
            masked_verbs = verb_logits.masked_fill(
                ~verb_mask_tensor, VERB_MASK_FILL
            )
            stop = start + len(rows)
            predicted_verbs[start:stop] = (
                masked_verbs.argmax(dim=1).detach().cpu().numpy()
            )
            predicted_aims[start:stop] = (
                aim_logits.argmax(dim=1).detach().cpu().numpy()
            )
    actions = dataset.actions[indices]
    masks = dataset.masks[indices]
    distance = v1._circular_bin_distance(predicted_aims, actions[:, 1])
    teacher_fire = actions[:, 0] == MACRO_FIRE
    student_fire = predicted_verbs == MACRO_FIRE
    true_positive = int(np.sum(teacher_fire & student_fire))
    teacher_fire_count = int(np.sum(teacher_fire))
    student_fire_count = int(np.sum(student_fire))
    return {
        "sample_count": int(indices.size),
        "deterministic_policy": (
            "masked_verb_argmax_plus_plain_aim_argmax"
        ),
        "deterministic_verbs_all_mask_legal": bool(
            np.all(masks[np.arange(indices.size), predicted_verbs])
        ),
        "verb_accuracy": float(
            np.mean(predicted_verbs == actions[:, 0])
        ),
        "aim_exact_accuracy": float(np.mean(distance == 0)),
        "aim_within_three_accuracy": float(np.mean(distance <= 3)),
        "fire_decisions": _aim_group(distance, teacher_fire),
        "teacher_fire_decisions": teacher_fire_count,
        "student_fire_predictions": student_fire_count,
        "fire_recall": (
            true_positive / teacher_fire_count
            if teacher_fire_count
            else None
        ),
        "fire_precision": (
            true_positive / student_fire_count
            if student_fire_count
            else None
        ),
        "fire_rate_calibration": {
            "policy": "deterministic_masked_verb_argmax",
            "student_fire_fraction": student_fire_count / indices.size,
            "teacher_fire_fraction": teacher_fire_count / indices.size,
            "student_to_teacher_fire_ratio": (
                student_fire_count / teacher_fire_count
                if teacher_fire_count
                else None
            ),
        },
    }


def optimize_macro_decisions(
    *,
    model: Any,
    dataset: MacroDecisionDataset,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    train_config: MacroDecisionTrainConfig,
    run_dir: Path | None = None,
    status_callback: (
        Callable[[int, list[dict[str, Any]], list[dict[str, Any]]], None]
        | None
    ) = None,
) -> dict[str, Any]:
    """Uniform shuffled minibatch epochs with per-epoch holdout metrics."""

    train_config.validate()
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.size == 0:
        raise RuntimeError("macro-decision training split is empty")
    holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
    evaluation_indices = (
        holdout_indices if holdout_indices.size else train_indices
    )
    evaluation_slice = "holdout" if holdout_indices.size else (
        "train_no_holdout"
    )
    rng = np.random.default_rng(int(train_config.shuffle_seed) + 104_729)
    batch_size = int(train_config.batch_size)
    updates_per_epoch = math.ceil(train_indices.size / batch_size)
    epoch_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for epoch in range(int(train_config.epochs)):
        order = rng.permutation(train_indices)
        batches: list[dict[str, Any]] = []
        for start in range(0, order.size, batch_size):
            metrics = _train_macro_batch(
                model=model,
                dataset=dataset,
                rows=order[start : start + batch_size],
                loss=train_config.loss,
            )
            metrics["batch_index"] = start // batch_size
            batches.append(metrics)
        holdout_metrics = evaluate_macro_decisions(
            model=model,
            dataset=dataset,
            indices=evaluation_indices,
            batch_size=batch_size,
        )
        holdout_metrics["slice"] = evaluation_slice
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "updates": len(batches),
                "mean_loss": float(
                    np.mean([row["loss"] for row in batches])
                ),
                "mean_verb_loss": float(
                    np.mean([row["verb_loss"] for row in batches])
                ),
                "mean_aim_loss": float(
                    np.mean([row["aim_loss"] for row in batches])
                ),
                "mean_verb_accuracy": float(
                    np.mean([row["verb_accuracy"] for row in batches])
                ),
                "mean_aim_exact_accuracy": float(
                    np.mean([row["aim_exact_accuracy"] for row in batches])
                ),
                "mean_aim_within_three_accuracy": float(
                    np.mean(
                        [
                            row["aim_within_three_accuracy"]
                            for row in batches
                        ]
                    )
                ),
                "last_batch": batches[-1],
                "holdout": holdout_metrics,
            }
        )
        interval = int(train_config.checkpoint_interval_epochs)
        if (
            run_dir is not None
            and interval > 0
            and (epoch + 1) % interval == 0
        ):
            checkpoint = run_dir / f"epoch_{epoch + 1:02d}_model.zip"
            model.save(checkpoint)
            checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "path": str(checkpoint),
                    "sha256": legacy._sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                }
            )
        if status_callback is not None:
            status_callback(epoch + 1, epoch_rows, checkpoints)
    return {
        "updates": sum(int(row["updates"]) for row in epoch_rows),
        "samples": int(train_indices.size),
        "updates_per_epoch": updates_per_epoch,
        "evaluation_slice": evaluation_slice,
        "epochs": epoch_rows,
        "checkpoints": checkpoints,
        "sampling": "uniform_shuffle",
        "aim_loss_scope": "all_decisions_wait_hold_carries_parked_target",
        "aim_loss_mode": AIM_LOSS_MODE,
        "aim_loss_smoothing": AIM_SMOOTHING,
    }


def _verify_round_trip(model: Any, final_model: Path) -> dict[str, Any]:
    """The saved zip must reload into a bitwise-equal macro-native policy.

    The reload resolves the pickled extractor class from
    ``tools.distill_alphazuma_55_park_settle_v3`` exactly the way
    ``probe_alphazuma_55_macro_native_eval_v1`` will.
    """

    import torch
    from sb3_contrib import MaskablePPO

    reloaded = MaskablePPO.load(str(final_model), device="cpu")
    nvec = tuple(
        int(value)
        for value in np.asarray(reloaded.action_space.nvec).reshape(-1)
    )
    if nvec != MACRO_NVEC:
        raise RuntimeError(
            f"round trip changed the action space: {nvec} != {MACRO_NVEC}"
        )
    if tuple(reloaded.observation_space.shape) != tuple(
        model.observation_space.shape
    ):
        raise RuntimeError("round trip changed the observation space")
    live = {
        key: value.detach().cpu()
        for key, value in model.policy.state_dict().items()
    }
    for key, value in reloaded.policy.state_dict().items():
        if not torch.equal(value.detach().cpu(), live[key]):
            raise RuntimeError(f"round trip drifted: {key}")
    extractor = reloaded.policy.features_extractor
    if (
        type(extractor).__module__ != EXTRACTOR_MODULE
        or type(extractor).__name__ != EXTRACTOR_CLASS_NAME
    ):
        raise RuntimeError(
            "round trip resolved a foreign extractor class: "
            f"{type(extractor)!r}"
        )
    return {
        "reloaded_action_nvec": list(nvec),
        "bitwise_state_dict_equal": True,
        "extractor_class_resolved_from": EXTRACTOR_MODULE,
        "loadable_by_macro_native_eval_v1": True,
    }


def execute_macro_training(
    *,
    model: Any,
    dataset: MacroDecisionDataset,
    run_dir: Path,
    model_config: MacroDecisionModelConfig,
    train_config: MacroDecisionTrainConfig,
    dataset_receipt: dict[str, Any] | None = None,
    stack_lags: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Offline macro-decision training with a receipted completion."""

    model_config.validate()
    train_config.validate()
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"macro-decision run directory exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    if dataset_receipt is None:
        dataset_receipt = {
            "source": "in_memory",
            "sample_count": dataset.sample_count,
        }
    lags = (
        [int(lag) for lag in parse_stack_lags(stack_lags)]
        if stack_lags is not None
        else None
    )
    extractor_path = Path(trainer_v3.__file__).resolve()
    legacy._write_json_atomic(
        run_dir / "config.json",
        v1._jsonable(
            {
                "schema": f"{SCHEMA_PREFIX}-config",
                "version": 1,
                "created_utc": legacy._utc_now(),
                "trainer": {
                    "path": str(SCRIPT_PATH),
                    "sha256": legacy._sha256(SCRIPT_PATH),
                },
                "extractor_module": {
                    "module": EXTRACTOR_MODULE,
                    "path": str(extractor_path),
                    "sha256": legacy._sha256(extractor_path),
                    "class": EXTRACTOR_CLASS_NAME,
                },
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "stack_lags": lags,
                "dataset": dataset_receipt,
                "formal_seed_consumption": False,
                "formal_candidate_authority": False,
            }
        ),
    )
    try:
        train_indices, holdout_indices, split = split_holdout_by_episode(
            dataset,
            holdout_fraction=float(train_config.holdout_fraction),
            seed=int(train_config.shuffle_seed),
        )

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            legacy._write_json_atomic(
                run_dir / "training_status.json",
                v1._jsonable(
                    {
                        "schema": f"{SCHEMA_PREFIX}-status",
                        "version": 1,
                        "status": "RUNNING",
                        "stage": "OPTIMIZING",
                        "updated_utc": legacy._utc_now(),
                        "completed_epochs": completed_epochs,
                        "expected_epochs": int(train_config.epochs),
                        "epochs": epoch_rows,
                        "checkpoints": checkpoints,
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
            )

        write_status(0, [], [])
        optimization = optimize_macro_decisions(
            model=model,
            dataset=dataset,
            train_indices=train_indices,
            holdout_indices=holdout_indices,
            train_config=train_config,
            run_dir=run_dir,
            status_callback=write_status,
        )
        final_model = run_dir / "final_model.zip"
        model.save(final_model)
        round_trip = _verify_round_trip(model, final_model)
        completion = {
            "schema": f"{SCHEMA_PREFIX}-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": legacy._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": POLICY_ARCHITECTURE,
            "trainer": {
                "path": str(SCRIPT_PATH),
                "sha256": legacy._sha256(SCRIPT_PATH),
            },
            "extractor_module": {
                "module": EXTRACTOR_MODULE,
                "path": str(extractor_path),
                "sha256": legacy._sha256(extractor_path),
                "class": EXTRACTOR_CLASS_NAME,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": str(train_config.device),
            },
            "policy_parameter_count": sum(
                parameter.numel()
                for parameter in model.policy.parameters()
            ),
            "model_spaces": {
                "action_space_nvec": list(MACRO_NVEC),
                "observation_shape": [
                    int(value)
                    for value in model.observation_space.shape
                ],
            },
            "stack_lags": lags,
            "dataset": dataset_receipt,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "recipe": {
                "labels": (
                    "teacher macro decisions through the shared "
                    "park-settle adapter (wait_hold carries the parked "
                    "target bin)"
                ),
                "verb_loss": "cross_entropy_over_macro_mask_filled_logits",
                "aim_loss_scope": (
                    "all_decisions_wait_hold_carries_parked_target"
                ),
                "aim_loss_mode": AIM_LOSS_MODE,
                "aim_loss_smoothing": AIM_SMOOTHING,
                "loss_weights": {
                    "verb": float(train_config.loss.verb_loss_weight),
                    "aim": float(train_config.loss.aim_loss_weight),
                },
                "verb_batch_rebalancing": "none",
                "verb_class_weights": (
                    train_config.loss.verb_class_weights
                ),
                "aim_head_init": (
                    "polar identity anchor (initialize_polar_aim_head, "
                    f"scale {float(model_config.aim_head_identity_scale)})"
                ),
                "value_head_init": VALUE_HEAD_INIT,
            },
            "split": split,
            "optimization": optimization,
            "final_holdout_metrics": optimization["epochs"][-1]["holdout"],
            "calibration": optimization["epochs"][-1]["holdout"],
            "round_trip": round_trip,
            "final_model": {
                "path": str(final_model),
                "sha256": legacy._sha256(final_model),
                "bytes": final_model.stat().st_size,
            },
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        legacy._write_json_atomic(
            run_dir / "completion.json", v1._jsonable(completion)
        )
        return completion
    except BaseException as error:
        legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": f"{SCHEMA_PREFIX}-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": legacy._utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        required=True,
        type=Path,
        help=(
            "collect_alphazuma_55_macro_decisions_v1 run dir (its "
            "episodes_manifest.json is loaded with sha256 verification)"
        ),
    )
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help=(
            "train on this level subset of the collection only (the "
            "per-level BC path); default: every collected level"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-4,
        help=(
            "fresh supervised training: deliberately higher than the "
            "3e-5/1e-5 PPO warm-start rates"
        ),
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--shuffle-seed", type=int, default=99_081_693)
    parser.add_argument("--model-seed", type=int, default=99_081_690)
    parser.add_argument("--learned-features-dim", type=int, default=2048)
    parser.add_argument(
        "--aim-head-identity-scale", type=float, default=5.0
    )
    parser.add_argument("--verb-loss-weight", type=float, default=1.0)
    parser.add_argument("--aim-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--verb-class-weights",
        type=float,
        nargs=MACRO_VERB_COUNT,
        default=None,
        metavar=tuple(name.upper() for name in MACRO_VERB_NAMES),
        help=(
            "per-verb CE class weights (order: wait_hold fire swap) "
            "applied as the weight= tensor inside the verb "
            "cross-entropy, the park-settle v1 reweighting hook; "
            "default: unweighted CE (current behavior)"
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint-interval-epochs", type=int, default=4
    )
    parser.add_argument(
        "--stack-lags",
        default=None,
        help=(
            "optional cross-check; must equal the collection manifest's "
            "stack_lags (which are always the source of truth)"
        ),
    )
    parser.add_argument("--max-ticks", type=int, default=30_000)
    parser.add_argument(
        "--observation-backing",
        choices=list(OBSERVATION_BACKING_MODES),
        default="auto",
        help=(
            "observation residency: 'ram' concatenates float32 "
            "observations in memory (REFUSED above the RAM budget), "
            "'memmap' extracts them once into a receipted raw .npy and "
            "trains on a read-only memmap view (the distill-v2 "
            "precedent), 'auto' picks by the manifest's projected bytes"
        ),
    )
    parser.add_argument(
        "--ram-budget-gib",
        type=float,
        default=DEFAULT_RAM_BUDGET_GIB,
        help=(
            "in-RAM observation budget (GiB) used by auto backing and "
            "the explicit-ram refusal; the full 55-level collection "
            "projects ~150-300 GB, far beyond it, and lands on memmap"
        ),
    )
    parser.add_argument(
        "--observation-cache-dir",
        type=Path,
        default=None,
        help=(
            "where memmap extraction writes the receipted .npy cache "
            f"(default: <collection-dir>/{OBSERVATION_CACHE_DIRNAME})"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = (
        args.collection_dir.expanduser().resolve(strict=True)
        / "episodes_manifest.json"
    )
    ram_budget_bytes = int(float(args.ram_budget_gib) * BYTES_PER_GIB)
    if ram_budget_bytes < 1:
        raise SystemExit("--ram-budget-gib must be positive")
    dataset, receipt = load_decision_dataset(
        manifest_path,
        levels=args.levels,
        observation_backing=str(args.observation_backing),
        ram_budget_bytes=ram_budget_bytes,
        observation_cache_dir=(
            args.observation_cache_dir.expanduser()
            if args.observation_cache_dir is not None
            else None
        ),
    )
    lags = parse_stack_lags(receipt["stack_lags"])
    if args.stack_lags is not None:
        requested = parse_stack_lags(str(args.stack_lags))
        if requested != lags:
            raise SystemExit(
                f"--stack-lags {requested} disagrees with the collection "
                f"manifest's {lags}; the manifest is the source of truth"
            )
    if args.validate_only:
        print(
            json.dumps(
                v1._jsonable({"status": "VALID", "dataset": receipt}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if args.original_root is None or args.run_dir is None:
        parser.error(
            "--original-root and --run-dir are required for training"
        )
    stacked_width = int(dataset.observations.shape[1])
    if stacked_width % len(lags):
        raise SystemExit(
            f"dataset observation width {stacked_width} is not divisible "
            f"by {len(lags)} stack frames"
        )
    base_width = stacked_width // len(lags)

    loss_config = MacroDecisionLossConfig(
        verb_loss_weight=float(args.verb_loss_weight),
        aim_loss_weight=float(args.aim_loss_weight),
        verb_class_weights=(
            tuple(float(value) for value in args.verb_class_weights)
            if args.verb_class_weights is not None
            else None
        ),
        max_grad_norm=float(args.max_grad_norm),
    )
    train_config = MacroDecisionTrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        holdout_fraction=float(args.holdout_fraction),
        shuffle_seed=int(args.shuffle_seed),
        device=str(args.device),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        loss=loss_config,
    )
    model_config = MacroDecisionModelConfig(
        learned_features_dim=int(args.learned_features_dim),
        aim_head_identity_scale=float(args.aim_head_identity_scale),
        model_seed=int(args.model_seed),
    )
    prototype = v1._launch_prototype(
        original_root=args.original_root.expanduser().resolve(strict=True),
        max_ticks=int(args.max_ticks),
        observation_width=base_width,
    )
    try:
        policy_kwargs = trainer_v3.stacked_entity_polar_policy_kwargs(
            prototype,
            learned_features_dim=int(model_config.learned_features_dim),
            stack_lags=lags,
        )
        observation_space = stacked_box(
            prototype.observation_space, len(lags)
        )
    finally:
        prototype.close()
    if int(observation_space.shape[0]) != stacked_width:
        raise SystemExit(
            f"stacked prototype width {int(observation_space.shape[0])} "
            f"disagrees with the dataset's {stacked_width}"
        )
    model = build_macro_student_model(
        observation_space=observation_space,
        policy_kwargs=policy_kwargs,
        model_config=model_config,
        train_config=train_config,
    )
    completion = execute_macro_training(
        model=model,
        dataset=dataset,
        run_dir=args.run_dir.expanduser(),
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=lags,
    )
    print(
        json.dumps(
            v1._jsonable(
                {
                    "status": completion["status"],
                    "final_model": completion["final_model"],
                    "levels": receipt["levels"],
                    "stack_lags": completion["stack_lags"],
                    "final_holdout_metrics": completion[
                        "final_holdout_metrics"
                    ],
                    "round_trip": completion["round_trip"],
                    "formal_seed_consumption": False,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
