"""RAM-safe runner: park-settle episode collection -> v4 coverage dataset.

Lineage
-------
``tools/collect_alphazuma_55_park_settle_episodes_v1.py`` writes full-tick
teacher episodes as one compressed NPZ per episode plus an
``episodes_manifest.json`` with relative paths and sha256 receipts.
``tools/build_alphazuma_55_motor_observable_replay_v4.py`` (the frozen
coverage-equalized builder) consumes full-tick episode records, but its
``main()`` only accepts inline-JSON episodes -- ~90 bytes per observation
float, impractical at the 55x3 scale (~2M ticks, ~183 GB raw
observations).  This runner drives the builder's Python API
``build_coverage_equalized_replay_dataset`` directly from a collection
run directory and writes EXACTLY the same three output files
``builder.main()`` would (``<prefix>.npz`` + ``<prefix>.manifest.json`` +
``<prefix>.dataset.json``), so
``tools.distill_alphazuma_55_park_settle_v1.load_replay_dataset`` accepts
the result unchanged.  A ``<prefix>.completion.json`` receipt is written
additively.

Receipts before use
-------------------
Every episode NPZ is verified against its manifest sha256 receipt BEFORE
any tick is hydrated.  Verification failures are collected across the
whole manifest and refused in one error that lists every bad file --
a partially corrupted collection never silently contributes episodes.

RAM safety (two-pass, no subsampling)
-------------------------------------
The builder API genuinely needs every episode record at once (it parses
all episodes up front to compute per-level quotas), but its RETENTION
PLAN is observation-independent: fire windows come from the raw intent
verb stream, danger ticks from per-tick ``info`` fields, quotas from tick
counts, and the uniform remainder from ``sampling_seed`` /
``round_index`` / level index -- ``_parse_episode`` only ever checks the
observation's width for consistency.  The runner exploits that:

* Pass 1 (planning): every episode is hydrated WITHOUT observations --
  the real per-tick ``raw_actions`` / ``effective_actions`` /
  ``training_masks`` / ``exact_masks`` / ``intent_mask_relaxed`` and the
  builder-relevant ``info`` fields (``score``, ``chain_length``,
  ``visible_balls``: every ``danger_field_priority`` candidate the
  collector NPZ schema records, plus the score the builder reads) are
  loaded, while a shared width-1 float32 stub stands in for each
  observation (NPZ members are decompressed lazily, so the observations
  member is never touched).  The builder runs on these records and
  returns the manifest and the exact per-row retention plan
  (``CoverageReplayProvenance``): real actions, real masks, real
  statistics -- only ``dataset_sha256`` is stub-poisoned and is
  recomputed in pass 2.  Peak memory is metadata only, roughly 2.5 KB
  per tick (~5 GB at 2M ticks) instead of ~92 KB per tick (~183 GB).
* Pass 2 (gather): retained observation rows are hydrated one episode at
  a time (the plan's rows are contiguous per episode) and appended to a
  disk spill file, exactly the collector's own spill+memmap pattern.
  Peak memory is ONE episode's decompressed observations (~2.75 GB at
  30k ticks x 22,904 float32) plus the retained actions/masks
  (~200 B/row).  The aggregate NPZ is then streamed from the read-only
  memmap by ``np.savez_compressed`` in bounded buffers, and the real
  ``dataset_sha256`` is recomputed row by row THROUGH the builder's own
  ``_dataset_sha256`` so the digest stays byte-compatible.

Nothing is subsampled beyond what the builder itself retains: the rows,
the manifest statistics, and the provenance arrays are identical to a
full in-memory hydration (proven against ``builder.main()`` in the
tests).

Honest outcomes
---------------
Outputs follow the frozen-output discipline (every writer refuses to
overwrite existing files).  ``formal_seed_consumption`` stays ``false``:
this runner reshapes engineering collections and never touches an
environment or a seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping
import zipfile

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import build_alphazuma_55_motor_observable_replay_v4 as builder


SCRIPT_PATH = Path(__file__).resolve()
COLLECTOR_PATH = (
    SCRIPT_PATH.parent / "collect_alphazuma_55_park_settle_episodes_v1.py"
)
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-coverage-replay-build"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}-completion"
VERSION = 1
# Must match tools.collect_alphazuma_55_park_settle_episodes_v1
# MANIFEST_SCHEMA (duplicated so this runner imports with numpy alone,
# exactly like the builder duplicates its VERB_NAMES).
COLLECTION_MANIFEST_SCHEMA = (
    "zuma-rl.alphazuma-55-park-settle-episodes-manifest"
)
# Episode NPZ members hydrated during the planning pass; the
# ``observations`` member is deliberately absent (see module docstring).
_METADATA_KEYS = (
    "raw_actions",
    "effective_actions",
    "training_masks",
    "exact_masks",
    "intent_mask_relaxed",
    "info_score",
    "info_chain_length",
    "info_visible_balls",
)
# The builder reads per-tick info for two things only: the danger tier
# (``danger_field_priority``) and the episode score.  These are all such
# fields the collector NPZ schema records; ``human_speedrun`` motor
# fields are never read by the builder and are not hydrated.
_PLANNING_INFO_FIELDS = ("score", "chain_length", "visible_balls")
# Shared width-1 stand-in observation for the planning pass.  The
# builder only checks observation width consistency; retention never
# depends on observation values.
_STUB_OBSERVATION = np.zeros(1, dtype=np.float32)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CoverageReplayBuildRunConfig:
    """One runner invocation: collection input, output prefix, knobs."""

    collection_run_dir: Path
    output_prefix: Path
    builder_config: builder.CoverageReplayV4Config = field(
        default_factory=builder.CoverageReplayV4Config
    )
    round_index: int = 0
    # ``None`` inherits the collection manifest's ``teacher_policy_id``.
    teacher_policy_id: str | None = None

    def __post_init__(self) -> None:
        if int(self.round_index) < 0:
            raise ValueError("round_index cannot be negative")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _entry_path(manifest_dir: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def load_collection_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and structurally validate the collector's episodes manifest."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != COLLECTION_MANIFEST_SCHEMA:
        raise ValueError(
            f"not a park-settle episodes manifest "
            f"(schema {manifest.get('schema')!r}): {manifest_path}"
        )
    semantics = manifest.get("mask_semantics")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("exact_masks") != builder.EXACT_MASK_SEMANTICS
    ):
        raise ValueError(
            f"collection manifest does not declare exact_masks as "
            f"{builder.EXACT_MASK_SEMANTICS!r}; refusing to feed the "
            f"park-settle chain: {manifest_path}"
        )
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest lists no episodes: {manifest_path}")
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest entry {ordinal} is not an object")
        for key in ("level_id", "seed", "path", "sha256", "tick_count"):
            if key not in entry:
                raise ValueError(f"manifest entry {ordinal} misses {key!r}")
    return manifest, [dict(entry) for entry in entries]


def verify_episode_receipts(
    manifest_dir: Path, entries: list[dict[str, Any]]
) -> None:
    """Verify EVERY episode NPZ sha256 receipt before any use.

    All failures are collected and refused together so one pass over a
    damaged collection reports every bad file instead of the first.
    """

    problems: list[str] = []
    for entry in entries:
        relative = str(entry["path"])
        path = _entry_path(manifest_dir, entry)
        if not path.is_file():
            problems.append(f"{relative}: missing on disk")
            continue
        digest = builder._file_sha256(path)
        expected = str(entry["sha256"])
        if digest != expected:
            problems.append(
                f"{relative}: sha256 mismatch "
                f"(manifest {expected}, disk {digest})"
            )
    if problems:
        listing = "\n  ".join(problems)
        raise ValueError(
            f"{len(problems)} episode file(s) failed their manifest "
            f"sha256 receipts; refusing to build:\n  {listing}"
        )


def _peek_observation_header(path: Path) -> tuple[int, int]:
    """(tick_count, observation_width) from the NPY header only.

    Reads the zip member's NPY header without decompressing the
    observation payload, so the planning pass can validate width
    agreement and manifest alignment across ALL episodes for free.
    """

    with zipfile.ZipFile(path) as archive:
        if "observations.npy" not in archive.namelist():
            raise ValueError(f"episode NPZ misses observations: {path}")
        with archive.open("observations.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                )
            elif version in {(2, 0), (3, 0)}:
                # 3.0 only widens header field names to utf-8; the
                # binary header layout matches 2.0.
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_2_0(stream)
                )
            else:
                raise ValueError(
                    f"unsupported NPY format version {version}: {path}"
                )
    if len(shape) != 2 or fortran_order or dtype != np.float32:
        raise ValueError(
            f"observations must be C-order 2-D float32, got shape "
            f"{shape} dtype {dtype}: {path}"
        )
    return int(shape[0]), int(shape[1])


def _load_metadata_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load every planning-pass NPZ member EXCEPT the observations.

    ``np.load`` decompresses NPZ members lazily on access, so skipping
    the ``observations`` key keeps the ~92 KB/tick payload on disk.
    """

    with np.load(path) as archive:
        missing = [key for key in _METADATA_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"episode NPZ misses {missing}: {path}")
        arrays = {key: archive[key] for key in _METADATA_KEYS}
    count = int(arrays["raw_actions"].shape[0])
    for key, value in arrays.items():
        if int(value.shape[0]) != count:
            raise ValueError(f"episode array {key} misaligned: {path}")
    return arrays


def _metadata_episode_record(
    manifest_dir: Path, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Builder episode record with real metadata and stub observations."""

    path = _entry_path(manifest_dir, entry).resolve(strict=True)
    arrays = _load_metadata_arrays(path)
    count = int(arrays["raw_actions"].shape[0])
    if count != int(entry["tick_count"]):
        raise ValueError(
            f"episode tick_count receipt disagrees with the NPZ "
            f"({int(entry['tick_count'])} != {count}): {path}"
        )
    ticks = [
        {
            "observation": _STUB_OBSERVATION,
            "raw_action": arrays["raw_actions"][tick_index],
            "training_mask": arrays["training_masks"][tick_index],
            "exact_mask": arrays["exact_masks"][tick_index],
            "effective_action": arrays["effective_actions"][tick_index],
            "intent_mask_relaxed": bool(
                arrays["intent_mask_relaxed"][tick_index]
            ),
            "info": {
                "score": int(arrays["info_score"][tick_index]),
                "chain_length": int(
                    arrays["info_chain_length"][tick_index]
                ),
                "visible_balls": int(
                    arrays["info_visible_balls"][tick_index]
                ),
            },
        }
        for tick_index in range(count)
    ]
    return {
        "level_id": str(entry["level_id"]),
        "seed": int(entry["seed"]),
        "outcome": str(entry.get("outcome", "")),
        "time_limit_truncated": bool(
            entry.get("time_limit_truncated", False)
        ),
        "observation_capacity_overflow": bool(
            entry.get("observation_capacity_overflow", False)
        ),
        "score": int(entry.get("score", 0)),
        "ticks": ticks,
    }


def _load_episode_observations(path: Path) -> np.ndarray:
    """Decompress ONE episode's observations (gather pass, per episode)."""

    with np.load(path) as archive:
        if "observations" not in archive.files:
            raise ValueError(f"episode NPZ misses observations: {path}")
        observations = archive["observations"]
    return np.asarray(observations, dtype=np.float32)


def _retained_episode_runs(
    provenance: builder.CoverageReplayProvenance,
) -> list[tuple[int, np.ndarray]]:
    """The retention plan as contiguous per-episode runs, in row order.

    The builder emits rows level-major with episodes in input order and
    ticks ascending, so each episode's retained rows form exactly one
    contiguous block; both invariants are asserted, never assumed.
    """

    episode_indices = np.asarray(provenance.episode_indices, dtype=np.int64)
    tick_indices = np.asarray(provenance.tick_indices, dtype=np.int64)
    if episode_indices.size != tick_indices.size:
        raise ValueError("provenance arrays disagree on row count")
    runs: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    boundaries = np.flatnonzero(np.diff(episode_indices) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [episode_indices.size]))
    for start, stop in zip(starts.tolist(), stops.tolist()):
        ordinal = int(episode_indices[start])
        if ordinal in seen:
            raise ValueError(
                f"retention plan revisits episode {ordinal}; "
                f"per-episode rows are no longer contiguous"
            )
        seen.add(ordinal)
        ticks = tick_indices[start:stop]
        if ticks.size > 1 and bool(np.any(np.diff(ticks) <= 0)):
            raise ValueError(
                f"retention plan ticks are not ascending for episode "
                f"{ordinal}"
            )
        runs.append((ordinal, ticks))
    return runs


def _iter_rows(
    observations: np.ndarray,
    actions: np.ndarray,
    training_masks: np.ndarray,
) -> Iterator[builder.SampleRow]:
    """Row-at-a-time view stream for the builder's ``_dataset_sha256``.

    The frozen digest helper only iterates its argument, so a lazy
    generator over the observation memmap keeps the recomputed digest
    byte-compatible without materializing the rows.
    """

    for row_index in range(int(actions.shape[0])):
        yield (
            observations[row_index],
            actions[row_index],
            training_masks[row_index],
        )


def _write_new_npz_streamed(
    path: Path,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    training_masks: np.ndarray,
    provenance: builder.CoverageReplayProvenance,
) -> None:
    """``builder._write_new_npz`` semantics without re-stacking rows.

    Identical member names in the identical order; ``observations`` may
    be a read-only memmap, which ``np.savez_compressed`` streams in
    bounded buffers instead of holding a second in-RAM copy.
    """

    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        training_masks=training_masks,
        exact_masks=provenance.exact_masks,
        episode_indices=provenance.episode_indices,
        tick_indices=provenance.tick_indices,
    )


def run_coverage_replay_build(
    config: CoverageReplayBuildRunConfig,
) -> dict[str, Any]:
    """Verify receipts, plan on metadata, gather observations, write."""

    started = time.perf_counter()
    run_dir = Path(config.collection_run_dir).expanduser().resolve(
        strict=True
    )
    manifest_path = (run_dir / "episodes_manifest.json").resolve(strict=True)
    collection_manifest, entries = load_collection_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    verify_episode_receipts(manifest_dir, entries)
    collection_manifest_sha256 = builder._file_sha256(manifest_path)

    # Observation geometry for every episode, without decompressing any
    # observation payload: width agreement + tick_count receipts.
    observation_width: int | None = None
    total_source_ticks = 0
    for entry in entries:
        path = _entry_path(manifest_dir, entry)
        tick_count, width = _peek_observation_header(path)
        if tick_count != int(entry["tick_count"]):
            raise ValueError(
                f"episode tick_count receipt disagrees with the NPZ "
                f"({int(entry['tick_count'])} != {tick_count}): {path}"
            )
        if observation_width is None:
            observation_width = width
        elif width != observation_width:
            raise ValueError(
                f"episode observation width changed across the "
                f"collection ({width} != {observation_width}): {path}"
            )
        total_source_ticks += tick_count
    assert observation_width is not None  # entries is non-empty

    teacher_policy_id = (
        str(config.teacher_policy_id)
        if config.teacher_policy_id is not None
        else collection_manifest.get("teacher_policy_id")
    )

    # ------------------------------------------------------------------
    # Pass 1 -- planning: real metadata, stub observations (see module
    # docstring for why retention is observation-independent).
    # ------------------------------------------------------------------
    planning_started = time.perf_counter()
    records = [
        _metadata_episode_record(manifest_dir, entry) for entry in entries
    ]
    rows, manifest, provenance = (
        builder.build_coverage_equalized_replay_dataset(
            records,
            config=config.builder_config,
            round_index=int(config.round_index),
            teacher_policy_id=teacher_policy_id,
        )
    )
    del records
    if not rows:
        raise ValueError("builder retained no rows")
    row_count = len(rows)
    actions = np.stack([row[1] for row in rows])
    training_masks = np.stack([row[2] for row in rows])
    del rows
    runs = _retained_episode_runs(provenance)
    planning_seconds = time.perf_counter() - planning_started

    # ------------------------------------------------------------------
    # Pass 2 -- gather: hydrate retained observation rows one episode at
    # a time into a disk spill, then stream the aggregate NPZ from a
    # read-only memmap (the collector's own spill pattern).
    # ------------------------------------------------------------------
    gather_started = time.perf_counter()
    output_prefix = Path(config.output_prefix).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.parent / (output_prefix.name + ".npz")
    manifest_out_path = output_prefix.parent / (
        output_prefix.name + ".manifest.json"
    )
    dataset_manifest_path = output_prefix.parent / (
        output_prefix.name + ".dataset.json"
    )
    completion_path = output_prefix.parent / (
        output_prefix.name + ".completion.json"
    )
    for planned in (
        npz_path,
        manifest_out_path,
        dataset_manifest_path,
        completion_path,
    ):
        if planned.exists():
            raise FileExistsError(f"output already exists: {planned}")
    spill_path = output_prefix.parent / (
        f".{output_prefix.name}.observations.{os.getpid()}.spill"
    )
    max_episode_observation_bytes = 0
    digest: str | None = None
    try:
        with spill_path.open("wb") as spill:
            written_rows = 0
            for ordinal, ticks in runs:
                entry = entries[ordinal]
                path = _entry_path(manifest_dir, entry)
                observations = _load_episode_observations(path)
                if int(observations.shape[1]) != observation_width:
                    raise ValueError(
                        f"episode observation width changed between "
                        f"passes: {path}"
                    )
                if int(ticks[-1]) >= int(observations.shape[0]):
                    raise ValueError(
                        f"retention plan addresses tick "
                        f"{int(ticks[-1])} beyond episode end: {path}"
                    )
                max_episode_observation_bytes = max(
                    max_episode_observation_bytes, int(observations.nbytes)
                )
                block = np.ascontiguousarray(
                    observations[ticks], dtype=np.float32
                )
                spill.write(block.tobytes())
                written_rows += int(block.shape[0])
                del observations, block
            if written_rows != row_count:
                raise ValueError(
                    f"gather pass produced {written_rows} rows for a "
                    f"{row_count}-row plan"
                )
            spill.flush()
            os.fsync(spill.fileno())
        observation_rows = np.memmap(
            spill_path,
            dtype=np.float32,
            mode="r",
            shape=(row_count, observation_width),
        )
        # Recompute the dataset digest over the REAL rows through the
        # builder's frozen helper (the planning-pass digest hashed the
        # stub observations and is discarded).
        digest = builder._dataset_sha256(
            _iter_rows(observation_rows, actions, training_masks)
        )
        manifest["dataset_sha256"] = digest
        _write_new_npz_streamed(
            npz_path,
            observations=observation_rows,
            actions=actions,
            training_masks=training_masks,
            provenance=provenance,
        )
        del observation_rows
    finally:
        spill_path.unlink(missing_ok=True)
    gather_seconds = time.perf_counter() - gather_started

    # Identical output trio, identical write order to ``builder.main``.
    dataset_manifest = builder._dataset_manifest(
        npz_path=npz_path.resolve(),
        rows=range(row_count),  # only len() is consumed
        manifest=manifest,
    )
    builder._write_new_json(dataset_manifest_path, dataset_manifest)
    manifest["trainer_dataset_manifest"] = dataset_manifest_path.name
    builder._write_new_json(manifest_out_path, manifest)

    completion = {
        "schema": COMPLETION_SCHEMA,
        "version": VERSION,
        "status": "BUILT",
        "completed_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "runner": {
            "path": str(SCRIPT_PATH),
            "sha256": builder._file_sha256(SCRIPT_PATH),
        },
        "builder": {
            "path": str(builder.SCRIPT_PATH),
            "sha256": builder._file_sha256(builder.SCRIPT_PATH),
            "builder_id": builder.BUILDER_ID,
            "builder_version": builder.BUILDER_VERSION,
        },
        "collection": {
            "run_dir": str(run_dir),
            "episodes_manifest": {
                "path": str(manifest_path),
                "sha256": collection_manifest_sha256,
            },
            "episodes": len(entries),
            "episodes_sha256_verified": len(entries),
            "total_source_ticks": total_source_ticks,
            "observation_width": observation_width,
            "teacher_policy_id": collection_manifest.get(
                "teacher_policy_id"
            ),
        },
        "builder_config": {
            **asdict(config.builder_config),
            "round_index": int(config.round_index),
            "teacher_policy_id": teacher_policy_id,
            "teacher_policy_id_source": (
                "cli"
                if config.teacher_policy_id is not None
                else "collection_manifest"
            ),
        },
        "ram_safety": {
            "strategy": (
                "two_pass_metadata_plan_then_streamed_observation_gather"
            ),
            "planning_pass": {
                "observations_resident": False,
                "stub_observation_width": int(_STUB_OBSERVATION.size),
                "hydrated_metadata_keys": list(_METADATA_KEYS),
                "info_fields": list(_PLANNING_INFO_FIELDS),
                "wall_seconds": planning_seconds,
            },
            "gather_pass": {
                "episodes_hydrated": len(runs),
                "max_single_episode_observation_bytes": (
                    max_episode_observation_bytes
                ),
                "retained_observation_bytes": (
                    row_count * observation_width * 4
                ),
                "aggregate_npz_write": "memmap_streamed",
                "wall_seconds": gather_seconds,
            },
        },
        "retained_samples": int(manifest["retained_samples"]),
        "retained_samples_by_verb": manifest["retained_samples_by_verb"],
        "dataset_sha256": digest,
        "runtime_illegal_raw_verb_fraction": manifest[
            "runtime_illegal_raw_verb_fraction"
        ],
        "coverage_equalization": manifest["coverage_equalization"],
        "per_level_coverage": manifest["per_level_coverage"],
        "outputs": {
            "samples_npz": {
                "path": str(npz_path),
                "sha256": builder._file_sha256(npz_path),
                "rows": row_count,
            },
            "manifest": {
                "path": str(manifest_out_path),
                "sha256": builder._file_sha256(manifest_out_path),
            },
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": builder._file_sha256(dataset_manifest_path),
            },
        },
        "formal_seed_consumption": False,
    }
    builder._write_new_json(completion_path, completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    defaults = builder.CoverageReplayV4Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-run-dir",
        required=True,
        type=Path,
        help="collector run dir holding episodes/ + episodes_manifest.json",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--target-coverage-ratio",
        type=float,
        default=defaults.target_coverage_ratio,
    )
    parser.add_argument(
        "--total-budget", type=int, default=defaults.total_budget
    )
    parser.add_argument(
        "--fire-window-before-ticks",
        type=int,
        default=defaults.fire_window_before_ticks,
    )
    parser.add_argument(
        "--fire-window-after-ticks",
        type=int,
        default=defaults.fire_window_after_ticks,
    )
    parser.add_argument(
        "--danger-threshold", type=float, default=defaults.danger_threshold
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=defaults.sampling_seed
    )
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument(
        "--teacher-policy-id",
        type=str,
        default=None,
        help="defaults to the collection manifest's teacher_policy_id",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CoverageReplayBuildRunConfig(
        collection_run_dir=args.collection_run_dir,
        output_prefix=args.output_prefix,
        builder_config=builder.CoverageReplayV4Config(
            target_coverage_ratio=args.target_coverage_ratio,
            total_budget=args.total_budget,
            fire_window_before_ticks=args.fire_window_before_ticks,
            fire_window_after_ticks=args.fire_window_after_ticks,
            danger_threshold=args.danger_threshold,
            sampling_seed=args.sampling_seed,
        ),
        round_index=int(args.round_index),
        teacher_policy_id=args.teacher_policy_id,
    )
    completion = run_coverage_replay_build(config)
    print(
        json.dumps(
            {
                "status": completion["status"],
                "builder": builder.BUILDER_ID,
                "retained_samples": completion["retained_samples"],
                "dataset_sha256": completion["dataset_sha256"],
                "samples_npz": completion["outputs"]["samples_npz"]["path"],
                "manifest": completion["outputs"]["manifest"]["path"],
                "dataset_manifest": completion["outputs"][
                    "dataset_manifest"
                ]["path"],
                "runtime_illegal_raw_verb_fraction": completion[
                    "runtime_illegal_raw_verb_fraction"
                ],
                "per_level_coverage": completion["per_level_coverage"],
                "ram_safety": completion["ram_safety"],
                "wall_seconds": completion["wall_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
