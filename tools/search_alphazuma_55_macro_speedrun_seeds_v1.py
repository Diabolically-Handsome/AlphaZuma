"""Tail-search fastest-win seeds for one macro-native model on one level.

``tools/probe_alphazuma_55_macro_native_eval_v1.py`` evaluates a
macro-native stacked checkpoint honestly but at panel scale: one seed
per level per invocation, so process startup (model load + SubprocVecEnv
spawn) dominates once thousands of seeds are wanted.  This tool is the
throughput restructuring of the same frozen interface for the speedrun
best-of-N convention: ONE level x ONE model x MANY seeds, streamed
through a persistent vector, reporting the fastest wins.

Convention: best-of-N seed search mirrors human IL record practice
(retry-until-RNG-cooperates); results are simulator times pending an RTA
timing-convention audit.  Nothing here is a formal metric: every receipt
carries ``formal_seed_consumption: false`` and no training or
formal-selection authority.

What is reused (imported, never copied)
---------------------------------------
* env factory: ``probe_alphazuma_55_macro_native_eval_v1
  ._make_stacked_macro_env_factory`` (ObservationStackWrapper UNDER
  ParkSettleActionWrapper over the probes' motor-observable stack);
* model gate: ``require_macro_native_model`` (MultiDiscrete([3, 180])
  only; per-tick checkpoints are refused with a pointer to the adapter
  probe) plus the stacked-width divisibility check;
* predict discipline: masked deterministic predict on the wrapper's own
  (3 + 180) ``action_masks()``, bit for bit.  There is no adapter and no
  fallback path: a prediction that violates its mask fails the run
  loudly (mirrors the native probe's zero-fallback contract).

Seed-queue streaming (throughput design)
----------------------------------------
A persistent SubprocVecEnv of ``--parallel-envs`` copies of the SAME
level streams a seed queue: whenever one env's episode ends, that env
alone is reseeded with the next queued seed and reset via
``vector.env_method("reset", seed=..., indices=[i])``.  Verified against
the installed stable-baselines3 2.9.0 SubprocVecEnv worker: the
``env_method`` command dispatches to the one worker process and calls
the gymnasium wrapper-chain ``reset(seed=...)`` directly, so per-env
seeded resets need NO synchronized generation batching (the fallback
this tool would otherwise take).  Two documented costs, correctness
unaffected:

* the worker auto-resets (unseeded) when an episode ends inside
  ``step``; our seeded reset immediately supersedes it, so one redundant
  reset is paid per episode;
* ``vector.step`` is synchronous across envs, so after the queue drains
  the last stragglers run with idle siblings.  Drained envs receive
  inert ``wait_hold`` macros and their telemetry is ignored -- the same
  finished-env masking ``probe_alphazuma_55_macro_eval_v1
  ._run_student_only`` uses for its fixed panel.

Crash-safe outputs and resume
------------------------------
``results.jsonl`` gains one line per finished seed the moment the
episode ends (single write + flush + fsync, so at most the FINAL line
can be torn by a crash) and ``results_index.json`` is rewritten
atomically after every row.  Resume re-runs the tool with the same
arguments: rows already in ``results.jsonl`` are skipped, a torn final
line is truncated away and its seed re-run, and a resume that finds
nothing pending just rebuilds ``completion.json`` from the rows.

Timing-convention groundwork (for the future RTA audit)
--------------------------------------------------------
A row's ``ticks`` is the native simulator tick count from episode reset
(tick 0) through the first tick at which the episode-ending boundary
was observed; the revenge env breaks its frame-skip loop on that exact
tick, so the value is tick-exact and includes every wrapper
park/settle/hold tick (ordinary simulated ticks).  A win ends the
episode at the FIRST irreversible boundary (``sim.win_pending`` or a
settled ``win`` outcome); each win row records that tick as
``win_trigger_tick`` and preserves ``win_pending_at_termination``.  No
distinct later "board cleared" tick exists in the info stream (the env
terminates AT the boundary, before retail's input-locked victory tail
is simulated), so ``board_cleared_tick`` is recorded as null and the
RTA audit can add the tail by explicit convention.

Seed discipline
----------------
Default ``--seed-start`` 1_496_000_000 x 10_000 seeds: engineering
only.  The range is validated in-tool against the frozen formal embargo
ranges AND the reserved engineering blocks already consumed by sibling
tools (probe-v4, teacher collection, DAgger-v2, macro-PPO engineering,
and the macro-decisions collection block around 1_494_000_000).

Seed-list TAS mode (``--seed-file``)
-------------------------------------
``--seed-file`` (mutually exclusive with ``--seed-start``/
``--seed-count``) streams an EXPLICIT seed list instead of a contiguous
range: a JSON/JSONL file in the golden-seed enumerator's top-K format
(``tools/enumerate_alphazuma_55_golden_seeds_v1.py`` ``top_k.jsonl``
rows carrying ``"seed"``), a JSON array of integers, or a JSON object
with a ``"seeds"`` array.  File order is preserved (rank order for
enumerator files), resume semantics are unchanged (``results.jsonl``
keyed by seed), and the receipts record the seed file's sha256 plus its
provenance (first-row schema and any sibling ``completion.json``).
Runs are labeled ``"convention": "seed_list_tas"`` -- seeds chosen
OFFLINE by exhaustive RNG-opening enumeration, honestly distinct from
the default best-of-N tail-search convention.  Frozen embargoes still
apply per seed (``--drop-embargoed`` skips and records them); reserved
engineering blocks refuse unless ``--allow-reserved`` (recorded).

Record-mode flags (variance on demand)
---------------------------------------
Best-of-N record hunting makes losses free, so variance is a feature.
``--stochastic-temperature`` (default None: masked deterministic
argmax, unchanged) samples each macro from the policy distribution
instead: the wrapper's own (3 + 180) action mask is applied to the raw
logits (masked-off entries dropped, the survivors renormalized by the
softmax), the surviving logits are divided by the temperature before
that softmax, and one (verb, aim) pair is drawn from a per-episode
numpy Generator derived from the episode seed
(``STOCHASTIC_RNG_SCHEME``), so identical CLI invocations reproduce
bit-for-bit and results are independent of ``--parallel-envs``
scheduling.  The run's ``convention`` gains a ``_stochasticT<temp>``
suffix so receipts self-describe.  ``--settle-tolerance-bins``
(default None: the wrapper default) overrides
``ParkSettleActionConfig.settle_tolerance_bins`` at env construction.
The imported factory's signature has no tolerance knob and frozen
files stay frozen, so the override REWRAPS: the imported factory
builds its usual stack -> ParkSettleActionWrapper chain, then the
wrapper is replaced by a fresh ParkSettleActionWrapper over the SAME
stacked env with the requested config (hold_ticks preserved).  The
effective value is echoed back by the wrapper's own ``contract()``
recorded in ``completion.json`` and verified in-tool.  Both flags
default to off and add receipt fields ONLY when set, so prior
conventions stay byte-compatible.
"""

from __future__ import annotations

import argparse
from collections import deque
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

# Imported for the side effect of making the stacked policy's pickled
# feature-extractor class importable before MaskablePPO.load runs.
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3  # noqa: F401
from tools import collect_alphazuma_55_macro_decisions_v1 as decisions_v1
from tools import collect_alphazuma_55_park_settle_episodes_v1 as episodes_v1
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_macro_native_eval_v1 as native_eval
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    parse_stack_lags,
)
from zuma_rl.park_settle_action_wrapper import (
    MACRO_VERB_NAMES,
    MACRO_WAIT_HOLD,
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
)


SCRIPT_PATH = Path(__file__).resolve()
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-macro-speedrun-seed-search-v1"
ROW_SCHEMA = f"{COMPLETION_SCHEMA}-row"
INDEX_SCHEMA = f"{COMPLETION_SCHEMA}-index"
RESULTS_FILENAME = "results.jsonl"
INDEX_FILENAME = "results_index.json"
MODE = "seed_stream_tail_search"
AIM_BINS = macro_eval.AIM_BINS
MACRO_MASK_WIDTH = len(MACRO_VERB_NAMES) + AIM_BINS
DEFAULT_LEVEL = "Jungle1"
DEFAULT_SEED_START = 1_496_000_000
DEFAULT_SEED_COUNT = 10_000
DEFAULT_MAX_TICKS = macro_eval.DEFAULT_MAX_TICKS
DEFAULT_HOLD_TICKS = macro_eval.DEFAULT_HOLD_TICKS
DEFAULT_PARALLEL_ENVS = 32
DEFAULT_TOP_K = 25
WIN_TICK_PERCENTILES = (1, 5, 10, 50)
EMBARGOED_SEED_RANGES = episodes_v1.EMBARGOED_SEED_RANGES
RESERVED_ENGINEERING_SEED_BLOCKS = (
    *decisions_v1.RESERVED_ENGINEERING_SEED_BLOCKS,
    # tools/collect_alphazuma_55_macro_decisions_v1.py collects from its
    # default base 1_494_000_000; reserved with headroom so tail
    # searches never collide with decision-collection (level, seed)
    # keys.  The planned tail-search block 1_496_000_000.. starts
    # immediately above this reservation.
    ("macro_decisions_collection_v1", 1_494_000_000, 1_495_999_999),
)
SPEEDRUN_CONVENTION_NOTE = (
    "best-of-N seed search mirrors human IL record practice "
    "(retry-until-RNG-cooperates); results are simulator times pending "
    "an RTA timing-convention audit"
)
BEST_OF_N_CONVENTION = "best_of_n_tail_search"
SEED_LIST_CONVENTION = "seed_list_tas"
SEED_LIST_CONVENTION_NOTE = (
    "seed-list TAS lane: seeds were chosen OFFLINE by exhaustive "
    "RNG-opening enumeration (AI-TAS, honestly labeled), not by "
    "best-of-N sampling; results are simulator times pending an RTA "
    "timing-convention audit"
)
SEED_SPACE = 1 << 32
# The frozen wrapper's own default (2): record-mode --settle-tolerance-bins
# defaults to None, meaning "leave the wrapper at this".
DEFAULT_SETTLE_TOLERANCE_BINS = ParkSettleActionConfig().settle_tolerance_bins
# "Zuma": a fixed stream tag mixed into the sampler's SeedSequence so
# its draws can never alias the env's own episode seeding.
STOCHASTIC_RNG_STREAM_TAG = 0x5A756D61
STOCHASTIC_RNG_SCHEME = (
    "per-episode numpy Generator(PCG64(SeedSequence((episode_seed, "
    f"{STOCHASTIC_RNG_STREAM_TAG})))) created at that episode's seeded "
    "reset; each macro decision draws verb then aim (two rng.random() "
    "draws, in that order) by inverse CDF over "
    "softmax(masked_logits / temperature) with masked-off entries "
    "dropped before normalization; the stream is per-episode, so "
    "results are independent of parallel-env scheduling and "
    "bit-for-bit reproducible for the same CLI"
)
# The macro MultiDiscrete([3, 180]) as concatenated logit/mask segments.
MACRO_SEGMENT_WIDTHS = (len(MACRO_VERB_NAMES), AIM_BINS)


def validate_seed_range(seed_start: int, seed_count: int) -> tuple[int, int]:
    """Refuse any overlap with formal embargoes or used engineering blocks.

    Interval arithmetic over the whole planned range
    ``[seed_start, seed_start + seed_count - 1]`` -- a plan that merely
    straddles a protected block is refused just like one inside it.
    Returns ``(first, last)`` on success.
    """

    first = int(seed_start)
    count = int(seed_count)
    if first < 0:
        raise ValueError("seed_start cannot be negative")
    if count < 1:
        raise ValueError("seed_count must be positive")
    last = first + count - 1
    for name, low, high in EMBARGOED_SEED_RANGES:
        if first <= int(high) and int(low) <= last:
            raise ValueError(
                f"seed range [{first}, {last}] overlaps the frozen "
                f"{name} embargo [{int(low)}, {int(high)}]; choose "
                "another --seed-start"
            )
    for name, low, high in RESERVED_ENGINEERING_SEED_BLOCKS:
        if first <= int(high) and int(low) <= last:
            raise ValueError(
                f"seed range [{first}, {last}] overlaps the reserved "
                f"{name} engineering block [{int(low)}, {int(high)}]; "
                "choose another --seed-start so runs keep unique "
                "(level, seed) keys"
            )
    return first, last


def _decode_seed_entry(entry: Any, where: str) -> int:
    if isinstance(entry, Mapping):
        if "seed" not in entry:
            raise ValueError(
                f"seed file entry at {where} has no 'seed' field"
            )
        entry = entry["seed"]
    if isinstance(entry, bool) or not isinstance(entry, int):
        raise ValueError(
            f"seed file entry at {where} is not an integer seed"
        )
    if not 0 <= int(entry) < SEED_SPACE:
        raise ValueError(
            f"seed file entry at {where} is outside [0, 2^32): {entry}"
        )
    return int(entry)


def load_seed_file(path: Path) -> list[int]:
    """Parse a JSON/JSONL seed list (the enumerator's top-K format).

    Accepted forms:

    * JSONL: one JSON object per line carrying ``"seed"`` (the golden
      seed enumerator's ``top_k.jsonl`` rows) or one bare integer;
    * a JSON array of integers or of objects carrying ``"seed"``;
    * a JSON object with a ``"seeds"`` array (or a single ``"seed"``).

    File order is preserved -- for enumerator files that is rank order,
    so the best-scored seeds stream first.  Empty files, duplicates,
    non-integers, and seeds outside [0, 2^32) are refused loudly.
    """

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"seed file {path} is empty")
    seeds: list[int] = []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, list):
        seeds = [
            _decode_seed_entry(entry, f"index {position}")
            for position, entry in enumerate(document)
        ]
    elif isinstance(document, Mapping):
        if "seeds" in document:
            listed = document["seeds"]
            if not isinstance(listed, list):
                raise ValueError(
                    f"seed file {path} 'seeds' is not an array"
                )
            seeds = [
                _decode_seed_entry(entry, f"seeds[{position}]")
                for position, entry in enumerate(listed)
            ]
        elif "seed" in document:
            seeds = [_decode_seed_entry(document, "the root object")]
        else:
            raise ValueError(
                f"seed file {path} is a JSON object without 'seeds' "
                "or 'seed'"
            )
    elif document is None:
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"seed file {path} line {number} is neither JSON "
                    f"nor part of a JSON document: {error}"
                ) from error
            seeds.append(_decode_seed_entry(entry, f"line {number}"))
    else:
        raise ValueError(
            f"seed file {path} is a JSON scalar, not a seed list"
        )
    if not seeds:
        raise ValueError(f"seed file {path} contains no seeds")
    seen: set[int] = set()
    for seed in seeds:
        if seed in seen:
            raise ValueError(
                f"seed file {path} lists seed {seed} more than once"
            )
        seen.add(seed)
    return seeds


def validate_seed_list(
    seeds: Sequence[int],
    *,
    drop_embargoed: bool = False,
    allow_reserved: bool = False,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-seed embargo/reserved discipline for explicit seed lists.

    Frozen formal embargoes always apply: an embargoed seed refuses the
    run loudly unless ``drop_embargoed``, which skips it and records it
    (an enumerator scans the whole 2^32 space, so its top-K may
    legitimately straddle an embargo -- dropping keeps the frozen
    ranges unconsumed without discarding the rest of the list).
    Reserved engineering blocks are (level, seed)-key collision guards,
    not embargoes: overlapping seeds refuse unless ``allow_reserved``,
    and every overlap is recorded in the receipts either way.  Returns
    ``(kept, dropped_embargoed, reserved_overlaps)``.
    """

    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    reserved: list[dict[str, Any]] = []
    for seed in seeds:
        value = int(seed)
        embargo = next(
            (
                name
                for name, low, high in EMBARGOED_SEED_RANGES
                if int(low) <= value <= int(high)
            ),
            None,
        )
        if embargo is not None:
            if drop_embargoed:
                dropped.append({"seed": value, "embargo": embargo})
                continue
            raise ValueError(
                f"seed {value} lies inside the frozen {embargo} "
                "embargo; pass --drop-embargoed to skip embargoed "
                "seeds (they are recorded, never run)"
            )
        block = next(
            (
                name
                for name, low, high in RESERVED_ENGINEERING_SEED_BLOCKS
                if int(low) <= value <= int(high)
            ),
            None,
        )
        if block is not None:
            reserved.append({"seed": value, "block": block})
            if not allow_reserved:
                raise ValueError(
                    f"seed {value} lies inside the reserved {block} "
                    "engineering block; pass --allow-reserved to run "
                    "it anyway (the overlap is recorded in receipts)"
                )
        kept.append(value)
    if not kept:
        raise ValueError(
            "seed list is empty after embargo filtering; nothing to run"
        )
    return kept, dropped, reserved


def _seed_file_provenance(path: Path) -> dict[str, Any]:
    """Self-describing receipts for where a seed list came from."""

    payload: dict[str, Any] = {
        "path": str(path),
        "sha256": base._sha256(path),
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(row, Mapping) and "schema" in row:
            payload["first_row_schema"] = str(row["schema"])
        break
    sibling = path.parent / "completion.json"
    if sibling.exists():
        try:
            data = base._read(sibling)
        except (ValueError, json.JSONDecodeError):
            data = {}
        payload["sibling_completion"] = {
            "path": str(sibling),
            "sha256": base._sha256(sibling),
            "schema": data.get("schema"),
        }
    return payload


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], top_k: int
) -> dict[str, Any]:
    """Best-of-N tail summary: win rate, win-tick percentiles, top-k.

    Percentiles use numpy's ``method="lower"`` so every reported value
    is an actually-achieved win tick (a seed that can be re-run), never
    an interpolated time no episode produced.  The top-k table is
    sorted by ``(ticks, seed)`` ascending -- deterministic under ties.
    With zero wins the percentiles are all None and the table is empty;
    ``win_rate`` over zero rows is NaN (serialized null), never a free
    pass.
    """

    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    total = len(rows)
    wins = [row for row in rows if row.get("outcome") == "win"]
    losses = sum(1 for row in rows if row.get("outcome") == "loss")
    truncations = sum(
        1 for row in rows if bool(row.get("time_limit_truncated", False))
    )
    win_ticks = sorted(int(row["ticks"]) for row in wins)
    percentiles: dict[str, int | None] = {}
    for quantile in WIN_TICK_PERCENTILES:
        percentiles[f"p{quantile}"] = (
            int(np.percentile(win_ticks, quantile, method="lower"))
            if win_ticks
            else None
        )
    fastest = sorted(
        wins, key=lambda row: (int(row["ticks"]), int(row["seed"]))
    )[: int(top_k)]
    return {
        "total_seeds": total,
        "wins": len(wins),
        "losses": losses,
        "truncations": truncations,
        "win_rate": v4._fraction(len(wins), total),
        "win_tick_percentiles": percentiles,
        "top_k": int(top_k),
        "top_k_fastest_wins": [
            {
                "seed": int(row["seed"]),
                "ticks": int(row["ticks"]),
                "shots": int(row["shots"]),
                "score": int(row["score"]),
            }
            for row in fastest
        ],
    }


def _timing_convention(
    convention_note: str = SPEEDRUN_CONVENTION_NOTE,
) -> dict[str, Any]:
    """Raw material for the future RTA timing-convention audit."""

    return {
        "tick_semantics": (
            "row 'ticks' is the native simulator tick count from "
            "episode reset (tick 0) through the first tick at which the "
            "episode-ending boundary was observed; the revenge env "
            "breaks its frame-skip loop on that exact tick, so the "
            "value is tick-exact and includes every wrapper "
            "park/settle/hold tick (those are ordinary simulated ticks)"
        ),
        "win_trigger": (
            "a win ends the episode at the FIRST irreversible boundary "
            "(sim.win_pending or a settled 'win' outcome); each win row "
            "records that tick as win_trigger_tick (== ticks)"
        ),
        "board_cleared_tick": (
            "recorded as null: the env terminates AT the irreversible "
            "boundary, before retail's input-locked victory tail is "
            "simulated, so no distinct later 'board cleared' tick "
            "exists in the info stream; win_pending_at_termination "
            "preserves whether the trigger was the pending boundary "
            "(tail not simulated) or a settled native outcome so an "
            "RTA convention can add the tail explicitly"
        ),
        "rta_audit_status": "PENDING",
        "convention_note": convention_note,
    }


def _seed_discipline_payload() -> dict[str, Any]:
    return {
        "embargoed_seed_ranges": {
            name: [int(low), int(high)]
            for name, low, high in EMBARGOED_SEED_RANGES
        },
        "reserved_engineering_seed_blocks": {
            name: [int(low), int(high)]
            for name, low, high in RESERVED_ENGINEERING_SEED_BLOCKS
        },
        "formal_seed_consumption": False,
    }


def _validate_stochastic_temperature(temperature: Any) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "stochastic temperature must be a finite positive float, "
            f"got {temperature!r}"
        )
    return value


def _stochastic_convention(convention: str, temperature: float) -> str:
    """Suffix the run convention so record-mode receipts self-describe."""

    value = _validate_stochastic_temperature(temperature)
    return f"{convention}_stochasticT{value:g}"


def stochastic_episode_rng(seed: int) -> np.random.Generator:
    """The per-episode sampler RNG (see ``STOCHASTIC_RNG_SCHEME``).

    Derived from the episode seed plus a fixed stream tag: one fresh
    generator per episode, created at that episode's seeded reset, so
    the draw stream depends only on the episode itself -- never on
    which env slot ran it or how many envs ran beside it.
    """

    sequence = np.random.SeedSequence(
        (int(seed), STOCHASTIC_RNG_STREAM_TAG)
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _maskable_policy_logits(model: Any, observations: Any) -> np.ndarray:
    """Raw per-segment macro logits from a MaskablePPO policy.

    Returns the concatenated (verb, aim) logits, shape
    ``(rows, 3 + 180)``, BEFORE any masking -- the sampler applies the
    wrapper's mask itself.  Torch's Categorical normalizes each
    segment's logits into log-probabilities; that constant is uniform
    within a segment, so it cancels inside the temperature softmax and
    sampling is unaffected.
    """

    import torch

    policy = model.policy
    policy.set_training_mode(False)
    with torch.no_grad():
        obs_tensor, _ = policy.obs_to_tensor(
            np.asarray(observations, dtype=np.float32)
        )
        distribution = policy.get_distribution(obs_tensor)
        parts = getattr(distribution, "distributions", None)
        if not parts:
            raise ValueError(
                "policy distribution exposes no per-segment "
                "categoricals; a macro-native MultiDiscrete([3, 180]) "
                "policy is required"
            )
        logits = torch.cat(
            [part.logits for part in parts], dim=1
        ).detach().cpu().numpy()
    return np.asarray(logits, dtype=np.float64)


def sample_masked_macro_action(
    logits: Any,
    mask: Any,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Temperature-sample one (verb, aim) macro from masked logits.

    Per segment (verb first, then aim): masked-off logits are dropped
    (set to -inf, never renormalized into), the survivors are divided
    by the temperature and softmaxed, and one index is drawn by
    inverse CDF on a single ``rng.random()`` draw.  Exactly two draws
    per call, in a fixed order, so the per-episode stream is
    deterministic.  A sample outside its mask is impossible by
    construction and double-checked before returning.
    """

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if logits.shape[0] != MACRO_MASK_WIDTH:
        raise ValueError(
            f"macro logits width must be {MACRO_MASK_WIDTH}, got "
            f"{logits.shape}"
        )
    if mask.shape[0] != MACRO_MASK_WIDTH:
        raise ValueError(
            f"macro mask width must be {MACRO_MASK_WIDTH}, got "
            f"{mask.shape}"
        )
    temperature = _validate_stochastic_temperature(temperature)
    choices: list[int] = []
    start = 0
    for width in MACRO_SEGMENT_WIDTHS:
        segment_logits = logits[start : start + width]
        segment_mask = mask[start : start + width]
        start += width
        if not bool(segment_mask.any()):
            raise RuntimeError(
                "macro action mask leaves no legal choice in a "
                f"segment of width {width}"
            )
        scaled = np.where(
            segment_mask, segment_logits / temperature, -np.inf
        )
        scaled = scaled - float(np.max(scaled[segment_mask]))
        probabilities = np.exp(scaled)
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "temperature softmax produced no probability mass "
                f"(total {total!r})"
            )
        probabilities /= total
        cumulative = np.cumsum(probabilities)
        draw = float(rng.random())
        index = int(np.searchsorted(cumulative, draw, side="right"))
        index = min(index, width - 1)
        # A zero-probability entry can never be selected mid-range
        # (its cumulative value equals its predecessor's); only float
        # roundoff at the very top can land past the last mass-bearing
        # index, so walk back to it.
        while index > 0 and not bool(segment_mask[index]):
            index -= 1
        if not bool(segment_mask[index]):
            raise RuntimeError(
                "stochastic sampler selected a masked-off index; the "
                "macro mask contract is broken"
            )
        choices.append(index)
    return np.asarray(choices, dtype=np.int64)


def _reset_env_seeded(vector: Any, index: int, seed: int) -> np.ndarray:
    """Seeded reset of ONE env inside the persistent vector.

    SB3 2.9 dispatches ``env_method("reset", seed=...)`` to the single
    target worker, which calls the gymnasium wrapper-chain
    ``reset(seed=...)`` directly -- the same full reset the vector's own
    synchronized ``reset()`` performs, minus the synchronization.  The
    worker's automatic unseeded auto-reset on done is superseded by
    this call (one redundant reset per episode, correctness unaffected).
    """

    payload = vector.env_method(
        "reset", indices=[int(index)], seed=int(seed)
    )
    observation, _reset_info = payload[0]
    return np.asarray(observation)


def run_seed_stream(
    *,
    model: Any,
    vector: Any,
    level_id: str,
    pending_seeds: Sequence[int],
    max_ticks: int,
    append_row: Callable[[Mapping[str, Any]], None],
    masks_provider: Callable[[Any], np.ndarray] | None = None,
    stochastic_temperature: float | None = None,
    logits_provider: Callable[[Any, np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Stream a seed queue through a persistent macro-wrapped vector.

    Every env always carries exactly one live episode (or is drained):
    the moment an episode ends its row is handed to ``append_row`` and
    the env alone is reseeded with the next queued seed.  Drained envs
    receive inert wait_hold macros and their telemetry is ignored --
    the finished-env masking convention of ``macro_eval
    ._run_student_only``.  With ``stochastic_temperature`` set, actions
    are temperature-sampled from the policy's masked logits with one
    RNG per episode (``STOCHASTIC_RNG_SCHEME``) instead of the default
    masked deterministic predict.  The model, vector, mask provider,
    and logits provider are injectable so both paths are CPU-testable
    with stubs.
    """

    if masks_provider is None:
        masks_provider = macro_eval._default_masks_provider
    stochastic = stochastic_temperature is not None
    if stochastic:
        temperature = _validate_stochastic_temperature(
            stochastic_temperature
        )
        if logits_provider is None:
            logits_provider = _maskable_policy_logits
    queue = deque(int(seed) for seed in pending_seeds)
    if not queue:
        raise ValueError("pending_seeds must be non-empty")
    count = int(vector.num_envs)
    if count < 1:
        raise ValueError("vector has no environments")
    if count > len(queue):
        raise ValueError(
            f"{count} parallel envs exceed {len(queue)} pending seeds; "
            "build the vector with min(parallel_envs, pending)"
        )
    started = time.perf_counter()
    width = int(vector.observation_space.shape[0])
    observations = np.zeros((count, width), dtype=np.float32)
    active_seed: list[int | None] = [None] * count
    episode_rng: list[np.random.Generator | None] = [None] * count
    shots = np.zeros(count, dtype=np.int64)
    macro_decisions = np.zeros(count, dtype=np.int64)
    native_ticks = np.zeros(count, dtype=np.int64)
    for index in range(count):
        seed = queue.popleft()
        observations[index] = _reset_env_seeded(vector, index, seed)
        active_seed[index] = seed
        if stochastic:
            episode_rng[index] = stochastic_episode_rng(seed)
    rows: list[dict[str, Any]] = []
    transitions = 0
    episodes_planned = len(pending_seeds)
    # Guard: each macro consumes >= 1 native tick and every episode is
    # bounded by max_ticks, so ceil(episodes/count) + 1 waves of
    # (max_ticks + 1) macro iterations is a generous upper bound.
    waves = (episodes_planned + count - 1) // count + 1
    guard = waves * (int(max_ticks) + 1)
    for _ in range(guard):
        if all(seed is None for seed in active_seed):
            break
        masks = np.asarray(
            masks_provider(vector), dtype=np.bool_
        ).reshape(count, -1)
        if masks.shape[1] != MACRO_MASK_WIDTH:
            raise ValueError(
                f"macro mask width must be {MACRO_MASK_WIDTH}, got "
                f"{masks.shape}"
            )
        if stochastic:
            logits = np.asarray(
                logits_provider(model, observations), dtype=np.float64
            ).reshape(count, -1)
            if logits.shape[1] != MACRO_MASK_WIDTH:
                raise ValueError(
                    f"macro logits width must be {MACRO_MASK_WIDTH}, "
                    f"got {logits.shape}"
                )
            actions = np.zeros((count, 2), dtype=np.int64)
            for index in range(count):
                if active_seed[index] is None:
                    actions[index] = (MACRO_WAIT_HOLD, 0)
                    continue
                actions[index] = sample_masked_macro_action(
                    logits[index],
                    masks[index],
                    temperature,
                    episode_rng[index],
                )
        else:
            predicted, _ = model.predict(
                observations, deterministic=True, action_masks=masks
            )
            actions = np.asarray(predicted, dtype=np.int64).reshape(
                count, 2
            )
            for index in range(count):
                if active_seed[index] is None:
                    actions[index] = (MACRO_WAIT_HOLD, 0)
                elif not bool(masks[index, int(actions[index][0])]):
                    raise RuntimeError(
                        "masked deterministic predict emitted a macro "
                        f"verb the wrapper masked off (env {index}, "
                        f"verb {int(actions[index][0])}): the "
                        "macro-native interface contract is broken"
                    )
        stepped, _, dones, infos = vector.step(actions)
        observations = np.asarray(stepped, dtype=np.float32).reshape(
            count, width
        )
        for index in range(count):
            seed = active_seed[index]
            if seed is None:
                continue
            transitions += 1
            info = dict(infos[index])
            macro = info.get("park_settle")
            if not isinstance(macro, Mapping):
                raise RuntimeError(
                    "macro step returned no park_settle telemetry; the "
                    "vector is not a park-settle wrapped stack"
                )
            macro_decisions[index] += 1
            native_ticks[index] += int(macro["ticks_consumed"])
            if bool(macro.get("released", False)):
                shots[index] += 1
            if not bool(dones[index]):
                continue
            outcome = info.get("outcome")
            win = outcome == "win"
            ticks = int(info.get("ticks", native_ticks[index]))
            row = {
                "schema": ROW_SCHEMA,
                "level_id": str(level_id),
                "seed": int(seed),
                "outcome": outcome,
                "native_outcome": info.get("native_outcome"),
                "time_limit_truncated": bool(
                    info.get("TimeLimit.truncated", False)
                ),
                "ticks": ticks,
                "score": int(info.get("score", 0)),
                "shots": int(shots[index]),
                "macro_decisions": int(macro_decisions[index]),
                "native_ticks_consumed": int(native_ticks[index]),
                "win_trigger_tick": ticks if win else None,
                "win_pending_at_termination": (
                    bool(info.get("win_pending", False)) if win else None
                ),
                "board_cleared_tick": None,
            }
            append_row(row)
            rows.append(row)
            shots[index] = 0
            macro_decisions[index] = 0
            native_ticks[index] = 0
            if queue:
                next_seed = queue.popleft()
                observations[index] = _reset_env_seeded(
                    vector, index, next_seed
                )
                active_seed[index] = next_seed
                if stochastic:
                    episode_rng[index] = stochastic_episode_rng(
                        next_seed
                    )
            else:
                active_seed[index] = None
                episode_rng[index] = None
    if any(seed is not None for seed in active_seed) or queue:
        unfinished = sorted(
            [seed for seed in active_seed if seed is not None] + list(queue)
        )
        raise RuntimeError(
            f"seed stream exceeded its iteration guard ({guard} macro "
            f"iterations) with seeds unfinished: {unfinished[:10]}"
        )
    if len(rows) != episodes_planned:
        raise RuntimeError(
            f"streamed {len(rows)} rows for {episodes_planned} pending "
            "seeds"
        )
    return {
        "rows": rows,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "active_macro_transitions": transitions,
            "parallel_envs": count,
            "total_native_ticks": int(
                sum(int(row["native_ticks_consumed"]) for row in rows)
            ),
        },
    }


def _load_completed_rows(
    results_path: Path, *, level_id: str
) -> tuple[list[dict[str, Any]], set[int]]:
    """Parse prior rows for resume; a torn FINAL line is truncated away.

    Rows are appended line-atomically (single write + flush + fsync),
    so the only expected corruption after a crash is a partial final
    line: it is dropped by truncating the file to the last complete
    line and its seed simply re-runs.  An unparseable line anywhere
    else, a level mismatch, or a duplicate seed refuses loudly.
    """

    if not results_path.exists():
        return [], set()
    blob = results_path.read_bytes()
    if not blob:
        return [], set()
    cut = blob.rfind(b"\n") + 1
    if cut != len(blob):
        with results_path.open("rb+") as stream:
            stream.truncate(cut)
            stream.flush()
            os.fsync(stream.fileno())
        blob = blob[:cut]
    rows: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for number, line in enumerate(blob.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"corrupt results line {number} in {results_path}: "
                f"{error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"results line {number} in {results_path} is not an "
                "object"
            )
        if str(row.get("level_id")) != str(level_id):
            raise ValueError(
                f"results line {number} belongs to level "
                f"{row.get('level_id')!r}, not {level_id!r}: refusing "
                "to mix runs"
            )
        seed = int(row["seed"])
        if seed in seeds:
            raise ValueError(
                f"results line {number} duplicates seed {seed}; two "
                "writers touched the same run dir"
            )
        seeds.add(seed)
        rows.append(row)
    return rows, seeds


def _append_row_line(results_path: Path, row: Mapping[str, Any]) -> None:
    line = json.dumps(
        v4._json_safe(dict(row)), ensure_ascii=False, allow_nan=False
    )
    with results_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_index(
    index_path: Path, *, rows: int, wins: int, last_seed: int | None
) -> None:
    base._write_atomic(
        index_path,
        {
            "schema": INDEX_SCHEMA,
            "version": 1,
            "rows": int(rows),
            "wins": int(wins),
            "last_appended_seed": (
                int(last_seed) if last_seed is not None else None
            ),
            "updated_utc": base._utc_now(),
        },
    )


def _park_settle_config(
    *, hold_ticks: int, settle_tolerance_bins: int | None
) -> ParkSettleActionConfig:
    """The effective wrapper config for the given CLI flags.

    ``None`` means the wrapper default
    (``DEFAULT_SETTLE_TOLERANCE_BINS``); constructing the dataclass
    runs its own validation eagerly, before any env exists.
    """

    if settle_tolerance_bins is None:
        return ParkSettleActionConfig(hold_ticks=int(hold_ticks))
    return ParkSettleActionConfig(
        hold_ticks=int(hold_ticks),
        settle_tolerance_bins=int(settle_tolerance_bins),
    )


def _make_search_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Sequence[int],
    settle_tolerance_bins: int | None,
) -> Callable[[], Any]:
    """native_eval's stacked macro factory, optionally re-configured.

    The imported factory's signature has no settle-tolerance knob and
    frozen files cannot change, so a non-default tolerance is applied
    by REWRAPPING: the imported factory builds its usual
    stack -> ParkSettleActionWrapper chain, then the wrapper is
    replaced by a fresh :class:`ParkSettleActionWrapper` over the SAME
    stacked env with a config carrying the requested tolerance
    (``hold_ticks`` preserved; the discarded wrapper holds no state and
    never steps).  With the default (``None``) the imported factory is
    returned untouched -- byte-identical to pre-flag behavior.
    """

    base_factory = native_eval._make_stacked_macro_env_factory(
        original_root=original_root,
        level_id=level_id,
        max_ticks=int(max_ticks),
        hold_ticks=int(hold_ticks),
        stack_lags=tuple(int(lag) for lag in stack_lags),
    )
    if settle_tolerance_bins is None:
        return base_factory
    config = _park_settle_config(
        hold_ticks=int(hold_ticks),
        settle_tolerance_bins=int(settle_tolerance_bins),
    )

    def make_env() -> ParkSettleActionWrapper:
        wrapped = base_factory()
        if not isinstance(wrapped, ParkSettleActionWrapper):
            raise RuntimeError(
                "native_eval._make_stacked_macro_env_factory no longer "
                f"returns a ParkSettleActionWrapper (got "
                f"{type(wrapped).__name__}); the settle-tolerance "
                "rewrap contract is broken"
            )
        return ParkSettleActionWrapper(wrapped.env, config=config)

    return make_env


def _build_vector(
    *,
    original_root: Path,
    level_id: str,
    env_count: int,
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Sequence[int],
    vec_env: str,
    settle_tolerance_bins: int | None = None,
) -> Any:
    """N copies of the same level behind the chosen vector backend.

    ``subproc`` is the production backend (one process per env);
    ``dummy`` runs in-process for CPU tests and debugging.  Both expose
    the per-env ``env_method("reset", seed=...)`` this tool relies on.
    """

    factory = _make_search_env_factory(
        original_root=original_root,
        level_id=level_id,
        max_ticks=int(max_ticks),
        hold_ticks=int(hold_ticks),
        stack_lags=tuple(int(lag) for lag in stack_lags),
        settle_tolerance_bins=settle_tolerance_bins,
    )
    factories = [factory] * int(env_count)
    if vec_env == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(factories, start_method="forkserver")
    if vec_env == "dummy":
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv(factories)
    raise ValueError(f"unknown vec_env backend: {vec_env!r}")


def run(
    *,
    model_path: Path,
    original_root: Path,
    run_dir: Path,
    level_id: str = DEFAULT_LEVEL,
    seed_start: int = DEFAULT_SEED_START,
    seed_count: int = DEFAULT_SEED_COUNT,
    max_ticks: int = DEFAULT_MAX_TICKS,
    hold_ticks: int = DEFAULT_HOLD_TICKS,
    stack_lags: Any = DEFAULT_STACK_LAGS,
    parallel_envs: int = DEFAULT_PARALLEL_ENVS,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
    vec_env: str = "subproc",
    seed_file: Path | None = None,
    seed_file_note: str | None = None,
    drop_embargoed: bool = False,
    allow_reserved: bool = False,
    stochastic_temperature: float | None = None,
    settle_tolerance_bins: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_path = model_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    lags = parse_stack_lags(stack_lags)
    (level,) = v4._validate_levels((str(level_id),))
    if int(parallel_envs) < 1:
        raise ValueError("parallel_envs must be positive")
    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    if stochastic_temperature is not None:
        stochastic_temperature = _validate_stochastic_temperature(
            stochastic_temperature
        )
    if settle_tolerance_bins is not None:
        # Fail fast on the wrapper config's own validation before any
        # receipts or envs exist (the aim-bins upper bound is checked
        # again by the wrapper itself at env construction).
        _park_settle_config(
            hold_ticks=int(hold_ticks),
            settle_tolerance_bins=int(settle_tolerance_bins),
        )
    source = {
        "model_path": str(model_path),
        "model_sha256": base._sha256(model_path),
    }
    # The resume identity: everything that changes WHAT is measured.
    # Throughput knobs (parallel_envs, device, vec_env, top_k) may vary
    # between invocations of the same run dir.
    seed_source: dict[str, Any] | None = None
    if seed_file is not None:
        if (
            int(seed_start) != DEFAULT_SEED_START
            or int(seed_count) != DEFAULT_SEED_COUNT
        ):
            raise ValueError(
                "--seed-file is mutually exclusive with --seed-start/"
                "--seed-count"
            )
        seed_file = Path(seed_file).resolve(strict=True)
        requested = load_seed_file(seed_file)
        planned_seeds, dropped_embargoed, reserved_overlaps = (
            validate_seed_list(
                requested,
                drop_embargoed=bool(drop_embargoed),
                allow_reserved=bool(allow_reserved),
            )
        )
        convention = SEED_LIST_CONVENTION
        convention_note = SEED_LIST_CONVENTION_NOTE
        seed_source = {
            "mode": SEED_LIST_CONVENTION,
            "seed_file": _seed_file_provenance(seed_file),
            "requested_seeds": len(requested),
            "planned_seeds": len(planned_seeds),
            "dropped_embargoed": dropped_embargoed,
            "reserved_overlaps": reserved_overlaps,
            "note": seed_file_note,
        }
        identity = {
            "level": str(level),
            "convention": SEED_LIST_CONVENTION,
            "seed_file_sha256": seed_source["seed_file"]["sha256"],
            "seed_count": len(planned_seeds),
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "stack_lags": [int(lag) for lag in lags],
            "model_sha256": source["model_sha256"],
        }
    else:
        first, last = validate_seed_range(seed_start, seed_count)
        planned_seeds = list(range(first, last + 1))
        convention = BEST_OF_N_CONVENTION
        convention_note = SPEEDRUN_CONVENTION_NOTE
        identity = {
            "level": str(level),
            "seed_start": int(first),
            "seed_count": int(seed_count),
            "seed_last": int(last),
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "stack_lags": [int(lag) for lag in lags],
            "model_sha256": source["model_sha256"],
        }
    # Record-mode flags change WHAT is measured, so when set they join
    # the resume identity and the convention label; when absent every
    # receipt stays byte-compatible with the prior conventions.
    stochastic_payload: dict[str, Any] | None = None
    if stochastic_temperature is not None:
        convention = _stochastic_convention(
            convention, stochastic_temperature
        )
        identity["stochastic_temperature"] = float(stochastic_temperature)
        stochastic_payload = {
            "temperature": float(stochastic_temperature),
            "rng_scheme": STOCHASTIC_RNG_SCHEME,
            "rng_stream_tag": int(STOCHASTIC_RNG_STREAM_TAG),
            "mask_rule": (
                "the wrapper's own (3 + 180) action mask is applied to "
                "the raw policy logits (masked-off entries dropped, the "
                "survivors renormalized by the softmax); temperature "
                "divides logits before the softmax"
            ),
            "replaces": (
                "masked deterministic argmax predict "
                "(native_interface.macro_interface)"
            ),
        }
    settle_payload: dict[str, Any] | None = None
    if settle_tolerance_bins is not None:
        identity["settle_tolerance_bins"] = int(settle_tolerance_bins)
        settle_payload = {
            "requested": int(settle_tolerance_bins),
            "effective": int(settle_tolerance_bins),
            "wrapper_default": int(DEFAULT_SETTLE_TOLERANCE_BINS),
            "echo_source": (
                "wrapper_contract.config.settle_tolerance_bins "
                "(compared against the requested value whenever envs "
                "are constructed; a mismatch fails the run)"
            ),
        }
    config_path = run_dir / "config.json"
    results_path = run_dir / RESULTS_FILENAME
    if run_dir.exists():
        if not config_path.exists():
            raise FileExistsError(
                f"run dir exists without a config receipt: {run_dir}"
            )
        recorded = base._read(config_path).get("resume_identity")
        if recorded != identity:
            raise ValueError(
                "resume identity mismatch: the existing run dir was "
                f"created with {recorded}, this invocation asks for "
                f"{identity}; refusing to mix results"
            )
    else:
        run_dir.mkdir(parents=True)
        config_payload: dict[str, Any] = {
            "schema": f"{COMPLETION_SCHEMA}-config",
            "version": 1,
            "status": "FROZEN",
            "probe": {
                "path": str(SCRIPT_PATH),
                "sha256": base._sha256(SCRIPT_PATH),
            },
            "native_eval_probe": {
                "module": (
                    "tools.probe_alphazuma_55_macro_native_eval_v1"
                ),
                "path": str(native_eval.SCRIPT_PATH),
                "sha256": base._sha256(native_eval.SCRIPT_PATH),
            },
            "wrapper_module": {
                "path": str(macro_eval.WRAPPER_PATH),
                "sha256": base._sha256(macro_eval.WRAPPER_PATH),
            },
            "observation_stack_wrapper": {
                "path": str(native_eval.STACK_WRAPPER_PATH),
                "sha256": base._sha256(native_eval.STACK_WRAPPER_PATH),
            },
            "source": source,
            "mode": MODE,
            "convention": convention,
            "seed_source": seed_source,
            "resume_identity": identity,
            "device": str(device),
            "vec_env": str(vec_env),
            "parallel_envs": int(parallel_envs),
            "top_k": int(top_k),
            "native_interface": native_eval._native_interface_contract(
                int(hold_ticks)
            ),
            "seed_discipline": _seed_discipline_payload(),
            "timing_convention": _timing_convention(convention_note),
            "convention_note": convention_note,
        }
        if stochastic_payload is not None:
            config_payload["stochastic_sampling"] = dict(
                stochastic_payload
            )
        if settle_payload is not None:
            config_payload["settle_tolerance_bins"] = dict(settle_payload)
        base._write_atomic(config_path, config_payload)
    resumed_rows, completed = _load_completed_rows(
        results_path, level_id=level
    )
    planned_set = set(planned_seeds)
    stray = sorted(seed for seed in completed if seed not in planned_set)
    if stray:
        raise ValueError(
            f"{RESULTS_FILENAME} contains seeds outside the plan: "
            f"{stray[:10]}"
        )
    pending = [
        seed for seed in planned_seeds if seed not in completed
    ]
    base._write_atomic(
        run_dir / "status.json",
        {
            "schema": f"{COMPLETION_SCHEMA}-status",
            "version": 1,
            "status": "RUNNING",
            "stage": MODE,
            "resumed_rows": len(resumed_rows),
            "pending_seeds": len(pending),
            "updated_utc": base._utc_now(),
        },
    )
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(str(model_path), device=str(device))
        native_eval.require_macro_native_model(model)
        stacked_width = int(model.observation_space.shape[0])
        if stacked_width % len(lags):
            raise ValueError(
                f"model observation width {stacked_width} is not "
                f"divisible by {len(lags)} stack frames; --stack-lags "
                "disagrees with the trained policy"
            )
        raw_observation_dim = stacked_width // len(lags)
        model_spaces = native_eval.model_space_signature(model)
        wrapper_contract: dict[str, Any] | None = None
        new_rows: list[dict[str, Any]] = []
        stream_runtime: dict[str, Any] | None = None
        env_count = min(int(parallel_envs), len(pending))
        if pending:
            vector = _build_vector(
                original_root=original_root,
                level_id=level,
                env_count=env_count,
                max_ticks=int(max_ticks),
                hold_ticks=int(hold_ticks),
                stack_lags=lags,
                vec_env=str(vec_env),
                settle_tolerance_bins=settle_tolerance_bins,
            )
            try:
                vector_width = int(vector.observation_space.shape[0])
                if vector_width != stacked_width:
                    raise ValueError(
                        f"stacked env width {vector_width} disagrees "
                        f"with the model's {stacked_width}"
                    )
                wrapper_contract = vector.env_method("contract")[0]
                if settle_tolerance_bins is not None:
                    echoed = int(
                        wrapper_contract["config"][
                            "settle_tolerance_bins"
                        ]
                    )
                    if echoed != int(settle_tolerance_bins):
                        raise RuntimeError(
                            "wrapper contract echoes "
                            f"settle_tolerance_bins {echoed}, not the "
                            f"requested {int(settle_tolerance_bins)}; "
                            "the env factory did not receive the "
                            "record-mode config"
                        )
                counters = {
                    "rows": len(resumed_rows),
                    "wins": sum(
                        1
                        for row in resumed_rows
                        if row.get("outcome") == "win"
                    ),
                }

                def _append(row: Mapping[str, Any]) -> None:
                    _append_row_line(results_path, row)
                    counters["rows"] += 1
                    if row.get("outcome") == "win":
                        counters["wins"] += 1
                    _write_index(
                        run_dir / INDEX_FILENAME,
                        rows=counters["rows"],
                        wins=counters["wins"],
                        last_seed=int(row["seed"]),
                    )

                stream = run_seed_stream(
                    model=model,
                    vector=vector,
                    level_id=level,
                    pending_seeds=pending,
                    max_ticks=int(max_ticks),
                    append_row=_append,
                    stochastic_temperature=stochastic_temperature,
                )
            finally:
                vector.close()
            new_rows = list(stream["rows"])
            stream_runtime = dict(stream["runtime"])
        all_rows = sorted(
            [*resumed_rows, *new_rows], key=lambda row: int(row["seed"])
        )
        if len(all_rows) != len(planned_seeds):
            raise RuntimeError(
                f"{len(all_rows)} rows for {len(planned_seeds)} "
                "planned seeds after streaming; results.jsonl is "
                "inconsistent"
            )
        summary = summarize_rows(all_rows, int(top_k))
        _write_index(
            run_dir / INDEX_FILENAME,
            rows=summary["total_seeds"],
            wins=summary["wins"],
            last_seed=int(all_rows[-1]["seed"]) if all_rows else None,
        )
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "vec_env": str(vec_env),
            "parallel_envs_requested": int(parallel_envs),
            "parallel_envs_used": int(env_count) if pending else 0,
        }
        if str(device).startswith("cuda"):
            runtime["cuda_device"] = torch.cuda.get_device_name(
                torch.device(str(device))
            )
        completion_payload: dict[str, Any] = (
            {
                "schema": COMPLETION_SCHEMA,
                "version": 1,
                "status": "COMPLETE",
                "completed_utc": base._utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "mode": MODE,
                "convention": convention,
                "level": str(level),
                "seed_start": (
                    int(first) if seed_file is None else None
                ),
                "seed_count": len(planned_seeds),
                "seed_last": (
                    int(last) if seed_file is None else None
                ),
                "seed_source": seed_source,
                "max_ticks": int(max_ticks),
                "hold_ticks": int(hold_ticks),
                "stack_lags": [int(lag) for lag in lags],
                "parallel_envs": int(parallel_envs),
                "device": str(device),
                "vec_env": str(vec_env),
                "model_sha256": source["model_sha256"],
                "source": source,
                "model_spaces": dict(model_spaces),
                "observation_stack": native_eval._observation_stack_payload(
                    stack_lags=lags,
                    raw_observation_dim=raw_observation_dim,
                ),
                "wrapper_contract": wrapper_contract,
                "native_interface": (
                    native_eval._native_interface_contract(int(hold_ticks))
                ),
                "results_file": RESULTS_FILENAME,
                "resumed_rows": len(resumed_rows),
                "new_rows": len(new_rows),
                **summary,
                "stream_runtime": stream_runtime,
                "runtime": runtime,
                "seed_discipline": _seed_discipline_payload(),
                "timing_convention": _timing_convention(convention_note),
                "convention_note": convention_note,
                "formal_seed_consumption": False,
                "formal_candidate_authority": False,
                "training_authority": False,
            }
        )
        if stochastic_payload is not None:
            completion_payload["stochastic_sampling"] = dict(
                stochastic_payload
            )
        if settle_payload is not None:
            completion_payload["settle_tolerance_bins"] = dict(
                settle_payload
            )
        completion = v4._json_safe(completion_payload)
        base._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base._write_atomic(
            run_dir / "failure.json",
            {
                "schema": f"{COMPLETION_SCHEMA}-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": base._utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
                "training_authority": False,
            },
        )
        raise
    finally:
        del model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        help="the SINGLE level id every streamed episode plays",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help=(
            f"first seed of a contiguous range (default "
            f"{DEFAULT_SEED_START}); mutually exclusive with "
            "--seed-file"
        ),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help=(
            f"range length (default {DEFAULT_SEED_COUNT}); mutually "
            "exclusive with --seed-file"
        ),
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help=(
            "explicit JSON/JSONL seed list (the golden-seed "
            "enumerator's top-K format); labels the run "
            f"convention={SEED_LIST_CONVENTION!r} and is mutually "
            "exclusive with --seed-start/--seed-count"
        ),
    )
    parser.add_argument(
        "--seed-file-note",
        default=None,
        help="free-text provenance note recorded in the receipts",
    )
    parser.add_argument(
        "--drop-embargoed",
        action="store_true",
        help=(
            "with --seed-file: skip (and record) seeds inside frozen "
            "formal embargoes instead of refusing the whole list"
        ),
    )
    parser.add_argument(
        "--allow-reserved",
        action="store_true",
        help=(
            "with --seed-file: run seeds inside reserved engineering "
            "blocks anyway (overlaps are recorded in the receipts)"
        ),
    )
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags the policy was trained "
            "with (feature-axis stack UNDER the macro wrapper); must "
            "start with 0 and match the checkpoint's training lags"
        ),
    )
    parser.add_argument(
        "--stochastic-temperature",
        type=float,
        default=None,
        help=(
            "RECORD MODE: sample each macro from the policy "
            "distribution instead of the default masked deterministic "
            "argmax; the wrapper's own action mask is applied to the "
            "raw logits (masked-off entries dropped, survivors "
            "renormalized), logits are divided by this temperature "
            "before the softmax, and draws come from a per-episode RNG "
            "derived from the episode seed, so identical CLIs "
            "reproduce bit-for-bit; the run convention gains a "
            "'_stochasticT<temp>' suffix"
        ),
    )
    parser.add_argument(
        "--settle-tolerance-bins",
        type=int,
        default=None,
        help=(
            "RECORD MODE: override the park-settle wrapper's "
            "settle_tolerance_bins at env construction (wrapper "
            f"default {DEFAULT_SETTLE_TOLERANCE_BINS}); the effective "
            "value is echoed back by the wrapper contract recorded in "
            "completion.json"
        ),
    )
    parser.add_argument(
        "--hold-ticks", type=int, default=DEFAULT_HOLD_TICKS
    )
    parser.add_argument(
        "--max-ticks", type=int, default=DEFAULT_MAX_TICKS
    )
    parser.add_argument(
        "--parallel-envs", type=int, default=DEFAULT_PARALLEL_ENVS
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help=(
            "vector backend: subproc (production) or dummy (in-process "
            "CPU tests/debugging)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed_file is not None and (
        args.seed_start is not None or args.seed_count is not None
    ):
        parser.error(
            "--seed-file is mutually exclusive with --seed-start/"
            "--seed-count"
        )
    if args.seed_file is None and (
        args.drop_embargoed or args.allow_reserved
    ):
        parser.error(
            "--drop-embargoed/--allow-reserved require --seed-file"
        )
    completion = run(
        model_path=args.model_path.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        run_dir=args.run_dir.expanduser(),
        level_id=str(args.level),
        seed_start=(
            DEFAULT_SEED_START
            if args.seed_start is None
            else int(args.seed_start)
        ),
        seed_count=(
            DEFAULT_SEED_COUNT
            if args.seed_count is None
            else int(args.seed_count)
        ),
        max_ticks=int(args.max_ticks),
        hold_ticks=int(args.hold_ticks),
        stack_lags=str(args.stack_lags),
        parallel_envs=int(args.parallel_envs),
        top_k=int(args.top_k),
        device=str(args.device),
        vec_env=str(args.vec_env),
        seed_file=(
            args.seed_file.expanduser().resolve(strict=True)
            if args.seed_file is not None
            else None
        ),
        seed_file_note=args.seed_file_note,
        drop_embargoed=bool(args.drop_embargoed),
        allow_reserved=bool(args.allow_reserved),
        stochastic_temperature=(
            None
            if args.stochastic_temperature is None
            else float(args.stochastic_temperature)
        ),
        settle_tolerance_bins=(
            None
            if args.settle_tolerance_bins is None
            else int(args.settle_tolerance_bins)
        ),
    )
    fastest = completion["top_k_fastest_wins"]
    print(
        json.dumps(
            {
                "status": completion["status"],
                "level": completion["level"],
                "total_seeds": completion["total_seeds"],
                "wins": completion["wins"],
                "win_rate": completion["win_rate"],
                "win_tick_percentiles": completion[
                    "win_tick_percentiles"
                ],
                "fastest_win": fastest[0] if fastest else None,
                "top_k_fastest_wins": fastest,
                "resumed_rows": completion["resumed_rows"],
                "new_rows": completion["new_rows"],
                "wall_seconds": completion["wall_seconds"],
                "convention": completion["convention"],
                "convention_note": completion["convention_note"],
                "formal_seed_consumption": False,
                "training_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
