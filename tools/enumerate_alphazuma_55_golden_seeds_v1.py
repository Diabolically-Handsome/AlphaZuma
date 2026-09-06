"""Exhaustively enumerate golden-opening seeds per level (AI-TAS lane).

The per-level RNG is the exact retail MTRand
(``zuma_rl.revenge_core.PopCapMTRandom``, src/zuma_rl/revenge_core.py:590,
including the signed-positive ``& 0x7FFFFFFF`` output quirk at
revenge_core.py:685).  A level's opening chain colors are a pure function
of the episode seed, so the globally best "golden opening" seeds out of
all 2^32 can be found OFFLINE with a numpy-vectorized MTRand batch --
no episodes are played and no model is involved.  The ranked top-K list
feeds ``tools/search_alphazuma_55_macro_speedrun_seeds_v1.py --seed-file``
(convention ``seed_list_tas``, honestly labeled).

Seed path, verified end-to-end (citations are file:line in this repo)
----------------------------------------------------------------------
1. ``RevengeEnv.reset(seed=...)`` (src/zuma_rl/revenge_env.py:803-810)
   calls ``super().reset(seed=seed)`` (gymnasium ``np_random`` only --
   never touches the game RNG) and then ``self.sim.reset(seed=seed)``.
   The constructor path is equivalent: ``RevengeSimulator.from_installed
   (..., seed=seed)`` or ``simulator.reset(seed=seed)``
   (revenge_env.py:212-230).
2. ``RevengeSimulator.reset`` (src/zuma_rl/revenge_core.py:1485-1491)
   sets ``_initial_mtrand_seed = int(seed) & 0xFFFFFFFF`` and calls
   ``self.rng.seed(...)`` on the shared global ``PopCapMTRandom``.  The
   same masked value also seeds the MSVC-CRT stream, but that stream
   only drives the SHOOTER gun colors through ``_BalancedColorChooser``
   (revenge_core.py:1036-1039, 1492-1495) -- never the chain feed.
3. ``PopCapMTRandom.seed`` (revenge_core.py:614-635): value is masked
   ``& 0xFFFFFFFF`` again and **seed 0 is replaced by 4357**
   (revenge_core.py:621-622), then the 624-word state is expanded with
   the Knuth multiplier 1812433253.  The env-episode-seed -> MTRand-seed
   mapping is therefore 1:1 on [0, 2^32) EXCEPT for the single alias
   seed 0 == seed 4357 (identical streams, identical openings).
4. Outputs: ``next_u31`` (revenge_core.py:676-685) = standard MT19937
   twist + tempering, then ``& 0x7FFFFFFF`` (the retail signed-positive
   quirk).  ``rand_mod(m)`` = ``next_u31() % m`` (revenge_core.py:687).

Opening draw layout, verified from level-init code
---------------------------------------------------
``RevengeSimulator.reset`` re-runs ``_reset_active_curve_runtime`` for
every curve in order (revenge_core.py:1522-1525), so curve 0's opening
draws always come FIRST on the freshly seeded stream.  This module
predicts and scores curve 0's opening: the WHOLE opening on
single-curve levels (every Jungle-zone campaign level except the
two-curve Jungle9), and the first-reset curve's opening on multi-curve
levels (later curves consume later draws and never shift curve 0's).
``_reset_active_curve_runtime`` (revenge_core.py:1475-1483) queues
``pending_count = min(num_balls, initial_pending_balls)`` when the
curve's ``num_balls`` parameter is positive, else
``initial_pending_balls`` = 10 (revenge_core.py:124, 1475-1478),
opening balls via ``_append_pending_color``
(revenge_core.py:1946-1991).  ``_roll_in_speed``
(revenge_core.py:1884-1912) consumes NO RNG.

Per ``_append_pending_color`` call the global MTRand draw order is:

* ball 0 only: ``previous = next_u31 % colors`` (revenge_core.py:1952).
  The re-draw guard at revenge_core.py:1953-1954 (``previous >=
  colors``) can NEVER fire at level init because every previous value
  is already reduced ``% colors``.
* every ball: ``roll = next_u31 % 100`` (revenge_core.py:1960).
* branch (revenge_core.py:1963-1979), consuming draws only in (d):
  (a) ``roll <= ball_repeat_chance`` AND tail-run(previous) <
      ``max_clump_size`` (``_pending_run_length``,
      revenge_core.py:1914-1924) -> repeat previous, NO draw;
  (b) elif the singles guard -- ``max_single < 10`` and
      ``_num_pending_singles(1) == 1`` and (``max_single == 0`` or
      ``_num_pending_singles(10) > max_single``)
      (``_num_pending_singles``, revenge_core.py:1926-1944) -> repeat
      previous, NO draw;
  (c) elif ``colors == 1`` -> color 0, NO draw;
  (d) else rejection loop: ``next_u31 % colors`` repeated until the
      value differs from previous (>= 1 draw, geometric tail).
* every ball: ONE unconditional sprite-frame ``next_u31``
  (revenge_core.py:1981-1985) -- rendering-only but shared-stream, so
  it must be consumed or every later color shifts.

Chain layout mapping: ``_add_ball`` (revenge_core.py:2026-2077) pops
``pending_colors[0]`` and inserts at ``balls[0]``, so the FIRST queued
color is the FRONT (lead) ball and the opening chain front->rear order
equals the pending-queue order predicted here.

Per-level draw parameters, DERIVED at run time (never hardcoded)
-----------------------------------------------------------------
``derive_level_draw_params`` constructs the level through the exact
catalog path the live check uses (``RevengeSimulator.from_installed``
-> ``OriginalGameCatalog.load_level``) and reads curve 0's decoded
CURV-header parameters (``CurveParameters``,
src/zuma_rl/original_data.py:139-159): ``colors``
(``active_num_colors``, revenge_core.py:1211-1215),
``ball_repeat_chance``, ``max_clump_size``, ``max_single``, and
``num_balls`` -- exactly the values ``_append_pending_color`` reads at
draw time through ``self.parameters`` = the ACTIVE curve's parameters
(revenge_core.py:1197-1198, 1956-1958, 1988) -- plus
``config.initial_pending_balls`` (revenge_core.py:124) and
``curve_count``.  Only members of the frozen
``zuma_rl.alphazuma_55.INCLUDED_LEVELS`` campaign set are derivable;
anything else fails closed, as does any level whose positive
``num_balls`` would truncate the opening below
``initial_pending_balls``.  Derivation alone is NOT trusted: an exotic
level could violate the modeled layout, so the mandatory live check
(``--original-root``: real ``reset(seed)`` openings compared
bit-for-bit against the predictor before any enumeration) remains the
gate.  The scalar reference predictor was validated against the real
simulator on Jungle1 over 307 seeds (0 mismatches; max MTRand
consumption observed: 33 draws, comfortably under the 64 precomputed
outputs) and is re-verified per level by the live check and the
real-env regression tests.

Vectorized batch MTRand (bitwise-identical, proven by parity test)
-------------------------------------------------------------------
``batch_mtrand_u31`` expands the 624-word state for B seeds at once
(vectorized across the batch axis) and produces the first N outputs
with exact tempering plus the signed-positive quirk.  Because the first
227 twisted words depend only on PRE-twist state words, the first
N <= 227 outputs need only state words 0..N+396, so the seeding
recurrence stops early and the twist is a single fully-vectorized
slice expression.  ``tests/test_enumerate_alphazuma_55_golden_seeds_v1
.py`` proves bitwise equality against ``PopCapMTRandom`` over > 1000
seeds x 64 draws.

Cascade-friendliness score (simple, documented, weight-configurable)
---------------------------------------------------------------------
Computed on the opening color sequence in chain order::

    score = w_longest_run * longest_run        # longest same-color run
          + w_runs3       * runs_ge3           # count of runs >= 3
          + w_adjacency   * adjacency          # adjacent equal pairs
          + w_distinct    * distinct_colors    # fewer distinct better

Defaults: w_longest_run=2.0, w_runs3=3.0, w_adjacency=1.0,
w_distinct=-2.0 (negative: fewer distinct colors is better).  Only the
RANKING matters; ties break by ascending seed so results are fully
deterministic and independent of chunking/threads.

Enumeration is chunked and resumable: per-chunk ``.npz`` checkpoints,
an atomically-rewritten merged running top-K, ``progress.json``, and a
``completion.json`` with receipts.  Prints
``CHUNK i/256 done rate=X seeds/s topscore=Y`` after every chunk.

No episodes are played by this tool: enumerating a seed's opening
colors does NOT consume any (level, seed) eval key, so the frozen
formal-seed embargoes are untouched (the downstream search tool
enforces them before any episode runs).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from zuma_rl.revenge_core import PopCapMTRandom


SCRIPT_PATH = Path(__file__).resolve()
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-golden-seed-enumeration-v1"
CONFIG_SCHEMA = f"{COMPLETION_SCHEMA}-config"
PROGRESS_SCHEMA = f"{COMPLETION_SCHEMA}-progress"
TOPK_ROW_SCHEMA = f"{COMPLETION_SCHEMA}-topk-row"
CHUNK_DIRNAME = "chunks"
TOPK_FILENAME = "top_k.jsonl"
RUNNING_TOPK_FILENAME = "topk_running.npz"
PROGRESS_FILENAME = "progress.json"
CONFIG_FILENAME = "config.json"
COMPLETION_FILENAME = "completion.json"
CONVENTION = "golden_opening_enumeration"

SEED_SPACE = 1 << 32
DEFAULT_CHUNK_SIZE = 1 << 24
DEFAULT_TOP_K = 100_000
DEFAULT_THREADS = 12
DEFAULT_BATCH_SIZE = 1 << 16
DEFAULT_MTRAND_OUTPUTS = 64
# The single-twist shortcut is exact only for the first 227 outputs
# (see batch_mtrand_u31).
MAX_MTRAND_OUTPUTS = 227
LIVE_CHECK_SAMPLER_SEED = 99_081_701

# MT19937 constants, byte for byte the PopCapMTRandom ones
# (src/zuma_rl/revenge_core.py:595-597, 624-633, 676-685).
_KNUTH = np.uint32(1_812_433_253)
_UPPER_MASK = np.uint32(0x8000_0000)
_LOWER_MASK = np.uint32(0x7FFF_FFFF)
_MATRIX_A = np.uint32(0x9908_B0DF)
_TEMPER_B = np.uint32(0x9D2C_5680)
_TEMPER_C = np.uint32(0xEFC6_0000)
_PERIOD_OFFSET = 397

DRAW_LAYOUT_CITATIONS = (
    "src/zuma_rl/revenge_env.py:803-810 (RevengeEnv.reset -> "
    "sim.reset(seed))",
    "src/zuma_rl/revenge_core.py:1485-1491 (seed & 0xFFFFFFFF -> "
    "PopCapMTRandom.seed)",
    "src/zuma_rl/revenge_core.py:614-635 (MTRand seeding; 0 -> 4357 "
    "alias)",
    "src/zuma_rl/revenge_core.py:676-685 (next_u31 tempering + "
    "signed-positive & 0x7FFFFFFF)",
    "src/zuma_rl/revenge_core.py:1522-1525 (reset re-runs every curve "
    "in order; curve 0's opening draws come first)",
    "src/zuma_rl/revenge_core.py:1475-1483 (pending_count = "
    "min(num_balls, initial_pending_balls) if num_balls > 0 else "
    "initial_pending_balls; _roll_in_speed consumes no RNG)",
    "src/zuma_rl/revenge_core.py:124 (initial_pending_balls = 10)",
    "src/zuma_rl/revenge_core.py:1197-1215 (parameters / "
    "active_num_colors read the ACTIVE curve's CURV parameters)",
    "src/zuma_rl/original_data.py:139-159 (CurveParameters decoded "
    "from the CURV header by OriginalGameCatalog.load_level)",
    "src/zuma_rl/revenge_core.py:1946-1991 (_append_pending_color: "
    "previous draw [ball 0], percent roll, repeat/singles-guard/"
    "rejection branch, sprite-frame draw)",
    "src/zuma_rl/revenge_core.py:1914-1924 (_pending_run_length)",
    "src/zuma_rl/revenge_core.py:1926-1944 (_num_pending_singles)",
    "src/zuma_rl/revenge_core.py:2026-2077 (_add_ball: pending order "
    "== chain front->rear order)",
)


@dataclass(frozen=True)
class LevelDrawParams:
    """Per-level parameters of the opening draw layout.

    Produced by ``derive_level_draw_params`` from the real level data;
    the live check re-reads the same values from a fresh simulator and
    verifies predicted openings bit-for-bit before enumeration.
    """

    colors: int
    ball_repeat_chance: int
    max_clump_size: int
    max_single: int
    num_balls: int
    opening_balls: int
    curve_count: int
    provenance: str


_DERIVED_PARAMS_CACHE: dict[tuple[str, str | None], LevelDrawParams] = {}


def included_levels() -> tuple[str, ...]:
    """The frozen AlphaZuma-55 campaign level set (lazy import)."""

    from zuma_rl.alphazuma_55 import INCLUDED_LEVELS

    return tuple(INCLUDED_LEVELS)


def derive_level_draw_params(
    level_id: str,
    *,
    original_root: Path | None = None,
) -> LevelDrawParams:
    """Derive a level's opening draw layout from the REAL level data.

    Constructs the level through the exact catalog path the live check
    uses (``RevengeSimulator.from_installed`` ->
    ``OriginalGameCatalog.load_level``; ``original_root=None`` uses the
    catalog's own installation discovery) and extracts curve 0's
    CURV-header parameters plus ``initial_pending_balls`` and
    ``curve_count`` (see the module docstring for citations).  Fails
    closed on levels outside the frozen campaign set and on layouts
    the predictor does not model (positive ``num_balls`` below
    ``initial_pending_balls``).  Derivation may still be wrong for
    exotic levels -- the mandatory live check remains the gate.
    """

    level_key = str(level_id)
    known = included_levels()
    if level_key not in known:
        raise ValueError(
            f"level {level_key!r} is not in the frozen AlphaZuma-55 "
            f"INCLUDED_LEVELS campaign set ({len(known)} levels); "
            "refusing to derive a draw layout outside it"
        )
    cache_key = (
        level_key,
        (
            str(Path(original_root).resolve())
            if original_root is not None
            else None
        ),
    )
    cached = _DERIVED_PARAMS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from zuma_rl.revenge_core import RevengeSimulator

    simulator = RevengeSimulator.from_installed(
        level_key, root=original_root, seed=1
    )
    num_balls = int(getattr(simulator.parameters, "num_balls", 0))
    opening_balls = int(simulator.config.initial_pending_balls)
    if 0 < num_balls < opening_balls:
        raise ValueError(
            f"level {level_key!r} caps its opening at num_balls="
            f"{num_balls} < initial_pending_balls={opening_balls}; the "
            "predictor and live check do not model truncated openings"
        )
    root_note = (
        f"root {cache_key[1]}"
        if cache_key[1] is not None
        else "the auto-discovered installation root"
    )
    params = LevelDrawParams(
        colors=int(simulator.active_num_colors),
        ball_repeat_chance=int(
            getattr(simulator.parameters, "ball_repeat_chance")
        ),
        max_clump_size=int(
            getattr(simulator.parameters, "max_clump_size")
        ),
        max_single=int(getattr(simulator.parameters, "max_single")),
        num_balls=num_balls,
        opening_balls=opening_balls,
        curve_count=int(simulator.curve_count),
        provenance=(
            "derived at run time from RevengeSimulator.from_installed("
            f"{level_key!r}) curve-0 CURV parameters under {root_note}; "
            "gated by the mandatory live check when --original-root "
            "is given"
        ),
    )
    _DERIVED_PARAMS_CACHE[cache_key] = params
    return params


@dataclass(frozen=True)
class OpeningScoreWeights:
    """Cascade-friendliness weights; only the RANKING matters."""

    longest_run: float = 2.0
    runs3: float = 3.0
    adjacency: float = 1.0
    distinct: float = -2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    )
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Vectorized batch MTRand (bitwise-identical to PopCapMTRandom).
# ---------------------------------------------------------------------------


def batch_mtrand_u31(
    seeds: Any, n_outputs: int = DEFAULT_MTRAND_OUTPUTS
) -> np.ndarray:
    """First ``n_outputs`` ``next_u31`` values for a batch of seeds.

    Bitwise-identical to ``PopCapMTRandom`` (revenge_core.py:590-695):
    same ``& 0xFFFFFFFF`` seed mask, same 0 -> 4357 alias, same Knuth
    state expansion, same twist, same tempering, same signed-positive
    ``& 0x7FFFFFFF``.  Restricted to ``n_outputs <= 227`` because the
    first 227 twisted words depend only on PRE-twist state words
    (``mt[i]`` for ``i < 227`` reads old ``mt[i]``, ``mt[i+1]``,
    ``mt[i+397]``), which keeps the whole twist one vectorized slice
    expression; the state recurrence is likewise stopped early at word
    ``n_outputs + 396``.
    """

    if not 1 <= int(n_outputs) <= MAX_MTRAND_OUTPUTS:
        raise ValueError(
            f"n_outputs must be in [1, {MAX_MTRAND_OUTPUTS}]"
        )
    n_outputs = int(n_outputs)
    seed_array = np.asarray(seeds)
    if seed_array.ndim != 1:
        raise ValueError("seeds must be a 1-D array")
    masked = (
        seed_array.astype(np.uint64) & np.uint64(0xFFFF_FFFF)
    ).astype(np.uint32)
    masked = np.where(masked == np.uint32(0), np.uint32(4357), masked)
    batch = masked.shape[0]
    words_needed = n_outputs + _PERIOD_OFFSET
    # State is laid out (word, batch) so every recurrence step writes
    # one contiguous row (cache-friendly), with in-place ufuncs to
    # avoid per-step temporaries.
    state = np.empty((words_needed, batch), dtype=np.uint32)
    state[0] = masked
    previous = state[0]
    for index in range(1, words_needed):
        row = state[index]
        np.right_shift(previous, 30, out=row)
        np.bitwise_xor(row, previous, out=row)
        np.multiply(row, _KNUTH, out=row)
        np.add(row, np.uint32(index), out=row)
        previous = row
    combined = (state[:n_outputs] & _UPPER_MASK) | (
        state[1 : n_outputs + 1] & _LOWER_MASK
    )
    magnitude = np.where(
        (combined & np.uint32(1)).astype(bool), _MATRIX_A, np.uint32(0)
    )
    twisted = (
        state[_PERIOD_OFFSET : _PERIOD_OFFSET + n_outputs]
        ^ (combined >> np.uint32(1))
        ^ magnitude
    )
    twisted ^= twisted >> np.uint32(11)
    twisted ^= (twisted << np.uint32(7)) & _TEMPER_B
    twisted ^= (twisted << np.uint32(15)) & _TEMPER_C
    twisted ^= twisted >> np.uint32(18)
    twisted &= _LOWER_MASK
    return np.ascontiguousarray(twisted.T)


# ---------------------------------------------------------------------------
# Opening-color prediction: scalar reference + vectorized batch.
# ---------------------------------------------------------------------------


def _num_pending_singles_scalar(
    pending: Sequence[int], group_limit: int
) -> int:
    """Exact mirror of ``_num_pending_singles`` (revenge_core.py:1926).

    At level init the chain is empty, so the sequence is just the
    reversed pending queue.
    """

    group_count = 0
    previous = -1
    singles = 0
    run_length = 0
    for color in reversed(pending):
        if group_count > group_limit:
            break
        if color != previous:
            if run_length == 1:
                singles += 1
            run_length = 1
            group_count += 1
            previous = color
        else:
            run_length += 1
    return singles


def predict_opening_colors_scalar(
    seed: int, params: LevelDrawParams
) -> list[int]:
    """Reference predictor: exact ``_append_pending_color`` replay.

    Uses the real ``PopCapMTRandom`` and mirrors
    revenge_core.py:1946-1991 draw for draw (see the module docstring
    for the layout).  Validated against real ``RevengeSimulator.reset``
    pending colors by the draw-layout regression test.
    """

    rng = PopCapMTRandom(int(seed))
    colors = int(params.colors)
    pending: list[int] = []
    for _ in range(int(params.opening_balls)):
        if pending:
            previous = pending[-1]
        else:
            previous = rng.rand_mod(colors)
        roll = rng.rand_mod(100)
        run = 0
        for candidate in reversed(pending):
            if candidate != previous:
                break
            run += 1
        if (
            roll <= int(params.ball_repeat_chance)
            and run < int(params.max_clump_size)
        ):
            color = previous
        elif (
            int(params.max_single) < 10
            and _num_pending_singles_scalar(pending, 1) == 1
            and (
                int(params.max_single) == 0
                or _num_pending_singles_scalar(pending, 10)
                > int(params.max_single)
            )
        ):
            color = previous
        elif colors == 1:
            color = 0
        else:
            color = rng.rand_mod(colors)
            while color == previous:
                color = rng.rand_mod(colors)
        rng.next_u31()  # sprite-frame draw (revenge_core.py:1985)
        pending.append(color)
    return pending


def _batch_pending_singles(
    colors: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``_num_pending_singles`` for group limits 1 and 10.

    Walks the already-chosen opening colors (``colors[:, :count]``)
    most-recent-first, exactly like revenge_core.py:1926-1944 with an
    empty chain.
    """

    batch = colors.shape[0]
    results: list[np.ndarray] = []
    for limit in (1, 10):
        group_count = np.zeros(batch, dtype=np.int32)
        previous = np.full(batch, -1, dtype=np.int8)
        singles = np.zeros(batch, dtype=np.int32)
        run_length = np.zeros(batch, dtype=np.int32)
        done = np.zeros(batch, dtype=bool)
        for index in range(count - 1, -1, -1):
            color = colors[:, index]
            done = done | (group_count > limit)
            active = ~done
            change = active & (color != previous)
            singles = singles + (change & (run_length == 1))
            run_length = np.where(
                change,
                np.int32(1),
                np.where(active, run_length + 1, run_length),
            )
            group_count = group_count + change
            previous = np.where(change, color, previous)
        results.append(singles)
    return results[0], results[1]


def batch_opening_colors(
    seeds: Any,
    params: LevelDrawParams,
    *,
    n_outputs: int = DEFAULT_MTRAND_OUTPUTS,
) -> tuple[np.ndarray, int]:
    """Opening chain colors for a batch of seeds, chain order.

    Walks the documented draw layout with a per-lane cursor over the
    precomputed batch MTRand outputs.  Lanes whose rejection loop would
    read past ``n_outputs`` (astronomically rare at the campaign's >= 4
    colors; the observed Jungle1 maximum over live probes was 33 of 64
    draws) are recomputed exactly with the scalar reference.  Returns
    ``(colors[batch, opening_balls] int8, scalar_fallback_count)``.
    """

    seed_array = np.asarray(seeds, dtype=np.int64)
    if seed_array.ndim != 1:
        raise ValueError("seeds must be a 1-D array")
    outputs = batch_mtrand_u31(seed_array, n_outputs)
    batch = seed_array.shape[0]
    colors_count = int(params.colors)
    balls = int(params.opening_balls)
    repeat = int(params.ball_repeat_chance)
    max_clump = int(params.max_clump_size)
    max_single = int(params.max_single)
    colors = np.full((batch, balls), -1, dtype=np.int8)
    lanes = np.arange(batch)
    cursor = np.zeros(batch, dtype=np.int64)
    overflow = np.zeros(batch, dtype=bool)

    def draw(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Next u31 for lanes in ``mask``; flags exhausted lanes."""

        exhausted = mask & (cursor >= n_outputs)
        overflow[exhausted] = True
        take = mask & ~exhausted
        values = outputs[lanes, np.minimum(cursor, n_outputs - 1)]
        cursor[take] += 1
        return values, take

    all_lanes = np.ones(batch, dtype=bool)
    # Ball 0's "previous" draw (revenge_core.py:1952).
    values, _ = draw(all_lanes)
    previous = (values % np.uint32(colors_count)).astype(np.int8)
    run_length = np.zeros(batch, dtype=np.int32)
    for ball in range(balls):
        # Percent roll (revenge_core.py:1960).
        values, _ = draw(all_lanes)
        roll = values % np.uint32(100)
        repeat_branch = (roll <= np.uint32(repeat)) & (
            run_length < max_clump
        )
        if max_single < 10:
            singles1, singles10 = _batch_pending_singles(colors, ball)
            singles_branch = (
                ~repeat_branch
                & (singles1 == 1)
                & ((max_single == 0) | (singles10 > max_single))
            )
        else:
            singles_branch = np.zeros(batch, dtype=bool)
        chosen = np.where(
            repeat_branch | singles_branch, previous, np.int8(-1)
        )
        need = chosen < 0
        if colors_count == 1:
            chosen[need] = 0
            need[:] = False
        while need.any():
            values, took = draw(need)
            need = took
            if not need.any():
                break
            candidate = (values % np.uint32(colors_count)).astype(
                np.int8
            )
            accepted = need & (candidate != previous)
            chosen = np.where(accepted, candidate, chosen)
            need &= ~accepted
        # Sprite-frame draw (revenge_core.py:1985).
        draw(all_lanes)
        colors[:, ball] = chosen
        run_length = np.where(
            chosen == previous, run_length + 1, np.int32(1)
        )
        previous = chosen
    fallback_count = int(overflow.sum())
    if fallback_count:
        for lane in np.nonzero(overflow)[0]:
            colors[lane] = predict_opening_colors_scalar(
                int(seed_array[lane]), params
            )
    if (colors < 0).any():
        raise RuntimeError("opening color prediction left unset lanes")
    return colors, fallback_count


# ---------------------------------------------------------------------------
# Cascade-friendliness scorer.
# ---------------------------------------------------------------------------


def opening_components(
    colors: np.ndarray, num_colors: int
) -> dict[str, np.ndarray]:
    """Component statistics of opening color sequences (chain order).

    * ``longest_run``: longest same-color run.
    * ``runs_ge3``: count of runs with length >= 3.
    * ``adjacency``: adjacent equal-color pairs (immediate-merge
      potential -- every pair is one shot from a 3-clear).
    * ``distinct_colors``: distinct colors present (fewer is better).
    """

    if colors.ndim != 2 or colors.shape[1] < 1:
        raise ValueError("colors must be a non-empty 2-D batch")
    batch, length = colors.shape
    adjacency = (
        (colors[:, 1:] == colors[:, :-1]).sum(axis=1).astype(np.int32)
    )
    run = np.ones(batch, dtype=np.int32)
    longest = np.ones(batch, dtype=np.int32)
    runs_ge3 = np.zeros(batch, dtype=np.int32)
    for index in range(1, length):
        same = colors[:, index] == colors[:, index - 1]
        runs_ge3 = runs_ge3 + (~same & (run >= 3))
        run = np.where(same, run + 1, np.int32(1))
        longest = np.maximum(longest, run)
    runs_ge3 = runs_ge3 + (run >= 3)
    distinct = np.zeros(batch, dtype=np.int32)
    for color in range(int(num_colors)):
        distinct = distinct + (colors == color).any(axis=1)
    return {
        "longest_run": longest,
        "runs_ge3": runs_ge3,
        "adjacency": adjacency,
        "distinct_colors": distinct,
    }


def score_openings(
    colors: np.ndarray,
    weights: OpeningScoreWeights,
    num_colors: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Weighted cascade-friendliness score (float64; ranking matters)."""

    components = opening_components(colors, num_colors)
    score = (
        float(weights.longest_run)
        * components["longest_run"].astype(np.float64)
        + float(weights.runs3) * components["runs_ge3"]
        + float(weights.adjacency) * components["adjacency"]
        + float(weights.distinct) * components["distinct_colors"]
    )
    return score, components


def _select_top_k(
    seeds: np.ndarray, scores: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k: score descending, then seed ascending."""

    order = np.lexsort((seeds, -scores))
    kept = order[: int(top_k)]
    return seeds[kept], scores[kept]


# ---------------------------------------------------------------------------
# Chunked, resumable enumeration.
# ---------------------------------------------------------------------------


def _chunk_path(out_dir: Path, chunk_index: int) -> Path:
    return out_dir / CHUNK_DIRNAME / f"chunk_{chunk_index:06d}.npz"


def enumerate_chunk(
    *,
    chunk_start: int,
    chunk_end: int,
    params: LevelDrawParams,
    weights: OpeningScoreWeights,
    top_k: int,
    batch_size: int,
    n_outputs: int = DEFAULT_MTRAND_OUTPUTS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score one contiguous seed range; returns its sorted top-k."""

    best_seeds = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float64)
    fallbacks = 0
    for batch_start in range(chunk_start, chunk_end, int(batch_size)):
        batch_end = min(batch_start + int(batch_size), chunk_end)
        seeds = np.arange(batch_start, batch_end, dtype=np.int64)
        colors, fallback = batch_opening_colors(
            seeds, params, n_outputs=n_outputs
        )
        fallbacks += fallback
        scores, _ = score_openings(colors, weights, params.colors)
        merged_seeds = np.concatenate([best_seeds, seeds])
        merged_scores = np.concatenate([best_scores, scores])
        best_seeds, best_scores = _select_top_k(
            merged_seeds, merged_scores, top_k
        )
    return best_seeds, best_scores, fallbacks


def _chunk_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Process pool entry point: enumerate + checkpoint one chunk."""

    started = time.perf_counter()
    params = LevelDrawParams(**payload["params"])
    weights = OpeningScoreWeights(**payload["weights"])
    chunk_index = int(payload["chunk_index"])
    chunk_start = int(payload["chunk_start"])
    chunk_end = int(payload["chunk_end"])
    seeds, scores, fallbacks = enumerate_chunk(
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        params=params,
        weights=weights,
        top_k=int(payload["top_k"]),
        batch_size=int(payload["batch_size"]),
        n_outputs=int(payload["n_outputs"]),
    )
    path = _chunk_path(Path(payload["out_dir"]), chunk_index)
    _write_npz_atomic(
        path,
        seeds=seeds.astype(np.uint32),
        scores=scores,
        chunk_start=np.int64(chunk_start),
        chunk_end=np.int64(chunk_end),
        fallbacks=np.int64(fallbacks),
    )
    elapsed = time.perf_counter() - started
    return {
        "chunk_index": chunk_index,
        "seeds_processed": chunk_end - chunk_start,
        "elapsed_seconds": elapsed,
        "fallbacks": fallbacks,
        "top_seeds": seeds,
        "top_scores": scores,
    }


def _load_chunk_checkpoint(
    path: Path, *, chunk_start: int, chunk_end: int
) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as data:
        for key in ("seeds", "scores", "chunk_start", "chunk_end"):
            if key not in data:
                raise ValueError(
                    f"chunk checkpoint {path} is missing {key!r}"
                )
        if (
            int(data["chunk_start"]) != int(chunk_start)
            or int(data["chunk_end"]) != int(chunk_end)
        ):
            raise ValueError(
                f"chunk checkpoint {path} covers "
                f"[{int(data['chunk_start'])}, {int(data['chunk_end'])})"
                f", expected [{chunk_start}, {chunk_end}); the out-dir "
                "mixes incompatible runs"
            )
        seeds = data["seeds"].astype(np.int64)
        scores = data["scores"].astype(np.float64)
        fallbacks = int(data["fallbacks"]) if "fallbacks" in data else 0
    if seeds.shape != scores.shape:
        raise ValueError(f"chunk checkpoint {path} is inconsistent")
    return seeds, scores, fallbacks


def run_live_check(
    *,
    level_id: str,
    original_root: Path,
    params: LevelDrawParams,
    start_seed: int,
    end_seed: int,
    sample_count: int,
    n_outputs: int = DEFAULT_MTRAND_OUTPUTS,
) -> dict[str, Any]:
    """Verify frozen params + predicted openings against the REAL sim.

    Constructs one real ``RevengeSimulator`` from the installed retail
    data, refuses on any frozen-parameter drift, then asserts the batch
    predictor reproduces ``sim.reset(seed=...)`` pending colors for a
    deterministic seed sample.  Fails closed: any mismatch raises.
    """

    from zuma_rl.revenge_core import RevengeSimulator

    simulator = RevengeSimulator.from_installed(
        level_id, root=original_root, seed=1
    )
    live = {
        "colors": int(simulator.active_num_colors),
        "ball_repeat_chance": int(
            getattr(simulator.parameters, "ball_repeat_chance")
        ),
        "max_clump_size": int(
            getattr(simulator.parameters, "max_clump_size")
        ),
        "max_single": int(getattr(simulator.parameters, "max_single")),
        "num_balls": int(
            getattr(simulator.parameters, "num_balls", 0)
        ),
        "opening_balls": int(simulator.config.initial_pending_balls),
        "curve_count": int(simulator.curve_count),
    }
    frozen = {key: getattr(params, key) for key in live}
    if live != frozen:
        raise ValueError(
            f"frozen draw params for {level_id!r} drifted from the "
            f"live level: frozen={frozen}, live={live}"
        )
    sampler = np.random.default_rng(LIVE_CHECK_SAMPLER_SEED)
    sampled = {
        int(seed)
        for seed in sampler.integers(
            int(start_seed), int(end_seed), size=max(0, sample_count)
        )
    }
    sampled.update({int(start_seed), int(end_seed) - 1})
    check_seeds = sorted(sampled)
    predicted, _ = batch_opening_colors(
        np.asarray(check_seeds, dtype=np.int64),
        params,
        n_outputs=n_outputs,
    )
    mismatches: list[int] = []
    for row, seed in enumerate(check_seeds):
        simulator.reset(seed=seed)
        actual = [int(color) for color in simulator.pending_colors]
        if actual != predicted[row].tolist():
            mismatches.append(seed)
    if mismatches:
        raise ValueError(
            "live draw-layout check FAILED for seeds "
            f"{mismatches[:10]}: the enumerator's layout no longer "
            "matches the real simulator"
        )
    return {
        "performed": True,
        "original_root": str(original_root),
        "checked_seeds": len(check_seeds),
        "sampler_seed": LIVE_CHECK_SAMPLER_SEED,
        "params_verified": live,
        "mismatches": 0,
    }


def _resume_identity(
    *,
    level_id: str,
    start_seed: int,
    end_seed: int,
    chunk_size: int,
    top_k: int,
    weights: OpeningScoreWeights,
    params: LevelDrawParams,
    n_outputs: int,
) -> dict[str, Any]:
    identity_params = asdict(params)
    identity_params.pop("provenance")
    return {
        "level": str(level_id),
        "start_seed": int(start_seed),
        "end_seed": int(end_seed),
        "chunk_size": int(chunk_size),
        "top_k": int(top_k),
        "weights": asdict(weights),
        "draw_params": identity_params,
        "mtrand_outputs": int(n_outputs),
    }


def run_enumeration(
    *,
    level_id: str = "Jungle1",
    out_dir: Path,
    start_seed: int = 0,
    end_seed: int = SEED_SPACE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    top_k: int = DEFAULT_TOP_K,
    threads: int = DEFAULT_THREADS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chunks: int | None = None,
    weights: OpeningScoreWeights = OpeningScoreWeights(),
    original_root: Path | None = None,
    live_check_seeds: int = 64,
    n_outputs: int = DEFAULT_MTRAND_OUTPUTS,
    params: LevelDrawParams | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Chunked, resumable golden-seed enumeration.

    Chunk ``i`` covers ``[start_seed + i*chunk_size, ...)`` on a grid
    anchored at ``start_seed``; each completed chunk persists an
    atomic ``.npz`` checkpoint, the merged running top-K is rewritten
    atomically after every chunk, and re-invocation with the same
    identity skips completed chunks.  ``max_chunks`` bounds how many
    PENDING chunks this invocation processes (status ``PARTIAL`` until
    every chunk exists).  Results are deterministic and independent of
    chunking, threads, and resume history.

    Draw-layout parameters are DERIVED from the real level data via
    ``derive_level_draw_params`` (using ``original_root`` when given,
    else the catalog's installation discovery).  ``params`` is an
    engineering/test override only -- it is never exposed on the CLI,
    and real enumerations should always pass ``original_root`` so the
    mandatory live check gates the layout before any chunk is trusted.
    """

    started = time.perf_counter()
    start_seed = int(start_seed)
    end_seed = int(end_seed)
    if not 0 <= start_seed < end_seed <= SEED_SPACE:
        raise ValueError(
            "seed range must satisfy 0 <= start < end <= 2^32 "
            f"(got [{start_seed}, {end_seed}))"
        )
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(threads) < 1:
        raise ValueError("threads must be positive")
    threads = min(int(threads), os.cpu_count() or int(threads))
    if max_chunks is not None and int(max_chunks) < 1:
        raise ValueError("max_chunks must be positive when given")

    params_source = "caller-override"
    if params is None:
        params = derive_level_draw_params(
            level_id, original_root=original_root
        )
        params_source = "derived-at-runtime"
    if int(params.colors) < 1 or int(params.opening_balls) < 1:
        raise ValueError(
            "draw params must have at least one colour and one "
            f"opening ball (got {params})"
        )

    out_dir = Path(out_dir).resolve()
    identity = _resume_identity(
        level_id=level_id,
        start_seed=start_seed,
        end_seed=end_seed,
        chunk_size=int(chunk_size),
        top_k=int(top_k),
        weights=weights,
        params=params,
        n_outputs=int(n_outputs),
    )
    config_path = out_dir / CONFIG_FILENAME
    if out_dir.exists() and config_path.exists():
        recorded = _read_json(config_path).get("resume_identity")
        if recorded != identity:
            raise ValueError(
                "resume identity mismatch: the existing out dir was "
                f"created with {recorded}, this invocation asks for "
                f"{identity}; refusing to mix enumerations"
            )
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"out dir {out_dir} exists without a config receipt"
        )
    (out_dir / CHUNK_DIRNAME).mkdir(parents=True, exist_ok=True)

    live_check: dict[str, Any] = {"performed": False}
    if original_root is not None:
        live_check = run_live_check(
            level_id=level_id,
            original_root=Path(original_root),
            params=params,
            start_seed=start_seed,
            end_seed=end_seed,
            sample_count=int(live_check_seeds),
            n_outputs=int(n_outputs),
        )

    if not config_path.exists():
        _write_json_atomic(
            config_path,
            {
                "schema": CONFIG_SCHEMA,
                "version": 1,
                "status": "FROZEN",
                "tool": {
                    "path": str(SCRIPT_PATH),
                    "sha256": _sha256(SCRIPT_PATH),
                },
                "resume_identity": identity,
                "convention": CONVENTION,
                "draw_layout_citations": list(DRAW_LAYOUT_CITATIONS),
                "draw_params_provenance": params.provenance,
                "draw_params_source": params_source,
                "seed_alias_note": (
                    "PopCapMTRandom.seed maps 0 -> 4357 "
                    "(revenge_core.py:621-622): env seeds 0 and 4357 "
                    "share one stream and one opening"
                ),
                "created_utc": _utc_now(),
            },
        )

    total_chunks = math.ceil((end_seed - start_seed) / int(chunk_size))
    chunk_bounds = [
        (
            index,
            start_seed + index * int(chunk_size),
            min(start_seed + (index + 1) * int(chunk_size), end_seed),
        )
        for index in range(total_chunks)
    ]
    merged_seeds = np.empty(0, dtype=np.int64)
    merged_scores = np.empty(0, dtype=np.float64)
    fallback_total = 0
    resumed_chunks: list[int] = []
    pending: list[tuple[int, int, int]] = []
    for index, chunk_start, chunk_end in chunk_bounds:
        path = _chunk_path(out_dir, index)
        if path.exists():
            seeds, scores, fallbacks = _load_chunk_checkpoint(
                path, chunk_start=chunk_start, chunk_end=chunk_end
            )
            merged_seeds, merged_scores = _select_top_k(
                np.concatenate([merged_seeds, seeds]),
                np.concatenate([merged_scores, scores]),
                int(top_k),
            )
            fallback_total += fallbacks
            resumed_chunks.append(index)
        else:
            pending.append((index, chunk_start, chunk_end))
    planned = (
        pending
        if max_chunks is None
        else pending[: int(max_chunks)]
    )

    def _persist_running(chunks_done: int, rate: float) -> None:
        best = (
            float(merged_scores[0]) if merged_scores.size else None
        )
        _write_npz_atomic(
            out_dir / RUNNING_TOPK_FILENAME,
            seeds=merged_seeds.astype(np.uint32),
            scores=merged_scores,
        )
        _write_json_atomic(
            out_dir / PROGRESS_FILENAME,
            {
                "schema": PROGRESS_SCHEMA,
                "version": 1,
                "chunks_done": chunks_done,
                "chunks_total": total_chunks,
                "best_score": best,
                "seeds_per_second": rate,
                "fallback_lanes": fallback_total,
                "updated_utc": _utc_now(),
            },
        )

    processed_seeds = 0
    stream_started = time.perf_counter()

    def _handle_result(result: Mapping[str, Any]) -> None:
        nonlocal merged_seeds, merged_scores, fallback_total
        nonlocal processed_seeds
        processed_seeds += int(result["seeds_processed"])
        fallback_total += int(result["fallbacks"])
        merged_seeds, merged_scores = _select_top_k(
            np.concatenate(
                [merged_seeds, np.asarray(result["top_seeds"])]
            ),
            np.concatenate(
                [merged_scores, np.asarray(result["top_scores"])]
            ),
            int(top_k),
        )
        chunks_done = len(resumed_chunks) + done_this_run[0]
        elapsed = max(time.perf_counter() - stream_started, 1e-9)
        rate = processed_seeds / elapsed
        _persist_running(chunks_done, rate)
        best = (
            float(merged_scores[0]) if merged_scores.size else float("nan")
        )
        progress(
            f"CHUNK {chunks_done}/{total_chunks} done "
            f"rate={rate:.0f} seeds/s topscore={best:.3f}"
        )

    done_this_run = [0]
    payload_base = {
        "params": asdict(params),
        "weights": asdict(weights),
        "top_k": int(top_k),
        "batch_size": int(batch_size),
        "n_outputs": int(n_outputs),
        "out_dir": str(out_dir),
    }
    if planned:
        if threads == 1:
            for index, chunk_start, chunk_end in planned:
                result = _chunk_worker(
                    {
                        **payload_base,
                        "chunk_index": index,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    }
                )
                done_this_run[0] += 1
                _handle_result(result)
        else:
            context = multiprocessing.get_context(
                "fork"
                if "fork" in multiprocessing.get_all_start_methods()
                else None
            )
            with ProcessPoolExecutor(
                max_workers=min(threads, len(planned)),
                mp_context=context,
            ) as pool:
                futures = [
                    pool.submit(
                        _chunk_worker,
                        {
                            **payload_base,
                            "chunk_index": index,
                            "chunk_start": chunk_start,
                            "chunk_end": chunk_end,
                        },
                    )
                    for index, chunk_start, chunk_end in planned
                ]
                for future in as_completed(futures):
                    result = future.result()
                    done_this_run[0] += 1
                    _handle_result(result)
    else:
        _persist_running(len(resumed_chunks), 0.0)

    chunks_done = len(resumed_chunks) + done_this_run[0]
    wall_seconds = time.perf_counter() - started
    stream_seconds = time.perf_counter() - stream_started
    throughput = {
        "seeds_processed_this_run": processed_seeds,
        "stream_seconds": stream_seconds,
        "seeds_per_second": (
            processed_seeds / stream_seconds if processed_seeds else None
        ),
        "threads": threads,
        "batch_size": int(batch_size),
    }
    if chunks_done < total_chunks:
        return {
            "schema": COMPLETION_SCHEMA,
            "status": "PARTIAL",
            "level": str(level_id),
            "chunks": {
                "total": total_chunks,
                "done": chunks_done,
                "resumed": len(resumed_chunks),
                "computed_this_run": done_this_run[0],
            },
            "best_score": (
                float(merged_scores[0]) if merged_scores.size else None
            ),
            "throughput": throughput,
            "wall_seconds": wall_seconds,
            "out_dir": str(out_dir),
        }

    # Every chunk exists: recompute the winners' openings for the
    # receipt (and as an internal consistency proof), then publish.
    final_seeds = merged_seeds
    final_scores = merged_scores
    final_colors = np.empty(
        (final_seeds.shape[0], params.opening_balls), dtype=np.int8
    )
    for batch_start in range(0, final_seeds.shape[0], int(batch_size)):
        batch_slice = slice(
            batch_start, batch_start + int(batch_size)
        )
        final_colors[batch_slice], _ = batch_opening_colors(
            final_seeds[batch_slice], params, n_outputs=int(n_outputs)
        )
    recomputed, final_components = score_openings(
        final_colors, weights, params.colors
    )
    if not np.array_equal(recomputed, final_scores):
        raise RuntimeError(
            "final top-K rescore disagrees with chunk checkpoints; "
            "the out dir is corrupt or the scorer changed mid-run"
        )
    topk_path = out_dir / TOPK_FILENAME
    tmp = topk_path.with_name(f"{topk_path.name}.tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for rank in range(final_seeds.shape[0]):
            row = {
                "schema": TOPK_ROW_SCHEMA,
                "level": str(level_id),
                "rank": rank + 1,
                "seed": int(final_seeds[rank]),
                "score": float(final_scores[rank]),
                "opening_colors": [
                    int(color) for color in final_colors[rank]
                ],
                "components": {
                    name: int(final_components[name][rank])
                    for name in final_components
                },
            }
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, topk_path)

    completion = {
        "schema": COMPLETION_SCHEMA,
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "wall_seconds": wall_seconds,
        "convention": CONVENTION,
        "level": str(level_id),
        "start_seed": start_seed,
        "end_seed": end_seed,
        "seed_count": end_seed - start_seed,
        "chunk_size": int(chunk_size),
        "chunks": {
            "total": total_chunks,
            "resumed": len(resumed_chunks),
            "computed_this_run": done_this_run[0],
        },
        "top_k": int(top_k),
        "top_k_kept": int(final_seeds.shape[0]),
        "top_k_file": TOPK_FILENAME,
        "scorer": {
            "weights": asdict(weights),
            "formula": (
                "score = w_longest_run*longest_run + w_runs3*runs_ge3 "
                "+ w_adjacency*adjacency + w_distinct*distinct_colors; "
                "ties break by ascending seed"
            ),
            "components": [
                "longest_run",
                "runs_ge3",
                "adjacency",
                "distinct_colors",
            ],
        },
        "draw_layout": {
            "params": asdict(params),
            "citations": list(DRAW_LAYOUT_CITATIONS),
            "seed_mapping": (
                "env episode seed -> int(seed) & 0xFFFFFFFF -> "
                "PopCapMTRandom.seed (0 aliases to 4357); 1:1 "
                "otherwise over [0, 2^32)"
            ),
        },
        "live_check": live_check,
        "scalar_fallback_lanes": fallback_total,
        "best": [
            {
                "rank": rank + 1,
                "seed": int(final_seeds[rank]),
                "score": float(final_scores[rank]),
                "opening_colors": [
                    int(color) for color in final_colors[rank]
                ],
            }
            for rank in range(min(10, final_seeds.shape[0]))
        ],
        "throughput": throughput,
        "tool": {
            "path": str(SCRIPT_PATH),
            "sha256": _sha256(SCRIPT_PATH),
        },
        "embargo_note": (
            "no episodes are played by this enumeration; no "
            "(level, seed) eval keys are consumed.  The downstream "
            "seed-list search tool enforces the frozen embargoes "
            "before any episode runs."
        ),
        "formal_seed_consumption": False,
        "training_authority": False,
        "out_dir": str(out_dir),
    }
    _write_json_atomic(out_dir / COMPLETION_FILENAME, completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        default="Jungle1",
        help=(
            "any zuma_rl.alphazuma_55.INCLUDED_LEVELS member; the "
            "draw layout is derived from the real level data at start"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--end-seed",
        type=int,
        default=SEED_SPACE,
        help="exclusive; defaults to the full 2^32 seed space",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "worker processes (one chunk per worker at a time); "
            "capped at the machine's CPU count"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="seeds per vectorized inner batch (memory/speed knob)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "process at most this many pending chunks then stop with "
            "status PARTIAL (throughput measurement / bounded resume)"
        ),
    )
    parser.add_argument(
        "--weight-longest-run", type=float, default=2.0
    )
    parser.add_argument("--weight-runs3", type=float, default=3.0)
    parser.add_argument("--weight-adjacency", type=float, default=1.0)
    parser.add_argument("--weight-distinct", type=float, default=-2.0)
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help=(
            "retail install root; when given, the derived draw params "
            "AND a deterministic seed sample are verified against the "
            "real simulator before enumerating (fails closed)"
        ),
    )
    parser.add_argument(
        "--live-check-seeds", type=int, default=64
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_enumeration(
        level_id=str(args.level),
        out_dir=args.out_dir.expanduser(),
        start_seed=int(args.start_seed),
        end_seed=int(args.end_seed),
        chunk_size=int(args.chunk_size),
        top_k=int(args.top_k),
        threads=int(args.threads),
        batch_size=int(args.batch_size),
        max_chunks=(
            int(args.max_chunks) if args.max_chunks is not None else None
        ),
        weights=OpeningScoreWeights(
            longest_run=float(args.weight_longest_run),
            runs3=float(args.weight_runs3),
            adjacency=float(args.weight_adjacency),
            distinct=float(args.weight_distinct),
        ),
        original_root=(
            args.original_root.expanduser().resolve(strict=True)
            if args.original_root is not None
            else None
        ),
        live_check_seeds=int(args.live_check_seeds),
    )
    summary = {
        "status": result["status"],
        "level": result["level"],
        "chunks": result["chunks"],
        "throughput": result["throughput"],
        "wall_seconds": result["wall_seconds"],
        "out_dir": result["out_dir"],
    }
    if result["status"] == "COMPLETE":
        summary["top_k_kept"] = result["top_k_kept"]
        summary["top_k_file"] = result["top_k_file"]
        summary["best"] = result["best"][:3]
    else:
        summary["best_score"] = result["best_score"]
    print(
        json.dumps(
            summary, ensure_ascii=False, indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
