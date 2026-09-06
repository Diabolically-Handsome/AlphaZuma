"""Coverage-equalized replay dataset builder for the motor-observable route.

Lineage
-------
``build_alphazuma_55_motor_observable_replay.py`` (V1), ``_v2.py`` and
``_v3.py`` freeze the preregistrations for the motor-observable replay
campaigns; the transition subsampling those routes freeze is implemented by
``tools/distill_alphazuma_55_polar_intent_wide_v1._collect_intent_round``,
which keeps independent per-episode, per-verb uniform reservoirs
(wait 256 / fire 256 / swap 192 / hop 192, with wait candidates strided
1-in-4).  Measured against the 55-level teacher anchor those caps compress
long hard levels hardest (volcano retains ~6.2% of teacher ticks while
jungle retains ~20.6%; ~201,315 transitions total over 55 levels), shred
teacher fire windows by sampling ticks independently, and give danger
states no weight at all.

V4 keeps the sample record format the replay-v2 distillation consumers
already read -- ``(observation float32, raw teacher intent action int64
[verb, aim_bin], training mask bool)`` triples plus a collection manifest
whose replay-v2 keys are unchanged (all new keys are additive and the
manifest is marked ``"builder": "v4-coverage"``) -- but replaces the
retention rule with three tiers computed per LEVEL over full-tick episode
records:

1. Fire-window tier: every tick inside ``[fire_edge -
   fire_window_before_ticks, fire_edge + fire_window_after_ticks]`` around
   every RAW teacher fire edge (a rising edge in the raw intent verb
   stream) is retained.  With the elite input chain (reaction delay 12
   ticks, FIRING 6 ticks before release) the default margins (24 before,
   6 after) cover the park-and-settle aim intents that decide the release
   angle, which independent tick sampling used to shred.
2. Danger tier: every tick whose danger value reaches
   ``danger_threshold`` is retained.  The danger value comes from the
   first field of ``danger_field_priority`` that the per-tick ``info``
   records actually carry; the chosen field and its normalization are
   reported in the manifest under ``coverage_equalization``.
3. Uniform remainder: the rest of the level budget is drawn uniformly
   without replacement from the remaining ticks -- no verb quotas.

Budget arithmetic
-----------------
Let ``S_L`` be the total source ticks recorded for level ``L`` and ``r``
the configured ``target_coverage_ratio``::

    Q_L = min(S_L, ceil(r * S_L))            # per-level quota

If ``sum(Q_L) > total_budget`` every quota is rescaled by
``total_budget / sum(Q_L)`` and rounded with a deterministic
largest-remainder rule so that ``sum(Q_L) == total_budget`` while
``Q_L / S_L`` stays equal across levels to within one transition: the
retained-coverage RATIO is equalized, instead of the per-episode uniform
caps that starved long levels.  The mandatory tiers are never dropped:
with ``M_L`` the union of fire-window and danger ticks,

    retained_L = M_L  UNION  UniformSample(non-mandatory ticks of L,
                                           max(0, Q_L - |M_L|))

so ``|retained_L| == max(Q_L, |M_L|)``.  A level whose mandatory tier
overflows its quota overshoots visibly in the manifest
(``per_level_coverage[level]["retained_ticks"] >
per_level_coverage[level]["quota_ticks"]``); the ``total_budget`` cap
therefore binds the uniform remainder, never the mandatory signal.

Danger-field selection
----------------------
The builder inspects the union of per-tick ``info`` keys across every
episode and picks the first field of ``danger_field_priority`` that is
present anywhere (default priority: ``danger_fraction``,
``furthest_waypoint_fraction``, ``chain_length``, ``visible_balls``; the
last two are what ``RevengeEnv._info`` emits today, the first two are
direct fractions a future recorder may add).  Values are compared against
``danger_threshold`` after normalization:

* ``raw_fraction`` -- if the global maximum of the field is <= 1.0 the
  values are already fractions and are used as-is;
* ``per_level_max`` -- otherwise each level's values are divided by that
  level's maximum.  Levels whose signal is flat (max == min) contribute
  no danger ticks, so a constant count field cannot mark a whole level
  mandatory.

The manifest reports ``danger_field_used``, ``danger_normalization`` and
``danger_field_candidates_present`` so the choice is auditable.

Episode record input
--------------------
The builder consumes FULL-tick episode records (the reservoir stage of
``_collect_intent_round`` is what V4 replaces, so records must not be
pre-subsampled)::

    {
        "level_id": str,
        "seed": int,
        "outcome": "win" | "loss",
        "time_limit_truncated": bool,               # optional
        "observation_capacity_overflow": bool,       # optional
        "ticks": [
            {
                "observation": float array,          # actor observation
                "raw_action": int array [verb, aim], # raw teacher intent
                "training_mask": bool array,         # verb-relaxed mask
                "exact_mask": bool array,            # EXACT runtime mask
                "effective_action": int array,       # optional
                "intent_mask_relaxed": bool,         # optional
                "info": {...},                       # optional env info
            },
            ...
        ],
    }

Mask semantics (both masks are REQUIRED per tick):

* ``training_mask`` is the relaxed replay-v2 training mask
  (``_intent_label`` unmasks the raw teacher verb); it validates the raw
  teacher verb and aim exactly like the replay-v2 chain and is what
  replay-v2 consumers keep reading.
* ``exact_mask`` is the exact runtime valid-action mask as the wrapper
  reported it, UNRELAXED.  The raw teacher verb is deliberately NOT
  validated against it -- roughly a quarter of raw teacher intents are
  runtime-illegal, and preserving that signal is the whole point: the
  park-settle trainer drops those ticks from its verb loss and reports
  the true ``illegal_teacher_verb_fraction`` instead of a vacuous 0.0.
  The manifest reports the observed fraction under
  ``runtime_illegal_raw_verb_fraction``.

Output rows are ordered level-major (levels in first-appearance order,
episodes in input order, ticks ascending) and every row keeps the exact
``_intent_label`` legality contract: the training mask must allow the raw
teacher verb and the raw teacher aim bin.

Outputs
-------
``main`` writes three sibling files per ``--output-prefix``:

* ``<prefix>.npz`` -- one aggregate NPZ with ``observations``,
  ``actions``, ``training_masks``, ``exact_masks``, and per-row
  ``episode_indices`` / ``tick_indices`` identity arrays, so temporal
  structure survives aggregation.
* ``<prefix>.manifest.json`` -- the replay-v2-compatible collection
  manifest (additive v4 keys only).
* ``<prefix>.dataset.json`` -- the trainer-facing dataset manifest:
  a ``shards`` list with per-file sha256 receipts plus a
  ``mask_semantics`` declaration.  ``tools.distill_alphazuma_55_park_
  settle_v1.load_replay_dataset`` consumes this file directly and loads
  it with ``temporal_structure == "episode_ticks"`` while picking the
  ``exact_masks`` key for its legality accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
BUILDER_ID = "v4-coverage"
BUILDER_VERSION = 4
# Mirrors tools.distill_alphazuma_55_settled_v3.VERB_NAMES; duplicated so
# this dataset-shaping tool imports with numpy alone instead of pulling the
# torch-backed distillation chain.
VERB_NAMES = ("wait", "fire", "swap", "hop")
FIRE_VERB_INDEX = VERB_NAMES.index("fire")
DEFAULT_DANGER_FIELD_PRIORITY = (
    "danger_fraction",
    "furthest_waypoint_fraction",
    "chain_length",
    "visible_balls",
)
EXACT_MASK_SEMANTICS = "exact_runtime_valid_action_mask"
RELAXED_MASK_SEMANTICS = "relaxed_training_mask_teacher_verb_unmasked"
MASK_SEMANTICS = {
    "exact_masks": EXACT_MASK_SEMANTICS,
    "training_masks": RELAXED_MASK_SEMANTICS,
}
DATASET_MANIFEST_SCHEMA = (
    "zuma-rl.alphazuma-55-motor-observable-replay-v4-dataset"
)

SampleRow = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CoverageReplayV4Config:
    """Retention configuration for the coverage-equalized replay builder."""

    target_coverage_ratio: float = 0.15
    total_budget: int = 200_000
    fire_window_before_ticks: int = 24
    fire_window_after_ticks: int = 6
    danger_threshold: float = 0.75
    danger_field_priority: tuple[str, ...] = DEFAULT_DANGER_FIELD_PRIORITY
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_coverage_ratio) <= 1.0:
            raise ValueError("target_coverage_ratio must be in (0, 1]")
        if int(self.total_budget) < 1:
            raise ValueError("total_budget must be at least 1")
        if int(self.fire_window_before_ticks) < 0:
            raise ValueError("fire_window_before_ticks cannot be negative")
        if int(self.fire_window_after_ticks) < 0:
            raise ValueError("fire_window_after_ticks cannot be negative")
        if not 0.0 < float(self.danger_threshold) <= 1.0:
            raise ValueError("danger_threshold must be in (0, 1]")
        if any(
            not isinstance(name, str) or not name
            for name in self.danger_field_priority
        ):
            raise ValueError("danger_field_priority entries must be names")
        if int(self.sampling_seed) < 0:
            raise ValueError("sampling_seed cannot be negative")


@dataclass
class _ParsedEpisode:
    """One validated full-tick episode record."""

    level_id: str
    seed: int
    outcome: str
    time_limit_truncated: bool
    observation_capacity_overflow: bool
    score: int
    intent_mask_relaxations: int
    triples: list[SampleRow]
    exact_masks: list[np.ndarray]
    runtime_illegal_verbs: np.ndarray
    infos: list[Mapping[str, Any]]
    raw_verbs: np.ndarray
    effective_verbs: np.ndarray


@dataclass
class CoverageReplayProvenance:
    """Per-row identity and exact runtime masks, aligned with the rows.

    ``episode_indices`` holds the source episode ordinal (input order),
    ``tick_indices`` the tick position inside that episode, and
    ``exact_masks`` the UNRELAXED runtime valid-action mask per retained
    row.  These arrays are what makes the aggregate NPZ loadable with
    ``temporal_structure == "episode_ticks"`` and lets the park-settle
    trainer compute an honest illegal-teacher-verb fraction.
    """

    episode_indices: np.ndarray
    tick_indices: np.ndarray
    exact_masks: np.ndarray


def _dataset_sha256(rows: Sequence[SampleRow]) -> str:
    """Order-sensitive dataset digest, byte-compatible with the V3 trainer.

    Mirrors ``distill_alphazuma_55_motor_observable_replay_v3
    ._dataset_sha256`` (duplicated to keep this module numpy-only).
    """

    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _parse_episode(
    record: Mapping[str, Any],
    *,
    index: int,
    sizes: dict[str, int | None],
) -> _ParsedEpisode:
    level_id = str(record.get("level_id", ""))
    if not level_id:
        raise ValueError(f"episode {index} has no level_id")
    if "seed" not in record:
        raise ValueError(f"episode {index} ({level_id}) has no seed")
    ticks = record.get("ticks")
    if not ticks:
        raise ValueError(f"episode {index} ({level_id}) has no ticks")
    label = f"episode {index} ({level_id})"
    triples: list[SampleRow] = []
    exact_masks: list[np.ndarray] = []
    infos: list[Mapping[str, Any]] = []
    raw_verbs = np.zeros(len(ticks), dtype=np.int64)
    effective_verbs = np.zeros(len(ticks), dtype=np.int64)
    runtime_illegal_verbs = np.zeros(len(ticks), dtype=np.bool_)
    relaxations = 0
    for tick_index, tick in enumerate(ticks):
        for key in ("observation", "raw_action", "training_mask", "exact_mask"):
            if key not in tick:
                raise ValueError(f"{label} tick {tick_index} misses {key}")
        observation = np.asarray(
            tick["observation"], dtype=np.float32
        ).reshape(-1)
        action = np.asarray(tick["raw_action"], dtype=np.int64).reshape(2)
        mask = np.asarray(tick["training_mask"], dtype=np.bool_).reshape(-1)
        exact_mask = np.asarray(
            tick["exact_mask"], dtype=np.bool_
        ).reshape(-1)
        if sizes["observation"] is None:
            sizes["observation"] = int(observation.size)
            sizes["mask"] = int(mask.size)
        if (
            int(observation.size) != sizes["observation"]
            or int(mask.size) != sizes["mask"]
            or int(exact_mask.size) != sizes["mask"]
        ):
            raise ValueError(f"{label} tick {tick_index} changes array sizes")
        aim_bins = int(mask.size) - len(VERB_NAMES)
        if aim_bins < 1:
            raise ValueError(f"{label} tick {tick_index} mask is too short")
        verb = int(action[0])
        aim = int(action[1])
        if not 0 <= verb < len(VERB_NAMES) or not 0 <= aim < aim_bins:
            raise ValueError(f"{label} tick {tick_index} action out of range")
        # Legality is validated against the RELAXED training mask only;
        # the exact runtime mask may legitimately forbid the raw teacher
        # verb (~26% of intents) and that signal must survive intact.
        if not bool(mask[verb]):
            raise ValueError(
                f"{label} tick {tick_index} training mask forbids the raw verb"
            )
        if not bool(mask[len(VERB_NAMES) + aim]):
            raise ValueError(
                f"{label} tick {tick_index} raw teacher aim is masked"
            )
        runtime_illegal_verbs[tick_index] = not bool(exact_mask[verb])
        effective = action
        if "effective_action" in tick:
            effective = np.asarray(
                tick["effective_action"], dtype=np.int64
            ).reshape(2)
            if not 0 <= int(effective[0]) < len(VERB_NAMES):
                raise ValueError(
                    f"{label} tick {tick_index} effective verb out of range"
                )
        relaxations += int(bool(tick.get("intent_mask_relaxed", False)))
        info = tick.get("info", {})
        if not isinstance(info, Mapping):
            raise ValueError(f"{label} tick {tick_index} info is not a mapping")
        triples.append((observation, action, mask))
        exact_masks.append(exact_mask)
        infos.append(info)
        raw_verbs[tick_index] = verb
        effective_verbs[tick_index] = int(effective[0])
    last_info = infos[-1]
    return _ParsedEpisode(
        level_id=level_id,
        seed=int(record["seed"]),
        outcome=str(record.get("outcome", "")),
        time_limit_truncated=bool(record.get("time_limit_truncated", False)),
        observation_capacity_overflow=bool(
            record.get("observation_capacity_overflow", False)
        ),
        score=int(last_info.get("score", record.get("score", 0)) or 0),
        intent_mask_relaxations=relaxations,
        triples=triples,
        exact_masks=exact_masks,
        runtime_illegal_verbs=runtime_illegal_verbs,
        infos=infos,
        raw_verbs=raw_verbs,
        effective_verbs=effective_verbs,
    )


def _equalized_quotas(
    source_ticks: Sequence[int],
    *,
    target_coverage_ratio: float,
    total_budget: int,
) -> tuple[list[int], bool]:
    """Per-level quotas with equal retained-coverage ratio (see docstring)."""

    quotas = [
        min(int(count), math.ceil(target_coverage_ratio * int(count)))
        for count in source_ticks
    ]
    total = sum(quotas)
    if total <= int(total_budget):
        return quotas, False
    scale = float(total_budget) / float(total)
    scaled = [quota * scale for quota in quotas]
    floors = [int(math.floor(value)) for value in scaled]
    leftover = int(total_budget) - sum(floors)
    order = sorted(
        range(len(quotas)),
        key=lambda position: (floors[position] - scaled[position], position),
    )
    for position in order[:leftover]:
        floors[position] += 1
    return floors, True


def _fire_window_mask(
    raw_verbs: np.ndarray, *, before: int, after: int
) -> tuple[np.ndarray, int]:
    """Boolean retention mask around raw-intent fire rising edges."""

    fire = raw_verbs == FIRE_VERB_INDEX
    previous = np.concatenate(([False], fire[:-1]))
    edges = np.flatnonzero(fire & ~previous)
    window = np.zeros(raw_verbs.size, dtype=np.bool_)
    for edge in edges:
        low = max(0, int(edge) - int(before))
        high = min(int(raw_verbs.size), int(edge) + int(after) + 1)
        window[low:high] = True
    return window, int(edges.size)


def _select_danger_field(
    parsed: Sequence[_ParsedEpisode], priority: Sequence[str]
) -> tuple[str | None, list[str]]:
    present = [
        field
        for field in priority
        if any(field in info for episode in parsed for info in episode.infos)
    ]
    return (present[0] if present else None), present


def _danger_raw_values(episode: _ParsedEpisode, field: str) -> np.ndarray:
    values = np.zeros(len(episode.infos), dtype=np.float64)
    for tick_index, info in enumerate(episode.infos):
        value = info.get(field, 0.0)
        values[tick_index] = 0.0 if value is None else float(value)
    return values


def build_coverage_equalized_replay(
    episodes: Sequence[Mapping[str, Any]],
    *,
    config: CoverageReplayV4Config | None = None,
    round_index: int = 0,
    teacher_policy_id: str | None = None,
) -> tuple[list[SampleRow], dict[str, Any]]:
    """Build replay-v2-compatible rows plus a v4-coverage manifest.

    Thin wrapper over ``build_coverage_equalized_replay_dataset`` for
    replay-v2-style in-memory consumers that only need the 3-tuple rows.
    """

    rows, manifest, _ = build_coverage_equalized_replay_dataset(
        episodes,
        config=config,
        round_index=round_index,
        teacher_policy_id=teacher_policy_id,
    )
    return rows, manifest


def build_coverage_equalized_replay_dataset(
    episodes: Sequence[Mapping[str, Any]],
    *,
    config: CoverageReplayV4Config | None = None,
    round_index: int = 0,
    teacher_policy_id: str | None = None,
) -> tuple[list[SampleRow], dict[str, Any], CoverageReplayProvenance]:
    """Build rows, the v4-coverage manifest, and per-row provenance."""

    started = time.perf_counter()
    config = config if config is not None else CoverageReplayV4Config()
    if not episodes:
        raise ValueError("no episode records were provided")
    sizes: dict[str, int | None] = {"observation": None, "mask": None}
    parsed = [
        _parse_episode(record, index=index, sizes=sizes)
        for index, record in enumerate(episodes)
    ]

    level_order: list[str] = []
    level_members: dict[str, list[int]] = {}
    for position, episode in enumerate(parsed):
        if episode.level_id not in level_members:
            level_order.append(episode.level_id)
            level_members[episode.level_id] = []
        level_members[episode.level_id].append(position)

    source_ticks = [
        sum(len(parsed[member].triples) for member in level_members[level_id])
        for level_id in level_order
    ]
    quotas, budget_cap_applied = _equalized_quotas(
        source_ticks,
        target_coverage_ratio=float(config.target_coverage_ratio),
        total_budget=int(config.total_budget),
    )

    danger_field, danger_candidates = _select_danger_field(
        parsed, config.danger_field_priority
    )
    danger_normalization: str | None = None
    if danger_field is not None:
        global_max = max(
            (
                float(_danger_raw_values(episode, danger_field).max())
                for episode in parsed
            ),
            default=0.0,
        )
        danger_normalization = (
            "raw_fraction" if global_max <= 1.0 else "per_level_max"
        )

    rows: list[SampleRow] = []
    provenance_episodes: list[int] = []
    provenance_ticks: list[int] = []
    provenance_exact_masks: list[np.ndarray] = []
    source_runtime_illegal = 0
    retained_runtime_illegal = 0
    per_level_coverage: dict[str, dict[str, Any]] = {}
    episode_reports: dict[int, dict[str, Any]] = {}
    for level_index, level_id in enumerate(level_order):
        members = level_members[level_id]
        fire_masks: list[np.ndarray] = []
        danger_masks: list[np.ndarray] = []
        fire_edges = 0
        level_danger_values: list[np.ndarray] = []
        for member in members:
            episode = parsed[member]
            window, edges = _fire_window_mask(
                episode.raw_verbs,
                before=int(config.fire_window_before_ticks),
                after=int(config.fire_window_after_ticks),
            )
            fire_masks.append(window)
            fire_edges += edges
            if danger_field is None:
                level_danger_values.append(
                    np.zeros(len(episode.triples), dtype=np.float64)
                )
            else:
                level_danger_values.append(
                    _danger_raw_values(episode, danger_field)
                )
        if danger_field is None:
            danger_masks = [
                np.zeros(values.size, dtype=np.bool_)
                for values in level_danger_values
            ]
        else:
            concatenated = np.concatenate(level_danger_values)
            if danger_normalization == "raw_fraction":
                normalized = level_danger_values
            else:
                level_max = float(concatenated.max())
                level_min = float(concatenated.min())
                if level_max > 0.0 and level_max > level_min:
                    normalized = [
                        values / level_max for values in level_danger_values
                    ]
                else:
                    # Flat signal: no contrast, no danger tier for the level.
                    normalized = [
                        np.zeros(values.size, dtype=np.float64)
                        for values in level_danger_values
                    ]
            danger_masks = [
                values >= float(config.danger_threshold)
                for values in normalized
            ]

        flat_fire = np.concatenate(fire_masks)
        flat_danger = np.concatenate(danger_masks)
        flat_mandatory = flat_fire | flat_danger
        total = int(flat_mandatory.size)
        quota = int(quotas[level_index])
        mandatory_count = int(flat_mandatory.sum())
        need = max(0, quota - mandatory_count)
        selected = flat_mandatory.copy()
        if need > 0:
            pool = np.flatnonzero(~flat_mandatory)
            rng = np.random.default_rng(
                int(config.sampling_seed)
                + int(round_index) * 1_000_003
                + level_index * 100_003
            )
            chosen = rng.choice(pool.size, size=need, replace=False)
            selected[pool[chosen]] = True
        retained_count = int(selected.sum())

        offset = 0
        retained_by_verb_level: Counter[str] = Counter()
        for member in members:
            episode = parsed[member]
            count = len(episode.triples)
            episode_selected = selected[offset : offset + count]
            episode_fire = flat_fire[offset : offset + count]
            episode_danger = flat_danger[offset : offset + count]
            offset += count
            retained_indices = np.flatnonzero(episode_selected)
            retained_counts: Counter[str] = Counter()
            for tick_index in retained_indices:
                rows.append(episode.triples[int(tick_index)])
                provenance_episodes.append(int(member))
                provenance_ticks.append(int(tick_index))
                provenance_exact_masks.append(
                    episode.exact_masks[int(tick_index)]
                )
                retained_counts[
                    VERB_NAMES[int(episode.raw_verbs[int(tick_index)])]
                ] += 1
            source_runtime_illegal += int(episode.runtime_illegal_verbs.sum())
            retained_runtime_illegal += int(
                episode.runtime_illegal_verbs[retained_indices].sum()
            )
            retained_by_verb_level.update(retained_counts)
            raw_counts = Counter(
                VERB_NAMES[int(verb)] for verb in episode.raw_verbs
            )
            effective_counts = Counter(
                VERB_NAMES[int(verb)] for verb in episode.effective_verbs
            )
            episode_reports[member] = {
                "level_id": episode.level_id,
                "seed": episode.seed,
                "steps": count,
                "outcome": episode.outcome,
                "time_limit_truncated": episode.time_limit_truncated,
                "score": episode.score,
                "raw_intent_action_counts": dict(raw_counts),
                "effective_execution_action_counts": dict(effective_counts),
                "intent_mask_relaxations": episode.intent_mask_relaxations,
                "runtime_illegal_raw_verb_ticks": int(
                    episode.runtime_illegal_verbs.sum()
                ),
                # Every source tick is a candidate in v4 (no verb strides).
                "candidate_samples_by_raw_verb": dict(raw_counts),
                "retained_samples_by_raw_verb": dict(retained_counts),
                "retained_samples": int(retained_indices.size),
                "observation_capacity_overflow": (
                    episode.observation_capacity_overflow
                ),
                "fire_window_ticks": int(episode_fire.sum()),
                "danger_ticks": int(episode_danger.sum()),
                "uniform_ticks": int(
                    retained_indices.size
                    - (episode_fire | episode_danger).sum()
                ),
            }

        per_level_coverage[level_id] = {
            "total_source_ticks": total,
            "quota_ticks": quota,
            "retained_ticks": retained_count,
            "coverage_ratio": retained_count / total,
            "fire_window_ticks": int(flat_fire.sum()),
            "danger_ticks": int(flat_danger.sum()),
            "mandatory_ticks": mandatory_count,
            "uniform_ticks": retained_count - mandatory_count,
            "fire_edges": fire_edges,
            "episodes_used": len(members),
        }

    ordered_reports = [
        episode_reports[position] for position in range(len(parsed))
    ]
    retained_by_verb = Counter(
        VERB_NAMES[int(row[1][0])] for row in rows
    )
    total_source = sum(source_ticks)
    manifest: dict[str, Any] = {
        "round_index": int(round_index),
        "wall_seconds": time.perf_counter() - started,
        "episodes": ordered_reports,
        "wins": sum(
            report["outcome"] == "win" for report in ordered_reports
        ),
        "losses": sum(
            report["outcome"] == "loss" for report in ordered_reports
        ),
        "truncations": sum(
            report["time_limit_truncated"] for report in ordered_reports
        ),
        "capacity_overflows": sum(
            report["observation_capacity_overflow"]
            for report in ordered_reports
        ),
        "retained_samples": len(rows),
        "retained_samples_by_verb": dict(retained_by_verb),
        "raw_intent_action_counts": dict(
            sum(
                (
                    Counter(report["raw_intent_action_counts"])
                    for report in ordered_reports
                ),
                Counter(),
            )
        ),
        "effective_execution_action_counts": dict(
            sum(
                (
                    Counter(report["effective_execution_action_counts"])
                    for report in ordered_reports
                ),
                Counter(),
            )
        ),
        "intent_mask_relaxations": sum(
            report["intent_mask_relaxations"] for report in ordered_reports
        ),
        "training_label_semantics": "raw_teacher_intent",
        "environment_execution_semantics": "exact_mask_effective_action",
        "teacher_policy_id": teacher_policy_id,
        "dataset_sha256": _dataset_sha256(rows),
        # Additive v4 keys below; replay-v2 consumers ignore them.
        "builder": BUILDER_ID,
        "builder_version": BUILDER_VERSION,
        "mask_semantics": dict(MASK_SEMANTICS),
        "runtime_illegal_raw_verb_fraction": {
            "source": (
                source_runtime_illegal / total_source if total_source else 0.0
            ),
            "retained": (
                retained_runtime_illegal / len(rows) if rows else 0.0
            ),
        },
        "coverage_equalization": {
            "target_coverage_ratio": float(config.target_coverage_ratio),
            "total_budget": int(config.total_budget),
            "budget_cap_applied": budget_cap_applied,
            "effective_coverage_ratio": (
                len(rows) / total_source if total_source else 0.0
            ),
            "fire_window_before_ticks": int(config.fire_window_before_ticks),
            "fire_window_after_ticks": int(config.fire_window_after_ticks),
            "danger_threshold": float(config.danger_threshold),
            "danger_field_priority": list(config.danger_field_priority),
            "danger_field_candidates_present": danger_candidates,
            "danger_field_used": danger_field,
            "danger_normalization": danger_normalization,
            "sampling_seed": int(config.sampling_seed),
        },
        "per_level_coverage": per_level_coverage,
    }
    mask_width = int(sizes["mask"] or 0)
    provenance = CoverageReplayProvenance(
        episode_indices=np.asarray(provenance_episodes, dtype=np.int64),
        tick_indices=np.asarray(provenance_ticks, dtype=np.int64),
        exact_masks=(
            np.stack(provenance_exact_masks)
            if provenance_exact_masks
            else np.zeros((0, mask_width), dtype=np.bool_)
        ),
    )
    return rows, manifest, provenance


def _load_episode_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("episodes") if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError(f"no episode list found in {path}")
    return records


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"output already exists: {path}")
    os.replace(temporary, path)


def _write_new_npz(
    path: Path,
    rows: Sequence[SampleRow],
    provenance: CoverageReplayProvenance,
) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.stack([row[0] for row in rows]),
        actions=np.stack([row[1] for row in rows]),
        training_masks=np.stack([row[2] for row in rows]),
        exact_masks=provenance.exact_masks,
        episode_indices=provenance.episode_indices,
        tick_indices=provenance.tick_indices,
    )


def _file_sha256(path: Path) -> str:
    """File-content digest, format-compatible with the trainer's loader.

    Mirrors ``tools.distill_alphazuma_55._sha256`` (duplicated to keep
    this module numpy-only instead of pulling the torch-backed chain).
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _dataset_manifest(
    *,
    npz_path: Path,
    rows: Sequence[SampleRow],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Trainer-facing dataset manifest for the aggregate NPZ shard."""

    return {
        "schema": DATASET_MANIFEST_SCHEMA,
        "version": BUILDER_VERSION,
        "builder": BUILDER_ID,
        "temporal_structure": "episode_ticks",
        "mask_semantics": dict(MASK_SEMANTICS),
        "dataset_sha256": manifest["dataset_sha256"],
        "runtime_illegal_raw_verb_fraction": manifest[
            "runtime_illegal_raw_verb_fraction"
        ],
        "shards": [
            {
                "path": npz_path.name,
                "sha256": _file_sha256(npz_path),
                "rows": len(rows),
            }
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = CoverageReplayV4Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
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
    parser.add_argument("--teacher-policy-id", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CoverageReplayV4Config(
        target_coverage_ratio=args.target_coverage_ratio,
        total_budget=args.total_budget,
        fire_window_before_ticks=args.fire_window_before_ticks,
        fire_window_after_ticks=args.fire_window_after_ticks,
        danger_threshold=args.danger_threshold,
        sampling_seed=args.sampling_seed,
    )
    records = _load_episode_records(args.episodes.resolve(strict=True))
    rows, manifest, provenance = build_coverage_equalized_replay_dataset(
        records,
        config=config,
        round_index=int(args.round_index),
        teacher_policy_id=args.teacher_policy_id,
    )
    npz_path = args.output_prefix.parent / (args.output_prefix.name + ".npz")
    manifest_path = args.output_prefix.parent / (
        args.output_prefix.name + ".manifest.json"
    )
    dataset_manifest_path = args.output_prefix.parent / (
        args.output_prefix.name + ".dataset.json"
    )
    _write_new_npz(npz_path, rows, provenance)
    dataset_manifest = _dataset_manifest(
        npz_path=npz_path.resolve(), rows=rows, manifest=manifest
    )
    _write_new_json(dataset_manifest_path, dataset_manifest)
    manifest["trainer_dataset_manifest"] = dataset_manifest_path.name
    _write_new_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "BUILT",
                "builder": BUILDER_ID,
                "retained_samples": manifest["retained_samples"],
                "dataset_sha256": manifest["dataset_sha256"],
                "samples_npz": str(npz_path),
                "manifest": str(manifest_path),
                "dataset_manifest": str(dataset_manifest_path),
                "runtime_illegal_raw_verb_fraction": manifest[
                    "runtime_illegal_raw_verb_fraction"
                ],
                "per_level_coverage": manifest["per_level_coverage"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
