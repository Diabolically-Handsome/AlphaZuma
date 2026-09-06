from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tools import build_alphazuma_55_motor_observable_replay_v4 as builder


OBS_DIM = 4
AIM_BINS = 8
MASK_SIZE = len(builder.VERB_NAMES) + AIM_BINS


def _make_episode(
    *,
    level_id: str,
    seed: int,
    episode_code: int,
    total_ticks: int,
    fire_edges: tuple[int, ...] = (),
    verb_overrides: dict[int, int] | None = None,
    runtime_illegal_ticks: tuple[int, ...] = (),
    danger_field: str | None = "chain_length",
    danger_baseline: float = 0.0,
    danger_values: dict[int, float] | None = None,
    extra_info: dict[str, dict[int, float]] | None = None,
    outcome: str = "win",
) -> dict[str, Any]:
    """Synthetic full-tick episode in the v4 episode-record format.

    ``observation[0]`` carries the episode code and ``observation[1]`` the
    tick index so retained rows can be mapped back to source ticks.
    ``runtime_illegal_ticks`` marks ticks whose EXACT runtime mask forbids
    the raw verb (the relaxed training mask still allows it, mirroring the
    ``_intent_label`` relaxation the builder must preserve, not validate
    away).
    """

    fire_set = {int(value) for value in fire_edges}
    verb_overrides = verb_overrides or {}
    illegal_set = {int(value) for value in runtime_illegal_ticks}
    danger_values = danger_values or {}
    extra_info = extra_info or {}
    ticks = []
    for tick_index in range(total_ticks):
        verb = builder.FIRE_VERB_INDEX if tick_index in fire_set else 0
        verb = int(verb_overrides.get(tick_index, verb))
        aim = tick_index % AIM_BINS
        info: dict[str, Any] = {}
        if danger_field is not None:
            info[danger_field] = float(
                danger_values.get(tick_index, danger_baseline)
            )
        for name, values in extra_info.items():
            info[name] = float(values.get(tick_index, 0.0))
        exact_mask = np.ones(MASK_SIZE, dtype=np.bool_)
        if tick_index in illegal_set:
            exact_mask[verb] = False
        ticks.append(
            {
                "observation": np.asarray(
                    [episode_code, tick_index, verb, aim], dtype=np.float32
                ),
                "raw_action": np.asarray([verb, aim], dtype=np.int64),
                "training_mask": np.ones(MASK_SIZE, dtype=np.bool_),
                "exact_mask": exact_mask,
                "info": info,
            }
        )
    return {
        "level_id": level_id,
        "seed": seed,
        "outcome": outcome,
        "time_limit_truncated": False,
        "observation_capacity_overflow": False,
        "ticks": ticks,
    }


def _retained_keys(rows: list[Any]) -> set[tuple[int, int]]:
    return {(int(row[0][0]), int(row[0][1])) for row in rows}


def _jungle_and_volcano() -> list[dict[str, Any]]:
    """Short easy level plus a 4x longer hard level (2 episodes each)."""

    return [
        _make_episode(
            level_id="jungle-01",
            seed=100,
            episode_code=0,
            total_ticks=500,
            fire_edges=(100,),
            danger_baseline=7.0,
        ),
        _make_episode(
            level_id="jungle-01",
            seed=101,
            episode_code=1,
            total_ticks=500,
            fire_edges=(100,),
            danger_baseline=7.0,
        ),
        _make_episode(
            level_id="volcano-13",
            seed=200,
            episode_code=2,
            total_ticks=2000,
            fire_edges=(200, 900, 1500),
            danger_baseline=7.0,
        ),
        _make_episode(
            level_id="volcano-13",
            seed=201,
            episode_code=3,
            total_ticks=2000,
            fire_edges=(200, 900, 1500),
            danger_baseline=7.0,
        ),
    ]


def _replay_v2_consumer_stub(
    rows: list[Any], collection: dict[str, Any]
) -> dict[str, list[Any]]:
    """Consume the dataset exactly the way the replay-v2 trainers do."""

    groups: dict[str, list[Any]] = {name: [] for name in builder.VERB_NAMES}
    for row in rows:
        observation, action, mask = row
        assert observation.dtype == np.float32 and observation.ndim == 1
        assert action.dtype == np.int64 and action.shape == (2,)
        assert mask.dtype == np.bool_ and mask.ndim == 1
        verb = int(action[0])
        aim = int(action[1])
        assert bool(mask[verb])
        assert bool(mask[len(builder.VERB_NAMES) + aim])
        groups[builder.VERB_NAMES[verb]].append(row)
    assert collection["retained_samples"] == len(rows)
    for name, group in groups.items():
        assert collection["retained_samples_by_verb"].get(name, 0) == len(
            group
        )
    for key in (
        "round_index",
        "wall_seconds",
        "episodes",
        "wins",
        "losses",
        "truncations",
        "capacity_overflows",
        "raw_intent_action_counts",
        "effective_execution_action_counts",
        "intent_mask_relaxations",
        "training_label_semantics",
        "environment_execution_semantics",
        "teacher_policy_id",
        "dataset_sha256",
    ):
        assert key in collection
    for episode in collection["episodes"]:
        for key in (
            "level_id",
            "seed",
            "steps",
            "outcome",
            "time_limit_truncated",
            "raw_intent_action_counts",
            "effective_execution_action_counts",
            "intent_mask_relaxations",
            "candidate_samples_by_raw_verb",
            "retained_samples_by_raw_verb",
            "retained_samples",
            "observation_capacity_overflow",
        ):
            assert key in episode
    return groups


def test_coverage_ratio_is_equalized_across_levels() -> None:
    rows, manifest = builder.build_coverage_equalized_replay(
        _jungle_and_volcano(),
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.25,
            total_budget=1_000_000,
            sampling_seed=7,
        ),
    )
    coverage = manifest["per_level_coverage"]
    jungle = coverage["jungle-01"]
    volcano = coverage["volcano-13"]
    assert jungle["total_source_ticks"] == 1000
    assert volcano["total_source_ticks"] == 4000
    assert abs(jungle["coverage_ratio"] - 0.25) < 1e-9
    assert abs(volcano["coverage_ratio"] - 0.25) < 1e-9
    assert (
        abs(jungle["coverage_ratio"] - volcano["coverage_ratio"]) < 1e-9
    )
    assert len(rows) == jungle["retained_ticks"] + volcano["retained_ticks"]
    assert manifest["coverage_equalization"]["budget_cap_applied"] is False
    # The flat baseline danger signal must not mark calm ticks mandatory.
    assert jungle["danger_ticks"] == 0
    assert volcano["danger_ticks"] == 0


def test_total_budget_caps_while_preserving_equal_ratios() -> None:
    rows, manifest = builder.build_coverage_equalized_replay(
        _jungle_and_volcano(),
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.25,
            total_budget=600,
            sampling_seed=7,
        ),
    )
    coverage = manifest["per_level_coverage"]
    jungle = coverage["jungle-01"]
    volcano = coverage["volcano-13"]
    assert len(rows) == 600
    assert jungle["retained_ticks"] + volcano["retained_ticks"] == 600
    assert abs(jungle["coverage_ratio"] - 0.12) < 1e-9
    assert abs(volcano["coverage_ratio"] - 0.12) < 1e-9
    assert manifest["coverage_equalization"]["budget_cap_applied"] is True
    # Mandatory tiers stayed below the quotas, so the cap is exact.
    assert jungle["mandatory_ticks"] <= jungle["quota_ticks"]
    assert volcano["mandatory_ticks"] <= volcano["quota_ticks"]


def test_fire_windows_are_fully_retained_even_when_quota_is_smaller() -> None:
    episode = _make_episode(
        level_id="temple-05",
        seed=300,
        episode_code=9,
        total_ticks=300,
        fire_edges=(10, 150, 151, 296),
        danger_field=None,
    )
    rows, manifest = builder.build_coverage_equalized_replay(
        [episode],
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.15,
            total_budget=50,
            fire_window_before_ticks=24,
            fire_window_after_ticks=6,
        ),
    )
    retained = _retained_keys(rows)
    expected_window_ticks: set[tuple[int, int]] = set()
    # 151 is a held fire intent, not a rising edge; windows clip at bounds.
    for edge in (10, 150, 296):
        for tick in range(max(0, edge - 24), min(300, edge + 6 + 1)):
            expected_window_ticks.add((9, tick))
    assert expected_window_ticks <= retained
    assert len(rows) == len(retained)
    coverage = manifest["per_level_coverage"]["temple-05"]
    assert coverage["fire_edges"] == 3
    assert coverage["fire_window_ticks"] == len(expected_window_ticks)
    # Mandatory tier overflows the quota and stays visible in the manifest.
    assert coverage["retained_ticks"] == len(expected_window_ticks)
    assert coverage["retained_ticks"] > coverage["quota_ticks"]
    assert coverage["uniform_ticks"] == 0


def test_danger_ticks_are_retained_and_field_is_reported() -> None:
    spike = {tick: 40.0 for tick in range(600, 700)}
    episode = _make_episode(
        level_id="volcano-13",
        seed=400,
        episode_code=5,
        total_ticks=1000,
        danger_field="chain_length",
        danger_baseline=10.0,
        danger_values=spike,
    )
    rows, manifest = builder.build_coverage_equalized_replay(
        [episode],
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.05,
            total_budget=1_000_000,
            danger_threshold=0.75,
        ),
    )
    retained = _retained_keys(rows)
    assert {(5, tick) for tick in range(600, 700)} <= retained
    equalization = manifest["coverage_equalization"]
    assert equalization["danger_field_used"] == "chain_length"
    assert equalization["danger_normalization"] == "per_level_max"
    coverage = manifest["per_level_coverage"]["volcano-13"]
    assert coverage["danger_ticks"] == 100
    assert coverage["retained_ticks"] >= 100


def test_danger_field_priority_prefers_direct_fraction() -> None:
    spike = {tick: 40.0 for tick in range(600, 700)}
    fraction = {tick: 0.9 for tick in range(50, 60)}
    episode = _make_episode(
        level_id="volcano-13",
        seed=401,
        episode_code=6,
        total_ticks=1000,
        danger_field="chain_length",
        danger_baseline=10.0,
        danger_values=spike,
        extra_info={"danger_fraction": fraction},
    )
    rows, manifest = builder.build_coverage_equalized_replay(
        [episode],
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.05,
            total_budget=1_000_000,
            danger_threshold=0.75,
        ),
    )
    retained = _retained_keys(rows)
    assert {(6, tick) for tick in range(50, 60)} <= retained
    equalization = manifest["coverage_equalization"]
    assert equalization["danger_field_used"] == "danger_fraction"
    assert equalization["danger_normalization"] == "raw_fraction"
    assert set(equalization["danger_field_candidates_present"]) == {
        "danger_fraction",
        "chain_length",
    }
    assert manifest["per_level_coverage"]["volcano-13"]["danger_ticks"] == 10


def test_rows_and_manifest_remain_replay_v2_compatible() -> None:
    episodes = _jungle_and_volcano()
    # Mixed verbs must group cleanly in a replay-v2-style consumer.
    episodes[0]["ticks"][50]["raw_action"] = np.asarray(
        [2, 50 % AIM_BINS], dtype=np.int64
    )
    episodes[0]["ticks"][51]["raw_action"] = np.asarray(
        [3, 51 % AIM_BINS], dtype=np.int64
    )
    rows, manifest = builder.build_coverage_equalized_replay(
        episodes,
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.25,
            total_budget=1_000_000,
        ),
        round_index=2,
        teacher_policy_id="curve-aware-settled-strategic-v3",
    )
    groups = _replay_v2_consumer_stub(rows, manifest)
    assert manifest["builder"] == "v4-coverage"
    assert manifest["builder_version"] == 4
    assert manifest["round_index"] == 2
    assert manifest["teacher_policy_id"] == (
        "curve-aware-settled-strategic-v3"
    )
    assert manifest["wins"] == 4
    assert manifest["losses"] == 0
    assert len(groups["fire"]) > 0
    assert manifest["raw_intent_action_counts"]["fire"] == 8
    assert manifest["training_label_semantics"] == "raw_teacher_intent"
    assert (
        manifest["environment_execution_semantics"]
        == "exact_mask_effective_action"
    )


def test_manifest_reports_per_level_starvation_fields() -> None:
    rows, manifest = builder.build_coverage_equalized_replay(
        _jungle_and_volcano(),
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=0.25,
            total_budget=600,
            sampling_seed=11,
        ),
    )
    coverage = manifest["per_level_coverage"]
    assert list(coverage) == ["jungle-01", "volcano-13"]
    for level_id, entry in coverage.items():
        for key in (
            "total_source_ticks",
            "quota_ticks",
            "retained_ticks",
            "coverage_ratio",
            "fire_window_ticks",
            "danger_ticks",
            "mandatory_ticks",
            "uniform_ticks",
            "fire_edges",
            "episodes_used",
        ):
            assert key in entry, (level_id, key)
        assert entry["episodes_used"] == 2
        assert entry["uniform_ticks"] == (
            entry["retained_ticks"] - entry["mandatory_ticks"]
        )
    assert (
        sum(entry["retained_ticks"] for entry in coverage.values())
        == len(rows)
        == manifest["retained_samples"]
    )


def test_same_seed_reproduces_identical_dataset_bytes() -> None:
    config = builder.CoverageReplayV4Config(
        target_coverage_ratio=0.2,
        total_budget=700,
        sampling_seed=13,
    )
    first_rows, first = builder.build_coverage_equalized_replay(
        _jungle_and_volcano(), config=config
    )
    second_rows, second = builder.build_coverage_equalized_replay(
        _jungle_and_volcano(), config=config
    )
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["per_level_coverage"] == second["per_level_coverage"]
    assert len(first_rows) == len(second_rows)


def test_masked_raw_aim_is_rejected() -> None:
    episode = _make_episode(
        level_id="jungle-01",
        seed=500,
        episode_code=0,
        total_ticks=10,
        danger_field=None,
    )
    bad_tick = episode["ticks"][3]
    aim = int(bad_tick["raw_action"][1])
    bad_tick["training_mask"][len(builder.VERB_NAMES) + aim] = False
    with pytest.raises(ValueError, match="raw teacher aim is masked"):
        builder.build_coverage_equalized_replay([episode])
    with pytest.raises(ValueError, match="no episode records"):
        builder.build_coverage_equalized_replay([])


def test_missing_or_wrong_size_exact_mask_is_rejected() -> None:
    episode = _make_episode(
        level_id="jungle-01",
        seed=600,
        episode_code=0,
        total_ticks=10,
        danger_field=None,
    )
    del episode["ticks"][4]["exact_mask"]
    with pytest.raises(ValueError, match="misses exact_mask"):
        builder.build_coverage_equalized_replay([episode])
    episode = _make_episode(
        level_id="jungle-01",
        seed=601,
        episode_code=0,
        total_ticks=10,
        danger_field=None,
    )
    episode["ticks"][2]["exact_mask"] = np.ones(
        MASK_SIZE - 1, dtype=np.bool_
    )
    with pytest.raises(ValueError, match="changes array sizes"):
        builder.build_coverage_equalized_replay([episode])


def test_runtime_illegal_raw_verbs_are_preserved_and_reported() -> None:
    """Exact-mask-illegal teacher verbs must survive into the dataset.

    The v1 blind-metric failure family: rejecting (or relaxing away)
    runtime-illegal raw verbs makes the downstream
    illegal_teacher_verb_fraction read a vacuous 0.0.  The builder must
    keep those ticks and report the true fraction.
    """

    episode = _make_episode(
        level_id="jungle-01",
        seed=602,
        episode_code=0,
        total_ticks=40,
        fire_edges=(10, 20),
        runtime_illegal_ticks=(20, 33),
        danger_field=None,
    )
    rows, manifest = builder.build_coverage_equalized_replay(
        [episode],
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=1.0, total_budget=1_000_000
        ),
    )
    assert len(rows) == 40
    fraction = manifest["runtime_illegal_raw_verb_fraction"]
    assert fraction["source"] == pytest.approx(2 / 40)
    assert fraction["retained"] == pytest.approx(2 / 40)
    assert manifest["mask_semantics"] == {
        "exact_masks": "exact_runtime_valid_action_mask",
        "training_masks": "relaxed_training_mask_teacher_verb_unmasked",
    }
    assert manifest["episodes"][0]["runtime_illegal_raw_verb_ticks"] == 2


def _trainer_scale_episode(
    *,
    level_id: str,
    seed: int,
    episode_code: int,
    total_ticks: int,
    mask_width: int,
    fire_edge: int,
    runtime_illegal_fire: int,
) -> dict[str, Any]:
    """JSON-serializable episode at the real 184-wide mask interface."""

    parked_bin = 10 + 7 * episode_code
    ticks = []
    for tick_index in range(total_ticks):
        verb = (
            builder.FIRE_VERB_INDEX
            if tick_index in (fire_edge, runtime_illegal_fire)
            else 0
        )
        exact_mask = [True] * mask_width
        if tick_index == runtime_illegal_fire:
            exact_mask[builder.FIRE_VERB_INDEX] = False
        ticks.append(
            {
                "observation": [float(episode_code), float(tick_index)],
                "raw_action": [verb, parked_bin],
                "training_mask": [True] * mask_width,
                "exact_mask": exact_mask,
                "info": {},
            }
        )
    return {
        "level_id": level_id,
        "seed": seed,
        "outcome": "win",
        "ticks": ticks,
    }


def test_builder_output_feeds_park_settle_trainer_end_to_end(
    tmp_path,
) -> None:
    """REAL builder output (not synthetic fixtures) must load as
    temporal_structure == 'episode_ticks' with exact runtime masks.

    Regression for the interoperability blocker: the v4 builder used to
    write one aggregate NPZ keyed 'training_masks' with no per-row
    identity arrays and a manifest without shard paths, which
    tools.distill_alphazuma_55_park_settle_v1.load_replay_dataset could
    not load at all -- and even a bridged load would have reported
    illegal_teacher_verb_fraction == 0.0 from the relaxed masks.
    """

    import json as json_module

    from tools import distill_alphazuma_55_park_settle_v1 as park_settle

    ticks_per_episode = 80
    episodes = [
        _trainer_scale_episode(
            level_id="jungle-01",
            seed=700,
            episode_code=0,
            total_ticks=ticks_per_episode,
            mask_width=park_settle.MASK_WIDTH,
            fire_edge=30,
            runtime_illegal_fire=50,
        ),
        _trainer_scale_episode(
            level_id="volcano-13",
            seed=701,
            episode_code=1,
            total_ticks=ticks_per_episode,
            mask_width=park_settle.MASK_WIDTH,
            fire_edge=40,
            runtime_illegal_fire=60,
        ),
    ]
    episodes_path = tmp_path / "episodes.json"
    episodes_path.write_text(
        json_module.dumps({"episodes": episodes}), encoding="utf-8"
    )
    prefix = tmp_path / "replay"
    assert (
        builder.main(
            [
                "--episodes",
                str(episodes_path),
                "--output-prefix",
                str(prefix),
                "--target-coverage-ratio",
                "1.0",
                "--total-budget",
                "1000000",
            ]
        )
        == 0
    )

    dataset_manifest_path = tmp_path / "replay.dataset.json"
    assert dataset_manifest_path.exists()
    dataset, receipt = park_settle.load_replay_dataset(
        dataset_manifest_path
    )
    assert dataset.temporal_structure == "episode_ticks"
    assert dataset.sample_count == 2 * ticks_per_episode
    assert len(np.unique(dataset.episode_indices)) == 2
    # Level-major output order: jungle episode first, ticks ascending.
    assert np.array_equal(
        dataset.tick_indices[:ticks_per_episode],
        np.arange(ticks_per_episode, dtype=np.int64),
    )
    # The trainer reads the EXACT runtime mask key, and the manifest
    # declares its semantics (recorded in the receipt).
    assert receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )
    assert receipt["entries"][0]["keys"]["episode_indices"] == (
        "episode_indices"
    )
    assert receipt["entries"][0]["keys"]["tick_indices"] == "tick_indices"

    annotations = park_settle.annotate_park_settle(
        dataset, park_settle.ParkSettleLossConfig()
    )
    # The two runtime-illegal fires survive with a nonzero honest
    # fraction (2/160), are excluded from the commits, and the exact
    # pre-fire windows resolve from real tick indices.
    assert int(np.sum(~annotations.verb_legal)) == 2
    assert int(np.sum(annotations.fire_commit)) == 2
    assert int(np.sum(annotations.fire_edge)) == 2
    assert int(np.sum(annotations.pre_fire_window)) == 2 * 25

    collection = json_module.loads(
        (tmp_path / "replay.manifest.json").read_text(encoding="utf-8")
    )
    assert collection["trainer_dataset_manifest"] == "replay.dataset.json"
    assert collection["runtime_illegal_raw_verb_fraction"][
        "source"
    ] == pytest.approx(2 / 160)
