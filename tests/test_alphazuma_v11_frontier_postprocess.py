from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.run_alphazuma_v11_frontier_postprocess import (
    _candidate_models,
    _paired_counts,
    _performance_report,
    _report_markdown,
    _selection_decision,
    _subset_metrics,
)


LEVEL_IDS = [
    "Jungle1",
    "Jungle2",
    "Jungle3",
    "Jungle4",
    "jungle6",
    "Jungle7",
    "Jungle8",
    "Jungle9",
    "Jungle10",
    "village1",
    "village2",
    "village4",
    "village5",
    "village6",
    "village7",
    "village8",
    "village10",
]
ZERO_WIN_IDS = ["Jungle7", "village6", "village10"]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _level_summary(level_id: str, wins: int, *, attempts: int = 16) -> dict[str, object]:
    return {
        "level_id": level_id,
        "attempts": attempts,
        "wins": wins,
        "losses": attempts - wins,
        "truncations": 0,
        "best_attempt": {"ticks": 1000 + LEVEL_IDS.index(level_id)} if wins else None,
    }


def _model_summary(
    model_id: str, wins: dict[str, int], *, attempts: int = 16
) -> dict[str, object]:
    return {
        "model": {"id": model_id, "training_steps": 1, "path": f"/{model_id}.zip"},
        "level_summaries": [
            _level_summary(level_id, wins.get(level_id, 0), attempts=attempts)
            for level_id in LEVEL_IDS
        ],
    }


def _write_matrix(
    root: Path,
    selected_wins: dict[str, int],
    baseline_wins: dict[str, int],
) -> None:
    shards = [[], []]
    for level_index, level_id in enumerate(LEVEL_IDS):
        for attempt_index in range(16):
            for model_id, wins in (
                ("baseline-v1", baseline_wins),
                ("selected-v11", selected_wins),
            ):
                shards[level_index % 2].append(
                    {
                        "level_id": level_id,
                        "attempt_index": attempt_index,
                        "model_id": model_id,
                        "outcome": "win" if attempt_index < wins.get(level_id, 0) else "loss",
                    }
                )
    for index, attempts in enumerate(shards):
        (root / f"matrix-shard-{index:02d}-of-02.json").write_text(
            json.dumps({"attempts": attempts}), encoding="utf-8"
        )


def test_zero_win_subset_uses_json_null_for_undefined_median() -> None:
    summary = {
        "level_summaries": [_level_summary(level_id, 0) for level_id in ZERO_WIN_IDS]
    }

    metrics = _subset_metrics(summary, ZERO_WIN_IDS)

    assert metrics["wins"] == 0
    assert metrics["median_best_win_ticks"] is None
    json.dumps(metrics, allow_nan=False)


def test_selection_ranking_is_anchor_first_and_strict_json_safe() -> None:
    candidates = [
        {"id": "baseline-v1", "training_steps": 1, "path": "/baseline.zip"},
        {"id": "recovered", "training_steps": 2, "path": "/recovered.zip"},
        {"id": "anchor-regressed", "training_steps": 3, "path": "/regressed.zip"},
    ]
    baseline_wins = {"Jungle2": 4}
    recovered_wins = {"Jungle2": 3, "Jungle7": 1}
    regressed_wins = {"Jungle2": 2, "Jungle7": 4, "village6": 4, "village10": 4}
    aggregate = {
        "created_utc": "2026-08-14T00:00:00Z",
        "preregistration": {"sha256": "sha256:prereg"},
        "models_manifest": {"sha256": "sha256:manifest"},
        "validation": {"complete": True},
        "models": [
            _model_summary("baseline-v1", baseline_wins, attempts=4),
            _model_summary("recovered", recovered_wins, attempts=4),
            _model_summary("anchor-regressed", regressed_wins, attempts=4),
        ],
    }
    master = {
        "zero_win_level_ids": ZERO_WIN_IDS,
        "selection": {
            "retention_anchor": "Jungle2",
            "ranking": ["anchor", "zero-win breadth", "zero-win wins"],
        },
    }

    decision = _selection_decision(master, aggregate, candidates)

    assert decision["selected_model"]["id"] == "recovered"
    assert decision["ranking"][-1]["model"]["id"] == "anchor-regressed"
    baseline = next(row for row in decision["ranking"] if row["model"]["id"] == "baseline-v1")
    assert baseline["metrics"]["median_best_win_ticks"] is None
    assert baseline["rank_key"][5] == -1_000_000_000.0
    json.dumps(decision, allow_nan=False)


def test_candidate_midpoint_is_measured_after_source_counter(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.zip"
    baseline.write_bytes(b"baseline")
    run_dir = tmp_path / "route"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for step in (150, 200, 250):
        (checkpoints / f"route_{step}_steps.zip").write_bytes(f"checkpoint-{step}".encode())
    final = run_dir / "final_model.zip"
    final.write_bytes(b"final")
    prereg_sha = "sha256:frozen-preregistration"
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "preregistration": {"sha256": prereg_sha},
                "run": {"id": "route-training-id"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "completion.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "source_steps": 100,
                "actual_steps": 300,
                "actual_steps_this_run": 200,
                "final_model": {"path": str(final), "sha256": _sha256(final)},
            }
        ),
        encoding="utf-8",
    )
    master = {
        "baseline_model": {
            "id": "baseline-v1",
            "training_steps": 900,
            "path": str(baseline),
            "sha256": _sha256(baseline),
        },
        "runs": [
            {
                "id": "route",
                "training_run_id": "route-training-id",
                "family": "test",
                "device": "cpu",
                "run_dir": str(run_dir),
                "preregistration_path": str(tmp_path / "unused-prereg.json"),
                "preregistration_sha256": prereg_sha,
                "source_counter_timesteps": 100,
                "source_effective_training_steps": 1000,
            }
        ],
    }

    candidates, receipts = _candidate_models(master)

    assert [candidate["id"] for candidate in candidates] == [
        "baseline-v1",
        "route-mid-200",
        "route-final-300",
    ]
    assert candidates[1]["frontier_steps"] == 100
    assert candidates[1]["training_steps"] == 1100
    assert candidates[2]["frontier_steps"] == 200
    assert candidates[2]["training_steps"] == 1200
    assert receipts[0]["midpoint_target_counter"] == 200


def test_full_report_is_paired_complete_and_strict_json_safe(tmp_path: Path) -> None:
    target_ids = [level_id for level_id in LEVEL_IDS if level_id != "Jungle2"]
    nonzero_targets = [level_id for level_id in target_ids if level_id not in ZERO_WIN_IDS]
    selected_wins = {level_id: 13 for level_id in nonzero_targets}
    selected_wins.update({"Jungle2": 14, "Jungle7": 4, "village6": 0, "village10": 0})
    baseline_wins = {level_id: 13 for level_id in nonzero_targets}
    baseline_wins.update({"Jungle2": 16, "Jungle7": 0, "village6": 0, "village10": 0})
    _write_matrix(tmp_path, selected_wins, baseline_wins)
    aggregate = {
        "validation": {
            "all_shards_complete": True,
            "attempt_matrix_exact": True,
            "paired_seed_matrix_exact": True,
        },
        "models": [
            _model_summary("baseline-v1", baseline_wins),
            _model_summary("selected-v11", selected_wins),
        ],
    }
    master = {
        "schedule": {"goal_end_utc": "2099-01-01T00:00:00Z"},
        "levels": [{"id": level_id} for level_id in LEVEL_IDS],
        "zero_win_level_ids": ZERO_WIN_IDS,
        "selection": {"retention_anchor": "Jungle2"},
        "promotion_gates": {
            "jungle2_wins_at_least": 14,
            "target_levels_cleared_at_least": 14,
            "target_wins_at_least": 168,
            "zero_win_levels_cleared_at_least": 1,
            "zero_win_total_wins_at_least": 4,
        },
        "stretch_goals": {
            "target_wins_at_least": 176,
            "zero_win_levels_cleared_at_least": 3,
            "zero_win_total_wins_at_least": 12,
        },
    }
    selection = {"selected_model": {"id": "synthetic-selected"}}

    report = _performance_report(
        master=master,
        selection=selection,
        aggregate=aggregate,
        output_root=tmp_path,
    )

    assert report["scientifically_valid"] is True
    assert report["v11_promotion_passed"] is True
    assert report["target_blind"]["selected"]["levels_cleared"] == 14
    assert report["target_blind"]["paired_comparison"]["pairs"] == 256
    assert report["zero_win_blind"]["baseline"]["median_best_win_ticks"] is None
    assert report["zero_win_blind"]["paired_comparison"]["pairs"] == 48
    assert "V1.1 promotion gate: PASS" in _report_markdown(report)
    json.dumps(report, allow_nan=False)


def test_paired_counts_close_exactly(tmp_path: Path) -> None:
    selected = {level_id: 1 for level_id in LEVEL_IDS}
    baseline = {level_id: 0 for level_id in LEVEL_IDS}
    _write_matrix(tmp_path, selected, baseline)

    paired = _paired_counts(
        output_root=tmp_path,
        shard_count=2,
        baseline_id="baseline-v1",
        selected_id="selected-v11",
        included_ids=LEVEL_IDS,
    )

    assert paired["pairs"] == 17 * 16
    assert sum(
        paired[key]
        for key in ("both_win", "selected_only_win", "baseline_only_win", "both_fail")
    ) == paired["pairs"]
