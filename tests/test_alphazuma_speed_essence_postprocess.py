from __future__ import annotations

import json

from tools.audit_alphazuma_speed_essence_result import (
    _independent_rankings,
    _new_bests as _audited_new_bests,
    _paired as _audited_paired,
)
from tools.run_alphazuma_speed_essence_postprocess import (
    SPEED_LEVEL_IDS,
    _new_level_bests,
    _paired_result,
    _selection_decision,
)


SELECTION_LEVELS = [
    "Jungle2",
    "Jungle4",
    "Jungle7",
    "Jungle9",
    "village4",
    "village5",
    "village6",
    "village7",
    "village10",
]


def _summary(model: dict[str, object], wins: dict[str, int], ticks: dict[str, int]):
    levels = []
    all_win_ticks = []
    for index, level_id in enumerate(SELECTION_LEVELS):
        count = wins.get(level_id, 0)
        best = None
        if count:
            tick = ticks.get(level_id, 2000 + index)
            all_win_ticks.extend([tick] * count)
            best = {
                "model_id": model["id"],
                "model_sha256": model["sha256"],
                "training_steps": model["training_steps"],
                "level_id": level_id,
                "attempt_index": 0,
                "seed": 100 + index,
                "outcome": "win",
                "ticks": tick,
                "seconds": tick / 100,
                "score": 1,
                "trajectory_sha256": f"sha256:{model['id']}-{level_id}",
            }
        levels.append(
            {
                "level_id": level_id,
                "attempts": 4,
                "wins": count,
                "losses": 4 - count,
                "truncations": 0,
                "best_attempt": best,
            }
        )
    return {
        "model": model,
        "levels": len(levels),
        "levels_cleared": sum(row["wins"] > 0 for row in levels),
        "attempts": len(levels) * 4,
        "wins": sum(row["wins"] for row in levels),
        "losses": len(levels) * 4 - sum(row["wins"] for row in levels),
        "truncations": 0,
        "win_rate": sum(row["wins"] for row in levels) / (len(levels) * 4),
        "median_win_ticks": None,
        "median_win_seconds": None,
        "best_attempt": None,
        "level_summaries": levels,
    }


def _master():
    return {
        "selection": {
            "robust_ranking": ["anchor", "hard breadth", "breadth", "wins", "median"],
            "speed_ranking": ["anchor", "speed breadth", "wins", "geomean"],
            "frozen_v1_best_tick_references": {
                "Jungle2": 1567,
                "Jungle4": 1865,
                "Jungle9": 5578,
                "village7": 3179,
            },
        }
    }


def test_controller_and_independent_auditor_choose_same_two_rankings() -> None:
    models = [
        {"id": "baseline", "training_steps": 1, "path": "/baseline", "sha256": "sha256:baseline"},
        {"id": "robust", "training_steps": 2, "path": "/robust", "sha256": "sha256:robust"},
        {"id": "speed", "training_steps": 3, "path": "/speed", "sha256": "sha256:speed"},
    ]
    baseline_wins = {level_id: 2 for level_id in SELECTION_LEVELS}
    robust_wins = {level_id: 1 for level_id in SELECTION_LEVELS}
    speed_wins = {level_id: 2 for level_id in SELECTION_LEVELS}
    baseline_wins["Jungle2"] = robust_wins["Jungle2"] = speed_wins["Jungle2"] = 3
    for level_id in ("Jungle7", "village6", "village10"):
        baseline_wins[level_id] = 0
        speed_wins[level_id] = 0
    # Robust owns the hard-level breadth; speed owns more total wins after the
    # four speed-reference levels tie on breadth.
    speed_ticks = {level_id: 500 for level_id in SPEED_LEVEL_IDS}
    aggregate = {
        "created_utc": "2026-08-14T00:00:00Z",
        "preregistration": {"sha256": "sha256:prereg"},
        "models_manifest": {"sha256": "sha256:manifest"},
        "validation": {"complete": True},
        "models": [
            _summary(models[0], baseline_wins, {}),
            _summary(models[1], robust_wins, {}),
            _summary(models[2], speed_wins, speed_ticks),
        ],
    }

    decision = _selection_decision(_master(), aggregate, models)
    audited_robust, audited_speed = _independent_rankings(_master(), aggregate, models)

    assert decision["robust_selected_model"]["id"] == audited_robust[0] == "robust"
    assert decision["speed_selected_model"]["id"] == audited_speed[0] == "speed"
    json.dumps(decision, allow_nan=False)


def test_incomplete_speed_reference_set_has_null_geometric_mean() -> None:
    model = {"id": "partial", "training_steps": 1, "path": "/partial", "sha256": "sha256:partial"}
    wins = {level_id: 1 for level_id in SELECTION_LEVELS if level_id != "Jungle9"}
    wins["Jungle2"] = 3
    aggregate = {
        "created_utc": "now",
        "preregistration": {},
        "models_manifest": {},
        "validation": {"complete": True},
        "models": [_summary(model, wins, {})],
    }

    decision = _selection_decision(_master(), aggregate, [model])

    metrics = decision["speed_ranking"][0]["metrics"]
    assert metrics["speed_levels_cleared"] == 3
    assert metrics["speed_reference_levels_complete"] is False
    assert metrics["speed_geometric_mean_ratio"] is None
    json.dumps(decision, allow_nan=False)


def test_new_best_and_paired_counts_recompute_identically() -> None:
    baseline = {"id": "baseline", "training_steps": 1, "sha256": "sha256:b", "path": "/b"}
    selected = {"id": "selected", "training_steps": 2, "sha256": "sha256:s", "path": "/s"}
    base_summary = _summary(baseline, {"Jungle2": 1}, {"Jungle2": 1600})
    selected_summary = _summary(selected, {"Jungle2": 1}, {"Jungle2": 1500})
    aggregate = {"models": [base_summary, selected_summary]}
    roles = {
        "v1_baseline": "baseline",
        "v11_baseline": "baseline",
        "robust_selected": "selected",
        "speed_selected": "selected",
    }
    rows = [
        {"model_id": "baseline", "level_id": "Jungle2", "attempt_index": 0, "outcome": "loss"},
        {"model_id": "selected", "level_id": "Jungle2", "attempt_index": 0, "outcome": "win"},
    ]

    assert _new_level_bests(aggregate, roles) == _audited_new_bests(aggregate, roles)
    assert _new_level_bests(aggregate, roles)[0]["improvement_ticks"] == 100
    assert _paired_result(rows, "selected", "baseline") == _audited_paired(rows, "selected", "baseline")
    assert _paired_result(rows, "selected", "selected")["both_win"] == 1
