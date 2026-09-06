from __future__ import annotations

from tools import audit_alphazuma_55_successor_result as independent
from tools.run_alphazuma_55_successor_certification import (
    _capability,
    _continuous_campaigns,
    _engineering_decision,
)


def _models() -> list[dict]:
    return [
        {"id": model_id, "path": f"/{model_id}.zip", "sha256": f"hash-{index}", "training_steps": index}
        for index, model_id in enumerate(independent.EXPECTED_IDS)
    ]


def _engineering_rows(win_counts: list[int]) -> list[dict]:
    rows = []
    for model_index, (model, wins) in enumerate(zip(_models(), win_counts, strict=True)):
        for level_index in range(55):
            won = level_index < wins
            rows.append(
                {
                    "model_id": model["id"],
                    "level_id": f"level-{level_index:02d}",
                    "outcome": "win" if won else "loss",
                    "ticks": 1000 + model_index * 10 + level_index,
                    "score": 100 + model_index if won else model_index,
                }
            )
    return rows


def test_engineering_ranking_and_independent_recomputation_match() -> None:
    models = _models()
    rows = _engineering_rows([34, 35, 40, 39])
    decision = _engineering_decision(
        models=models,
        rows=rows,
        minimum_levels=35,
        minimum_wins=35,
    )
    independently_ranked = independent._rank(models, rows)
    assert decision["ranking"] == independently_ranked
    assert decision["selected_policy"]["id"] == "polar-dagger-round-01"
    assert decision["promotion_gate"]["passed"] is True
    assert decision["successor_formal_seed_consumption_authorized"] is True


def test_engineering_gate_keeps_formal_seeds_embargoed_below_threshold() -> None:
    decision = _engineering_decision(
        models=_models(),
        rows=_engineering_rows([20, 30, 34, 33]),
        minimum_levels=35,
        minimum_wins=35,
    )
    assert decision["status"] == "NO_PROMOTION"
    assert decision["promotion_gate"]["passed"] is False
    assert decision["successor_formal_seed_consumption_authorized"] is False


def test_capability_gate_requires_all_four_conditions() -> None:
    summary = {
        "levels_cleared": 55,
        "wins": 220,
        "attempts": 440,
        "level_summaries": [{"wins": 4} for _ in range(55)],
    }
    actual, passed = _capability(summary)
    assert actual == {
        "levels_cleared": 55,
        "minimum_wins_per_level": 4,
        "total_wins": 220,
        "attempts": 440,
    }
    assert passed is True
    summary["level_summaries"][17]["wins"] = 0
    _, passed = _capability(summary)
    assert passed is False


def test_continuous_campaign_requires_one_complete_55_level_run() -> None:
    level_ids = [f"level-{index:02d}" for index in range(55)]
    rows = []
    for campaign in range(4):
        for level_index, level_id in enumerate(level_ids):
            won = campaign == 2 or level_index != 20
            rows.append(
                {
                    "model_id": "policy",
                    "attempt_index": campaign,
                    "level_id": level_id,
                    "outcome": "win" if won else "loss",
                    "seed": 2_200_000_000 + level_index * 4 + campaign,
                }
            )
    result = _continuous_campaigns(rows, level_ids=level_ids, model_id="policy")
    independently = independent._continuous(rows, level_ids, "policy")
    assert result == independently
    assert result["passed"] is True
    assert result["campaigns_cleared"] == 1
