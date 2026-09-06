from __future__ import annotations

import json
from types import SimpleNamespace

from tools import build_alphazuma_55_eval_contract as eval_contract
from tools.run_alphazuma_55_postprocess import (
    _continuous_campaigns,
    _selection_decision,
)
from tools import build_alphazuma_55_postprocess_parallel_v2 as parallel_builder
from tools import run_alphazuma_55_postprocess_parallel_v2 as parallel_controller
from tools.audit_alphazuma_55_result import (
    _continuous as _audited_continuous,
    _rankings as _audited_rankings,
)


def _candidate(model_id: str, digest_suffix: str) -> dict[str, object]:
    return {
        "id": model_id,
        "training_steps": 1,
        "path": f"/{model_id}.zip",
        "sha256": f"sha256:{digest_suffix:0>64}",
    }


def _summary(model: dict[str, object], wins: list[int], median: float):
    return {
        "model": model,
        "attempts": 220,
        "wins": sum(wins),
        "losses": 220 - sum(wins),
        "truncations": 0,
        "win_rate": sum(wins) / 220,
        "levels": 55,
        "levels_cleared": sum(value > 0 for value in wins),
        "median_win_ticks": median,
        "median_win_seconds": median / 100.0,
        "best_attempt": None,
        "level_summaries": [
            {
                "level_id": f"level-{index:02d}",
                "wins": value,
                "attempts": 4,
                "best_attempt": None,
            }
            for index, value in enumerate(wins)
        ],
    }


def test_robust_and_speed_rankings_use_frozen_different_priorities() -> None:
    total = _candidate("total", "2")
    floor = _candidate("floor", "1")
    aggregate = {
        "models": [
            _summary(total, [4] * 54 + [1], 200.0),
            _summary(floor, [2] * 55, 100.0),
        ]
    }

    decision = _selection_decision(aggregate, [total, floor])

    assert decision["selected_single_policy"]["id"] == "floor"
    assert decision["speed_report_model"]["id"] == "total"
    assert decision["robust_ranking"][0]["metrics"]["minimum_wins_per_level"] == 2
    assert decision["speed_ranking"][0]["metrics"]["total_wins"] == 217
    assert _audited_rankings(aggregate, [total, floor]) == (
        ["floor", "total"],
        ["total", "floor"],
    )


def test_continuous_gate_requires_one_complete_no_retry_campaign() -> None:
    levels = [f"level-{index:02d}" for index in range(55)]
    rows = []
    for campaign in range(4):
        for level_index, level_id in enumerate(levels):
            rows.append(
                {
                    "model_id": "single",
                    "attempt_index": campaign,
                    "level_id": level_id,
                    "seed": 1_800_000_000 + level_index * 4 + campaign,
                    "outcome": "win" if campaign == 2 or level_index != 17 else "loss",
                }
            )

    result = _continuous_campaigns(rows, level_ids=levels, model_id="single")

    assert result["passed"] is True
    assert result["campaigns_cleared"] == 1
    assert result["campaigns"][0]["first_failure_level"] == "level-17"
    assert result["campaigns"][2]["first_failure_level"] is None
    assert _audited_continuous(rows, levels, "single") == result


def test_continuous_contract_uses_all_220_frozen_seeds(tmp_path, monkeypatch) -> None:
    class FakeCatalog:
        def __init__(self, _root):
            pass

        def level(self, level_id):
            return SimpleNamespace(id=level_id, display_name=level_id, curve_names=(level_id,))

        def load_level(self, _level_id, hard=False):
            assert hard is False
            return SimpleNamespace(curves=(object(),))

    monkeypatch.setattr(eval_contract, "OriginalGameCatalog", FakeCatalog)
    master_path = tmp_path / "master.json"
    master_path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
                "version": 1,
                "campaign_id": "fixture",
                "scope": {"included_levels_in_adventure_order": list(eval_contract.INCLUDED_LEVELS)},
                "seed_registry": {
                    "continuous_campaign_challenge": {
                        "first": 1_800_000_000,
                        "last": 1_800_000_219,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# fixture\n", encoding="utf-8")

    _, preregistration = eval_contract.build_contracts(
        master_path=master_path,
        original_root=tmp_path,
        evaluator_path=evaluator,
        stage="continuous",
        output_root=tmp_path / "out",
        models=[{"id": "single", "training_steps": 1, "path": "/single.zip", "sha256": "sha256:x"}],
        parallel_envs_per_shard=24,
    )

    assert preregistration["attempts_per_level"] == 4
    assert preregistration["total_attempts"] == 220
    assert preregistration["seed_plan"] == {
        "registry": "continuous_campaign_challenge",
        "base_seed": 1_800_000_000,
        "last_seed": 1_800_000_219,
    }
    assert preregistration["environment"]["base_config"]["max_ticks"] == 180_000


def test_parallel_controller_uses_frozen_evaluation_width(tmp_path, monkeypatch) -> None:
    captured = {}
    (tmp_path / "controller").mkdir()

    def fake_build_contracts(**kwargs):
        captured.update(kwargs)
        return {"schema": "manifest"}, {"schema": "preregistration"}

    monkeypatch.setattr(parallel_controller, "build_contracts", fake_build_contracts)
    parallel_controller._write_evaluation_contracts(
        contract={
            "master_path": tmp_path / "master.json",
            "original_root": tmp_path / "retail",
            "parallel_envs_per_shard": parallel_builder.EVALUATION_PARALLEL_ENVS_PER_SHARD,
        },
        stage="selection",
        models=[_candidate("single", "1")],
        output_root=tmp_path / "evaluation",
        controller_root=tmp_path / "controller",
        manifest_name="manifest.json",
        preregistration_name="preregistration.json",
        extra_manifest={"purpose": "fixture"},
    )

    assert captured["parallel_envs_per_shard"] == 112
    assert json.loads((tmp_path / "controller" / "manifest.json").read_text())[
        "purpose"
    ] == "fixture"


def test_six_routes_freeze_exactly_fifteen_candidates(monkeypatch) -> None:
    migrations = [_candidate(f"baseline-{index}", str(index + 1)) for index in range(3)]
    routes = [{"id": f"route-{index}"} for index in range(6)]

    def fake_route_candidates(run):
        index = int(str(run["id"]).split("-")[-1])
        return (
            [
                _candidate(f"{run['id']}-mid", str(10 + index * 2)),
                _candidate(f"{run['id']}-final", str(11 + index * 2)),
            ],
            {"route_id": run["id"], "status": "FROZEN"},
        )

    monkeypatch.setattr(
        parallel_controller,
        "_frozen_route_candidates",
        fake_route_candidates,
    )

    candidates, receipts = parallel_controller._freeze_candidates(
        {"migrations": migrations, "routes": routes}
    )

    assert len(candidates) == 15
    assert len(receipts) == 6
    assert len({str(candidate["id"]) for candidate in candidates}) == 15
    assert [receipt["route_id"] for receipt in receipts] == [
        f"route-{index}" for index in range(6)
    ]
