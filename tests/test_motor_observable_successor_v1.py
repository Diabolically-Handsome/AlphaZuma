from __future__ import annotations

from pathlib import Path

from tools import build_alphazuma_55_motor_observable_successor_master_v1 as builder
from tools import run_alphazuma_55_motor_observable_successor_v1 as controller


def test_master_freezes_unseen_formal_ranges_and_real_goal(tmp_path: Path) -> None:
    output = tmp_path / "master.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1], output=output
    )
    assert value["engineering"]["promotion_gate"]["minimum_wins"] == 35
    assert value["engineering"]["promotion_gate"][
        "minimum_cleared_levels"
    ] == 35
    assert value["seed_registry"]["final_blind"] == {
        "first": 3_300_000_000,
        "last": 3_300_000_439,
        "matrix": "55 levels x 8 attempts for one frozen policy",
        "embargo_until_engineering_promotion_is_frozen": True,
    }
    assert value["seed_registry"]["continuous_campaign_challenge"][
        "first"
    ] == 3_400_000_000
    assert value["formal_gates"]["single_policy_capability_gate"] == {
        "levels_cleared": 55,
        "minimum_wins_per_level": 1,
        "total_wins": 220,
        "attempts": 440,
    }
    assert value["formal_gates"]["continuous_full_game_gate"][
        "campaigns_cleared"
    ] == 1
    assert value["authority_boundary"]["engineering_result_unseen_at_freeze"]


def test_contract_builder_uses_level_major_seed_formula(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    contract = {
        "path": tmp_path / "master.json",
        "template": {
            "levels": [
                {"id": f"level-{index}", "display_name": "x", "curve_count": 1}
                for index in range(55)
            ],
            "environment": {"profile_mode": "tutorials_completed"},
        },
        "implementation": {
            "evaluator": controller.PROJECT_ROOT
            / "tools/evaluate_alphazuma_55_motor_observable.py"
        },
        "outputs": {
            "controller": controller_root,
            "final_blind": tmp_path / "final",
            "continuous": tmp_path / "continuous",
        },
        "devices": ["cuda:0", "cuda:1"],
        "parallel_envs_per_shard": 24,
    }
    Path(contract["path"]).write_text("{}", encoding="utf-8")
    decision = controller_root / "decision.json"
    decision.write_text("{}", encoding="utf-8")
    selected = {
        "id": "motor",
        "path": str(model),
        "sha256": controller._sha256(model),
        "training_steps": 1,
        "observation_profile": "motor-observable-v1",
        "promotable": True,
    }
    manifest, prereg = controller._build_contracts(
        contract=contract,
        stage="final_blind",
        model=selected,
        decision_path=decision,
    )
    value = controller._read(prereg)
    assert controller._read(manifest)["models"] == [selected]
    assert value["attempts_per_level"] == 8
    assert value["total_attempts"] == 440
    assert value["seed_plan"]["base_seed"] == 3_300_000_000
    assert value["seed_plan"]["last_seed"] == 3_300_000_439
    assert value["seed_plan"]["seed_formula"].startswith("base_seed")
