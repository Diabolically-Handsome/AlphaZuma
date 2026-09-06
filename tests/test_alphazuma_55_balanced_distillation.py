from __future__ import annotations

import json

from tools.distill_alphazuma_55_balanced import _validate_preregistration


def test_balanced_teacher_map_and_preregistration_validate(tmp_path):
    from tools import build_alphazuma_55_balanced_distillation as prereg_builder
    from tools import build_alphazuma_55_balanced_teacher_map as map_builder

    teacher_map_path = tmp_path / "teacher-map.json"
    teacher_map = map_builder.build()
    teacher_map_path.write_text(json.dumps(teacher_map), encoding="utf-8")

    source_root = __import__("pathlib").Path(
        "/mnt/d/ZumaTraining/alphazuma-55-specialist-distillation-s85081501-v1"
    )
    value = prereg_builder.build_preregistration(
        master_path=__import__("pathlib").Path(
            "/mnt/c/Users/Laure/Documents/祖玛/diagnostics/"
            "alphazuma-55-weekend-s81081401-preregistration-v1.json"
        ),
        teacher_map_path=teacher_map_path,
        source_model_path=source_root / "final_model.zip",
        source_id="test-balanced-source",
        source_training_steps=34_444_283,
        source_completion=source_root / "completion.json",
        run_dir=tmp_path / "run",
        device="cuda:0",
        training_seed_base=1_521_000_000,
        model_seed=99,
    )
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(value), encoding="utf-8")

    validated = _validate_preregistration(prereg_path)

    assert validated["run"]["fire_samples_per_level_per_round"] == 192
    assert validated["run"]["teacher_execution_probability_after_round_zero"] == 0.75
    assert len(teacher_map["levels"]) == 55
    assert {
        row["observation_teacher_policy_id"] for row in teacher_map["levels"]
    } == {
        "geometric-actor-observable-v1",
        "strategic-actor-observable-v1",
    }
