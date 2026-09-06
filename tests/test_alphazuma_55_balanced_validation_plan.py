from __future__ import annotations

import json
from pathlib import Path

from tools import build_alphazuma_55_balanced_validation_plan as builder
from tools.materialize_alphazuma_55_balanced_validation import validate_plan_static


def test_balanced_validation_plan_freezes_unused_paired_matrix(tmp_path):
    source_root = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-specialist-distillation-s85081501-v1"
    )
    balanced_root = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-balanced-distillation-s99081510-v1"
    )
    plan = builder.build(
        master_path=Path(
            "/mnt/c/Users/Laure/Documents/祖玛/diagnostics/"
            "alphazuma-55-weekend-s81081401-preregistration-v1.json"
        ),
        balanced_preregistration=Path(
            "/mnt/c/Users/Laure/Documents/祖玛/diagnostics/"
            "alphazuma-55-balanced-distillation-s99081510-preregistration-v1.json"
        ),
        source_model=source_root / "final_model.zip",
        source_completion=source_root / "completion.json",
        target_expected_path=balanced_root / "final_model.zip",
        target_expected_completion=balanced_root / "completion.json",
        original_root=Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"),
        output_root=tmp_path / "run",
        models_manifest=tmp_path / "models.json",
        preregistration_output=tmp_path / "prereg.json",
        audit_output=tmp_path / "audit.json",
        base_seed=1_550_000_110,
        device="cuda:1",
        parallel_envs=24,
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    validated = validate_plan_static(path, builder._sha256(path))

    assert validated["matrix"]["base_seed"] == 1_550_000_110
    assert validated["matrix"]["last_seed"] == 1_550_000_164
    assert validated["matrix"]["prior_consumed_ranges"] == [
        [1_550_000_000, 1_550_000_054],
        [1_550_000_055, 1_550_000_109],
    ]
    assert validated["authority_boundary"]["formal_seed_consumption"] is False
