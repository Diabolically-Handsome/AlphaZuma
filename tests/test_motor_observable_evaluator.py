from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import distill_alphazuma_55_motor_observable_replay_v2 as trainer
from tools import evaluate_alphazuma_55_motor_observable as evaluator
from zuma_rl.motor_observable_migration import (
    transplant_appended_global_policy,
)
from zuma_rl.observable_human_speedrun import MOTOR_OBSERVATION_PROFILE


def _real_root() -> Path:
    return Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


def test_manifest_requires_explicit_observation_profile(tmp_path: Path) -> None:
    source = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-wide-"
        "s99081540-v1/epoch_10_model.zip"
    )
    if not source.exists():
        pytest.skip("source model is unavailable")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "models": [
                    {
                        "id": "missing-profile",
                        "training_steps": 0,
                        "path": str(source),
                        "sha256": evaluator._sha256(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observation profile"):
        evaluator._load_models_manifest(manifest)


def test_base_and_motor_models_validate_on_matching_profiles(
    tmp_path: Path,
) -> None:
    from sb3_contrib import MaskablePPO

    original_root = _real_root()
    if not original_root.exists():
        pytest.skip("retail extraction is unavailable")
    project_root = trainer.SCRIPT_PATH.parents[1]
    route = json.loads(
        (
            project_root
            / "diagnostics/alphazuma-55-motor-observable-replay-"
            "s99081614-preregistration-v2.json"
        ).read_text(encoding="utf-8-sig")
    )
    source_path = Path(route["source"]["model_path"])
    config = dict(route["run"])
    config["device"] = "cpu"
    prototype = trainer._prototype(
        original_root=original_root,
        max_ticks=int(config["max_ticks"]),
    )
    motor_path = tmp_path / "motor.zip"
    try:
        source = MaskablePPO.load(source_path, device="cpu")
        motor = trainer._build_motor_model(prototype, config)
        transplant_appended_global_policy(
            source_policy=source.policy,
            target_policy=motor.policy,
            source_global_feature_size=int(
                prototype.revenge_env.global_feature_size
            ),
            target_global_feature_size=int(prototype.global_feature_size),
        )
        motor.save(motor_path)
    finally:
        prototype.close()

    prereg = json.loads(
        (
            project_root
            / "diagnostics/alphazuma-55-polar-intent-wide-validation-"
            "s99081544-preregistration-v1.json"
        ).read_text(encoding="utf-8-sig")
    )
    prereg["levels"] = prereg["levels"][:1]
    result = evaluator._validate_model_spaces(
        prereg=prereg,
        original_root=original_root,
        model_specs=[
            {
                "id": "base",
                "path": str(source_path),
                "observation_profile": evaluator.BASE_OBSERVATION_PROFILE,
            },
            {
                "id": "motor",
                "path": str(motor_path),
                "observation_profile": MOTOR_OBSERVATION_PROFILE,
            },
        ],
    )
    assert result["models_verified"] == 2
    assert [
        row["observation_profile"]
        for row in result["model_space_validation"]
    ] == [evaluator.BASE_OBSERVATION_PROFILE, MOTOR_OBSERVATION_PROFILE]
