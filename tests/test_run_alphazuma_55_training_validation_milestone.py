from pathlib import Path

import pytest

from tools import run_alphazuma_55_training_validation_milestone as controller


def _healthy_snapshot() -> dict:
    return {
        "status": "PASS",
        "training": {
            "route_count": 4,
            "failed_routes": 0,
            "running_routes": 4,
            "completed_routes": 0,
        },
        "controller": {"phase": "WAITING_FOR_TRAINING"},
        "formal_output_boundary": {
            "embargo_boundary_intact": True,
            "roots": {
                "selection": {"exists": False},
                "final_blind": {"exists": False},
                "continuous": {"exists": False},
            },
        },
        "hardware": {
            "memory": {
                "available_bytes": 64 * 1024**3,
                "swap_used_bytes": 0,
            },
            "gpus": [{"temperature_c": 45}, {"temperature_c": 53}],
        },
    }


def _plan() -> dict:
    return {
        "activation_gate": {
            "formal_controller_phase_must_equal": "WAITING_FOR_TRAINING",
            "wsl_available_memory_gib_at_least": 40,
            "wsl_swap_used_bytes_must_equal": 0,
            "gpu_temperature_c_below": 82,
        }
    }


def test_validate_activation_accepts_healthy_frozen_boundary() -> None:
    controller._validate_activation(_healthy_snapshot(), _plan())


def test_validate_activation_rejects_formal_output(tmp_path: Path) -> None:
    snapshot = _healthy_snapshot()
    snapshot["formal_output_boundary"]["roots"]["selection"]["exists"] = True

    with pytest.raises(ValueError, match="formal output embargo"):
        controller._validate_activation(snapshot, _plan())


def test_validate_activation_rejects_swap_use() -> None:
    snapshot = _healthy_snapshot()
    snapshot["hardware"]["memory"]["swap_used_bytes"] = 4096

    with pytest.raises(ValueError, match="swap"):
        controller._validate_activation(snapshot, _plan())


def test_expected_hash_rejects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        controller._expected_hash(path, "sha256:" + "0" * 64, "artifact")
