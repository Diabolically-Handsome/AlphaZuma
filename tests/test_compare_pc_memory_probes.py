from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tools.compare_pc_memory_probes as comparator
from tools.compare_pc_memory_probes import (
    TransitionValidationError,
    validate_transition,
)


_RUNTIME_SHA256 = "sha256:" + "12" * 32
_DMO_SHA256 = "sha256:" + "34" * 32


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _ball(ball_id: int, color_id: int, distance: float) -> dict[str, Any]:
    return {
        "ball": {
            "ball_id": ball_id,
            "color_id": color_id,
            "curve_distance": distance,
            "position_x": 100.0 + ball_id,
            "position_y": 200.0,
        }
    }


def _bullet(
    pointer_offset: int,
    ball_id: int,
    color_id: int,
) -> dict[str, Any]:
    value = _ball(ball_id, color_id, 0.0)
    value["shooter_pointer_offset"] = pointer_offset
    value["subclass_fields"] = {
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "fired": False,
    }
    return value


def _probe(*, after: bool) -> dict[str, Any]:
    chain = [
        _ball(1, 3, 11.5 if after else 10.0),
        _ball(2, 4, 21.5 if after else 20.0),
    ]
    if after:
        current = _bullet(0x130, 99, 2)
        next_bullet = _bullet(0x134, 110, 3)
        fired = _ball(104, 0, 0.0)
        fired["ball"]["position_x"] = 356.0
        fired["ball"]["position_y"] = 232.0
        fired["subclass_fields"] = {
            "velocity_x": -2.75,
            "velocity_y": -7.5,
            "fired": False,
        }
        fired_records = [fired]
    else:
        current = _bullet(0x130, 104, 0)
        current["ball"]["position_x"] = 405.0
        current["ball"]["position_y"] = 319.0
        next_bullet = _bullet(0x134, 99, 2)
        fired_records = []
    return {
        "schema": "zuma-rl.pc-memory-int32-probe",
        "framework_update": 7750 if after else 7738,
        "runtime_executable_sha256": _RUNTIME_SHA256,
        "dmo_sha256": _DMO_SHA256,
        "active_board": {
            "score": 7950,
            "displayed_score": 7950,
            "score_target": 9650,
            "curve_manager": {
                "curves": [
                    {
                        "intrusive_lists": [
                            {
                                "container_offset": 0x5C,
                                "payload_count": len(chain),
                                "records": chain,
                            }
                        ]
                    }
                ]
            },
            "primary_child": {
                "bullets": [current, next_bullet],
            },
            "fired_bullets": {
                "board_container_offset": 0x698,
                "declared_count": len(fired_records),
                "traversed_count": len(fired_records),
                "records": fired_records,
            },
        },
    }


def _pixel_report(
    probe_path: Path,
    *,
    fired: bool,
) -> dict[str, Any]:
    probe = json.loads(probe_path.read_text(encoding="ascii"))
    return {
        "schema": "zuma-rl.pc-ball-pixel-validation",
        "version": 3,
        "status": "PASS",
        "memory_probe": {
            "sha256": _digest(probe_path),
            "framework_update": probe["framework_update"],
        },
        "visible_count": 2,
        "match_count": 2,
        "mismatch_count": 0,
        "fired_bullets": {
            "visible_count": 1 if fired else 0,
            "match_count": 1 if fired else 0,
            "mismatch_count": 0,
            "results": (
                [{"ball_id": 104, "color_id": 0}]
                if fired
                else []
            ),
        },
        "raw_payload_validation": {
            "status": "PASS",
            "probe_sha256": _digest(probe_path),
        },
    }


@pytest.fixture(autouse=True)
def _stub_raw_payload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validate(
        path: Path,
        *,
        expected_runtime_sha256: str | None = None,
        expected_dmo_sha256: str | None = None,
        expected_update: int | None = None,
        expected_score: int | None = None,
        require_freeze_state: bool = False,
    ) -> dict[str, Any]:
        probe = json.loads(path.read_text(encoding="ascii"))
        board = probe["active_board"]
        chain = board["curve_manager"]["curves"][0][
            "intrusive_lists"
        ][0]["records"]
        bullets = {
            row["shooter_pointer_offset"]: row
            for row in board["primary_child"]["bullets"]
        }
        runtime = probe["runtime_executable_sha256"]
        dmo = probe["dmo_sha256"]
        assert expected_runtime_sha256 in {None, runtime}
        assert expected_dmo_sha256 in {None, dmo}
        assert expected_update in {None, probe["framework_update"]}
        assert expected_score in {None, board["score"]}
        assert require_freeze_state is True
        return {
            "runtime_executable_sha256": runtime,
            "dmo_sha256": dmo,
            "score": board["score"],
            "displayed_score": board["displayed_score"],
            "score_target": board["score_target"],
            "active_chain": chain,
            "shooter_current": bullets[0x130],
            "shooter_next": bullets[0x134],
            "fired_bullets": board["fired_bullets"]["records"],
        }

    monkeypatch.setattr(comparator, "validate_probe_payloads", validate)


def _case(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    before_probe = tmp_path / "before-probe.json"
    after_probe = tmp_path / "after-probe.json"
    before_pixel = tmp_path / "before-pixel.json"
    after_pixel = tmp_path / "after-pixel.json"
    _write_json(before_probe, _probe(after=False))
    _write_json(after_probe, _probe(after=True))
    _write_json(
        before_pixel,
        _pixel_report(before_probe, fired=False),
    )
    _write_json(
        after_pixel,
        _pixel_report(after_probe, fired=True),
    )
    return before_probe, after_probe, before_pixel, after_pixel


def _validate(paths: tuple[Path, Path, Path, Path]) -> dict[str, Any]:
    return validate_transition(
        *paths,
        expected_before_update=7738,
        expected_after_update=7750,
        expected_score=7950,
        expected_chain_distance_delta=1.5,
        distance_tolerance=1e-6,
        minimum_chain_count=2,
    )


def test_transition_proves_transfer_promotion_and_chain_motion(
    tmp_path: Path,
) -> None:
    result = _validate(_case(tmp_path))

    assert result["status"] == "PASS"
    assert result["active_chain"]["count"] == 2
    assert result["active_chain"]["mean_curve_distance_delta"] == 1.5
    assert result["shot_transition"]["before_current"] == {
        "ball_id": 104,
        "color_id": 0,
    }
    assert result["shot_transition"]["after_current"] == {
        "ball_id": 99,
        "color_id": 2,
    }
    assert result["shot_transition"]["transferred_fired_bullet"][
        "transient_flag_16a_after_transfer"
    ] is False


def test_transition_rejects_active_chain_color_change(
    tmp_path: Path,
) -> None:
    paths = _case(tmp_path)
    after = json.loads(paths[1].read_text(encoding="ascii"))
    after["active_board"]["curve_manager"]["curves"][0][
        "intrusive_lists"
    ][0]["records"][1]["ball"]["color_id"] = 2
    _write_json(paths[1], after)
    _write_json(paths[3], _pixel_report(paths[1], fired=True))

    with pytest.raises(
        TransitionValidationError,
        match="active_chain_identity_or_color_changed",
    ):
        _validate(paths)


def test_transition_rejects_wrong_transferred_bullet(
    tmp_path: Path,
) -> None:
    paths = _case(tmp_path)
    after = json.loads(paths[1].read_text(encoding="ascii"))
    after["active_board"]["fired_bullets"]["records"][0]["ball"][
        "ball_id"
    ] = 777
    _write_json(paths[1], after)
    _write_json(paths[3], _pixel_report(paths[1], fired=True))

    with pytest.raises(
        TransitionValidationError,
        match="shooter_current_was_not_transferred",
    ):
        _validate(paths)


def test_transition_rejects_unbound_pixel_report(
    tmp_path: Path,
) -> None:
    paths = _case(tmp_path)
    report = json.loads(paths[3].read_text(encoding="ascii"))
    report["memory_probe"]["sha256"] = "sha256:" + "00" * 32
    _write_json(paths[3], report)

    with pytest.raises(
        TransitionValidationError,
        match="pixel_validation_probe_hash_mismatch",
    ):
        _validate(paths)
