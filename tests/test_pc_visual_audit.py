"""Tests for the fail-closed retail visual-derivability audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from zuma_rl import pc_visual_audit
from zuma_rl.pc_golden import ComparisonStatus


def _verification() -> SimpleNamespace:
    return SimpleNamespace(
        status=ComparisonStatus.PASS,
        summary={
            "semantic_limits": {
                "deterministic_replay_attested": True,
                "memory_measurement_provenance_attested": True,
            },
            "replay_determinism": {
                "machine_verifiable_evidence": True,
                "run_count": 2,
                "native_tick_count": 78,
                "full_viewport_rgb24_per_tick_matched": True,
                "normalized_trace_matched": True,
                "dmo_binding_matched": True,
            },
            "measurement_provenance": {
                "machine_verifiable_evidence": True,
                "gameplay_viewport_pixels_matched": True,
                "active_visible_count": 91,
                "active_pixel_match_count": 91,
                "active_color_ids": [0, 1, 2, 3],
                "fired_visible_count": 1,
                "fired_pixel_match_count": 1,
                "fired_color_ids": [0],
                "fired_ball_id": 104,
                "fired_color_id": 0,
                "fired_speed": 8.0,
                "curve_geometry_binding": {
                    "resolved_against_original_data": True,
                    "phases": [{"phase": "after"}],
                },
            },
        },
    )


def _actor_report(feature_contract: list[dict[str, str]]) -> dict:
    return {
        "status": "PASS",
        "audit_fingerprint": "sha256:" + "1" * 64,
        "implementation_fingerprint": "sha256:" + "2" * 64,
        "environment_fingerprint": "sha256:" + "3" * 64,
        "feature_contract": feature_contract,
    }


def _install_case_stubs(monkeypatch) -> None:
    monkeypatch.setattr(
        pc_visual_audit.PcGoldenManifest,
        "read_json",
        lambda path: SimpleNamespace(case_id="case-1"),
    )
    monkeypatch.setattr(
        pc_visual_audit,
        "pc_golden_native_source_fingerprint",
        lambda manifest: "sha256:" + "4" * 64,
    )
    monkeypatch.setattr(
        pc_visual_audit,
        "verify_pc_golden_case",
        lambda *args, **kwargs: _verification(),
    )


def test_pc_capabilities_require_pixel_counts_colors_and_geometry() -> None:
    capabilities = pc_visual_audit._verification_capabilities(
        _verification()
    )

    assert capabilities == frozenset(
        {
            "active_ball_color_0",
            "active_ball_color_1",
            "active_ball_color_2",
            "active_ball_color_3",
            "active_ball_presence_position",
            "curve_geometry_pc_binding",
            "full_viewport_tick_history",
            "projectile_color_0",
            "projectile_presence_position",
        }
    )


def test_missing_powerup_visual_keeps_audit_incomparable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "case" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    _install_case_stubs(monkeypatch)
    monkeypatch.setattr(
        pc_visual_audit,
        "audit_actor_observation",
        lambda **kwargs: _actor_report(
            [
                {
                    "section": "balls",
                    "feature": "x",
                    "visibility": "direct_visible",
                },
                {
                    "section": "balls",
                    "feature": "powerup_3",
                    "visibility": "direct_visible",
                },
                {
                    "section": "globals",
                    "feature": "tick_progress",
                    "visibility": "agent_clock",
                },
            ]
        ),
    )

    report = pc_visual_audit.audit_actor_visual_derivability(
        [manifest],
        evidence_root=tmp_path,
        original_root=tmp_path,
    )

    assert report["status"] == "INCOMPARABLE"
    assert report["summary"]["feature_count"] == 3
    assert report["summary"]["derivable_feature_count"] == 2
    assert report["summary"]["missing_feature_count"] == 1
    assert report["summary"]["missing_capabilities"] == [
        "active_ball_powerup_3"
    ]
    assert report["cases"][0]["manifest_path"] == "case/manifest.json"
    assert report["audit_fingerprint"].startswith("sha256:")


def test_pass_report_is_recomputed_and_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "case" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    _install_case_stubs(monkeypatch)
    monkeypatch.setattr(
        pc_visual_audit,
        "audit_actor_observation",
        lambda **kwargs: _actor_report(
            [
                {
                    "section": "balls",
                    "feature": "x",
                    "visibility": "direct_visible",
                },
                {
                    "section": "globals",
                    "feature": "outcome",
                    "visibility": "terminal_api",
                },
            ]
        ),
    )
    report = pc_visual_audit.audit_actor_visual_derivability(
        [manifest],
        evidence_root=tmp_path,
        original_root=tmp_path,
    )
    assert report["status"] == "PASS"

    reason = pc_visual_audit.validate_actor_visual_audit_report(
        report,
        evidence_root=tmp_path,
        original_root=tmp_path,
        level_id="Jungle2",
        hard=False,
        profile_mode="tutorials_completed",
    )

    assert reason is None
    report["audit_fingerprint"] = "sha256:" + "9" * 64
    assert "fingerprint differs" in (
        pc_visual_audit.validate_actor_visual_audit_report(
            report,
            evidence_root=tmp_path,
            original_root=tmp_path,
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
        )
        or ""
    )


def test_manifest_path_must_stay_inside_evidence_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-manifest.json"
    outside.write_text("{}", encoding="utf-8")

    try:
        pc_visual_audit.audit_actor_visual_derivability(
            [outside],
            evidence_root=tmp_path,
            original_root=tmp_path,
        )
    except ValueError as error:
        assert "escapes evidence_root" in str(error)
    else:
        raise AssertionError("path escape unexpectedly accepted")
