"""Tests for the headless actor-observation audit protocol."""

from __future__ import annotations

from copy import deepcopy

from zuma_rl import actor_audit


def test_visibility_contract_classifies_public_and_privileged_fields() -> None:
    assert (
        actor_audit._visibility_class("balls", "powerup_3")
        == "direct_visible"
    )
    assert (
        actor_audit._visibility_class("balls", "waypoint")
        == "geometry_derived"
    )
    assert (
        actor_audit._visibility_class("projectiles", "velocity_x")
        == "history_derived"
    )
    assert (
        actor_audit._visibility_class("globals", "pending_fraction")
        == "privileged_zero"
    )
    assert (
        actor_audit._visibility_class("globals", "tick_progress")
        == "agent_clock"
    )
    assert (
        actor_audit._visibility_class("globals", "fruit_present")
        == "direct_visible"
    )
    assert actor_audit._visibility_class("globals", "secret") is None


def test_audit_fingerprint_binds_the_complete_payload(
    monkeypatch,
) -> None:
    payload = {
        "schema": actor_audit.AUDIT_SCHEMA,
        "version": actor_audit.AUDIT_VERSION,
        "status": "PASS",
        "audit_type": actor_audit.AUDIT_TYPE,
        "scope": {"policy": actor_audit.POLICY_ID},
        "checks": [],
        "feature_contract": [],
        "failure_reasons": [],
        "summary": {},
    }
    monkeypatch.setattr(
        actor_audit,
        "_audit_payload",
        lambda **kwargs: deepcopy(payload),
    )

    first = actor_audit.audit_actor_observation()
    changed_payload = deepcopy(payload)
    changed_payload["summary"] = {"feature_count": 1}
    monkeypatch.setattr(
        actor_audit,
        "_audit_payload",
        lambda **kwargs: deepcopy(changed_payload),
    )
    second = actor_audit.audit_actor_observation()

    assert first["audit_fingerprint"].startswith("sha256:")
    assert first["audit_fingerprint"] != second["audit_fingerprint"]


def test_live_validation_rejects_stale_or_incomplete_report(
    monkeypatch,
) -> None:
    fingerprint = "sha256:" + "1" * 64
    live = {
        "status": "PASS",
        "audit_fingerprint": fingerprint,
    }
    monkeypatch.setattr(
        actor_audit,
        "audit_actor_observation",
        lambda **kwargs: live,
    )
    report = {
        "status": "PASS",
        "audit_type": actor_audit.AUDIT_TYPE,
        "audit_fingerprint": fingerprint,
        "checks": [{"name": "boundary", "status": "PASS"}],
        "summary": {
            "unmapped_feature_count": 0,
            "hidden_perturbation_actor_equal": True,
            "privileged_control_changed": True,
            "visible_powerup_channel_verified": True,
            "visible_fruit_channels_verified": True,
        },
    }

    assert (
        actor_audit.validate_actor_audit_report(
            report,
            root=None,
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
        )
        is None
    )

    stale = deepcopy(report)
    stale["audit_fingerprint"] = "sha256:" + "2" * 64
    assert "fingerprint differs" in actor_audit.validate_actor_audit_report(
        stale,
        root=None,
        level_id="Jungle2",
        hard=False,
        profile_mode="tutorials_completed",
    )

    incomplete = deepcopy(report)
    incomplete["summary"]["privileged_control_changed"] = False
    assert "does not prove" in actor_audit.validate_actor_audit_report(
        incomplete,
        root=None,
        level_id="Jungle2",
        hard=False,
        profile_mode="tutorials_completed",
    )
