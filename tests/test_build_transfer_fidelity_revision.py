"""Regression tests for the explicit transfer Fidelity revision registry."""

from __future__ import annotations

from tools import build_transfer_fidelity_revision as revision


_CURRENT_EXTENSION_ENTRIES = {
    "a16-audit-c158-natural-loss": {
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c158-natural-loss-live-v1.json",
        "features": ["pc_mechanism_coverage"],
    },
    "a17-audit-c164-fruit-collection": {
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c164-fruit-collection-live-v1.json",
        "features": ["pc_mechanism_coverage"],
    },
    "a18-audit-c185-fruit-powerup-collision": {
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c185-fruit-powerup-collision-live-v2.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    "p8-pc-source-c158-natural-loss": {
        "kind": "pc_source",
        "path": (
            "sources/"
            "c158-u11200-u11830-source-bound-natural-loss-v1.json"
        ),
        "features": ["natural_loss", "source_authenticity"],
    },
    "p9-pc-source-c164-fruit-collection": {
        "kind": "pc_source",
        "path": (
            "sources/"
            "c164-u5220-u5310-source-bound-fruit-collection-v1.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_collection_score",
            "fruit_projectile_collision",
            "source_authenticity",
        ],
    },
    "p10-pc-source-c183-fruit-powerup-collision": {
        "kind": "pc_source",
        "path": (
            "sources/"
            "c183-u8120-u8270-source-bound-fruit-powerup-collision-v1.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_powerup_collision",
            "powerup_proximity_bomb",
            "source_authenticity",
        ],
    },
    "q12-sim-c158-natural-loss": {
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c158-u11341-u11519-natural-loss-"
            "diff-v4-source-bound.json"
        ),
        "features": ["natural_loss"],
    },
    "q13-sim-c164-fruit-collection": {
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c164-u5278-u5305-fruit-collection-"
            "diff-v3-reconciled-bound.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_collection_score",
            "fruit_projectile_collision",
        ],
    },
    "q14-sim-c183-fruit-powerup-collision": {
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c183-u8213-u8260-fruit-powerup-collision-"
            "diff-v6-reconciled-bound.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_powerup_collision",
            "powerup_proximity_bomb",
        ],
    },
}

_C200_REFRESH_PATHS = {
    "l-audit-actor-hidden-state-current": (
        "diagnostics/actor-audit-jungle2-current-v12.json"
    ),
    "a7-audit-c35-natural-shot": (
        "diagnostics/pc-mechanism-c35-natural-shot-v2-current.json"
    ),
    "a11-audit-c126-natural-win": (
        "diagnostics/pc-mechanism-c126-natural-win-live-v6-current.json"
    ),
    "a13-audit-c131b1-reverse": (
        "diagnostics/pc-mechanism-c131b1-reverse-live-v4-current.json"
    ),
    "q7-sim-c126-powerup-proximity-bomb": (
        "diagnostics/jungle2-c126-u7222-u7248-powerup-proximity-bomb-"
        "diff-v14-nosync-bound-current.json"
    ),
    "q8-sim-c131b1-powerup-reverse": (
        "diagnostics/jungle2-c131b1-u5060-u5098-powerup-reverse-"
        "diff-v9-nosync-bound-current.json"
    ),
    "q12-sim-c158-natural-loss": (
        "diagnostics/jungle2-c158-u11341-u11519-natural-loss-"
        "diff-v4-source-bound.json"
    ),
}


def test_current_registry_extensions_match_frozen_contracts() -> None:
    entries = {entry["id"]: entry for entry in revision.CURRENT_ENTRIES}

    assert len(entries) == len(revision.CURRENT_ENTRIES)
    for evidence_id, expected in _CURRENT_EXTENSION_ENTRIES.items():
        assert entries[evidence_id] == {"id": evidence_id, **expected}


def test_c200_refreshes_only_the_registered_evidence_paths() -> None:
    entries = {entry["id"]: entry for entry in revision.CURRENT_ENTRIES}

    for evidence_id, path in _C200_REFRESH_PATHS.items():
        assert entries[evidence_id]["path"] == path


def test_revision_registry_features_are_unique_and_sorted() -> None:
    for entry in revision.CURRENT_ENTRIES:
        assert entry["features"] == sorted(set(entry["features"]))


def test_c200_does_not_change_transfer_policy_generation() -> None:
    assert revision.POLICY_ID == "original-transfer-jungle2-v4"
    assert revision.BASE_POLICY_IDS == {
        "original-transfer-jungle2-v3",
        "original-transfer-jungle2-v4",
    }
