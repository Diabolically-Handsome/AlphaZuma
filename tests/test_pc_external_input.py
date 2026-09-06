from __future__ import annotations

from copy import deepcopy

import pytest

from zuma_rl.pc_external_input import (
    EXTERNAL_INPUT_GUARD_SCHEMA,
    EXTERNAL_INPUT_GUARD_VERSION,
    EXTERNAL_INPUT_POLICY,
    EXTERNAL_INPUT_PRIVACY,
    PcExternalInputError,
    REPAINT_INPUT_EXTRA_INFO,
    SELF_TEST_INPUT_EXTRA_INFO,
    canonical_external_input_bytes,
    validate_external_input_guard_receipt,
)


def _receipt() -> dict[str, object]:
    return {
        "schema": EXTERNAL_INPUT_GUARD_SCHEMA,
        "version": EXTERNAL_INPUT_GUARD_VERSION,
        "status": "PASS",
        "policy": EXTERNAL_INPUT_POLICY,
        "privacy": EXTERNAL_INPUT_PRIVACY,
        "coverage": {
            "start_perf_counter_ns": 100,
            "end_perf_counter_ns": 200,
        },
        "hooks": {
            "keyboard_installed": True,
            "mouse_installed": True,
            "keyboard_unhooked": True,
            "mouse_unhooked": True,
        },
        "self_tests": [
            {
                "label": label,
                "status": "PASS",
                "keyboard_event_count": 2,
                "mouse_event_count": 2,
            }
            for label in ("start", "end")
        ],
        "markers": {
            "repaint_input_extra_info_hex": (
                f"0x{REPAINT_INPUT_EXTRA_INFO:08x}"
            ),
            "self_test_input_extra_info_hex": (
                f"0x{SELF_TEST_INPUT_EXTRA_INFO:08x}"
            ),
        },
        "event_counts": {
            "external": 0,
            "allowed_repaint": 24,
            "self_test": 8,
        },
        "external_events": [],
        "protocol_errors": [],
    }


def test_external_input_receipt_passes_and_is_canonical_lf() -> None:
    receipt = _receipt()

    assert validate_external_input_guard_receipt(receipt) == receipt
    payload = canonical_external_input_bytes(receipt)
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload


def test_external_input_receipt_rejects_false_pass() -> None:
    receipt = _receipt()
    counts = receipt["event_counts"]
    assert isinstance(counts, dict)
    counts["external"] = 1
    receipt["external_events"] = [
        {
            "perf_counter_ns": 150,
            "device": "keyboard",
            "action": "down",
            "injected": False,
        }
    ]

    with pytest.raises(PcExternalInputError, match="false_pass"):
        validate_external_input_guard_receipt(receipt)


def test_external_input_receipt_accepts_failed_audit_but_not_formal() -> None:
    receipt = deepcopy(_receipt())
    receipt["status"] = "FAIL"
    counts = receipt["event_counts"]
    assert isinstance(counts, dict)
    counts["external"] = 1
    receipt["external_events"] = [
        {
            "perf_counter_ns": 150,
            "device": "mouse",
            "action": "move",
            "injected": False,
        }
    ]

    assert validate_external_input_guard_receipt(
        receipt, require_pass=False
    ) == receipt
    with pytest.raises(PcExternalInputError, match="guard_not_pass"):
        validate_external_input_guard_receipt(receipt)
