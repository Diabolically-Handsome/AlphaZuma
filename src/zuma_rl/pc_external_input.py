"""Formal evidence contract for global external-input contamination guards."""

from __future__ import annotations

import json
from typing import Any, Mapping


EXTERNAL_INPUT_GUARD_SCHEMA = "zuma-rl.pc-external-input-guard"
EXTERNAL_INPUT_GUARD_VERSION = 1
EXTERNAL_INPUT_GUARD_BINDING_SCHEMA = (
    "zuma-rl.pc-external-input-guard-binding"
)
EXTERNAL_INPUT_GUARD_BINDING_VERSION = 1
EXTERNAL_INPUT_POLICY = (
    "reject_all_unrecognized_global_keyboard_or_mouse_events"
)
EXTERNAL_INPUT_PRIVACY = (
    "no_key_identity_text_or_pointer_coordinates_recorded"
)
REPAINT_INPUT_EXTRA_INFO = 0x5A524C50
SELF_TEST_INPUT_EXTRA_INFO = 0x5A475244


class PcExternalInputError(ValueError):
    """The input-contamination receipt is malformed or does not pass."""


def canonical_external_input_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PcExternalInputError(f"{name}_invalid")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PcExternalInputError(f"{name}_invalid")
    return value


def validate_external_input_guard_receipt(
    value: Any,
    *,
    require_pass: bool = True,
) -> Mapping[str, Any]:
    """Strictly validate a privacy-preserving Windows input receipt."""

    receipt = _mapping(value, "external_input_guard")
    required = {
        "schema",
        "version",
        "status",
        "policy",
        "privacy",
        "coverage",
        "hooks",
        "self_tests",
        "markers",
        "event_counts",
        "external_events",
        "protocol_errors",
    }
    if set(receipt) != required:
        raise PcExternalInputError("external_input_guard_fields_invalid")
    if (
        receipt.get("schema") != EXTERNAL_INPUT_GUARD_SCHEMA
        or receipt.get("version") != EXTERNAL_INPUT_GUARD_VERSION
        or receipt.get("status") not in {"PASS", "FAIL"}
        or receipt.get("policy") != EXTERNAL_INPUT_POLICY
        or receipt.get("privacy") != EXTERNAL_INPUT_PRIVACY
    ):
        raise PcExternalInputError("external_input_guard_header_invalid")

    coverage = _mapping(receipt.get("coverage"), "external_input_coverage")
    if set(coverage) != {
        "start_perf_counter_ns",
        "end_perf_counter_ns",
    }:
        raise PcExternalInputError("external_input_coverage_fields_invalid")
    start = _integer(
        coverage.get("start_perf_counter_ns"),
        "external_input_coverage_start",
        minimum=1,
    )
    end = _integer(
        coverage.get("end_perf_counter_ns"),
        "external_input_coverage_end",
        minimum=1,
    )
    if start >= end:
        raise PcExternalInputError("external_input_coverage_order_invalid")

    hooks = _mapping(receipt.get("hooks"), "external_input_hooks")
    expected_hook_fields = {
        "keyboard_installed",
        "mouse_installed",
        "keyboard_unhooked",
        "mouse_unhooked",
    }
    if set(hooks) != expected_hook_fields or any(
        not isinstance(hooks.get(field), bool) for field in expected_hook_fields
    ):
        raise PcExternalInputError("external_input_hooks_invalid")

    markers = _mapping(receipt.get("markers"), "external_input_markers")
    if markers != {
        "repaint_input_extra_info_hex": (
            f"0x{REPAINT_INPUT_EXTRA_INFO:08x}"
        ),
        "self_test_input_extra_info_hex": (
            f"0x{SELF_TEST_INPUT_EXTRA_INFO:08x}"
        ),
    }:
        raise PcExternalInputError("external_input_markers_invalid")

    self_tests = receipt.get("self_tests")
    if not isinstance(self_tests, list) or len(self_tests) != 2:
        raise PcExternalInputError("external_input_self_tests_invalid")
    for expected_label, value in zip(("start", "end"), self_tests, strict=True):
        row = _mapping(value, "external_input_self_test")
        if set(row) != {
            "label",
            "status",
            "keyboard_event_count",
            "mouse_event_count",
        }:
            raise PcExternalInputError("external_input_self_test_fields_invalid")
        keyboard_count = _integer(
            row.get("keyboard_event_count"),
            "external_input_self_test_keyboard_count",
        )
        mouse_count = _integer(
            row.get("mouse_event_count"),
            "external_input_self_test_mouse_count",
        )
        if (
            row.get("label") != expected_label
            or row.get("status") not in {"PASS", "FAIL"}
            or (
                row.get("status") == "PASS"
                and (keyboard_count < 2 or mouse_count < 2)
            )
        ):
            raise PcExternalInputError("external_input_self_test_invalid")

    counts = _mapping(receipt.get("event_counts"), "external_input_counts")
    if set(counts) != {
        "external",
        "allowed_repaint",
        "self_test",
    }:
        raise PcExternalInputError("external_input_counts_fields_invalid")
    external_count = _integer(
        counts.get("external"), "external_input_external_count"
    )
    _integer(
        counts.get("allowed_repaint"),
        "external_input_repaint_count",
    )
    self_test_count = _integer(
        counts.get("self_test"),
        "external_input_self_test_count",
    )
    events = receipt.get("external_events")
    if not isinstance(events, list) or external_count < len(events):
        raise PcExternalInputError("external_input_events_invalid")
    for value in events:
        event = _mapping(value, "external_input_event")
        if set(event) != {
            "perf_counter_ns",
            "device",
            "action",
            "injected",
        }:
            raise PcExternalInputError("external_input_event_fields_invalid")
        _integer(
            event.get("perf_counter_ns"),
            "external_input_event_time",
            minimum=start,
        )
        if (
            event.get("device") not in {"keyboard", "mouse"}
            or not isinstance(event.get("action"), str)
            or not event["action"]
            or not isinstance(event.get("injected"), bool)
        ):
            raise PcExternalInputError("external_input_event_invalid")
    errors = receipt.get("protocol_errors")
    if not isinstance(errors, list) or any(
        not isinstance(error, str) or not error for error in errors
    ):
        raise PcExternalInputError("external_input_protocol_errors_invalid")

    passed = (
        receipt.get("status") == "PASS"
        and all(hooks.values())
        and all(row.get("status") == "PASS" for row in self_tests)
        and external_count == 0
        and events == []
        and self_test_count >= 8
        and errors == []
    )
    if receipt.get("status") == "PASS" and not passed:
        raise PcExternalInputError("external_input_false_pass")
    if require_pass and not passed:
        raise PcExternalInputError("external_input_guard_not_pass")
    return receipt


__all__ = [
    "EXTERNAL_INPUT_GUARD_BINDING_SCHEMA",
    "EXTERNAL_INPUT_GUARD_BINDING_VERSION",
    "EXTERNAL_INPUT_GUARD_SCHEMA",
    "EXTERNAL_INPUT_GUARD_VERSION",
    "EXTERNAL_INPUT_POLICY",
    "EXTERNAL_INPUT_PRIVACY",
    "PcExternalInputError",
    "REPAINT_INPUT_EXTRA_INFO",
    "SELF_TEST_INPUT_EXTRA_INFO",
    "canonical_external_input_bytes",
    "validate_external_input_guard_receipt",
]
