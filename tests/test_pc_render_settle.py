from __future__ import annotations

import json

import pytest

from zuma_rl.pc_golden import PcGoldenValidationError
from zuma_rl.pc_render_settle import (
    PcRenderSettledUpdateMap,
    RenderSettledUpdateRecord,
)


def _digest(byte: str) -> str:
    return "sha256:" + byte * 64


def _map() -> PcRenderSettledUpdateMap:
    return PcRenderSettledUpdateMap(
        capture_metadata_sha256=_digest("1"),
        frames_csv_sha256=_digest("2"),
        framework_update_map_sha256=_digest("3"),
        framework_state_sidecar_sha256=_digest("4"),
        framework_poll_sha256=_digest("5"),
        calibration_preregistration_sha256=_digest("6"),
        calibration_execution_binding_sha256=_digest("7"),
        calibration_holdout_report_sha256=_digest("8"),
        process_id=100,
        process_creation_filetime_100ns=200,
        executable_sha256=_digest("9"),
        records=(
            RenderSettledUpdateRecord(
                sequence=0,
                present_ticks=10,
                present_perf_counter_ns=1_000_000,
                source_update_before=30,
                source_update_after=31,
                assigned_draw_count=40,
                assigned_framework_update=30,
                draw_event_sample_after_perf_counter_ns=500_000,
                settle_age_ns=500_000,
            ),
            RenderSettledUpdateRecord(
                sequence=1,
                present_ticks=20,
                present_perf_counter_ns=2_000_000,
                source_update_before=31,
                source_update_after=31,
                assigned_draw_count=41,
                assigned_framework_update=31,
                draw_event_sample_after_perf_counter_ns=1_500_000,
                settle_age_ns=500_000,
            ),
        ),
    )


def test_render_settled_map_round_trips_canonical_json() -> None:
    value = _map()

    parsed = PcRenderSettledUpdateMap.from_json(value.to_json())

    assert parsed == value
    assert parsed.records[0].assigned_framework_update == 30


def test_render_settled_map_rejects_posthoc_delay_change() -> None:
    payload = json.loads(_map().to_json())
    payload["render_settle_delay_ns"] = 399_999
    text = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"

    with pytest.raises(
        PcGoldenValidationError,
        match="differs from the calibrated contract",
    ):
        PcRenderSettledUpdateMap.from_json(text)


def test_render_settled_record_rejects_invented_settle_age() -> None:
    with pytest.raises(
        PcGoldenValidationError,
        match="age differs from its timestamps",
    ):
        RenderSettledUpdateRecord(
            sequence=0,
            present_ticks=10,
            present_perf_counter_ns=1_000_000,
            source_update_before=30,
            source_update_after=30,
            assigned_draw_count=40,
            assigned_framework_update=30,
            draw_event_sample_after_perf_counter_ns=500_000,
            settle_age_ns=499_999,
        )
