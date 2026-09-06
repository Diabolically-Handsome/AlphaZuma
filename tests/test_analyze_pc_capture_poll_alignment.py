from __future__ import annotations

from difflib import SequenceMatcher

from tools.analyze_pc_capture_draw_alignment import FrameRecord
from tools.analyze_pc_capture_poll_alignment import (
    CollapsedVisual,
    analyze_delay,
)


def _frame(sequence: int, update: int) -> FrameRecord:
    return FrameRecord(
        sequence=sequence,
        present_ticks=100 + sequence,
        host_perf_counter_ns=1000 + sequence,
        frame_sha256="sha256:" + f"{sequence:02x}" * 32,
        update_before=update,
        update_after=update,
        draw_before=0,
        draw_after=0,
    )


def test_delay_analysis_recovers_modal_draw_offset_and_unresolved_pair() -> None:
    left_digests = ("a", "b", "c", "x")
    right_digests = ("a", "b", "c", "y")
    left_collapsed = tuple(
        CollapsedVisual(value, index, index)
        for index, value in enumerate(left_digests)
    )
    right_collapsed = tuple(
        CollapsedVisual(value, index, index)
        for index, value in enumerate(right_digests)
    )
    matcher = SequenceMatcher(
        None, left_digests, right_digests, autojunk=False
    )

    row = analyze_delay(
        delay_ns=250_000,
        left_records=tuple(_frame(index, 10 + index) for index in range(4)),
        right_records=tuple(_frame(index, 10 + index) for index in range(4)),
        left_digests=left_digests,
        right_digests=right_digests,
        left_draws=(20, 21, 22, 23),
        right_draws=(15, 16, 17, 18),
        matcher=matcher,
        left_collapsed=left_collapsed,
        right_collapsed=right_collapsed,
    )

    assert row["modal_draw_delta"] == 5
    assert row["modal_draw_delta_support"] == 3
    assert row["paired_draw_count"] == 4
    assert row["exact_hash_overlap_count"] == 3
    assert row["unresolved_pair_count"] == 1
    assert row["unresolved_pairs"][0]["left_draw_count"] == 23
