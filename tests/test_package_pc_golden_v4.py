from __future__ import annotations

from pathlib import Path
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools.package_pc_golden_v4 import (
    FrameCandidate,
    MatchingTick,
    PackagingError,
    _matching_ticks,
    _select_packaging_window,
    _verify_memory_video_binding,
    _verify_selected_raw_frames,
    _window_inputs,
)


def _edge_candidate(
    *,
    sequence: int,
    update: int,
    comparison: str,
    full: str,
    changed_pixels: int,
) -> FrameCandidate:
    edge = np.zeros((8, 4), dtype=np.uint8)
    edge[:changed_pixels, 0] = 1
    return FrameCandidate(
        sequence=sequence,
        update=update,
        pts=sequence + 1,
        frame_sha256=full,
        comparison_sha256=comparison,
        excluded_edge_bgra=edge.tobytes(),
    )


def _matching_tick(update: int) -> MatchingTick:
    return MatchingTick(
        update=update,
        r1=FrameCandidate(
            sequence=update,
            update=update,
            pts=update,
            frame_sha256="sha256:" + "11" * 32,
        ),
        r2=FrameCandidate(
            sequence=update + 1000,
            update=update,
            pts=update + 1000,
            frame_sha256="sha256:" + "11" * 32,
        ),
    )


def test_packaging_window_skips_longer_partial_input_edge() -> None:
    demo = SimpleNamespace(
        input_commands=(
            SimpleNamespace(
                update=101,
                kind="mouse_button",
                payload={"down": True},
            ),
        )
    )
    selected, inputs = _select_packaging_window(
        (
            _matching_tick(100),
            _matching_tick(101),
            _matching_tick(200),
        ),
        demo,
        minimum_ticks=1,
        allow_idle_window=True,
    )

    assert [row.update for row in selected] == [200]
    assert inputs == ()


def test_packaging_window_honors_preregistered_update_range() -> None:
    demo = SimpleNamespace(input_commands=())
    selected, inputs = _select_packaging_window(
        tuple(
            _matching_tick(update)
            for update in (100, 101, 102, 103, 104, 200, 201, 202)
        ),
        demo,
        minimum_ticks=2,
        allow_idle_window=True,
        required_update_range=(201, 202),
    )

    assert [row.update for row in selected] == [200, 201, 202]
    assert inputs == ()


def test_packaging_window_can_select_exact_preregistered_range() -> None:
    demo = SimpleNamespace(input_commands=())
    selected, inputs = _select_packaging_window(
        tuple(_matching_tick(update) for update in range(100, 110)),
        demo,
        minimum_ticks=3,
        allow_idle_window=True,
        required_update_range=(103, 105),
        select_exact_required_range=True,
    )

    assert [row.update for row in selected] == [103, 104, 105]
    assert inputs == ()


def test_packaging_window_rejects_missing_preregistered_range() -> None:
    demo = SimpleNamespace(input_commands=())

    with pytest.raises(
        PackagingError,
        match="preregistered update range",
    ):
        _select_packaging_window(
            tuple(_matching_tick(update) for update in range(100, 105)),
            demo,
            minimum_ticks=2,
            allow_idle_window=True,
            required_update_range=(200, 201),
        )


def test_matching_ticks_accepts_only_the_bounded_final_raster_row() -> None:
    comparison = "sha256:" + "33" * 32
    left = {
        100: (
            _edge_candidate(
                sequence=1,
                update=100,
                comparison=comparison,
                full="sha256:" + "11" * 32,
                changed_pixels=0,
            ),
        )
    }
    right = {
        100: (
            _edge_candidate(
                sequence=2,
                update=100,
                comparison=comparison,
                full="sha256:" + "22" * 32,
                changed_pixels=5,
            ),
        )
    }

    accepted = _matching_ticks(
        left,
        right,
        maximum_excluded_edge_mismatches=5,
    )

    assert len(accepted) == 1
    assert accepted[0].excluded_edge_mismatch_count == 5
    assert (
        _matching_ticks(
            left,
            right,
            maximum_excluded_edge_mismatches=4,
        )
        == ()
    )


def test_selected_raw_frames_keep_the_complete_interior_exact(
    tmp_path: Path,
) -> None:
    left = np.zeros((45, 140, 4), dtype=np.uint8)
    left[:, :, 3] = 255
    right = left.copy()
    right[-1, :5, 0] = 17
    candidates = []
    for run_id, frame in (("r1", left), ("r2", right)):
        capture = tmp_path / f"run-{run_id}" / "capture"
        capture.mkdir(parents=True)
        (capture / "frames.bgra.raw").write_bytes(frame.tobytes())
        candidates.append(
            FrameCandidate(
                sequence=0,
                update=100,
                pts=1,
                frame_sha256=(
                    "sha256:" + hashlib.sha256(frame.tobytes()).hexdigest()
                ),
            )
        )
    selected = (
        MatchingTick(
            update=100,
            r1=candidates[0],
            r2=candidates[1],
            excluded_edge_mismatch_count=5,
        ),
    )

    crop_hash = _verify_selected_raw_frames(
        tmp_path,
        selected,
        width=140,
        height=45,
        excluded_bottom_rows=1,
        maximum_excluded_edge_mismatches=5,
    )

    assert crop_hash.startswith("sha256:")
    right[20, 135, 0] = 9
    (tmp_path / "run-r2" / "capture" / "frames.bgra.raw").write_bytes(
        right.tobytes()
    )
    changed = FrameCandidate(
        sequence=0,
        update=100,
        pts=1,
        frame_sha256=(
            "sha256:" + hashlib.sha256(right.tobytes()).hexdigest()
        ),
    )
    with pytest.raises(PackagingError, match="inside gameplay viewport"):
        _verify_selected_raw_frames(
            tmp_path,
            (
                MatchingTick(
                    update=100,
                    r1=candidates[0],
                    r2=changed,
                    excluded_edge_mismatch_count=5,
                ),
            ),
            width=140,
            height=45,
            excluded_bottom_rows=1,
            maximum_excluded_edge_mismatches=5,
        )


def _binding_fixture(
    tmp_path: Path,
    *,
    interior_mismatch: bool = False,
) -> tuple[tuple[MatchingTick, ...], Path, Path]:
    capture = tmp_path / "run-r1" / "capture"
    capture.mkdir(parents=True)

    rgb = np.zeros((2, 600, 800, 3), dtype=np.uint8)
    rgb[0, :, :, :] = (11, 23, 47)
    rgb[1, :, :, :] = (53, 71, 89)
    bgra = np.empty((2, 600, 800, 4), dtype=np.uint8)
    bgra[:, :, :, 0] = rgb[:, :, :, 2]
    bgra[:, :, :, 1] = rgb[:, :, :, 1]
    bgra[:, :, :, 2] = rgb[:, :, :, 0]
    bgra[:, :, :, 3] = 255
    (capture / "frames.bgra.raw").write_bytes(bgra.tobytes())

    before_rgb = rgb[0].copy()
    before_rgb[-1, :4, :] = 255
    if interior_mismatch:
        before_rgb[100, 100, :] = 255
    before = tmp_path / "before.bmp"
    after = tmp_path / "after.bmp"
    Image.fromarray(before_rgb, "RGB").save(before, format="BMP")
    Image.fromarray(rgb[1], "RGB").save(after, format="BMP")

    selected = (
        MatchingTick(
            update=100,
            r1=FrameCandidate(
                sequence=0,
                update=100,
                pts=7,
                frame_sha256="sha256:" + "11" * 32,
            ),
            r2=FrameCandidate(
                sequence=4,
                update=100,
                pts=9,
                frame_sha256="sha256:" + "11" * 32,
            ),
        ),
        MatchingTick(
            update=101,
            r1=FrameCandidate(
                sequence=1,
                update=101,
                pts=13,
                frame_sha256="sha256:" + "22" * 32,
            ),
            r2=FrameCandidate(
                sequence=5,
                update=101,
                pts=15,
                frame_sha256="sha256:" + "22" * 32,
            ),
        ),
    )
    return selected, before, after


def test_memory_video_binding_accepts_only_declared_bottom_edge_budget(
    tmp_path: Path,
) -> None:
    selected, before, after = _binding_fixture(tmp_path)

    rows = _verify_memory_video_binding(
        tmp_path,
        selected,
        before_update=100,
        before_frame=before,
        after_update=101,
        after_frame=after,
        excluded_bottom_rows=1,
        maximum_excluded_edge_mismatches=4,
    )

    assert [row["phase"] for row in rows] == ["before", "after"]
    assert rows[0]["interior_mismatch_count"] == 0
    assert rows[0]["excluded_edge_mismatch_count"] == 4
    assert rows[1]["excluded_edge_mismatch_count"] == 0
    assert rows[1]["native_tick"] == 1


def test_memory_video_binding_rejects_gameplay_viewport_difference(
    tmp_path: Path,
) -> None:
    selected, before, after = _binding_fixture(
        tmp_path,
        interior_mismatch=True,
    )

    with pytest.raises(
        PackagingError,
        match="differs inside gameplay viewport",
    ):
        _verify_memory_video_binding(
            tmp_path,
            selected,
            before_update=100,
            before_frame=before,
            after_update=101,
            after_frame=after,
            excluded_bottom_rows=1,
            maximum_excluded_edge_mismatches=4,
        )


def test_selected_raw_frame_verification_derives_capture_length(
    tmp_path: Path,
) -> None:
    payload = np.zeros((501, 42, 130, 4), dtype=np.uint8)
    payload[500, 20, 79] = (1, 2, 3, 255)
    for run_id in ("r1", "r2"):
        capture = tmp_path / f"run-{run_id}" / "capture"
        capture.mkdir(parents=True)
        (capture / "frames.bgra.raw").write_bytes(payload.tobytes())
    frame = payload[500].tobytes()
    digest = "sha256:" + hashlib.sha256(frame).hexdigest()
    selected = (
        MatchingTick(
            update=100,
            r1=FrameCandidate(
                sequence=500,
                update=100,
                pts=1,
                frame_sha256=digest,
            ),
            r2=FrameCandidate(
                sequence=500,
                update=100,
                pts=1,
                frame_sha256=digest,
            ),
        ),
    )

    crop_sha256 = _verify_selected_raw_frames(
        tmp_path,
        selected,
        width=130,
        height=42,
    )

    assert crop_sha256.startswith("sha256:")


def _score_crop_selected(payload: np.ndarray) -> tuple[MatchingTick, ...]:
    selected: list[MatchingTick] = []
    for sequence, frame in enumerate(payload):
        digest = "sha256:" + hashlib.sha256(frame.tobytes()).hexdigest()
        candidate = FrameCandidate(
            sequence=sequence,
            update=100 + sequence,
            pts=sequence + 1,
            frame_sha256=digest,
        )
        selected.append(
            MatchingTick(
                update=100 + sequence,
                r1=candidate,
                r2=candidate,
            )
        )
    return tuple(selected)


def _write_score_crop_payload(tmp_path: Path, payload: np.ndarray) -> None:
    for run_id in ("r1", "r2"):
        capture = tmp_path / f"run-{run_id}" / "capture"
        capture.mkdir(parents=True)
        (capture / "frames.bgra.raw").write_bytes(payload.tobytes())


def test_score_crop_excludes_animated_hud_boundary_scanlines(
    tmp_path: Path,
) -> None:
    payload = np.zeros((2, 45, 130, 4), dtype=np.uint8)
    payload[:, :, :, 3] = 255
    payload[1, 7, 75, :3] = (7, 8, 9)
    payload[1, 42, 79, :3] = (9, 8, 7)
    payload[1, 43:45, 79:87, :3] = (11, 22, 33)
    _write_score_crop_payload(tmp_path, payload)

    crop_sha256 = _verify_selected_raw_frames(
        tmp_path,
        _score_crop_selected(payload),
        width=130,
        height=45,
    )

    expected = hashlib.sha256(payload[0, 8:42, :130].tobytes()).hexdigest()
    assert crop_sha256 == "sha256:" + expected


def test_score_crop_still_rejects_readout_change(tmp_path: Path) -> None:
    payload = np.zeros((2, 45, 130, 4), dtype=np.uint8)
    payload[:, :, :, 3] = 255
    payload[1, 20, 79, :3] = (11, 22, 33)
    _write_score_crop_payload(tmp_path, payload)

    with pytest.raises(PackagingError, match="score annotation crop changes"):
        _verify_selected_raw_frames(
            tmp_path,
            _score_crop_selected(payload),
            width=130,
            height=45,
        )


def test_idle_input_window_requires_explicit_authorization() -> None:
    demo = SimpleNamespace(input_commands=())

    with pytest.raises(
        PackagingError,
        match="complete mouse-button edge",
    ):
        _window_inputs(
            demo,
            first_update=100,
            last_update=200,
        )

    assert _window_inputs(
        demo,
        first_update=100,
        last_update=200,
        allow_idle_window=True,
    ) == ()
