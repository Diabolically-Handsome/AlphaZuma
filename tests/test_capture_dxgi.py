from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tools import capture_dxgi as capture
from zuma_rl.pc_protocol_evidence import PcFrameworkUpdateMap


class _ScriptedClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> int:
        self._last = next(self._values, self._last)
        return self._last


def _framework_state(
    update: int,
    *,
    draw: int,
    state: int = 3,
    pending: bool = False,
) -> capture.FrameworkStateSnapshot:
    return capture.FrameworkStateSnapshot(
        non_draw_count=0,
        frame_time_ms=10,
        is_drawing=False,
        last_draw_was_empty=False,
        has_pending_draw=pending,
        pending_updates_acc=0.25,
        update_f_time_acc=1.5,
        last_time_check=990,
        last_time=995,
        last_user_input_tick=900,
        sleep_count=100,
        draw_count=draw,
        update_count=update,
        update_app_state=state,
        update_app_depth=1,
        update_multiplier=1.0,
        paused=False,
        fast_forward_target=0,
        fast_forward_to_marker=False,
        fast_forward_step=False,
        last_draw_tick=1000,
        next_draw_tick=1000,
        step_mode=0,
    )


class _FakeCamera:
    def __init__(
        self,
        frames: list[Any],
        counters: list[tuple[int, int, int] | None],
        *,
        device_idx: int = 2,
        output_idx: int = 3,
    ) -> None:
        assert len(frames) == len(counters)
        self._frames = frames
        self._counters = counters
        self._index = 0
        self.device_idx = device_idx
        self.output_idx = output_idx
        self._duplicator = SimpleNamespace(
            latest_frame_ticks=0,
            performance_frequency=0,
            accumulated_frames=0,
        )

    def grab(
        self,
        *,
        region: capture.Region,
        copy: bool,
        new_frame_only: bool,
    ) -> Any:
        assert region == (0, 0, 2, 1)
        assert copy is True
        assert new_frame_only is True
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        counters = self._counters[self._index]
        self._index += 1
        if counters is not None:
            (
                self._duplicator.latest_frame_ticks,
                self._duplicator.performance_frequency,
                self._duplicator.accumulated_frames,
            ) = counters
        return frame


class _FakeIntoCamera(_FakeCamera):
    def __init__(
        self,
        frames: list[Any],
        counters: list[tuple[int, int, int] | None],
    ) -> None:
        super().__init__(frames, counters)
        self.destination_ids: list[int] = []

    def grab(
        self,
        *,
        region: capture.Region,
        copy: bool,
        new_frame_only: bool,
    ) -> Any:
        del region, copy, new_frame_only
        raise AssertionError("preallocated path must not call grab")

    def _grab_into(
        self,
        region: capture.Region,
        destination: Any,
    ) -> tuple[bool, int, int, int]:
        assert region == (0, 0, 2, 1)
        assert destination.shape == (1, 2, 4)
        assert destination.dtype == np.uint8
        assert destination.flags.c_contiguous
        if self._index >= len(self._frames):
            return False, 0, 0, 0
        frame = self._frames[self._index]
        counters = self._counters[self._index]
        self._index += 1
        if counters is not None:
            (
                self._duplicator.latest_frame_ticks,
                self._duplicator.performance_frequency,
                self._duplicator.accumulated_frames,
            ) = counters
        if frame is None:
            return False, 0, 0, 0
        np.copyto(destination, frame)
        self.destination_ids.append(id(destination))
        return (
            True,
            self._duplicator.latest_frame_ticks,
            destination.shape[1],
            destination.shape[0],
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.25", 0.25),
        ("3", 3.0),
        (5, 5.0),
    ],
)
def test_parse_duration_accepts_bounded_finite_values(
    value: object,
    expected: float,
) -> None:
    assert capture.parse_duration(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "",
        "nan",
        "inf",
        "-inf",
        "0.249",
            "6.001",
        object(),
    ],
)
def test_parse_duration_rejects_invalid_values(value: object) -> None:
    with pytest.raises(capture.CaptureError, match="invalid_duration"):
        capture.parse_duration(value)


@pytest.mark.parametrize("value", ["60", 60, "240", 240])
def test_parse_frame_budget_fps_accepts_integer_budget_bounds(
    value: object,
) -> None:
    assert capture.parse_frame_budget_fps(value) == int(value)


@pytest.mark.parametrize(
    "value",
    [True, None, "59", 59, "241", 241, "60.0", 60.0, "1e2"],
)
def test_parse_frame_budget_fps_rejects_noncanonical_or_out_of_range_values(
    value: object,
) -> None:
    with pytest.raises(
        capture.CaptureError,
        match="invalid_frame_budget_fps",
    ):
        capture.parse_frame_budget_fps(value)


@pytest.mark.parametrize("value", ["0", 0, "15", 15])
def test_parse_capture_index_accepts_canonical_bounded_values(
    value: object,
) -> None:
    assert capture.parse_capture_index(
        value,
        error_code="invalid_device_index",
    ) == int(value)


@pytest.mark.parametrize("value", [True, None, -1, 16, "01", 1.0])
def test_parse_capture_index_uses_requested_safe_error_code(
    value: object,
) -> None:
    with pytest.raises(
        capture.CaptureError,
        match="invalid_output_index",
    ):
        capture.parse_capture_index(
            value,
            error_code="invalid_output_index",
        )


def test_region_parsing_dimensions_and_raw_budget() -> None:
    region = capture.parse_region(("10", "20", "14", "23"))

    assert region == (10, 20, 14, 23)
    assert capture.region_dimensions(region) == (4, 3)
    assert capture.estimated_raw_bytes(region, 0.25, 60) == (
        4 * 3 * 4 * (15 + 2)
    )


@pytest.mark.parametrize(
    "values",
    [
        (),
        ("0", "0", "1"),
        ("0", "0", "1", "2", "3"),
        ("0", "0", "0", "1"),
        ("0", "0", "1", "0"),
        ("0.0", "0", "1", "1"),
        (False, 0, 1, 1),
    ],
)
def test_parse_region_rejects_invalid_rectangles(
    values: tuple[object, ...],
) -> None:
    with pytest.raises(capture.CaptureError, match="invalid_region"):
        capture.parse_region(values)


def test_parse_region_accepts_negative_desktop_coordinates() -> None:
    assert capture.parse_region(("-1920", "-20", "0", "1060")) == (
        -1920,
        -20,
        0,
        1060,
    )


def test_raw_budget_rejects_capture_over_two_gibibytes() -> None:
    with pytest.raises(capture.CaptureError, match="raw_budget_exceeded"):
        capture.estimated_raw_bytes((0, 0, 3840, 2160), 5.0, 240)


def test_validate_process_name_is_strict_and_does_not_rewrite() -> None:
    assert (
        capture.validate_process_name("ZumasRevenge.exe")
        == "ZumasRevenge.exe"
    )
    for invalid in (
        "",
        "ZumasRevenge",
        "../ZumasRevenge.exe",
        r"C:\Games\ZumasRevenge.exe",
        "Zuma Revenge.exe",
        7,
    ):
        with pytest.raises(
            capture.CaptureError,
            match="invalid_process_name",
        ):
            capture.validate_process_name(invalid)


def test_target_executable_hash_fields_is_strict_by_default() -> None:
    def locked(unused: Path) -> str:
        raise capture.CaptureError(
            "target_executable_hash_unavailable"
        )

    with pytest.raises(
        capture.CaptureError,
        match="target_executable_hash_unavailable",
    ):
        capture._target_executable_hash_fields(
            Path("locked.exe"),
            allow_deferred=False,
            hash_file_fn=locked,
        )


def test_target_executable_hash_fields_marks_probe_only_deferral() -> None:
    def locked(unused: Path) -> str:
        raise capture.CaptureError(
            "target_executable_hash_unavailable"
        )

    assert capture._target_executable_hash_fields(
        Path("locked.exe"),
        allow_deferred=True,
        hash_file_fn=locked,
    ) == {
        "executable_sha256": None,
        "executable_sha256_status": (
            "deferred_locked_runtime_payload_probe_only"
        ),
    }


def test_target_executable_hash_fields_preserves_verified_hash() -> None:
    digest = "sha256:" + "a" * 64

    assert capture._target_executable_hash_fields(
        Path("target.exe"),
        allow_deferred=True,
        hash_file_fn=lambda unused: digest,
    ) == {
        "executable_sha256": digest,
        "executable_hash_provenance": {
            "method": "direct_file_sha256",
        },
    }


def test_target_executable_hash_fields_uses_embedded_signed_pe() -> None:
    payload_digest = "sha256:" + "b" * 64
    identity = capture.EmbeddedPeIdentity(
        source_bytes=1_300,
        source_sha256="sha256:" + "a" * 64,
        payload_offset=100,
        payload_bytes=1_000,
        payload_sha256=payload_digest,
        payload_extent_basis="certificate_table_end",
        trailing_source_bytes=200,
        machine=0x14C,
        section_count=4,
        pe_timestamp=123,
        pe_checksum=456,
        size_of_image=2_000,
        size_of_headers=200,
        certificate_table_offset=900,
        certificate_table_bytes=100,
    )
    seen: dict[str, object] = {}

    def inspect(
        path: Path,
        *,
        expected_payload_bytes: int,
    ) -> capture.EmbeddedPeIdentity:
        seen["path"] = path
        seen["expected"] = expected_payload_bytes
        return identity

    result = capture._target_executable_hash_fields(
        Path("locked-runtime.exe"),
        allow_deferred=False,
        executable_bytes=1_000,
        runtime_source_executable=Path("persistent-launcher.exe"),
        hash_file_fn=lambda unused: (_ for _ in ()).throw(
            capture.CaptureError(
                "target_executable_hash_unavailable"
            )
        ),
        inspect_embedded_fn=inspect,
    )

    assert seen == {
        "path": Path("persistent-launcher.exe"),
        "expected": 1_000,
    }
    assert result["executable_sha256"] == payload_digest
    assert result["executable_hash_provenance"] == {
        "method": "embedded_signed_pe",
        **identity.to_dict(),
        "runtime_file_direct_hash_verified": False,
    }


def test_target_executable_hash_fields_cross_checks_readable_runtime() -> None:
    identity = capture.EmbeddedPeIdentity(
        source_bytes=1_300,
        source_sha256="sha256:" + "a" * 64,
        payload_offset=100,
        payload_bytes=1_000,
        payload_sha256="sha256:" + "b" * 64,
        payload_extent_basis="certificate_table_end",
        trailing_source_bytes=200,
        machine=0x14C,
        section_count=4,
        pe_timestamp=123,
        pe_checksum=456,
        size_of_image=2_000,
        size_of_headers=200,
        certificate_table_offset=900,
        certificate_table_bytes=100,
    )

    with pytest.raises(
        capture.CaptureError,
        match="runtime_source_identity_mismatch",
    ):
        capture._target_executable_hash_fields(
            Path("runtime.exe"),
            allow_deferred=False,
            executable_bytes=1_000,
            runtime_source_executable=Path("launcher.exe"),
            hash_file_fn=lambda unused: "sha256:" + "c" * 64,
            inspect_embedded_fn=lambda *args, **kwargs: identity,
        )


def test_ensure_empty_output_directory_accepts_only_empty_absolute_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert capture.ensure_empty_output_directory(empty) == empty

    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        capture.CaptureError,
        match="output_directory_not_absolute",
    ):
        capture.ensure_empty_output_directory(Path("empty"))

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(
        capture.CaptureError,
        match="output_directory_invalid",
    ):
        capture.ensure_empty_output_directory(regular_file)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_bytes(b"preserve me")
    with pytest.raises(
        capture.CaptureError,
        match="output_directory_not_empty",
    ):
        capture.ensure_empty_output_directory(nonempty)
    assert (nonempty / "sentinel").read_bytes() == b"preserve me"


def test_ensure_empty_output_directory_rejects_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(
        capture.CaptureError,
        match="output_directory_invalid",
    ):
        capture.ensure_empty_output_directory(link)


def test_resolve_global_region_uses_exact_unique_process_and_window() -> None:
    seen: dict[str, object] = {}

    def process_ids(name: str) -> tuple[int, ...]:
        seen["name"] = name
        return (17,)

    def client_regions(pids: object) -> tuple[capture.Region, ...]:
        seen["pids"] = tuple(pids)  # type: ignore[arg-type]
        return ((100, 200, 900, 800),)

    assert capture.resolve_global_region(
        process_name="ZumasRevenge.exe",
        explicit_region=(120, 220, 880, 780),
        process_ids_fn=process_ids,
        client_regions_fn=client_regions,
    ) == (120, 220, 880, 780)
    assert seen == {"name": "ZumasRevenge.exe", "pids": (17,)}


@pytest.mark.parametrize(
    ("pids", "regions", "code"),
    [
        ((), (), "target_process_missing"),
        ((1, 2), (), "target_process_not_unique"),
        ((1,), (), "target_window_missing"),
        (
            (1,),
            ((0, 0, 4, 4), (10, 10, 14, 14)),
            "target_window_not_unique",
        ),
    ],
)
def test_resolve_global_region_rejects_ambiguous_inventory(
    pids: tuple[int, ...],
    regions: tuple[capture.Region, ...],
    code: str,
) -> None:
    with pytest.raises(capture.CaptureError, match=code):
        capture.resolve_global_region(
            process_name="ZumasRevenge.exe",
            explicit_region=None,
            process_ids_fn=lambda unused: pids,
            client_regions_fn=lambda unused: regions,
        )


def test_resolve_global_region_rejects_rectangle_outside_client() -> None:
    with pytest.raises(
        capture.CaptureError,
        match="region_outside_target_window",
    ):
        capture.resolve_global_region(
            process_name="ZumasRevenge.exe",
            explicit_region=(0, 0, 101, 100),
            process_ids_fn=lambda unused: (1,),
            client_regions_fn=lambda unused: ((1, 1, 100, 100),),
        )


def test_resolve_windows_target_binds_pid_hwnd_and_region() -> None:
    target = capture.WindowTarget(
        process_id=17,
        window_handle=0x1234,
        client_region=(-100, 20, 700, 620),
    )

    resolved, region = capture.resolve_windows_target(
        process_name="ZumasRevenge.exe",
        explicit_region=(-50, 40, 650, 600),
        process_ids_fn=lambda unused: (17,),
        client_windows_fn=lambda unused: (target,),
    )

    assert resolved == target
    assert region == (-50, 40, 650, 600)


def test_resolve_windows_target_selects_unique_foreground_window() -> None:
    background = capture.WindowTarget(
        process_id=17,
        window_handle=0x1111,
        client_region=(0, 0, 3840, 2160),
    )
    foreground = capture.WindowTarget(
        process_id=17,
        window_handle=0x2222,
        client_region=(0, 0, 3840, 2160),
    )

    resolved, region = capture.resolve_windows_target(
        process_name="popcapgame1.exe",
        explicit_region=None,
        process_ids_fn=lambda unused: (17,),
        client_windows_fn=lambda unused: (background, foreground),
        foreground_window_fn=lambda: foreground.window_handle,
    )

    assert resolved == foreground
    assert region == foreground.client_region


def test_resolve_windows_target_rejects_multiple_nonforeground_windows() -> None:
    windows = (
        capture.WindowTarget(17, 0x1111, (0, 0, 800, 600)),
        capture.WindowTarget(17, 0x2222, (0, 0, 800, 600)),
    )

    with pytest.raises(capture.CaptureError, match="target_window_not_unique"):
        capture.resolve_windows_target(
            process_name="popcapgame1.exe",
            explicit_region=None,
            process_ids_fn=lambda unused: (17,),
            client_windows_fn=lambda unused: windows,
            foreground_window_fn=lambda: 0x3333,
        )


def test_output_local_region_translates_desktop_coordinates() -> None:
    camera = SimpleNamespace(
        _output=SimpleNamespace(
            rotation_angle=0,
            desc=SimpleNamespace(
                DesktopCoordinates=SimpleNamespace(
                    left=-1920,
                    top=0,
                    right=0,
                    bottom=1080,
                )
            ),
        )
    )
    assert capture.output_local_region(
        camera,
        (-1800, 20, -100, 1020),
    ) == (120, 20, 1820, 1020)

    with pytest.raises(
        capture.CaptureError,
        match="region_outside_capture_output",
    ):
        capture.output_local_region(camera, (-1800, 20, 100, 1020))


def test_capture_writes_transactional_raw_csv_metadata_and_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    frame_1 = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8]]],
        dtype=np.uint8,
    )
    frame_2 = np.array(
        [[[9, 10, 11, 12], [13, 14, 15, 16]]],
        dtype=np.uint8,
    )
    camera = _FakeCamera(
        [frame_1, frame_1, frame_2],
        [
            (90, 10_000, 1),
            (100, 10_000, 1),
            (300, 10_000, 1),
        ],
    )
    metadata = capture.capture_to_directory(
        camera=camera,
        output=output,
        global_region=(10, 20, 12, 21),
        local_region=(0, 0, 2, 1),
        duration_seconds=0.25,
        frame_budget_fps=60,
        process_name="ZumasRevenge.exe",
        device_index=2,
        output_index=3,
        monotonic_ns=_ScriptedClock(
            [0, 0, 0, 0, 10, 20, 30, 250_000_001]
        ),
        idle_sleep=lambda unused: None,
    )

    payload_1 = frame_1.tobytes()
    payload_2 = frame_2.tobytes()
    raw = payload_1 + payload_2
    aggregate = hashlib.sha256(capture._PIXEL_HASH_DOMAIN)
    for payload in (payload_1, payload_2):
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)

    assert (output / "frames.bgra.raw").read_bytes() == raw
    on_disk_metadata = json.loads(
        (output / "metadata.json").read_text(encoding="ascii")
    )
    assert metadata == on_disk_metadata
    assert metadata["status"] == "acquisition_complete"
    assert metadata["frame_count"] == 2
    assert metadata["first_present_ticks"] == 100
    assert metadata["last_present_ticks"] == 300
    assert metadata["qpc_frequency"] == 10_000
    assert metadata["missed_presentations"] == 0
    assert metadata["raw_bytes"] == len(raw)
    assert metadata["raw_sha256"] == (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    assert metadata["aggregate_pixel_sha256"] == (
        "sha256:" + aggregate.hexdigest()
    )
    assert metadata["global_region"] == [10, 20, 12, 21]
    assert metadata["output_local_region"] == [0, 0, 2, 1]
    assert metadata["device_index"] == 2
    assert metadata["output_index"] == 3
    assert metadata["frame_budget_fps"] == 60
    assert metadata["warmup_baseline_present_ticks"] == 90
    assert metadata["present_span_ticks"] == 200
    assert metadata["observed_mean_present_fps"] == 50.0
    assert metadata["frames_csv_rows"] == 2
    assert metadata["frames_csv_bytes"] == (
        output / "frames.csv"
    ).stat().st_size
    assert metadata["frames_csv_sha256"] == (
        "sha256:"
        + hashlib.sha256((output / "frames.csv").read_bytes()).hexdigest()
    )

    with (output / "frames.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sequence"] for row in rows] == ["0", "1"]
    assert [row["present_ticks"] for row in rows] == ["100", "300"]
    assert [row["raw_offset"] for row in rows] == ["0", "8"]
    assert [row["raw_bytes"] for row in rows] == ["8", "8"]
    assert [row["frame_sha256"] for row in rows] == [
        "sha256:" + hashlib.sha256(payload_1).hexdigest(),
        "sha256:" + hashlib.sha256(payload_2).hexdigest(),
    ]
    assert not list(output.glob("*.part"))


def test_capture_samples_and_publishes_framework_update_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture-with-updates"
    output.mkdir()
    frame_1 = np.zeros((1, 2, 4), dtype=np.uint8)
    frame_2 = np.ones((1, 2, 4), dtype=np.uint8)
    camera = _FakeCamera(
        [frame_1, frame_1, frame_2],
        [
            (90, 10_000, 1),
            (100, 10_000, 1),
            (300, 10_000, 1),
        ],
    )
    update_values = iter((50, 50, 51, 51))
    samples: list[capture.FrameworkUpdateSample] = []
    gate_calls: list[str] = []

    capture.capture_to_directory(
        camera=camera,
        output=output,
        global_region=(10, 20, 12, 21),
        local_region=(0, 0, 2, 1),
        duration_seconds=0.25,
        frame_budget_fps=60,
        process_name="popcapgame1.exe",
        framework_update_sampler=lambda: next(update_values),
        framework_update_samples=samples,
        capture_start_gate=lambda: gate_calls.append("opened"),
        monotonic_ns=_ScriptedClock(
            [0, 0, 0, 0, 10, 20, 30, 250_000_001]
        ),
        idle_sleep=lambda unused: None,
    )

    assert [(item.update_before, item.update_after) for item in samples] == [
        (50, 50),
        (51, 51),
    ]
    assert gate_calls == ["opened"]
    sidecar = tmp_path / "framework-updates.json"
    identity = {
        "process_id": 10,
        "process_creation_filetime_100ns": 1000,
        "executable_sha256": "sha256:" + "ab" * 32,
    }
    capture.write_framework_update_sidecar(
        sidecar,
        metadata_bytes=(output / "metadata.json").read_bytes(),
        target_identity=identity,
        samples=samples,
    )

    parsed = PcFrameworkUpdateMap.read(sidecar)
    assert parsed.process_instance == (10, 1000)
    assert [item.present_ticks for item in parsed.records] == [100, 300]
    assert [item.stable_update for item in parsed.records] == [50, 51]
    assert not sidecar.with_name(sidecar.name + ".part").exists()


def test_framework_state_decoder_uses_verified_retail_offsets() -> None:
    payload = bytearray(capture.FRAMEWORK_STATE_BYTES)
    struct.pack_into("<i", payload, 0, 4)
    struct.pack_into("<i", payload, 4, 10)
    payload[8:11] = b"\x00\x01\x01"
    struct.pack_into("<d", payload, 12, 0.25)
    struct.pack_into("<d", payload, 20, 1.5)
    struct.pack_into("<I", payload, 28, 990)
    struct.pack_into("<I", payload, 32, 995)
    struct.pack_into("<I", payload, 36, 900)
    struct.pack_into("<i", payload, 40, 91)
    struct.pack_into("<i", payload, 44, 777)
    struct.pack_into("<i", payload, 48, 4271)
    struct.pack_into("<i", payload, 52, 3)
    struct.pack_into("<i", payload, 56, 1)
    struct.pack_into("<d", payload, 60, 1.0)
    payload[68] = 0
    struct.pack_into("<i", payload, 72, 4250)
    payload[76:78] = b"\x00\x00"
    struct.pack_into("<I", payload, 80, 0xFFFFFFF0)
    struct.pack_into("<I", payload, 84, 12)
    struct.pack_into("<i", payload, 88, 0)

    state = capture.FrameworkUpdateReader.decode_state(bytes(payload))

    assert state.non_draw_count == 4
    assert state.frame_time_ms == 10
    assert state.last_draw_was_empty is True
    assert state.has_pending_draw is True
    assert state.pending_updates_acc == 0.25
    assert state.update_f_time_acc == 1.5
    assert state.last_time_check == 990
    assert state.last_time == 995
    assert state.last_user_input_tick == 900
    assert state.draw_count == 777
    assert state.update_count == 4271
    assert state.update_app_state == 3
    assert state.update_app_depth == 1
    assert state.fast_forward_target == 4250
    assert state.last_draw_tick == 0xFFFFFFF0
    assert state.next_draw_tick == 12


def test_capture_publishes_bound_framework_state_diagnostic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture-with-state"
    output.mkdir()
    frame_1 = np.zeros((1, 2, 4), dtype=np.uint8)
    frame_2 = np.ones((1, 2, 4), dtype=np.uint8)
    camera = _FakeCamera(
        [frame_1, frame_1, frame_2],
        [
            (90, 10_000, 1),
            (100, 10_000, 1),
            (300, 10_000, 1),
        ],
    )
    states = iter(
        (
            _framework_state(50, draw=70, pending=True),
            _framework_state(50, draw=70),
            _framework_state(51, draw=71, pending=True),
            _framework_state(51, draw=71),
        )
    )
    update_samples: list[capture.FrameworkUpdateSample] = []
    state_samples: list[capture.FrameworkStateSample] = []

    capture.capture_to_directory(
        camera=camera,
        output=output,
        global_region=(10, 20, 12, 21),
        local_region=(0, 0, 2, 1),
        duration_seconds=0.25,
        frame_budget_fps=60,
        process_name="popcapgame1.exe",
        framework_update_samples=update_samples,
        framework_state_sampler=lambda: next(states),
        framework_state_samples=state_samples,
        monotonic_ns=_ScriptedClock(
            [0, 0, 0, 0, 10, 20, 30, 250_000_001]
        ),
        idle_sleep=lambda unused: None,
    )

    assert [(item.update_before, item.update_after) for item in update_samples] == [
        (50, 50),
        (51, 51),
    ]
    assert [item.before.has_pending_draw for item in state_samples] == [
        True,
        True,
    ]
    identity = {
        "process_id": 10,
        "process_creation_filetime_100ns": 1000,
        "executable_sha256": "sha256:" + "ab" * 32,
    }
    metadata_bytes = (output / "metadata.json").read_bytes()
    update_path = tmp_path / "framework-updates.json"
    capture.write_framework_update_sidecar(
        update_path,
        metadata_bytes=metadata_bytes,
        target_identity=identity,
        samples=update_samples,
    )
    update_bytes = update_path.read_bytes()
    state_path = tmp_path / "framework-state.json"
    digest = capture.write_framework_state_sidecar(
        state_path,
        metadata_bytes=metadata_bytes,
        framework_update_bytes=update_bytes,
        target_identity=identity,
        samples=state_samples,
    )

    parsed = json.loads(state_path.read_text(encoding="ascii"))
    assert parsed["diagnostic_only"] is True
    assert parsed["capture_metadata_sha256"] == (
        "sha256:" + hashlib.sha256(metadata_bytes).hexdigest()
    )
    assert parsed["framework_update_map_sha256"] == (
        "sha256:" + hashlib.sha256(update_bytes).hexdigest()
    )
    assert parsed["records"][1]["after"]["update_count"] == 51
    assert digest == "sha256:" + hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert not state_path.with_name(state_path.name + ".part").exists()


def test_capture_uses_preallocated_owned_grab_into_buffers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preallocated"
    output.mkdir()
    warmup = np.zeros((1, 2, 4), dtype=np.uint8)
    frame_1 = np.full((1, 2, 4), 17, dtype=np.uint8)
    frame_2 = np.full((1, 2, 4), 23, dtype=np.uint8)
    camera = _FakeIntoCamera(
        [warmup, frame_1, frame_2],
        [
            (90, 10_000, 1),
            (100, 10_000, 1),
            (300, 10_000, 1),
        ],
    )

    metadata = capture.capture_to_directory(
        camera=camera,
        output=output,
        global_region=(10, 20, 12, 21),
        local_region=(0, 0, 2, 1),
        duration_seconds=0.25,
        frame_budget_fps=60,
        process_name="ZumasRevenge.exe",
        monotonic_ns=_ScriptedClock(
            [0, 0, 0, 0, 10, 20, 30, 250_000_001]
        ),
        idle_sleep=lambda unused: None,
    )

    assert camera.destination_ids[0] == camera.destination_ids[1]
    assert camera.destination_ids[1] != camera.destination_ids[2]
    assert metadata["frame_count"] == 2
    assert (output / "frames.bgra.raw").read_bytes() == (
        frame_1.tobytes() + frame_2.tobytes()
    )


def test_preallocated_grab_into_rejects_malformed_no_frame_result(
    tmp_path: Path,
) -> None:
    class MalformedIntoCamera(_FakeIntoCamera):
        def _grab_into(
            self,
            region: capture.Region,
            destination: Any,
        ) -> tuple[bool, int, int, int]:
            del region, destination
            return False, 1, 0, 0

    output = tmp_path / "malformed-into"
    output.mkdir()
    frame = np.zeros((1, 2, 4), dtype=np.uint8)
    camera = MalformedIntoCamera(
        [frame],
        [(50, 10_000, 1)],
    )

    with pytest.raises(
        capture.CaptureError,
        match="frame_layout_invalid",
    ):
        capture.capture_to_directory(
            camera=camera,
            output=output,
            global_region=(10, 20, 12, 21),
            local_region=(0, 0, 2, 1),
            duration_seconds=0.25,
            frame_budget_fps=60,
            process_name="ZumasRevenge.exe",
            monotonic_ns=_ScriptedClock([0, 0, 250_000_001]),
            idle_sleep=lambda unused: None,
        )

    assert list(output.glob("*.part"))
    assert not (output / "metadata.json").exists()


def test_capture_skips_pointer_only_updates_without_recording_them(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pointer-only"
    output.mkdir()
    frame = np.zeros((1, 2, 4), dtype=np.uint8)
    camera = _FakeCamera(
        [frame, frame, frame],
        [
            (50, 10_000, 1),
            (75, 10_000, 0),
            (100, 10_000, 1),
        ],
    )

    metadata = capture.capture_to_directory(
        camera=camera,
        output=output,
        global_region=(10, 20, 12, 21),
        local_region=(0, 0, 2, 1),
        duration_seconds=0.25,
        frame_budget_fps=60,
        process_name="ZumasRevenge.exe",
        monotonic_ns=_ScriptedClock(
            [0, 0, 0, 0, 10, 20, 30, 250_000_001]
        ),
        idle_sleep=lambda unused: None,
    )

    assert metadata["pointer_only_update_count"] == 1
    assert metadata["frame_count"] == 1
    assert (output / "frames.bgra.raw").read_bytes() == frame.tobytes()


def test_capture_rejects_duplicator_recovery(
    tmp_path: Path,
) -> None:
    class RecoveringCamera(_FakeCamera):
        def grab(
            self,
            *,
            region: capture.Region,
            copy: bool,
            new_frame_only: bool,
        ) -> Any:
            frame = super().grab(
                region=region,
                copy=copy,
                new_frame_only=new_frame_only,
            )
            if self._index == 2:
                self._duplicator = SimpleNamespace(
                    latest_frame_ticks=100,
                    performance_frequency=10_000,
                    accumulated_frames=1,
                )
            return frame

    output = tmp_path / "duplicator-change"
    output.mkdir()
    frame = np.zeros((1, 2, 4), dtype=np.uint8)
    camera = RecoveringCamera(
        [frame, frame],
        [(50, 10_000, 1), (100, 10_000, 1)],
    )

    with pytest.raises(
        capture.CaptureError,
        match="capture_source_changed",
    ):
        capture.capture_to_directory(
            camera=camera,
            output=output,
            global_region=(10, 20, 12, 21),
            local_region=(0, 0, 2, 1),
            duration_seconds=0.25,
            frame_budget_fps=60,
            process_name="ZumasRevenge.exe",
            monotonic_ns=_ScriptedClock(
                [0, 0, 0, 0, 10, 250_000_001]
            ),
            idle_sleep=lambda unused: None,
        )

    assert not (output / "metadata.json").exists()
    assert list(output.glob("*.part"))


def test_runtime_version_preflight_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture.importlib.metadata,
        "version",
        lambda package: capture.SUPPORTED_RUNTIME_VERSIONS[package],
    )
    assert capture.validate_runtime_versions() == (
        capture.SUPPORTED_RUNTIME_VERSIONS
    )

    monkeypatch.setattr(
        capture.importlib.metadata,
        "version",
        lambda package: "9.9.9",
    )
    with pytest.raises(
        capture.CaptureError,
        match="unsupported_capture_runtime",
    ):
        capture.validate_runtime_versions()


@pytest.mark.parametrize(
    ("frames", "counters", "clock_values", "code"),
    [
        (
            [
                np.zeros((1, 2, 4), dtype=np.uint8),
                np.zeros((1, 2, 4), dtype=np.uint8),
                np.ones((1, 2, 4), dtype=np.uint8),
            ],
            [
                (50, 10_000, 1),
                (100, 10_000, 1),
                (100, 10_000, 1),
            ],
            [0, 0, 0, 0, 10, 20, 30, 250_000_001],
            "dxgi_ticks_not_increasing",
        ),
        (
            [np.zeros((1, 2, 4), dtype=np.uint8), None],
            [(50, 10_000, 1), None],
            [0, 0, 0, 0, 10, 250_000_001],
            "no_presented_frames",
        ),
        (
            [
                np.zeros((1, 2, 4), dtype=np.uint8),
                np.zeros((1, 2, 3), dtype=np.uint8),
            ],
            [(50, 10_000, 1), (100, 10_000, 1)],
            [0, 0, 0, 0, 10, 250_000_001],
            "frame_layout_invalid",
        ),
        (
            [
                np.zeros((1, 2, 4), dtype=np.uint8),
                np.ones((1, 2, 4), dtype=np.uint8),
            ],
            [(50, 10_000, 1), (100, 10_000, 2)],
            [0, 0, 0, 0, 10, 250_000_001],
            "missed_presentations_detected",
        ),
    ],
)
def test_capture_failure_keeps_parts_and_never_publishes_commit_marker(
    tmp_path: Path,
    frames: list[Any],
    counters: list[tuple[int, int, int] | None],
    clock_values: list[int],
    code: str,
) -> None:
    output = tmp_path / code
    output.mkdir()
    camera = _FakeCamera(frames, counters)

    with pytest.raises(capture.CaptureError, match=code):
        capture.capture_to_directory(
            camera=camera,
            output=output,
            global_region=(10, 20, 12, 21),
            local_region=(0, 0, 2, 1),
            duration_seconds=0.25,
            frame_budget_fps=60,
            process_name="ZumasRevenge.exe",
            monotonic_ns=_ScriptedClock(clock_values),
            idle_sleep=lambda unused: None,
        )

    assert (output / "frames.bgra.raw.part").is_file()
    assert (output / "frames.csv.part").is_file()
    assert not (output / "frames.bgra.raw").exists()
    assert not (output / "frames.csv").exists()
    assert not (output / "metadata.json").exists()


def test_safe_cli_parser_never_echoes_unknown_private_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-user-path-do-not-echo"

    status = capture.main(["--unknown", secret])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert captured.err == "capture error: invalid_arguments\n"
    assert secret not in captured.err
    assert "Traceback" not in captured.err


def test_safe_cli_masks_unexpected_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-exception-payload"

    def explode(argv: list[str] | None = None) -> int:
        del argv
        raise RuntimeError(secret)

    monkeypatch.setattr(capture, "run", explode)
    status = capture.main([])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert captured.err == "capture error: unexpected_failure\n"
    assert secret not in captured.err
    assert "Traceback" not in captured.err
