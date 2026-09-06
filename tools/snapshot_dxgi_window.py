"""Save one lossless BGRA snapshot of a Windows client area as a BMP.

This is a single-frame companion to ``capture_dxgi.py``.  By default it is a
diagnostic capture.  A higher-level collector may explicitly label the frame
as formal exact-step evidence after it has established the native update
barrier, process identity, repaint handshake, and transport contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.capture_dxgi import (
    CaptureError,
    output_local_region,
    parse_capture_index,
    region_dimensions,
    resolve_windows_target,
    validate_process_name,
    verify_windows_target,
)


DIAGNOSTIC_SNAPSHOT_SEMANTICS = (
    "diagnostic_single_frame_not_formal_evidence"
)
FORMAL_EXACT_STEP_SNAPSHOT_SEMANTICS = (
    "formal_exact_step_external_lossless_evidence"
)
_SNAPSHOT_SEMANTICS = frozenset(
    {
        DIAGNOSTIC_SNAPSHOT_SEMANTICS,
        FORMAL_EXACT_STEP_SNAPSHOT_SEMANTICS,
    }
)


def _top_down_bgra_bmp(frame: Any, width: int, height: int) -> bytes:
    """Return an uncompressed top-down 32-bit BMP for one BGRA ndarray."""

    try:
        shape = tuple(int(value) for value in frame.shape)
        contiguous = bool(frame.flags.c_contiguous)
    except (AttributeError, TypeError, ValueError):
        raise CaptureError("snapshot_frame_invalid") from None
    if shape != (height, width, 4) or not contiguous:
        raise CaptureError("snapshot_frame_invalid")

    pixels = frame.tobytes(order="C")
    expected_bytes = width * height * 4
    if len(pixels) != expected_bytes:
        raise CaptureError("snapshot_frame_invalid")

    pixel_offset = 14 + 40
    file_size = pixel_offset + expected_bytes
    file_header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0,
        0,
        pixel_offset,
    )
    dib_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        -height,
        1,
        32,
        0,
        expected_bytes,
        2835,
        2835,
        0,
        0,
    )
    return file_header + dib_header + pixels


def snapshot(
    *,
    output: Path,
    process_name: str,
    device_index: int,
    output_index: int,
    timeout: float,
    evidence_semantics: str = DIAGNOSTIC_SNAPSHOT_SEMANTICS,
) -> dict[str, object]:
    """Acquire and publish one lossless client-area snapshot."""

    process_name = validate_process_name(process_name)
    device_index = parse_capture_index(
        device_index,
        error_code="invalid_device_index",
    )
    output_index = parse_capture_index(
        output_index,
        error_code="invalid_output_index",
    )
    if timeout <= 0:
        raise CaptureError("invalid_timeout")
    if evidence_semantics not in _SNAPSHOT_SEMANTICS:
        raise CaptureError("snapshot_semantics_invalid")
    output = output.resolve()
    if output.exists():
        raise CaptureError("snapshot_output_exists")
    if not output.parent.is_dir():
        raise CaptureError("snapshot_output_parent_missing")

    target, global_region = resolve_windows_target(
        process_name=process_name,
        explicit_region=None,
    )
    verify_windows_target(target)
    width, height = region_dimensions(global_region)

    try:
        import dxcam
    except (ImportError, OSError, RuntimeError):
        raise CaptureError("dxcam_unavailable") from None

    camera = None
    try:
        camera = dxcam.create(
            device_idx=device_index,
            output_idx=output_index,
            output_color="BGRA",
            backend="dxgi",
            processor_backend="numpy",
        )
        local_region = output_local_region(camera, global_region)
        deadline = time.monotonic() + timeout
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = camera.grab(region=local_region, copy=True)
            if frame is None:
                time.sleep(0.005)
        if frame is None:
            raise CaptureError("snapshot_timeout")
        verify_windows_target(target)
        if output_local_region(camera, global_region) != local_region:
            raise CaptureError("capture_source_changed")
        encoded = _top_down_bgra_bmp(frame, width, height)
        with output.open("xb") as stream:
            stream.write(encoded)
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass

    return {
        "output": str(output),
        "process_id": target.process_id,
        "window_handle_hex": f"0x{target.window_handle:016x}",
        "client_region": list(global_region),
        "width": width,
        "height": height,
        "bytes": len(encoded),
        "semantics": evidence_semantics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process", default="popcapgame1.exe")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = snapshot(
            output=args.output,
            process_name=args.process,
            device_index=args.device_index,
            output_index=args.output_index,
            timeout=args.timeout,
        )
    except CaptureError as error:
        print(f"snapshot error: {error.code}", file=sys.stderr)
        return 1
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
