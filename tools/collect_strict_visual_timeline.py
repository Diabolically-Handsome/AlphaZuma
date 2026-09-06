"""Collect audited UI snapshots along one strict retail DMO replay.

This is a diagnostic collector, not PC Golden evidence.  It keeps the strict
service-brokered command trace attached, takes screenshots only after a real
non-client repaint handshake, and restores the user's scoped Zuma state in a
``finally`` path.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.autoplay_live_zuma import LiveBoardUnavailable, read_live_board
from tools.capture_dxgi import FrameworkUpdateReader
from tools.collect_pc_golden_v4 import (
    CollectionError,
    EXPECTED_RUNTIME_SHA256,
    OVERLAY_NAMES,
    _activate_startup_window,
    _child_environment,
    _require_game_stopped,
    _terminate_exact_runtime,
    _wait_for_capture_window,
    _wait_for_runtime,
    _wait_game_stopped,
    _window_repaint_handshake,
)
from tools.launch_popcap_replay import process_ids_by_name
from tools.pc_state_transaction import (
    capture_state,
    overlay_snapshot_files,
    restore_state,
    write_snapshot_exclusive,
)
from tools.snapshot_dxgi_window import snapshot
from zuma_rl.popcap_dmo import PopCapDemo


SCHEMA = "zuma-rl.strict-visual-timeline-diagnostic"
VERSION = 4
DEFAULT_RUNTIME = Path(
    r"D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe"
)
DEFAULT_CHANGEDIR = Path(
    r"D:\SteamLibrary\steamapps\common\Zuma's Revenge"
)
DEFAULT_STEAM = Path(r"C:\Program Files (x86)\Steam\steam.exe")
DEFAULT_PRESTATE = Path(
    r"D:\ZumaGolden\session-20260728-203153"
    r"\clean-level1-001-prestate-reconstructed"
)
DEFAULT_BOARD_SEED_ADDRESS = 0x0065B828
DEFAULT_BOARD_SEED = 1162643045
DEFAULT_SERVICE_WAIT_TIMEOUT = 60.0
STILL_ACTIVE = 259
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _target_update(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("target update must be non-negative")
    return value


def _session_nonce(text: str) -> str:
    if not _NONCE_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(
            "session nonce must contain exactly 32 lowercase hex digits"
        )
    return text


def _compact_board(board: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "native_game_time",
        "score",
        "displayed_score",
        "score_target",
        "active_ball_count",
        "pending_ball_count",
        "inserting_ball_count",
        "fired_ball_count",
        "runtime_active",
        "curve_plan_exhausted",
        "loss_counter",
    )
    return {key: board.get(key) for key in keys}


def _stable_repaint_handshake(
    pid: int,
    target: Any,
    *,
    maximum_attempts: int = 3,
) -> tuple[Any, dict[str, Any]]:
    """Retry only a game-driven geometry transition, then audit success."""

    failures: list[dict[str, Any]] = []
    current = target
    for attempt in range(1, maximum_attempts + 1):
        try:
            evidence = dict(_window_repaint_handshake(current))
            evidence["diagnostic_attempt"] = attempt
            evidence["diagnostic_prior_failures"] = failures
            return current, evidence
        except CollectionError as error:
            if (
                error.code != "window_repaint_geometry_not_restored"
                or attempt == maximum_attempts
            ):
                raise
            failures.append(
                {
                    "attempt": attempt,
                    "code": error.code,
                    "perf_counter_ns": time.perf_counter_ns(),
                }
            )
            time.sleep(0.25)
            current = _wait_for_capture_window(pid)
            _activate_startup_window(current)
    raise AssertionError("unreachable repaint retry state")


def _runtime_sha256(path: Path) -> str:
    return _sha256(path)


def _normal_close(pid: int, window_handle: int, timeout: float = 10.0) -> int | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    user32.PostMessageW.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x00100000 | 0x00001000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not user32.PostMessageW(
            wintypes.HWND(window_handle),
            0x0010,
            0,
            0,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        wait_result = kernel32.WaitForSingleObject(
            handle,
            max(1, round(timeout * 1000)),
        )
        if wait_result != 0:
            return None
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if int(exit_code.value) == STILL_ACTIVE:
            return None
        return int(exit_code.value)
    finally:
        kernel32.CloseHandle(handle)


def _trace_arguments(
    *,
    project_root: Path,
    dmo: Path,
    runtime: Path,
    changedir: Path,
    steam: Path,
    seed: int,
    board_seed: int,
    board_seed_address: int,
    stop_update: int,
    trace_timeout: float,
    result_path: Path,
    detach_at_update: int | None,
    reattach_at_update: int | None,
    allow_post_blackout_attach_stabilization: bool,
    allow_blackout_file_write_order_rebase: bool,
) -> list[str]:
    arguments = [
        sys.executable,
        str(project_root / "tools" / "trace_popcap_demo_commands.py"),
        "--dmo",
        str(dmo),
        "--steam-exe",
        str(steam),
        "--runtime-exe",
        str(runtime),
        "--direct-runtime-exe",
        str(runtime),
        "--changedir",
        str(changedir),
        "--crt-rand-seed",
        str(seed),
        "--startup-seed-transport",
        "debugger_register",
        "--board-seed-address",
        str(board_seed_address),
        "--board-seed",
        str(board_seed),
        "--global-rng-seed",
        str(seed),
        "--thread-crt-rng-seed",
        str(seed),
        "--maximum-hits",
        "50000",
        "--launch-timeout",
        "30",
        "--trace-timeout",
        str(trace_timeout),
        "--broker-service-blocks",
        "--service-wait-timeout",
        str(DEFAULT_SERVICE_WAIT_TIMEOUT),
        "--quiet-nonservice",
        "--progress-every-updates",
        "500",
        "--allow-pre-stream-commands",
        "--result-json",
        str(result_path),
    ]
    if detach_at_update is None:
        arguments.extend(("--stop-after-update", str(stop_update)))
    else:
        arguments.extend(
            ("--detach-at-update", str(detach_at_update))
        )
        arguments.extend(("--attach-timeout", str(trace_timeout)))
        arguments.append("--accept-normal-exit")
        arguments.append("--close-after-terminal-command")
    if reattach_at_update is not None:
        arguments.extend(
            ("--reattach-at-update", str(reattach_at_update))
        )
    if allow_post_blackout_attach_stabilization:
        arguments.append(
            "--allow-post-blackout-attach-stabilization"
        )
    if allow_blackout_file_write_order_rebase:
        arguments.append(
            "--allow-blackout-file-write-order-rebase"
        )
    return arguments


def collect_timeline(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    dmo = args.dmo.resolve()
    runtime = args.runtime.resolve()
    changedir = args.changedir.resolve()
    prestate = args.prestate.resolve()
    steam = args.steam.resolve()
    targets = tuple(sorted(set(args.target_update)))
    if not targets:
        raise ValueError("at least one target update is required")
    stop_update = targets[-1]
    demo = PopCapDemo.read(dmo)
    seed = demo.random_seed if args.seed is None else args.seed
    if (
        output_root.exists()
        or not output_root.parent.is_dir()
        or not runtime.is_file()
        or not changedir.is_dir()
        or not prestate.is_dir()
        or not steam.is_file()
        or _runtime_sha256(runtime) != EXPECTED_RUNTIME_SHA256
        or stop_update > demo.length_updates
    ):
        raise ValueError("timeline input identity is invalid")
    if (args.detach_at_update is None) != (
        args.reattach_at_update is None
    ):
        raise ValueError(
            "detach and reattach updates must be provided together"
        )
    if args.detach_at_update is not None:
        if not (
            0 <= args.detach_at_update
            < args.reattach_at_update
            < stop_update
        ):
            raise ValueError("timeline detach/reattach schedule is invalid")
    elif (
        args.allow_post_blackout_attach_stabilization
        or args.allow_blackout_file_write_order_rebase
    ):
        raise ValueError(
            "post-blackout auditing requires detach/reattach"
        )
    for name in OVERLAY_NAMES:
        if not (prestate / name).is_file():
            raise ValueError(f"prestate overlay is missing: {name}")

    _require_game_stopped()
    output_root.mkdir()
    snapshots_root = output_root / "snapshots"
    repaint_root = output_root / "repaints"
    snapshots_root.mkdir()
    repaint_root.mkdir()
    host = capture_state(session_nonce=args.session_nonce, phase="host-pre")
    write_snapshot_exclusive(host, output_root / "host-pre.json")
    template = overlay_snapshot_files(
        host,
        tuple((name, prestate / name) for name in OVERLAY_NAMES),
        phase="diagnostic-pre",
        captured_perf_counter_ns=time.perf_counter_ns(),
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "diagnostic-not-pc-golden",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dmo": {
            "path": str(dmo),
            "sha256": f"sha256:{demo.artifact_sha256}",
            "length_updates": demo.length_updates,
            "random_seed": demo.random_seed,
        },
        "runtime": {
            "path": str(runtime),
            "sha256": EXPECTED_RUNTIME_SHA256,
        },
        "target_updates": list(targets),
        "stop_update": stop_update,
        "trace_schedule": {
            "detach_at_update": args.detach_at_update,
            "reattach_at_update": args.reattach_at_update,
            "allow_post_blackout_attach_stabilization": (
                args.allow_post_blackout_attach_stabilization
            ),
            "allow_blackout_file_write_order_rebase": (
                args.allow_blackout_file_write_order_rebase
            ),
        },
        "host_pre_state_root": host.state_root,
        "template_state_root": template.state_root,
        "rows": [],
        "trace_return_code": None,
        "normal_close_exit_code": None,
        "normal_close_exit_zero": False,
        "error": None,
    }
    trace_process: subprocess.Popen[bytes] | None = None
    runtime_pid: int | None = None
    target_window = None
    primary_error: BaseException | None = None
    try:
        applied = restore_state(template)
        if applied.state_root != template.state_root:
            raise RuntimeError("diagnostic prestate restore mismatch")
        write_snapshot_exclusive(applied, output_root / "applied-pre.json")
        existing = set(process_ids_by_name(runtime.name))
        trace_result_path = output_root / "strict-replay.json"
        trace_args = _trace_arguments(
            project_root=project_root,
            dmo=dmo,
            runtime=runtime,
            changedir=changedir,
            steam=steam,
            seed=seed,
            board_seed=args.board_seed,
            board_seed_address=args.board_seed_address,
            stop_update=stop_update,
            trace_timeout=args.trace_timeout,
            result_path=trace_result_path,
            detach_at_update=args.detach_at_update,
            reattach_at_update=args.reattach_at_update,
            allow_post_blackout_attach_stabilization=(
                args.allow_post_blackout_attach_stabilization
            ),
            allow_blackout_file_write_order_rebase=(
                args.allow_blackout_file_write_order_rebase
            ),
        )
        with (output_root / "strict-replay.log").open("xb") as trace_log:
            trace_process = subprocess.Popen(
                trace_args,
                cwd=project_root,
                env=_child_environment(project_root),
                stdin=subprocess.DEVNULL,
                stdout=trace_log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            runtime_pid, _ = _wait_for_runtime(
                trace_process,
                existing=existing,
                expected_runtime=runtime,
                timeout=35.0,
            )
            result["pid"] = runtime_pid
            target_window = _wait_for_capture_window(runtime_pid)
            activation = _activate_startup_window(target_window)
            _write_json_exclusive(
                output_root / "startup-window-activation.json",
                activation,
            )

            deadline = time.monotonic() + args.trace_timeout
            with FrameworkUpdateReader(runtime_pid) as update_reader:
                for target_update in targets:
                    observed_update = -1
                    while time.monotonic() < deadline:
                        observed_update = update_reader.sample()
                        if observed_update >= target_update:
                            break
                        return_code = trace_process.poll()
                        if return_code is not None:
                            raise RuntimeError(
                                "strict trace exited before target update "
                                f"{target_update}: {return_code}"
                            )
                        time.sleep(0.001)
                    else:
                        raise TimeoutError(
                            f"target update timed out: {target_update}"
                        )

                    target_window, repaint = _stable_repaint_handshake(
                        runtime_pid,
                        target_window,
                    )
                    repaint_path = repaint_root / f"u{target_update}.json"
                    _write_json_exclusive(repaint_path, repaint)
                    image_path = snapshots_root / f"u{target_update}.bmp"
                    capture_evidence = snapshot(
                        output=image_path,
                        process_name=runtime.name,
                        device_index=0,
                        output_index=0,
                        timeout=2.0,
                    )
                    board: dict[str, Any] | None = None
                    board_error: str | None = None
                    try:
                        board = _compact_board(read_live_board(runtime_pid))
                    except (
                        LiveBoardUnavailable,
                        OSError,
                        RuntimeError,
                        ValueError,
                    ) as error:
                        board_error = f"{type(error).__name__}: {error}"
                    observed_after = update_reader.sample()
                    row = {
                        "target_update": target_update,
                        "observed_update_before_repaint": observed_update,
                        "observed_update_after_capture": observed_after,
                        "image": str(image_path),
                        "image_sha256": _sha256(image_path),
                        "capture": capture_evidence,
                        "repaint": {
                            "artifact": str(repaint_path),
                            "geometry_restored": repaint[
                                "geometry_restored"
                            ],
                            "actual_temporary_delta": repaint[
                                "actual_temporary_delta"
                            ],
                        },
                        "board": board,
                        "board_error": board_error,
                    }
                    result["rows"].append(row)
                    print(
                        "timeline_snapshot "
                        f"target={target_update} "
                        f"observed={observed_update}:{observed_after} "
                        f"board={'yes' if board is not None else 'no'}",
                        flush=True,
                    )

            result["trace_return_code"] = trace_process.wait(timeout=30.0)
            if result["trace_return_code"] != 0:
                raise RuntimeError("strict trace failed")
            if args.detach_at_update is None:
                exit_code = _normal_close(
                    runtime_pid,
                    target_window.window_handle,
                )
            else:
                trace_report = json.loads(
                    trace_result_path.read_text(encoding="utf-8")
                )
                exit_code = trace_report["result"]["exit_code"]
            result["normal_close_exit_code"] = exit_code
            result["normal_close_exit_zero"] = exit_code == 0
            if exit_code != 0:
                raise RuntimeError("runtime did not exit normally")
    except BaseException as error:
        primary_error = error
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if trace_process is not None and trace_process.poll() is None:
            trace_process.terminate()
            try:
                trace_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                trace_process.kill()
                trace_process.wait(timeout=10.0)
        if runtime_pid is not None:
            _terminate_exact_runtime(runtime_pid, runtime)
        for process_id in process_ids_by_name(runtime.name):
            _terminate_exact_runtime(process_id, runtime)
        _wait_game_stopped()
        restored_host = restore_state(host)
        if restored_host.state_root != host.state_root:
            raise RuntimeError("host state restore mismatch")
        host_after = capture_state(
            session_nonce=args.session_nonce,
            phase="host-restored",
        )
        write_snapshot_exclusive(
            host_after,
            output_root / "host-restored.json",
        )
        if host_after.state_root != host.state_root:
            raise RuntimeError("host state verification mismatch")
        result["host_restored_state_root"] = host_after.state_root
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_exclusive(output_root / "result.json", result)

    if primary_error is not None:
        raise primary_error
    print(output_root / "result.json", flush=True)
    return output_root / "result.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--changedir", type=Path, default=DEFAULT_CHANGEDIR)
    parser.add_argument("--steam", type=Path, default=DEFAULT_STEAM)
    parser.add_argument("--prestate", type=Path, default=DEFAULT_PRESTATE)
    parser.add_argument(
        "--target-update",
        action="append",
        required=True,
        type=_target_update,
    )
    parser.add_argument("--seed", type=_target_update)
    parser.add_argument(
        "--board-seed",
        type=_target_update,
        default=DEFAULT_BOARD_SEED,
    )
    parser.add_argument(
        "--board-seed-address",
        type=_target_update,
        default=DEFAULT_BOARD_SEED_ADDRESS,
    )
    parser.add_argument("--trace-timeout", type=float, default=1200.0)
    parser.add_argument("--detach-at-update", type=_target_update)
    parser.add_argument("--reattach-at-update", type=_target_update)
    parser.add_argument(
        "--allow-post-blackout-attach-stabilization",
        action="store_true",
    )
    parser.add_argument(
        "--allow-blackout-file-write-order-rebase",
        action="store_true",
    )
    parser.add_argument(
        "--session-nonce",
        type=_session_nonce,
        default="c2390000000000000000000000000000",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.trace_timeout <= 0:
        raise ValueError("trace timeout must be positive")
    collect_timeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
