"""Launch and immediately slow a PopCap DMO replay on Windows.

Steam may play a short DMO to completion less than a second after the game
window appears.  This helper removes human/UI-automation latency from the
capture boundary:

1. snapshot existing runtime PIDs;
2. ask Steam to launch the selected DMO;
3. bind the newly created, path-verified runtime and its ``MainWindow``;
4. post the framework's documented ``-`` replay control immediately; and
5. verify the resulting multiplier through read-only process inspection.

The helper never writes target-process memory.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.control_popcap_replay import (
    expected_multiplier,
    format_state,
    locate_replay_state,
    post_char,
    windows_for_pid,
)
from tools.inspect_popcap_replay import ReplayState, close_process


TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
DEFAULT_STEAM_EXE = Path(r"C:\Program Files (x86)\Steam\steam.exe")
DEFAULT_RUNTIME_EXE = Path(
    r"C:\ProgramData\PopCap Games\ZumasRevenge\popcapgame1.exe"
)


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def process_ids_by_name(executable_name: str) -> tuple[int, ...]:
    """Return current PIDs with an exact case-insensitive image name."""

    if os.name != "nt":
        raise RuntimeError("PopCap replay launch control requires Windows")
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    wanted = executable_name.casefold()
    found: list[int] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.casefold() == wanted:
                found.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return tuple(sorted(found))


def process_image_path(pid: int) -> Path | None:
    """Read a process image path using query-only access."""

    if os.name != "nt":
        raise RuntimeError("process path inspection requires Windows")
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        capacity = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(capacity.value)
        if not kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(capacity)
        ):
            return None
        return Path(buffer.value)
    finally:
        kernel32.CloseHandle(handle)


def same_windows_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def replay_main_window(pid: int) -> int | None:
    candidates = [
        hwnd
        for hwnd, class_name, title in windows_for_pid(pid)
        if class_name == "MainWindow" and title.startswith("Zuma's Revenge!")
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"multiple replay MainWindow handles for PID {pid}")
    return candidates[0] if candidates else None


def wait_for_new_runtime_window(
    *,
    existing_pids: set[int],
    runtime_exe: Path,
    timeout: float,
) -> tuple[int, int]:
    """Bind a newly created path-verified runtime and its main window."""

    deadline = time.monotonic() + timeout
    observed: set[int] = set()
    while time.monotonic() < deadline:
        current = set(process_ids_by_name(runtime_exe.name))
        for pid in sorted(current - existing_pids):
            observed.add(pid)
            image = process_image_path(pid)
            if image is None or not same_windows_path(image, runtime_exe):
                continue
            hwnd = replay_main_window(pid)
            if hwnd is not None:
                return pid, hwnd
        time.sleep(0.002)
    raise TimeoutError(
        "timed out waiting for a new path-verified PopCap replay window; "
        f"observed runtime PIDs={sorted(observed)}"
    )


def wait_for_pid_window(
    *,
    pid: int,
    runtime_exe: Path,
    timeout: float,
) -> int:
    """Wait for one known path-verified process to create its main window."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        image = process_image_path(pid)
        if image is None:
            raise RuntimeError(f"direct replay PID {pid} exited before its window")
        if not same_windows_path(image, runtime_exe):
            raise RuntimeError(
                f"direct replay PID {pid} path mismatch: {image} != {runtime_exe}"
            )
        hwnd = replay_main_window(pid)
        if hwnd is not None:
            return hwnd
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for direct replay PID {pid} window")


def wait_for_verified_state(
    *,
    pid: int,
    minus_count: int,
    demo_length: int,
    timeout: float,
):
    """Wait until every posted minus was consumed, then return live state."""

    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return locate_replay_state(pid, minus_count, demo_length)
        except (OSError, RuntimeError) as error:
            last_error = error
            time.sleep(0.005)
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(f"could not verify slowed replay state{detail}")


def direct_launch_environment(app_id: int) -> dict[str, str]:
    """Return the retail direct-launch environment without DPI scaling."""

    if app_id <= 0:
        raise ValueError("app_id must be positive")
    environment = os.environ.copy()
    environment["SteamAppId"] = str(app_id)
    environment["SteamGameId"] = str(app_id)
    environment["__COMPAT_LAYER"] = "HIGHDPIAWARE"
    return environment


def launch_replay(
    *,
    steam_exe: Path,
    app_id: int,
    dmo: Path,
    runtime_exe: Path,
    minus_count: int,
    demo_length: int,
    timeout: float,
    direct_runtime_exe: Path | None = None,
    changedir: Path | None = None,
) -> tuple[int, int, ReplayState]:
    """Launch, slow, and read-only verify one replay runtime."""

    if os.name != "nt":
        raise RuntimeError("PopCap replay launch control requires Windows")
    if minus_count < 0:
        raise ValueError("minus_count must be non-negative")
    required_paths = [(dmo, "DMO")]
    if direct_runtime_exe is None:
        required_paths.append((steam_exe, "Steam executable"))
    else:
        required_paths.append((direct_runtime_exe, "direct runtime executable"))
        if changedir is None or not changedir.is_dir():
            raise FileNotFoundError(
                f"direct runtime changedir does not exist: {changedir}"
            )
    for path, label in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    expected_runtime = (
        direct_runtime_exe.resolve()
        if direct_runtime_exe is not None
        else runtime_exe
    )
    existing = set(process_ids_by_name(expected_runtime.name))
    if existing:
        raise RuntimeError(
            "refusing to launch with an existing PopCap runtime: "
            f"{sorted(existing)}"
        )

    if direct_runtime_exe is None:
        command = [
            str(steam_exe),
            "-silent",
            "-applaunch",
            str(app_id),
            "-play",
            f"-demofile={dmo}",
        ]
        launched = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        del launched
        pid, hwnd = wait_for_new_runtime_window(
            existing_pids=existing,
            runtime_exe=expected_runtime,
            timeout=timeout,
        )
    else:
        command = [
            str(direct_runtime_exe),
            "-play",
            f"-demofile={dmo}",
            f"-changedir={changedir.resolve()}\\",
        ]
        direct_environment = direct_launch_environment(app_id)
        launched = subprocess.Popen(
            command,
            cwd=changedir,
            env=direct_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        pid = launched.pid
        hwnd = wait_for_pid_window(
            pid=pid,
            runtime_exe=expected_runtime,
            timeout=timeout,
        )
    for _ in range(minus_count):
        post_char(hwnd, "-")

    handle, address, state = wait_for_verified_state(
        pid=pid,
        minus_count=minus_count,
        demo_length=demo_length,
        timeout=timeout,
    )
    close_process(handle)
    if state.update_multiplier != expected_multiplier(minus_count):
        raise RuntimeError("verified replay multiplier changed unexpectedly")
    if state.update_count >= demo_length:
        raise RuntimeError("replay reached its declared end before verification")
    return pid, hwnd, state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically launch and slow a Zuma's Revenge DMO replay."
    )
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--demo-length", required=True, type=int)
    parser.add_argument("--minus-count", type=int, default=6)
    parser.add_argument("--steam-exe", type=Path, default=DEFAULT_STEAM_EXE)
    parser.add_argument("--app-id", type=int, default=3620)
    parser.add_argument("--runtime-exe", type=Path, default=DEFAULT_RUNTIME_EXE)
    parser.add_argument(
        "--direct-runtime-exe",
        type=Path,
        help="Diagnostic: bypass Steam and run this extracted runtime directly.",
    )
    parser.add_argument(
        "--changedir",
        type=Path,
        help="Required with --direct-runtime-exe; original game asset directory.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--monitor-until-exit",
        action="store_true",
        help=(
            "After the read-only launch verification, stay alive until the "
            "exact runtime exits so a parent collector can monitor launch "
            "liveness without attaching a debugger."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pid, hwnd, state = launch_replay(
        steam_exe=args.steam_exe,
        app_id=args.app_id,
        dmo=args.dmo.resolve(),
        runtime_exe=args.runtime_exe,
        minus_count=args.minus_count,
        demo_length=args.demo_length,
        timeout=args.timeout,
        direct_runtime_exe=args.direct_runtime_exe,
        changedir=args.changedir,
    )
    print(f"pid={pid}")
    print(f"hwnd=0x{hwnd:X}")
    print(format_state(state))
    if args.monitor_until_exit:
        sys.stdout.flush()
        while True:
            observed = process_image_path(pid)
            if observed is None:
                break
            if not same_windows_path(
                observed,
                (
                    args.direct_runtime_exe.resolve()
                    if args.direct_runtime_exe is not None
                    else args.runtime_exe
                ),
            ):
                raise RuntimeError("monitored replay process identity changed")
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
