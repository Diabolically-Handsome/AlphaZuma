"""Capture read-only Windows thread runtime metadata for a retail process."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import time
from typing import Any, Iterable, Mapping


THREAD_QUERY_INFORMATION = 0x0040
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
THREAD_QUERY_ACCESS = (
    THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION
)
THREAD_BASIC_INFORMATION_CLASS = 0
THREAD_QUERY_SET_WIN32_START_ADDRESS_CLASS = 9
THREAD_SUSPEND_COUNT_CLASS = 35
STILL_ACTIVE = 259
THREAD_PRIORITY_ERROR_RETURN = 0x7FFFFFFF


class CLIENT_ID(ctypes.Structure):
    _fields_ = [
        ("UniqueProcess", ctypes.c_void_p),
        ("UniqueThread", ctypes.c_void_p),
    ]


class THREAD_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("ExitStatus", ctypes.c_long),
        ("TebBaseAddress", ctypes.c_void_p),
        ("ClientId", CLIENT_ID),
        ("AffinityMask", ctypes.c_size_t),
        ("Priority", ctypes.c_long),
        ("BasePriority", ctypes.c_long),
    ]


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

    kernel32.OpenThread.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetThreadTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetThreadTimes.restype = wintypes.BOOL
    kernel32.QueryThreadCycleTime.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_ulonglong),
    )
    kernel32.QueryThreadCycleTime.restype = wintypes.BOOL
    kernel32.GetThreadPriority.argtypes = (wintypes.HANDLE,)
    kernel32.GetThreadPriority.restype = ctypes.c_int
    kernel32.GetExitCodeThread.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeThread.restype = wintypes.BOOL
    ntdll.NtQueryInformationThread.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    )
    ntdll.NtQueryInformationThread.restype = ctypes.c_long
else:
    kernel32 = None
    ntdll = None


def _filetime_100ns(value: wintypes.FILETIME) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _format_address(value: int | None) -> str | None:
    return f"0x{value:016x}" if value is not None else None


def _nt_query(
    handle: int,
    information_class: int,
    output: Any,
) -> int:
    if ntdll is None:
        raise OSError("Windows thread queries require Windows")
    return int(
        ntdll.NtQueryInformationThread(
            handle,
            information_class,
            ctypes.byref(output),
            ctypes.sizeof(output),
            None,
        )
    )


def _query_thread_runtime(thread_id: int) -> dict[str, Any]:
    if kernel32 is None:
        raise OSError("Windows thread queries require Windows")
    row: dict[str, Any] = {
        "thread_id": thread_id,
        "query_errors": [],
    }
    handle = kernel32.OpenThread(
        THREAD_QUERY_ACCESS,
        False,
        thread_id,
    )
    if not handle:
        row["query_errors"].append(
            {
                "operation": "OpenThread",
                "winerror": ctypes.get_last_error(),
            }
        )
        row["accessible"] = False
        return row
    row["accessible"] = True
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if kernel32.GetThreadTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            row.update(
                {
                    "creation_filetime_100ns": _filetime_100ns(creation),
                    "exit_filetime_100ns": _filetime_100ns(exit_time),
                    "kernel_time_100ns": _filetime_100ns(kernel_time),
                    "user_time_100ns": _filetime_100ns(user_time),
                }
            )
        else:
            row["query_errors"].append(
                {
                    "operation": "GetThreadTimes",
                    "winerror": ctypes.get_last_error(),
                }
            )

        cycle_time = ctypes.c_ulonglong()
        if kernel32.QueryThreadCycleTime(
            handle,
            ctypes.byref(cycle_time),
        ):
            row["cycle_time"] = int(cycle_time.value)
        else:
            row["query_errors"].append(
                {
                    "operation": "QueryThreadCycleTime",
                    "winerror": ctypes.get_last_error(),
                }
            )

        priority = int(kernel32.GetThreadPriority(handle))
        if priority != THREAD_PRIORITY_ERROR_RETURN:
            row["current_priority"] = priority
        else:
            row["query_errors"].append(
                {
                    "operation": "GetThreadPriority",
                    "winerror": ctypes.get_last_error(),
                }
            )

        exit_code = wintypes.DWORD()
        if kernel32.GetExitCodeThread(handle, ctypes.byref(exit_code)):
            row["exit_code"] = int(exit_code.value)
            row["active"] = int(exit_code.value) == STILL_ACTIVE
        else:
            row["query_errors"].append(
                {
                    "operation": "GetExitCodeThread",
                    "winerror": ctypes.get_last_error(),
                }
            )

        start_address = ctypes.c_void_p()
        start_status = _nt_query(
            handle,
            THREAD_QUERY_SET_WIN32_START_ADDRESS_CLASS,
            start_address,
        )
        if start_status >= 0:
            start_value = (
                int(start_address.value)
                if start_address.value is not None
                else 0
            )
            row["start_address"] = start_value
            row["start_address_hex"] = _format_address(start_value)
        else:
            row["query_errors"].append(
                {
                    "operation": "NtQueryInformationThreadStartAddress",
                    "ntstatus": start_status,
                }
            )

        basic = THREAD_BASIC_INFORMATION()
        basic_status = _nt_query(
            handle,
            THREAD_BASIC_INFORMATION_CLASS,
            basic,
        )
        if basic_status >= 0:
            teb = int(basic.TebBaseAddress or 0)
            row.update(
                {
                    "teb_address": teb,
                    "teb_address_hex": _format_address(teb),
                    "native_process_id": int(
                        basic.ClientId.UniqueProcess or 0
                    ),
                    "native_thread_id": int(
                        basic.ClientId.UniqueThread or 0
                    ),
                    "affinity_mask": int(basic.AffinityMask),
                    "native_priority": int(basic.Priority),
                    "native_base_priority": int(basic.BasePriority),
                    "native_exit_status": int(basic.ExitStatus),
                }
            )
        else:
            row["query_errors"].append(
                {
                    "operation": "NtQueryInformationThreadBasic",
                    "ntstatus": basic_status,
                }
            )

        suspend_count = wintypes.ULONG()
        suspend_status = _nt_query(
            handle,
            THREAD_SUSPEND_COUNT_CLASS,
            suspend_count,
        )
        if suspend_status >= 0:
            row["suspend_count"] = int(suspend_count.value)
        else:
            row["suspend_count"] = None
            row["query_errors"].append(
                {
                    "operation": "NtQueryInformationThreadSuspendCount",
                    "ntstatus": suspend_status,
                }
            )
    finally:
        kernel32.CloseHandle(handle)
    return row


def capture_thread_runtime_snapshot(
    *,
    process_id: int,
    main_thread_id: int,
    phase: str,
    thread_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Return a read-only point-in-time thread runtime snapshot."""

    if process_id <= 0 or main_thread_id <= 0 or not phase:
        raise ValueError("invalid thread runtime snapshot target")
    started_ns = time.perf_counter_ns()
    if os.name != "nt":
        raise OSError("Windows thread snapshots require Windows")
    from tools.trace_popcap_demo_commands import _thread_ids_for_pid

    ids = sorted(
        {
            int(thread_id)
            for thread_id in (
                _thread_ids_for_pid(process_id)
                if thread_ids is None
                else thread_ids
            )
            if int(thread_id) > 0
        }
    )
    rows = []
    for thread_id in ids:
        row = _query_thread_runtime(thread_id)
        row["is_main_thread"] = thread_id == main_thread_id
        rows.append(row)
    finished_ns = time.perf_counter_ns()
    query_error_count = sum(
        len(row.get("query_errors", [])) for row in rows
    )
    accessible_count = sum(
        1 for row in rows if row.get("accessible") is True
    )
    return {
        "schema": "zuma-rl.pc-thread-runtime-snapshot",
        "version": 1,
        "classification": "diagnostic-read-only-os-thread-metadata",
        "phase": phase,
        "process_id": process_id,
        "main_thread_id": main_thread_id,
        "captured_perf_counter_ns": finished_ns,
        "capture_duration_ns": finished_ns - started_ns,
        "thread_count": len(rows),
        "accessible_thread_count": accessible_count,
        "query_error_count": query_error_count,
        "process_memory_reads": 0,
        "process_memory_writes": 0,
        "process_context_reads": 0,
        "process_context_writes": 0,
        "threads": rows,
    }


def compute_thread_runtime_delta(
    ready: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two snapshots without assuming thread IDs persist."""

    if (
        ready.get("process_id") != boundary.get("process_id")
        or ready.get("main_thread_id") != boundary.get("main_thread_id")
    ):
        raise ValueError("thread runtime snapshot identity mismatch")
    ready_rows = {
        int(row["thread_id"]): row
        for row in ready.get("threads", [])
        if isinstance(row, Mapping) and isinstance(row.get("thread_id"), int)
    }
    boundary_rows = {
        int(row["thread_id"]): row
        for row in boundary.get("threads", [])
        if isinstance(row, Mapping) and isinstance(row.get("thread_id"), int)
    }
    rows: list[dict[str, Any]] = []
    groups: dict[int | None, dict[str, Any]] = {}
    for thread_id in sorted(set(ready_rows) | set(boundary_rows)):
        before = ready_rows.get(thread_id)
        after = boundary_rows.get(thread_id)
        reference = after if after is not None else before
        assert reference is not None
        start_address = reference.get("start_address")
        if not isinstance(start_address, int):
            start_address = None
        row: dict[str, Any] = {
            "thread_id": thread_id,
            "is_main_thread": bool(reference.get("is_main_thread")),
            "start_address": start_address,
            "start_address_hex": _format_address(start_address),
            "present_at_ready": before is not None,
            "present_at_boundary": after is not None,
        }
        for key in (
            "kernel_time_100ns",
            "user_time_100ns",
            "cycle_time",
        ):
            before_value = before.get(key) if before is not None else None
            after_value = after.get(key) if after is not None else None
            delta_key = f"{key}_delta"
            if (
                isinstance(before_value, int)
                and isinstance(after_value, int)
                and after_value >= before_value
            ):
                row[delta_key] = after_value - before_value
            elif before is None and isinstance(after_value, int):
                # A thread created after the ready snapshot accrued all of
                # its observed runtime inside the measured interval.
                row[delta_key] = after_value
            else:
                row[delta_key] = None
        rows.append(row)

        group = groups.setdefault(
            start_address,
            {
                "start_address": start_address,
                "start_address_hex": _format_address(start_address),
                "thread_ids": [],
                "ready_count": 0,
                "boundary_count": 0,
                "persistent_count": 0,
                "created_count": 0,
                "exited_count": 0,
                "kernel_time_100ns_delta": 0,
                "user_time_100ns_delta": 0,
                "cycle_time_delta": 0,
            },
        )
        group["thread_ids"].append(thread_id)
        group["ready_count"] += int(before is not None)
        group["boundary_count"] += int(after is not None)
        group["persistent_count"] += int(
            before is not None and after is not None
        )
        group["created_count"] += int(
            before is None and after is not None
        )
        group["exited_count"] += int(
            before is not None and after is None
        )
        for key in (
            "kernel_time_100ns_delta",
            "user_time_100ns_delta",
            "cycle_time_delta",
        ):
            value = row.get(key)
            if isinstance(value, int):
                group[key] += value

    group_rows = sorted(
        groups.values(),
        key=lambda row: (
            -int(row["cycle_time_delta"]),
            -int(row["user_time_100ns_delta"]),
            -int(row["kernel_time_100ns_delta"]),
            -1 if row["start_address"] is None else int(row["start_address"]),
        ),
    )
    return {
        "schema": "zuma-rl.pc-thread-runtime-delta",
        "version": 1,
        "process_id": ready["process_id"],
        "main_thread_id": ready["main_thread_id"],
        "ready_thread_count": len(ready_rows),
        "boundary_thread_count": len(boundary_rows),
        "persistent_thread_count": len(set(ready_rows) & set(boundary_rows)),
        "created_thread_count": len(set(boundary_rows) - set(ready_rows)),
        "exited_thread_count": len(set(ready_rows) - set(boundary_rows)),
        "elapsed_perf_counter_ns": int(
            boundary["captured_perf_counter_ns"]
        )
        - int(ready["captured_perf_counter_ns"]),
        "threads": rows,
        "aggregate_by_start_address": group_rows,
    }
