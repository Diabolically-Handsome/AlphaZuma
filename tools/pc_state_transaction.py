"""Capture and restore the exact Zuma users/registry state.

The snapshot format embeds regular-file bytes and raw registry value bytes.
Restore is deliberately scoped to the one known users directory and the two
fixed HKCU registry roots.  It never follows reparse points and verifies the
result by taking a fresh snapshot.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import stat
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_protocol_evidence import (
    REGISTRY_ROOTS,
    PcStateSnapshot,
    SnapshotFile,
    SnapshotRegistryNode,
    SnapshotRegistryValue,
)

DEFAULT_USERS_ROOT = Path(r"C:\ProgramData\Steam\ZumasRevenge\users")

_FILE_ATTRIBUTE_READONLY = 0x1
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_DEVICE = 0x40
_FILE_ATTRIBUTE_NORMAL = 0x80
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_ATTRIBUTE_ENCRYPTED = 0x4000
_UNSAFE_ATTRIBUTES = (
    _FILE_ATTRIBUTE_DEVICE
    | _FILE_ATTRIBUTE_REPARSE_POINT
    | _FILE_ATTRIBUTE_ENCRYPTED
)
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_REGISTRY_VIEW = 0x0200  # KEY_WOW64_32KEY


class StateTransactionError(RuntimeError):
    """A fixed-code state operation failure."""

    def __init__(self, code: str) -> None:
        if not code.replace("_", "").isalnum():
            code = "state_transaction_failed"
        self.code = code
        super().__init__(code)


def _require_windows() -> None:
    if os.name != "nt":
        raise StateTransactionError("windows_required")


def _attributes(info: os.stat_result) -> int:
    value = int(getattr(info, "st_file_attributes", 0))
    if value < 0 or value > 0xFFFFFFFF:
        raise StateTransactionError("users_attributes_invalid")
    return value


def _assert_safe_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise StateTransactionError("users_root_unavailable") from None
    attributes = _attributes(info)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or attributes & _UNSAFE_ATTRIBUTES
    ):
        raise StateTransactionError("users_tree_unsafe")


def capture_users_tree(root: Path = DEFAULT_USERS_ROOT) -> tuple[SnapshotFile, ...]:
    """Read a complete, no-follow regular-file inventory."""

    _require_windows()
    _assert_safe_directory(root)
    files: list[SnapshotFile] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise StateTransactionError("users_tree_read_failed") from None
        for entry in entries:
            path = Path(entry.path)
            try:
                # On Windows, ``DirEntry.stat()`` can report ``st_ino == 0``
                # while ``Path.stat()`` for the same file reports its real
                # file index.  Use the same stat path on both sides of the
                # read so the change guard compares like with like.
                before = path.stat(follow_symlinks=False)
            except OSError:
                raise StateTransactionError("users_tree_read_failed") from None
            attributes = _attributes(before)
            if (
                stat.S_ISLNK(before.st_mode)
                or attributes & _UNSAFE_ATTRIBUTES
            ):
                raise StateTransactionError("users_tree_unsafe")
            if stat.S_ISDIR(before.st_mode):
                if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
                    raise StateTransactionError("users_tree_unsafe")
                pending.append(path)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise StateTransactionError("users_tree_unsafe")
            try:
                payload = path.read_bytes()
                after = path.stat(follow_symlinks=False)
            except OSError:
                raise StateTransactionError("users_tree_read_failed") from None
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ino != after.st_ino
                or len(payload) != before.st_size
            ):
                raise StateTransactionError("users_tree_changed") from None
            relative = path.relative_to(root).as_posix()
            files.append(
                SnapshotFile(
                    path=relative,
                    data=payload,
                    windows_attributes=attributes,
                )
            )
    return tuple(
        sorted(files, key=lambda item: (item.path.casefold(), item.path))
    )


def _raw_registry_value(handle: int, name: str) -> tuple[int, bytes]:
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.RegQueryValueExW.argtypes = (
        wintypes.HKEY,
        wintypes.LPCWSTR,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPBYTE,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.RegQueryValueExW.restype = wintypes.LONG
    value_type = wintypes.DWORD()
    size = wintypes.DWORD()
    result = advapi32.RegQueryValueExW(
        handle,
        name,
        None,
        ctypes.byref(value_type),
        None,
        ctypes.byref(size),
    )
    if result != 0:
        raise StateTransactionError("registry_value_read_failed")
    buffer = (ctypes.c_ubyte * max(1, size.value))()
    result = advapi32.RegQueryValueExW(
        handle,
        name,
        None,
        ctypes.byref(value_type),
        buffer,
        ctypes.byref(size),
    )
    if result != 0:
        raise StateTransactionError("registry_value_read_failed")
    return int(value_type.value), bytes(buffer[: size.value])


def _capture_registry_root(root: str) -> tuple[SnapshotRegistryNode, ...]:
    import winreg

    prefix = "HKCU\\"
    if not root.startswith(prefix):
        raise StateTransactionError("registry_scope_invalid")
    root_path = root[len(prefix) :]
    access = winreg.KEY_READ | _REGISTRY_VIEW
    try:
        base = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            root_path,
            0,
            access,
        )
    except FileNotFoundError:
        return (
            SnapshotRegistryNode(
                root=root,
                subkey="",
                exists=False,
                values=(),
            ),
        )
    except OSError:
        raise StateTransactionError("registry_root_read_failed") from None

    nodes: list[SnapshotRegistryNode] = []
    pending: list[tuple[str, object]] = [("", base)]
    try:
        while pending:
            relative, key = pending.pop()
            try:
                subkey_count, value_count, _ = winreg.QueryInfoKey(key)
                value_names = [
                    winreg.EnumValue(key, index)[0]
                    for index in range(value_count)
                ]
                subkey_names = [
                    winreg.EnumKey(key, index)
                    for index in range(subkey_count)
                ]
            except OSError:
                raise StateTransactionError(
                    "registry_root_read_failed"
                ) from None
            captured_values: list[SnapshotRegistryValue] = []
            for name in value_names:
                value_type, value_data = _raw_registry_value(
                    int(key),
                    name,
                )
                captured_values.append(
                    SnapshotRegistryValue(
                        name=name,
                        value_type=value_type,
                        data=value_data,
                    )
                )
            values = tuple(
                sorted(
                    captured_values,
                    key=lambda item: (item.name.casefold(), item.name),
                )
            )
            nodes.append(
                SnapshotRegistryNode(
                    root=root,
                    subkey=relative.replace("\\", "/"),
                    exists=True,
                    values=values,
                )
            )
            for subkey_name in reversed(
                sorted(subkey_names, key=lambda item: (item.casefold(), item))
            ):
                child_relative = (
                    subkey_name
                    if not relative
                    else f"{relative}\\{subkey_name}"
                )
                try:
                    child = winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER,
                        f"{root_path}\\{child_relative}",
                        0,
                        access,
                    )
                except OSError:
                    raise StateTransactionError(
                        "registry_root_read_failed"
                    ) from None
                pending.append((child_relative, child))
            if key is not base:
                key.Close()
    finally:
        base.Close()
        for _, key in pending:
            try:
                key.Close()
            except OSError:
                pass
    root_order = {item: index for index, item in enumerate(REGISTRY_ROOTS)}
    return tuple(
        sorted(
            nodes,
            key=lambda item: (
                root_order[item.root],
                item.subkey.casefold(),
                item.subkey,
            ),
        )
    )


def capture_registry_tree() -> tuple[SnapshotRegistryNode, ...]:
    _require_windows()
    nodes = tuple(
        node
        for root in REGISTRY_ROOTS
        for node in _capture_registry_root(root)
    )
    return nodes


def capture_state(
    *,
    session_nonce: str,
    phase: str,
    users_root: Path = DEFAULT_USERS_ROOT,
    captured_perf_counter_ns: int | None = None,
) -> PcStateSnapshot:
    return PcStateSnapshot(
        session_nonce=session_nonce,
        phase=phase,
        captured_perf_counter_ns=(
            time.perf_counter_ns()
            if captured_perf_counter_ns is None
            else captured_perf_counter_ns
        ),
        files=capture_users_tree(users_root),
        registry_nodes=capture_registry_tree(),
    )


def overlay_snapshot_files(
    snapshot: PcStateSnapshot,
    replacements: Iterable[tuple[str, Path]],
    *,
    phase: str | None = None,
    captured_perf_counter_ns: int | None = None,
) -> PcStateSnapshot:
    """Return a state bundle with a closed set of explicit file overlays."""

    by_path = {item.path.casefold(): item for item in snapshot.files}
    for relative, source in replacements:
        key = relative.casefold()
        if key not in by_path:
            raise StateTransactionError("overlay_target_missing")
        try:
            payload = source.read_bytes()
        except OSError:
            raise StateTransactionError("overlay_source_read_failed") from None
        original = by_path[key]
        by_path[key] = SnapshotFile(
            path=original.path,
            data=payload,
            windows_attributes=original.windows_attributes,
        )
    files = tuple(
        sorted(
            by_path.values(),
            key=lambda item: (item.path.casefold(), item.path),
        )
    )
    return PcStateSnapshot(
        session_nonce=snapshot.session_nonce,
        phase=snapshot.phase if phase is None else phase,
        captured_perf_counter_ns=(
            snapshot.captured_perf_counter_ns
            if captured_perf_counter_ns is None
            else captured_perf_counter_ns
        ),
        files=files,
        registry_nodes=snapshot.registry_nodes,
    )


def overlay_snapshot_registry_values(
    snapshot: PcStateSnapshot,
    replacements: Iterable[tuple[str, str, str, int, bytes]],
    *,
    phase: str | None = None,
    captured_perf_counter_ns: int | None = None,
) -> PcStateSnapshot:
    """Return a state bundle with explicit existing registry-value overlays."""

    nodes = {
        (item.root.casefold(), item.subkey.casefold()): item
        for item in snapshot.registry_nodes
    }
    for root, subkey, value_name, value_type, payload in replacements:
        key = (root.casefold(), subkey.casefold())
        node = nodes.get(key)
        if node is None or not node.exists:
            raise StateTransactionError("overlay_registry_node_missing")
        values = {item.name.casefold(): item for item in node.values}
        value_key = value_name.casefold()
        if value_key not in values:
            raise StateTransactionError("overlay_registry_value_missing")
        original = values[value_key]
        values[value_key] = SnapshotRegistryValue(
            name=original.name,
            value_type=value_type,
            data=payload,
        )
        nodes[key] = SnapshotRegistryNode(
            root=node.root,
            subkey=node.subkey,
            exists=True,
            values=tuple(
                sorted(
                    values.values(),
                    key=lambda item: (
                        item.name.casefold(),
                        item.name,
                    ),
                )
            ),
        )
    root_order = {root: index for index, root in enumerate(REGISTRY_ROOTS)}
    ordered_nodes = tuple(
        sorted(
            nodes.values(),
            key=lambda item: (
                root_order[item.root],
                item.subkey.casefold(),
                item.subkey,
            ),
        )
    )
    return PcStateSnapshot(
        session_nonce=snapshot.session_nonce,
        phase=snapshot.phase if phase is None else phase,
        captured_perf_counter_ns=(
            snapshot.captured_perf_counter_ns
            if captured_perf_counter_ns is None
            else captured_perf_counter_ns
        ),
        files=snapshot.files,
        registry_nodes=ordered_nodes,
    )


def _delete_users_tree_contents(root: Path) -> None:
    _assert_safe_directory(root)
    files: list[tuple[Path, int]] = []
    directories: list[tuple[Path, int]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise StateTransactionError("users_tree_restore_failed") from None
        for entry in entries:
            path = Path(entry.path)
            try:
                info = path.stat(follow_symlinks=False)
            except OSError:
                raise StateTransactionError(
                    "users_tree_restore_failed"
                ) from None
            attributes = _attributes(info)
            if stat.S_ISLNK(info.st_mode) or attributes & _UNSAFE_ATTRIBUTES:
                raise StateTransactionError("users_tree_unsafe")
            if stat.S_ISDIR(info.st_mode):
                directories.append((path, attributes))
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.append((path, attributes))
            else:
                raise StateTransactionError("users_tree_unsafe")
    try:
        for path, attributes in files:
            if attributes & _FILE_ATTRIBUTE_READONLY:
                writable = attributes & ~_FILE_ATTRIBUTE_READONLY
                _set_windows_attributes(
                    path,
                    writable or _FILE_ATTRIBUTE_NORMAL,
                )
            path.unlink()
        for path, attributes in sorted(
            directories,
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if attributes & _FILE_ATTRIBUTE_READONLY:
                writable = attributes & ~_FILE_ATTRIBUTE_READONLY
                _set_windows_attributes(
                    path,
                    writable or _FILE_ATTRIBUTE_DIRECTORY,
                )
            path.rmdir()
    except OSError:
        raise StateTransactionError("users_tree_restore_failed") from None


def _set_windows_attributes(path: Path, attributes: int) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileAttributesW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
    )
    kernel32.SetFileAttributesW.restype = wintypes.BOOL
    if not kernel32.SetFileAttributesW(str(path), attributes):
        raise StateTransactionError("users_attributes_restore_failed")


def restore_users_tree(
    snapshot: PcStateSnapshot,
    root: Path = DEFAULT_USERS_ROOT,
) -> None:
    _require_windows()
    if root.resolve() != DEFAULT_USERS_ROOT.resolve():
        raise StateTransactionError("restore_users_scope_invalid")
    _delete_users_tree_contents(root)
    try:
        for item in snapshot.files:
            target = root / Path(item.path)
            if not target.resolve().is_relative_to(root.resolve()):
                raise StateTransactionError("restore_users_scope_invalid")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.zuma-rl-{snapshot.session_nonce}.tmp"
            )
            if temporary.exists():
                raise StateTransactionError("restore_temp_conflict")
            with temporary.open("xb") as stream:
                if stream.write(item.data) != len(item.data):
                    raise StateTransactionError(
                        "users_tree_restore_failed"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _set_windows_attributes(target, item.windows_attributes)
    except StateTransactionError:
        raise
    except OSError:
        raise StateTransactionError("users_tree_restore_failed") from None


def _delete_registry_root(root: str) -> None:
    import winreg

    root_path = root.removeprefix("HKCU\\")

    def delete_children(relative: str) -> None:
        access = winreg.KEY_READ | winreg.KEY_WRITE | _REGISTRY_VIEW
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                relative,
                0,
                access,
            )
        except FileNotFoundError:
            return
        try:
            while True:
                try:
                    child = winreg.EnumKey(key, 0)
                except OSError:
                    break
                delete_children(f"{relative}\\{child}")
        finally:
            key.Close()
        try:
            winreg.DeleteKeyEx(
                winreg.HKEY_CURRENT_USER,
                relative,
                _REGISTRY_VIEW,
                0,
            )
        except AttributeError:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, relative)

    try:
        delete_children(root_path)
    except FileNotFoundError:
        return
    except OSError as error:
        if getattr(error, "winerror", None) not in {
            _ERROR_FILE_NOT_FOUND,
            _ERROR_PATH_NOT_FOUND,
        }:
            raise StateTransactionError(
                "registry_restore_failed"
            ) from None


def _set_raw_registry_value(
    handle: int,
    value: SnapshotRegistryValue,
) -> None:
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.RegSetValueExW.argtypes = (
        wintypes.HKEY,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPBYTE,
        wintypes.DWORD,
    )
    advapi32.RegSetValueExW.restype = wintypes.LONG
    buffer = (ctypes.c_ubyte * max(1, len(value.data)))()
    if value.data:
        ctypes.memmove(buffer, value.data, len(value.data))
    result = advapi32.RegSetValueExW(
        handle,
        value.name,
        0,
        value.value_type,
        buffer,
        len(value.data),
    )
    if result != 0:
        raise StateTransactionError("registry_restore_failed")


def restore_registry_tree(snapshot: PcStateSnapshot) -> None:
    _require_windows()
    import winreg

    for root in REGISTRY_ROOTS:
        _delete_registry_root(root)
        nodes = [
            node for node in snapshot.registry_nodes if node.root == root
        ]
        base = next(node for node in nodes if node.subkey == "")
        if not base.exists:
            continue
        root_path = root.removeprefix("HKCU\\")
        for node in nodes:
            relative = node.subkey.replace("/", "\\")
            path = root_path if not relative else f"{root_path}\\{relative}"
            try:
                key = winreg.CreateKeyEx(
                    winreg.HKEY_CURRENT_USER,
                    path,
                    0,
                    winreg.KEY_WRITE | _REGISTRY_VIEW,
                )
            except OSError:
                raise StateTransactionError(
                    "registry_restore_failed"
                ) from None
            try:
                for value in node.values:
                    _set_raw_registry_value(int(key), value)
                winreg.FlushKey(key)
            finally:
                key.Close()


def restore_state(
    snapshot: PcStateSnapshot,
    *,
    users_root: Path = DEFAULT_USERS_ROOT,
) -> PcStateSnapshot:
    """Restore and independently recapture the scoped composite state."""

    restore_users_tree(snapshot, users_root)
    restore_registry_tree(snapshot)
    verified = capture_state(
        session_nonce=snapshot.session_nonce,
        phase=snapshot.phase,
        users_root=users_root,
    )
    if verified.state_root != snapshot.state_root:
        raise StateTransactionError("restored_state_root_mismatch")
    return verified


def write_snapshot_exclusive(
    snapshot: PcStateSnapshot,
    path: Path,
) -> None:
    if not path.is_absolute() or path.exists():
        raise StateTransactionError("snapshot_output_invalid")
    part = path.with_name(path.name + ".part")
    if part.exists() or not path.parent.is_dir():
        raise StateTransactionError("snapshot_output_invalid")
    payload = snapshot.to_json().encode("ascii")
    try:
        with part.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise StateTransactionError("snapshot_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(part, path)
        part.unlink()
    except StateTransactionError:
        raise
    except (FileExistsError, OSError):
        raise StateTransactionError("snapshot_publish_failed") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or restore the scoped Zuma users/registry state."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--session-nonce", required=True)
    snapshot.add_argument("--phase", required=True)
    snapshot.add_argument("--output", required=True, type=Path)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--snapshot", required=True, type=Path)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        snapshot = capture_state(
            session_nonce=args.session_nonce,
            phase=args.phase,
        )
        write_snapshot_exclusive(snapshot, args.output)
        print(snapshot.state_root)
        return 0
    snapshot = PcStateSnapshot.read(args.snapshot)
    verified = restore_state(snapshot)
    print(verified.state_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except StateTransactionError as error:
        print(f"state transaction error: {error.code}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("state transaction error: interrupted", file=sys.stderr)
        return 130
    except Exception:
        print(
            "state transaction error: unexpected_failure",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
