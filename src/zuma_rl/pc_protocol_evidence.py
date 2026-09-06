"""Strict raw-evidence formats for PC golden protocol v4.

The manifest binds artifact bytes, but protocol acceptance must be recomputed
from those bytes.  This module therefore defines canonical state bundles,
an append-only hash-chain journal, a compact raw process timeline, and a
strict reader for the native DXGI acquisition metadata.  None of the readers
accept producer-authored ``restored`` or ``deterministic`` booleans.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from zuma_rl.pc_golden import (
    PcGoldenValidationError,
    _canonical_json_text,
    _check_keys,
    _json_loads_strict,
    _mapping,
    _nonempty_string,
    _sequence,
    _sha256,
    _strict_bool,
    _strict_int,
    canonical_sha256,
)

STATE_SNAPSHOT_SCHEMA = "zuma-rl.pc-state-snapshot"
STATE_SNAPSHOT_VERSION = 1
SAVE_JOURNAL_SCHEMA = "zuma-rl.pc-save-journal"
SAVE_JOURNAL_VERSION = 1
PROCESS_TIMELINE_MAGIC = b"ZRPCPRC1"
PROCESS_TIMELINE_VERSION = 1
FRAMEWORK_UPDATE_SCHEMA = "zuma-rl.pc-framework-update-map"
FRAMEWORK_UPDATE_VERSION = 1
DXGI_CAPTURE_SCHEMA = "zuma-rl.dxgi-bgra-capture"
DXGI_CAPTURE_VERSION = 2

REGISTRY_ROOTS = (
    r"HKCU\Software\SteamPopCap\ZumasRevenge",
    r"HKCU\Software\PopCap\ZumasRevenge",
)

_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}")
_PHASE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,63}")
_RUN_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_MAX_SNAPSHOT_FILES = 100_000
_MAX_SNAPSHOT_FILE_BYTES = 128 * 1024**2
_MAX_SNAPSHOT_TOTAL_BYTES = 512 * 1024**2
_MAX_REGISTRY_NODES = 10_000
_MAX_REGISTRY_VALUES = 100_000
_MAX_REGISTRY_VALUE_BYTES = 16 * 1024**2
_MAX_JOURNAL_RECORDS = 10_000
_MAX_PROCESS_EVENTS = 10_000
_MAX_FRAMEWORK_UPDATE_RECORDS = 100_000

_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_DEVICE = 0x40
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_ATTRIBUTE_ENCRYPTED = 0x4000
_FORBIDDEN_FILE_ATTRIBUTES = (
    _FILE_ATTRIBUTE_DIRECTORY
    | _FILE_ATTRIBUTE_DEVICE
    | _FILE_ATTRIBUTE_REPARSE_POINT
    | _FILE_ATTRIBUTE_ENCRYPTED
)

_PROCESS_HEADER = struct.Struct("<8sI16sI")
_PROCESS_RECORD = struct.Struct("<IB3xIQQI32s")
_PROCESS_START = 1
_PROCESS_STOP = 2
_NO_EXIT_CODE = 0xFFFFFFFF

_DXGI_METADATA_KEYS = {
    "schema",
    "version",
    "status",
    "target_process",
    "device_index",
    "output_index",
    "global_region",
    "output_local_region",
    "width",
    "height",
    "pixel_format",
    "bytes_per_pixel",
    "frame_budget_fps",
    "capture_mode",
    "capture_storage_mode",
    "capture_surface",
    "foreground_geometry_guard",
    "topmost_or_injected_overlay_detection",
    "requires_full_frame_visual_review",
    "requested_duration_seconds",
    "capture_start_perf_counter_ns",
    "capture_end_perf_counter_ns",
    "capture_elapsed_seconds",
    "estimated_raw_budget_bytes",
    "frame_budget",
    "frame_count",
    "warmup_baseline_present_ticks",
    "first_present_ticks",
    "last_present_ticks",
    "present_span_ticks",
    "present_span_seconds",
    "observed_mean_present_fps",
    "qpc_frequency",
    "missed_presentations",
    "idle_poll_count",
    "pointer_only_update_count",
    "warmup_idle_poll_count",
    "warmup_pointer_only_update_count",
    "runtime_versions",
    "runtime_versions_verified",
    "source_identity",
    "target_identity",
    "raw_bytes",
    "raw_sha256",
    "frames_csv_bytes",
    "frames_csv_rows",
    "frames_csv_sha256",
    "aggregate_pixel_sha256",
    "frames_csv",
    "raw_frames",
}
_DXGI_TARGET_IDENTITY_KEYS = {
    "process_id",
    "process_creation_filetime_100ns",
    "executable_bytes",
    "executable_sha256",
    "executable_hash_provenance",
    "window_handle_hex",
    "window_client_region",
}


def _nonce(value: Any, name: str = "session_nonce") -> str:
    if not isinstance(value, str) or _NONCE_PATTERN.fullmatch(value) is None:
        raise PcGoldenValidationError(
            f"{name} must contain exactly 32 lowercase hexadecimal digits"
        )
    return value


def _phase(value: Any, name: str = "phase") -> str:
    if not isinstance(value, str) or _PHASE_PATTERN.fullmatch(value) is None:
        raise PcGoldenValidationError(
            f"{name} must use canonical lowercase ASCII phase syntax"
        )
    return value


def _run_id(value: Any, name: str = "run_id") -> str:
    if not isinstance(value, str) or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise PcGoldenValidationError(
            f"{name} must use canonical lowercase ASCII run syntax"
        )
    return value


def _decode_base64(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise PcGoldenValidationError(f"{name} must be a base64 string")
    try:
        result = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise PcGoldenValidationError(
            f"{name} is not canonical base64"
        ) from None
    if base64.b64encode(result).decode("ascii") != value:
        raise PcGoldenValidationError(f"{name} is not canonical base64")
    return result


def _encode_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _canonical_json_file(value: Mapping[str, Any]) -> str:
    return _canonical_json_text(value) + "\n"


def _line_sha256(value: Mapping[str, Any]) -> str:
    payload = _canonical_json_file(value).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _relative_posix_path(value: Any, name: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise PcGoldenValidationError(f"{name} must be a string")
    if allow_empty and value == "":
        return value
    if not value:
        raise PcGoldenValidationError(f"{name} must not be empty")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or "." in pure.parts
    ):
        raise PcGoldenValidationError(
            f"{name} must be a normalized relative POSIX path"
        )
    return value


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One complete regular-file member of the users tree."""

    path: str
    data: bytes
    windows_attributes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _relative_posix_path(
                self.path,
                "snapshot file path",
                allow_empty=False,
            ),
        )
        if not isinstance(self.data, bytes):
            raise PcGoldenValidationError("snapshot file data must be bytes")
        if len(self.data) > _MAX_SNAPSHOT_FILE_BYTES:
            raise PcGoldenValidationError(
                "snapshot file exceeds the per-file byte limit"
            )
        attributes = _strict_int(
            self.windows_attributes,
            "snapshot file windows_attributes",
            minimum=0,
        )
        if attributes > 0xFFFFFFFF:
            raise PcGoldenValidationError(
                "snapshot file windows_attributes exceeds uint32"
            )
        if attributes & _FORBIDDEN_FILE_ATTRIBUTES:
            raise PcGoldenValidationError(
                "snapshot file cannot be a directory, device, reparse point, "
                "or encrypted member"
            )
        object.__setattr__(self, "windows_attributes", attributes)

    @property
    def sha256(self) -> str:
        return f"sha256:{hashlib.sha256(self.data).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "windows_attributes": self.windows_attributes,
            "data_base64": _encode_base64(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SnapshotFile":
        data = _mapping(value, "snapshot file")
        _check_keys(
            data,
            required={"path", "windows_attributes", "data_base64"},
            name="snapshot file",
        )
        return cls(
            path=data["path"],
            windows_attributes=data["windows_attributes"],
            data=_decode_base64(
                data["data_base64"],
                "snapshot file data_base64",
            ),
        )


@dataclass(frozen=True, slots=True)
class SnapshotRegistryValue:
    """One raw Windows registry value."""

    name: str
    value_type: int
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise PcGoldenValidationError(
                "snapshot registry value name must be a string"
            )
        if "\x00" in self.name:
            raise PcGoldenValidationError(
                "snapshot registry value name must not contain NUL"
            )
        value_type = _strict_int(
            self.value_type,
            "snapshot registry value_type",
            minimum=0,
        )
        if value_type > 11 or value_type == 10:
            raise PcGoldenValidationError(
                "snapshot registry value_type is unsupported"
            )
        if not isinstance(self.data, bytes):
            raise PcGoldenValidationError(
                "snapshot registry data must be bytes"
            )
        if len(self.data) > _MAX_REGISTRY_VALUE_BYTES:
            raise PcGoldenValidationError(
                "snapshot registry value exceeds the byte limit"
            )
        if value_type == 4 and len(self.data) != 4:
            raise PcGoldenValidationError("REG_DWORD data must be four bytes")
        if value_type == 11 and len(self.data) != 8:
            raise PcGoldenValidationError("REG_QWORD data must be eight bytes")
        object.__setattr__(self, "value_type", value_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "data_base64": _encode_base64(self.data),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SnapshotRegistryValue":
        data = _mapping(value, "snapshot registry value")
        _check_keys(
            data,
            required={"name", "value_type", "data_base64"},
            name="snapshot registry value",
        )
        return cls(
            name=data["name"],
            value_type=data["value_type"],
            data=_decode_base64(
                data["data_base64"],
                "snapshot registry value data_base64",
            ),
        )


@dataclass(frozen=True, slots=True)
class SnapshotRegistryNode:
    """One key beneath one of the two fixed Zuma registry roots."""

    root: str
    subkey: str
    exists: bool
    values: tuple[SnapshotRegistryValue, ...]

    def __post_init__(self) -> None:
        if self.root not in REGISTRY_ROOTS:
            raise PcGoldenValidationError(
                "snapshot registry root is outside the fixed Zuma scope"
            )
        object.__setattr__(
            self,
            "subkey",
            _relative_posix_path(
                self.subkey,
                "snapshot registry subkey",
                allow_empty=True,
            ),
        )
        exists = _strict_bool(self.exists, "snapshot registry exists")
        values = tuple(self.values)
        if any(
            not isinstance(item, SnapshotRegistryValue) for item in values
        ):
            raise PcGoldenValidationError(
                "snapshot registry node values must contain registry values"
            )
        ordered = tuple(
            sorted(values, key=lambda item: (item.name.casefold(), item.name))
        )
        if values != ordered:
            raise PcGoldenValidationError(
                "snapshot registry values must use canonical name order"
            )
        folded_names = [item.name.casefold() for item in values]
        if len(set(folded_names)) != len(folded_names):
            raise PcGoldenValidationError(
                "snapshot registry values must not duplicate Windows names"
            )
        if not exists and values:
            raise PcGoldenValidationError(
                "a missing registry node cannot contain values"
            )
        if not exists and self.subkey:
            raise PcGoldenValidationError(
                "only a fixed registry root may be represented as missing"
            )
        object.__setattr__(self, "exists", exists)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "subkey": self.subkey,
            "exists": self.exists,
            "values": [item.to_dict() for item in self.values],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SnapshotRegistryNode":
        data = _mapping(value, "snapshot registry node")
        _check_keys(
            data,
            required={"root", "subkey", "exists", "values"},
            name="snapshot registry node",
        )
        return cls(
            root=data["root"],
            subkey=data["subkey"],
            exists=data["exists"],
            values=tuple(
                SnapshotRegistryValue.from_dict(
                    _mapping(item, "snapshot registry value")
                )
                for item in _sequence(
                    data["values"],
                    "snapshot registry values",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PcStateSnapshot:
    """Canonical complete users-tree and fixed-registry state bundle."""

    session_nonce: str
    phase: str
    captured_perf_counter_ns: int
    files: tuple[SnapshotFile, ...]
    registry_nodes: tuple[SnapshotRegistryNode, ...]
    state_root: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_nonce", _nonce(self.session_nonce))
        object.__setattr__(self, "phase", _phase(self.phase))
        object.__setattr__(
            self,
            "captured_perf_counter_ns",
            _strict_int(
                self.captured_perf_counter_ns,
                "snapshot captured_perf_counter_ns",
                minimum=1,
            ),
        )
        files = tuple(self.files)
        if len(files) > _MAX_SNAPSHOT_FILES:
            raise PcGoldenValidationError(
                "snapshot users tree exceeds the file-count limit"
            )
        if any(not isinstance(item, SnapshotFile) for item in files):
            raise PcGoldenValidationError(
                "snapshot files must contain SnapshotFile objects"
            )
        ordered_files = tuple(
            sorted(files, key=lambda item: (item.path.casefold(), item.path))
        )
        if files != ordered_files:
            raise PcGoldenValidationError(
                "snapshot files must use canonical Windows path order"
            )
        folded_paths = [item.path.casefold() for item in files]
        if len(set(folded_paths)) != len(folded_paths):
            raise PcGoldenValidationError(
                "snapshot files must not duplicate Windows paths"
            )
        if sum(len(item.data) for item in files) > _MAX_SNAPSHOT_TOTAL_BYTES:
            raise PcGoldenValidationError(
                "snapshot users tree exceeds the total byte limit"
            )

        nodes = tuple(self.registry_nodes)
        if len(nodes) > _MAX_REGISTRY_NODES:
            raise PcGoldenValidationError(
                "snapshot registry tree exceeds the node-count limit"
            )
        if any(not isinstance(item, SnapshotRegistryNode) for item in nodes):
            raise PcGoldenValidationError(
                "snapshot registry_nodes must contain registry nodes"
            )
        root_order = {root: index for index, root in enumerate(REGISTRY_ROOTS)}
        ordered_nodes = tuple(
            sorted(
                nodes,
                key=lambda item: (
                    root_order[item.root],
                    item.subkey.casefold(),
                    item.subkey,
                ),
            )
        )
        if nodes != ordered_nodes:
            raise PcGoldenValidationError(
                "snapshot registry nodes must use canonical root/subkey order"
            )
        node_keys = [
            (item.root.casefold(), item.subkey.casefold()) for item in nodes
        ]
        if len(set(node_keys)) != len(node_keys):
            raise PcGoldenValidationError(
                "snapshot registry nodes must not duplicate Windows keys"
            )
        if sum(len(item.values) for item in nodes) > _MAX_REGISTRY_VALUES:
            raise PcGoldenValidationError(
                "snapshot registry tree exceeds the value-count limit"
            )
        for root in REGISTRY_ROOTS:
            base = [
                item
                for item in nodes
                if item.root == root and item.subkey == ""
            ]
            if len(base) != 1:
                raise PcGoldenValidationError(
                    "snapshot must contain exactly one base node for each "
                    "fixed registry root"
                )
            if not base[0].exists and any(
                item.root == root and item.subkey for item in nodes
            ):
                raise PcGoldenValidationError(
                    "a missing registry root cannot contain child nodes"
                )
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "registry_nodes", nodes)
        object.__setattr__(
            self,
            "state_root",
            canonical_sha256(self._state_payload()),
        )

    def _state_payload(self) -> dict[str, Any]:
        return {
            "users_files": [item.to_dict() for item in self.files],
            "registry_nodes": [
                item.to_dict() for item in self.registry_nodes
            ],
        }

    def projected_state_root(
        self,
        excluded_registry_values: Sequence[
            tuple[str, str, str]
        ],
    ) -> str:
        """Hash complete state after removing an explicit value allowlist."""

        exclusions = tuple(excluded_registry_values)
        if not exclusions:
            raise PcGoldenValidationError(
                "state projection requires excluded registry values"
            )
        folded: dict[
            tuple[str, str, str],
            tuple[str, str, str],
        ] = {}
        for item in exclusions:
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or not all(isinstance(part, str) for part in item)
            ):
                raise PcGoldenValidationError(
                    "state projection exclusions must be registry triples"
                )
            root, subkey, name = item
            if root not in REGISTRY_ROOTS or not name or "\0" in name:
                raise PcGoldenValidationError(
                    "state projection exclusion is outside the fixed scope"
                )
            key = (root.casefold(), subkey.casefold(), name.casefold())
            if key in folded:
                raise PcGoldenValidationError(
                    "state projection exclusions must be unique"
                )
            folded[key] = item

        found: set[tuple[str, str, str]] = set()
        nodes: list[dict[str, Any]] = []
        for node in self.registry_nodes:
            values: list[dict[str, Any]] = []
            for value in node.values:
                key = (
                    node.root.casefold(),
                    node.subkey.casefold(),
                    value.name.casefold(),
                )
                if key in folded:
                    found.add(key)
                else:
                    values.append(value.to_dict())
            encoded = node.to_dict()
            encoded["values"] = values
            nodes.append(encoded)
        if found != set(folded):
            raise PcGoldenValidationError(
                "state projection exclusion is absent from the snapshot"
            )
        return canonical_sha256(
            {
                "users_files": [
                    item.to_dict() for item in self.files
                ],
                "registry_nodes": nodes,
            }
        )

    @property
    def users_bytes(self) -> int:
        return sum(len(item.data) for item in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SNAPSHOT_SCHEMA,
            "version": STATE_SNAPSHOT_VERSION,
            "session_nonce": self.session_nonce,
            "phase": self.phase,
            "captured_perf_counter_ns": self.captured_perf_counter_ns,
            **self._state_payload(),
            "state_root": self.state_root,
        }

    def to_json(self) -> str:
        return _canonical_json_file(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PcStateSnapshot":
        data = _mapping(value, "PC state snapshot")
        _check_keys(
            data,
            required={
                "schema",
                "version",
                "session_nonce",
                "phase",
                "captured_perf_counter_ns",
                "users_files",
                "registry_nodes",
                "state_root",
            },
            name="PC state snapshot",
        )
        if (
            data["schema"] != STATE_SNAPSHOT_SCHEMA
            or _strict_int(
                data["version"],
                "snapshot.version",
            )
            != STATE_SNAPSHOT_VERSION
        ):
            raise PcGoldenValidationError(
                "unsupported PC state snapshot schema/version"
            )
        result = cls(
            session_nonce=data["session_nonce"],
            phase=data["phase"],
            captured_perf_counter_ns=data["captured_perf_counter_ns"],
            files=tuple(
                SnapshotFile.from_dict(_mapping(item, "snapshot file"))
                for item in _sequence(
                    data["users_files"],
                    "snapshot users_files",
                )
            ),
            registry_nodes=tuple(
                SnapshotRegistryNode.from_dict(
                    _mapping(item, "snapshot registry node")
                )
                for item in _sequence(
                    data["registry_nodes"],
                    "snapshot registry_nodes",
                )
            ),
        )
        if _sha256(data["state_root"], "snapshot state_root") != result.state_root:
            raise PcGoldenValidationError(
                "snapshot state_root does not match canonical state bytes"
            )
        return result

    @classmethod
    def from_json(cls, text: str) -> "PcStateSnapshot":
        if not isinstance(text, str):
            raise TypeError("snapshot text must be a string")
        result = cls.from_dict(
            _mapping(
                _json_loads_strict(text, "PC state snapshot"),
                "PC state snapshot",
            )
        )
        if text != result.to_json():
            raise PcGoldenValidationError(
                "PC state snapshot JSON is not canonical"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcStateSnapshot":
        try:
            text = Path(path).read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "PC state snapshot could not be read as ASCII"
            ) from error
        return cls.from_json(text)


@dataclass(frozen=True, slots=True)
class SaveJournalRecord:
    """One append-only phase transition in the save transaction."""

    sequence: int
    previous_record_sha256: str
    phase: str
    snapshot_artifact: str
    snapshot_state_root: str
    snapshot_perf_counter_ns: int
    process_id: int | None
    process_creation_filetime_100ns: int | None
    exit_code: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _strict_int(
                self.sequence,
                "save journal sequence",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "previous_record_sha256",
            _sha256(
                self.previous_record_sha256,
                "save journal previous_record_sha256",
            ),
        )
        object.__setattr__(self, "phase", _phase(self.phase))
        object.__setattr__(
            self,
            "snapshot_artifact",
            _nonempty_string(
                self.snapshot_artifact,
                "save journal snapshot_artifact",
            ),
        )
        object.__setattr__(
            self,
            "snapshot_state_root",
            _sha256(
                self.snapshot_state_root,
                "save journal snapshot_state_root",
            ),
        )
        object.__setattr__(
            self,
            "snapshot_perf_counter_ns",
            _strict_int(
                self.snapshot_perf_counter_ns,
                "save journal snapshot_perf_counter_ns",
                minimum=1,
            ),
        )
        identity_is_partial = (
            (self.process_id is None)
            != (self.process_creation_filetime_100ns is None)
        )
        if identity_is_partial:
            raise PcGoldenValidationError(
                "save journal process identity must be wholly present or null"
            )
        if self.process_id is not None:
            object.__setattr__(
                self,
                "process_id",
                _strict_int(
                    self.process_id,
                    "save journal process_id",
                    minimum=1,
                ),
            )
            object.__setattr__(
                self,
                "process_creation_filetime_100ns",
                _strict_int(
                    self.process_creation_filetime_100ns,
                    "save journal process_creation_filetime_100ns",
                    minimum=1,
                ),
            )
        if self.exit_code is not None:
            exit_code = _strict_int(
                self.exit_code,
                "save journal exit_code",
                minimum=0,
            )
            if exit_code > 0xFFFFFFFF:
                raise PcGoldenValidationError(
                    "save journal exit_code exceeds uint32"
                )
            object.__setattr__(self, "exit_code", exit_code)

    @property
    def process_instance(self) -> tuple[int, int] | None:
        if self.process_id is None:
            return None
        assert self.process_creation_filetime_100ns is not None
        return self.process_id, self.process_creation_filetime_100ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "previous_record_sha256": self.previous_record_sha256,
            "phase": self.phase,
            "snapshot_artifact": self.snapshot_artifact,
            "snapshot_state_root": self.snapshot_state_root,
            "snapshot_perf_counter_ns": self.snapshot_perf_counter_ns,
            "process_id": self.process_id,
            "process_creation_filetime_100ns": (
                self.process_creation_filetime_100ns
            ),
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SaveJournalRecord":
        data = _mapping(value, "save journal record")
        fields = {
            "sequence",
            "previous_record_sha256",
            "phase",
            "snapshot_artifact",
            "snapshot_state_root",
            "snapshot_perf_counter_ns",
            "process_id",
            "process_creation_filetime_100ns",
            "exit_code",
        }
        _check_keys(data, required=fields, name="save journal record")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class PcSaveJournal:
    """Canonical NDJSON journal with a verifiable SHA-256 record chain."""

    session_nonce: str
    records: tuple[SaveJournalRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_nonce", _nonce(self.session_nonce))
        records = tuple(self.records)
        if not records:
            raise PcGoldenValidationError("save journal must not be empty")
        if len(records) > _MAX_JOURNAL_RECORDS:
            raise PcGoldenValidationError(
                "save journal exceeds the record-count limit"
            )
        if any(not isinstance(item, SaveJournalRecord) for item in records):
            raise PcGoldenValidationError(
                "save journal records must contain SaveJournalRecord objects"
            )
        previous_hash = _line_sha256(self._header_dict())
        previous_perf_ns: int | None = None
        for expected_sequence, record in enumerate(records):
            if record.sequence != expected_sequence:
                raise PcGoldenValidationError(
                    "save journal sequences must be contiguous and start at "
                    "zero"
                )
            if record.previous_record_sha256 != previous_hash:
                raise PcGoldenValidationError(
                    "save journal previous-record hash chain is broken"
                )
            if (
                previous_perf_ns is not None
                and record.snapshot_perf_counter_ns <= previous_perf_ns
            ):
                raise PcGoldenValidationError(
                    "save journal snapshot times must be strictly increasing"
                )
            previous_hash = _line_sha256(record.to_dict())
            previous_perf_ns = record.snapshot_perf_counter_ns
        object.__setattr__(self, "records", records)

    def _header_dict(self) -> dict[str, Any]:
        return {
            "schema": SAVE_JOURNAL_SCHEMA,
            "version": SAVE_JOURNAL_VERSION,
            "session_nonce": self.session_nonce,
        }

    def to_ndjson(self) -> str:
        lines = [_canonical_json_text(self._header_dict())]
        lines.extend(_canonical_json_text(item.to_dict()) for item in self.records)
        return "\n".join(lines) + "\n"

    @classmethod
    def build(
        cls,
        *,
        session_nonce: str,
        records: Sequence[Mapping[str, Any]],
    ) -> "PcSaveJournal":
        """Build a canonical journal while computing its hash chain."""

        nonce = _nonce(session_nonce)
        header = {
            "schema": SAVE_JOURNAL_SCHEMA,
            "version": SAVE_JOURNAL_VERSION,
            "session_nonce": nonce,
        }
        previous_hash = _line_sha256(header)
        built: list[SaveJournalRecord] = []
        for sequence, raw_record in enumerate(records):
            data = dict(_mapping(raw_record, "save journal build record"))
            if "sequence" in data or "previous_record_sha256" in data:
                raise PcGoldenValidationError(
                    "journal builder records must not predeclare chain fields"
                )
            record = SaveJournalRecord(
                sequence=sequence,
                previous_record_sha256=previous_hash,
                **data,
            )
            built.append(record)
            previous_hash = _line_sha256(record.to_dict())
        return cls(session_nonce=nonce, records=tuple(built))

    @classmethod
    def from_ndjson(cls, text: str) -> "PcSaveJournal":
        if not isinstance(text, str):
            raise TypeError("save journal text must be a string")
        lines = text.splitlines()
        if not lines or any(not line for line in lines):
            raise PcGoldenValidationError(
                "save journal must contain non-blank NDJSON lines"
            )
        header = _mapping(
            _json_loads_strict(lines[0], "save journal header"),
            "save journal header",
        )
        _check_keys(
            header,
            required={"schema", "version", "session_nonce"},
            name="save journal header",
        )
        if (
            header["schema"] != SAVE_JOURNAL_SCHEMA
            or _strict_int(header["version"], "save journal version")
            != SAVE_JOURNAL_VERSION
        ):
            raise PcGoldenValidationError(
                "unsupported save journal schema/version"
            )
        result = cls(
            session_nonce=header["session_nonce"],
            records=tuple(
                SaveJournalRecord.from_dict(
                    _mapping(
                        _json_loads_strict(
                            line,
                            f"save journal line {line_number}",
                        ),
                        f"save journal line {line_number}",
                    )
                )
                for line_number, line in enumerate(lines[1:], start=2)
            ),
        )
        if text != result.to_ndjson():
            raise PcGoldenValidationError(
                "save journal NDJSON is not canonical"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcSaveJournal":
        try:
            text = Path(path).read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "save journal could not be read as ASCII"
            ) from error
        return cls.from_ndjson(text)


@dataclass(frozen=True, slots=True)
class PcProcessEvent:
    """One raw process start or stop observation."""

    sequence: int
    kind: str
    process_id: int
    process_creation_filetime_100ns: int
    perf_counter_ns: int
    exit_code: int | None
    executable_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _strict_int(
                self.sequence,
                "process event sequence",
                minimum=0,
            ),
        )
        if self.kind not in {"start", "stop"}:
            raise PcGoldenValidationError(
                "process event kind must be start or stop"
            )
        object.__setattr__(
            self,
            "process_id",
            _strict_int(
                self.process_id,
                "process event process_id",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "process_creation_filetime_100ns",
            _strict_int(
                self.process_creation_filetime_100ns,
                "process event process_creation_filetime_100ns",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "perf_counter_ns",
            _strict_int(
                self.perf_counter_ns,
                "process event perf_counter_ns",
                minimum=1,
            ),
        )
        if self.kind == "start":
            if self.exit_code is not None:
                raise PcGoldenValidationError(
                    "process start event must not contain an exit code"
                )
        else:
            if self.exit_code is None:
                raise PcGoldenValidationError(
                    "process stop event must contain an exit code"
                )
            exit_code = _strict_int(
                self.exit_code,
                "process event exit_code",
                minimum=0,
            )
            if exit_code >= _NO_EXIT_CODE:
                raise PcGoldenValidationError(
                    "process event exit_code exceeds the binary format"
                )
            object.__setattr__(self, "exit_code", exit_code)
        object.__setattr__(
            self,
            "executable_sha256",
            _sha256(
                self.executable_sha256,
                "process event executable_sha256",
            ),
        )

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns


@dataclass(frozen=True, slots=True)
class PcProcessTimeline:
    """Canonical binary process timeline parsed independently by verifier."""

    session_nonce: str
    events: tuple[PcProcessEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_nonce", _nonce(self.session_nonce))
        events = tuple(self.events)
        if not events:
            raise PcGoldenValidationError(
                "process timeline must not be empty"
            )
        if len(events) > _MAX_PROCESS_EVENTS:
            raise PcGoldenValidationError(
                "process timeline exceeds the event-count limit"
            )
        if any(not isinstance(item, PcProcessEvent) for item in events):
            raise PcGoldenValidationError(
                "process timeline events must contain PcProcessEvent objects"
            )
        active: dict[tuple[int, int], PcProcessEvent] = {}
        completed: set[tuple[int, int]] = set()
        previous_time: int | None = None
        for expected_sequence, event in enumerate(events):
            if event.sequence != expected_sequence:
                raise PcGoldenValidationError(
                    "process event sequences must be contiguous and start "
                    "at zero"
                )
            if (
                previous_time is not None
                and event.perf_counter_ns <= previous_time
            ):
                raise PcGoldenValidationError(
                    "process event times must be strictly increasing"
                )
            instance = event.process_instance
            if event.kind == "start":
                if instance in active or instance in completed:
                    raise PcGoldenValidationError(
                        "process instance has duplicate start events"
                    )
                active[instance] = event
            else:
                start = active.pop(instance, None)
                if start is None:
                    raise PcGoldenValidationError(
                        "process stop event has no matching start event"
                    )
                if start.executable_sha256 != event.executable_sha256:
                    raise PcGoldenValidationError(
                        "process executable identity changed within an "
                        "instance"
                    )
                completed.add(instance)
            previous_time = event.perf_counter_ns
        if active:
            raise PcGoldenValidationError(
                "process timeline contains an unterminated process instance"
            )
        object.__setattr__(self, "events", events)

    def to_bytes(self) -> bytes:
        nonce_bytes = bytes.fromhex(self.session_nonce)
        payload = bytearray(
            _PROCESS_HEADER.pack(
                PROCESS_TIMELINE_MAGIC,
                PROCESS_TIMELINE_VERSION,
                nonce_bytes,
                len(self.events),
            )
        )
        for event in self.events:
            payload.extend(
                _PROCESS_RECORD.pack(
                    event.sequence,
                    _PROCESS_START if event.kind == "start" else _PROCESS_STOP,
                    event.process_id,
                    event.process_creation_filetime_100ns,
                    event.perf_counter_ns,
                    (
                        _NO_EXIT_CODE
                        if event.exit_code is None
                        else event.exit_code
                    ),
                    bytes.fromhex(event.executable_sha256.removeprefix("sha256:")),
                )
            )
        return bytes(payload)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "PcProcessTimeline":
        if not isinstance(payload, bytes):
            raise TypeError("process timeline payload must be bytes")
        if len(payload) < _PROCESS_HEADER.size:
            raise PcGoldenValidationError(
                "process timeline header is truncated"
            )
        magic, version, nonce_bytes, count = _PROCESS_HEADER.unpack_from(
            payload,
            0,
        )
        if (
            magic != PROCESS_TIMELINE_MAGIC
            or version != PROCESS_TIMELINE_VERSION
        ):
            raise PcGoldenValidationError(
                "unsupported process timeline schema/version"
            )
        if count == 0 or count > _MAX_PROCESS_EVENTS:
            raise PcGoldenValidationError(
                "process timeline event count is invalid"
            )
        expected_bytes = _PROCESS_HEADER.size + count * _PROCESS_RECORD.size
        if len(payload) != expected_bytes:
            raise PcGoldenValidationError(
                "process timeline byte count does not match its header"
            )
        events: list[PcProcessEvent] = []
        offset = _PROCESS_HEADER.size
        for _ in range(count):
            (
                sequence,
                kind_code,
                process_id,
                creation_filetime,
                perf_counter_ns,
                exit_code,
                executable_digest,
            ) = _PROCESS_RECORD.unpack_from(payload, offset)
            offset += _PROCESS_RECORD.size
            if kind_code not in {_PROCESS_START, _PROCESS_STOP}:
                raise PcGoldenValidationError(
                    "process timeline contains an unknown event kind"
                )
            events.append(
                PcProcessEvent(
                    sequence=sequence,
                    kind=(
                        "start"
                        if kind_code == _PROCESS_START
                        else "stop"
                    ),
                    process_id=process_id,
                    process_creation_filetime_100ns=creation_filetime,
                    perf_counter_ns=perf_counter_ns,
                    exit_code=(
                        None if exit_code == _NO_EXIT_CODE else exit_code
                    ),
                    executable_sha256=(
                        f"sha256:{executable_digest.hex()}"
                    ),
                )
            )
        result = cls(
            session_nonce=nonce_bytes.hex(),
            events=tuple(events),
        )
        if payload != result.to_bytes():
            raise PcGoldenValidationError(
                "process timeline binary is not canonical"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcProcessTimeline":
        try:
            payload = Path(path).read_bytes()
        except OSError as error:
            raise PcGoldenValidationError(
                "process timeline could not be read"
            ) from error
        return cls.from_bytes(payload)

    def event_pair(
        self,
        process_instance: tuple[int, int],
    ) -> tuple[PcProcessEvent, PcProcessEvent]:
        matches = tuple(
            event
            for event in self.events
            if event.process_instance == process_instance
        )
        if len(matches) != 2:
            raise PcGoldenValidationError(
                "process timeline does not contain one start/stop pair"
            )
        return matches[0], matches[1]


@dataclass(frozen=True, slots=True)
class FrameworkUpdateRecord:
    """Framework-update samples bracketing one captured presentation."""

    sequence: int
    present_ticks: int
    host_perf_counter_ns: int
    update_before: int
    update_after: int

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "present_ticks",
            "host_perf_counter_ns",
            "update_before",
            "update_after",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(
                    getattr(self, name),
                    f"framework update {name}",
                    minimum=0,
                ),
            )
        if self.host_perf_counter_ns == 0:
            raise PcGoldenValidationError(
                "framework update host_perf_counter_ns must be positive"
            )

    @property
    def stable_update(self) -> int | None:
        if self.update_before != self.update_after:
            return None
        return self.update_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "present_ticks": self.present_ticks,
            "host_perf_counter_ns": self.host_perf_counter_ns,
            "update_before": self.update_before,
            "update_after": self.update_after,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "FrameworkUpdateRecord":
        data = _mapping(value, "framework update record")
        fields = {
            "sequence",
            "present_ticks",
            "host_perf_counter_ns",
            "update_before",
            "update_after",
        }
        _check_keys(data, required=fields, name="framework update record")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class PcFrameworkUpdateMap:
    """Canonical per-frame native framework-update observations."""

    capture_metadata_sha256: str
    process_id: int
    process_creation_filetime_100ns: int
    executable_sha256: str
    records: tuple[FrameworkUpdateRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capture_metadata_sha256",
            _sha256(
                self.capture_metadata_sha256,
                "framework update capture_metadata_sha256",
            ),
        )
        object.__setattr__(
            self,
            "process_id",
            _strict_int(
                self.process_id,
                "framework update process_id",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "process_creation_filetime_100ns",
            _strict_int(
                self.process_creation_filetime_100ns,
                "framework update process_creation_filetime_100ns",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "executable_sha256",
            _sha256(
                self.executable_sha256,
                "framework update executable_sha256",
            ),
        )
        records = tuple(self.records)
        if not records:
            raise PcGoldenValidationError(
                "framework update map must not be empty"
            )
        if len(records) > _MAX_FRAMEWORK_UPDATE_RECORDS:
            raise PcGoldenValidationError(
                "framework update map exceeds the record-count limit"
            )
        if any(
            not isinstance(item, FrameworkUpdateRecord) for item in records
        ):
            raise PcGoldenValidationError(
                "framework update map records must contain update records"
            )
        previous_present: int | None = None
        previous_host: int | None = None
        for expected_sequence, record in enumerate(records):
            if record.sequence != expected_sequence:
                raise PcGoldenValidationError(
                    "framework update sequences must be contiguous and "
                    "start at zero"
                )
            if (
                previous_present is not None
                and record.present_ticks <= previous_present
            ):
                raise PcGoldenValidationError(
                    "framework update present ticks must be strictly "
                    "increasing"
                )
            if (
                previous_host is not None
                and record.host_perf_counter_ns <= previous_host
            ):
                raise PcGoldenValidationError(
                    "framework update host times must be strictly increasing"
                )
            if record.update_after < record.update_before:
                raise PcGoldenValidationError(
                    "framework updates must be non-decreasing"
                )
            previous_present = record.present_ticks
            previous_host = record.host_perf_counter_ns
        for previous, current in zip(records, records[1:]):
            if (
                current.update_before < previous.update_before
                or current.update_after < previous.update_after
            ):
                raise PcGoldenValidationError(
                    "framework updates must be non-decreasing"
                )
        object.__setattr__(self, "records", records)

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FRAMEWORK_UPDATE_SCHEMA,
            "version": FRAMEWORK_UPDATE_VERSION,
            "capture_metadata_sha256": self.capture_metadata_sha256,
            "process_id": self.process_id,
            "process_creation_filetime_100ns": (
                self.process_creation_filetime_100ns
            ),
            "executable_sha256": self.executable_sha256,
            "records": [item.to_dict() for item in self.records],
        }

    def to_json(self) -> str:
        return _canonical_json_file(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "PcFrameworkUpdateMap":
        data = _mapping(value, "framework update map")
        _check_keys(
            data,
            required={
                "schema",
                "version",
                "capture_metadata_sha256",
                "process_id",
                "process_creation_filetime_100ns",
                "executable_sha256",
                "records",
            },
            name="framework update map",
        )
        if (
            data["schema"] != FRAMEWORK_UPDATE_SCHEMA
            or _strict_int(
                data["version"],
                "framework update map version",
            )
            != FRAMEWORK_UPDATE_VERSION
        ):
            raise PcGoldenValidationError(
                "unsupported framework update map schema/version"
            )
        return cls(
            capture_metadata_sha256=data["capture_metadata_sha256"],
            process_id=data["process_id"],
            process_creation_filetime_100ns=data[
                "process_creation_filetime_100ns"
            ],
            executable_sha256=data["executable_sha256"],
            records=tuple(
                FrameworkUpdateRecord.from_dict(
                    _mapping(item, "framework update record")
                )
                for item in _sequence(
                    data["records"],
                    "framework update records",
                )
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "PcFrameworkUpdateMap":
        if not isinstance(text, str):
            raise TypeError("framework update map text must be a string")
        result = cls.from_dict(
            _mapping(
                _json_loads_strict(text, "framework update map"),
                "framework update map",
            )
        )
        if text != result.to_json():
            raise PcGoldenValidationError(
                "framework update map JSON is not canonical"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcFrameworkUpdateMap":
        try:
            text = Path(path).read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "framework update map could not be read as ASCII"
            ) from error
        return cls.from_json(text)


@dataclass(frozen=True, slots=True)
class PcDxgiCaptureMetadata:
    """Security-relevant facts parsed from canonical DXGI metadata v2."""

    process_id: int
    process_creation_filetime_100ns: int
    executable_sha256: str
    capture_start_perf_counter_ns: int
    capture_end_perf_counter_ns: int
    width: int
    height: int
    frame_count: int
    qpc_frequency: int
    raw_sha256: str
    frames_csv_sha256: str
    aggregate_pixel_sha256: str
    canonical_bytes: bytes = field(repr=False)
    raw_mapping: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        for name in (
            "process_id",
            "process_creation_filetime_100ns",
            "capture_start_perf_counter_ns",
            "capture_end_perf_counter_ns",
            "width",
            "height",
            "frame_count",
            "qpc_frequency",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(
                    getattr(self, name),
                    f"DXGI metadata {name}",
                    minimum=1,
                ),
            )
        if (
            self.capture_end_perf_counter_ns
            <= self.capture_start_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "DXGI capture end must be after capture start"
            )
        for name in (
            "executable_sha256",
            "raw_sha256",
            "frames_csv_sha256",
            "aggregate_pixel_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), f"DXGI metadata {name}"),
            )
        if not isinstance(self.canonical_bytes, bytes):
            raise PcGoldenValidationError(
                "DXGI canonical metadata must be bytes"
            )
        if not isinstance(self.raw_mapping, Mapping):
            raise PcGoldenValidationError(
                "DXGI raw metadata must be a mapping"
            )
        object.__setattr__(
            self,
            "raw_mapping",
            MappingProxyType(dict(self.raw_mapping)),
        )

    @property
    def sha256(self) -> str:
        return f"sha256:{hashlib.sha256(self.canonical_bytes).hexdigest()}"

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns

    @classmethod
    def from_json(cls, text: str) -> "PcDxgiCaptureMetadata":
        if not isinstance(text, str):
            raise TypeError("DXGI metadata text must be a string")
        data = _mapping(
            _json_loads_strict(text, "DXGI capture metadata"),
            "DXGI capture metadata",
        )
        if set(data) != _DXGI_METADATA_KEYS:
            raise PcGoldenValidationError(
                "DXGI capture metadata fields do not match v2"
            )
        canonical = (
            json.dumps(
                data,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        if text != canonical:
            raise PcGoldenValidationError(
                "DXGI capture metadata JSON is not canonical"
            )
        if (
            data["schema"] != DXGI_CAPTURE_SCHEMA
            or _strict_int(data["version"], "DXGI metadata version")
            != DXGI_CAPTURE_VERSION
            or data["status"] != "acquisition_complete"
        ):
            raise PcGoldenValidationError(
                "unsupported DXGI capture metadata schema/version"
            )
        if (
            data["target_process"] != "popcapgame1.exe"
            or data["pixel_format"] != "bgra"
            or data["bytes_per_pixel"] != 4
            or data["capture_mode"] != "every_new_present"
            or data["capture_storage_mode"] != "memory_then_publish"
            or data["capture_surface"]
            != "dxgi_desktop_client_region_crop"
            or data["foreground_geometry_guard"] is not True
            or data["topmost_or_injected_overlay_detection"] is not False
            or data["requires_full_frame_visual_review"] is not True
            or data["runtime_versions_verified"] is not True
            or data["missed_presentations"] != 0
            or data["frames_csv"] != "frames.csv"
            or data["raw_frames"] != "frames.bgra.raw"
        ):
            raise PcGoldenValidationError(
                "DXGI capture metadata violates the lossless capture contract"
            )
        target = _mapping(data["target_identity"], "DXGI target_identity")
        if set(target) != _DXGI_TARGET_IDENTITY_KEYS:
            raise PcGoldenValidationError(
                "DXGI target_identity fields do not match v2"
            )
        start_ns = _strict_int(
            data["capture_start_perf_counter_ns"],
            "DXGI capture start",
            minimum=1,
        )
        end_ns = _strict_int(
            data["capture_end_perf_counter_ns"],
            "DXGI capture end",
            minimum=1,
        )
        elapsed = data["capture_elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
            or abs(float(elapsed) - (end_ns - start_ns) / 1_000_000_000)
            > 1e-9
        ):
            raise PcGoldenValidationError(
                "DXGI capture elapsed time is inconsistent"
            )
        frame_count = _strict_int(
            data["frame_count"],
            "DXGI frame_count",
            minimum=1,
        )
        if (
            _strict_int(
                data["frames_csv_rows"],
                "DXGI frames_csv_rows",
                minimum=1,
            )
            != frame_count
        ):
            raise PcGoldenValidationError(
                "DXGI frame count and CSV row count differ"
            )
        return cls(
            process_id=target["process_id"],
            process_creation_filetime_100ns=target[
                "process_creation_filetime_100ns"
            ],
            executable_sha256=target["executable_sha256"],
            capture_start_perf_counter_ns=start_ns,
            capture_end_perf_counter_ns=end_ns,
            width=data["width"],
            height=data["height"],
            frame_count=frame_count,
            qpc_frequency=data["qpc_frequency"],
            raw_sha256=data["raw_sha256"],
            frames_csv_sha256=data["frames_csv_sha256"],
            aggregate_pixel_sha256=data["aggregate_pixel_sha256"],
            canonical_bytes=canonical.encode("ascii"),
            raw_mapping=data,
        )

    @classmethod
    def read(cls, path: str | Path) -> "PcDxgiCaptureMetadata":
        try:
            text = Path(path).read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "DXGI capture metadata could not be read as ASCII"
            ) from error
        return cls.from_json(text)


__all__ = [
    "DXGI_CAPTURE_SCHEMA",
    "DXGI_CAPTURE_VERSION",
    "FRAMEWORK_UPDATE_SCHEMA",
    "FRAMEWORK_UPDATE_VERSION",
    "PROCESS_TIMELINE_MAGIC",
    "PROCESS_TIMELINE_VERSION",
    "REGISTRY_ROOTS",
    "SAVE_JOURNAL_SCHEMA",
    "SAVE_JOURNAL_VERSION",
    "STATE_SNAPSHOT_SCHEMA",
    "STATE_SNAPSHOT_VERSION",
    "PcDxgiCaptureMetadata",
    "PcFrameworkUpdateMap",
    "PcProcessEvent",
    "PcProcessTimeline",
    "PcSaveJournal",
    "PcStateSnapshot",
    "SaveJournalRecord",
    "SnapshotFile",
    "FrameworkUpdateRecord",
    "SnapshotRegistryNode",
    "SnapshotRegistryValue",
]
