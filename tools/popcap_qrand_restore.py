"""Validate and encode a captured retail QRand state for replay restore."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Mapping


QRAND_OBJECT_SIZE = 0x48
QRAND_VECTOR_LAYOUT = (
    (0x08, "weights", "float32"),
    (0x18, "sways", "float32"),
    (0x28, "last_hit", "int32"),
    (0x38, "previous_hit", "int32"),
)
MAX_QRAND_VECTOR_ITEMS = 64


@dataclass(frozen=True, slots=True)
class QRandRestoreState:
    source_path: Path
    source_sha256: str
    source_framework_update: int
    source_native_game_time: int
    update_count: int
    selected_index: int
    vectors: Mapping[str, tuple[int | float, ...]]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "selected_index": self.selected_index,
            "vectors": {
                name: list(self.vectors[name])
                for _, name, _ in QRAND_VECTOR_LAYOUT
            },
        }

    @property
    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.semantic_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _parse_vectors(raw: Any) -> Mapping[str, tuple[int | float, ...]]:
    if not isinstance(raw, Mapping):
        raise ValueError("QRand vectors must be an object")
    expected_names = {name for _, name, _ in QRAND_VECTOR_LAYOUT}
    if set(raw) != expected_names:
        raise ValueError("QRand vector names differ from the retail layout")

    vectors: dict[str, tuple[int | float, ...]] = {}
    expected_length: int | None = None
    for _, name, value_type in QRAND_VECTOR_LAYOUT:
        values = raw[name]
        if (
            not isinstance(values, list)
            or len(values) > MAX_QRAND_VECTOR_ITEMS
        ):
            raise ValueError(f"QRand {name} vector length is invalid")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("QRand vector lengths differ")

        parsed: list[int | float] = []
        for value in values:
            if value_type == "int32":
                integer = _strict_int(value, f"QRand {name} value")
                if not -(2**31) <= integer < 2**31:
                    raise ValueError(f"QRand {name} value exceeds int32")
                parsed.append(integer)
            else:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"QRand {name} value must be finite"
                    )
                parsed.append(float(value))
        vectors[name] = tuple(parsed)
    return MappingProxyType(vectors)


def load_qrand_restore_state(
    path: Path,
    *,
    framework_update: int,
) -> QRandRestoreState:
    """Load exactly one read-only monitor row at a declared update."""

    if framework_update < 0:
        raise ValueError("QRand source update must not be negative")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("QRand monitor log is not UTF-8") from error
    if not lines:
        raise ValueError("QRand monitor log is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("QRand monitor header is invalid") from error
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != "zuma-rl.live-rng-change-monitor"
        or header.get("version") != 1
        or header.get("process_memory_writes") != 0
    ):
        raise ValueError("QRand monitor header contract mismatch")

    matches: list[Mapping[str, Any]] = []
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("QRand monitor row is invalid") from error
        if (
            isinstance(row, Mapping)
            and row.get("type") == "rng"
            and row.get("framework_update") == framework_update
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "QRand monitor must contain exactly one row at the source update"
        )
    row = matches[0]
    native_game_time = _strict_int(
        row.get("native_game_time"),
        "QRand source native game time",
    )
    qrand = row.get("qrand")
    if not isinstance(qrand, Mapping):
        raise ValueError("QRand monitor row lacks a QRand object")
    update_count = _strict_int(
        qrand.get("update_count"),
        "QRand update count",
    )
    selected_index = _strict_int(
        qrand.get("selected_index"),
        "QRand selected index",
    )
    vectors = _parse_vectors(qrand.get("vectors"))
    palette_size = len(vectors["weights"])
    if (
        update_count < 0
        or not -1 <= selected_index < palette_size
        or any(
            value < 0 or value > update_count
            for name in ("last_hit", "previous_hit")
            for value in vectors[name]
        )
    ):
        raise ValueError("QRand scalar/history values are invalid")
    return QRandRestoreState(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_framework_update=framework_update,
        source_native_game_time=native_game_time,
        update_count=update_count,
        selected_index=selected_index,
        vectors=vectors,
    )


def qrand_vector_write_plan(
    object_bytes: bytes,
    state: QRandRestoreState,
) -> tuple[tuple[int, bytes], ...]:
    """Validate live vector headers and return address/payload writes."""

    if len(object_bytes) != QRAND_OBJECT_SIZE:
        raise ValueError("live QRand object size mismatch")
    writes: list[tuple[int, bytes]] = []
    for vector_offset, name, value_type in QRAND_VECTOR_LAYOUT:
        begin, end, capacity = struct.unpack_from(
            "<III",
            object_bytes,
            vector_offset + 4,
        )
        values = state.vectors[name]
        if not values:
            if begin != 0 or end != 0 or capacity != 0:
                raise ValueError(
                    f"live QRand {name} empty vector layout mismatch"
                )
            continue
        expected_bytes = len(values) * 4
        if (
            begin == 0
            or begin % 4
            or end != begin + expected_bytes
            or capacity < end
            or capacity % 4
        ):
            raise ValueError(f"live QRand {name} vector layout mismatch")
        format_code = "f" if value_type == "float32" else "i"
        payload = struct.pack(
            f"<{len(values)}{format_code}",
            *values,
        )
        writes.append((begin, payload))
    return tuple(writes)


def qrand_scalar_payload(state: QRandRestoreState) -> bytes:
    return struct.pack("<ii", state.update_count, state.selected_index)
