"""Strictly validate and analyze compact retail global-RNG trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence

from .pc_memory_trajectory import (
    INITIAL_SAMPLE_BARRIER,
    STEPPED_SAMPLE_BARRIER,
    TRAJECTORY_SAMPLE_PHASE,
)
from .revenge_core import PopCapMTRandom


RNG_TRAJECTORY_SCHEMA = "zuma-rl.pc-rng-trajectory"
RNG_TRAJECTORY_VERSION = 2
RNG_ANALYSIS_SCHEMA = "zuma-rl.pc-rng-trajectory-analysis"
RNG_ANALYSIS_VERSION = 2
RNG_TRACE_ALIGNMENT_SCHEMA = "zuma-rl.pc-rng-trace-alignment"
RNG_TRACE_ALIGNMENT_VERSION = 1
POWERUP_SPAWN_VERIFICATION_SCHEMA = (
    "zuma-rl.pc-powerup-spawn-verification"
)
POWERUP_SPAWN_VERIFICATION_VERSION = 1
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
CURVE_LIST_OFFSETS = (0x50, 0x5C, 0x68)
POWERUP_NONE_TYPE = 14
POWERUP_SPAWN_CALLERS = (
    (0x0045CEE0, "chance_roll"),
    (0x0045CF3B, "weighted_type"),
    (0x0045D2A7, "eligible_color"),
    (0x0045D695, "eligible_ball"),
)


class PcRngTrajectoryError(ValueError):
    """A compact trajectory or one of its bound artifacts is invalid."""


@dataclass(frozen=True, slots=True)
class RngIdentity:
    ball_id: int
    color_id: int
    kind: str
    powerup_previous_type: int | None = None
    powerup_primary_type: int | None = None
    powerup_secondary_type: int | None = None


@dataclass(frozen=True, slots=True)
class RngCurveList:
    curve_index: int
    container_offset: int
    entities: tuple[RngIdentity | None, ...]
    topology_sha256: str


@dataclass(frozen=True, slots=True)
class RngTrajectoryFrame:
    update: int
    score: int
    displayed_score: int
    score_target: int
    board_color_counts: tuple[int, ...]
    chain_ball_count: int
    pending_ball_count: int
    inserting_ball_count: int
    fired_bullet_count: int
    current_ball: RngIdentity | None
    next_ball: RngIdentity | None
    qrand_update_count: int
    qrand_selected_index: int
    qrand_vectors: tuple[tuple[str, tuple[int | float, ...]], ...]
    thread_crt_rand_state: int | None
    curve_lists: tuple[RngCurveList, ...]
    mtrand_words: tuple[int, ...]
    mtrand_index: int

    def colors_for_list(self, container_offset: int) -> tuple[int, ...]:
        return tuple(
            entity.color_id
            for curve_list in self.curve_lists
            if curve_list.container_offset == container_offset
            for entity in curve_list.entities
            if entity is not None
        )

    @property
    def pending_colors(self) -> tuple[int, ...]:
        return self.colors_for_list(0x68)


def _fail(code: str) -> None:
    raise PcRngTrajectoryError(code)


def _reject_constant(value: str) -> None:
    _fail(f"rng_trajectory_nonfinite_json_number:{value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"rng_trajectory_duplicate_json_key:{key}")
        result[key] = value
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except PcRngTrajectoryError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PcRngTrajectoryError(
            "rng_trajectory_json_invalid"
        ) from error
    if not isinstance(value, dict):
        _fail("rng_trajectory_json_root_invalid")
    if payload != _canonical_bytes(value):
        _fail("rng_trajectory_json_not_canonical")
    return value


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _integer(
    value: Any,
    code: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        _fail(code)
    return value


def _artifact_path(root: Path, member: Any) -> Path:
    if not isinstance(member, str) or not member or "\\" in member or ":" in member:
        _fail("rng_trajectory_artifact_path_invalid")
    pure = PurePosixPath(member)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("rng_trajectory_artifact_path_invalid")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise PcRngTrajectoryError(
            "rng_trajectory_artifact_path_escape"
        ) from error
    return path


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _identity(value: Any, code: str) -> RngIdentity | None:
    if value is None:
        return None
    record = _mapping(value, code)
    kind = record.get("kind")
    if kind not in {"ball", "bullet"}:
        _fail(code)
    powerups: list[int | None] = []
    for key in (
        "powerup_previous_type",
        "powerup_primary_type",
        "powerup_secondary_type",
    ):
        raw = record.get(key)
        powerups.append(
            None
            if raw is None
            else _integer(raw, code, maximum=14)
        )
    return RngIdentity(
        ball_id=_integer(
            record.get("ball_id"),
            code,
            maximum=0xFFFFFFFF,
        ),
        color_id=_integer(
            record.get("color_id"),
            code,
            maximum=5,
        ),
        kind=kind,
        powerup_previous_type=powerups[0],
        powerup_primary_type=powerups[1],
        powerup_secondary_type=powerups[2],
    )


def _curve_lists(value: Any) -> tuple[RngCurveList, ...]:
    rows = _sequence(value, "rng_trajectory_curve_lists_invalid")
    decoded: list[RngCurveList] = []
    seen: set[tuple[int, int]] = set()
    for value in rows:
        row = _mapping(value, "rng_trajectory_curve_list_invalid")
        curve_index = _integer(
            row.get("curve_index"),
            "rng_trajectory_curve_index_invalid",
        )
        offset = _integer(
            row.get("container_offset"),
            "rng_trajectory_curve_list_offset_invalid",
        )
        if offset not in CURVE_LIST_OFFSETS or (curve_index, offset) in seen:
            _fail("rng_trajectory_curve_list_identity_invalid")
        seen.add((curve_index, offset))
        entities = tuple(
            _identity(item, "rng_trajectory_curve_entity_invalid")
            for item in _sequence(
                row.get("entities"),
                "rng_trajectory_curve_entities_invalid",
            )
        )
        declared = _integer(
            row.get("declared_count"),
            "rng_trajectory_curve_declared_count_invalid",
        )
        traversed = _integer(
            row.get("traversed_count"),
            "rng_trajectory_curve_traversed_count_invalid",
        )
        if declared != traversed or traversed != len(entities):
            _fail("rng_trajectory_curve_list_count_mismatch")
        digest = row.get("topology_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
        ):
            _fail("rng_trajectory_curve_topology_sha256_invalid")
        decoded.append(
            RngCurveList(
                curve_index=curve_index,
                container_offset=offset,
                entities=entities,
                topology_sha256=digest,
            )
        )
    if not decoded:
        _fail("rng_trajectory_curve_lists_empty")
    curve_indices = {row.curve_index for row in decoded}
    if seen != {
        (curve_index, offset)
        for curve_index in curve_indices
        for offset in CURVE_LIST_OFFSETS
    }:
        _fail("rng_trajectory_curve_list_set_incomplete")
    return tuple(sorted(decoded, key=lambda row: (row.curve_index, row.container_offset)))


def _qrand(value: Any) -> tuple[int, int, tuple[tuple[str, tuple[int | float, ...]], ...]]:
    record = _mapping(value, "rng_trajectory_qrand_invalid")
    vectors = _mapping(
        record.get("vectors"),
        "rng_trajectory_qrand_vectors_invalid",
    )
    decoded: list[tuple[str, tuple[int | float, ...]]] = []
    for name, value_type in (
        ("weights", "float"),
        ("sways", "float"),
        ("last_hit", "int"),
        ("previous_hit", "int"),
    ):
        values = _sequence(
            vectors.get(name),
            f"rng_trajectory_qrand_{name}_invalid",
        )
        if len(values) not in {0, 6}:
            _fail(f"rng_trajectory_qrand_{name}_length_invalid")
        if value_type == "float":
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in values
            ):
                _fail(f"rng_trajectory_qrand_{name}_value_invalid")
            decoded_values: tuple[int | float, ...] = tuple(
                float(item) for item in values
            )
        else:
            if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
                _fail(f"rng_trajectory_qrand_{name}_value_invalid")
            decoded_values = tuple(int(item) for item in values)
        decoded.append((name, decoded_values))
    update_count = _integer(
        record.get("update_count"),
        "rng_trajectory_qrand_update_count_invalid",
    )
    selected_index = record.get("selected_index")
    vector_lengths = {len(items) for _, items in decoded}
    if vector_lengths == {0}:
        if update_count != 0 or selected_index != -1:
            _fail("rng_trajectory_qrand_uninitialized_state_invalid")
        return update_count, -1, tuple(decoded)
    if vector_lengths != {6}:
        _fail("rng_trajectory_qrand_vector_initialization_mismatch")
    if update_count == 0 and selected_index == -1:
        return update_count, -1, tuple(decoded)
    return (
        update_count,
        _integer(
            selected_index,
            "rng_trajectory_qrand_selected_index_invalid",
            maximum=5,
        ),
        tuple(decoded),
    )


def load_rng_trajectory(index_path: Path) -> tuple[RngTrajectoryFrame, ...]:
    """Load a compact trajectory and verify every bound MT state record."""

    index_path = index_path.resolve()
    root = index_path.parent
    index = _read_canonical_json(index_path)
    if (
        index.get("schema") != RNG_TRAJECTORY_SCHEMA
        or index.get("version") != RNG_TRAJECTORY_VERSION
        or index.get("sample_phase") != TRAJECTORY_SAMPLE_PHASE
    ):
        _fail("rng_trajectory_schema_invalid")
    start = _integer(index.get("start_update"), "rng_trajectory_start_invalid")
    end = _integer(index.get("end_update"), "rng_trajectory_end_invalid")
    freeze = _integer(
        index.get("freeze_update", start),
        "rng_trajectory_freeze_update_invalid",
    )
    warmup_count = _integer(
        index.get("warmup_tick_count", start - freeze),
        "rng_trajectory_warmup_count_invalid",
    )
    count = _integer(
        index.get("tick_count"),
        "rng_trajectory_tick_count_invalid",
        minimum=1,
    )
    rows = _sequence(index.get("ticks"), "rng_trajectory_ticks_invalid")
    if (
        freeze > start
        or warmup_count != start - freeze
        or end < start
        or count != end - start + 1
        or len(rows) != count
    ):
        _fail("rng_trajectory_update_range_invalid")
    first_barrier = (
        INITIAL_SAMPLE_BARRIER
        if freeze == start
        else STEPPED_SAMPLE_BARRIER
    )

    mt = _mapping(index.get("global_mtrand"), "rng_trajectory_mtrand_invalid")
    if (
        mt.get("state_word_count") != MTRAND_STATE_WORDS
        or mt.get("record_bytes") != MTRAND_STATE_BYTES
    ):
        _fail("rng_trajectory_mtrand_layout_invalid")
    artifact = _artifact_path(root, mt.get("artifact"))
    try:
        blob = artifact.read_bytes()
    except OSError as error:
        raise PcRngTrajectoryError(
            "rng_trajectory_mtrand_artifact_missing"
        ) from error
    if (
        mt.get("artifact_bytes") != len(blob)
        or mt.get("artifact_sha256") != _sha256_bytes(blob)
        or len(blob) != count * MTRAND_STATE_BYTES
    ):
        _fail("rng_trajectory_mtrand_artifact_mismatch")

    frames: list[RngTrajectoryFrame] = []
    for frame_index, value in enumerate(rows):
        row = _mapping(value, "rng_trajectory_tick_invalid")
        update = _integer(
            row.get("framework_update"),
            "rng_trajectory_tick_update_invalid",
        )
        if (
            update != start + frame_index
            or row.get("sample_phase") != TRAJECTORY_SAMPLE_PHASE
            or row.get("sample_barrier")
            != (
                first_barrier
                if frame_index == 0
                else STEPPED_SAMPLE_BARRIER
            )
        ):
            _fail("rng_trajectory_tick_sequence_invalid")
        replay = _mapping(
            row.get("replay_state"),
            "rng_trajectory_replay_state_invalid",
        )
        expected_barrier = (
            first_barrier
            if frame_index == 0
            else STEPPED_SAMPLE_BARRIER
        )
        if (
            replay.get("update_count") != update
            or replay.get("frame_time_ms") != 10
            or not isinstance(replay.get("update_multiplier"), (int, float))
            or float(replay["update_multiplier"]) > 0.1
            or replay.get("fast_forward_to_marker") is not False
            or replay.get("fast_forward_step") is not False
            or not isinstance(replay.get("fast_forward_target"), int)
            or (
                expected_barrier == INITIAL_SAMPLE_BARRIER
                and not 0 <= replay["fast_forward_target"] <= update
            )
            or (
                expected_barrier == STEPPED_SAMPLE_BARRIER
                and replay["fast_forward_target"] != update
            )
        ):
            _fail("rng_trajectory_replay_state_invalid")
        offset = _integer(
            row.get("global_mtrand_record_offset"),
            "rng_trajectory_mtrand_record_offset_invalid",
        )
        if (
            offset != frame_index * MTRAND_STATE_BYTES
            or row.get("global_mtrand_record_bytes") != MTRAND_STATE_BYTES
        ):
            _fail("rng_trajectory_mtrand_record_layout_invalid")
        payload = blob[offset : offset + MTRAND_STATE_BYTES]
        if row.get("global_mtrand_record_sha256") != _sha256_bytes(payload):
            _fail("rng_trajectory_mtrand_record_sha256_mismatch")
        unpacked = struct.unpack("<625I", payload)
        words = tuple(unpacked[:MTRAND_STATE_WORDS])
        mt_index = unpacked[MTRAND_STATE_WORDS]
        if (
            mt_index > MTRAND_STATE_WORDS
            or row.get("global_mtrand_index") != mt_index
        ):
            _fail("rng_trajectory_mtrand_index_mismatch")

        curve_lists = _curve_lists(row.get("curve_lists"))
        counts = {
            offset: sum(
                len(curve_list.entities)
                for curve_list in curve_lists
                if curve_list.container_offset == offset
            )
            for offset in CURVE_LIST_OFFSETS
        }
        chain_count = _integer(
            row.get("chain_ball_count"),
            "rng_trajectory_chain_count_invalid",
        )
        pending_count = _integer(
            row.get("pending_ball_count"),
            "rng_trajectory_pending_count_invalid",
        )
        inserting_count = _integer(
            row.get("inserting_ball_count"),
            "rng_trajectory_inserting_count_invalid",
        )
        if (
            counts[0x5C] != chain_count
            or counts[0x68] != pending_count
            or counts[0x50] != inserting_count
        ):
            _fail("rng_trajectory_summary_count_mismatch")
        board_counts = tuple(
            _integer(
                item,
                "rng_trajectory_board_color_count_invalid",
            )
            for item in _sequence(
                row.get("board_color_counts"),
                "rng_trajectory_board_color_counts_invalid",
            )
        )
        if len(board_counts) != 6:
            _fail("rng_trajectory_board_color_counts_length_invalid")
        qrand_update, qrand_selected, qrand_vectors = _qrand(row.get("qrand"))
        crt_value = row.get("thread_crt_rand_state")
        if crt_value is not None:
            crt_value = _integer(
                crt_value,
                "rng_trajectory_crt_state_invalid",
                maximum=0xFFFFFFFF,
            )
        frames.append(
            RngTrajectoryFrame(
                update=update,
                score=_integer(row.get("score"), "rng_trajectory_score_invalid"),
                displayed_score=_integer(
                    row.get("displayed_score"),
                    "rng_trajectory_displayed_score_invalid",
                ),
                score_target=_integer(
                    row.get("score_target"),
                    "rng_trajectory_score_target_invalid",
                ),
                board_color_counts=board_counts,
                chain_ball_count=chain_count,
                pending_ball_count=pending_count,
                inserting_ball_count=inserting_count,
                fired_bullet_count=_integer(
                    row.get("fired_bullet_count"),
                    "rng_trajectory_fired_count_invalid",
                ),
                current_ball=_identity(
                    row.get("current_ball"),
                    "rng_trajectory_current_ball_invalid",
                ),
                next_ball=_identity(
                    row.get("next_ball"),
                    "rng_trajectory_next_ball_invalid",
                ),
                qrand_update_count=qrand_update,
                qrand_selected_index=qrand_selected,
                qrand_vectors=qrand_vectors,
                thread_crt_rand_state=crt_value,
                curve_lists=curve_lists,
                mtrand_words=words,
                mtrand_index=mt_index,
            )
        )
    return tuple(frames)


def mtrand_outputs_between(
    before: RngTrajectoryFrame,
    after: RngTrajectoryFrame,
    *,
    maximum_calls: int = 64,
) -> tuple[int, ...]:
    """Return the exact MT outputs required to advance one captured state."""

    if maximum_calls < 0:
        raise ValueError("maximum_calls must be non-negative")
    target = (after.mtrand_words, after.mtrand_index)
    rng = PopCapMTRandom(1)
    rng.load_state(before.mtrand_words, before.mtrand_index)
    outputs: list[int] = []
    for _ in range(maximum_calls + 1):
        if rng.state == target:
            return tuple(outputs)
        if len(outputs) == maximum_calls:
            break
        outputs.append(rng.next_u31())
    raise PcRngTrajectoryError(
        "rng_trajectory_mtrand_transition_unreachable:"
        f"{before.update}->{after.update}"
    )


def _powerup_fields(
    entity: RngIdentity | None,
) -> tuple[int | None, int | None, int | None] | None:
    if entity is None:
        return None
    return (
        entity.powerup_previous_type,
        entity.powerup_primary_type,
        entity.powerup_secondary_type,
    )


def _powerup_snapshot(
    frame: RngTrajectoryFrame,
) -> dict[tuple[int, int, int], RngIdentity | None]:
    snapshot: dict[tuple[int, int, int], RngIdentity | None] = {}
    for curve_list in frame.curve_lists:
        for entity_index, entity in enumerate(curve_list.entities):
            key = (
                curve_list.curve_index,
                curve_list.container_offset,
                entity_index,
            )
            if key in snapshot:
                _fail("powerup_snapshot_duplicate_entity_slot")
            snapshot[key] = entity
    return snapshot


def _powerup_record(
    entity: RngIdentity | None,
) -> dict[str, int | None] | None:
    if entity is None:
        return None
    return {
        "previous_type": entity.powerup_previous_type,
        "primary_type": entity.powerup_primary_type,
        "secondary_type": entity.powerup_secondary_type,
    }


def powerup_changes_between(
    before: RngTrajectoryFrame,
    after: RngTrajectoryFrame,
) -> tuple[dict[str, Any], ...]:
    """Return every per-entity power-up field mutation between two frames."""

    before_snapshot = _powerup_snapshot(before)
    after_snapshot = _powerup_snapshot(after)
    changes: list[dict[str, Any]] = []
    for key in sorted(before_snapshot.keys() | after_snapshot.keys()):
        before_entity = before_snapshot.get(key)
        after_entity = after_snapshot.get(key)
        if _powerup_fields(before_entity) == _powerup_fields(after_entity):
            continue
        curve_index, container_offset, entity_index = key
        changes.append(
            {
                "curve_index": curve_index,
                "container_offset": container_offset,
                "entity_index": entity_index,
                "ball_id_before": (
                    before_entity.ball_id
                    if before_entity is not None
                    else None
                ),
                "ball_id_after": (
                    after_entity.ball_id
                    if after_entity is not None
                    else None
                ),
                "color_id_before": (
                    before_entity.color_id
                    if before_entity is not None
                    else None
                ),
                "color_id_after": (
                    after_entity.color_id
                    if after_entity is not None
                    else None
                ),
                "before": _powerup_record(before_entity),
                "after": _powerup_record(after_entity),
            }
        )
    return tuple(changes)


def analyze_rng_trajectory(
    frames: Sequence[RngTrajectoryFrame],
    *,
    maximum_calls_per_tick: int = 64,
) -> dict[str, Any]:
    """Produce an auditable transition report with exact consumed outputs."""

    if not frames:
        raise ValueError("frames must not be empty")
    transitions: list[dict[str, Any]] = []
    histogram: Counter[int] = Counter()
    total_calls = 0
    for before, after in zip(frames, frames[1:]):
        outputs = mtrand_outputs_between(
            before,
            after,
            maximum_calls=maximum_calls_per_tick,
        )
        call_count = len(outputs)
        histogram[call_count] += 1
        total_calls += call_count
        mutations: list[str] = []
        if before.chain_ball_count != after.chain_ball_count:
            mutations.append("chain_count")
        if before.pending_ball_count != after.pending_ball_count:
            mutations.append("pending_count")
        if before.pending_colors != after.pending_colors:
            mutations.append("pending_colors")
        if before.current_ball != after.current_ball:
            mutations.append("current_ball")
        if before.next_ball != after.next_ball:
            mutations.append("next_ball")
        if before.qrand_update_count != after.qrand_update_count:
            mutations.append("qrand")
        if before.thread_crt_rand_state != after.thread_crt_rand_state:
            mutations.append("thread_crt_rand")
        powerup_changes = powerup_changes_between(before, after)
        if powerup_changes:
            mutations.append("powerups")
        transitions.append(
            {
                "from_update": before.update,
                "to_update": after.update,
                "mtrand_index_before": before.mtrand_index,
                "mtrand_index_after": after.mtrand_index,
                "mtrand_call_count": call_count,
                "mtrand_outputs": list(outputs),
                "mutations": mutations,
                "powerup_changes": list(powerup_changes),
                "chain_ball_count_before": before.chain_ball_count,
                "chain_ball_count_after": after.chain_ball_count,
                "pending_colors_before": list(before.pending_colors),
                "pending_colors_after": list(after.pending_colors),
                "current_color_before": (
                    before.current_ball.color_id
                    if before.current_ball is not None
                    else None
                ),
                "current_color_after": (
                    after.current_ball.color_id
                    if after.current_ball is not None
                    else None
                ),
                "next_color_before": (
                    before.next_ball.color_id
                    if before.next_ball is not None
                    else None
                ),
                "next_color_after": (
                    after.next_ball.color_id
                    if after.next_ball is not None
                    else None
                ),
            }
        )
    return {
        "schema": RNG_ANALYSIS_SCHEMA,
        "version": RNG_ANALYSIS_VERSION,
        "status": "PASS",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "tick_count": len(frames),
        "transition_count": len(transitions),
        "total_mtrand_calls": total_calls,
        "mtrand_call_count_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "non_two_call_transition_count": sum(
            row["mtrand_call_count"] != 2 for row in transitions
        ),
        "topology_or_shooter_transition_count": sum(
            bool(row["mutations"]) for row in transitions
        ),
        "transitions": transitions,
    }


def verify_rng_call_trace_alignment(
    frames: Sequence[RngTrajectoryFrame],
    trace_path: Path,
    *,
    maximum_calls_per_tick: int = 64,
) -> dict[str, Any]:
    """Bind each dynamic wrapper hit to the exact adjacent MT states."""

    if not frames:
        raise ValueError("frames must not be empty")
    trace_path = trace_path.resolve()
    trace = _read_canonical_json(trace_path)
    calls = _sequence(
        trace.get("calls"),
        "rng_call_trace_calls_invalid",
    )
    if (
        trace.get("schema") != "zuma-rl.pc-global-mtrand-call-trace"
        or trace.get("version") != 1
        or trace.get("status") != "PASS"
        or trace.get("start_update") != frames[0].update
        or trace.get("end_update") != frames[-1].update
        or trace.get("call_count") != len(calls)
    ):
        _fail("rng_call_trace_contract_invalid")

    cursor = 0
    transitions: list[dict[str, Any]] = []
    caller_counts: Counter[str] = Counter()
    thread_ids: set[int] = set()
    for before, after in zip(frames, frames[1:]):
        expected_outputs = mtrand_outputs_between(
            before,
            after,
            maximum_calls=maximum_calls_per_tick,
        )
        rng = PopCapMTRandom(1)
        rng.load_state(before.mtrand_words, before.mtrand_index)
        transition_calls: list[dict[str, Any]] = []
        for expected_output in expected_outputs:
            if cursor >= len(calls):
                _fail("rng_call_trace_call_count_mismatch")
            call = _mapping(
                calls[cursor],
                "rng_call_trace_call_invalid",
            )
            state_payload = struct.pack(
                "<625I",
                *rng.words,
                rng.index,
            )
            index_before = rng.index
            observed_output = rng.next_u31()
            index_after = rng.index
            caller = _integer(
                call.get("caller"),
                "rng_call_trace_caller_invalid",
                minimum=1,
            )
            caller_hex = f"0x{caller:08x}"
            thread_id = _integer(
                call.get("thread_id"),
                "rng_call_trace_thread_invalid",
                minimum=1,
            )
            if (
                call.get("order") != cursor
                or call.get("framework_update") != after.update
                or call.get("caller_hex") != caller_hex
                or call.get("mtrand_index_before") != index_before
                or call.get("mtrand_index_after") != index_after
                or call.get("mtrand_output") != expected_output
                or observed_output != expected_output
                or call.get("mtrand_state_sha256_before")
                != _sha256_bytes(state_payload)
            ):
                _fail("rng_call_trace_transition_mismatch")
            caller_counts[caller_hex] += 1
            thread_ids.add(thread_id)
            transition_calls.append(
                {
                    "order": cursor,
                    "caller": caller_hex,
                    "mtrand_index_before": index_before,
                    "mtrand_index_after": index_after,
                    "mtrand_output": expected_output,
                }
            )
            cursor += 1
        if rng.state != (after.mtrand_words, after.mtrand_index):
            _fail("rng_call_trace_terminal_state_mismatch")
        transitions.append(
            {
                "from_update": before.update,
                "to_update": after.update,
                "call_count": len(transition_calls),
                "calls": transition_calls,
            }
        )
    if cursor != len(calls):
        _fail("rng_call_trace_call_count_mismatch")

    trace_payload = trace_path.read_bytes()
    return {
        "schema": RNG_TRACE_ALIGNMENT_SCHEMA,
        "version": RNG_TRACE_ALIGNMENT_VERSION,
        "status": "PASS",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "tick_count": len(frames),
        "transition_count": len(transitions),
        "call_count": cursor,
        "thread_ids": sorted(thread_ids),
        "caller_counts": {
            key: caller_counts[key] for key in sorted(caller_counts)
        },
        "trace_artifact": trace_path.name,
        "trace_artifact_sha256": _sha256_bytes(trace_payload),
        "runtime_executable_sha256": trace.get(
            "runtime_executable_sha256"
        ),
        "transitions": transitions,
    }


def _curve_non_powerup_signature(
    frame: RngTrajectoryFrame,
) -> tuple[Any, ...]:
    return tuple(
        (
            curve_list.curve_index,
            curve_list.container_offset,
            curve_list.topology_sha256,
            tuple(
                (
                    entity.ball_id,
                    entity.color_id,
                    entity.kind,
                )
                if entity is not None
                else None
                for entity in curve_list.entities
            ),
        )
        for curve_list in frame.curve_lists
    )


def _weighted_powerup_type(
    roll: int,
    weights: Sequence[int],
) -> int:
    cursor = 0
    for powerup_type, weight in enumerate(weights):
        cursor += weight
        if roll < cursor:
            return powerup_type
    _fail("powerup_spawn_weighted_roll_out_of_range")


def verify_powerup_spawn_transition(
    frames: Sequence[RngTrajectoryFrame],
    trace_path: Path,
    *,
    to_update: int,
    curve_index: int,
    num_colors: int,
    chance_denominator: int,
    powerup_weights: Sequence[int],
    expected_powerup_type: int | None = None,
    expected_ball_id: int | None = None,
    maximum_calls_per_tick: int = 64,
) -> dict[str, Any]:
    """Verify one retail power-up spawn from trigger roll through ball mutation."""

    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    if (
        isinstance(to_update, bool)
        or not isinstance(to_update, int)
        or to_update < 1
    ):
        raise ValueError("to_update must be a positive integer")
    if (
        isinstance(curve_index, bool)
        or not isinstance(curve_index, int)
        or curve_index < 0
    ):
        raise ValueError("curve_index must be a non-negative integer")
    if (
        isinstance(num_colors, bool)
        or not isinstance(num_colors, int)
        or not 1 <= num_colors <= 6
    ):
        raise ValueError("num_colors must be in [1, 6]")
    if (
        isinstance(chance_denominator, bool)
        or not isinstance(chance_denominator, int)
        or chance_denominator < 1
    ):
        raise ValueError("chance_denominator must be positive")
    if len(powerup_weights) != POWERUP_NONE_TYPE:
        raise ValueError("powerup_weights must contain exactly 14 entries")
    weights = tuple(
        _integer(
            weight,
            "powerup_spawn_weight_invalid",
            maximum=0x7FFFFFFF,
        )
        for weight in powerup_weights
    )
    total_weight = sum(weights)
    if total_weight < 1:
        raise ValueError("powerup_weights must contain a positive weight")
    if expected_powerup_type is not None:
        _integer(
            expected_powerup_type,
            "powerup_spawn_expected_type_invalid",
            maximum=POWERUP_NONE_TYPE - 1,
        )
    if expected_ball_id is not None:
        _integer(
            expected_ball_id,
            "powerup_spawn_expected_ball_invalid",
            maximum=0xFFFFFFFF,
        )

    matching = [
        (before, after)
        for before, after in zip(frames, frames[1:])
        if after.update == to_update
    ]
    if len(matching) != 1:
        _fail("powerup_spawn_transition_not_unique")
    before, after = matching[0]
    if after.update != before.update + 1:
        _fail("powerup_spawn_updates_not_consecutive")

    outputs = mtrand_outputs_between(
        before,
        after,
        maximum_calls=maximum_calls_per_tick,
    )
    if len(outputs) != len(POWERUP_SPAWN_CALLERS):
        _fail("powerup_spawn_mtrand_call_count_mismatch")

    invariants = {
        "score_unchanged": before.score == after.score,
        "displayed_score_unchanged": (
            before.displayed_score == after.displayed_score
        ),
        "score_target_unchanged": before.score_target == after.score_target,
        "board_color_counts_unchanged": (
            before.board_color_counts == after.board_color_counts
        ),
        "list_counts_unchanged": (
            before.chain_ball_count == after.chain_ball_count
            and before.pending_ball_count == after.pending_ball_count
            and before.inserting_ball_count == after.inserting_ball_count
            and before.fired_bullet_count == after.fired_bullet_count
        ),
        "non_powerup_topology_unchanged": (
            _curve_non_powerup_signature(before)
            == _curve_non_powerup_signature(after)
        ),
        "shooter_unchanged": (
            before.current_ball == after.current_ball
            and before.next_ball == after.next_ball
        ),
        "qrand_unchanged": (
            before.qrand_update_count == after.qrand_update_count
            and before.qrand_selected_index == after.qrand_selected_index
            and before.qrand_vectors == after.qrand_vectors
        ),
        "thread_crt_rand_unchanged": (
            before.thread_crt_rand_state == after.thread_crt_rand_state
        ),
    }
    failed_invariants = [
        name for name, passed in invariants.items() if not passed
    ]
    if failed_invariants:
        _fail(
            "powerup_spawn_unrelated_mutation:"
            + ",".join(failed_invariants)
        )

    changes = powerup_changes_between(before, after)
    if len(changes) != 1:
        _fail("powerup_spawn_change_count_mismatch")
    change = changes[0]
    before_powerup = _mapping(
        change.get("before"),
        "powerup_spawn_before_state_invalid",
    )
    after_powerup = _mapping(
        change.get("after"),
        "powerup_spawn_after_state_invalid",
    )
    if (
        change["curve_index"] != curve_index
        or change["container_offset"] != 0x5C
        or change["ball_id_before"] != change["ball_id_after"]
        or change["color_id_before"] != change["color_id_after"]
        or before_powerup.get("previous_type") != POWERUP_NONE_TYPE
        or before_powerup.get("primary_type") != POWERUP_NONE_TYPE
        or before_powerup.get("secondary_type") != POWERUP_NONE_TYPE
        or after_powerup.get("previous_type") != POWERUP_NONE_TYPE
        or after_powerup.get("primary_type") != POWERUP_NONE_TYPE
        or after_powerup.get("secondary_type") == POWERUP_NONE_TYPE
    ):
        _fail("powerup_spawn_ball_mutation_invalid")
    observed_powerup_type = _integer(
        after_powerup.get("secondary_type"),
        "powerup_spawn_observed_type_invalid",
        maximum=POWERUP_NONE_TYPE - 1,
    )
    observed_ball_id = _integer(
        change.get("ball_id_after"),
        "powerup_spawn_observed_ball_invalid",
        maximum=0xFFFFFFFF,
    )
    observed_color_id = _integer(
        change.get("color_id_after"),
        "powerup_spawn_observed_color_invalid",
        maximum=num_colors - 1,
    )

    trigger_mod = outputs[0] % chance_denominator
    if trigger_mod != 0:
        _fail("powerup_spawn_trigger_roll_mismatch")
    weighted_roll = outputs[1] % total_weight
    selected_powerup_type = _weighted_powerup_type(
        weighted_roll,
        weights,
    )

    active_lists = [
        curve_list
        for curve_list in before.curve_lists
        if curve_list.curve_index == curve_index
        and curve_list.container_offset == 0x5C
    ]
    if len(active_lists) != 1:
        _fail("powerup_spawn_active_list_not_unique")
    active_entities = active_lists[0].entities
    if any(
        entity is None
        or entity.kind != "ball"
        or entity.powerup_primary_type is None
        or entity.powerup_secondary_type is None
        for entity in active_entities
    ):
        _fail("powerup_spawn_active_identity_stream_incomplete")
    balls = tuple(
        entity for entity in active_entities if entity is not None
    )
    eligible_colors = tuple(
        color
        for color in range(num_colors)
        if not any(
            entity.color_id == color
            and (
                entity.powerup_primary_type != POWERUP_NONE_TYPE
                or entity.powerup_secondary_type != POWERUP_NONE_TYPE
            )
            for entity in balls
        )
    )
    if not eligible_colors:
        _fail("powerup_spawn_no_eligible_colors")
    color_roll = outputs[2] % len(eligible_colors)
    selected_color = eligible_colors[color_roll]
    eligible_ball_ids = tuple(
        entity.ball_id
        for entity in balls
        if entity.color_id == selected_color
        and entity.powerup_primary_type == POWERUP_NONE_TYPE
        and entity.powerup_secondary_type == POWERUP_NONE_TYPE
    )
    if not eligible_ball_ids:
        _fail("powerup_spawn_no_eligible_balls")
    ball_roll = outputs[3] % len(eligible_ball_ids)
    selected_ball_id = eligible_ball_ids[ball_roll]
    if (
        selected_powerup_type != observed_powerup_type
        or selected_color != observed_color_id
        or selected_ball_id != observed_ball_id
    ):
        _fail("powerup_spawn_rng_prediction_mismatch")
    if (
        expected_powerup_type is not None
        and selected_powerup_type != expected_powerup_type
    ):
        _fail("powerup_spawn_expected_type_mismatch")
    if (
        expected_ball_id is not None
        and selected_ball_id != expected_ball_id
    ):
        _fail("powerup_spawn_expected_ball_mismatch")

    alignment = verify_rng_call_trace_alignment(
        frames,
        trace_path,
        maximum_calls_per_tick=maximum_calls_per_tick,
    )
    trace = _read_canonical_json(trace_path.resolve())
    raw_calls = _sequence(
        trace.get("calls"),
        "rng_call_trace_calls_invalid",
    )
    spawn_calls = [
        _mapping(call, "rng_call_trace_call_invalid")
        for call in raw_calls
        if isinstance(call, dict)
        and call.get("framework_update") == to_update
    ]
    observed_callers = tuple(
        _integer(
            call.get("caller"),
            "rng_call_trace_caller_invalid",
            minimum=1,
        )
        for call in spawn_calls
    )
    expected_callers = tuple(
        address for address, _ in POWERUP_SPAWN_CALLERS
    )
    if (
        observed_callers != expected_callers
        or tuple(call.get("mtrand_output") for call in spawn_calls)
        != outputs
    ):
        _fail("powerup_spawn_caller_sequence_mismatch")

    trigger_observations: list[dict[str, int | bool]] = []
    for raw_call in raw_calls:
        call = _mapping(raw_call, "rng_call_trace_call_invalid")
        if call.get("caller") != POWERUP_SPAWN_CALLERS[0][0]:
            continue
        update = _integer(
            call.get("framework_update"),
            "rng_call_trace_update_invalid",
        )
        output = _integer(
            call.get("mtrand_output"),
            "rng_call_trace_output_invalid",
            maximum=0x7FFFFFFF,
        )
        remainder = output % chance_denominator
        hit = remainder == 0
        if hit != (update == to_update):
            _fail("powerup_spawn_trigger_observation_mismatch")
        trigger_observations.append(
            {
                "framework_update": update,
                "mtrand_output": output,
                "remainder": remainder,
                "triggered": hit,
            }
        )

    return {
        "schema": POWERUP_SPAWN_VERIFICATION_SCHEMA,
        "version": POWERUP_SPAWN_VERIFICATION_VERSION,
        "status": "PASS",
        "from_update": before.update,
        "to_update": after.update,
        "curve_index": curve_index,
        "invariants": invariants,
        "mtrand_outputs": list(outputs),
        "chance_roll": {
            "caller": f"0x{POWERUP_SPAWN_CALLERS[0][0]:08x}",
            "denominator": chance_denominator,
            "output": outputs[0],
            "remainder": trigger_mod,
        },
        "weighted_type_roll": {
            "caller": f"0x{POWERUP_SPAWN_CALLERS[1][0]:08x}",
            "weights": list(weights),
            "total_weight": total_weight,
            "output": outputs[1],
            "roll": weighted_roll,
            "selected_powerup_type": selected_powerup_type,
        },
        "eligible_color_roll": {
            "caller": f"0x{POWERUP_SPAWN_CALLERS[2][0]:08x}",
            "eligible_colors": list(eligible_colors),
            "output": outputs[2],
            "roll": color_roll,
            "selected_color": selected_color,
        },
        "eligible_ball_roll": {
            "caller": f"0x{POWERUP_SPAWN_CALLERS[3][0]:08x}",
            "eligible_ball_ids": list(eligible_ball_ids),
            "output": outputs[3],
            "roll": ball_roll,
            "selected_ball_id": selected_ball_id,
        },
        "observed_change": change,
        "trigger_observations": trigger_observations,
        "call_trace": {
            "artifact": alignment["trace_artifact"],
            "artifact_sha256": alignment["trace_artifact_sha256"],
            "runtime_executable_sha256": alignment[
                "runtime_executable_sha256"
            ],
            "thread_ids": alignment["thread_ids"],
            "spawn_calls": [
                {
                    "caller": f"0x{address:08x}",
                    "role": role,
                    "mtrand_output": output,
                }
                for (address, role), output in zip(
                    POWERUP_SPAWN_CALLERS,
                    outputs,
                    strict=True,
                )
            ],
        },
    }


__all__ = [
    "POWERUP_NONE_TYPE",
    "POWERUP_SPAWN_CALLERS",
    "POWERUP_SPAWN_VERIFICATION_SCHEMA",
    "POWERUP_SPAWN_VERIFICATION_VERSION",
    "PcRngTrajectoryError",
    "RNG_ANALYSIS_SCHEMA",
    "RNG_ANALYSIS_VERSION",
    "RNG_TRAJECTORY_SCHEMA",
    "RNG_TRAJECTORY_VERSION",
    "RNG_TRACE_ALIGNMENT_SCHEMA",
    "RNG_TRACE_ALIGNMENT_VERSION",
    "RngCurveList",
    "RngIdentity",
    "RngTrajectoryFrame",
    "analyze_rng_trajectory",
    "load_rng_trajectory",
    "mtrand_outputs_between",
    "powerup_changes_between",
    "verify_powerup_spawn_transition",
    "verify_rng_call_trace_alignment",
]
