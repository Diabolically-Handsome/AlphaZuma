"""Validate a complete main-thread global-MTRand call oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from tools.popcap_mtrand_call_oracle import (
    EXPECTED_RETAIL_RUNTIME_SHA256,
)
from zuma_rl.revenge_core import PopCapMTRandom


MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
TRACE_SCHEMA = "zuma-rl.pc-gameplay-mtrand-call-trace"


@dataclass(frozen=True, slots=True)
class GlobalMTRandOracleEntry:
    order: int
    source_order: int
    framework_update: int
    native_game_time: int
    score: int
    score_target: int
    caller: int
    output: int
    pre_draw_count: int
    pre_index: int
    pre_state_sha256: str
    pre_payload: bytes
    post_draw_count: int
    post_index: int
    post_state_sha256: str
    post_payload: bytes

    def evidence_dict(self) -> dict[str, int | str]:
        return {
            "order": self.order,
            "source_order": self.source_order,
            "framework_update": self.framework_update,
            "native_game_time": self.native_game_time,
            "score": self.score,
            "score_target": self.score_target,
            "caller": self.caller,
            "output": self.output,
            "pre_draw_count": self.pre_draw_count,
            "pre_index": self.pre_index,
            "pre_state_sha256": self.pre_state_sha256,
            "post_draw_count": self.post_draw_count,
            "post_index": self.post_index,
            "post_state_sha256": self.post_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class GlobalMTRandCallOracle:
    source_path: Path
    source_sha256: str
    source_process_id: int
    source_main_thread_id: int
    runtime_executable_sha256: str
    seed: int
    start_after_update: int
    end_at_update: int | None
    entries: tuple[GlobalMTRandOracleEntry, ...]

    @property
    def semantic_sha256(self) -> str:
        value = {
            "source_sha256": self.source_sha256,
            "source_process_id": self.source_process_id,
            "source_main_thread_id": self.source_main_thread_id,
            "runtime_executable_sha256": (
                self.runtime_executable_sha256
            ),
            "seed": self.seed,
            "start_after_update": self.start_after_update,
            "entries": [entry.evidence_dict() for entry in self.entries],
        }
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class InitialGlobalMTRandCallOracle:
    """Source-bound state immediately before the first global wrapper call."""

    source_path: Path
    source_sha256: str
    source_process_id: int
    source_main_thread_id: int
    runtime_executable_sha256: str
    seed: int
    wrapper_address: int
    call_address: int
    caller: int
    output: int
    pre_index: int
    pre_state_sha256: str
    pre_payload: bytes
    post_index: int
    post_state_sha256: str
    post_payload: bytes

    @property
    def semantic_sha256(self) -> str:
        value = {
            "source_sha256": self.source_sha256,
            "source_process_id": self.source_process_id,
            "source_main_thread_id": self.source_main_thread_id,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "seed": self.seed,
            "wrapper_address": self.wrapper_address,
            "call_address": self.call_address,
            "caller": self.caller,
            "output": self.output,
            "pre_index": self.pre_index,
            "pre_state_sha256": self.pre_state_sha256,
            "post_index": self.post_index,
            "post_state_sha256": self.post_state_sha256,
        }
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StartupGlobalMTRandObservationEntry:
    """One source wrapper call, allowing hidden draws between calls."""

    source_order: int
    framework_update: int
    caller: int
    output: int
    pre_draw_count: int
    pre_index: int
    pre_state_sha256: str
    post_draw_count: int
    post_index: int
    post_state_sha256: str

    def evidence_dict(self) -> dict[str, int | str]:
        return {
            "source_order": self.source_order,
            "framework_update": self.framework_update,
            "caller": self.caller,
            "output": self.output,
            "pre_draw_count": self.pre_draw_count,
            "pre_index": self.pre_index,
            "pre_state_sha256": self.pre_state_sha256,
            "post_draw_count": self.post_draw_count,
            "post_index": self.post_index,
            "post_state_sha256": self.post_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class StartupGlobalMTRandObservationOracle:
    source_path: Path
    source_sha256: str
    source_process_id: int
    source_main_thread_id: int
    runtime_executable_sha256: str
    seed: int
    wrapper_address: int
    end_at_update: int
    maximum_draws: int
    entries: tuple[StartupGlobalMTRandObservationEntry, ...]

    @property
    def semantic_sha256(self) -> str:
        value = {
            "source_sha256": self.source_sha256,
            "source_process_id": self.source_process_id,
            "source_main_thread_id": self.source_main_thread_id,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "seed": self.seed,
            "wrapper_address": self.wrapper_address,
            "end_at_update": self.end_at_update,
            "entries": [entry.evidence_dict() for entry in self.entries],
        }
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _state_payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack(
        f"<{MTRAND_STATE_WORDS + 1}I",
        *rng.words,
        rng.index,
    )


def _state_hash(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(
            character not in "0123456789abcdef"
            for character in value[7:]
        )
    ):
        raise ValueError(f"{context} is invalid")
    return value


def reconstruct_mtrand_draw_counts(
    *,
    seed: int,
    state_sha256: set[str],
    maximum_draws: int,
) -> dict[str, int]:
    """Map exact seeded MTRand state hashes to their draw counts."""

    if (
        not 0 <= seed <= 0xFFFFFFFF
        or maximum_draws < 0
        or not state_sha256
    ):
        raise ValueError("MTRand draw-count reconstruction input is invalid")
    targets = {
        _state_hash(value, "MTRand reconstruction state hash")
        for value in state_sha256
    }
    observed: dict[str, int] = {}
    rng = PopCapMTRandom(seed)
    for draw_count in range(maximum_draws + 1):
        payload = _state_payload(rng)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest in targets:
            observed[digest] = draw_count
            if len(observed) == len(targets):
                break
        if draw_count != maximum_draws:
            rng.next_u31()
    if len(observed) != len(targets):
        raise ValueError(
            "MTRand states were not reconstructed within the draw bound"
        )
    return observed


def load_initial_global_mtrand_call_oracle(
    path: Path,
    *,
    seed: int,
) -> InitialGlobalMTRandCallOracle:
    """Validate and reconstruct source call order zero before execution."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("initial global MTRand seed must fit uint32")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("initial global MTRand oracle JSON is invalid") from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema") != TRACE_SCHEMA
        or raw.get("version") != 1
        or raw.get("status") != "PASS"
        or raw.get("failure") is not None
        or raw.get("breakpoint_kind") != "global_wrapper_entry"
        or raw.get("process_memory_writes") != 0
        or raw.get("persistent_file_modified") is not False
        or raw.get("hardware_breakpoint_restored") is not True
        or raw.get("stop_break_requested") is not True
        or raw.get("stop_break_observed") is not True
        or raw.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
    ):
        raise ValueError("initial global MTRand oracle contract mismatch")
    process_id = _strict_int(
        raw.get("process_id"),
        "initial global MTRand source process ID",
    )
    main_thread_id = _strict_int(
        raw.get("main_thread_id"),
        "initial global MTRand source main thread ID",
    )
    wrapper_address = _strict_int(
        raw.get("address"),
        "initial global MTRand wrapper address",
    )
    calls = raw.get("calls")
    call_count = _strict_int(
        raw.get("call_count"),
        "initial global MTRand source call count",
    )
    if (
        process_id <= 0
        or main_thread_id <= 0
        or wrapper_address <= 0
        or not isinstance(calls, list)
        or call_count <= 0
        or len(calls) != call_count
        or not isinstance(calls[0], Mapping)
    ):
        raise ValueError("initial global MTRand call list is invalid")
    call = calls[0]
    order = _strict_int(
        call.get("order"),
        "initial global MTRand source order",
    )
    thread_id = _strict_int(
        call.get("thread_id"),
        "initial global MTRand thread ID",
    )
    framework_update = _strict_int(
        call.get("framework_update"),
        "initial global MTRand framework update",
    )
    caller = _strict_int(
        call.get("caller"),
        "initial global MTRand caller",
    )
    output = _strict_int(
        call.get("output"),
        "initial global MTRand output",
    )
    pre_index = _strict_int(
        call.get("pre_index"),
        "initial global MTRand pre index",
    )
    post_index = _strict_int(
        call.get("post_index"),
        "initial global MTRand post index",
    )
    pre_hash = _state_hash(
        call.get("pre_state_sha256"),
        "initial global MTRand pre-state hash",
    )
    post_hash = _state_hash(
        call.get("post_state_sha256"),
        "initial global MTRand post-state hash",
    )
    call_address = caller - 5
    if (
        order != 0
        or thread_id != main_thread_id
        or framework_update != 0
        or call_address <= 0
        or not 0 <= output <= 0x7FFFFFFF
        or not 0 <= pre_index <= MTRAND_STATE_WORDS
        or not 0 <= post_index <= MTRAND_STATE_WORDS
    ):
        raise ValueError("initial global MTRand call contract mismatch")

    rng = PopCapMTRandom(seed)
    pre_payload = _state_payload(rng)
    observed_pre_hash = "sha256:" + hashlib.sha256(pre_payload).hexdigest()
    observed_pre_index = struct.unpack_from(
        "<I",
        pre_payload,
        MTRAND_STATE_WORDS * 4,
    )[0]
    observed_output = rng.next_u31()
    post_payload = _state_payload(rng)
    observed_post_hash = "sha256:" + hashlib.sha256(post_payload).hexdigest()
    observed_post_index = struct.unpack_from(
        "<I",
        post_payload,
        MTRAND_STATE_WORDS * 4,
    )[0]
    if (
        pre_hash != observed_pre_hash
        or pre_index != observed_pre_index
        or output != observed_output
        or post_hash != observed_post_hash
        or post_index != observed_post_index
    ):
        raise ValueError("initial global MTRand seeded transition mismatch")
    return InitialGlobalMTRandCallOracle(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        source_main_thread_id=main_thread_id,
        runtime_executable_sha256=str(raw["runtime_executable_sha256"]),
        seed=seed,
        wrapper_address=wrapper_address,
        call_address=call_address,
        caller=caller,
        output=output,
        pre_index=pre_index,
        pre_state_sha256=pre_hash,
        pre_payload=pre_payload,
        post_index=post_index,
        post_state_sha256=post_hash,
        post_payload=post_payload,
    )


def load_startup_global_mtrand_observation_oracle(
    path: Path,
    *,
    seed: int,
    end_at_update: int,
    maximum_draws: int = 100_000,
) -> StartupGlobalMTRandObservationOracle:
    """Load every source wrapper call through a startup update boundary."""

    if (
        not 0 <= seed <= 0xFFFFFFFF
        or end_at_update < 0
        or maximum_draws <= 0
    ):
        raise ValueError("startup global MTRand oracle bounds are invalid")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "startup global MTRand oracle JSON is invalid"
        ) from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema") != TRACE_SCHEMA
        or raw.get("version") != 1
        or raw.get("status") != "PASS"
        or raw.get("failure") is not None
        or raw.get("breakpoint_kind") != "global_wrapper_entry"
        or raw.get("process_memory_writes") != 0
        or raw.get("persistent_file_modified") is not False
        or raw.get("hardware_breakpoint_restored") is not True
        or raw.get("stop_break_requested") is not True
        or raw.get("stop_break_observed") is not True
        or raw.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
    ):
        raise ValueError("startup global MTRand oracle contract mismatch")
    process_id = _strict_int(
        raw.get("process_id"),
        "startup global MTRand source process ID",
    )
    main_thread_id = _strict_int(
        raw.get("main_thread_id"),
        "startup global MTRand source main thread ID",
    )
    wrapper_address = _strict_int(
        raw.get("address"),
        "startup global MTRand wrapper address",
    )
    calls = raw.get("calls")
    call_count = _strict_int(
        raw.get("call_count"),
        "startup global MTRand source call count",
    )
    if (
        process_id <= 0
        or main_thread_id <= 0
        or wrapper_address <= 0
        or not isinstance(calls, list)
        or call_count <= 0
        or len(calls) != call_count
    ):
        raise ValueError("startup global MTRand call list is invalid")

    selected: list[dict[str, int | str]] = []
    previous_update = -1
    for source_order, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError("startup global MTRand call is invalid")
        if (
            _strict_int(
                call.get("order"),
                "startup global MTRand source order",
            )
            != source_order
        ):
            raise ValueError("startup global MTRand source order is invalid")
        update = _strict_int(
            call.get("framework_update"),
            "startup global MTRand framework update",
        )
        if update > end_at_update:
            continue
        thread_id = _strict_int(
            call.get("thread_id"),
            "startup global MTRand thread ID",
        )
        caller = _strict_int(
            call.get("caller"),
            "startup global MTRand caller",
        )
        output = _strict_int(
            call.get("output"),
            "startup global MTRand output",
        )
        pre_index = _strict_int(
            call.get("pre_index"),
            "startup global MTRand pre index",
        )
        post_index = _strict_int(
            call.get("post_index"),
            "startup global MTRand post index",
        )
        pre_hash = _state_hash(
            call.get("pre_state_sha256"),
            "startup global MTRand pre-state hash",
        )
        post_hash = _state_hash(
            call.get("post_state_sha256"),
            "startup global MTRand post-state hash",
        )
        if (
            update < previous_update
            or thread_id != main_thread_id
            or caller <= 0
            or not 0 <= output <= 0x7FFFFFFF
            or not 0 <= pre_index <= MTRAND_STATE_WORDS
            or not 0 <= post_index <= MTRAND_STATE_WORDS
        ):
            raise ValueError(
                "startup global MTRand call contract mismatch"
            )
        previous_update = update
        selected.append(
            {
                "source_order": source_order,
                "framework_update": update,
                "caller": caller,
                "output": output,
                "pre_index": pre_index,
                "pre_state_sha256": pre_hash,
                "post_index": post_index,
                "post_state_sha256": post_hash,
            }
        )
    if not selected or selected[0]["source_order"] != 0:
        raise ValueError("startup global MTRand selection is incomplete")

    state_hashes = {
        str(row[key])
        for row in selected
        for key in ("pre_state_sha256", "post_state_sha256")
    }
    draw_counts = reconstruct_mtrand_draw_counts(
        seed=seed,
        state_sha256=state_hashes,
        maximum_draws=maximum_draws,
    )
    entries: list[StartupGlobalMTRandObservationEntry] = []
    previous_post_draw = -1
    for row in selected:
        pre_hash = str(row["pre_state_sha256"])
        post_hash = str(row["post_state_sha256"])
        pre_draw = draw_counts[pre_hash]
        post_draw = draw_counts[post_hash]
        rng = PopCapMTRandom(seed)
        for _ in range(pre_draw):
            rng.next_u31()
        observed_output = rng.next_u31()
        if (
            pre_draw < previous_post_draw
            or post_draw != pre_draw + 1
            or observed_output != row["output"]
            or rng.index != row["post_index"]
        ):
            raise ValueError(
                "startup global MTRand seeded reconstruction mismatch"
            )
        previous_post_draw = post_draw
        entries.append(
            StartupGlobalMTRandObservationEntry(
                source_order=int(row["source_order"]),
                framework_update=int(row["framework_update"]),
                caller=int(row["caller"]),
                output=int(row["output"]),
                pre_draw_count=pre_draw,
                pre_index=int(row["pre_index"]),
                pre_state_sha256=pre_hash,
                post_draw_count=post_draw,
                post_index=int(row["post_index"]),
                post_state_sha256=post_hash,
            )
        )
    if entries[0].pre_draw_count != 0:
        raise ValueError("startup global MTRand source does not start at seed")
    return StartupGlobalMTRandObservationOracle(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        source_main_thread_id=main_thread_id,
        runtime_executable_sha256=str(raw["runtime_executable_sha256"]),
        seed=seed,
        wrapper_address=wrapper_address,
        end_at_update=end_at_update,
        maximum_draws=maximum_draws,
        entries=tuple(entries),
    )


def load_global_mtrand_call_oracle(
    path: Path,
    *,
    seed: int,
    start_after_update: int,
    end_at_update: int | None = None,
    maximum_draws: int = 100_000,
) -> GlobalMTRandCallOracle:
    """Verify a contiguous post-boundary global call sequence."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("global MTRand oracle seed must fit uint32")
    if (
        start_after_update < 0
        or maximum_draws < 0
        or (
            end_at_update is not None
            and end_at_update <= start_after_update
        )
    ):
        raise ValueError("global MTRand oracle bounds are invalid")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("global MTRand oracle JSON is invalid") from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema") != TRACE_SCHEMA
        or raw.get("version") != 1
        or raw.get("status") != "PASS"
        or raw.get("failure") is not None
        or raw.get("breakpoint_kind") != "global_wrapper_entry"
        or raw.get("process_memory_writes") != 0
        or raw.get("persistent_file_modified") is not False
        or raw.get("hardware_breakpoint_restored") is not True
        or raw.get("stop_break_requested") is not True
        or raw.get("stop_break_observed") is not True
        or raw.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
    ):
        raise ValueError("global MTRand oracle contract mismatch")
    process_id = _strict_int(
        raw.get("process_id"),
        "global MTRand oracle process ID",
    )
    main_thread_id = _strict_int(
        raw.get("main_thread_id"),
        "global MTRand oracle main thread ID",
    )
    calls = raw.get("calls")
    call_count = _strict_int(
        raw.get("call_count"),
        "global MTRand oracle call count",
    )
    if (
        process_id <= 0
        or main_thread_id <= 0
        or not isinstance(calls, list)
        or call_count <= 0
        or len(calls) != call_count
    ):
        raise ValueError("global MTRand oracle call list is invalid")

    selected: list[dict[str, int | str]] = []
    for source_order, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError("global MTRand oracle call is invalid")
        if (
            _strict_int(
                call.get("order"),
                "global MTRand oracle source order",
            )
            != source_order
        ):
            raise ValueError("global MTRand source order is invalid")
        update = _strict_int(
            call.get("framework_update"),
            "global MTRand oracle framework update",
        )
        if update <= start_after_update:
            continue
        if end_at_update is not None and update > end_at_update:
            continue
        native = _strict_int(
            call.get("native_game_time"),
            "global MTRand oracle native game time",
        )
        score = _strict_int(
            call.get("score"),
            "global MTRand oracle score",
        )
        score_target = _strict_int(
            call.get("score_target"),
            "global MTRand oracle score target",
        )
        thread_id = _strict_int(
            call.get("thread_id"),
            "global MTRand oracle thread ID",
        )
        caller = _strict_int(
            call.get("caller"),
            "global MTRand oracle caller",
        )
        output = _strict_int(
            call.get("output"),
            "global MTRand oracle output",
        )
        pre_index = _strict_int(
            call.get("pre_index"),
            "global MTRand oracle pre index",
        )
        post_index = _strict_int(
            call.get("post_index"),
            "global MTRand oracle post index",
        )
        pre_hash = _state_hash(
            call.get("pre_state_sha256"),
            "global MTRand oracle pre-state hash",
        )
        post_hash = _state_hash(
            call.get("post_state_sha256"),
            "global MTRand oracle post-state hash",
        )
        if (
            thread_id != main_thread_id
            or native < 0
            or score < 0
            or score_target <= 0
            or caller <= 0
            or not 0 <= output <= 0x7FFFFFFF
            or not 0 <= pre_index <= MTRAND_STATE_WORDS
            or not 0 <= post_index <= MTRAND_STATE_WORDS
        ):
            raise ValueError("global MTRand oracle call contract mismatch")
        selected.append(
            {
                "source_order": source_order,
                "framework_update": update,
                "native_game_time": native,
                "score": score,
                "score_target": score_target,
                "caller": caller,
                "output": output,
                "pre_index": pre_index,
                "post_index": post_index,
                "pre_state_sha256": pre_hash,
                "post_state_sha256": post_hash,
            }
        )
    if not selected:
        raise ValueError("global MTRand oracle selection is empty")
    for left, right in zip(selected, selected[1:]):
        if (
            left["post_state_sha256"]
            != right["pre_state_sha256"]
        ):
            raise ValueError(
                "global MTRand oracle has an unobserved inter-call draw"
            )

    target_hashes = {
        str(call[key])
        for call in selected
        for key in ("pre_state_sha256", "post_state_sha256")
    }
    reconstructed: dict[str, tuple[int, int | None, bytes]] = {}
    rng = PopCapMTRandom(seed)
    previous_output: int | None = None
    for draw_count in range(maximum_draws + 1):
        payload = _state_payload(rng)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest in target_hashes:
            reconstructed[digest] = (
                draw_count,
                previous_output,
                payload,
            )
            if len(reconstructed) == len(target_hashes):
                break
        if draw_count != maximum_draws:
            previous_output = rng.next_u31()
    if len(reconstructed) != len(target_hashes):
        raise ValueError(
            "global MTRand oracle states were not reconstructed "
            "within the declared draw bound"
        )

    entries: list[GlobalMTRandOracleEntry] = []
    previous_update = -1
    previous_native = -1
    for order, call in enumerate(selected):
        pre_hash = str(call["pre_state_sha256"])
        post_hash = str(call["post_state_sha256"])
        pre_draw, _, pre_payload = reconstructed[pre_hash]
        post_draw, output, post_payload = reconstructed[post_hash]
        update = int(call["framework_update"])
        native = int(call["native_game_time"])
        if (
            update < previous_update
            or native < previous_native
            or post_draw != pre_draw + 1
            or output != call["output"]
            or struct.unpack_from(
                "<I",
                pre_payload,
                MTRAND_STATE_WORDS * 4,
            )[0]
            != call["pre_index"]
        ):
            raise ValueError(
                "global MTRand oracle seeded reconstruction mismatch"
            )
        previous_update = update
        previous_native = native
        entries.append(
            GlobalMTRandOracleEntry(
                order=order,
                source_order=int(call["source_order"]),
                framework_update=update,
                native_game_time=native,
                score=int(call["score"]),
                score_target=int(call["score_target"]),
                caller=int(call["caller"]),
                output=int(call["output"]),
                pre_draw_count=pre_draw,
                pre_index=int(call["pre_index"]),
                pre_state_sha256=pre_hash,
                pre_payload=pre_payload,
                post_draw_count=post_draw,
                post_index=int(call["post_index"]),
                post_state_sha256=post_hash,
                post_payload=post_payload,
            )
        )

    return GlobalMTRandCallOracle(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        source_main_thread_id=main_thread_id,
        runtime_executable_sha256=str(
            raw["runtime_executable_sha256"]
        ),
        seed=seed,
        start_after_update=start_after_update,
        end_at_update=end_at_update,
        entries=tuple(entries),
    )
