"""Validate and reconstruct an exact gameplay-MTRand call oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from zuma_rl.revenge_core import PopCapMTRandom


MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
TRACE_SCHEMA = "zuma-rl.pc-gameplay-mtrand-call-trace"
EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:"
    "2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)


@dataclass(frozen=True, slots=True)
class GameplayMTRandOracleEntry:
    order: int
    caller: int
    framework_update: int
    native_game_time: int
    score: int
    score_target: int
    output: int
    post_draw_count: int
    post_index: int
    post_state_sha256: str
    post_payload: bytes

    def evidence_dict(self) -> dict[str, int | str]:
        return {
            "order": self.order,
            "caller": self.caller,
            "framework_update": self.framework_update,
            "native_game_time": self.native_game_time,
            "score": self.score,
            "score_target": self.score_target,
            "output": self.output,
            "post_draw_count": self.post_draw_count,
            "post_index": self.post_index,
            "post_state_sha256": self.post_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class GameplayMTRandOracle:
    source_path: Path
    source_sha256: str
    source_process_id: int
    runtime_executable_sha256: str
    seed: int
    entries: tuple[GameplayMTRandOracleEntry, ...]

    @property
    def semantic_sha256(self) -> str:
        value = {
            "source_sha256": self.source_sha256,
            "source_process_id": self.source_process_id,
            "runtime_executable_sha256": (
                self.runtime_executable_sha256
            ),
            "seed": self.seed,
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


def load_gameplay_mtrand_oracle(
    path: Path,
    *,
    seed: int,
    maximum_draws: int = 100_000,
) -> GameplayMTRandOracle:
    """Verify call evidence and recover each exact post-call state payload."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("gameplay MTRand oracle seed must fit uint32")
    if maximum_draws < 0:
        raise ValueError("gameplay MTRand maximum draws is invalid")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("gameplay MTRand oracle JSON is invalid") from error
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema") != TRACE_SCHEMA
        or raw.get("version") != 1
        or raw.get("status") != "PASS"
        or raw.get("failure") is not None
        or raw.get("process_memory_writes") != 0
        or raw.get("persistent_file_modified") is not False
        or raw.get("hardware_breakpoint_restored") is not True
        or raw.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
    ):
        raise ValueError("gameplay MTRand oracle contract mismatch")
    process_id = _strict_int(
        raw.get("process_id"),
        "gameplay MTRand oracle process ID",
    )
    calls = raw.get("calls")
    call_count = _strict_int(
        raw.get("call_count"),
        "gameplay MTRand oracle call count",
    )
    if (
        process_id <= 0
        or not isinstance(calls, list)
        or call_count <= 0
        or len(calls) != call_count
    ):
        raise ValueError("gameplay MTRand oracle call list is invalid")

    parsed: list[dict[str, int | str]] = []
    target_hashes: set[str] = set()
    previous_update = -1
    previous_native = -1
    for order, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError("gameplay MTRand oracle call is invalid")
        observed_order = _strict_int(
            call.get("order"),
            "gameplay MTRand oracle order",
        )
        update = _strict_int(
            call.get("framework_update"),
            "gameplay MTRand oracle framework update",
        )
        native = _strict_int(
            call.get("native_game_time"),
            "gameplay MTRand oracle native game time",
        )
        score = _strict_int(
            call.get("score"),
            "gameplay MTRand oracle score",
        )
        score_target = _strict_int(
            call.get("score_target"),
            "gameplay MTRand oracle score target",
        )
        output = _strict_int(
            call.get("output"),
            "gameplay MTRand oracle output",
        )
        caller = _strict_int(
            call.get("caller"),
            "gameplay MTRand oracle caller",
        )
        post_index = _strict_int(
            call.get("post_index"),
            "gameplay MTRand oracle post index",
        )
        post_sha256 = call.get("post_state_sha256")
        if (
            observed_order != order
            or update < previous_update
            or native < previous_native
            or score < 0
            or score_target <= 0
            or not 0 < caller <= 0xFFFFFFFF
            or not 0 <= output <= 0x7FFFFFFF
            or not 0 <= post_index <= MTRAND_STATE_WORDS
            or not isinstance(post_sha256, str)
            or not post_sha256.startswith("sha256:")
            or len(post_sha256) != 71
            or any(
                character not in "0123456789abcdef"
                for character in post_sha256[7:]
            )
            or post_sha256 in target_hashes
        ):
            raise ValueError(
                "gameplay MTRand oracle call contract mismatch"
            )
        previous_update = update
        previous_native = native
        target_hashes.add(post_sha256)
        parsed.append(
            {
                "order": order,
                "caller": caller,
                "framework_update": update,
                "native_game_time": native,
                "score": score,
                "score_target": score_target,
                "output": output,
                "post_index": post_index,
                "post_state_sha256": post_sha256,
            }
        )

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
            "gameplay MTRand oracle states were not reconstructed "
            "within the declared draw bound"
        )

    entries: list[GameplayMTRandOracleEntry] = []
    previous_draw_count = -1
    for call in parsed:
        post_sha256 = str(call["post_state_sha256"])
        draw_count, output, payload = reconstructed[post_sha256]
        if (
            draw_count <= previous_draw_count
            or output is None
            or output != call["output"]
            or struct.unpack_from(
                "<I",
                payload,
                MTRAND_STATE_WORDS * 4,
            )[0]
            != call["post_index"]
        ):
            raise ValueError(
                "gameplay MTRand oracle seeded reconstruction mismatch"
            )
        previous_draw_count = draw_count
        entries.append(
            GameplayMTRandOracleEntry(
                order=int(call["order"]),
                caller=int(call["caller"]),
                framework_update=int(call["framework_update"]),
                native_game_time=int(call["native_game_time"]),
                score=int(call["score"]),
                score_target=int(call["score_target"]),
                output=int(call["output"]),
                post_draw_count=draw_count,
                post_index=int(call["post_index"]),
                post_state_sha256=post_sha256,
                post_payload=payload,
            )
        )

    return GameplayMTRandOracle(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        runtime_executable_sha256=str(
            raw["runtime_executable_sha256"]
        ),
        seed=seed,
        entries=tuple(entries),
    )
