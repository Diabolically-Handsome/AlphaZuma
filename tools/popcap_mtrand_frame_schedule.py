"""Build a per-Board-update global-MTRand synchronization schedule."""

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
DEFAULT_MAXIMUM_DRAWS = 100_000


@dataclass(frozen=True, slots=True)
class FrameMTRandSyncEntry:
    framework_update: int
    source_framework_update: int
    source_native_game_time: int
    source_draw_count: int
    source_index: int
    source_state_sha256: str
    output: int
    post_draw_count: int
    post_index: int
    post_state_sha256: str
    post_state_observed: bool
    post_payload: bytes

    def evidence_dict(self) -> dict[str, int | str]:
        return {
            "framework_update": self.framework_update,
            "source_framework_update": self.source_framework_update,
            "source_native_game_time": self.source_native_game_time,
            "source_draw_count": self.source_draw_count,
            "source_index": self.source_index,
            "source_state_sha256": self.source_state_sha256,
            "output": self.output,
            "post_draw_count": self.post_draw_count,
            "post_index": self.post_index,
            "post_state_sha256": self.post_state_sha256,
            "post_state_observed": self.post_state_observed,
        }


@dataclass(frozen=True, slots=True)
class FrameMTRandSyncSchedule:
    source_path: Path
    source_sha256: str
    source_process_id: int
    seed: int
    start_update: int
    end_update: int
    entries: tuple[FrameMTRandSyncEntry, ...]

    @property
    def by_update(self) -> dict[int, FrameMTRandSyncEntry]:
        return {entry.framework_update: entry for entry in self.entries}

    @property
    def semantic_sha256(self) -> str:
        value = {
            "source_sha256": self.source_sha256,
            "source_process_id": self.source_process_id,
            "seed": self.seed,
            "start_update": self.start_update,
            "end_update": self.end_update,
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
class _CapturedState:
    framework_update: int
    native_game_time: int
    index: int
    sha256: str


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


def _load_capture(
    path: Path,
    *,
    first_required_update: int,
    end_update: int,
) -> tuple[int, str, dict[int, list[_CapturedState]]]:
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("frame schedule monitor log is not UTF-8") from error
    if not lines:
        raise ValueError("frame schedule monitor log is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("frame schedule monitor header is invalid") from error
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != "zuma-rl.live-rng-change-monitor"
        or header.get("version") != 1
        or header.get("process_memory_writes") != 0
    ):
        raise ValueError("frame schedule monitor header contract mismatch")
    process_id = _strict_int(
        header.get("process_id"),
        "frame schedule source process ID",
    )
    if process_id <= 0:
        raise ValueError("frame schedule source process ID is invalid")

    by_update: dict[int, list[_CapturedState]] = {}
    previous_update = -1
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("frame schedule monitor row is invalid") from error
        if not isinstance(row, Mapping) or row.get("type") != "rng":
            continue
        update = _strict_int(
            row.get("framework_update"),
            "frame schedule framework update",
        )
        if update < previous_update:
            # A normal retail exit can begin a later Board segment.  The
            # requested first-board interval was already consumed.
            if previous_update >= end_update:
                break
            raise ValueError("frame schedule updates are not monotonic")
        previous_update = update
        if update < first_required_update or update > end_update:
            continue
        native_game_time = _strict_int(
            row.get("native_game_time"),
            "frame schedule native game time",
        )
        index = _strict_int(
            row.get("global_mtrand_index"),
            "frame schedule MTRand index",
        )
        state_sha256 = row.get("global_mtrand_sha256")
        if (
            native_game_time < 0
            or not 0 <= index <= MTRAND_STATE_WORDS
            or not isinstance(state_sha256, str)
            or len(state_sha256) != 64
            or state_sha256.lower() != state_sha256
            or any(
                character not in "0123456789abcdef"
                for character in state_sha256
            )
        ):
            raise ValueError("frame schedule captured state is invalid")
        by_update.setdefault(update, []).append(
            _CapturedState(
                framework_update=update,
                native_game_time=native_game_time,
                index=index,
                sha256="sha256:" + state_sha256,
            )
        )
    return process_id, source_sha256, by_update


def _reconstruct_states(
    *,
    seed: int,
    captures: tuple[_CapturedState, ...],
    maximum_draws: int,
) -> dict[str, tuple[int, bytes]]:
    targets = {capture.sha256 for capture in captures}
    target_indices = {capture.index for capture in captures}
    found: dict[str, tuple[int, bytes]] = {}
    rng = PopCapMTRandom(seed)
    for draw_count in range(maximum_draws + 1):
        if rng.index in target_indices:
            payload = _state_payload(rng)
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            if digest in targets:
                if digest in found:
                    raise ValueError(
                        "frame schedule state reconstruction is ambiguous"
                    )
                found[digest] = (draw_count, payload)
                if len(found) == len(targets):
                    break
        if draw_count != maximum_draws:
            rng.next_u31()
    missing = targets.difference(found)
    if missing:
        raise ValueError(
            "frame schedule states were not reconstructed within the "
            "declared draw bound"
        )
    return found


def load_frame_mtrand_sync_schedule(
    path: Path,
    *,
    seed: int,
    start_update: int,
    end_update: int,
    maximum_draws: int = DEFAULT_MAXIMUM_DRAWS,
) -> FrameMTRandSyncSchedule:
    """Verify a complete capture and derive one post-call state per update."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("frame schedule seed must fit uint32")
    if start_update <= 0 or end_update < start_update:
        raise ValueError("frame schedule update interval is invalid")
    if maximum_draws < 0:
        raise ValueError("frame schedule maximum draws is invalid")

    first_required = start_update - 1
    process_id, source_sha256, by_update = _load_capture(
        path,
        first_required_update=first_required,
        end_update=end_update,
    )
    missing_updates = [
        update
        for update in range(first_required, end_update + 1)
        if update not in by_update
    ]
    if missing_updates:
        raise ValueError(
            "frame schedule capture does not cover every requested update"
        )

    all_captures = tuple(
        capture
        for update in range(first_required, end_update + 1)
        for capture in by_update[update]
    )
    reconstructed = _reconstruct_states(
        seed=seed,
        captures=all_captures,
        maximum_draws=maximum_draws,
    )

    entries: list[FrameMTRandSyncEntry] = []
    for update in range(start_update, end_update + 1):
        source = by_update[update - 1][-1]
        source_draw_count, source_payload = reconstructed[source.sha256]
        unpacked = struct.unpack(
            f"<{MTRAND_STATE_WORDS + 1}I",
            source_payload,
        )
        rng = PopCapMTRandom(1)
        rng.load_state(
            unpacked[:MTRAND_STATE_WORDS],
            unpacked[MTRAND_STATE_WORDS],
        )
        output = rng.next_u31()
        post_payload = _state_payload(rng)
        post_sha256 = "sha256:" + hashlib.sha256(
            post_payload
        ).hexdigest()
        observed_digests = {
            capture.sha256 for capture in by_update[update]
        }
        observed_draw_counts = {
            reconstructed[digest][0] for digest in observed_digests
        }
        post_state_observed = post_sha256 in observed_digests
        if not post_state_observed:
            if observed_draw_counts == {source_draw_count}:
                # The global stream did not advance in this natural update.
                # Keep it out of the expected-hit schedule; a replay hit at
                # this update is then an explicit divergence.
                continue
            if (
                min(observed_draw_counts) >= source_draw_count
                and max(observed_draw_counts) >= source_draw_count + 1
            ):
                # Several synchronous calls can complete before the read-only
                # monitor gets a scheduling opportunity.  The seeded stream
                # still proves the first call's intermediate post-state.
                pass
            else:
                raise ValueError(
                    "frame schedule observed a non-forward MTRand transition "
                    f"at update {update}"
                )
        if (
            post_state_observed
            and source_draw_count + 1 not in observed_draw_counts
        ):
            raise ValueError(
                "frame schedule expected first call is not present in the "
                f"natural capture at update {update}"
            )
        entries.append(
            FrameMTRandSyncEntry(
                framework_update=update,
                source_framework_update=source.framework_update,
                source_native_game_time=source.native_game_time,
                source_draw_count=source_draw_count,
                source_index=source.index,
                source_state_sha256=source.sha256,
                output=output,
                post_draw_count=source_draw_count + 1,
                post_index=rng.index,
                post_state_sha256=post_sha256,
                post_state_observed=post_state_observed,
                post_payload=post_payload,
            )
        )

    return FrameMTRandSyncSchedule(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        seed=seed,
        start_update=start_update,
        end_update=end_update,
        entries=tuple(entries),
    )
