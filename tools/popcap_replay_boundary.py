"""Pure boundary checks for strict PopCap DMO replay tracing.

These helpers intentionally contain no Windows debugger imports so the replay
ordering rules can be tested on every development platform.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import struct


MAIN_DEMO_CALLER = 0x00686E70
DEFERRED_FILE_WRITE_CALLER = 0x00680FB2
DEFERRED_FILE_WRITE_RESUME = 0x00680FCD
REGISTRY_WRITE_DEMO_CALLER = 0x0067F13A
REGISTRY_WRITE_LIVE_PAYLOAD_COMMAND = 24
SERVICE_COMMAND_NUMBERS = frozenset((*range(10, 21), 22))
SERVICE_POLL_INTERVAL_SECONDS = 0.002
SERVICE_PROGRESS_PROBE_TIMEOUT_SECONDS = 1.0
STARTUP_POST_BYPASS_WORKER_SETTLE_SECONDS = 0.002
ATTACH_STABILIZATION_MAX_SKIPPED_HITS = 64
ATTACH_STABILIZATION_MAX_UPDATE_DELTA = 60
SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES = 15
FONT_CACHE_MANIFEST_PREFIX = "cached\\fonts\\600\\"
FONT_CACHE_MANIFEST_EXPECTED_ENTRIES = 35
MAIN_FILE_READ_CORRIDOR_MIN_BYTES = 16 * 1024
MAIN_FILE_READ_CORRIDOR_MAX_BYTES = 256 * 1024
MAIN_FILE_READ_CORRIDOR_MAX_ROWS = 32
POST_BLACKOUT_FILE_WRITE_SPILLOVER_MAX_ROWS = 32
SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS = 32
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4


def _is_successful_file_write_result(row: Mapping[str, object]) -> bool:
    """Return whether one row is an exact successful file-write result."""

    payload = row.get("payload")
    return (
        not bool(row["short_form"])
        and int(row["command_number"]) == 16
        and row.get("kind") == "file_write"
        and int(row["end"]) - int(row["start"]) == 11
        and isinstance(payload, dict)
        and set(payload) == {"success"}
        and payload["success"] is True
    )


def _is_uniform_successful_file_write_corridor(
    rows: list[dict[str, object]],
    start_index: int,
    end_index: int,
) -> bool:
    """Return whether a bounded corridor is one exact successful write run."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or end_index - start_index + 1
        > POST_BLACKOUT_FILE_WRITE_SPILLOVER_MAX_ROWS
    ):
        return False
    corridor = rows[start_index : end_index + 1]
    update = int(corridor[0]["update"])
    return all(
        int(row["update"]) == update
        and _is_successful_file_write_result(row)
        for row in corridor
    )


def _is_bounded_late_successful_file_write_corridor(
    rows: list[dict[str, object]],
    snapshot: Mapping[str, int],
    start_index: int,
    end_index: int,
) -> bool:
    """Return whether a post-load write run is late by at most one tick per row."""

    if not _is_uniform_successful_file_write_corridor(
        rows,
        start_index,
        end_index,
    ):
        return False
    corridor_update = int(rows[start_index]["update"])
    update_delta = snapshot["update"] - corridor_update
    return (
        snapshot.get("demo_loading_complete") == 1
        and snapshot["update"] == snapshot["last_demo_update"]
        and 1 <= update_delta <= end_index - start_index + 1
    )


def _is_bounded_late_preloading_failed_file_write_corridor(
    rows: list[dict[str, object]],
    snapshot: Mapping[str, int],
    start_index: int,
    end_index: int,
) -> bool:
    """Accept one exact pre-loading false-write corridor one tick late."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or end_index - start_index + 1
        > FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
    ):
        return False
    corridor = rows[start_index : end_index + 1]
    previous_end: int | None = None
    previous_update: int | None = None
    for row in corridor:
        row_start = int(row["start"])
        row_end = int(row["end"])
        row_update = int(row["update"])
        if (
            (previous_end is not None and row_start != previous_end)
            or (previous_update is not None and row_update < previous_update)
            or row_update < 0
            or bool(row["short_form"])
            or int(row["command_number"]) != 16
            or row.get("kind") != "file_write"
            or row_end - row_start != 11
            or not isinstance(row.get("payload"), dict)
            or set(row["payload"]) != {"success"}
            or row["payload"]["success"] is not False
        ):
            return False
        previous_end = row_end
        previous_update = row_update
    assert previous_update is not None
    return (
        snapshot.get("demo_loading_complete") == 0
        and snapshot["update"] == snapshot["last_demo_update"]
        and snapshot["update"] == previous_update + 1
    )


def _is_exact_startup_font_cache_failed_write_partition(
    rows: list[dict[str, object]],
    *,
    start_index: int,
    end_index: int,
    direct_row_indices: tuple[int, ...],
    manifest_entry_count: int,
) -> bool:
    """Prove direct calls plus one continuation cover all startup writes."""

    if (
        manifest_entry_count != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or not direct_row_indices
        or tuple(sorted(set(direct_row_indices))) != direct_row_indices
    ):
        return False
    loading_complete_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("kind") == "loading_complete"
            and int(row["command_number"]) == 9
        ),
        None,
    )
    if loading_complete_index is None or end_index >= loading_complete_index:
        return False
    startup_rows = tuple(
        index
        for index, row in enumerate(rows[:loading_complete_index])
        if (
            not bool(row["short_form"])
            and int(row["command_number"]) == 16
            and row.get("kind") == "file_write"
            and int(row["end"]) - int(row["start"]) == 11
            and row.get("payload") == {"success": False}
        )
    )
    continuation_rows = tuple(range(start_index, end_index + 1))
    return (
        len(startup_rows) == manifest_entry_count
        and direct_row_indices + continuation_rows == startup_rows
        and _is_bounded_late_preloading_failed_file_write_corridor(
            rows,
            {
                "update": int(rows[end_index]["update"]) + 1,
                "last_demo_update": int(rows[end_index]["update"]) + 1,
                "demo_loading_complete": 0,
            },
            start_index,
            end_index,
        )
    )


def _normalize_font_cache_member(value: str) -> str | None:
    """Return one canonical retail font-cache member from a live path."""

    normalized = value.replace("/", "\\").casefold()
    marker = normalized.rfind(FONT_CACHE_MANIFEST_PREFIX)
    if marker < 0:
        return None
    member = normalized[marker:]
    if (
        not member.endswith(".cfw2")
        or "\x00" in member
        or member == FONT_CACHE_MANIFEST_PREFIX + ".cfw2"
    ):
        return None
    return member


def _board_seed_override_applies(
    hit_index: int,
    skip_count: int,
) -> bool:
    """Return whether this zero-based reseed-site hit is the target call."""

    if hit_index < 0 or skip_count < 0:
        raise ValueError(
            "board seed hit index and skip count must be non-negative"
        )
    return hit_index >= skip_count


def _source_bound_board_anchor_applies(
    *,
    framework_update: int,
    target_framework_update: int,
    thread_id: int,
    main_thread_id: int,
    rng_owner_vtable: int,
    expected_board_vtable: int,
    observed_seed: int,
    expected_seed: int,
    global_post_state_match: bool,
    bounded_global_correction: bool = False,
    active_board_context_match: bool,
    attach_stabilization_collision: bool,
) -> bool:
    """Match one exact source-bound board-seed RNG handoff.

    The object owning the seeded MTRand is not SexyAppBase's active Board.
    Bind both identities independently and require the complete natural
    post-call global state, rather than conflating their addresses.
    """

    values = (
        framework_update,
        target_framework_update,
        thread_id,
        main_thread_id,
        rng_owner_vtable,
        expected_board_vtable,
        observed_seed,
        expected_seed,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in values
    ):
        raise ValueError("source-bound board selector values must be unsigned")
    if main_thread_id == 0 or expected_board_vtable == 0:
        raise ValueError("source-bound board identities must be positive")
    flags = (
        global_post_state_match,
        bounded_global_correction,
        active_board_context_match,
        attach_stabilization_collision,
    )
    if any(not isinstance(value, bool) for value in flags):
        raise ValueError("source-bound board selector flags must be boolean")
    return (
        framework_update == target_framework_update
        and thread_id == main_thread_id
        and rng_owner_vtable == expected_board_vtable
        and (
            (
                observed_seed == expected_seed
                and global_post_state_match
            )
            or bounded_global_correction
        )
        and active_board_context_match
        and not attach_stabilization_collision
    )


def _source_bound_global_correction_eligible(
    *,
    allow_correction: bool,
    live_post_draw_verified: bool,
    correction_index_delta: int | None,
    observed_seed_match: bool,
    global_post_state_match: bool,
) -> bool:
    """Decide correction eligibility from the current anchor observation."""

    boolean_values = (
        allow_correction,
        live_post_draw_verified,
        observed_seed_match,
        global_post_state_match,
    )
    if not all(isinstance(value, bool) for value in boolean_values) or (
        correction_index_delta is not None
        and (
            isinstance(correction_index_delta, bool)
            or not isinstance(correction_index_delta, int)
            or correction_index_delta < 0
        )
    ):
        raise ValueError("source-bound correction inputs are invalid")
    return (
        allow_correction
        and live_post_draw_verified
        and correction_index_delta is not None
        and not (observed_seed_match and global_post_state_match)
    )


def _source_bound_global_post_correction_delta(
    live_post: bytes,
    expected_post: bytes,
    *,
    maximum_draw_delta: int = SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS,
) -> int | None:
    """Return one bounded same-state-word MTRand index correction."""

    if (
        not isinstance(live_post, bytes)
        or not isinstance(expected_post, bytes)
        or len(live_post) != MTRAND_STATE_BYTES
        or len(expected_post) != MTRAND_STATE_BYTES
        or isinstance(maximum_draw_delta, bool)
        or not isinstance(maximum_draw_delta, int)
        or maximum_draw_delta < 0
    ):
        raise ValueError("source-bound global correction inputs are invalid")
    live_index = struct.unpack_from(
        "<I",
        live_post,
        MTRAND_STATE_WORDS * 4,
    )[0]
    expected_index = struct.unpack_from(
        "<I",
        expected_post,
        MTRAND_STATE_WORDS * 4,
    )[0]
    if (
        live_index > MTRAND_STATE_WORDS
        or expected_index > MTRAND_STATE_WORDS
        or live_post[: MTRAND_STATE_WORDS * 4]
        != expected_post[: MTRAND_STATE_WORDS * 4]
    ):
        return None
    delta = expected_index - live_index
    return delta if abs(delta) <= maximum_draw_delta else None


def _service_block(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    allow_prepared_command: bool = False,
) -> tuple[int, int, int] | None:
    """Return the contiguous service run a racing main call must yield to."""

    if (
        snapshot["return_address"] != MAIN_DEMO_CALLER
        or snapshot["command_number"] not in SERVICE_COMMAND_NUMBERS
        or snapshot["needs_command"] not in (
            0,
            1 if allow_prepared_command else 0,
        )
    ):
        return None
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return None
    index, row = matched
    if (
        int(row["command_number"]) != snapshot["command_number"]
        or int(row["update"]) != snapshot["last_demo_update"]
    ):
        return None
    if snapshot["needs_command"] == 1 and (
        bool(row["short_form"])
        or snapshot["buffer_read_bit_position"]
        != int(row["start"]) + 10
    ):
        return None
    block_update = int(row["update"])
    end_index = index
    while end_index + 1 < len(rows):
        following = rows[end_index + 1]
        if (
            int(following["command_number"])
            not in SERVICE_COMMAND_NUMBERS
            or int(following["update"]) != block_update
        ):
            break
        end_index += 1
    return index, end_index, int(rows[end_index]["end"])


def _service_settle_end(
    rows: list[dict[str, object]],
    end_index: int,
    target_end: int,
) -> int:
    """Allow one parsed header at the next service/settle command."""

    if end_index < 0 or end_index >= len(rows) or target_end < 0:
        raise ValueError("service settle boundary is invalid")
    if end_index + 1 >= len(rows):
        return target_end
    following = rows[end_index + 1]
    if int(following["command_number"]) in (
        SERVICE_COMMAND_NUMBERS | {9, 31}
    ):
        return int(following["end"])
    return target_end


def _service_continuation_corridor(
    rows: list[dict[str, object]],
    start_index: int,
) -> tuple[int, int]:
    """Return the exact contiguous service-only corridor from one row."""

    if (
        start_index < 0
        or start_index >= len(rows)
        or int(rows[start_index]["command_number"])
        not in SERVICE_COMMAND_NUMBERS
    ):
        raise ValueError("service continuation start is invalid")
    end_index = start_index
    while end_index + 1 < len(rows):
        following = rows[end_index + 1]
        if int(following["command_number"]) not in SERVICE_COMMAND_NUMBERS:
            break
        end_index += 1
    return end_index, int(rows[end_index]["end"])


def _audited_main_file_read_corridor(
    rows: list[dict[str, object]],
    *,
    start_index: int,
    end_index: int,
) -> dict[str, int | str] | None:
    """Identify one exact main-owned repeated save-hydration corridor.

    The main thread may consume this corridor natively only after the normal
    service broker has proved that no worker thread advances it.  Eligibility
    is intentionally narrow: one bounded same-update run of read-only service
    commands must contain exactly one large successful file-read payload, and
    the identical large payload must occur exactly twice in the immutable DMO.
    The tracer still checks every live command boundary while native replay is
    active; this helper authorizes scheduling only and never changes payloads.
    """

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or end_index - start_index + 1 > MAIN_FILE_READ_CORRIDOR_MAX_ROWS
    ):
        return None
    corridor = rows[start_index : end_index + 1]
    update = int(corridor[0]["update"])
    if any(
        bool(row["short_form"])
        or int(row["update"]) != update
        or int(row["command_number"]) not in {11, 12, 14, 15}
        for row in corridor
    ):
        return None
    large_reads: list[tuple[int, int, str]] = []
    file_read_count = 0
    for row_index, row in enumerate(
        corridor,
        start=start_index,
    ):
        if int(row["command_number"]) != 15:
            continue
        file_read_count += 1
        payload = row.get("payload")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        payload_bytes = data.get("bytes")
        payload_sha256 = data.get("sha256")
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes < 0
            or not isinstance(payload_sha256, str)
            or len(payload_sha256) != 71
            or not payload_sha256.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in payload_sha256.removeprefix("sha256:")
            )
        ):
            return None
        if payload_bytes >= MAIN_FILE_READ_CORRIDOR_MIN_BYTES:
            if payload_bytes > MAIN_FILE_READ_CORRIDOR_MAX_BYTES:
                return None
            large_reads.append(
                (row_index, payload_bytes, payload_sha256)
            )
    if file_read_count == 0 or len(large_reads) != 1:
        return None
    large_row_index, large_bytes, large_sha256 = large_reads[0]
    matching_large_rows = 0
    for row in rows:
        if int(row["command_number"]) != 15:
            continue
        payload = row.get("payload")
        data = payload.get("data") if isinstance(payload, dict) else None
        if (
            isinstance(data, dict)
            and data.get("bytes") == large_bytes
            and data.get("sha256") == large_sha256
            and payload.get("success") is True
        ):
            matching_large_rows += 1
    if matching_large_rows != 2:
        return None
    return {
        "start_index": start_index,
        "end_index": end_index,
        "update": update,
        "file_read_count": file_read_count,
        "large_file_read_row_index": large_row_index,
        "large_file_read_bytes": large_bytes,
        "large_file_read_sha256": large_sha256,
        "matching_large_payload_occurrences": matching_large_rows,
    }


def _audited_service_payload_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    start_index: int,
    end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int = -1,
    previous_command_order: int = -1,
    previous_reentry_kind: int = 0,
    command_order_offset: int = 0,
) -> tuple[int | None, str | None]:
    """Audit a non-boundary PrepareDemoCommand call inside service payload."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < -1
        or previous_command_order < -1
        or previous_reentry_kind not in (0, 1, 2, 3)
        or command_order_offset < 0
    ):
        raise ValueError("service payload re-entry boundary is invalid")
    command_bit_position = snapshot["command_bit_position"]
    matched_index: int | None = None
    for index in range(start_index, end_index + 1):
        row = rows[index]
        if (
            not bool(row["short_form"])
            and int(row["command_number"]) in SERVICE_COMMAND_NUMBERS
            and int(row["start"]) + 10 == command_bit_position
        ):
            matched_index = index
            break
    if matched_index is None:
        return None, (
            "service payload re-entry is not an exact long-form header: "
            f"{command_bit_position}"
        )
    expected_order = matched_index + 1 - command_order_offset
    corridor_end = int(rows[end_index]["end"])
    observed = {
        "return_address": snapshot["return_address"],
        "command_number": snapshot["command_number"],
        "is_short": snapshot["is_short"],
        "command_order": snapshot["command_order"],
    }
    matched_row = rows[matched_index]
    matched_payload = matched_row.get("payload")
    registry_write_payload_callback = (
        matched_row.get("kind") == "registry_write"
        and int(matched_row["command_number"]) == 12
        and int(matched_row["end"]) - int(matched_row["start"]) == 11
        and isinstance(matched_payload, dict)
        and set(matched_payload) == {"success"}
        and isinstance(matched_payload["success"], bool)
    )
    expected = {
        "return_address": (
            REGISTRY_WRITE_DEMO_CALLER
            if registry_write_payload_callback
            else MAIN_DEMO_CALLER
        ),
        "command_number": (
            REGISTRY_WRITE_LIVE_PAYLOAD_COMMAND
            if registry_write_payload_callback
            else 0
        ),
        "is_short": 0,
        "command_order": expected_order,
    }
    for key, value in expected.items():
        if observed[key] != value:
            return matched_index, (
                f"service payload re-entry {matched_index} {key} mismatch: "
                f"{observed[key]} != {value}"
            )
    read_bit_position = snapshot["buffer_read_bit_position"]
    if snapshot["needs_command"] == 0:
        expected_read = command_bit_position + 10
        if read_bit_position != expected_read:
            return matched_index, (
                "service payload header read mismatch: "
                f"{read_bit_position} != {expected_read}"
            )
    elif snapshot["needs_command"] == 1:
        same_payload_forced_read = (
            previous_reentry_kind == 1
            and command_bit_position == previous_command_bit_position
            and snapshot["command_order"] == previous_command_order
        )
        exact_start_payload_entry = (
            previous_reentry_kind in (2, 3)
            and previous_command_bit_position
            == int(rows[matched_index]["start"])
            and command_bit_position == previous_command_bit_position + 10
            and snapshot["command_order"] == previous_command_order + 1
        )
        exact_registry_payload_chain_entry = False
        if (
            registry_write_payload_callback
            and previous_reentry_kind == 1
            and matched_index > start_index
        ):
            previous_row = rows[matched_index - 1]
            previous_payload = previous_row.get("payload")
            exact_registry_payload_chain_entry = (
                previous_row.get("kind") == "registry_write"
                and not bool(previous_row["short_form"])
                and int(previous_row["command_number"]) == 12
                and int(previous_row["end"])
                - int(previous_row["start"])
                == 11
                and int(previous_row["end"])
                == int(matched_row["start"])
                and int(previous_row["update"])
                == int(matched_row["update"])
                and isinstance(previous_payload, dict)
                and set(previous_payload) == {"success"}
                and isinstance(previous_payload["success"], bool)
                and previous_command_bit_position
                == int(previous_row["start"]) + 10
                and previous_read_bit_position == command_bit_position
                and snapshot["command_order"]
                == previous_command_order + 1
            )
        if not (
            same_payload_forced_read
            or exact_start_payload_entry
            or exact_registry_payload_chain_entry
        ):
            return matched_index, (
                "service payload forced read changed its prepared command"
            )
        exact_service_ends = {
            int(rows[index]["end"])
            for index in range(matched_index, end_index + 1)
        }
        registry_following_payload_start = False
        if (
            registry_write_payload_callback
            and matched_index + 1 <= end_index
        ):
            following_row = rows[matched_index + 1]
            following_payload = following_row.get("payload")
            registry_following_payload_start = (
                following_row.get("kind") == "registry_write"
                and not bool(following_row["short_form"])
                and int(following_row["command_number"]) == 12
                and int(following_row["start"])
                == int(matched_row["end"])
                and int(following_row["end"])
                - int(following_row["start"])
                == 11
                and int(following_row["update"])
                == int(matched_row["update"])
                and isinstance(following_payload, dict)
                and set(following_payload) == {"success"}
                and isinstance(following_payload["success"], bool)
                and read_bit_position
                == int(following_row["start"]) + 10
            )
        if (
            read_bit_position not in exact_service_ends
            and not registry_following_payload_start
        ):
            return matched_index, (
                "service payload forced read is not at a service row end: "
                f"{read_bit_position}"
            )
    else:
        return matched_index, (
            "service payload needs_command is invalid: "
            f"{snapshot['needs_command']}"
        )
    if read_bit_position <= previous_read_bit_position:
        return matched_index, (
            "service payload re-entry did not advance: "
            f"{read_bit_position} <= "
            f"{previous_read_bit_position}"
        )
    if read_bit_position > corridor_end:
        return matched_index, (
            "service payload re-entry escaped its corridor: "
            f"{read_bit_position} > {corridor_end}"
        )
    return matched_index, None


def _audited_terminal_registry_write_eof_exit(
    rows: list[dict[str, object]],
    continuation: Mapping[str, object],
    *,
    process_id: int,
    exited: bool,
    exit_code: int | None,
    command_order_offset: int,
) -> tuple[dict[str, int | bool | str] | None, str | None]:
    """Audit a normal exit at the final registry-write result bit.

    The retail registry callback chains each successful result bit with the
    next ten-bit header.  At DMO EOF there is no following header, and the
    process can exit normally while parked exactly at the final result bit.
    This accepts only the maximal terminal corridor and exactly one remaining
    successful payload bit.
    """

    expected_keys = {
        "base",
        "thread_id",
        "start_index",
        "end_index",
        "end_bit_position",
        "last_read_bit_position",
        "last_command_bit_position",
        "last_command_order",
        "last_reentry_update",
        "last_reentry_kind",
        "reentries",
        "allow_bounded_late_successful_file_write_corridor",
        "allow_bounded_late_preloading_failed_file_write_corridor",
    }
    if (
        process_id <= 0
        or exited is not True
        or exit_code != 0
        or command_order_offset < 0
        or set(continuation) != expected_keys
    ):
        return None, "terminal registry-write exit precondition mismatch"
    integer_fields = expected_keys - {
        "allow_bounded_late_successful_file_write_corridor",
        "allow_bounded_late_preloading_failed_file_write_corridor",
    }
    if any(
        isinstance(continuation.get(name), bool)
        or not isinstance(continuation.get(name), int)
        for name in integer_fields
    ):
        return None, "terminal registry-write continuation is malformed"
    if (
        continuation["base"] <= 0
        or continuation["thread_id"] <= 0
        or continuation[
            "allow_bounded_late_successful_file_write_corridor"
        ]
        is not False
        or continuation[
            "allow_bounded_late_preloading_failed_file_write_corridor"
        ]
        is not False
    ):
        return None, "terminal registry-write continuation metadata mismatch"
    start_index = int(continuation["start_index"])
    end_index = int(continuation["end_index"])
    if (
        start_index < 0
        or end_index <= start_index
        or end_index != len(rows) - 1
    ):
        return None, "terminal registry-write corridor is not maximal at EOF"

    corridor_update = int(rows[start_index]["update"])
    previous_end: int | None = None
    for index in range(start_index, end_index + 1):
        row = rows[index]
        payload = row.get("payload")
        row_start = int(row["start"])
        row_end = int(row["end"])
        if (
            row.get("kind") != "registry_write"
            or bool(row["short_form"])
            or int(row["command_number"]) != 12
            or row_end - row_start != 11
            or int(row["update"]) != corridor_update
            or (previous_end is not None and row_start != previous_end)
            or not isinstance(payload, dict)
            or set(payload) != {"success"}
            or payload["success"] is not True
        ):
            return None, (
                "terminal registry-write corridor contains an ineligible row"
            )
        previous_end = row_end
    if start_index > 0:
        previous_row = rows[start_index - 1]
        previous_payload = previous_row.get("payload")
        if (
            previous_row.get("kind") == "registry_write"
            and not bool(previous_row["short_form"])
            and int(previous_row["command_number"]) == 12
            and int(previous_row["end"]) == int(rows[start_index]["start"])
            and int(previous_row["update"]) == corridor_update
            and isinstance(previous_payload, dict)
            and previous_payload == {"success": True}
        ):
            return None, "terminal registry-write corridor omits its prefix"

    terminal_row = rows[end_index]
    penultimate_row = rows[end_index - 1]
    terminal_payload_bit = int(terminal_row["start"]) + 10
    row_count = end_index - start_index + 1
    if (
        continuation["end_bit_position"] != int(terminal_row["end"])
        or continuation["last_read_bit_position"] != terminal_payload_bit
        or continuation["last_command_bit_position"]
        != int(penultimate_row["start"]) + 10
        or continuation["last_command_order"]
        != end_index - command_order_offset
        or continuation["last_reentry_update"] != corridor_update
        or continuation["last_reentry_kind"] != 1
        or continuation["reentries"] != row_count
        or int(terminal_row["end"]) - terminal_payload_bit != 1
    ):
        return None, "terminal registry-write EOF boundary mismatch"
    return {
        "process_id": process_id,
        "thread_id": int(continuation["thread_id"]),
        "framework_update": int(continuation["last_reentry_update"]),
        "start_index": start_index,
        "end_index": end_index,
        "terminal_row_index": end_index,
        "terminal_row_start_bit_position": int(terminal_row["start"]),
        "terminal_payload_bit_position": terminal_payload_bit,
        "terminal_row_end_bit_position": int(terminal_row["end"]),
        "last_command_bit_position": int(
            continuation["last_command_bit_position"]
        ),
        "last_read_bit_position": int(
            continuation["last_read_bit_position"]
        ),
        "last_command_order": int(continuation["last_command_order"]),
        "command_order_offset": command_order_offset,
        "reentries": int(continuation["reentries"]),
        "unconsumed_result_bits": 1,
        "recorded_success": True,
        "exit_code": 0,
        "mechanism": "normal_exit_at_terminal_registry_write_result_bit",
    }, None


def _audited_service_prepared_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    start_index: int,
    end_index: int,
    previous_read_bit_position: int,
    previous_command_order: int,
    previous_command_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int = 0,
    require_before_recorded_update: bool = True,
    allow_bounded_late_successful_file_write_corridor: bool = False,
    allow_bounded_late_preloading_failed_file_write_corridor: bool = False,
) -> tuple[int | None, str | None]:
    """Audit an exact service start prepared before its recorded update."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_order < -1
        or previous_command_bit_position < -1
        or previous_reentry_kind not in (1, 3)
        or command_order_offset < 0
        or not isinstance(require_before_recorded_update, bool)
        or not isinstance(
            allow_bounded_late_successful_file_write_corridor,
            bool,
        )
        or not isinstance(
            allow_bounded_late_preloading_failed_file_write_corridor,
            bool,
        )
    ):
        raise ValueError("service prepared re-entry boundary is invalid")
    command_start = snapshot["command_bit_position"]
    matched_index: int | None = None
    for index in range(start_index, end_index + 1):
        if int(rows[index]["start"]) == command_start:
            matched_index = index
            break
    if matched_index is None:
        return None, "service prepared re-entry has no exact offline start"
    row = rows[matched_index]
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "buffer_read_bit_position": command_start + 10,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return matched_index, (
                f"service prepared re-entry {matched_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    new_prepared_entry = (
        previous_reentry_kind == 1
        and command_start == previous_read_bit_position
        and snapshot["command_order"] == previous_command_order + 1
        and snapshot["buffer_read_bit_position"]
        > previous_read_bit_position
    )
    header_to_prepared_transition = (
        previous_reentry_kind == 3
        and command_start == previous_command_bit_position
        and snapshot["command_order"] == previous_command_order
        and snapshot["buffer_read_bit_position"]
        == previous_read_bit_position
    )
    if not (new_prepared_entry or header_to_prepared_transition):
        return matched_index, (
            "service prepared re-entry does not follow its audited payload "
            "or exact header state"
        )
    if (
        bool(row["short_form"])
        or int(row["command_number"]) not in SERVICE_COMMAND_NUMBERS
        or int(row["end"]) - int(row["start"]) <= 10
    ):
        return matched_index, (
            f"service prepared re-entry row is ineligible: {matched_index}"
        )
    if snapshot["command_order"] > matched_index - command_order_offset:
        return matched_index, (
            "service prepared re-entry command order moved past its row"
        )
    row_update = int(row["update"])
    if require_before_recorded_update:
        if not (
            snapshot["last_demo_update"] < row_update
            and snapshot["update"] < row_update
        ):
            return matched_index, (
                "service prepared re-entry is not before its recorded update: "
                f"framework={snapshot['update']}, "
                f"last={snapshot['last_demo_update']}, row={row_update}"
            )
    else:
        corridor = rows[start_index : end_index + 1]
        corridor_updates = [int(item["update"]) for item in corridor]
        exact_corridor_timing = (
            snapshot["update"] == snapshot["last_demo_update"]
            and min(corridor_updates)
            <= snapshot["update"]
            <= max(corridor_updates)
        )
        bounded_late_file_write_spillover = (
            (
                allow_bounded_late_successful_file_write_corridor
                and _is_bounded_late_successful_file_write_corridor(
                    rows,
                    snapshot,
                    start_index,
                    end_index,
                )
            )
            or (
                allow_bounded_late_preloading_failed_file_write_corridor
                and _is_bounded_late_preloading_failed_file_write_corridor(
                    rows,
                    snapshot,
                    start_index,
                    end_index,
                )
            )
        )
        if (
            not exact_corridor_timing
            and not bounded_late_file_write_spillover
        ):
            return matched_index, (
                "service prepared payload re-entry update is outside its "
                "audited corridor: "
                f"framework={snapshot['update']}, "
                f"last={snapshot['last_demo_update']}"
            )
    return matched_index, None


def _audited_service_header_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    start_index: int,
    end_index: int,
    previous_read_bit_position: int,
    previous_command_order: int,
    previous_reentry_kind: int,
    command_order_offset: int = 0,
    require_before_recorded_update: bool = True,
    allow_bounded_late_successful_file_write_corridor: bool = False,
    allow_bounded_late_preloading_failed_file_write_corridor: bool = False,
) -> tuple[int | None, str | None]:
    """Audit an exact service header reached early from a payload read."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_order < -1
        or previous_reentry_kind != 1
        or command_order_offset < 0
        or not isinstance(require_before_recorded_update, bool)
        or not isinstance(
            allow_bounded_late_successful_file_write_corridor,
            bool,
        )
        or not isinstance(
            allow_bounded_late_preloading_failed_file_write_corridor,
            bool,
        )
    ):
        raise ValueError("service header re-entry boundary is invalid")
    command_start = snapshot["command_bit_position"]
    matched_index: int | None = None
    for index in range(start_index, end_index + 1):
        if int(rows[index]["start"]) == command_start:
            matched_index = index
            break
    if matched_index is None:
        return None, "service header re-entry has no exact offline start"
    row = rows[matched_index]
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "command_order": previous_command_order + 1,
        "buffer_read_bit_position": command_start + 10,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return matched_index, (
                f"service header re-entry {matched_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    if (
        bool(row["short_form"])
        or int(row["command_number"]) not in SERVICE_COMMAND_NUMBERS
        or int(row["end"]) - int(row["start"]) <= 10
    ):
        return matched_index, (
            f"service header re-entry row is ineligible: {matched_index}"
        )
    if command_start != previous_read_bit_position:
        return matched_index, (
            "service header re-entry did not begin at the prior payload end: "
            f"{command_start} != {previous_read_bit_position}"
        )
    if snapshot["command_order"] > matched_index - command_order_offset:
        return matched_index, (
            "service header re-entry command order moved past its row"
        )
    row_update = int(row["update"])
    if require_before_recorded_update:
        if not (
            snapshot["last_demo_update"] < row_update
            and snapshot["update"] < row_update
        ):
            return matched_index, (
                "service header re-entry is not before its recorded update: "
                f"framework={snapshot['update']}, "
                f"last={snapshot['last_demo_update']}, row={row_update}"
            )
    else:
        corridor = rows[start_index : end_index + 1]
        corridor_updates = [int(item["update"]) for item in corridor]
        exact_corridor_timing = (
            snapshot["update"] == snapshot["last_demo_update"]
            and min(corridor_updates)
            <= snapshot["update"]
            <= max(corridor_updates)
        )
        bounded_late_file_write_spillover = (
            (
                allow_bounded_late_successful_file_write_corridor
                and _is_bounded_late_successful_file_write_corridor(
                    rows,
                    snapshot,
                    start_index,
                    end_index,
                )
            )
            or (
                allow_bounded_late_preloading_failed_file_write_corridor
                and _is_bounded_late_preloading_failed_file_write_corridor(
                    rows,
                    snapshot,
                    start_index,
                    end_index,
                )
            )
        )
        if (
            not exact_corridor_timing
            and not bounded_late_file_write_spillover
        ):
            return matched_index, (
                "service header payload re-entry update is outside its "
                "audited corridor: "
                f"framework={snapshot['update']}, "
                f"last={snapshot['last_demo_update']}"
            )
    return matched_index, None


def _audited_service_file_write_header_claim(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    start_index: int,
    end_index: int,
    previous_read_bit_position: int,
    previous_command_order: int,
    previous_reentry_kind: int,
    command_order_offset: int = 0,
    require_before_recorded_update: bool = True,
) -> tuple[int | None, str | None]:
    """Audit a real file-write worker claiming one prepared service row."""

    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_order < -1
        or previous_reentry_kind != 1
        or command_order_offset < 0
        or not isinstance(require_before_recorded_update, bool)
    ):
        raise ValueError("service file-write header claim is invalid")
    command_start = snapshot["command_bit_position"]
    matched_index: int | None = None
    for index in range(start_index, end_index + 1):
        if int(rows[index]["start"]) == command_start:
            matched_index = index
            break
    if matched_index is None:
        return None, "service file-write claim has no exact offline start"
    row = rows[matched_index]
    payload = row.get("payload")
    expected = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "needs_command": 0,
        "command_number": 16,
        "is_short": 0,
        "command_order": previous_command_order + 1,
        "buffer_read_bit_position": command_start + 10,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return matched_index, (
                f"service file-write claim {matched_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    if (
        bool(row["short_form"])
        or int(row["command_number"]) != 16
        or int(row["end"]) - int(row["start"]) != 11
        or not isinstance(payload, dict)
        or not isinstance(payload.get("success"), bool)
    ):
        return matched_index, (
            f"service file-write claim row is ineligible: {matched_index}"
        )
    if command_start != previous_read_bit_position:
        return matched_index, (
            "service file-write claim did not begin at the prior payload end: "
            f"{command_start} != {previous_read_bit_position}"
        )
    if snapshot["command_order"] > matched_index - command_order_offset:
        return matched_index, (
            "service file-write claim command order moved past its row"
        )
    row_update = int(row["update"])
    if require_before_recorded_update:
        if not (
            snapshot["last_demo_update"] < row_update
            and snapshot["update"] < row_update
        ):
            return matched_index, (
                "service file-write claim is not before its recorded update: "
                f"framework={snapshot['update']}, "
                f"last={snapshot['last_demo_update']}, row={row_update}"
            )
    else:
        corridor_updates = [
            int(rows[index]["update"])
            for index in range(start_index, end_index + 1)
        ]
        if (
            snapshot["update"] != snapshot["last_demo_update"]
            or not (
                min(corridor_updates)
                <= snapshot["update"]
                <= max(corridor_updates)
            )
        ):
            return matched_index, (
                "service file-write claim update is outside its audited "
                "corridor"
            )
    return matched_index, None


def _audited_service_file_write_payload_completion(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    row_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_reentry_kind: int,
    command_order_offset: int = 0,
) -> tuple[int | None, str | None]:
    """Audit the exact one-bit result after a real startup write claim.

    A loading worker can enter the real file-write wrapper after a brokered
    failed-write row has prepared the immediately following header.  The main
    playback thread then observes the same immutable command with only its
    recorded one-bit ``False`` result consumed.  Bind both sides of that
    handoff so an arbitrary prepared command cannot be treated as progress.
    """

    if (
        row_index < 0
        or row_index >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_reentry_kind != 13
        or command_order_offset < 0
    ):
        raise ValueError(
            "service file-write payload completion boundary is invalid"
        )
    row = rows[row_index]
    row_start = int(row["start"])
    row_end = int(row["end"])
    row_update = int(row["update"])
    if (
        bool(row["short_form"])
        or int(row["command_number"]) != 16
        or row.get("kind") != "file_write"
        or row.get("payload") != {"success": False}
        or row_end - row_start != 11
    ):
        return row_index, (
            "service file-write payload completion row is ineligible"
        )
    expected_order = row_index - command_order_offset
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 16,
        "is_short": 0,
        "command_order": expected_order,
        "command_bit_position": row_start,
        "buffer_read_bit_position": row_end,
        "update": row_update,
        "last_demo_update": row_update,
        "demo_loading_complete": 0,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return row_index, (
                "service file-write payload completion "
                f"{key} mismatch: {snapshot[key]} != {value}"
            )
    if (
        previous_command_bit_position != row_start
        or previous_read_bit_position != row_start + 10
        or previous_command_order != expected_order
        or snapshot["buffer_read_bit_position"]
        != previous_read_bit_position + 1
    ):
        return row_index, (
            "service file-write payload completion changed its exact "
            "claimed header state"
        )
    return row_index, None


def _may_bypass_stalled_service_block(
    snapshot: dict[str, int],
    *,
    already_bypassed: bool,
    allow_prepared_orphan: bool = False,
) -> bool:
    """Allow one exact main-thread continuation when no service thread moved."""

    return (
        snapshot["return_address"] == MAIN_DEMO_CALLER
        and not already_bypassed
        and (
            snapshot["needs_command"] == 0
            or (
                allow_prepared_orphan
                and snapshot["needs_command"] == 1
            )
        )
    )


def _may_adopt_prepared_service_block(
    *,
    already_bypassed: bool,
    allow_orphan: bool,
    orphan_already_used: bool,
) -> bool:
    """Allow a known retry or one exact prepared block after reattachment."""

    return already_bypassed or (
        allow_orphan and not orphan_already_used
    )


def _may_broker_terminal_prepared_service_row(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    prepared_row: int,
    corridor_end_index: int,
) -> bool:
    """Identify the final service payload handoff without widening parsing.

    The long-form file-write result bit follows its ten-bit command header.
    If the main loop re-enters while the final corridor row is prepared but
    that result bit is still unread, the original service thread must finish
    it.  Letting the main loop parse again would reinterpret the payload as a
    new command header and permanently desynchronize the DMO stream.
    """

    if (
        prepared_row < 0
        or corridor_end_index < 0
        or prepared_row >= len(rows)
        or corridor_end_index >= len(rows)
    ):
        raise ValueError("terminal prepared service row is invalid")
    row = rows[prepared_row]
    return (
        prepared_row == corridor_end_index
        and int(row["command_number"]) in SERVICE_COMMAND_NUMBERS
        and not bool(row["short_form"])
        and snapshot["return_address"] == MAIN_DEMO_CALLER
        and snapshot["needs_command"] == 1
        and snapshot["is_short"] == 0
        and snapshot["command_number"] == int(row["command_number"])
        and snapshot["command_bit_position"] == int(row["start"])
        and snapshot["buffer_read_bit_position"]
        == int(row["start"]) + 10
    )


def _snapshot_boundary_failure(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    snapshot: dict[str, int],
    *,
    command_order_offset: int = 0,
) -> str | None:
    """Describe the first offline/live command-boundary mismatch, if any."""

    if command_order_offset < 0:
        raise ValueError("command order offset must be non-negative")
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return (
            "command start is absent from the offline DMO: "
            f"{snapshot['command_bit_position']}"
        )
    index, row = matched
    expected = {
        "command_order": index - command_order_offset,
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "last_demo_update": int(row["update"]),
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return (
                f"offline boundary {index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    return None


def _audited_deferred_file_write_result(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    command_order_offset: int,
    accounted_rows: tuple[int, ...],
    discharge_count: int,
    pre_attach_rows: tuple[int, ...] = (),
    pre_attach_before_update: int | None = None,
    excluded_accounted_rows: tuple[int, ...] = (),
    service_consumed_rows: tuple[int, ...] = (),
    allow_one_tick_overdue_prepared: bool = False,
    allow_pre_loading_after_service_continuation: bool = False,
) -> tuple[int | None, bool | None, str | None]:
    """Map one late file-write playback call to an already consumed result.

    The retail file-write wrapper calls ``PrepareDemoCommand`` at
    ``DEFERRED_FILE_WRITE_CALLER`` and then unconditionally consumes one result
    bit.  Startup service races can consume exact file-write rows without that
    wrapper entering PrepareDemoCommand, leaving the command-order offset as a
    concrete debt.  A delayed wrapper call may discharge only one such audited
    row while the live stream is parked at a complete, due, non-file-write
    boundary.  A genuinely due file-write row is left to normal replay.
    """

    if (
        command_order_offset < 0
        or discharge_count < 0
        or not isinstance(allow_one_tick_overdue_prepared, bool)
    ):
        raise ValueError("deferred file-write counters must be non-negative")
    if snapshot["return_address"] != DEFERRED_FILE_WRITE_CALLER:
        return None, None, (
            "deferred file-write caller mismatch: "
            f"{snapshot['return_address']} != {DEFERRED_FILE_WRITE_CALLER}"
        )
    matched_boundary = rows_by_start.get(snapshot["command_bit_position"])
    boundary_snapshot = snapshot
    if (
        allow_one_tick_overdue_prepared
        and matched_boundary is not None
        and snapshot["needs_command"] == 0
        and snapshot["update"] == int(matched_boundary[1]["update"]) + 1
        and snapshot["last_demo_update"] == snapshot["update"]
    ):
        # A successful deferred-exit receipt may clamp an idle that was
        # already one framework tick late to the current live timeline.  The
        # offline boundary still identifies the row by its recorded update;
        # normalize only that comparison and retain every other live field.
        boundary_snapshot = dict(snapshot)
        boundary_snapshot["last_demo_update"] = int(
            matched_boundary[1]["update"]
        )
    boundary_failure = _snapshot_boundary_failure(
        rows_by_start,
        boundary_snapshot,
        command_order_offset=command_order_offset,
    )
    if boundary_failure is not None:
        return None, None, (
            "deferred file-write current boundary is invalid: "
            f"{boundary_failure}"
        )
    current_index, current_row = rows_by_start[
        snapshot["command_bit_position"]
    ]
    if snapshot["needs_command"] not in (0, 1):
        return None, None, (
            "deferred file-write needs state is invalid: needs="
            f"{snapshot['needs_command']}"
        )
    current_payload = current_row.get("payload")
    native_file_write_progress = (
        current_row.get("kind") == "file_write"
        and not bool(current_row["short_form"])
        and int(current_row["command_number"]) == 16
        and int(current_row["end"]) - int(current_row["start"]) == 11
        and isinstance(current_payload, dict)
        and set(current_payload) == {"success"}
        and isinstance(current_payload["success"], bool)
        and snapshot["update"] + 1 == int(current_row["update"])
        and (
            (
                snapshot["needs_command"] == 0
                and snapshot["buffer_read_bit_position"]
                == int(current_row["end"]) - 1
            )
            or (
                snapshot["needs_command"] == 1
                and snapshot["buffer_read_bit_position"]
                == int(current_row["end"])
            )
        )
    )
    if native_file_write_progress:
        # The retail worker can claim a due write one framework tick before
        # its recorded update.  The first entry has consumed the ten-bit
        # header; the retry has consumed the one-bit result.  Both are exact
        # native progress and must remain on the original replay path rather
        # than being mistaken for a deferred-result debt discharge.
        return None, None, None
    if snapshot["buffer_read_bit_position"] != int(current_row["end"]):
        return None, None, (
            "deferred file-write current row is not fully consumed: "
            f"{snapshot['buffer_read_bit_position']} != {current_row['end']}"
        )
    if snapshot["needs_command"] == 0:
        if snapshot["update"] > int(current_row["update"]):
            if not (
                allow_one_tick_overdue_prepared
                and snapshot["update"] == int(current_row["update"]) + 1
            ):
                return None, None, (
                    "deferred file-write prepared row is overdue: "
                    f"{snapshot['update']} > {current_row['update']}"
                )
        if (
            int(current_row["command_number"]) == 16
            and int(current_row["update"]) == snapshot["update"]
        ):
            return None, None, None
    else:
        if snapshot["update"] != int(current_row["update"]):
            return None, None, (
                "deferred file-write framework update mismatch: "
                f"{snapshot['update']} != {current_row['update']}"
            )
        if current_index + 1 >= len(rows):
            return None, None, (
                "deferred file-write has no following offline row"
            )
        following = rows[current_index + 1]
        if (
            int(following["command_number"]) == 16
            and int(following["update"]) == snapshot["update"]
        ):
            return None, None, None
    if snapshot["demo_loading_complete"] != 1:
        if not allow_pre_loading_after_service_continuation:
            return None, None, (
                "deferred file-write is outside the post-loading boundary: "
                f"loading_complete={snapshot['demo_loading_complete']}"
            )
        if (
            snapshot["demo_loading_complete"] != 0
            or pre_attach_rows
            or pre_attach_before_update is not None
        ):
            return None, None, (
                "deferred file-write pre-loading service-continuation "
                "receipt is incompatible with late-attach debt"
            )
    if (
        not accounted_rows
        and not pre_attach_rows
        and not service_consumed_rows
    ):
        return None, None, "deferred file-write has no accounted result debt"
    if tuple(sorted(set(accounted_rows))) != accounted_rows:
        return None, None, (
            "deferred file-write accounted rows are not unique and ordered"
        )
    if command_order_offset != len(accounted_rows):
        return None, None, (
            "deferred file-write offset/debt mismatch: "
            f"{command_order_offset} != {len(accounted_rows)}"
        )
    if tuple(sorted(set(excluded_accounted_rows))) != excluded_accounted_rows:
        return None, None, (
            "deferred file-write excluded rows are not unique and ordered"
        )
    if not set(excluded_accounted_rows).issubset(accounted_rows):
        return None, None, (
            "deferred file-write excluded rows are not accounted"
        )
    if tuple(sorted(set(service_consumed_rows))) != service_consumed_rows:
        return None, None, (
            "deferred file-write service-consumed rows are not unique and "
            "ordered"
        )
    if set(service_consumed_rows) & set(accounted_rows):
        return None, None, (
            "deferred file-write accounted and service-consumed rows overlap"
        )
    if bool(pre_attach_rows) != (pre_attach_before_update is not None):
        return None, None, (
            "deferred file-write pre-attach debt metadata is incomplete"
        )
    if tuple(sorted(set(pre_attach_rows))) != pre_attach_rows:
        return None, None, (
            "deferred file-write pre-attach rows are not unique and ordered"
        )
    effective_accounted_rows = tuple(
        row_index
        for row_index in accounted_rows
        if row_index not in excluded_accounted_rows
    )
    if (
        set(effective_accounted_rows) & set(pre_attach_rows)
        or set(service_consumed_rows) & set(pre_attach_rows)
    ):
        return None, None, (
            "deferred file-write debt sources overlap"
        )
    debt_rows = (
        accounted_rows + pre_attach_rows
        if not excluded_accounted_rows and not service_consumed_rows
        else tuple(
            sorted(
                set(effective_accounted_rows)
                | set(service_consumed_rows)
                | set(pre_attach_rows)
            )
        )
    )
    if discharge_count >= len(debt_rows):
        return None, None, (
            "deferred file-write result debt is exhausted: "
            f"{discharge_count} >= {len(debt_rows)}"
        )
    for row_index in debt_rows:
        if row_index < 0 or row_index >= len(rows):
            return None, None, (
                "deferred file-write accounted row is out of range: "
                f"{row_index}"
            )
        debt_row = rows[row_index]
        payload = debt_row.get("payload")
        if (
            row_index >= current_index
            or bool(debt_row["short_form"])
            or int(debt_row["command_number"]) != 16
            or not isinstance(payload, dict)
            or not isinstance(payload.get("success"), bool)
        ):
            return None, None, (
                "deferred file-write debt row is not an exact prior long "
                f"file-write result: row={row_index}"
            )
        if row_index in pre_attach_rows and (
            pre_attach_before_update is None
            or int(debt_row["update"]) >= pre_attach_before_update
        ):
            return None, None, (
                "deferred file-write pre-attach row is not before the "
                f"audited attach: row={row_index}, "
                f"update={debt_row['update']}, "
                f"attach={pre_attach_before_update}"
            )
    debt_index = debt_rows[discharge_count]
    recorded_success = bool(rows[debt_index]["payload"]["success"])
    return debt_index, recorded_success, None


def _audited_successful_file_write_debt_barrier(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    command_order_offset: int,
    accounted_rows: tuple[int, ...],
    excluded_accounted_rows: tuple[int, ...],
    service_consumed_rows: tuple[int, ...],
    discharged_rows: tuple[int, ...],
    diagnostic_padding_rows: tuple[int, ...],
) -> tuple[tuple[int, ...], str | None]:
    """Retire only an exact unclaimed tail at a later service barrier.

    A fixed-width replay service read can consume more successful file-write
    results than the retail file-write wrappers subsequently request.  Those
    rows are not debt merely because they affected command order.  They become
    auditable surplus only after every observed wrapper discharge is an exact
    prefix and playback reaches a later, non-file-write prepared service row,
    with no intervening or future file-write result in the DMO.
    """

    registries = (
        accounted_rows,
        excluded_accounted_rows,
        service_consumed_rows,
        discharged_rows,
        diagnostic_padding_rows,
    )
    if (
        command_order_offset < 0
        or command_order_offset != len(accounted_rows)
        or any(tuple(sorted(set(values))) != values for values in registries)
        or not set(excluded_accounted_rows).issubset(accounted_rows)
        or set(service_consumed_rows) & set(accounted_rows)
        or set(diagnostic_padding_rows) != set(excluded_accounted_rows)
    ):
        raise ValueError("successful file-write debt barrier registry is invalid")

    current_index: int | None = None
    for index, row in enumerate(rows):
        if int(row["start"]) == snapshot["command_bit_position"]:
            current_index = index
            break
    if current_index is None:
        return (), "file-write debt barrier has no exact offline start"
    current = rows[current_index]
    current_payload = current.get("payload")
    if (
        snapshot["return_address"] != MAIN_DEMO_CALLER
        or snapshot["needs_command"] != 1
        or snapshot["is_short"] != 0
        or snapshot["command_number"] != int(current["command_number"])
        or snapshot["command_order"]
        != current_index - command_order_offset
        or snapshot["buffer_read_bit_position"] != int(current["start"]) + 10
        or snapshot["update"] != int(current["update"])
        or bool(current["short_form"])
        or int(current["command_number"]) not in SERVICE_COMMAND_NUMBERS
        or int(current["command_number"]) == 16
        or int(current["end"]) - int(current["start"]) <= 10
        or not isinstance(current_payload, dict)
        or not isinstance(current_payload.get("success"), bool)
    ):
        return (), "file-write debt barrier is not a later prepared service row"

    def exact_successful_file_write(row_index: int) -> bool:
        if row_index < 0 or row_index >= len(rows):
            return False
        row = rows[row_index]
        return (
            row.get("kind") == "file_write"
            and not bool(row["short_form"])
            and int(row["command_number"]) == 16
            and int(row["end"]) - int(row["start"]) == 11
            and row.get("payload") == {"success": True}
        )

    effective_accounted = tuple(
        row_index
        for row_index in accounted_rows
        if row_index not in excluded_accounted_rows
    )
    debt_rows = tuple(
        sorted(set(effective_accounted) | set(service_consumed_rows))
    )
    if (
        not debt_rows
        or any(not exact_successful_file_write(index) for index in debt_rows)
        or len(discharged_rows) >= len(debt_rows)
        or discharged_rows != debt_rows[: len(discharged_rows)]
    ):
        return (), "file-write debt barrier lacks an exact discharged prefix"
    surplus_rows = debt_rows[len(discharged_rows) :]
    if not surplus_rows or surplus_rows[-1] >= current_index:
        return (), "file-write debt barrier surplus is not strictly prior"

    corridor_update = int(rows[surplus_rows[0]]["update"])
    corridor_start = surplus_rows[0]
    while (
        corridor_start > 0
        and exact_successful_file_write(corridor_start - 1)
        and int(rows[corridor_start - 1]["update"]) == corridor_update
    ):
        corridor_start -= 1
    corridor_end = surplus_rows[-1]
    while (
        corridor_end + 1 < len(rows)
        and exact_successful_file_write(corridor_end + 1)
        and int(rows[corridor_end + 1]["update"]) == corridor_update
    ):
        corridor_end += 1
    allowed_tail = set(surplus_rows) | set(diagnostic_padding_rows)
    if (
        any(
            row_index not in allowed_tail
            for row_index in range(surplus_rows[0], corridor_end + 1)
        )
        or not set(diagnostic_padding_rows).issubset(
            range(corridor_start, corridor_end + 1)
        )
        or any(
            exact_successful_file_write(row_index)
            for row_index in range(corridor_end + 1, len(rows))
        )
        or int(current["update"]) <= corridor_update
    ):
        return (), (
            "file-write debt barrier does not close the final successful "
            "file-write corridor"
        )
    return surplus_rows, None


def _audited_font_cache_manifest_completion(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    command_order_offset: int,
    accounted_rows: tuple[int, ...],
    discharge_count: int,
    pre_attach_rows: tuple[int, ...],
    pre_attach_before_update: int | None,
    startup_file_write_rows: tuple[int, ...],
    manifest: Mapping[str, int],
    observed_members: tuple[tuple[str, int], ...],
    current_member: str,
    current_size: int,
    completion_count: int,
    service_handoff_count: int,
    direct_match_count: int = 0,
    excluded_accounted_rows: tuple[int, ...] = (),
    service_consumed_rows: tuple[int, ...] = (),
    allow_one_tick_overdue_prepared: bool = False,
    allow_pre_loading_after_service_continuation: bool = False,
) -> tuple[bool | None, str | None]:
    """Authorize bounded uniform results until the retail manifest closes.

    This is deliberately narrower than another generic debt row.  It is valid
    only after every explicit post-attach and pre-attach result debt has been
    discharged, while parked at the same audited boundary, and only when the
    original PAK catalog and the startup DMO contain the same 35 font-cache
    operations.  Because every recorded result is false, member ordering is
    immaterial; each observed live identity and byte size still has to be
    unique and present in the immutable retail catalog.  The pre-attach
    operation count plus all observed post-attach calls may never exceed the
    catalog cardinality, so a 36th operation fails closed.
    """

    if (
        completion_count < 0
        or service_handoff_count < 0
        or direct_match_count < 0
    ):
        raise ValueError("font-cache completion counters must be non-negative")
    debt_index, _, debt_failure = _audited_deferred_file_write_result(
        rows_by_start,
        rows,
        snapshot,
        command_order_offset=command_order_offset,
        accounted_rows=accounted_rows,
        discharge_count=discharge_count,
        pre_attach_rows=pre_attach_rows,
        pre_attach_before_update=pre_attach_before_update,
        excluded_accounted_rows=excluded_accounted_rows,
        service_consumed_rows=service_consumed_rows,
        allow_one_tick_overdue_prepared=(
            allow_one_tick_overdue_prepared
        ),
        allow_pre_loading_after_service_continuation=(
            allow_pre_loading_after_service_continuation
        ),
    )
    if debt_index is not None or debt_failure is None:
        return None, (
            "font-cache manifest completion requires exhausted explicit debt"
        )
    if "result debt is exhausted" not in debt_failure:
        return None, (
            "font-cache manifest completion boundary is invalid: "
            f"{debt_failure}"
        )

    normalized_manifest: dict[str, int] = {}
    for raw_member, raw_size in manifest.items():
        member = _normalize_font_cache_member(raw_member)
        if (
            member is None
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size <= 0
            or member in normalized_manifest
        ):
            return None, "font-cache retail manifest is malformed"
        normalized_manifest[member] = raw_size
    if len(normalized_manifest) != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES:
        return None, (
            "font-cache retail manifest entry count mismatch: "
            f"{len(normalized_manifest)} != "
            f"{FONT_CACHE_MANIFEST_EXPECTED_ENTRIES}"
        )

    loading_complete_rows = tuple(
        index
        for index, row in enumerate(rows)
        if row.get("kind") == "loading_complete"
        and int(row["command_number"]) == 9
    )
    if not loading_complete_rows:
        return None, "font-cache startup has no loading-complete boundary"
    loading_complete_index = loading_complete_rows[0]
    exact_startup_rows = tuple(
        index
        for index, row in enumerate(rows[:loading_complete_index])
        if (
            not bool(row["short_form"])
            and int(row["command_number"]) == 16
            and isinstance(row.get("payload"), dict)
            and isinstance(row["payload"].get("success"), bool)
        )
    )
    if startup_file_write_rows != exact_startup_rows:
        return None, "font-cache startup row receipt is not exact"
    if len(startup_file_write_rows) != len(normalized_manifest):
        return None, (
            "font-cache DMO/manifest cardinality mismatch: "
            f"{len(startup_file_write_rows)} != {len(normalized_manifest)}"
        )
    if any(
        bool(rows[row_index]["payload"]["success"])
        for row_index in startup_file_write_rows
    ):
        return None, "font-cache startup results are not uniformly false"
    expected_pre_attach_rows = tuple(
        row_index
        for row_index in startup_file_write_rows
        if (
            pre_attach_before_update is not None
            and int(rows[row_index]["update"])
            < pre_attach_before_update
        )
    )
    if pre_attach_rows != expected_pre_attach_rows:
        return None, "font-cache pre-attach operation receipt is not exact"

    expected_observed_count = (
        discharge_count
        + completion_count
        + service_handoff_count
        + direct_match_count
    )
    if len(observed_members) != expected_observed_count:
        return None, (
            "font-cache observed-call receipt count mismatch: "
            f"observed={len(observed_members)}, discharged={discharge_count}, "
            f"completed={completion_count}, "
            f"service_handoffs={service_handoff_count}, "
            f"direct_matches={direct_match_count}"
        )
    if (
        len(pre_attach_rows) + len(observed_members) + 1
        > len(normalized_manifest)
    ):
        return None, (
            "font-cache retail operation cardinality is exhausted: "
            f"pre_attach={len(pre_attach_rows)}, "
            f"observed={len(observed_members)}, manifest="
            f"{len(normalized_manifest)}"
        )
    seen: set[str] = set()
    for raw_member, raw_size in observed_members:
        member = _normalize_font_cache_member(raw_member)
        if (
            member is None
            or member in seen
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or normalized_manifest.get(member) != raw_size
        ):
            return None, (
                "font-cache prior live member is absent, duplicated, or "
                "size-mismatched"
            )
        seen.add(member)

    member = _normalize_font_cache_member(current_member)
    if member is None:
        return None, "font-cache current live path is not a manifest member"
    if member in seen:
        return None, "font-cache current live member is duplicated"
    if (
        isinstance(current_size, bool)
        or not isinstance(current_size, int)
        or normalized_manifest.get(member) != current_size
    ):
        return None, "font-cache current live member size mismatch"
    if len(seen) + 1 > len(normalized_manifest):
        return None, "font-cache observed members exceed the retail manifest"
    return False, None


def _audited_terminal_file_write_payload_handoff(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    row_index: int,
    command_order_offset: int,
) -> tuple[bool | None, str | None]:
    """Validate one terminal long file-write payload before emulation."""

    if (
        row_index < 0
        or row_index >= len(rows)
        or command_order_offset < 0
    ):
        raise ValueError("terminal file-write handoff index is invalid")
    row = rows[row_index]
    payload = row.get("payload")
    if (
        bool(row["short_form"])
        or int(row["command_number"]) != 16
        or int(row["end"]) - int(row["start"]) != 11
        or not isinstance(payload, dict)
        or not isinstance(payload.get("success"), bool)
    ):
        return None, (
            "terminal file-write handoff row is not one exact long result: "
            f"row={row_index}"
        )
    expected = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "command_bit_position": int(row["start"]),
        "buffer_read_bit_position": int(row["end"]) - 1,
        "needs_command": 1,
        "is_short": 0,
        "command_number": 16,
        "command_order": row_index - command_order_offset,
        "last_demo_update": int(row["update"]) - 1,
        "update": int(row["update"]) - 1,
        "demo_loading_complete": 0,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return None, (
                f"terminal file-write handoff row {row_index} {key} "
                f"mismatch: {snapshot[key]} != {value}"
            )
    return bool(payload["success"]), None


def _attach_stabilization_boundary_failure(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    snapshot: dict[str, int],
    *,
    command_order_offset: int = 0,
) -> str | None:
    """Require a complete main-loop boundary before auditing a late attach.

    A debugger can attach while ``PrepareDemoCommand`` is already active.  The
    first software-breakpoint hits can therefore expose payload/re-entry state
    that was created before instrumentation.  Such hits are safe to observe,
    but they are not valid live/offline boundaries.  Stabilization ends only
    at a main-loop call whose complete command has been consumed and whose
    offline identity is exact.
    """

    if snapshot["return_address"] != MAIN_DEMO_CALLER:
        return (
            "attach stabilization caller mismatch: "
            f"{snapshot['return_address']} != {MAIN_DEMO_CALLER}"
        )
    if snapshot["needs_command"] != 0:
        return (
            "attach stabilization requires a complete command: needs="
            f"{snapshot['needs_command']}"
        )
    boundary_failure = _snapshot_boundary_failure(
        rows_by_start,
        snapshot,
        command_order_offset=command_order_offset,
    )
    if boundary_failure is not None:
        return boundary_failure
    _, row = rows_by_start[snapshot["command_bit_position"]]
    expected_end = int(row["end"])
    if snapshot["buffer_read_bit_position"] != expected_end:
        return (
            "attach stabilization command is not fully consumed: "
            f"{snapshot['buffer_read_bit_position']} != {expected_end}"
        )
    expected_update = int(row["update"])
    if snapshot["update"] != expected_update:
        return (
            "attach stabilization framework update mismatch: "
            f"{snapshot['update']} != {expected_update}"
        )
    return None


def _blackout_file_write_rebase_candidates(
    rows: list[dict[str, object]],
    *,
    before_index: int,
    detached_after_update: int,
) -> tuple[int, ...]:
    """Return the exact long-result candidate envelope in one blackout."""

    if (
        before_index < 0
        or before_index > len(rows)
        or detached_after_update < 0
    ):
        raise ValueError("blackout candidate boundary is invalid")
    eligible: list[int] = []
    for row_index, candidate in enumerate(rows[:before_index]):
        if int(candidate["update"]) <= detached_after_update:
            continue
        payload = candidate.get("payload")
        if (
            candidate.get("kind") == "file_write"
            and int(candidate["command_number"]) == 16
            and not bool(candidate["short_form"])
            and int(candidate["end"]) - int(candidate["start"]) == 11
            and isinstance(payload, dict)
            and set(payload) == {"success"}
            and isinstance(payload["success"], bool)
        ):
            eligible.append(row_index)
    return tuple(eligible)


def _audited_file_write_command_order_rebase(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    detached_after_update: int,
) -> tuple[int, tuple[int, ...], str | None]:
    """Audit one post-blackout order offset against direct file-write results.

    PopCap's replay service can consume long-form file-write result bits
    without entering PrepareDemoCommand.  During an explicitly untraced
    interval this leaves the live entry counter behind the offline row index,
    while the bit position and all other command fields remain exact.  Accept
    that offset only when every missing entry is accounted for by one exact
    long-form file-write result row in the blackout.
    """

    if detached_after_update < 0:
        raise ValueError("detached update must be non-negative")
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return (
            0,
            (),
            "command start is absent from the offline DMO: "
            f"{snapshot['command_bit_position']}",
        )
    index, row = matched
    non_order_expected = {
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "last_demo_update": int(row["update"]),
    }
    for key, value in non_order_expected.items():
        if snapshot[key] != value:
            return (
                0,
                (),
                f"offline boundary {index} {key} mismatch before rebase: "
                f"{snapshot[key]} != {value}",
            )
    live_order = snapshot["command_order"]
    offset = index - live_order
    if live_order < 0 or offset < 0:
        return (
            0,
            (),
            "post-blackout command order cannot be rebased: "
            f"offline={index}, live={live_order}",
        )

    eligible = _blackout_file_write_rebase_candidates(
        rows,
        before_index=index,
        detached_after_update=detached_after_update,
    )

    if offset != len(eligible):
        return (
            0,
            eligible,
            "post-blackout command order offset is not fully explained by "
            "long-form file-write results: "
            f"offset={offset}, eligible={len(eligible)}, "
            f"offline={index}, live={live_order}",
        )
    return offset, eligible, None


def _audited_post_blackout_file_write_offset_envelope(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    detached_after_update: int,
    expected_command_order_offset: int,
    expected_native_timeline_offset: int,
) -> tuple[int, tuple[int, ...], int, str | None]:
    """Audit a preregistered cross-corridor blackout offset envelope.

    A long debugger blackout can let retail service state created by one
    successful-write corridor mature only when a later corridor is consumed.
    The resulting entry deficit therefore cannot honestly be assigned to
    individual rows from endpoint observations.  This mode freezes the whole
    exact long-result candidate set before launch, requires preregistered
    order and native-timeline offsets, and accepts only a complete main-loop
    boundary after applying those fixed offsets.  Every later boundary must
    continue to satisfy the same offsets.
    """

    if (
        detached_after_update < 0
        or expected_command_order_offset <= 0
        or expected_native_timeline_offset < 0
    ):
        raise ValueError("blackout offset envelope is invalid")
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return 0, (), 0, (
            "command start is absent from the offline DMO: "
            f"{snapshot['command_bit_position']}"
        )
    index, row = matched
    expected_fields = {
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
    }
    for key, value in expected_fields.items():
        if snapshot[key] != value:
            return 0, (), 0, (
                f"offline boundary {index} {key} mismatch before envelope: "
                f"{snapshot[key]} != {value}"
            )

    live_order = snapshot["command_order"]
    observed_order_offset = index - live_order
    observed_last_offset = (
        snapshot["last_demo_update"] - int(row["update"])
    )
    observed_update_offset = snapshot["update"] - int(row["update"])
    candidates = _blackout_file_write_rebase_candidates(
        rows,
        before_index=index,
        detached_after_update=detached_after_update,
    )
    if (
        live_order < 0
        or observed_order_offset != expected_command_order_offset
        or observed_last_offset != expected_native_timeline_offset
        or observed_update_offset != expected_native_timeline_offset
        or len(candidates) < expected_command_order_offset
    ):
        return 0, candidates, 0, (
            "post-blackout offset envelope mismatch: "
            f"order={observed_order_offset}/"
            f"{expected_command_order_offset}, last_timeline="
            f"{observed_last_offset}/"
            f"{expected_native_timeline_offset}, update_timeline="
            f"{observed_update_offset}/"
            f"{expected_native_timeline_offset}, candidates="
            f"{len(candidates)}"
        )

    normalized = dict(snapshot)
    normalized["last_demo_update"] -= expected_native_timeline_offset
    normalized["update"] -= expected_native_timeline_offset
    failure = _attach_stabilization_boundary_failure(
        rows_by_start,
        normalized,
        command_order_offset=expected_command_order_offset,
    )
    if failure is not None:
        return 0, candidates, 0, failure
    return (
        expected_command_order_offset,
        candidates,
        expected_native_timeline_offset,
        None,
    )


def _audited_post_blackout_file_write_offset_envelope_set(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    detached_after_update: int,
    allowed_offset_pairs: tuple[tuple[int, int], ...],
) -> tuple[int, tuple[int, ...], int, str | None]:
    """Audit one member of a preregistered finite blackout-offset set."""

    if (
        detached_after_update < 0
        or not allowed_offset_pairs
        or len(set(allowed_offset_pairs)) != len(allowed_offset_pairs)
        or any(
            isinstance(command_offset, bool)
            or not isinstance(command_offset, int)
            or command_offset < 0
            or isinstance(timeline_offset, bool)
            or not isinstance(timeline_offset, int)
            or timeline_offset < 0
            for command_offset, timeline_offset in allowed_offset_pairs
        )
    ):
        raise ValueError("blackout offset envelope set is invalid")
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return 0, (), 0, (
            "command start is absent from the offline DMO: "
            f"{snapshot['command_bit_position']}"
        )
    index, row = matched
    expected_fields = {
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
    }
    for key, value in expected_fields.items():
        if snapshot[key] != value:
            return 0, (), 0, (
                f"offline boundary {index} {key} mismatch before envelope: "
                f"{snapshot[key]} != {value}"
            )

    live_order = snapshot["command_order"]
    observed_order_offset = index - live_order
    observed_last_offset = (
        snapshot["last_demo_update"] - int(row["update"])
    )
    observed_update_offset = snapshot["update"] - int(row["update"])
    candidates = _blackout_file_write_rebase_candidates(
        rows,
        before_index=index,
        detached_after_update=detached_after_update,
    )
    observed_pair = (observed_order_offset, observed_last_offset)
    if (
        live_order < 0
        or observed_last_offset != observed_update_offset
        or observed_pair not in allowed_offset_pairs
        or len(candidates) < observed_order_offset
    ):
        allowed_label = ",".join(
            f"{command}/{timeline}"
            for command, timeline in allowed_offset_pairs
        )
        return 0, candidates, 0, (
            "post-blackout offset envelope-set mismatch: "
            f"order={observed_order_offset}, last_timeline="
            f"{observed_last_offset}, update_timeline="
            f"{observed_update_offset}, allowed={allowed_label}, "
            f"candidates={len(candidates)}"
        )

    normalized = dict(snapshot)
    normalized["last_demo_update"] -= observed_last_offset
    normalized["update"] -= observed_last_offset
    failure = _attach_stabilization_boundary_failure(
        rows_by_start,
        normalized,
        command_order_offset=observed_order_offset,
    )
    if failure is not None:
        return 0, candidates, 0, failure
    return observed_order_offset, candidates, observed_last_offset, None


def _normalize_blackout_native_timeline_snapshot(
    snapshot: Mapping[str, object],
    *,
    native_timeline_offset: int,
) -> dict[str, object]:
    """Normalize one live snapshot to its frozen post-blackout timeline.

    The original native values are retained so repeated normalization is
    idempotent and any accidental offset change fails closed.
    """

    if (
        isinstance(native_timeline_offset, bool)
        or not isinstance(native_timeline_offset, int)
        or native_timeline_offset < 0
    ):
        raise ValueError("blackout native timeline offset is invalid")
    update = snapshot.get("update")
    last_demo_update = snapshot.get("last_demo_update")
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or isinstance(last_demo_update, bool)
        or not isinstance(last_demo_update, int)
    ):
        raise ValueError("blackout snapshot timeline is invalid")
    marker_keys = (
        "native_update",
        "native_last_demo_update",
    )
    marker_presence = tuple(key in snapshot for key in marker_keys)
    if marker_presence == (True, True):
        native_update = snapshot["native_update"]
        native_last_demo_update = snapshot["native_last_demo_update"]
        if (
            isinstance(native_update, bool)
            or not isinstance(native_update, int)
            or isinstance(native_last_demo_update, bool)
            or not isinstance(native_last_demo_update, int)
            or update != native_update - native_timeline_offset
            or last_demo_update
            != native_last_demo_update - native_timeline_offset
        ):
            raise ValueError(
                "blackout snapshot timeline normalization changed"
            )
        return dict(snapshot)
    if marker_presence != (False, False):
        raise ValueError("blackout snapshot timeline marker is incomplete")
    if update < native_timeline_offset or (
        last_demo_update < native_timeline_offset
    ):
        raise ValueError("blackout snapshot timeline would be negative")
    return {
        **snapshot,
        "native_update": update,
        "native_last_demo_update": last_demo_update,
        "update": update - native_timeline_offset,
        "last_demo_update": (
            last_demo_update - native_timeline_offset
        ),
    }


def _audited_post_blackout_attach_stabilization(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    detached_after_update: int,
) -> tuple[int, tuple[int, ...], str | None]:
    """Atomically stabilize a late attach and audit its order rebase.

    A post-blackout breakpoint can first land inside a command that began
    before the debugger attached.  Do not commit the file-write order offset
    merely because that in-flight snapshot happens to identify an offline
    row.  The candidate offset becomes authoritative only when the same
    snapshot is also one exact, fully consumed main-loop boundary.
    """

    offset, accounted_rows, failure = (
        _audited_file_write_command_order_rebase(
            rows_by_start,
            rows,
            snapshot,
            detached_after_update=detached_after_update,
        )
    )
    if failure is not None:
        return 0, accounted_rows, failure
    failure = _attach_stabilization_boundary_failure(
        rows_by_start,
        snapshot,
        command_order_offset=offset,
    )
    if failure is not None:
        return 0, accounted_rows, failure
    return offset, accounted_rows, None


def _audited_service_command_order_rebase(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    command_order_offset: int,
    accounted_rows: tuple[int, ...],
    require_last_demo_update: bool = True,
    allow_exit_row: bool = False,
) -> tuple[int, tuple[int, ...], str | None]:
    """Account for service rows consumed inside an audited payload re-entry."""

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index >= len(rows)
        or command_order_offset < 0
        or not isinstance(allow_exit_row, bool)
    ):
        raise ValueError("service command-order rebase boundary is invalid")
    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return command_order_offset, accounted_rows, (
            "service command-order rebase requires an exact offline start"
        )
    index, row = matched
    if allow_exit_row:
        if (
            index != corridor_end_index + 1
            or int(row["command_number"]) in SERVICE_COMMAND_NUMBERS
        ):
            return command_order_offset, accounted_rows, (
                "service command-order rebase exit row is invalid: "
                f"{index}"
            )
    expected_fields = {
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
    }
    if require_last_demo_update:
        expected_fields["last_demo_update"] = int(row["update"])
    for key, value in expected_fields.items():
        if snapshot[key] != value:
            return command_order_offset, accounted_rows, (
                f"service command-order rebase row {index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    expected_live_order = index - command_order_offset
    missing_count = expected_live_order - snapshot["command_order"]
    if missing_count <= 0:
        return command_order_offset, accounted_rows, (
            "service command-order rebase is not a positive deficit: "
            f"offline={index}, live={snapshot['command_order']}, "
            f"offset={command_order_offset}"
        )
    first_missing = index - missing_count
    missing_rows = tuple(range(first_missing, index))
    if (
        first_missing < corridor_start_index
        or index
        > corridor_end_index + (1 if allow_exit_row else 0)
        or any(row_index in accounted_rows for row_index in missing_rows)
    ):
        return command_order_offset, accounted_rows, (
            "service command-order deficit escapes or repeats its corridor: "
            f"rows={missing_rows}"
        )
    command_start = snapshot["command_bit_position"]
    for row_index in missing_rows:
        missing = rows[row_index]
        if (
            row_index > corridor_end_index
            or bool(missing["short_form"])
            or int(missing["command_number"])
            not in SERVICE_COMMAND_NUMBERS
            or int(missing["end"]) - int(missing["start"]) <= 10
            or int(missing["end"]) > command_start
        ):
            return command_order_offset, accounted_rows, (
                "service command-order deficit contains an ineligible row: "
                f"{row_index}"
            )
    return (
        command_order_offset + missing_count,
        accounted_rows + missing_rows,
        None,
    )


def _audited_successful_file_write_corridor_prepared_exit(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit a complete non-service exit after post-load write spillover."""

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError(
            "successful file-write prepared exit boundary is invalid"
        )
    corridor = rows[corridor_start_index : corridor_end_index + 1]
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, (
            "successful file-write prepared exit corridor is ineligible"
        )
    corridor_update = int(corridor[0]["update"])
    exit_index = corridor_end_index + 1
    exit_row = rows[exit_index]
    command_start = int(exit_row["start"])
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": int(exit_row["command_number"]),
        "is_short": int(bool(exit_row["short_form"])),
        "command_order": exit_index - command_order_offset,
        "command_bit_position": command_start,
        "buffer_read_bit_position": int(exit_row["end"]),
        "demo_loading_complete": 1,
    }
    if (
        int(corridor[-1]["end"]) != command_start
        or previous_read_bit_position != command_start
        or int(exit_row["command_number"]) in SERVICE_COMMAND_NUMBERS
        or int(exit_row["update"]) != corridor_update
    ):
        return exit_index, (
            "successful file-write prepared exit is not the exact "
            "same-update corridor successor"
        )
    for key, value in expected.items():
        if snapshot.get(key) != value:
            return exit_index, (
                "successful file-write prepared exit "
                f"{exit_index} {key} mismatch: "
                f"{snapshot.get(key)} != {value}"
            )
    update_delta = snapshot["update"] - corridor_update
    if not (
        snapshot["update"] == snapshot["last_demo_update"]
        and 1 <= update_delta <= len(corridor)
    ):
        return exit_index, (
            "successful file-write prepared exit is outside its bounded "
            "spillover: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}, "
            f"corridor={corridor_update}, rows={len(corridor)}"
        )
    return exit_index, None


def _audited_successful_file_write_deferred_exit(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit an exact idle boundary returned to a delayed write wrapper."""

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError(
            "successful file-write deferred exit boundary is invalid"
        )
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, (
            "successful file-write deferred exit corridor is ineligible"
        )

    corridor_update = int(rows[corridor_end_index]["update"])
    exit_index = corridor_end_index + 1
    exit_row = rows[exit_index]
    command_start = int(exit_row["start"])
    command_end = int(exit_row["end"])
    exit_update = int(exit_row["update"])
    expected = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "needs_command": 0,
        "command_number": 31,
        "is_short": 0,
        "command_order": exit_index - command_order_offset,
        "command_bit_position": command_start,
        "buffer_read_bit_position": command_end,
        "demo_loading_complete": 1,
    }
    if (
        int(rows[corridor_end_index]["end"]) != command_start
        or previous_read_bit_position != command_start
        or exit_row.get("kind") != "idle"
        or exit_row.get("payload") != {}
        or command_end - command_start != 10
    ):
        return exit_index, (
            "successful file-write deferred exit is not the exact long-idle "
            "corridor successor"
        )
    for key, value in expected.items():
        if snapshot.get(key) != value:
            return exit_index, (
                "successful file-write deferred exit "
                f"{exit_index} {key} mismatch: "
                f"{snapshot.get(key)} != {value}"
            )

    framework_update = snapshot["update"]
    live_last_demo_update = snapshot["last_demo_update"]
    corridor_length = corridor_end_index - corridor_start_index + 1
    corridor_to_framework_limit = max(
        SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES,
        corridor_length,
    )
    if not (
        corridor_update < framework_update <= exit_update + 1
        and exit_update < live_last_demo_update
        and framework_update - corridor_update
        <= corridor_to_framework_limit
        and -1
        <= exit_update - framework_update
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and live_last_demo_update - exit_update
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
    ):
        return exit_index, (
            "successful file-write deferred exit is outside its bounded "
            "timeline: "
            f"corridor={corridor_update}, framework={framework_update}, "
            f"exit={exit_update}, last={live_last_demo_update}"
        )
    return exit_index, None


def _audited_successful_file_write_deferred_exit_timeline_commit(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, int | None, str | None]:
    """Derive the only timeline write allowed by an audited deferred exit.

    The exact idle row normally commits its recorded update.  If the bounded
    audit proves that the framework has already advanced exactly one tick
    beyond that row, writing the recorded value would place the replay clock
    in the past and the retail loop will not revisit the prepared idle.  In
    that one case the effective commit is clamped to the current framework
    update.  The underlying exit audit rejects every later observation.
    """

    exit_index, failure = _audited_successful_file_write_deferred_exit(
        rows,
        snapshot,
        corridor_start_index=corridor_start_index,
        corridor_end_index=corridor_end_index,
        previous_read_bit_position=previous_read_bit_position,
        previous_reentry_kind=previous_reentry_kind,
        command_order_offset=command_order_offset,
    )
    if failure is not None or exit_index is None:
        return exit_index, None, failure
    recorded_update = int(rows[exit_index]["update"])
    if snapshot["update"] > recorded_update + 1:
        return exit_index, None, (
            "successful file-write deferred exit timeline commit is more "
            "than one tick overdue"
        )
    return exit_index, max(recorded_update, snapshot["update"]), None


def _audited_successful_file_write_deferred_exit_committed_boundary_snapshot(
    rows_by_start: dict[int, tuple[int, dict[str, object]]],
    snapshot: dict[str, int],
    receipt: Mapping[str, object],
    *,
    process_id: int,
    thread_id: int,
) -> tuple[dict[str, int], bool]:
    """Normalize one exact clamped idle for the offline boundary audit.

    The live ``last_demo_update`` intentionally equals the framework update
    after a one-tick-late deferred-exit commit, while the immutable DMO row
    retains its original update.  Normalize only this comparison and only
    when the complete commit receipt, process, thread, row, and bit boundary
    all identify the same long idle.
    """

    matched = rows_by_start.get(snapshot["command_bit_position"])
    if matched is None:
        return snapshot, False
    row_index, row = matched
    recorded_update = int(row["update"])
    effective_update = recorded_update + 1
    corridor_end_index = receipt.get("corridor_end_row_index")
    corridor_end_update = receipt.get("corridor_end_recorded_update")
    if (
        process_id <= 0
        or thread_id <= 0
        or row.get("kind") != "idle"
        or bool(row["short_form"])
        or int(row["command_number"]) != 31
        or row.get("payload") != {}
        or int(row["end"]) - int(row["start"]) != 10
        or snapshot["return_address"] != MAIN_DEMO_CALLER
        or snapshot["needs_command"] not in (0, 1)
        or snapshot["command_number"] != 31
        or snapshot["is_short"] != 0
        or snapshot["update"] != effective_update
        or snapshot["last_demo_update"] != effective_update
        or snapshot["buffer_read_bit_position"] != int(row["end"])
        or receipt.get("process_id") != process_id
        or receipt.get("thread_id") != thread_id
        or receipt.get("row_index") != row_index
        or receipt.get("command_bit_position") != int(row["start"])
        or receipt.get("buffer_read_bit_position") != int(row["end"])
        or receipt.get("framework_update") != effective_update
        or receipt.get("recorded_update") != recorded_update
        or receipt.get("effective_committed_update") != effective_update
        or receipt.get("last_demo_update_after") != effective_update
        or receipt.get("late_commit_clamp_updates") != 1
        or receipt.get("recorded_timeline_delta_updates") != -1
        or receipt.get("timeline_delta_updates") != 0
        or receipt.get("bytes_written") != 4
        or isinstance(corridor_end_index, bool)
        or not isinstance(corridor_end_index, int)
        or corridor_end_index < 0
        or corridor_end_index >= row_index
        or isinstance(corridor_end_update, bool)
        or not isinstance(corridor_end_update, int)
        or corridor_end_update > recorded_update
    ):
        return snapshot, False
    normalized = dict(snapshot)
    normalized["last_demo_update"] = recorded_update
    return normalized, True


def _audited_successful_file_write_deferred_exit_clamped_suffix_rebase(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    receipt: Mapping[str, object],
    *,
    process_id: int,
    thread_id: int,
    command_order_offset: int,
) -> tuple[int | None, int | None, str | None]:
    """Carry an exact one-tick deferred-exit clamp into its DMO suffix."""

    if (
        process_id <= 0
        or thread_id <= 0
        or command_order_offset < 0
    ):
        raise ValueError("clamped suffix rebase boundary is invalid")
    commit_row_index = receipt.get("row_index")
    if (
        isinstance(commit_row_index, bool)
        or not isinstance(commit_row_index, int)
        or commit_row_index < 0
        or commit_row_index + 1 >= len(rows)
    ):
        return None, None, "clamped suffix commit row is invalid"
    commit_row = rows[commit_row_index]
    suffix_start_index = commit_row_index + 1
    suffix_start = rows[suffix_start_index]
    recorded_update = int(commit_row["update"])
    suffix_recorded_update = int(suffix_start["update"])
    expected_live_update = suffix_recorded_update + 1
    if (
        commit_row.get("kind") != "idle"
        or bool(commit_row["short_form"])
        or int(commit_row["command_number"]) != 31
        or commit_row.get("payload") != {}
        or int(commit_row["end"]) - int(commit_row["start"]) != 10
        or suffix_start.get("kind") != "idle"
        or bool(suffix_start["short_form"])
        or int(suffix_start["command_number"]) != 31
        or suffix_start.get("payload") != {}
        or int(suffix_start["end"]) - int(suffix_start["start"]) != 10
        or int(commit_row["end"]) != int(suffix_start["start"])
        or suffix_recorded_update - recorded_update
        != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        or receipt.get("process_id") != process_id
        or receipt.get("thread_id") != thread_id
        or receipt.get("recorded_update") != recorded_update
        or receipt.get("framework_update") != recorded_update + 1
        or receipt.get("effective_committed_update") != recorded_update + 1
        or receipt.get("last_demo_update_after") != recorded_update + 1
        or receipt.get("late_commit_clamp_updates") != 1
        or receipt.get("recorded_timeline_delta_updates") != -1
        or receipt.get("timeline_delta_updates") != 0
        or receipt.get("bytes_written") != 4
    ):
        return suffix_start_index, None, (
            "clamped suffix commit receipt or idle anchor mismatch"
        )
    expected_snapshot = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 31,
        "is_short": 0,
        "command_order": suffix_start_index - command_order_offset,
        "command_bit_position": int(suffix_start["start"]),
        "buffer_read_bit_position": int(suffix_start["end"]),
        "last_demo_update": expected_live_update,
        "update": expected_live_update,
    }
    for key, value in expected_snapshot.items():
        if snapshot.get(key) != value:
            return suffix_start_index, None, (
                "clamped suffix first boundary "
                f"{key} mismatch: {snapshot.get(key)} != {value}"
            )

    shifted_updates: list[int] = []
    previous_update = recorded_update
    for row in rows[suffix_start_index:]:
        update = int(row["update"])
        if update < previous_update or update >= 0x7FFFFFFF:
            return suffix_start_index, None, (
                "clamped suffix is not a bounded monotonic int32 timeline"
            )
        shifted_updates.append(update + 1)
        previous_update = update
    for row, shifted_update in zip(
        rows[suffix_start_index:], shifted_updates, strict=True
    ):
        row["update"] = shifted_update
    return suffix_start_index, 1, None


def _audited_service_exit_late_idle_bridge(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit one saturated DMO timing bridge consumed by a write callback.

    A DMO command can encode at most fifteen updates of delay.  A successful
    asynchronous file-write corridor can therefore be followed by a long-form
    idle at that exact limit and another adjacent idle for the remaining
    delay.  The callback may consume the first idle while returning at the
    second idle's exact update.  Accept only that fully bound ``15 + n``
    bridge; command payloads, bit positions, order, and the following update
    must all be exact.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index + 2 >= len(rows)
        or previous_read_bit_position < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError("service late-idle bridge boundary is invalid")
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, "service late-idle bridge corridor is ineligible"

    bridge_index = corridor_end_index + 1
    following_index = bridge_index + 1
    corridor_end = rows[corridor_end_index]
    bridge = rows[bridge_index]
    following = rows[following_index]
    bridge_start = int(bridge["start"])
    bridge_end = int(bridge["end"])
    corridor_update = int(corridor_end["update"])
    bridge_update = int(bridge["update"])
    following_update = int(following["update"])

    if (
        int(corridor_end["end"]) != bridge_start
        or previous_read_bit_position != bridge_start
        or bridge.get("kind") != "idle"
        or bool(bridge["short_form"])
        or int(bridge["command_number"]) != 31
        or bridge.get("payload") != {}
        or bridge_end - bridge_start != 10
        or int(following["start"]) != bridge_end
        or following.get("kind") != "idle"
        or bool(following["short_form"])
        or int(following["command_number"]) != 31
        or following.get("payload") != {}
        or int(following["end"]) - int(following["start"]) != 10
    ):
        return bridge_index, (
            "service late-idle bridge is not two exact adjacent long idles"
        )

    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 31,
        "is_short": 0,
        "command_order": bridge_index - command_order_offset,
        "command_bit_position": bridge_start,
        "buffer_read_bit_position": bridge_end,
        "demo_loading_complete": 1,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            return bridge_index, (
                f"service late-idle bridge {bridge_index} {key} mismatch: "
                f"{snapshot.get(key)} != {value}"
            )

    bridge_lateness = snapshot["update"] - bridge_update
    if not (
        bridge_update - corridor_update
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and 1
        <= bridge_lateness
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and snapshot["update"] == snapshot["last_demo_update"]
        and snapshot["update"] == following_update
    ):
        return bridge_index, (
            "service late-idle bridge is outside its exact saturated "
            "timeline: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}, "
            f"corridor={corridor_update}, bridge={bridge_update}, "
            f"following={following_update}"
        )
    return bridge_index, None


def _audited_service_exit_overdue_idle_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit one overdue idle strictly between two saturated idle rows.

    A successful asynchronous file-write corridor can return after the first
    saturated fifteen-update idle is already due, while the following exact
    long idle is still in the future.  This is narrower than the late-idle
    bridge above: it accepts only the open interval between two adjacent,
    payload-free long idles and never changes the recorded timeline.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index + 2 >= len(rows)
        or previous_read_bit_position < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError("service overdue-idle re-entry boundary is invalid")
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, "service overdue-idle re-entry corridor is ineligible"

    bridge_index = corridor_end_index + 1
    following_index = bridge_index + 1
    corridor_end = rows[corridor_end_index]
    bridge = rows[bridge_index]
    following = rows[following_index]
    bridge_start = int(bridge["start"])
    bridge_end = int(bridge["end"])
    following_start = int(following["start"])
    following_end = int(following["end"])
    corridor_update = int(corridor_end["update"])
    bridge_update = int(bridge["update"])
    following_update = int(following["update"])

    if (
        int(corridor_end["end"]) != bridge_start
        or previous_read_bit_position != bridge_start
        or bridge.get("kind") != "idle"
        or bool(bridge["short_form"])
        or int(bridge["command_number"]) != 31
        or bridge.get("payload") != {}
        or bridge_end - bridge_start != 10
        or following_start != bridge_end
        or following.get("kind") != "idle"
        or bool(following["short_form"])
        or int(following["command_number"]) != 31
        or following.get("payload") != {}
        or following_end - following_start != 10
    ):
        return bridge_index, (
            "service overdue-idle re-entry is not two exact adjacent "
            "long idles"
        )

    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 31,
        "is_short": 0,
        "command_order": bridge_index - command_order_offset,
        "command_bit_position": bridge_start,
        "buffer_read_bit_position": bridge_end,
        "demo_loading_complete": 1,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            return bridge_index, (
                f"service overdue-idle re-entry {bridge_index} {key} "
                f"mismatch: {snapshot.get(key)} != {value}"
            )

    bridge_lateness = snapshot["update"] - bridge_update
    following_remaining = following_update - snapshot["update"]
    if not (
        bridge_update - corridor_update
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and following_update - bridge_update
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and 1
        <= bridge_lateness
        < SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and 1
        <= following_remaining
        < SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and bridge_lateness + following_remaining
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and snapshot["update"] == snapshot["last_demo_update"]
        and bridge_update < snapshot["update"] < following_update
    ):
        return bridge_index, (
            "service overdue-idle re-entry is outside its exact open "
            "saturated timeline: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}, "
            f"corridor={corridor_update}, bridge={bridge_update}, "
            f"following={following_update}"
        )
    return bridge_index, None


def _audited_service_exit_overdue_idle_prepared_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit the second exact call that advances an overdue idle row."""

    if (
        previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind != 10
    ):
        raise ValueError(
            "service overdue-idle prepared re-entry is invalid"
        )
    if snapshot["needs_command"] != 1:
        return None, (
            "service overdue-idle prepared re-entry requires a pending "
            "command"
        )
    if (
        previous_read_bit_position
        != snapshot["buffer_read_bit_position"]
        or previous_command_bit_position
        != snapshot["command_bit_position"]
        or previous_command_order != snapshot["command_order"]
        or previous_update != snapshot["update"]
    ):
        return None, (
            "service overdue-idle prepared re-entry changed its exact "
            "first-stage boundary"
        )

    completed_snapshot = dict(snapshot)
    completed_snapshot["needs_command"] = 0
    return _audited_service_exit_overdue_idle_reentry(
        rows,
        completed_snapshot,
        corridor_start_index=corridor_start_index,
        corridor_end_index=corridor_end_index,
        previous_read_bit_position=int(
            rows[corridor_end_index + 1]["start"]
        ),
        previous_reentry_kind=1,
        command_order_offset=command_order_offset,
    )


def _audited_service_exit_late_idle_bridge_prepared_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit the second same-update call that advances a late idle bridge."""

    if (
        previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind != 8
    ):
        raise ValueError(
            "service late-idle bridge prepared re-entry is invalid"
        )
    if snapshot["needs_command"] != 1:
        return None, (
            "service late-idle bridge prepared re-entry requires a pending "
            "command"
        )
    if (
        previous_read_bit_position
        != snapshot["buffer_read_bit_position"]
        or previous_command_bit_position
        != snapshot["command_bit_position"]
        or previous_command_order != snapshot["command_order"]
        or previous_update != snapshot["update"]
    ):
        return None, (
            "service late-idle bridge prepared re-entry changed its exact "
            "first-stage boundary"
        )

    completed_snapshot = dict(snapshot)
    completed_snapshot["needs_command"] = 0
    return _audited_service_exit_late_idle_bridge(
        rows,
        completed_snapshot,
        corridor_start_index=corridor_start_index,
        corridor_end_index=corridor_end_index,
        previous_read_bit_position=int(
            rows[corridor_end_index + 1]["start"]
        ),
        previous_reentry_kind=1,
        command_order_offset=command_order_offset,
    )


def _audited_late_idle_native_timeline_rebase(
    rows: list[dict[str, object]],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    bridge_index: int,
    live_bridge_update: int,
) -> tuple[int | None, int | None, str | None]:
    """Shift only the native verification timeline after an audited bridge.

    A saturated 15-update DMO wait can finish late when a successful retail
    file-write callback occupies more than 15 framework updates.  The command
    stream still contains the correct relative waits.  Once both exact bridge
    handshakes have been audited, carry the measured lateness into every later
    *verification* row while leaving the bridge and caller-owned offline rows
    unchanged.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or bridge_index != corridor_end_index + 1
        or bridge_index + 1 >= len(rows)
        or live_bridge_update < 0
    ):
        raise ValueError("late-idle native timeline rebase boundary is invalid")
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, None, (
            "late-idle native timeline rebase corridor is ineligible"
        )

    bridge = rows[bridge_index]
    following = rows[bridge_index + 1]
    bridge_start = int(bridge["start"])
    bridge_end = int(bridge["end"])
    bridge_update = int(bridge["update"])
    following_update = int(following["update"])
    if (
        bridge.get("kind") != "idle"
        or bool(bridge["short_form"])
        or int(bridge["command_number"]) != 31
        or bridge.get("payload") != {}
        or bridge_end - bridge_start != 10
        or int(following["start"]) != bridge_end
        or following.get("kind") != "idle"
        or bool(following["short_form"])
        or int(following["command_number"]) != 31
        or following.get("payload") != {}
        or int(following["end"]) - int(following["start"]) != 10
    ):
        return None, None, (
            "late-idle native timeline rebase is not two exact adjacent long idles"
        )

    delta = live_bridge_update - bridge_update
    if not (
        bridge_update
        - int(rows[corridor_end_index]["update"])
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and 1 <= delta <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and following_update == live_bridge_update
    ):
        return None, None, (
            "late-idle native timeline rebase is outside its exact saturated "
            "timeline"
        )

    shifted_updates: list[int] = []
    previous_update = bridge_update
    for row in rows[bridge_index + 1 :]:
        update = int(row["update"])
        if update < previous_update or update > 0x7FFFFFFF - delta:
            return None, None, (
                "late-idle native timeline suffix is not a bounded monotonic "
                "int32 timeline"
            )
        shifted_updates.append(update + delta)
        previous_update = update

    suffix_start_index = bridge_index + 1
    for row, shifted_update in zip(
        rows[suffix_start_index:],
        shifted_updates,
        strict=True,
    ):
        row["update"] = shifted_update
    return suffix_start_index, delta, None


def _audited_successful_file_write_tail_prefetch(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit one exact 24-bit prefetch across a successful write tail.

    Retail playback can refill its bit reader while the final successful
    file-write result remains prepared.  The observed refill consumes the
    remaining write-tail bits, one complete not-yet-due long idle, and exactly
    one timing bit of the following long idle.  It does not execute either idle
    command.  Bind every state field and both inert suffix rows before treating
    that refill as a transient continuation.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index + 2
        or corridor_end_index + 2 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError("successful file-write tail prefetch boundary is invalid")
    if not _is_uniform_successful_file_write_corridor(
        rows,
        corridor_start_index,
        corridor_end_index,
    ):
        return None, "successful file-write tail prefetch corridor is ineligible"

    matched_index = corridor_end_index - 2
    matched = rows[matched_index]
    corridor_end = int(rows[corridor_end_index]["end"])
    first_idle_index = corridor_end_index + 1
    second_idle_index = corridor_end_index + 2
    first_idle = rows[first_idle_index]
    second_idle = rows[second_idle_index]
    first_idle_start = int(first_idle["start"])
    first_idle_end = int(first_idle["end"])
    if (
        first_idle_start != corridor_end
        or first_idle.get("kind") != "idle"
        or bool(first_idle["short_form"])
        or int(first_idle["command_number"]) != 31
        or first_idle.get("payload") != {}
        or first_idle_end - first_idle_start != 10
        or int(second_idle["start"]) != first_idle_end
        or second_idle.get("kind") != "idle"
        or bool(second_idle["short_form"])
        or int(second_idle["command_number"]) != 31
        or second_idle.get("payload") != {}
        or int(second_idle["end"]) - int(second_idle["start"]) != 10
    ):
        return None, (
            "successful file-write tail prefetch lacks two exact inert long idles"
        )

    expected_command_bit_position = int(matched["start"]) + 10
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 0,
        "is_short": 0,
        "command_order": matched_index + 1 - command_order_offset,
        "command_bit_position": expected_command_bit_position,
        "buffer_read_bit_position": corridor_end + 11,
        "update": previous_update,
        "last_demo_update": previous_update,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return second_idle_index, (
                f"successful file-write tail prefetch {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    if (
        previous_command_bit_position != expected_command_bit_position
        or previous_read_bit_position != expected_command_bit_position + 10
        or previous_command_order != snapshot["command_order"]
        or snapshot["buffer_read_bit_position"]
        != previous_read_bit_position + 24
    ):
        return second_idle_index, (
            "successful file-write tail prefetch changed its exact prepared state"
        )
    if not (
        int(rows[corridor_end_index]["update"])
        <= snapshot["update"]
        < int(first_idle["update"])
        and int(first_idle["update"]) - snapshot["update"]
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
    ):
        return second_idle_index, (
            "successful file-write tail prefetch crossed a due idle timeline"
        )
    return second_idle_index, None


def _audited_preloading_failed_file_write_tail_prefetch(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
    command_order_rebase_rows: tuple[int, ...] = (),
) -> tuple[int | None, str | None]:
    """Audit one exact pre-loading failed-write tail prefetch.

    The usual form begins from an already prepared payload and refills 24
    bits.  A real current-update file-write callback can instead return at the
    preceding command header; that independently observed header requires one
    exact 34-bit refill.  Both variants are bound to the same immutable failed
    write corridor and two inert long-idle suffix rows.  When earlier payload
    reads have exactly rebased two immediately preceding rows, retail can
    retain the penultimate payload command and refill the same 24 bits one
    row later.  That third form is accepted only when the complete rebase
    registry proves those two rows were already accounted.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index + 2
        or corridor_end_index + 2 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind not in (1, 2)
        or command_order_offset < 0
        or command_order_offset != len(command_order_rebase_rows)
        or tuple(sorted(set(command_order_rebase_rows)))
        != command_order_rebase_rows
        or corridor_end_index - corridor_start_index + 1
        > FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
    ):
        raise ValueError(
            "preloading failed file-write tail prefetch boundary is invalid"
        )

    previous_end: int | None = None
    previous_row_update: int | None = None
    for row in rows[corridor_start_index : corridor_end_index + 1]:
        row_start = int(row["start"])
        row_end = int(row["end"])
        row_update = int(row["update"])
        if (
            (previous_end is not None and row_start != previous_end)
            or (
                previous_row_update is not None
                and row_update < previous_row_update
            )
            or bool(row["short_form"])
            or int(row["command_number"]) != 16
            or row.get("kind") != "file_write"
            or row_end - row_start != 11
            or row.get("payload") != {"success": False}
        ):
            return None, (
                "preloading failed file-write tail prefetch corridor is "
                "ineligible"
            )
        previous_end = row_end
        previous_row_update = row_update

    corridor_end = int(rows[corridor_end_index]["end"])
    first_idle_index = corridor_end_index + 1
    second_idle_index = corridor_end_index + 2
    first_idle = rows[first_idle_index]
    second_idle = rows[second_idle_index]
    first_idle_start = int(first_idle["start"])
    first_idle_end = int(first_idle["end"])
    if (
        first_idle_start != corridor_end
        or first_idle.get("kind") != "idle"
        or bool(first_idle["short_form"])
        or int(first_idle["command_number"]) != 31
        or first_idle.get("payload") != {}
        or first_idle_end - first_idle_start != 10
        or int(second_idle["start"]) != first_idle_end
        or second_idle.get("kind") != "idle"
        or bool(second_idle["short_form"])
        or int(second_idle["command_number"]) != 31
        or second_idle.get("payload") != {}
        or int(second_idle["end"]) - int(second_idle["start"]) != 10
    ):
        return None, (
            "preloading failed file-write tail prefetch lacks two exact "
            "inert long idles"
        )

    standard_matched_index = corridor_end_index - 2
    late_rebased_matched_index = corridor_end_index - 1
    standard_command_bit_position = (
        int(rows[standard_matched_index]["start"]) + 10
    )
    late_rebased_command_bit_position = (
        int(rows[late_rebased_matched_index]["start"]) + 10
    )
    prepared_payload_refill = (
        previous_reentry_kind == 1
        and previous_command_bit_position
        == standard_command_bit_position
        and previous_read_bit_position
        == standard_command_bit_position + 10
        and previous_command_order == snapshot["command_order"]
        and snapshot["command_bit_position"]
        == standard_command_bit_position
    )
    current_header_refill = (
        previous_reentry_kind == 2
        and previous_command_bit_position
        == int(rows[standard_matched_index]["start"])
        and previous_read_bit_position
        == standard_command_bit_position
        and previous_command_order == snapshot["command_order"] - 1
        and snapshot["command_bit_position"]
        == standard_command_bit_position
    )
    rebased_late_payload_refill = (
        previous_reentry_kind == 1
        and command_order_offset >= 2
        and {
            late_rebased_matched_index - 2,
            late_rebased_matched_index - 1,
        }.issubset(command_order_rebase_rows)
        and previous_command_bit_position
        == late_rebased_command_bit_position
        and previous_read_bit_position
        == late_rebased_command_bit_position + 10
        and previous_command_order == snapshot["command_order"]
        and snapshot["command_bit_position"]
        == late_rebased_command_bit_position
    )
    if not (
        prepared_payload_refill
        or current_header_refill
        or rebased_late_payload_refill
    ):
        return second_idle_index, (
            "preloading failed file-write tail prefetch changed its exact "
            "prepared state"
        )
    matched_index = (
        late_rebased_matched_index
        if rebased_late_payload_refill
        else standard_matched_index
    )
    matched = rows[matched_index]
    expected_command_bit_position = int(matched["start"]) + 10
    expected_buffer_read_bit_position = (
        corridor_end + 22
        if rebased_late_payload_refill
        else corridor_end + 11
    )
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 0,
        "is_short": 0,
        "command_order": matched_index + 1 - command_order_offset,
        "command_bit_position": expected_command_bit_position,
        "buffer_read_bit_position": expected_buffer_read_bit_position,
        "update": previous_update,
        "last_demo_update": previous_update,
        "demo_loading_complete": 0,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return second_idle_index, (
                f"preloading failed file-write tail prefetch {key} "
                f"mismatch: {snapshot[key]} != {value}"
            )
    matched_update = int(matched["update"])
    corridor_end_update = int(rows[corridor_end_index]["update"])
    first_idle_update = int(first_idle["update"])
    second_idle_update = int(second_idle["update"])
    exact_preloading_timeline = (
        previous_reentry_kind == 1
        and matched_update == snapshot["update"] + 1
        and corridor_end_update == snapshot["update"] + 2
    )
    exact_current_header_timeline = (
        previous_reentry_kind == 2
        and matched_update == snapshot["update"]
        and corridor_end_update == snapshot["update"] + 1
    )
    exact_rebased_late_payload_timeline = (
        rebased_late_payload_refill
        and matched_update == snapshot["update"]
        and corridor_end_update == snapshot["update"] + 1
    )
    if not (
        (
            exact_preloading_timeline
            or exact_current_header_timeline
            or exact_rebased_late_payload_timeline
        )
        and first_idle_update - corridor_end_update
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        and second_idle_update - first_idle_update
        == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
    ):
        return second_idle_index, (
            "preloading failed file-write tail prefetch is outside its exact "
            "pre-loading timeline"
        )
    return second_idle_index, None


def _audited_preloading_failed_file_write_tail_short_header_recovery(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
    command_order_rebase_rows: tuple[int, ...],
) -> tuple[int | None, dict[str, int] | None, str | None]:
    """Audit and undo one exact false short header after a tail prefetch.

    The rebased pre-loading failed-write tail can refill through two complete
    idle rows and two bits into a third saturated idle.  If that physical
    refill cursor is then mistaken for a logical command boundary, the
    remaining idle bits deterministically decode as an eleven-update short
    mouse-button header.  The payload has not run while ``needs_command`` is
    zero, so this is the last safe interception point.

    Accept only the immutable live suffix and complete rebase registry, then
    return the canonical state immediately after the final failed-write row.
    The caller may write that state atomically before resuming retail code.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index + 7
        or corridor_end_index + 4 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind != 11
        or command_order_offset != 4
        or command_order_offset != len(command_order_rebase_rows)
        or tuple(sorted(set(command_order_rebase_rows)))
        != command_order_rebase_rows
    ):
        raise ValueError(
            "preloading failed file-write tail short-header recovery "
            "boundary is invalid"
        )

    expected_rebase_rows = (
        corridor_end_index - 7,
        corridor_end_index - 6,
        corridor_end_index - 3,
        corridor_end_index - 2,
    )
    if command_order_rebase_rows != expected_rebase_rows:
        return None, None, (
            "preloading failed file-write tail short-header recovery "
            "registry mismatch"
        )

    tail_snapshot = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 0,
        "is_short": 0,
        "command_order": previous_command_order,
        "command_bit_position": previous_command_bit_position,
        "buffer_read_bit_position": previous_read_bit_position,
        "update": previous_update,
        "last_demo_update": previous_update,
        "demo_loading_complete": 0,
    }
    tail_row, tail_failure = (
        _audited_preloading_failed_file_write_tail_prefetch(
            rows,
            tail_snapshot,
            corridor_start_index=corridor_start_index,
            corridor_end_index=corridor_end_index,
            previous_read_bit_position=(
                previous_command_bit_position + 10
            ),
            previous_command_bit_position=previous_command_bit_position,
            previous_command_order=previous_command_order,
            previous_update=previous_update,
            previous_reentry_kind=1,
            command_order_offset=command_order_offset,
            command_order_rebase_rows=command_order_rebase_rows,
        )
    )
    second_idle_index = corridor_end_index + 2
    if tail_failure is not None or tail_row != second_idle_index:
        return None, None, (
            "preloading failed file-write tail short-header recovery lacks "
            "its exact audited predecessor: "
            f"{tail_failure or tail_row}"
        )

    third_idle_index = corridor_end_index + 3
    loading_complete_index = corridor_end_index + 4
    second_idle = rows[second_idle_index]
    third_idle = rows[third_idle_index]
    loading_complete = rows[loading_complete_index]
    third_start = int(third_idle["start"])
    third_end = int(third_idle["end"])
    if (
        int(second_idle["end"]) != third_start
        or third_idle.get("kind") != "idle"
        or bool(third_idle["short_form"])
        or int(third_idle["command_number"]) != 31
        or third_idle.get("payload") != {}
        or third_end - third_start != 10
        or int(third_idle["update"])
        - int(second_idle["update"])
        != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        or int(loading_complete["start"]) != third_end
        or loading_complete.get("kind") != "loading_complete"
        or bool(loading_complete["short_form"])
        or int(loading_complete["command_number"]) != 9
        or loading_complete.get("payload") != {}
        or int(loading_complete["end"])
        - int(loading_complete["start"])
        != 10
        or int(loading_complete["update"])
        - int(third_idle["update"])
        != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES - 1
    ):
        return third_idle_index, None, (
            "preloading failed file-write tail short-header recovery "
            "suffix mismatch"
        )

    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 1,
        "is_short": 1,
        "command_order": corridor_end_index + 1 - command_order_offset,
        "command_bit_position": third_start + 2,
        "buffer_read_bit_position": third_start + 8,
        "update": previous_update + 11,
        "last_demo_update": previous_update + 11,
        "demo_loading_complete": 0,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            return third_idle_index, None, (
                "preloading failed file-write tail short-header recovery "
                f"{key} mismatch: {snapshot.get(key)} != {value}"
            )
    if (
        previous_read_bit_position != third_start + 2
        or snapshot["command_bit_position"]
        != previous_read_bit_position
        or snapshot["buffer_read_bit_position"]
        != previous_read_bit_position + 6
        or snapshot["command_order"] != previous_command_order + 1
        or int(rows[corridor_end_index]["update"])
        != previous_update + 1
        or not (
            int(rows[corridor_end_index]["update"])
            < snapshot["update"]
            < int(rows[corridor_end_index + 1]["update"])
        )
    ):
        return third_idle_index, None, (
            "preloading failed file-write tail short-header recovery "
            "transition mismatch"
        )

    final_write = rows[corridor_end_index]
    recovery = {
        "buffer_read_bit_position": int(final_write["end"]),
        "last_demo_update": int(final_write["update"]),
        "needs_command": 1,
        "is_short": 0,
        "command_number": 16,
        "command_order": corridor_end_index - command_order_offset,
        "command_bit_position": int(final_write["start"]),
        "demo_loading_complete": 0,
    }
    return third_idle_index, recovery, None


def _audited_service_exit_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit one bounded early non-service boundary after a service corridor."""

    if (
        corridor_end_index < 0
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_reentry_kind != 1
        or command_order_offset < 0
    ):
        raise ValueError("service exit re-entry boundary is invalid")
    exit_index = corridor_end_index + 1
    row = rows[exit_index]
    command_start = int(row["start"])
    if (
        int(rows[corridor_end_index]["end"]) != command_start
        or snapshot["command_bit_position"] != command_start
        or previous_read_bit_position != command_start
        or int(row["command_number"]) in SERVICE_COMMAND_NUMBERS
    ):
        return None, "service exit re-entry is not the exact corridor exit"
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "command_order": exit_index - command_order_offset,
        "buffer_read_bit_position": int(row["end"]),
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return exit_index, (
                f"service exit re-entry {exit_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    row_update = int(row["update"])
    corridor_end_update = int(rows[corridor_end_index]["update"])
    if not (
        snapshot["update"] == snapshot["last_demo_update"]
        and corridor_end_update < snapshot["update"] < row_update
        and row_update - snapshot["update"]
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
    ):
        return exit_index, (
            "service exit re-entry is outside the bounded post-corridor "
            "timeline gap: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}, "
            f"corridor_end={corridor_end_update}, row={row_update}"
        )
    return exit_index, None


def _audited_service_exit_prepared_reentry(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit one complete exit header in the bounded timeline gap."""

    if (
        corridor_end_index < 0
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind != 4
        or command_order_offset < 0
    ):
        raise ValueError("service exit prepared re-entry boundary is invalid")
    exit_index = corridor_end_index + 1
    row = rows[exit_index]
    command_start = int(row["start"])
    command_end = int(row["end"])
    expected_order = exit_index - command_order_offset
    if (
        int(rows[corridor_end_index]["end"]) != command_start
        or int(row["command_number"]) in SERVICE_COMMAND_NUMBERS
        or previous_command_bit_position != command_start
        or previous_read_bit_position != command_end
        or previous_command_order != expected_order
        or previous_update != snapshot["update"]
    ):
        return None, (
            "service exit prepared re-entry does not follow its exact header"
        )
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": int(row["command_number"]),
        "is_short": int(bool(row["short_form"])),
        "command_order": expected_order,
        "command_bit_position": command_start,
        "buffer_read_bit_position": command_end,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return exit_index, (
                f"service exit prepared re-entry {exit_index} {key} "
                f"mismatch: {snapshot[key]} != {value}"
            )
    row_update = int(row["update"])
    corridor_end_update = int(rows[corridor_end_index]["update"])
    if not (
        snapshot["update"] == snapshot["last_demo_update"]
        and corridor_end_update < snapshot["update"] < row_update
        and row_update - snapshot["update"]
        <= SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
    ):
        return exit_index, (
            "service exit prepared re-entry is outside the bounded "
            "post-corridor timeline gap: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}, "
            f"corridor_end={corridor_end_update}, row={row_update}"
        )
    return exit_index, None


def _audited_service_exit_partial_header(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit the retail parser's split service-payload/idle-header exit."""

    if (
        corridor_end_index < 0
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_reentry_kind != 2
        or command_order_offset < 0
    ):
        raise ValueError("service partial-exit boundary is invalid")
    service_row = rows[corridor_end_index]
    exit_index = corridor_end_index + 1
    exit_row = rows[exit_index]
    service_start = int(service_row["start"])
    service_end = int(service_row["end"])
    exit_start = int(exit_row["start"])
    if (
        bool(service_row["short_form"])
        or int(service_row["command_number"])
        not in SERVICE_COMMAND_NUMBERS
        or service_end - service_start != 11
        or exit_start != service_end
        or bool(exit_row["short_form"])
        or int(exit_row["command_number"]) != 31
        or exit_row.get("kind") != "idle"
        or int(exit_row["end"]) - exit_start != 10
    ):
        return None, "service partial-exit rows are ineligible"
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 0,
        "is_short": 1,
        "command_bit_position": service_start + 10,
        "command_order": exit_index - command_order_offset,
        "buffer_read_bit_position": exit_start + 5,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return exit_index, (
                f"service partial-exit {exit_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    if (
        previous_command_bit_position != service_start
        or previous_read_bit_position != service_start + 10
        or snapshot["command_order"] != previous_command_order + 1
    ):
        return exit_index, (
            "service partial-exit does not follow its prepared final row"
        )
    if not (
        snapshot["update"] == snapshot["last_demo_update"]
        and int(service_row["update"])
        < snapshot["update"]
        < int(exit_row["update"])
    ):
        return exit_index, (
            "service partial-exit timing is not between corridor and exit: "
            f"framework={snapshot['update']}, "
            f"last={snapshot['last_demo_update']}"
        )
    return exit_index, None


def _audited_service_exit_forced_read(
    rows: list[dict[str, object]],
    snapshot: dict[str, int],
    *,
    corridor_end_index: int,
    previous_read_bit_position: int,
    previous_command_bit_position: int,
    previous_command_order: int,
    previous_update: int,
    previous_reentry_kind: int,
    command_order_offset: int,
) -> tuple[int | None, str | None]:
    """Audit a fixed-size forced read inside the inert idle exit suffix."""

    if (
        corridor_end_index < 0
        or corridor_end_index + 1 >= len(rows)
        or previous_read_bit_position < 0
        or previous_command_bit_position < 0
        or previous_command_order < 0
        or previous_update < 0
        or previous_reentry_kind not in (5, 6)
        or command_order_offset < 0
    ):
        raise ValueError("service exit forced-read boundary is invalid")
    service_row = rows[corridor_end_index]
    service_start = int(service_row["start"])
    exit_index = corridor_end_index + 1
    idle_end_index = exit_index - 1
    prior_end = int(service_row["end"])
    for index in range(exit_index, len(rows)):
        row = rows[index]
        if (
            int(row["start"]) != prior_end
            or bool(row["short_form"])
            or int(row["command_number"]) != 31
            or row.get("kind") != "idle"
            or int(row["end"]) - int(row["start"]) != 10
        ):
            break
        idle_end_index = index
        prior_end = int(row["end"])
    if idle_end_index < exit_index:
        return None, "service exit forced-read has no inert idle suffix"
    expected = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 0,
        "is_short": 1,
        "command_bit_position": service_start + 10,
        "command_order": exit_index - command_order_offset,
        "buffer_read_bit_position": previous_read_bit_position + 12,
        "update": previous_update,
        "last_demo_update": previous_update,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            return idle_end_index, (
                f"service exit forced-read {idle_end_index} {key} mismatch: "
                f"{snapshot[key]} != {value}"
            )
    if (
        previous_command_bit_position != service_start + 10
        or snapshot["command_order"] != previous_command_order
        or snapshot["buffer_read_bit_position"] > int(rows[idle_end_index]["end"])
    ):
        return idle_end_index, (
            "service exit forced-read escaped its inert idle suffix"
        )
    return idle_end_index, None


def _wait_for_service_boundary(
    *,
    target_end: int,
    settle_end: int | None = None,
    timeout: float,
    read_state: Callable[[], tuple[int, int]],
    monotonic: Callable[[], float],
    delay: Callable[[float], None],
    poll_interval: float = SERVICE_POLL_INTERVAL_SECONDS,
    allow_intermediate_prepared: bool = True,
) -> tuple[int, int, str | None]:
    """Wait for a service end or stable next-header handoff."""

    effective_settle_end = (
        target_end if settle_end is None else settle_end
    )
    if (
        target_end < 0
        or effective_settle_end < target_end
        or timeout <= 0
        or poll_interval <= 0
        or not isinstance(allow_intermediate_prepared, bool)
    ):
        raise ValueError("service boundary wait parameters must be positive")
    deadline = monotonic() + timeout
    observed_read = -1
    observed_needs = -1
    while monotonic() < deadline:
        try:
            observed_read, observed_needs = read_state()
        except OSError as error:
            return (
                observed_read,
                observed_needs,
                f"service broker read failed: {error}",
            )
        if (
            observed_read in (target_end, effective_settle_end)
            or (
                allow_intermediate_prepared
                and
                target_end < observed_read < effective_settle_end
                and observed_needs == 1
            )
        ):
            return observed_read, observed_needs, None
        if observed_read > effective_settle_end:
            return (
                observed_read,
                observed_needs,
                "service broker overshot target: "
                f"{observed_read} > {effective_settle_end}",
            )
        delay(poll_interval)
    return (
        observed_read,
        observed_needs,
        "service broker timed out: "
        f"read={observed_read}, target={target_end}",
    )


def _probe_service_progress(
    *,
    initial_read: int,
    target_end: int,
    settle_end: int | None = None,
    timeout: float,
    read_state: Callable[[], tuple[int, int]],
    monotonic: Callable[[], float],
    delay: Callable[[float], None],
    poll_interval: float = SERVICE_POLL_INTERVAL_SECONDS,
    allow_intermediate_prepared: bool = True,
) -> tuple[int, int, str, str | None]:
    """Classify whether another service thread is consuming the held block."""

    effective_settle_end = (
        target_end if settle_end is None else settle_end
    )
    if (
        initial_read < 0
        or target_end < initial_read
        or effective_settle_end < target_end
        or timeout <= 0
        or poll_interval <= 0
        or not isinstance(allow_intermediate_prepared, bool)
    ):
        raise ValueError("service progress probe parameters are invalid")
    deadline = monotonic() + timeout
    observed_read = initial_read
    observed_needs = -1
    while monotonic() < deadline:
        try:
            observed_read, observed_needs = read_state()
        except OSError as error:
            return (
                observed_read,
                observed_needs,
                "failure",
                f"service broker read failed: {error}",
            )
        if (
            observed_read in (target_end, effective_settle_end)
            or (
                allow_intermediate_prepared
                and
                target_end < observed_read < effective_settle_end
                and observed_needs == 1
            )
        ):
            return observed_read, observed_needs, "boundary", None
        if observed_read > effective_settle_end:
            return (
                observed_read,
                observed_needs,
                "failure",
                "service broker overshot target: "
                f"{observed_read} > {effective_settle_end}",
            )
        if observed_read < initial_read:
            return (
                observed_read,
                observed_needs,
                "failure",
                "service broker read position regressed: "
                f"{observed_read} < {initial_read}",
            )
        if observed_read > initial_read:
            return observed_read, observed_needs, "progress", None
        delay(poll_interval)
    return observed_read, observed_needs, "no_progress", None


def _wait_for_startup_post_bypass_worker_boundary(
    rows: list[dict[str, object]],
    *,
    corridor_start_index: int,
    corridor_end_index: int,
    target_row_index: int,
    initial_read_bit_position: int,
    command_order_offset: int,
    timeout: float,
    read_state: Callable[[], Mapping[str, int]],
    monotonic: Callable[[], float],
    delay: Callable[[float], None],
    alternate_target_row_index: int | None = None,
    close_window: Callable[[], None] | None = None,
    poll_interval: float = SERVICE_POLL_INTERVAL_SECONDS,
    settle_interval: float = (
        STARTUP_POST_BYPASS_WORKER_SETTLE_SECONDS
    ),
) -> tuple[dict[str, int], str, str | None]:
    """Wait for workers to settle at the exact broker target boundary.

    The replay main thread is held after its current PrepareDemoCommand call
    has returned.  Worker callbacks may consume only the manifest-bound
    failed-write corridor.  The handoff is usable only when the broker's
    original final row has been consumed exactly.  An earlier stable row is
    not sufficient: resuming there leaves a prepared callback ahead of the
    broker and can hide a startup command-order rebase.
    """

    if (
        corridor_start_index < 0
        or corridor_end_index < corridor_start_index
        or corridor_end_index >= len(rows)
        or target_row_index < corridor_start_index
        or target_row_index > corridor_end_index
        or (
            alternate_target_row_index is not None
            and (
                isinstance(alternate_target_row_index, bool)
                or not isinstance(alternate_target_row_index, int)
                or alternate_target_row_index <= target_row_index
                or alternate_target_row_index > corridor_end_index
            )
        )
        or initial_read_bit_position < 0
        or command_order_offset < 0
        or timeout <= 0
        or poll_interval <= 0
        or settle_interval <= 0
        or settle_interval >= timeout
    ):
        raise ValueError(
            "startup post-bypass worker boundary parameters are invalid"
        )
    if alternate_target_row_index is not None:
        target_update = int(rows[target_row_index]["update"])
        alternate_update = int(
            rows[alternate_target_row_index]["update"]
        )
        alternate_rows = rows[
            target_row_index + 1 : alternate_target_row_index + 1
        ]
        if (
            alternate_update != target_update + 1
            or not alternate_rows
            or any(
                int(row["update"]) != alternate_update
                for row in alternate_rows
            )
            or (
                alternate_target_row_index < corridor_end_index
                and int(
                    rows[alternate_target_row_index + 1]["update"]
                )
                == alternate_update
            )
        ):
            raise ValueError(
                "startup post-bypass alternate target is not one exact "
                "next-update block"
            )
    maximum_target_row_index = (
        target_row_index
        if alternate_target_row_index is None
        else alternate_target_row_index
    )
    target_end_bit_position = int(
        rows[maximum_target_row_index]["end"]
    )
    expected_consumed: dict[int, tuple[int, int]] = {}
    for row_index in range(
        corridor_start_index,
        corridor_end_index + 1,
    ):
        row = rows[row_index]
        row_start = int(row["start"])
        row_end = int(row["end"])
        if (
            bool(row["short_form"])
            or int(row["command_number"]) != 16
            or row.get("kind") != "file_write"
            or row_end - row_start != 11
            or row.get("payload") != {"success": False}
        ):
            raise ValueError(
                "startup post-bypass worker corridor is not exact"
            )
        if (
            row_index == target_row_index
            or (
                alternate_target_row_index is not None
                and target_row_index
                < row_index
                <= alternate_target_row_index
            )
        ):
            expected_consumed[row_end] = (row_index, row_start)

    required_fields = (
        "buffer_read_bit_position",
        "needs_command",
        "command_order",
        "command_bit_position",
        "is_short",
        "command_number",
        "demo_loading_complete",
        "update",
    )
    observed = {name: -1 for name in required_fields}
    progress_seen = False
    candidate_key: tuple[int, ...] | None = None
    candidate_since: float | None = None
    boundary_window_closed = False
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            raw_observed = read_state()
        except OSError as error:
            return observed, "failure", (
                "startup post-bypass worker state read failed: "
                f"{error}"
            )
        if any(
            name not in raw_observed
            or isinstance(raw_observed[name], bool)
            or not isinstance(raw_observed[name], int)
            for name in required_fields
        ):
            return observed, "failure", (
                "startup post-bypass worker state is malformed"
            )
        observed = {
            name: int(raw_observed[name])
            for name in required_fields
        }
        observed_read = observed["buffer_read_bit_position"]
        if observed_read < initial_read_bit_position:
            return observed, "failure", (
                "startup post-bypass worker read position regressed: "
                f"{observed_read} < {initial_read_bit_position}"
            )
        if observed_read > target_end_bit_position:
            return observed, "failure", (
                "startup post-bypass worker escaped its exact target: "
                f"{observed_read} > {target_end_bit_position}"
            )
        if observed["demo_loading_complete"] != 0:
            return observed, "failure", (
                "startup post-bypass worker crossed loading completion"
            )
        progress_seen = progress_seen or (
            observed_read > initial_read_bit_position
        )

        settled_row_index = -1
        settled_after_payload = 0
        settled_at_corridor_end = 0
        consumed = expected_consumed.get(observed_read)
        if consumed is not None:
            row_index, row_start = consumed
            if (
                observed["needs_command"] == 1
                and observed["command_order"]
                == row_index - command_order_offset
                and observed["command_bit_position"] == row_start
                and observed["is_short"] == 0
                and observed["command_number"] == 16
            ):
                settled_row_index = row_index
                settled_after_payload = 1
                settled_at_corridor_end = int(row_index == corridor_end_index)

        if progress_seen and settled_row_index >= 0:
            key = (
                observed_read,
                observed["needs_command"],
                observed["command_order"],
                observed["command_bit_position"],
                observed["is_short"],
                observed["command_number"],
                settled_row_index,
                settled_after_payload,
                settled_at_corridor_end,
            )
            now = monotonic()
            if key != candidate_key:
                candidate_key = key
                candidate_since = now
            elif (
                candidate_since is not None
                and now - candidate_since >= settle_interval
            ):
                if close_window is not None and not boundary_window_closed:
                    # Close the scheduling window before returning the
                    # accepted boundary.  The next loop iteration rereads
                    # every field while the competing workers are held; an
                    # advance that races this callback still fails through
                    # the ordinary exact-target checks above.
                    close_window()
                    boundary_window_closed = True
                    continue
                return {
                    **observed,
                    "settled_row_index": settled_row_index,
                    "settled_after_payload": settled_after_payload,
                    "settled_at_corridor_end": (
                        settled_at_corridor_end
                    ),
                }, "boundary", None
        else:
            candidate_key = None
            candidate_since = None
        delay(poll_interval)

    if not progress_seen:
        return observed, "no_progress", (
            "startup post-bypass worker made no progress: "
            f"read={observed['buffer_read_bit_position']}"
        )
    return observed, "failure", (
        "startup post-bypass worker did not settle at an exact offline "
        "boundary: read="
        f"{observed['buffer_read_bit_position']}, "
        f"needs={observed['needs_command']}, "
        f"order={observed['command_order']}"
    )


def _commit_startup_post_bypass_worker_continuation(
    rows: list[dict[str, object]],
    continuation: Mapping[str, object],
    observation: Mapping[str, int],
    *,
    target_row_index: int,
    alternate_target_row_index: int | None,
    initial_read_bit_position: int,
    command_order_offset: int,
) -> tuple[
    dict[str, object] | None,
    dict[str, int | str] | None,
    str | None,
]:
    """Commit one atomically frozen startup worker payload boundary.

    The worker can finish the broker target or one row in the immediately
    following update block.  Its frozen state is authoritative for the
    continuation ledger.  Kind 12 is deliberately transient: the resumed
    main thread must next echo this exact already-consumed prepared command
    before any ordinary continuation transition is accepted.
    """

    integer_fields = (
        "base",
        "thread_id",
        "start_index",
        "end_index",
        "end_bit_position",
        "last_read_bit_position",
        "last_command_bit_position",
        "last_command_order",
        "last_reentry_update",
        "last_reentry_kind",
        "reentries",
    )
    if (
        not isinstance(continuation, Mapping)
        or any(
            name not in continuation
            or isinstance(continuation[name], bool)
            or not isinstance(continuation[name], int)
            for name in integer_fields
        )
        or continuation.get(
            "allow_bounded_late_successful_file_write_corridor"
        )
        is not False
        or continuation.get(
            "allow_bounded_late_preloading_failed_file_write_corridor"
        )
        is not True
        or isinstance(target_row_index, bool)
        or not isinstance(target_row_index, int)
        or (
            alternate_target_row_index is not None
            and (
                isinstance(alternate_target_row_index, bool)
                or not isinstance(alternate_target_row_index, int)
            )
        )
        or isinstance(initial_read_bit_position, bool)
        or not isinstance(initial_read_bit_position, int)
        or initial_read_bit_position < 0
        or isinstance(command_order_offset, bool)
        or not isinstance(command_order_offset, int)
        or command_order_offset < 0
    ):
        raise ValueError(
            "startup worker continuation commit parameters are invalid"
        )

    start_index = int(continuation["start_index"])
    end_index = int(continuation["end_index"])
    if (
        start_index < 0
        or end_index < start_index
        or end_index >= len(rows)
        or target_row_index < start_index
        or target_row_index > end_index
        or (
            alternate_target_row_index is not None
            and (
                alternate_target_row_index <= target_row_index
                or alternate_target_row_index > end_index
            )
        )
    ):
        raise ValueError(
            "startup worker continuation commit corridor is invalid"
        )
    if alternate_target_row_index is not None:
        target_update = int(rows[target_row_index]["update"])
        alternate_update = int(
            rows[alternate_target_row_index]["update"]
        )
        if (
            alternate_update != target_update + 1
            or any(
                int(rows[row_index]["update"]) != alternate_update
                for row_index in range(
                    target_row_index + 1,
                    alternate_target_row_index + 1,
                )
            )
            or (
                alternate_target_row_index < end_index
                and int(
                    rows[alternate_target_row_index + 1]["update"]
                )
                == alternate_update
            )
        ):
            raise ValueError(
                "startup worker continuation alternate target is not "
                "one exact next-update block"
            )
    if (
        int(continuation["last_read_bit_position"])
        != initial_read_bit_position
        or int(continuation["last_command_bit_position"]) != -1
        or int(continuation["last_command_order"]) != -1
        or int(continuation["last_reentry_kind"]) != 0
        or int(continuation["reentries"]) != 0
        or initial_read_bit_position
        != int(rows[start_index]["start"]) + 10
        or int(continuation["end_bit_position"])
        != int(rows[end_index]["end"])
    ):
        return None, None, (
            "startup worker continuation ledger is not at its exact "
            "pre-yield boundary"
        )

    required_observation_fields = (
        "buffer_read_bit_position",
        "needs_command",
        "command_order",
        "command_bit_position",
        "is_short",
        "command_number",
        "demo_loading_complete",
        "update",
        "settled_row_index",
        "settled_after_payload",
        "settled_at_corridor_end",
    )
    if any(
        name not in observation
        or isinstance(observation[name], bool)
        or not isinstance(observation[name], int)
        for name in required_observation_fields
    ):
        return None, None, (
            "startup worker continuation observation is malformed"
        )
    settled_row_index = int(observation["settled_row_index"])
    maximum_row_index = (
        target_row_index
        if alternate_target_row_index is None
        else alternate_target_row_index
    )
    if not target_row_index <= settled_row_index <= maximum_row_index:
        return None, None, (
            "startup worker continuation settled outside its audited "
            "target window"
        )

    previous_end: int | None = None
    for row_index in range(start_index, settled_row_index + 1):
        row = rows[row_index]
        row_start = int(row["start"])
        row_end = int(row["end"])
        if (
            bool(row["short_form"])
            or int(row["command_number"]) != 16
            or row.get("kind") != "file_write"
            or row.get("payload") != {"success": False}
            or row_end - row_start != 11
            or (previous_end is not None and row_start != previous_end)
        ):
            return None, None, (
                "startup worker continuation consumed a non-exact "
                "failed-write corridor"
            )
        previous_end = row_end

    settled_row = rows[settled_row_index]
    expected = {
        "buffer_read_bit_position": int(settled_row["end"]),
        "needs_command": 1,
        "command_order": settled_row_index - command_order_offset,
        "command_bit_position": int(settled_row["start"]),
        "is_short": 0,
        "command_number": 16,
        "demo_loading_complete": 0,
        # A worker may preload the immediate next-update block while the
        # framework clock remains at the broker target update.  The frozen
        # command identity proves the preloaded row; the unchanged clock
        # independently proves that this did not escape the bounded window.
        "update": int(rows[target_row_index]["update"]),
        "settled_after_payload": 1,
        "settled_at_corridor_end": int(settled_row_index == end_index),
    }
    for name, value in expected.items():
        if int(observation[name]) != value:
            return None, None, (
                "startup worker continuation observation mismatch: "
                f"{name}={observation[name]} != {value}"
            )

    committed = dict(continuation)
    committed.update(
        {
            "last_read_bit_position": int(
                observation["buffer_read_bit_position"]
            ),
            "last_command_bit_position": int(
                observation["command_bit_position"]
            ),
            "last_command_order": int(observation["command_order"]),
            "last_reentry_update": int(observation["update"]),
            "last_reentry_kind": 12,
            "reentries": settled_row_index - start_index + 1,
        }
    )
    receipt: dict[str, int | str] = {
        "mechanism": "atomic_startup_worker_payload_boundary",
        "target_row_index": target_row_index,
        "maximum_target_row_index": maximum_row_index,
        "settled_row_index": settled_row_index,
        "initial_read_bit_position": initial_read_bit_position,
        "committed_read_bit_position": int(
            observation["buffer_read_bit_position"]
        ),
        "committed_command_bit_position": int(
            observation["command_bit_position"]
        ),
        "committed_command_order": int(observation["command_order"]),
        "committed_framework_update": int(observation["update"]),
        "settled_recorded_update": int(settled_row["update"]),
        "consumed_row_count": settled_row_index - start_index + 1,
        "transient_reentry_kind": 12,
    }
    return committed, receipt, None


def _consume_startup_post_bypass_worker_echo(
    rows: list[dict[str, object]],
    continuation: Mapping[str, object],
    snapshot: Mapping[str, int],
    *,
    thread_id: int,
    command_order_offset: int,
) -> tuple[
    dict[str, object] | None,
    dict[str, int | str] | None,
    str | None,
]:
    """Consume the one exact main-thread echo after a worker commit."""

    if (
        isinstance(thread_id, bool)
        or not isinstance(thread_id, int)
        or thread_id <= 0
        or isinstance(command_order_offset, bool)
        or not isinstance(command_order_offset, int)
        or command_order_offset < 0
    ):
        raise ValueError("startup worker echo parameters are invalid")
    required_continuation_fields = (
        "base",
        "thread_id",
        "start_index",
        "end_index",
        "last_read_bit_position",
        "last_command_bit_position",
        "last_command_order",
        "last_reentry_update",
        "last_reentry_kind",
        "reentries",
    )
    if any(
        name not in continuation
        or isinstance(continuation[name], bool)
        or not isinstance(continuation[name], int)
        for name in required_continuation_fields
    ):
        return None, None, "startup worker echo continuation is malformed"
    if int(continuation["last_reentry_kind"]) != 12:
        return None, None, "startup worker echo state is not armed"

    row_index = (
        int(continuation["last_command_order"])
        + command_order_offset
    )
    if not (
        int(continuation["start_index"])
        <= row_index
        <= int(continuation["end_index"])
        < len(rows)
    ):
        return None, None, "startup worker echo row is outside its corridor"
    row = rows[row_index]
    if (
        bool(row["short_form"])
        or int(row["command_number"]) != 16
        or row.get("kind") != "file_write"
        or row.get("payload") != {"success": False}
        or int(row["end"]) - int(row["start"]) != 11
    ):
        return None, None, "startup worker echo row is ineligible"

    expected = {
        "base": int(continuation["base"]),
        "return_address": MAIN_DEMO_CALLER,
        "buffer_read_bit_position": int(
            continuation["last_read_bit_position"]
        ),
        "needs_command": 1,
        "command_order": int(continuation["last_command_order"]),
        "command_bit_position": int(
            continuation["last_command_bit_position"]
        ),
        "is_short": 0,
        "command_number": 16,
        "demo_loading_complete": 0,
        "update": int(row["update"]),
        "last_demo_update": int(row["update"]),
    }
    if thread_id != int(continuation["thread_id"]):
        return None, None, "startup worker echo thread identity changed"
    for name, value in expected.items():
        observed = snapshot.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != value
        ):
            return None, None, (
                "startup worker echo observation mismatch: "
                f"{name}={observed} != {value}"
            )

    same_update_start_index = -1
    same_update_end_index = -1
    same_update_target_end_bit_position = -1
    if row_index + 1 <= int(continuation["end_index"]):
        following_index = row_index + 1
        if int(rows[following_index]["update"]) == int(row["update"]):
            same_update_start_index = following_index
            same_update_end_index = following_index
            while (
                same_update_end_index + 1
                <= int(continuation["end_index"])
                and int(rows[same_update_end_index + 1]["update"])
                == int(row["update"])
            ):
                same_update_end_index += 1
            same_update_target_end_bit_position = int(
                rows[same_update_end_index]["end"]
            )

    # The echo itself consumes no new offline row.  Return to the neutral
    # continuation state with the exact worker boundary retained.  If more
    # service rows belong to this same update, the caller can broker that
    # immutable suffix immediately, before another thread races its payload.
    advanced = dict(continuation)
    advanced.update(
        {
            "last_reentry_update": int(snapshot["update"]),
            "last_reentry_kind": 0,
        }
    )
    receipt: dict[str, int | str] = {
        "mechanism": "exact_main_echo_of_atomic_worker_boundary",
        "row_index": row_index,
        "framework_update": int(snapshot["update"]),
        "read_bit_position": int(snapshot["buffer_read_bit_position"]),
        "command_bit_position": int(snapshot["command_bit_position"]),
        "command_order": int(snapshot["command_order"]),
        "same_update_service_block_start_index": (
            same_update_start_index
        ),
        "same_update_service_block_end_index": same_update_end_index,
        "same_update_service_block_target_end_bit_position": (
            same_update_target_end_bit_position
        ),
        "next_reentry_kind": 0,
    }
    return advanced, receipt, None
