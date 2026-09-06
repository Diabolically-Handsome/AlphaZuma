from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest

from zuma_rl.pc_rng_trajectory import (
    POWERUP_SPAWN_CALLERS,
    PcRngTrajectoryError,
    RngCurveList,
    RngIdentity,
    RngTrajectoryFrame,
    analyze_rng_trajectory,
    load_rng_trajectory,
    mtrand_outputs_between,
    powerup_changes_between,
    verify_powerup_spawn_transition,
    verify_rng_call_trace_alignment,
)
from zuma_rl.pc_memory_trajectory import (
    INITIAL_SAMPLE_BARRIER,
    STEPPED_SAMPLE_BARRIER,
    TRAJECTORY_SAMPLE_PHASE,
)
from zuma_rl.revenge_core import PopCapMTRandom


TYPE3_ONLY_WEIGHTS = tuple(
    1 if powerup_type == 3 else 0
    for powerup_type in range(14)
)


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _identity(ball_id: int, color_id: int) -> dict[str, object]:
    return {
        "address": 0x10000000 + ball_id * 0x100,
        "address_hex": f"0x{0x10000000 + ball_id * 0x100:08x}",
        "kind": "ball",
        "vtable": 0x00960160,
        "ball_id": ball_id,
        "color_id": color_id,
        "powerup_previous_type": 14,
        "powerup_primary_type": 14,
        "powerup_secondary_type": 14,
    }


def _curve_list(offset: int, entities: list[dict[str, object]]) -> dict[str, object]:
    return {
        "curve_index": 0,
        "container_offset": offset,
        "sentinel_address": 0x20000000 + offset,
        "first_node_address": 0x20000100 + offset,
        "last_node_address": 0x20000100 + offset,
        "declared_count": len(entities),
        "traversed_count": len(entities),
        "topology_sha256": "sha256:" + "0" * 64,
        "entities": entities,
    }


def _write_fixture(root: Path) -> Path:
    rng = PopCapMTRandom(5489)
    states = [rng.state]
    expected_outputs: list[list[int]] = []
    for count in (2, 5):
        expected_outputs.append([rng.next_u31() for _ in range(count)])
        states.append(rng.state)
    records = [
        struct.pack("<625I", *words, index)
        for words, index in states
    ]
    blob = b"".join(records)
    blob_path = root / "global-mtrand-frames.bin"
    blob_path.write_bytes(blob)

    qrand = {
        "address": 0x30000000,
        "update_count": 2,
        "selected_index": 1,
        "vectors": {
            "weights": [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
            "sways": [0.0] * 6,
            "last_hit": [0] * 6,
            "previous_hit": [0] * 6,
        },
    }
    ticks = []
    for frame_index, record in enumerate(records):
        update = 100 + frame_index
        ticks.append(
            {
                "framework_update": update,
                "sample_phase": TRAJECTORY_SAMPLE_PHASE,
                "sample_barrier": (
                    INITIAL_SAMPLE_BARRIER
                    if frame_index == 0
                    else STEPPED_SAMPLE_BARRIER
                ),
                "replay_state": {
                    "update_count": update,
                    "frame_time_ms": 10,
                    "update_multiplier": 0.0877914951989026,
                    "fast_forward_target": (
                        0 if frame_index == 0 else update
                    ),
                    "fast_forward_to_marker": False,
                    "fast_forward_step": False,
                },
                "score": 10,
                "displayed_score": 10,
                "score_target": 100,
                "board_color_counts": [0, 0, 1, 0, 0, 0],
                "chain_ball_count": 1,
                "pending_ball_count": 1,
                "inserting_ball_count": 0,
                "fired_bullet_count": 0,
                "current_ball": _identity(2, 1),
                "next_ball": _identity(3, 3),
                "qrand": qrand,
                "thread_crt_rand_state": 1234,
                "global_mtrand_index": states[frame_index][1],
                "curve_lists": [
                    _curve_list(0x50, []),
                    _curve_list(0x5C, [_identity(1, 2)]),
                    _curve_list(0x68, [_identity(4, 0)]),
                ],
                "global_mtrand_record_offset": frame_index * 2500,
                "global_mtrand_record_bytes": 2500,
                "global_mtrand_record_sha256": _sha256(record),
            }
        )
    index = {
        "schema": "zuma-rl.pc-rng-trajectory",
        "version": 2,
        "sample_phase": TRAJECTORY_SAMPLE_PHASE,
        "start_update": 100,
        "end_update": 102,
        "tick_count": 3,
        "global_mtrand": {
            "address": 0x00A313C0,
            "address_hex": "0x00a313c0",
            "state_word_count": 624,
            "record_bytes": 2500,
            "artifact": blob_path.name,
            "artifact_bytes": len(blob),
            "artifact_sha256": _sha256(blob),
        },
        "ticks": ticks,
        "test_expected_outputs": expected_outputs,
    }
    index_path = root / "index.json"
    index_path.write_bytes(
        (
            json.dumps(
                index,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    return index_path


def _rewrite_qrand(
    index_path: Path,
    *,
    update_count: int,
    selected_index: int,
    lengths: tuple[int, int, int, int],
) -> None:
    payload = json.loads(index_path.read_text(encoding="ascii"))
    for tick in payload["ticks"]:
        qrand = tick["qrand"]
        qrand["update_count"] = update_count
        qrand["selected_index"] = selected_index
        for (name, _), length in zip(
            qrand["vectors"].items(),
            lengths,
            strict=True,
        ):
            qrand["vectors"][name] = [0] * length
    index_path.write_bytes(
        (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )


def test_loader_accepts_strict_uninitialized_qrand_sentinel(
    tmp_path: Path,
) -> None:
    index_path = _write_fixture(tmp_path)
    _rewrite_qrand(
        index_path,
        update_count=0,
        selected_index=-1,
        lengths=(0, 0, 0, 0),
    )

    frames = load_rng_trajectory(index_path)

    assert all(frame.qrand_update_count == 0 for frame in frames)
    assert all(frame.qrand_selected_index == -1 for frame in frames)
    assert all(
        all(not values for _, values in frame.qrand_vectors)
        for frame in frames
    )


def test_loader_accepts_initialized_qrand_before_first_selection(
    tmp_path: Path,
) -> None:
    index_path = _write_fixture(tmp_path)
    _rewrite_qrand(
        index_path,
        update_count=0,
        selected_index=-1,
        lengths=(6, 6, 6, 6),
    )

    frames = load_rng_trajectory(index_path)

    assert all(frame.qrand_update_count == 0 for frame in frames)
    assert all(frame.qrand_selected_index == -1 for frame in frames)
    assert all(
        all(len(values) == 6 for _, values in frame.qrand_vectors)
        for frame in frames
    )


@pytest.mark.parametrize(
    ("update_count", "selected_index", "lengths", "error"),
    [
        (
            0,
            -1,
            (0, 6, 0, 0),
            "rng_trajectory_qrand_vector_initialization_mismatch",
        ),
        (
            0,
            0,
            (0, 0, 0, 0),
            "rng_trajectory_qrand_uninitialized_state_invalid",
        ),
        (
            1,
            -1,
            (0, 0, 0, 0),
            "rng_trajectory_qrand_uninitialized_state_invalid",
        ),
    ],
)
def test_loader_rejects_partial_or_inconsistent_uninitialized_qrand(
    tmp_path: Path,
    update_count: int,
    selected_index: int,
    lengths: tuple[int, int, int, int],
    error: str,
) -> None:
    index_path = _write_fixture(tmp_path)
    _rewrite_qrand(
        index_path,
        update_count=update_count,
        selected_index=selected_index,
        lengths=lengths,
    )

    with pytest.raises(PcRngTrajectoryError, match=error):
        load_rng_trajectory(index_path)


def _write_call_trace(
    root: Path,
    frames: tuple[RngTrajectoryFrame, ...],
) -> Path:
    calls: list[dict[str, object]] = []
    for before, after in zip(frames, frames[1:]):
        rng = PopCapMTRandom(1)
        rng.load_state(before.mtrand_words, before.mtrand_index)
        outputs = mtrand_outputs_between(before, after)
        for expected_output in outputs:
            payload = struct.pack("<625I", *rng.words, rng.index)
            before_index = rng.index
            assert rng.next_u31() == expected_output
            caller = 0x00401000 + len(calls)
            calls.append(
                {
                    "order": len(calls),
                    "framework_update": after.update,
                    "thread_id": 123,
                    "caller": caller,
                    "caller_hex": f"0x{caller:08x}",
                    "mtrand_index_before": before_index,
                    "mtrand_index_after": rng.index,
                    "mtrand_output": expected_output,
                    "mtrand_state_sha256_before": _sha256(payload),
                    "stack_code": [],
                }
            )
    trace = {
        "schema": "zuma-rl.pc-global-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "call_count": len(calls),
        "runtime_executable_sha256": "sha256:" + "1" * 64,
        "calls": calls,
    }
    path = root / "call-trace.json"
    path.write_bytes(
        (
            json.dumps(
                trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    return path


def _powerup_frames_and_trace(
    root: Path,
) -> tuple[
    tuple[RngTrajectoryFrame, RngTrajectoryFrame],
    Path,
    int,
    int,
    tuple[int, ...],
]:
    base = load_rng_trajectory(_write_fixture(root))[0]
    rng = PopCapMTRandom(1)
    rng.load_state(base.mtrand_words, base.mtrand_index)
    outputs = tuple(rng.next_u31() for _ in range(4))
    selected_color = outputs[2] % 2
    other_color = 1 - selected_color
    candidate_ids = (109, 101, 86)
    selected_ball_id = candidate_ids[outputs[3] % len(candidate_ids)]
    before_entities = tuple(
        RngIdentity(
            ball_id=ball_id,
            color_id=selected_color,
            kind="ball",
            powerup_previous_type=14,
            powerup_primary_type=14,
            powerup_secondary_type=14,
        )
        for ball_id in candidate_ids
    ) + (
        RngIdentity(
            ball_id=9,
            color_id=other_color,
            kind="ball",
            powerup_previous_type=14,
            powerup_primary_type=14,
            powerup_secondary_type=14,
        ),
    )
    after_entities = tuple(
        replace(
            entity,
            powerup_secondary_type=3,
        )
        if entity.ball_id == selected_ball_id
        else entity
        for entity in before_entities
    )
    before_chain = RngCurveList(
        curve_index=0,
        container_offset=0x5C,
        entities=before_entities,
        topology_sha256="sha256:" + "2" * 64,
    )
    after_chain = replace(before_chain, entities=after_entities)

    def with_chain(
        frame: RngTrajectoryFrame,
        chain: RngCurveList,
    ) -> tuple[RngCurveList, ...]:
        return tuple(
            chain if item.container_offset == 0x5C else item
            for item in frame.curve_lists
        )

    before = replace(
        base,
        chain_ball_count=len(before_entities),
        curve_lists=with_chain(base, before_chain),
    )
    after = replace(
        before,
        update=before.update + 1,
        curve_lists=with_chain(before, after_chain),
        mtrand_words=rng.words,
        mtrand_index=rng.index,
    )

    trace_rng = PopCapMTRandom(1)
    trace_rng.load_state(before.mtrand_words, before.mtrand_index)
    calls: list[dict[str, object]] = []
    for order, ((caller, _), expected_output) in enumerate(
        zip(POWERUP_SPAWN_CALLERS, outputs, strict=True)
    ):
        payload = struct.pack(
            "<625I",
            *trace_rng.words,
            trace_rng.index,
        )
        index_before = trace_rng.index
        assert trace_rng.next_u31() == expected_output
        calls.append(
            {
                "order": order,
                "framework_update": after.update,
                "thread_id": 123,
                "caller": caller,
                "caller_hex": f"0x{caller:08x}",
                "mtrand_index_before": index_before,
                "mtrand_index_after": trace_rng.index,
                "mtrand_output": expected_output,
                "mtrand_state_sha256_before": _sha256(payload),
                "stack_code": [],
            }
        )
    trace = {
        "schema": "zuma-rl.pc-global-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "start_update": before.update,
        "end_update": after.update,
        "call_count": len(calls),
        "runtime_executable_sha256": "sha256:" + "1" * 64,
        "calls": calls,
    }
    trace_path = root / "powerup-call-trace.json"
    trace_path.write_bytes(
        (
            json.dumps(
                trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    return (
        (before, after),
        trace_path,
        selected_color,
        selected_ball_id,
        outputs,
    )


def test_load_and_analyze_exact_mtrand_transitions(tmp_path: Path) -> None:
    index_path = _write_fixture(tmp_path)
    frames = load_rng_trajectory(index_path)

    report = analyze_rng_trajectory(frames)

    source = json.loads(index_path.read_text(encoding="ascii"))
    assert report["status"] == "PASS"
    assert report["mtrand_call_count_histogram"] == {"2": 1, "5": 1}
    assert report["total_mtrand_calls"] == 7
    assert frames[0].current_ball is not None
    assert frames[0].current_ball.powerup_primary_type == 14
    assert report["transitions"][0]["mtrand_outputs"] == source[
        "test_expected_outputs"
    ][0]
    assert report["transitions"][1]["mtrand_outputs"] == source[
        "test_expected_outputs"
    ][1]


def test_powerup_spawn_verifier_replays_all_four_choices(
    tmp_path: Path,
) -> None:
    (
        frames,
        trace_path,
        selected_color,
        selected_ball_id,
        outputs,
    ) = _powerup_frames_and_trace(tmp_path)

    report = verify_powerup_spawn_transition(
        frames,
        trace_path,
        to_update=frames[1].update,
        curve_index=0,
        num_colors=2,
        chance_denominator=1,
        powerup_weights=TYPE3_ONLY_WEIGHTS,
        expected_powerup_type=3,
        expected_ball_id=selected_ball_id,
    )

    assert report["status"] == "PASS"
    assert report["mtrand_outputs"] == list(outputs)
    assert report["weighted_type_roll"]["selected_powerup_type"] == 3
    assert report["eligible_color_roll"]["selected_color"] == selected_color
    assert report["eligible_ball_roll"]["selected_ball_id"] == selected_ball_id
    assert report["observed_change"]["ball_id_after"] == selected_ball_id
    assert powerup_changes_between(*frames) == (
        report["observed_change"],
    )
    analysis = analyze_rng_trajectory(frames)
    assert analysis["version"] == 2
    assert analysis["transitions"][0]["powerup_changes"] == [
        report["observed_change"]
    ]


def test_powerup_spawn_verifier_rejects_wrong_caller(
    tmp_path: Path,
) -> None:
    frames, trace_path, _, selected_ball_id, _ = _powerup_frames_and_trace(
        tmp_path
    )
    trace = json.loads(trace_path.read_text(encoding="ascii"))
    trace["calls"][2]["caller"] += 1
    trace["calls"][2]["caller_hex"] = (
        f"0x{trace['calls'][2]['caller']:08x}"
    )
    trace_path.write_bytes(
        (
            json.dumps(
                trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )

    with pytest.raises(
        PcRngTrajectoryError,
        match="powerup_spawn_caller_sequence_mismatch",
    ):
        verify_powerup_spawn_transition(
            frames,
            trace_path,
            to_update=frames[1].update,
            curve_index=0,
            num_colors=2,
            chance_denominator=1,
            powerup_weights=TYPE3_ONLY_WEIGHTS,
            expected_powerup_type=3,
            expected_ball_id=selected_ball_id,
        )


def test_powerup_spawn_verifier_rejects_unrelated_score_mutation(
    tmp_path: Path,
) -> None:
    frames, trace_path, _, selected_ball_id, _ = _powerup_frames_and_trace(
        tmp_path
    )
    tampered = (frames[0], replace(frames[1], score=frames[1].score + 1))

    with pytest.raises(
        PcRngTrajectoryError,
        match="powerup_spawn_unrelated_mutation:score_unchanged",
    ):
        verify_powerup_spawn_transition(
            tampered,
            trace_path,
            to_update=tampered[1].update,
            curve_index=0,
            num_colors=2,
            chance_denominator=1,
            powerup_weights=TYPE3_ONLY_WEIGHTS,
            expected_powerup_type=3,
            expected_ball_id=selected_ball_id,
        )


def test_dynamic_call_trace_binds_to_every_transition(tmp_path: Path) -> None:
    frames = load_rng_trajectory(_write_fixture(tmp_path))
    trace_path = _write_call_trace(tmp_path, frames)

    report = verify_rng_call_trace_alignment(frames, trace_path)

    assert report["status"] == "PASS"
    assert report["call_count"] == 7
    assert report["thread_ids"] == [123]
    assert [row["call_count"] for row in report["transitions"]] == [2, 5]


def test_dynamic_call_trace_rejects_wrong_output(tmp_path: Path) -> None:
    frames = load_rng_trajectory(_write_fixture(tmp_path))
    trace_path = _write_call_trace(tmp_path, frames)
    trace = json.loads(trace_path.read_text(encoding="ascii"))
    trace["calls"][0]["mtrand_output"] ^= 1
    trace_path.write_bytes(
        (
            json.dumps(
                trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )

    with pytest.raises(
        PcRngTrajectoryError,
        match="rng_call_trace_transition_mismatch",
    ):
        verify_rng_call_trace_alignment(frames, trace_path)


def test_loader_rejects_tampered_mtrand_stream(tmp_path: Path) -> None:
    index_path = _write_fixture(tmp_path)
    stream = tmp_path / "global-mtrand-frames.bin"
    payload = bytearray(stream.read_bytes())
    payload[100] ^= 1
    stream.write_bytes(payload)

    with pytest.raises(
        PcRngTrajectoryError,
        match="rng_trajectory_mtrand_artifact_mismatch",
    ):
        load_rng_trajectory(index_path)


def test_loader_rejects_unfinished_replay_step(tmp_path: Path) -> None:
    index_path = _write_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="ascii"))
    index["ticks"][1]["replay_state"]["fast_forward_step"] = True
    index_path.write_bytes(
        (
            json.dumps(
                index,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )

    with pytest.raises(
        PcRngTrajectoryError,
        match="rng_trajectory_replay_state_invalid",
    ):
        load_rng_trajectory(index_path)


def test_loader_rejects_legacy_unproven_sample_phase(tmp_path: Path) -> None:
    index_path = _write_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="ascii"))
    index["version"] = 1
    index["sample_phase"] = "frozen_post_framework_update"
    index_path.write_bytes(
        (
            json.dumps(
                index,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )

    with pytest.raises(
        PcRngTrajectoryError,
        match="rng_trajectory_schema_invalid",
    ):
        load_rng_trajectory(index_path)


def test_transition_fails_closed_when_call_bound_is_too_small(
    tmp_path: Path,
) -> None:
    frames = load_rng_trajectory(_write_fixture(tmp_path))
    unrelated = PopCapMTRandom(123).state
    unreachable = replace(
        frames[1],
        mtrand_words=unrelated[0],
        mtrand_index=unrelated[1],
    )

    with pytest.raises(
        PcRngTrajectoryError,
        match="rng_trajectory_mtrand_transition_unreachable",
    ):
        mtrand_outputs_between(frames[0], unreachable, maximum_calls=8)
