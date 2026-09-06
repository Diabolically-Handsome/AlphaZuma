"""Validate post-video, v2 retail memory-trajectory follow-up tasks.

PC Golden collection records two clean visible replays.  A separate third
replay is needed for frozen, single-step Board/Ball/Bullet evidence because the
memory collector intentionally perturbs timing.  This module turns that third
replay into a content-addressed task with no feature labels and no guessed
score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError

PLAN_SCHEMA = "zuma-rl.pc-memory-followup-plan"
PLAN_VERSION = 1
REPORT_SCHEMA = "zuma-rl.pc-memory-followup-verification"
REPORT_VERSION = 1
COLLECTION_PLAN_SCHEMA = "zuma-rl.pc-golden-v4-collection-plan"
COLLECTION_PLAN_VERSION = 5
COLLECTION_RESULT_SCHEMA = "zuma-rl.pc-golden-v4-collection-result"
COLLECTION_RESULT_VERSION = 5
MAX_PLAN_BYTES = 1024 * 1024
MAX_TASKS = 64
MAX_TRAJECTORY_TICKS = 2048
MIN_REPAINT_TO_SLOWDOWN_TICKS = 35
MIN_SLOWDOWN_TO_PROBE_TICKS = 5

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class PcMemoryFollowupError(ValueError):
    """A follow-up plan or one of its prepared source sessions is invalid."""


def _fail(message: str) -> None:
    raise PcMemoryFollowupError(message)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json_value(path: Path) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                _fail("JSON contains a duplicate key")
            value[key] = item
        return value

    def constant(value: str) -> None:
        _fail(f"JSON contains a non-finite constant: {value}")

    try:
        if path.stat().st_size > MAX_PLAN_BYTES:
            _fail("JSON input is too large")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except PcMemoryFollowupError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PcMemoryFollowupError("JSON input is invalid") from error
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, Mapping):
        _fail("JSON root must be an object")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        _fail(f"{name} is not a valid identifier")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{name} is not a SHA-256 digest")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer of at least {minimum}")
    return value


def _relative(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{name} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        _fail(f"{name} must be a normalized relative POSIX path")
    return path


def _existing_directory(
    root: Path,
    relative: PurePosixPath,
    name: str,
) -> Path:
    try:
        path = root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcMemoryFollowupError(
            f"{name} is missing or escapes evidence_root"
        ) from error
    if not path.is_dir():
        _fail(f"{name} must identify a directory")
    return path


def _new_output_path(
    root: Path,
    relative: PurePosixPath,
    name: str,
) -> Path:
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PcMemoryFollowupError(
            f"{name} escapes evidence_root"
        ) from error
    if not path.parent.is_dir():
        _fail(f"{name} parent directory is missing")
    return path


@dataclass(frozen=True, slots=True)
class _Task:
    id: str
    session_path: PurePosixPath
    session_plan_sha256: str
    output_path: PurePosixPath
    slowdown_update: int
    probe_update: int
    trajectory_end_update: int
    maximum_attempts: int


@dataclass(frozen=True, slots=True)
class PcMemoryFollowupReport:
    status: str
    tasks: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "PASS" else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "status": self.status,
            "tasks": [dict(row) for row in self.tasks],
            "reasons": list(self.reasons),
            "summary": dict(self.summary),
        }


def _load_tasks(value: Any) -> tuple[_Task, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or len(value) > MAX_TASKS
    ):
        _fail("follow-up tasks must be a non-empty bounded array")
    result: list[_Task] = []
    expected = {
        "id",
        "session_path",
        "session_plan_sha256",
        "output_path",
        "slowdown_update",
        "probe_update",
        "trajectory_end_update",
        "maximum_attempts",
    }
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            _fail(f"follow-up task {index} fields are invalid")
        name = f"tasks[{index}]"
        slowdown = _integer(raw["slowdown_update"], f"{name}.slowdown_update")
        probe = _integer(raw["probe_update"], f"{name}.probe_update")
        end = _integer(
            raw["trajectory_end_update"],
            f"{name}.trajectory_end_update",
        )
        attempts = _integer(
            raw["maximum_attempts"],
            f"{name}.maximum_attempts",
            minimum=1,
        )
        if not slowdown < probe <= end:
            _fail(f"{name} timing is invalid")
        if end - probe + 1 > MAX_TRAJECTORY_TICKS:
            _fail(f"{name} trajectory exceeds the collector limit")
        result.append(
            _Task(
                id=_identifier(raw["id"], f"{name}.id"),
                session_path=_relative(
                    raw["session_path"],
                    f"{name}.session_path",
                ),
                session_plan_sha256=_sha256(
                    raw["session_plan_sha256"],
                    f"{name}.session_plan_sha256",
                ),
                output_path=_relative(
                    raw["output_path"],
                    f"{name}.output_path",
                ),
                slowdown_update=slowdown,
                probe_update=probe,
                trajectory_end_update=end,
                maximum_attempts=attempts,
            )
        )
    ids = [task.id for task in result]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        _fail("follow-up task IDs must be sorted and unique")
    sessions = [task.session_path for task in result]
    outputs = [task.output_path for task in result]
    if len(sessions) != len(set(sessions)):
        _fail("follow-up source sessions must be unique")
    if len(outputs) != len(set(outputs)):
        _fail("follow-up output paths must be unique")
    return tuple(result)


def _verify_collection_result(
    session: Path,
    *,
    session_nonce: str,
) -> str:
    result_path = session / "collection.json"
    if not result_path.is_file():
        return "waiting_for_pc_golden_collection"
    result = _read_json(result_path)
    if (
        result.get("schema") != COLLECTION_RESULT_SCHEMA
        or result.get("version") != COLLECTION_RESULT_VERSION
        or result.get("status") != "complete"
        or result.get("session_nonce") != session_nonce
    ):
        _fail("PC Golden collection result is invalid")
    return "ready_for_memory_collection"


def _successful_probe(output: Path) -> Path | None:
    if not output.is_dir():
        return None
    attempts_path = output / "attempts.json"
    if not attempts_path.is_file():
        _fail("existing follow-up output is incomplete")
    rows = _read_json_value(attempts_path)
    if not isinstance(rows, list):
        _fail("existing follow-up attempts report is invalid")
    passing = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("status") == "PASS"
    ]
    if len(passing) != 1:
        _fail("existing follow-up output has no unique successful attempt")
    attempt = _integer(passing[0].get("attempt"), "successful attempt", minimum=1)
    probe = output / f"attempt-{attempt:02d}" / "memory-probe.json"
    if (
        not probe.is_file()
        or passing[0].get("probe_sha256") != _sha256_path(probe)
    ):
        _fail("existing follow-up probe identity differs")
    return probe


def _verify_task(
    task: _Task,
    *,
    root: Path,
) -> dict[str, Any]:
    session = _existing_directory(
        root,
        task.session_path,
        f"task {task.id} session",
    )
    plan_path = session / "plan.json"
    if (
        not plan_path.is_file()
        or _sha256_path(plan_path) != task.session_plan_sha256
    ):
        _fail(f"task {task.id} session plan SHA-256 differs")
    plan = _read_json(plan_path)
    if (
        plan.get("schema") != COLLECTION_PLAN_SCHEMA
        or plan.get("version") != COLLECTION_PLAN_VERSION
    ):
        _fail(f"task {task.id} collection plan schema/version is invalid")
    session_nonce = plan.get("session_nonce")
    if not isinstance(session_nonce, str) or not session_nonce:
        _fail(f"task {task.id} session nonce is invalid")

    dmo = plan.get("dmo")
    trace = plan.get("trace")
    capture = plan.get("capture")
    if not all(isinstance(row, Mapping) for row in (dmo, trace, capture)):
        _fail(f"task {task.id} collection plan sections are missing")
    guard = capture.get("window_repaint_guard")
    if not isinstance(guard, Mapping):
        _fail(f"task {task.id} repaint guard is missing")
    dmo_relative = _relative(dmo.get("artifact"), "collection dmo artifact")
    try:
        dmo_path = session.joinpath(*dmo_relative.parts).resolve(strict=True)
        dmo_path.relative_to(session)
    except (OSError, ValueError) as error:
        raise PcMemoryFollowupError(
            f"task {task.id} DMO is missing or escapes its session"
        ) from error
    if (
        dmo.get("bytes") != dmo_path.stat().st_size
        or dmo.get("sha256") != _sha256_path(dmo_path)
    ):
        _fail(f"task {task.id} DMO identity differs")
    try:
        demo = PopCapDemo.read(dmo_path)
    except (OSError, UnicodeError, PopCapDemoError) as error:
        raise PcMemoryFollowupError(
            f"task {task.id} DMO cannot be parsed"
        ) from error
    if (
        demo.random_seed != dmo.get("random_seed")
        or demo.length_updates != dmo.get("length_updates")
    ):
        _fail(f"task {task.id} DMO semantics differ")

    repaint = _integer(
        capture.get("window_repaint_update"),
        "window_repaint_update",
    )
    previous = _integer(
        guard.get("previous_effectful_command_update"),
        "previous_effectful_command_update",
    )
    following = _integer(
        guard.get("next_effectful_command_update"),
        "next_effectful_command_update",
    )
    detach = _integer(trace.get("detach_at_update"), "detach_at_update")
    reattach = _integer(
        trace.get("reattach_at_update"),
        "reattach_at_update",
    )
    effectful = sorted(
        {
            command.update
            for command in demo.commands
            if command.kind != "idle"
        }
    )
    actual_previous = max(
        (update for update in effectful if update < repaint),
        default=None,
    )
    actual_following = min(
        (update for update in effectful if update >= repaint),
        default=None,
    )
    if actual_previous != previous or actual_following != following:
        _fail(f"task {task.id} effectful-command anchors differ")
    legacy_idle_window_timing = (
        detach
        < previous
        < repaint
        < task.slowdown_update
        < task.probe_update
        <= task.trajectory_end_update
        < reattach
        < following
        < demo.length_updates
    )
    source_anchor = trace.get("source_bound_board_anchor")
    source_restore = trace.get(
        "source_bound_board_precall_global_restore"
    )
    source_bound_common = (
        trace.get("startup_trace_handoff") is True
        and isinstance(source_anchor, Mapping)
        and isinstance(source_restore, Mapping)
        and isinstance(source_anchor.get("framework_update"), int)
        and not isinstance(source_anchor.get("framework_update"), bool)
        and source_restore.get("framework_update")
        == source_anchor.get("framework_update")
        and source_restore.get(
            "suspend_other_threads_until_board_call"
        )
        is True
    )
    source_bound_blackout_timing = (
        source_bound_common
        and source_anchor.get("framework_update") == previous
        and previous
        < detach
        < repaint
        < task.slowdown_update
        < task.probe_update
        <= task.trajectory_end_update
        < reattach
        < demo.length_updates
        and repaint < following < demo.length_updates
    )
    source_bound_post_input_blackout_timing = (
        source_bound_common
        and source_anchor.get("framework_update")
        < detach
        < previous
        < repaint
        < task.slowdown_update
        < task.probe_update
        <= task.trajectory_end_update
        < reattach
        < demo.length_updates
        and repaint < following < demo.length_updates
    )
    if not (
        legacy_idle_window_timing
        or source_bound_blackout_timing
        or source_bound_post_input_blackout_timing
    ):
        _fail(f"task {task.id} replay/memory timing order is invalid")
    timing_mode = (
        "legacy_idle_window"
        if legacy_idle_window_timing
        else (
            "source_bound_blackout"
            if source_bound_blackout_timing
            else "source_bound_post_input_blackout"
        )
    )
    if (
        task.slowdown_update - repaint
        < MIN_REPAINT_TO_SLOWDOWN_TICKS
    ):
        _fail(f"task {task.id} lacks repaint-handshake headroom")
    if (
        task.probe_update - task.slowdown_update
        < MIN_SLOWDOWN_TO_PROBE_TICKS
    ):
        _fail(f"task {task.id} lacks slowdown-to-freeze headroom")

    prestate = session / "protocol" / "pre-template.json"
    host_restore = session / "safety" / "host-pre.json"
    if not prestate.is_file() or not host_restore.is_file():
        _fail(f"task {task.id} protocol snapshots are missing")
    output = _new_output_path(
        root,
        task.output_path,
        f"task {task.id} output",
    )
    existing_probe = _successful_probe(output) if output.exists() else None
    prerequisite = _verify_collection_result(
        session,
        session_nonce=session_nonce,
    )
    state = (
        "memory_collection_complete"
        if existing_probe is not None
        else prerequisite
    )
    arguments = [
        "--plan",
        str(plan_path),
        "--prestate",
        str(prestate),
        "--host-restore",
        str(host_restore),
        "--output-root",
        str(output),
        "--probe-update",
        str(task.probe_update),
        "--slowdown-update",
        str(task.slowdown_update),
        "--discover-score",
        "--maximum-attempts",
        str(task.maximum_attempts),
        "--trajectory-end-update",
        str(task.trajectory_end_update),
        "--trajectory-mode",
        "full",
    ]
    return {
        "id": task.id,
        "status": state,
        "session_path": task.session_path.as_posix(),
        "session_plan_sha256": task.session_plan_sha256,
        "output_path": task.output_path.as_posix(),
        "random_seed": demo.random_seed,
        "dmo_sha256": dmo["sha256"],
        "repaint_update": repaint,
        "timing_mode": timing_mode,
        "slowdown_update": task.slowdown_update,
        "probe_update": task.probe_update,
        "trajectory_end_update": task.trajectory_end_update,
        "trajectory_tick_count": (
            task.trajectory_end_update - task.probe_update + 1
        ),
        "next_effectful_command_update": following,
        "collection_prerequisite": prerequisite,
        "existing_probe": (
            None
            if existing_probe is None
            else str(existing_probe)
        ),
        "collector_arguments": arguments,
    }


def verify_pc_memory_followup_plan(
    plan_path: str | Path,
    *,
    evidence_root: str | Path,
) -> PcMemoryFollowupReport:
    source = Path(plan_path)
    try:
        plan = _read_json(source)
        if (
            set(plan) != {"schema", "version", "tasks"}
            or plan.get("schema") != PLAN_SCHEMA
            or plan.get("version") != PLAN_VERSION
        ):
            _fail("follow-up plan schema/version is invalid")
        root = Path(evidence_root).resolve(strict=True)
        if not root.is_dir():
            _fail("evidence_root must identify a directory")
        tasks = _load_tasks(plan.get("tasks"))
        rows = tuple(_verify_task(task, root=root) for task in tasks)
        return PcMemoryFollowupReport(
            status="PASS",
            tasks=rows,
            summary={
                "task_count": len(rows),
                "waiting_for_pc_golden_count": sum(
                    row["status"] == "waiting_for_pc_golden_collection"
                    for row in rows
                ),
                "ready_for_memory_collection_count": sum(
                    row["status"] == "ready_for_memory_collection"
                    for row in rows
                ),
                "memory_collection_complete_count": sum(
                    row["status"] == "memory_collection_complete"
                    for row in rows
                ),
                "feature_claims_allowed": False,
                "score_mode": "active_board_read_only",
                "trajectory_schema_version": 2,
            },
        )
    except (
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        PcMemoryFollowupError,
        ValueError,
    ) as error:
        return PcMemoryFollowupReport(
            status="FAIL",
            reasons=(str(error) or type(error).__name__,),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify content-addressed, post-PC-Golden memory trajectory tasks "
            "without launching the original game."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_pc_memory_followup_plan(
        args.plan,
        evidence_root=args.evidence_root,
    )
    print(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
        )
    )
    return report.exit_code


__all__ = [
    "PLAN_SCHEMA",
    "PLAN_VERSION",
    "PcMemoryFollowupError",
    "PcMemoryFollowupReport",
    "build_parser",
    "main",
    "verify_pc_memory_followup_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
