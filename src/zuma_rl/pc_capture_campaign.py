"""Validate and materialize a fail-closed PC Golden capture campaign.

This planner is deliberately read-only.  It proves that every ready task binds
an immutable DMO and that its repaint/capture interval sits inside a genuine
effect-free command gap.  It does not claim that a planned window contains a
mechanism until the resulting pixels and gameplay-memory evidence are
classified by their independent verifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from zuma_rl.fidelity_gate import GATE_POLICIES
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError

CAMPAIGN_SCHEMA = "zuma-rl.pc-capture-campaign"
CAMPAIGN_VERSION = 1
REPORT_SCHEMA = "zuma-rl.pc-capture-campaign-verification"
REPORT_VERSION = 1
MAX_CAMPAIGN_BYTES = 1024 * 1024
MAX_SOURCE_COUNT = 32
MAX_TASK_COUNT = 64

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TASK_STATUSES = frozenset(
    {"existing", "ready_for_collection", "requires_recording"}
)
_SOURCE_CLASSIFICATIONS = frozenset(
    {"accepted_existing", "certifying_candidate", "diagnostic_only"}
)


class PcCaptureCampaignError(ValueError):
    """The campaign or one of its immutable inputs is invalid."""


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        if path.stat().st_size > MAX_CAMPAIGN_BYTES:
            raise PcCaptureCampaignError("campaign file is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except PcCaptureCampaignError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PcCaptureCampaignError("campaign JSON is invalid") from error
    if not isinstance(value, Mapping):
        raise PcCaptureCampaignError("campaign root must be an object")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PcCaptureCampaignError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise PcCaptureCampaignError(f"{name} must be finite and positive")
    return float(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise PcCaptureCampaignError(f"{name} is not a valid identifier")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PcCaptureCampaignError(f"{name} is not a SHA-256 digest")
    return value


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PcCaptureCampaignError(f"{name} must be a relative path")
    result = PurePosixPath(value)
    if (
        result.is_absolute()
        or not result.parts
        or any(part in {"", ".", ".."} for part in result.parts)
    ):
        raise PcCaptureCampaignError(
            f"{name} must be a normalized relative POSIX path"
        )
    return result


def _resolve(root: Path, relative: PurePosixPath, name: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcCaptureCampaignError(
            f"{name} does not exist or escapes evidence_root"
        ) from error
    if not resolved.is_file():
        raise PcCaptureCampaignError(f"{name} must identify a file")
    return resolved


def _resolve_directory(
    root: Path,
    relative: PurePosixPath,
    name: str,
) -> Path:
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcCaptureCampaignError(
            f"{name} does not exist or escapes evidence_root"
        ) from error
    if not resolved.is_dir():
        raise PcCaptureCampaignError(f"{name} must identify a directory")
    return resolved


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _feature_list(
    value: Any,
    *,
    allowed: frozenset[str],
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PcCaptureCampaignError(f"{name} must be an array")
    result = tuple(value)
    if (
        not result
        or any(not isinstance(item, str) for item in result)
        or list(result) != sorted(set(result))
        or not set(result).issubset(allowed)
    ):
        raise PcCaptureCampaignError(
            f"{name} must be sorted, unique policy features"
        )
    return result


@dataclass(frozen=True, slots=True)
class _Source:
    id: str
    path: PurePosixPath
    sha256: str
    random_seed: int
    length_updates: int
    classification: str
    exclusion_reason: str | None
    prestate_path: PurePosixPath | None
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class _Window:
    attach_at_update: int
    detach_at_update: int
    repaint_update: int
    wait_until_update: int
    reattach_at_update: int
    previous_effectful_update: int
    next_effectful_update: int
    duration_seconds: float
    frame_budget_fps: int
    minimum_required_ticks: int


@dataclass(frozen=True, slots=True)
class _Task:
    id: str
    status: str
    candidate_features: tuple[str, ...]
    source_id: str | None
    window: _Window | None
    blocker: str | None


@dataclass(frozen=True, slots=True)
class PcCaptureCampaignReport:
    status: str
    policy: str | None
    sources: tuple[Mapping[str, Any], ...] = ()
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
            "policy": self.policy,
            "sources": [dict(item) for item in self.sources],
            "tasks": [dict(item) for item in self.tasks],
            "reasons": list(self.reasons),
            "summary": dict(self.summary),
        }


def _load_sources(
    raw_sources: Any,
    *,
    evidence_root: Path,
) -> tuple[_Source, ...]:
    if (
        isinstance(raw_sources, (str, bytes))
        or not isinstance(raw_sources, Sequence)
        or len(raw_sources) > MAX_SOURCE_COUNT
    ):
        raise PcCaptureCampaignError("campaign.sources is invalid")
    result: list[_Source] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise PcCaptureCampaignError("campaign source must be an object")
        expected = {
            "id",
            "path",
            "sha256",
            "random_seed",
            "length_updates",
            "classification",
            "exclusion_reason",
        }
        if set(raw) not in (expected, expected | {"prestate_path"}):
            raise PcCaptureCampaignError(
                f"campaign.sources[{index}] fields are invalid"
            )
        classification = raw["classification"]
        if classification not in _SOURCE_CLASSIFICATIONS:
            raise PcCaptureCampaignError("source classification is invalid")
        exclusion_reason = raw["exclusion_reason"]
        if classification == "diagnostic_only":
            if not isinstance(exclusion_reason, str) or not exclusion_reason:
                raise PcCaptureCampaignError(
                    "diagnostic source requires an exclusion reason"
                )
        elif exclusion_reason is not None:
            raise PcCaptureCampaignError(
                "certifying source may not declare an exclusion reason"
            )
        relative = _relative_path(
            raw["path"],
            f"campaign.sources[{index}].path",
        )
        resolved = _resolve(
            evidence_root,
            relative,
            f"campaign.sources[{index}].path",
        )
        expected_sha256 = _sha256(
            raw["sha256"],
            f"campaign.sources[{index}].sha256",
        )
        if _sha256_path(resolved) != expected_sha256:
            raise PcCaptureCampaignError("campaign source SHA-256 differs")
        try:
            demo = PopCapDemo.read(resolved)
        except (OSError, UnicodeError, PopCapDemoError) as error:
            raise PcCaptureCampaignError(
                "campaign source DMO is invalid"
            ) from error
        random_seed = _strict_int(
            raw["random_seed"],
            f"campaign.sources[{index}].random_seed",
        )
        length_updates = _strict_int(
            raw["length_updates"],
            f"campaign.sources[{index}].length_updates",
            minimum=1,
        )
        if (
            demo.random_seed != random_seed
            or demo.length_updates != length_updates
        ):
            raise PcCaptureCampaignError(
                "campaign source DMO metadata differs"
            )
        prestate_path = None
        if "prestate_path" in raw:
            prestate_path = _relative_path(
                raw["prestate_path"],
                f"campaign.sources[{index}].prestate_path",
            )
            _validate_prestate_directory(
                _resolve_directory(
                    evidence_root,
                    prestate_path,
                    f"campaign.sources[{index}].prestate_path",
                ),
                f"campaign.sources[{index}].prestate_path",
            )
        result.append(
            _Source(
                id=_identifier(
                    raw["id"],
                    f"campaign.sources[{index}].id",
                ),
                path=relative,
                sha256=expected_sha256,
                random_seed=random_seed,
                length_updates=length_updates,
                classification=classification,
                exclusion_reason=exclusion_reason,
                prestate_path=prestate_path,
                resolved_path=resolved,
            )
        )
    ids = [item.id for item in result]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise PcCaptureCampaignError(
            "campaign sources must have sorted unique ids"
        )
    return tuple(result)


def _validate_prestate_directory(path: Path, name: str) -> None:
    required = {
        "users.dat",
        "user2.dat",
        "adv_in_game2.sav",
    }
    if not required.issubset(
        child.name for child in path.iterdir() if child.is_file()
    ):
        raise PcCaptureCampaignError(
            f"{name} is missing required user files"
        )


def _window(raw: Any, *, task_name: str) -> _Window:
    if not isinstance(raw, Mapping):
        raise PcCaptureCampaignError(f"{task_name}.window must be an object")
    expected = {
        "attach_at_update",
        "detach_at_update",
        "repaint_update",
        "wait_until_update",
        "reattach_at_update",
        "previous_effectful_update",
        "next_effectful_update",
        "duration_seconds",
        "frame_budget_fps",
        "minimum_required_ticks",
    }
    if set(raw) != expected:
        raise PcCaptureCampaignError(f"{task_name}.window fields are invalid")
    return _Window(
        attach_at_update=_strict_int(
            raw["attach_at_update"],
            f"{task_name}.window.attach_at_update",
        ),
        detach_at_update=_strict_int(
            raw["detach_at_update"],
            f"{task_name}.window.detach_at_update",
        ),
        repaint_update=_strict_int(
            raw["repaint_update"],
            f"{task_name}.window.repaint_update",
        ),
        wait_until_update=_strict_int(
            raw["wait_until_update"],
            f"{task_name}.window.wait_until_update",
        ),
        reattach_at_update=_strict_int(
            raw["reattach_at_update"],
            f"{task_name}.window.reattach_at_update",
        ),
        previous_effectful_update=_strict_int(
            raw["previous_effectful_update"],
            f"{task_name}.window.previous_effectful_update",
        ),
        next_effectful_update=_strict_int(
            raw["next_effectful_update"],
            f"{task_name}.window.next_effectful_update",
        ),
        duration_seconds=_positive_float(
            raw["duration_seconds"],
            f"{task_name}.window.duration_seconds",
        ),
        frame_budget_fps=_strict_int(
            raw["frame_budget_fps"],
            f"{task_name}.window.frame_budget_fps",
            minimum=1,
        ),
        minimum_required_ticks=_strict_int(
            raw["minimum_required_ticks"],
            f"{task_name}.window.minimum_required_ticks",
        ),
    )


def _load_tasks(
    raw_tasks: Any,
    *,
    policy: str,
) -> tuple[_Task, ...]:
    if (
        isinstance(raw_tasks, (str, bytes))
        or not isinstance(raw_tasks, Sequence)
        or not raw_tasks
        or len(raw_tasks) > MAX_TASK_COUNT
    ):
        raise PcCaptureCampaignError("campaign.tasks is invalid")
    allowed = frozenset(
        requirement.feature
        for requirement in GATE_POLICIES[policy].requirements
    )
    result: list[_Task] = []
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise PcCaptureCampaignError("campaign task must be an object")
        expected = {
            "id",
            "status",
            "candidate_features",
            "source_id",
            "window",
            "blocker",
        }
        if set(raw) != expected:
            raise PcCaptureCampaignError(
                f"campaign.tasks[{index}] fields are invalid"
            )
        name = f"campaign.tasks[{index}]"
        status = raw["status"]
        if status not in _TASK_STATUSES:
            raise PcCaptureCampaignError(f"{name}.status is invalid")
        source_id = raw["source_id"]
        blocker = raw["blocker"]
        raw_window = raw["window"]
        if status == "ready_for_collection":
            if not isinstance(source_id, str) or not source_id:
                raise PcCaptureCampaignError(
                    f"{name} ready task requires source_id"
                )
            if blocker is not None:
                raise PcCaptureCampaignError(
                    f"{name} ready task may not have a blocker"
                )
            window = _window(raw_window, task_name=name)
        elif status == "requires_recording":
            if source_id is not None or raw_window is not None:
                raise PcCaptureCampaignError(
                    f"{name} recording task may not bind a source/window"
                )
            if not isinstance(blocker, str) or not blocker:
                raise PcCaptureCampaignError(
                    f"{name} recording task requires a blocker"
                )
            window = None
        else:
            if (
                not isinstance(source_id, str)
                or not source_id
                or raw_window is not None
                or blocker is not None
            ):
                raise PcCaptureCampaignError(
                    f"{name} existing task fields are invalid"
                )
            window = None
        result.append(
            _Task(
                id=_identifier(raw["id"], f"{name}.id"),
                status=status,
                candidate_features=_feature_list(
                    raw["candidate_features"],
                    allowed=allowed,
                    name=f"{name}.candidate_features",
                ),
                source_id=source_id,
                window=window,
                blocker=blocker,
            )
        )
    ids = [item.id for item in result]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise PcCaptureCampaignError(
            "campaign tasks must have sorted unique ids"
        )
    return tuple(result)


def _verify_window(
    source: _Source,
    window: _Window,
) -> tuple[int, int]:
    try:
        demo = PopCapDemo.read(source.resolved_path)
    except (OSError, UnicodeError, PopCapDemoError) as error:
        raise PcCaptureCampaignError("campaign source DMO is invalid") from error
    effectful_updates = sorted(
        {
            command.update
            for command in demo.commands
            if command.kind != "idle"
        }
    )
    previous = [
        update
        for update in effectful_updates
        if update < window.repaint_update
    ]
    following = [
        update
        for update in effectful_updates
        if update >= window.repaint_update
    ]
    if not previous or not following:
        raise PcCaptureCampaignError("capture repaint is outside DMO activity")
    actual_previous = previous[-1]
    actual_next = following[0]
    if (
        actual_previous != window.previous_effectful_update
        or actual_next != window.next_effectful_update
    ):
        raise PcCaptureCampaignError(
            "capture window effectful-command anchors differ"
        )
    if not (
        0
        <= window.attach_at_update
        < window.detach_at_update
        < window.repaint_update
        <= window.wait_until_update
        < window.reattach_at_update
        < actual_next
        < source.length_updates
        and actual_previous < window.repaint_update
    ):
        raise PcCaptureCampaignError(
            "capture trace/repaint ordering is invalid"
        )
    idle_budget = actual_next - window.wait_until_update
    if idle_budget < window.minimum_required_ticks:
        raise PcCaptureCampaignError(
            "capture idle budget is below required ticks"
        )
    idle_commands = sum(
        command.kind == "idle"
        and actual_previous < command.update < actual_next
        for command in demo.commands
    )
    return idle_budget, idle_commands


def verify_pc_capture_campaign(
    campaign_path: str | Path,
    *,
    evidence_root: str | Path,
) -> PcCaptureCampaignReport:
    source = Path(campaign_path)
    policy: str | None = None
    try:
        campaign = _read_json(source)
        expected = {
            "schema",
            "version",
            "policy",
            "prestate_path",
            "direct_runtime_path",
            "direct_runtime_sha256",
            "sources",
            "tasks",
        }
        if (
            set(campaign) != expected
            or campaign.get("schema") != CAMPAIGN_SCHEMA
            or campaign.get("version") != CAMPAIGN_VERSION
        ):
            raise PcCaptureCampaignError("campaign schema/version is invalid")
        policy = _identifier(campaign["policy"], "campaign.policy")
        if policy not in GATE_POLICIES:
            raise PcCaptureCampaignError("campaign policy is unsupported")
        root = Path(evidence_root).resolve(strict=True)
        if not root.is_dir():
            raise PcCaptureCampaignError(
                "evidence_root must identify a directory"
            )
        prestate_relative = _relative_path(
            campaign["prestate_path"],
            "campaign.prestate_path",
        )
        prestate = _resolve_directory(
            root,
            prestate_relative,
            "campaign.prestate_path",
        )
        _validate_prestate_directory(prestate, "campaign.prestate_path")
        runtime_relative = _relative_path(
            campaign["direct_runtime_path"],
            "campaign.direct_runtime_path",
        )
        runtime = _resolve(
            root,
            runtime_relative,
            "campaign.direct_runtime_path",
        )
        runtime_sha256 = _sha256(
            campaign["direct_runtime_sha256"],
            "campaign.direct_runtime_sha256",
        )
        if _sha256_path(runtime) != runtime_sha256:
            raise PcCaptureCampaignError(
                "campaign direct runtime SHA-256 differs"
            )
        sources = _load_sources(
            campaign["sources"],
            evidence_root=root,
        )
        source_by_id = {item.id: item for item in sources}
        tasks = _load_tasks(campaign["tasks"], policy=policy)

        source_rows = tuple(
            {
                "id": item.id,
                "path": item.path.as_posix(),
                "sha256": item.sha256,
                "random_seed": item.random_seed,
                "length_updates": item.length_updates,
                "classification": item.classification,
                "exclusion_reason": item.exclusion_reason,
                "prestate_path": (
                    None
                    if item.prestate_path is None
                    else item.prestate_path.as_posix()
                ),
                "status": "PASS",
            }
            for item in sources
        )
        task_rows: list[dict[str, Any]] = []
        for task in tasks:
            if task.source_id is not None and task.source_id not in source_by_id:
                raise PcCaptureCampaignError(
                    f"task {task.id!r} references an unknown source"
                )
            bound_source = (
                None
                if task.source_id is None
                else source_by_id[task.source_id]
            )
            row: dict[str, Any] = {
                "id": task.id,
                "status": task.status,
                "candidate_features": list(task.candidate_features),
                "source_id": task.source_id,
                "blocker": task.blocker,
            }
            if task.status == "ready_for_collection":
                assert bound_source is not None
                assert task.window is not None
                if bound_source.classification == "diagnostic_only":
                    raise PcCaptureCampaignError(
                        f"task {task.id!r} uses a diagnostic-only DMO"
                    )
                idle_budget, idle_commands = _verify_window(
                    bound_source,
                    task.window,
                )
                task_prestate = (
                    prestate_relative
                    if bound_source.prestate_path is None
                    else bound_source.prestate_path
                )
                row.update(
                    {
                        "idle_update_budget": idle_budget,
                        "guarded_idle_command_count": idle_commands,
                        "collector_prepare_options": {
                            "dmo": bound_source.path.as_posix(),
                            "prestate_dir": task_prestate.as_posix(),
                            "direct_runtime_executable": (
                                runtime_relative.as_posix()
                            ),
                            "crt_rand_seed": bound_source.random_seed,
                            "board_seed": 1162643045,
                            "global_rng_seed": bound_source.random_seed,
                            "thread_crt_rng_seed": bound_source.random_seed,
                            "duration_seconds": (
                                task.window.duration_seconds
                            ),
                            "frame_budget_fps": (
                                task.window.frame_budget_fps
                            ),
                            "wait_until_framework_update": (
                                task.window.wait_until_update
                            ),
                            "window_repaint_update": (
                                task.window.repaint_update
                            ),
                            "attach_at_update": (
                                task.window.attach_at_update
                            ),
                            "allow_pre_stream_commands": (
                                task.window.attach_at_update == 0
                            ),
                            "detach_at_update": (
                                task.window.detach_at_update
                            ),
                            "reattach_at_update": (
                                task.window.reattach_at_update
                            ),
                        },
                    }
                )
            elif task.status == "existing":
                assert bound_source is not None
                if bound_source.classification != "accepted_existing":
                    raise PcCaptureCampaignError(
                        f"task {task.id!r} is not bound to accepted evidence"
                    )
            task_rows.append(row)

        ready = sum(
            task.status == "ready_for_collection" for task in tasks
        )
        pending = sum(task.status == "requires_recording" for task in tasks)
        existing = sum(task.status == "existing" for task in tasks)
        return PcCaptureCampaignReport(
            status="PASS",
            policy=policy,
            sources=source_rows,
            tasks=tuple(task_rows),
            summary={
                "existing_case_count": existing,
                "ready_for_collection_count": ready,
                "requires_recording_count": pending,
                "diagnostic_only_source_count": sum(
                    item.classification == "diagnostic_only"
                    for item in sources
                ),
                "campaign_complete": pending == 0,
                "mechanism_claims_are_candidates_only": True,
            },
        )
    except (
        OSError,
        PcCaptureCampaignError,
        RecursionError,
    ) as error:
        return PcCaptureCampaignReport(
            status="FAIL",
            policy=policy,
            reasons=(str(error),),
            summary={"campaign_complete": False},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify immutable DMO identities and effect-free timing windows "
            "for a PC Golden capture campaign without launching the game."
        )
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_pc_capture_campaign(
        args.campaign,
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
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_VERSION",
    "PcCaptureCampaignError",
    "PcCaptureCampaignReport",
    "build_parser",
    "main",
    "verify_pc_capture_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
