"""Read-only acceptance verification for one PC golden case.

The verifier composes the strict manifest, trace, artifact, PopCap DMO,
calibration, and full-video readers.  It intentionally does not invoke the
simulator and never serializes decoded DMO command payloads.  Its JSON report
contains only validation outcomes, content-addressed metadata, safe header
facts, and counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.pc_calibration import PcCalibrationSidecar
from zuma_rl.pc_golden import (
    DMO_FILE_ID,
    DMO_FORMAT,
    DMO_VERSION,
    EXACT_STEP_MANIFEST_VERSION,
    LEGACY_MANIFEST_VERSION,
    MANIFEST_VERSION,
    MEASUREMENT_CHANNELS,
    TRUSTED_PC_EVIDENCE_SOURCES,
    ComparisonStatus,
    ComparisonResult,
    CoverageStatus,
    MeasurementStatus,
    PcGoldenArtifactError,
    PcGoldenManifest,
    PcGoldenTrace,
    PcGoldenValidationError,
    ReplayRunContract,
    SAVE_VOLATILE_REGISTRY_ROLES,
    SaveRunContract,
)
from zuma_rl.pc_exact_step_evidence import (
    COMPARISON_SCHEMA as EXACT_STEP_COMPARISON_SCHEMA,
    COMPARISON_VERSION as EXACT_STEP_COMPARISON_VERSION,
    PcExactStepEvidenceError,
    compare_formal_exact_step_runs,
    load_formal_exact_step_run,
    read_canonical_json as read_exact_step_canonical_json,
)
from zuma_rl.pc_evidence import PcTickMap
from zuma_rl.pc_memory_evidence import (
    MEMORY_CURVE_GEOMETRY_ARTIFACT,
    MEMORY_TRANSITION_CONTRACT_ARTIFACT,
    PcMemoryEvidenceError,
    PcMemoryImageDecoderUnavailable,
    PcMemoryTransitionContract,
    build_memory_curve_geometry_binding,
    read_canonical_report,
    validate_probe_payloads,
    validate_shot_transition,
)
from zuma_rl.pc_protocol_evidence import (
    PcDxgiCaptureMetadata,
    PcFrameworkUpdateMap,
    PcProcessEvent,
    PcProcessTimeline,
    PcSaveJournal,
    PcStateSnapshot,
)
from zuma_rl.pc_rng_trajectory import PcRngTrajectoryError
from zuma_rl.pc_render_settle import (
    PcRenderSettledUpdateMap,
    verify_render_settled_update_map,
)
from zuma_rl.original_data import OriginalDataError, OriginalGameCatalog
from zuma_rl.pc_video import (
    MAX_NORMALIZED_DECODE_BYTES,
    MAX_VIDEO_ARTIFACT_BYTES,
    PcVideoInspection,
    PcVideoError,
    VideoDecoderUnavailable,
    VideoResourceLimitError,
    decode_pc_video_rgb24_at_pts,
    inspect_pc_video,
    inspect_pc_video_artifact,
    validate_pc_video_artifact_declaration,
)
from zuma_rl.popcap_dmo import (
    DEMO_FILE_ID,
    PopCapDemo,
    PopCapDemoError,
)
from zuma_rl.retail_dmo_provenance import (
    COLLECTOR_PLAN_ARTIFACT,
    PLAYBACK_DMO_ARTIFACT,
    PROVENANCE_ARTIFACT,
    RAW_RECORDING_DMO_ARTIFACT,
    RECORDING_REPORT_ARTIFACT,
    RetailDmoProvenanceError,
    read_certifying_provenance,
)

REPORT_SCHEMA = "zuma-rl.pc-golden-verification"
REPORT_VERSION = 1
MAX_PC_ARTIFACT_COUNT = 4_096
MAX_NONVIDEO_ARTIFACT_BYTES = 512 * 1024**2
MAX_TOTAL_NONVIDEO_ARTIFACT_BYTES = 4 * 1024**3
MAX_TOTAL_VIDEO_ARTIFACT_BYTES = 2 * MAX_VIDEO_ARTIFACT_BYTES
MAX_TOTAL_NORMALIZED_VIDEO_BYTES = 2 * MAX_NORMALIZED_DECODE_BYTES
EXACT_STEP_PREREGISTRATION_SCHEMA = (
    "zuma-rl.pc-exact-step-golden-preregistration"
)
EXACT_STEP_EXECUTION_BINDING_SCHEMA = (
    "zuma-rl.pc-exact-step-golden-execution-binding"
)
EXACT_STEP_CONTROL_VERSION = 1
EXACT_STEP_HOLDOUT_CLASSIFICATION = "fresh_formal_pc_golden_holdout"


class VerificationCheckStatus(str, Enum):
    """Outcome of one verifier stage."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPARABLE = "INCOMPARABLE"
    SKIPPED = "SKIPPED"


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _safe_json(value: Any, name: str = "report value") -> Any:
    """Detach and freeze JSON-safe report data.

    This helper is intentionally independent from the golden-format internals.
    It prevents custom objects or non-finite values from reaching the report.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} keys must be strings")
        return MappingProxyType(
            {
                key: _safe_json(item, f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _safe_json(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{name} is not JSON-safe")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _plain_json(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One safe, structured acceptance check."""

    name: str
    status: VerificationCheckStatus
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "check.name"))
        if not isinstance(self.status, VerificationCheckStatus):
            try:
                status = VerificationCheckStatus(self.status)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid check status") from error
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "message",
            _nonempty(self.message, "check.message"),
        )
        if not isinstance(self.details, Mapping):
            raise ValueError("check.details must be a mapping")
        object.__setattr__(
            self,
            "details",
            _safe_json(self.details, "check.details"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": _plain_json(self.details),
        }


@dataclass(frozen=True, slots=True)
class PcGoldenVerificationReport:
    """Safe machine-readable result of accepting one PC golden case."""

    status: ComparisonStatus
    case_id: str | None
    checks: tuple[VerificationCheck, ...]
    reasons: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ComparisonStatus):
            try:
                status = ComparisonStatus(self.status)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid report status") from error
            object.__setattr__(self, "status", status)
        if self.case_id is not None:
            object.__setattr__(
                self,
                "case_id",
                _nonempty(self.case_id, "report.case_id"),
            )
        checks = tuple(self.checks)
        if any(not isinstance(check, VerificationCheck) for check in checks):
            raise ValueError(
                "report.checks must contain VerificationCheck objects"
            )
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("report check names must be unique")
        object.__setattr__(self, "checks", checks)
        reasons = tuple(
            _nonempty(reason, "report reason")
            for reason in self.reasons
        )
        if self.status is ComparisonStatus.PASS and reasons:
            raise ValueError("PASS reports must not contain reasons")
        if self.status is not ComparisonStatus.PASS and not reasons:
            raise ValueError(
                "FAIL and INCOMPARABLE reports require a reason"
            )
        object.__setattr__(self, "reasons", reasons)
        if not isinstance(self.summary, Mapping):
            raise ValueError("report.summary must be a mapping")
        object.__setattr__(
            self,
            "summary",
            _safe_json(self.summary, "report.summary"),
        )

    @property
    def exit_code(self) -> int:
        """CLI exit code: PASS=0, FAIL=1, INCOMPARABLE=2."""

        return {
            ComparisonStatus.PASS: 0,
            ComparisonStatus.FAIL: 1,
            ComparisonStatus.INCOMPARABLE: 2,
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "status": self.status.value,
            "case_id": self.case_id,
            "checks": [check.to_dict() for check in self.checks],
            "reasons": list(self.reasons),
            "summary": _plain_json(self.summary),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )


class _InputSequenceMismatch(ValueError):
    """Safe failure whose text never contains an input payload."""


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _scenario_semantics_mismatches(
    manifest: PcGoldenManifest,
    original_root: str | Path,
) -> tuple[str, ...]:
    """Resolve scenario identity against original data without leaking paths."""

    catalog = OriginalGameCatalog(original_root)
    loaded = catalog.load_level(
        manifest.scenario.level_id,
        hard=manifest.scenario.hard,
    )
    definition = loaded.definition
    environment = manifest.pc_environment
    mismatches: list[str] = []

    if definition.id != manifest.scenario.level_id:
        mismatches.append("level_id")
    if manifest.scenario.mode != "adventure":
        mismatches.append("mode")
    if not 0 <= manifest.scenario.curve_index < len(loaded.curves):
        mismatches.append("curve_index")
    positions = definition.gun.positions
    if not 0 <= manifest.scenario.gun_index < len(positions):
        mismatches.append("gun_index")
    if len(loaded.curves) != 1:
        mismatches.append("independent_curve_count")
    if len(positions) != 1:
        mismatches.append("frog_position_count")
    if definition.gun.type.casefold() != "normal":
        mismatches.append("gun_type")
    level_key = definition.id.casefold()
    if level_key.startswith(("boss", "debugboss")):
        mismatches.append("boss_state_machine")
    if (
        definition.attributes.get("ironfrog", "false").casefold()
        == "true"
    ):
        mismatches.append("iron_frog_state_machine")

    launcher_sha256 = environment.game_settings.get(
        "runtime_source_executable_sha256"
    )
    expected_source_sha256 = (
        environment.executable_sha256
        if launcher_sha256 is None
        else launcher_sha256
    )
    if (
        not isinstance(expected_source_sha256, str)
        or _sha256_path(catalog.root / "ZumasRevenge.exe")
        != expected_source_sha256
    ):
        mismatches.append(
            "executable_sha256"
            if launcher_sha256 is None
            else "runtime_source_executable_sha256"
        )
    if (
        _sha256_path(catalog.root / "main.pak")
        != environment.main_pak_sha256
    ):
        mismatches.append("main_pak_sha256")
    levels_xml = catalog.archive.read_member(r"levels\levels.xml")
    if _sha256_bytes(levels_xml) != environment.levels_xml_sha256:
        mismatches.append("levels_xml_sha256")

    actual_curve_hashes = sorted(
        _sha256_path(curve.source_path)
        for curve in loaded.curves
        if curve.source_path is not None
    )
    if (
        len(actual_curve_hashes) != len(loaded.curves)
        or actual_curve_hashes
        != sorted(environment.curve_sha256.values())
    ):
        mismatches.append("curve_sha256")
    return tuple(dict.fromkeys(mismatches))


def _failure_report(
    *,
    case_id: str | None,
    checks: list[VerificationCheck],
    stage: str,
    message: str,
    error: BaseException | None = None,
    summary: Mapping[str, Any] | None = None,
) -> PcGoldenVerificationReport:
    details: dict[str, Any] = {}
    if error is not None:
        details["error_type"] = type(error).__name__
    checks.append(
        VerificationCheck(
            name=stage,
            status=VerificationCheckStatus.FAIL,
            message=message,
            details=details,
        )
    )
    return PcGoldenVerificationReport(
        status=ComparisonStatus.FAIL,
        case_id=case_id,
        checks=tuple(checks),
        reasons=(message,),
        summary={} if summary is None else summary,
    )


def _dmo_header_mismatches(
    manifest: PcGoldenManifest,
    demo: PopCapDemo,
) -> tuple[str, ...]:
    """Return header field names only, never decoded command payloads."""

    timeline = manifest.input_timeline
    artifact = manifest.artifacts[timeline.artifact]
    checks = {
        "format": (
            timeline.format == DMO_FORMAT
            and demo.version == DMO_VERSION
        ),
        "file_id": (
            timeline.file_id
            == DMO_FILE_ID
            == DEMO_FILE_ID
        ),
        "dmo_version": timeline.dmo_version == demo.version == DMO_VERSION,
        "product_version": timeline.product_version == demo.product_version,
        "random_seed": timeline.random_seed == demo.random_seed,
        "length_updates": timeline.length_updates == demo.length_updates,
        "artifact_bytes": artifact.bytes == demo.artifact_bytes,
        "artifact_sha256": artifact.sha256 == demo.artifact_sha256,
    }
    return tuple(name for name, matches in checks.items() if not matches)


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _input_sequence_check(
    manifest: PcGoldenManifest,
    trace: PcGoldenTrace,
    demo: PopCapDemo,
) -> VerificationCheck:
    """Cross-check trace input order when explicit DMO bindings are present.

    ``GoldenInput.payload`` may bind an entry to the original global DMO
    command using integer ``dmo_sequence`` and ``dmo_update`` fields.  Without
    those fields, update-to-tick phase is intentionally unspecified, so the
    check is reported as incomparable rather than guessed from video PTS.
    """

    trace_inputs = tuple(
        (record.tick, item)
        for record in trace.records
        for item in record.inputs
    )
    trace_tick_start = trace.records[0].tick
    trace_tick_end = trace.records[-1].tick
    dmo_inputs = tuple(
        item
        for item in demo.input_commands
        if trace_tick_start
        <= (
            item.update
            + manifest.input_timeline.native_tick_offset
        )
        <= trace_tick_end
    )
    counts = {
        "trace_input_count": len(trace_inputs),
        "dmo_input_count": len(dmo_inputs),
        "dmo_total_input_count": len(demo.input_commands),
        "trace_tick_start": trace_tick_start,
        "trace_tick_end": trace_tick_end,
    }
    if not trace_inputs and not dmo_inputs:
        return VerificationCheck(
            name="input_sequence",
            status=VerificationCheckStatus.PASS,
            message="trace and DMO both contain no input events",
            details={**counts, "mode": "empty_exact"},
        )

    binding_presence = tuple(
        (
            "dmo_sequence" in item.payload,
            "dmo_update" in item.payload,
        )
        for _, item in trace_inputs
    )
    if not binding_presence or not any(
        has_sequence or has_update
        for has_sequence, has_update in binding_presence
    ):
        return VerificationCheck(
            name="input_sequence",
            status=VerificationCheckStatus.INCOMPARABLE,
            message=(
                "trace inputs do not declare explicit DMO sequence/update "
                "bindings"
            ),
            details={**counts, "mode": "unbound"},
        )
    if not all(
        has_sequence and has_update
        for has_sequence, has_update in binding_presence
    ):
        raise _InputSequenceMismatch(
            "trace DMO input bindings are partial"
        )
    if len(trace_inputs) != len(dmo_inputs):
        raise _InputSequenceMismatch(
            "trace and DMO input counts differ under explicit binding"
        )

    for index, ((native_tick, trace_input), dmo_input) in enumerate(
        zip(trace_inputs, dmo_inputs, strict=True)
    ):
        if set(trace_input.payload) != {"dmo_sequence", "dmo_update"}:
            raise _InputSequenceMismatch(
                "explicit DMO-bound trace inputs may contain only binding "
                f"fields at input index {index}"
            )
        sequence = _strict_nonnegative_int(
            trace_input.payload["dmo_sequence"]
        )
        update = _strict_nonnegative_int(
            trace_input.payload["dmo_update"]
        )
        if sequence is None or update is None:
            raise _InputSequenceMismatch(
                f"trace DMO binding types are invalid at input index {index}"
            )
        if sequence != dmo_input.sequence:
            raise _InputSequenceMismatch(
                f"DMO sequence binding differs at input index {index}"
            )
        if update != dmo_input.update:
            raise _InputSequenceMismatch(
                f"DMO update binding differs at input index {index}"
            )
        expected_native_tick = (
            dmo_input.update
            + manifest.input_timeline.native_tick_offset
        )
        if native_tick != expected_native_tick:
            raise _InputSequenceMismatch(
                "DMO update maps to a different native tick at input "
                f"index {index}"
            )
        if trace_input.kind != dmo_input.kind:
            raise _InputSequenceMismatch(
                f"DMO input kind differs at input index {index}"
            )

    return VerificationCheck(
        name="input_sequence",
        status=VerificationCheckStatus.PASS,
        message="trace input order matches explicit DMO bindings",
        details={**counts, "mode": "explicit_dmo_binding"},
    )


def _trace_coverage_readiness(
    manifest: PcGoldenManifest,
    trace: PcGoldenTrace,
) -> ComparisonResult:
    """Validate required trace evidence instead of trusting declarations."""

    reasons: list[str] = []
    total_reasons = 0

    def add(reason: str) -> None:
        nonlocal total_reasons
        total_reasons += 1
        if len(reasons) < 32:
            reasons.append(reason)

    ranges_by_channel: dict[str, list[Any]] = {}
    for item in manifest.coverage:
        ranges_by_channel.setdefault(item.channel, []).append(item)

    for channel in ("frames", "input_events"):
        ranges = ranges_by_channel.get(channel, ())
        if not ranges or not all(item.required for item in ranges):
            add(
                f"PC golden acceptance requires channel {channel!r} "
                "on every tick"
            )

    required_measurements = [
        item
        for item in manifest.coverage
        if item.required and item.channel in MEASUREMENT_CHANNELS
    ]
    complete_measurements = [
        item
        for item in manifest.coverage
        if (
            item.status is CoverageStatus.COMPLETE
            and item.channel in MEASUREMENT_CHANNELS
        )
    ]
    required_events = [
        item
        for item in manifest.coverage
        if item.required and item.channel == "events"
    ]
    has_observed_event = any(
        trace.records[tick].events
        for coverage in required_events
        for tick in range(coverage.start_tick, coverage.end_tick + 1)
    )
    if not required_measurements and not (
        required_events and has_observed_event
    ):
        add(
            "PC golden acceptance requires at least one required "
            "measurement channel or one observed required event"
        )

    for coverage in complete_measurements:
        for tick in range(coverage.start_tick, coverage.end_tick + 1):
            measurement = trace.records[tick].measurements.get(
                coverage.channel
            )
            if measurement is None:
                add(
                    f"complete measurement {coverage.channel!r} is missing "
                    f"at tick {tick}; the coverage declaration is not "
                    "supported by the trace"
                )
                continue
            allowed_statuses = {MeasurementStatus.OBSERVED}
            if not coverage.required:
                allowed_statuses.add(MeasurementStatus.INFERRED)
            if measurement.status not in allowed_statuses:
                add(
                    f"complete measurement {coverage.channel!r} is "
                    f"{measurement.status.value} at tick {tick}"
                    + (
                        "; required comparison evidence must be directly "
                        "observed"
                        if coverage.required
                        else ""
                    )
                )
            elif measurement.source not in TRUSTED_PC_EVIDENCE_SOURCES:
                add(
                    f"complete measurement {coverage.channel!r} uses an "
                    f"untrusted provenance source at tick {tick}"
                )

    complete_event_ranges = [
        item
        for item in manifest.coverage
        if (
            item.channel == "events"
            and item.status is CoverageStatus.COMPLETE
        )
    ]
    for coverage in complete_event_ranges:
        for tick in range(coverage.start_tick, coverage.end_tick + 1):
            for event in trace.records[tick].events:
                if event.source not in TRUSTED_PC_EVIDENCE_SOURCES:
                    add(
                        "complete event evidence uses an untrusted "
                        f"provenance source at tick {tick}"
                    )

    if total_reasons > len(reasons):
        reasons.append(
            f"{total_reasons - len(reasons)} additional trace coverage "
            "reasons omitted"
        )
    if reasons:
        return ComparisonResult(
            status=ComparisonStatus.INCOMPARABLE,
            reasons=tuple(reasons),
            metrics={"trace_coverage_issue_count": total_reasons},
        )
    return ComparisonResult(
        status=ComparisonStatus.PASS,
        reasons=(),
        metrics={"trace_coverage_issue_count": 0},
    )


class _ProtocolIncomparable(RuntimeError):
    """Required raw protocol evidence or a decoder is unavailable."""


@dataclass(frozen=True, slots=True)
class _SaveProtocolContext:
    snapshots: Mapping[str, PcStateSnapshot]
    process_pairs: Mapping[
        str,
        tuple[PcProcessEvent, PcProcessEvent],
    ]


_ZUMA_REGISTRY_ROOT = r"HKCU\Software\SteamPopCap\ZumasRevenge"
_VOLATILE_END_STATE_VALUES = (
    (_ZUMA_REGISTRY_ROOT, "", "LastGamePID"),
    (_ZUMA_REGISTRY_ROOT, "", "RegExData"),
)


def _projected_end_state_root(
    snapshot: PcStateSnapshot,
    *,
    run: SaveRunContract,
    pre_snapshot: PcStateSnapshot,
    launcher_registry_contract: str,
) -> str:
    nodes = [
        node
        for node in snapshot.registry_nodes
        if node.root == _ZUMA_REGISTRY_ROOT and node.subkey == ""
    ]
    if len(nodes) != 1 or not nodes[0].exists:
        raise PcGoldenValidationError(
            "replay end state lacks the Zuma launcher registry root"
        )
    values = {value.name.casefold(): value for value in nodes[0].values}
    last_pid = values.get("lastgamepid")
    registration_data = values.get("regexdata")
    if (
        last_pid is None
        or last_pid.name != "LastGamePID"
        or last_pid.value_type != 4
        or len(last_pid.data) != 4
    ):
        raise PcGoldenValidationError(
            "LastGamePID is not the expected launcher DWORD"
        )
    if (
        registration_data is None
        or registration_data.name != "RegExData"
        or registration_data.value_type != 4
        or len(registration_data.data) != 4
    ):
        raise PcGoldenValidationError(
            "RegExData is not the expected launcher DWORD"
        )
    if launcher_registry_contract == "process_bound":
        if int.from_bytes(last_pid.data, "little") != run.process_id:
            raise PcGoldenValidationError(
                "LastGamePID does not bind the replay process identity"
            )
    elif launcher_registry_contract == "preserved_from_prestate":
        pre_nodes = [
            node
            for node in pre_snapshot.registry_nodes
            if node.root == _ZUMA_REGISTRY_ROOT and node.subkey == ""
        ]
        if len(pre_nodes) != 1 or not pre_nodes[0].exists:
            raise PcGoldenValidationError(
                "pre-state lacks the Zuma launcher registry root"
            )
        pre_values = {
            value.name.casefold(): value
            for value in pre_nodes[0].values
        }
        pre_last_pid = pre_values.get("lastgamepid")
        pre_registration_data = pre_values.get("regexdata")
        if (
            pre_last_pid is None
            or pre_registration_data is None
            or (
                last_pid.value_type,
                last_pid.data,
            )
            != (
                pre_last_pid.value_type,
                pre_last_pid.data,
            )
            or (
                registration_data.value_type,
                registration_data.data,
            )
            != (
                pre_registration_data.value_type,
                pre_registration_data.data,
            )
        ):
            raise PcGoldenValidationError(
                "direct-runtime replay changed launcher registry values"
            )
    else:
        raise PcGoldenValidationError(
            "launcher registry contract is unsupported"
        )
    return snapshot.projected_state_root(
        _VOLATILE_END_STATE_VALUES
    )


def _snapshot_phase_artifacts(
    manifest: PcGoldenManifest,
) -> tuple[tuple[str, str, SaveRunContract | None], ...]:
    contract = manifest.save_transaction
    if contract is None:
        return ()
    phases: list[tuple[str, str, SaveRunContract | None]] = [
        ("pre", contract.pre_snapshot_artifact, None)
    ]
    for run in contract.runs:
        phases.extend(
            (
                (
                    f"{run.run_id}-start",
                    run.start_snapshot_artifact,
                    run,
                ),
                (
                    f"{run.run_id}-end",
                    run.end_snapshot_artifact,
                    run,
                ),
            )
        )
    phases.append(("restored", contract.restored_snapshot_artifact, None))
    return tuple(phases)


def _load_save_protocol_evidence(
    manifest: PcGoldenManifest,
    verified_artifacts: Mapping[str, Path],
) -> tuple[_SaveProtocolContext, Mapping[str, Any]]:
    contract = manifest.save_transaction
    if contract is None:
        raise _ProtocolIncomparable(
            "manifest does not declare a v4 save transaction"
        )
    phase_artifacts = _snapshot_phase_artifacts(manifest)
    snapshots: dict[str, PcStateSnapshot] = {}
    for phase, artifact_name, _ in phase_artifacts:
        snapshot = PcStateSnapshot.read(verified_artifacts[artifact_name])
        if snapshot.phase != phase:
            raise PcGoldenValidationError(
                "state snapshot phase does not match its save contract"
            )
        snapshots[phase] = snapshot

    journal = PcSaveJournal.read(
        verified_artifacts[contract.journal_artifact]
    )
    timeline = PcProcessTimeline.read(
        verified_artifacts[contract.process_timeline_artifact]
    )
    session_nonces = {
        journal.session_nonce,
        timeline.session_nonce,
        *(snapshot.session_nonce for snapshot in snapshots.values()),
    }
    if len(session_nonces) != 1:
        raise PcGoldenValidationError(
            "save evidence session nonces do not match"
        )
    if len(journal.records) != len(phase_artifacts):
        raise PcGoldenValidationError(
            "save journal does not contain exactly one record per phase"
        )

    process_pairs: dict[
        str,
        tuple[PcProcessEvent, PcProcessEvent],
    ] = {}
    for (
        expected_phase,
        expected_artifact,
        save_run,
    ), record in zip(
        phase_artifacts,
        journal.records,
        strict=True,
    ):
        snapshot = snapshots[expected_phase]
        if (
            record.phase != expected_phase
            or record.snapshot_artifact != expected_artifact
            or record.snapshot_state_root != snapshot.state_root
            or record.snapshot_perf_counter_ns
            != snapshot.captured_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "save journal phase/snapshot binding differs from raw "
                "snapshot evidence"
            )
        if save_run is None:
            if record.process_instance is not None or record.exit_code is not None:
                raise PcGoldenValidationError(
                    "pre/restored journal phases must not bind a process"
                )
            continue
        if record.process_instance != save_run.process_instance:
            raise PcGoldenValidationError(
                "save journal process instance differs from its run contract"
            )
        if expected_phase.endswith("-start"):
            if record.exit_code is not None:
                raise PcGoldenValidationError(
                    "run-start journal phase must not contain an exit code"
                )
            continue
        if record.exit_code != 0:
            raise PcGoldenValidationError(
                "run-end journal phase does not prove a normal exit"
            )

    pre = snapshots["pre"]
    restored = snapshots["restored"]
    if pre.state_root != manifest.pc_environment.pre_capture_save_sha256:
        raise PcGoldenValidationError(
            "v4 pre-state root differs from pc_environment "
            "pre_capture_save_sha256"
        )
    if restored.state_root != pre.state_root:
        raise PcGoldenValidationError(
            "restored state root differs from the pre-state root"
        )
    end_gameplay_roots: set[str] = set()
    end_full_roots: set[str] = set()
    expected_process_instances: set[tuple[int, int]] = set()
    previous_end_time: int | None = None
    launcher_registry_contract = manifest.pc_environment.game_settings.get(
        "launcher_registry_contract",
        "process_bound",
    )
    if launcher_registry_contract not in {
        "process_bound",
        "preserved_from_prestate",
    }:
        raise PcGoldenValidationError(
            "launcher registry contract is unsupported"
        )
    for run in contract.runs:
        start_snapshot = snapshots[f"{run.run_id}-start"]
        end_snapshot = snapshots[f"{run.run_id}-end"]
        if start_snapshot.state_root != pre.state_root:
            raise PcGoldenValidationError(
                "replay run did not start from the canonical pre-state root"
            )
        end_full_roots.add(end_snapshot.state_root)
        end_gameplay_roots.add(
            _projected_end_state_root(
                end_snapshot,
                run=run,
                pre_snapshot=pre,
                launcher_registry_contract=launcher_registry_contract,
            )
        )
        start_event, stop_event = timeline.event_pair(run.process_instance)
        if (
            start_event.executable_sha256
            != manifest.pc_environment.executable_sha256
            or stop_event.executable_sha256
            != manifest.pc_environment.executable_sha256
        ):
            raise PcGoldenValidationError(
                "native process executable differs from pc_environment"
            )
        if stop_event.exit_code != 0:
            raise PcGoldenValidationError(
                "native replay process did not exit normally"
            )
        if not (
            start_snapshot.captured_perf_counter_ns
            < start_event.perf_counter_ns
            < stop_event.perf_counter_ns
            < end_snapshot.captured_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "save snapshots do not bracket the native process lifetime"
            )
        if (
            previous_end_time is not None
            and start_snapshot.captured_perf_counter_ns <= previous_end_time
        ):
            raise PcGoldenValidationError(
                "save transaction replay runs overlap or are out of order"
            )
        previous_end_time = end_snapshot.captured_perf_counter_ns
        expected_process_instances.add(run.process_instance)
        process_pairs[run.run_id] = (start_event, stop_event)
    if len(end_gameplay_roots) != 1:
        raise PcGoldenValidationError(
            "independent replay projected gameplay-state roots differ"
        )
    observed_process_instances = {
        event.process_instance for event in timeline.events
    }
    if observed_process_instances != expected_process_instances:
        raise PcGoldenValidationError(
            "process timeline contains undeclared or missing process instances"
        )

    context = _SaveProtocolContext(
        snapshots=MappingProxyType(snapshots),
        process_pairs=MappingProxyType(process_pairs),
    )
    details = {
        "machine_verifiable_evidence": True,
        "session_nonce_matched": True,
        "run_count": len(contract.runs),
        "snapshot_count": len(snapshots),
        "process_event_count": len(timeline.events),
        "pre_restored_root_matched": True,
        "run_start_roots_matched": True,
        "run_end_roots_matched": True,
        "run_end_root_scope": "gameplay_state_projection_v1",
        "run_end_full_root_count": len(end_full_roots),
        "volatile_registry_roles": list(
            SAVE_VOLATILE_REGISTRY_ROLES
        ),
        "launcher_registry_contract": launcher_registry_contract,
        "last_game_pid_bound_to_process": (
            launcher_registry_contract == "process_bound"
        ),
        "launcher_registry_preserved_from_prestate": (
            launcher_registry_contract == "preserved_from_prestate"
        ),
        "full_snapshot_roots_retained": True,
        "normal_exit_count": len(contract.runs),
        "users_file_count": len(pre.files),
        "users_bytes": pre.users_bytes,
    }
    return context, details


def _normalized_replay_trace(
    trace: PcGoldenTrace,
) -> tuple[Mapping[str, Any], ...]:
    """Remove only run-specific frame indices and presentation timestamps."""

    normalized: list[Mapping[str, Any]] = []
    for record in trace.records:
        item = record.to_dict()
        item.pop("frames")
        for input_item in item["inputs"]:
            input_item.pop("pts")
        normalized.append(item)
    return tuple(normalized)


def _per_tick_pixel_hashes(
    inspection: PcVideoInspection,
    tick_map: PcTickMap,
    *,
    excluded_bottom_rows: int = 0,
) -> tuple[str, ...]:
    if excluded_bottom_rows == 0:
        frame_hashes = inspection.frame_pixel_sha256
    elif excluded_bottom_rows == 1:
        frame_hashes = inspection.frame_pixel_sha256_without_last_row
    else:
        raise PcGoldenValidationError(
            "replay pixel comparison excludes an unsupported raster edge"
        )
    if len(frame_hashes) != inspection.frame_count:
        raise _ProtocolIncomparable(
            "video decoder did not expose per-frame normalized pixel hashes"
        )
    pts_to_index = {
        pts: index for index, pts in enumerate(inspection.frame_pts)
    }
    if len(pts_to_index) != len(inspection.frame_pts):
        raise PcGoldenValidationError(
            "decoded video contains duplicate presentation timestamps"
        )
    try:
        return tuple(
            frame_hashes[pts_to_index[record.pts]]
            for record in tick_map.records
        )
    except KeyError:
        raise PcGoldenValidationError(
            "tick-map PTS does not identify a decoded video frame"
        ) from None


def _per_tick_last_row_pixels(
    inspection: PcVideoInspection,
    tick_map: PcTickMap,
) -> tuple[bytes, ...]:
    if len(inspection.frame_last_row_rgb24) != inspection.frame_count:
        raise _ProtocolIncomparable(
            "video decoder did not expose the final raster row"
        )
    pts_to_index = {
        pts: index for index, pts in enumerate(inspection.frame_pts)
    }
    if len(pts_to_index) != len(inspection.frame_pts):
        raise PcGoldenValidationError(
            "decoded video contains duplicate presentation timestamps"
        )
    try:
        return tuple(
            inspection.frame_last_row_rgb24[pts_to_index[record.pts]]
            for record in tick_map.records
        )
    except KeyError:
        raise PcGoldenValidationError(
            "tick-map PTS does not identify a decoded video frame"
        ) from None


def _exact_step_fixed_contract(manifest: PcGoldenManifest) -> dict[str, Any]:
    contract = manifest.exact_step_replay
    if contract is None:
        raise PcGoldenValidationError(
            "manifest does not declare exact-step replay evidence"
        )
    return {
        "freeze_update": contract.freeze_update,
        "source_start_update": contract.source_start_update,
        "source_end_update": contract.source_end_update,
        "warmup_tick_count": contract.warmup_tick_count,
        "retained_tick_count": (
            contract.source_end_update - contract.source_start_update + 1
        ),
        "maximum_startup_attempts": contract.maximum_startup_attempts,
        "sample_phase": contract.sample_phase,
        "pixel_comparison": contract.pixel_comparison,
        "run_order": [run.run_id for run in contract.runs],
    }


def _verify_exact_step_control_evidence(
    manifest: PcGoldenManifest,
    verified_artifacts: Mapping[str, Path],
) -> None:
    """Validate the immutable holdout freeze and pre-output binding."""

    contract = manifest.exact_step_replay
    if contract is None:
        raise PcGoldenValidationError(
            "manifest does not declare exact-step replay evidence"
        )
    expected_contract = _exact_step_fixed_contract(manifest)
    expected_absence = {run.run_id: True for run in contract.runs}
    preregistration = read_exact_step_canonical_json(
        verified_artifacts[contract.preregistration_artifact]
    )
    if not isinstance(preregistration, dict) or (
        preregistration.get("schema")
        != EXACT_STEP_PREREGISTRATION_SCHEMA
        or preregistration.get("version") != EXACT_STEP_CONTROL_VERSION
        or preregistration.get("status") != "FROZEN_BEFORE_RUN_OUTPUTS"
        or preregistration.get("evidence_classification")
        != EXACT_STEP_HOLDOUT_CLASSIFICATION
        or preregistration.get("fixed_contract") != expected_contract
        or preregistration.get("output_roots_absent_at_freeze")
        != expected_absence
        or preregistration.get("case_output_absent_at_freeze") is not True
    ):
        raise PcGoldenValidationError(
            "exact-step preregistration does not freeze the declared fresh "
            "holdout before outputs"
        )

    execution = read_exact_step_canonical_json(
        verified_artifacts[contract.execution_binding_artifact]
    )
    preregistration_sha256 = manifest.artifacts[
        contract.preregistration_artifact
    ].sha256
    if not isinstance(execution, dict) or (
        execution.get("schema") != EXACT_STEP_EXECUTION_BINDING_SCHEMA
        or execution.get("version") != EXACT_STEP_CONTROL_VERSION
        or execution.get("status") != "BOUND_BEFORE_RUN_OUTPUTS"
        or execution.get("evidence_classification")
        != EXACT_STEP_HOLDOUT_CLASSIFICATION
        or execution.get("parent_preregistration_sha256")
        != preregistration_sha256
        or execution.get("fixed_contract") != expected_contract
        or execution.get("output_roots_absent_at_binding")
        != expected_absence
        or execution.get("case_output_absent_at_binding") is not True
    ):
        raise PcGoldenValidationError(
            "exact-step execution binding does not bind the frozen contract "
            "before outputs"
        )


def _exact_step_source_integer(
    row: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PcGoldenValidationError(
            f"exact-step source field {field!r} is not a valid integer"
        )
    return value


def _validate_exact_step_trace_binding(
    manifest: PcGoldenManifest,
    *,
    trace: PcGoldenTrace,
    tick_map: PcTickMap,
    video_inspection: PcVideoInspection,
    source_run: Any,
) -> None:
    """Bind a packaged trace/video back to retained source tick rows/BMPs."""

    contract = manifest.exact_step_replay
    if contract is None:
        raise PcGoldenValidationError(
            "manifest does not declare exact-step replay evidence"
        )
    tick_count = contract.source_end_update - contract.source_start_update + 1
    if (
        len(trace.records) != tick_count
        or len(tick_map.records) != tick_count
        or video_inspection.frame_count != tick_count
        or manifest.input_timeline.native_tick_offset
        != -contract.source_start_update
    ):
        raise PcGoldenValidationError(
            "exact-step trace, tick map, video, or DMO update phase differs "
            "from the retained source range"
        )
    expected_pts = tuple(range(tick_count))
    if (
        tuple(record.pts for record in tick_map.records) != expected_pts
        or video_inspection.frame_pts != expected_pts
    ):
        raise PcGoldenValidationError(
            "exact-step video and tick map must use the frozen 100 Hz PTS grid"
        )

    expected_video_pixels = tuple(
        frame.pc_video_rgb24_sha256 for frame in source_run.visual_frames
    )
    if _per_tick_pixel_hashes(video_inspection, tick_map) != expected_video_pixels:
        raise PcGoldenValidationError(
            "decoded exact-step video pixels differ from retained source BMPs"
        )

    for tick, (record, source_row) in enumerate(
        zip(trace.records, source_run.tick_rows, strict=True)
    ):
        expected_measurements = {
            "chain_count": {
                "status": MeasurementStatus.OBSERVED.value,
                "value": _exact_step_source_integer(
                    source_row, "chain_ball_count"
                ),
                "source": "pc_memory_probe",
                "uncertainty": 0.0,
            },
            "projectile_count": {
                "status": MeasurementStatus.OBSERVED.value,
                "value": _exact_step_source_integer(
                    source_row, "fired_bullet_count"
                ),
                "source": "pc_memory_probe",
                "uncertainty": 0.0,
            },
            "score": {
                "status": MeasurementStatus.OBSERVED.value,
                "value": _exact_step_source_integer(source_row, "score"),
                "source": "pc_memory_probe",
                "uncertainty": 0.0,
            },
        }
        observed_measurements = {
            name: measurement.to_dict()
            for name, measurement in record.measurements.items()
        }
        if (
            source_row.get("framework_update")
            != contract.source_start_update + tick
            or len(record.frames) != 1
            or record.frames[0].frame_index != tick
            or record.frames[0].pts != tick
            or record.events
            or observed_measurements != expected_measurements
        ):
            raise PcGoldenValidationError(
                "exact-step packaged trace differs from its retained source "
                f"at native tick {tick}"
            )


def _verify_exact_step_replay_evidence(
    manifest: PcGoldenManifest,
    verified_artifacts: Mapping[str, Path],
    *,
    demo: PopCapDemo,
    primary_trace: PcGoldenTrace,
    primary_tick_map: PcTickMap,
    primary_video_inspection: PcVideoInspection | None,
) -> Mapping[str, Any]:
    """Independently recompute a formal exact-step holdout decision."""

    contract = manifest.exact_step_replay
    if contract is None:
        raise _ProtocolIncomparable(
            "manifest does not declare formal exact-step replay evidence"
        )
    if primary_video_inspection is None:
        raise _ProtocolIncomparable(
            "video decoder is unavailable for formal exact-step evidence"
        )
    _verify_exact_step_control_evidence(manifest, verified_artifacts)

    replay_pre_snapshot = PcStateSnapshot.read(
        verified_artifacts[contract.pre_snapshot_artifact]
    )
    host_pre_snapshot = PcStateSnapshot.read(
        verified_artifacts[contract.host_pre_snapshot_artifact]
    )
    if (
        replay_pre_snapshot.state_root
        != manifest.pc_environment.pre_capture_save_sha256
    ):
        raise PcGoldenValidationError(
            "exact-step replay prestate root differs from the environment"
        )

    declared_paths = tuple(verified_artifacts.values())
    loaded_runs: list[Any] = []
    traces: list[PcGoldenTrace] = []
    run_summaries: list[Mapping[str, Any]] = []
    post_snapshots: list[PcStateSnapshot] = []
    for index, run in enumerate(contract.runs):
        loaded = load_formal_exact_step_run(
            run_id=run.run_id,
            selected_attempt=run.selected_attempt,
            attempts_path=verified_artifacts[run.attempts_artifact],
            probe_path=verified_artifacts[run.memory_probe_artifact],
            index_path=verified_artifacts[run.trajectory_index_artifact],
            expected_process_id=run.process_id,
            expected_process_creation_filetime_100ns=(
                run.process_creation_filetime_100ns
            ),
            expected_runtime_sha256=(
                manifest.pc_environment.executable_sha256
            ),
            expected_dmo_sha256=(
                manifest.artifacts[manifest.input_timeline.artifact].sha256
            ),
            expected_freeze_update=contract.freeze_update,
            expected_start_update=contract.source_start_update,
            expected_end_update=contract.source_end_update,
            expected_warmup_tick_count=contract.warmup_tick_count,
            maximum_startup_attempts=contract.maximum_startup_attempts,
            declared_artifact_paths=declared_paths,
        )
        if index == 0:
            trace = primary_trace
            tick_map = primary_tick_map
            inspection = primary_video_inspection
        else:
            trace = PcGoldenTrace.read_ndjson(
                verified_artifacts[run.trace_artifact]
            )
            trace.validate_against_run(
                manifest,
                trace_artifact=run.trace_artifact,
                video=run.video,
                clock=run.clock,
            )
            tick_map = PcTickMap.read(
                verified_artifacts[run.clock.tick_map_artifact]
            )
            tick_map.validate_against_run(
                manifest,
                trace,
                tick_map_artifact=run.clock.tick_map_artifact,
                video=run.video,
                clock=run.clock,
            )
            inspection = inspect_pc_video_artifact(
                verified_artifacts[run.video.artifact],
                metadata=run.video,
                artifact=manifest.artifacts[run.video.artifact],
            )
        input_check = _input_sequence_check(manifest, trace, demo)
        if input_check.status is not VerificationCheckStatus.PASS:
            raise _ProtocolIncomparable(
                "exact-step replay trace lacks exact DMO sequence binding"
            )
        _validate_exact_step_trace_binding(
            manifest,
            trace=trace,
            tick_map=tick_map,
            video_inspection=inspection,
            source_run=loaded,
        )

        post_snapshot = PcStateSnapshot.read(
            verified_artifacts[run.post_snapshot_artifact]
        )
        if (
            post_snapshot.state_root != host_pre_snapshot.state_root
            or post_snapshot.captured_perf_counter_ns
            <= loaded.attempt_finished_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "exact-step post-run state was not restored after capture"
            )
        loaded_runs.append(loaded)
        traces.append(trace)
        post_snapshots.append(post_snapshot)
        run_summaries.append(
            {
                "run_id": run.run_id,
                "selected_attempt": run.selected_attempt,
                "process_id": run.process_id,
                "process_creation_filetime_100ns": (
                    run.process_creation_filetime_100ns
                ),
                "tick_count": len(tick_map.records),
                "source_bmp_count": len(loaded.visual_frames),
                "video_frame_count": inspection.frame_count,
                "dmo_binding": "exact",
                "video_rgb24_bound_to_source_bmp": True,
                "post_capture_state_restored": True,
            }
        )

    if host_pre_snapshot.captured_perf_counter_ns >= min(
        run.campaign_started_perf_counter_ns for run in loaded_runs
    ):
        raise PcGoldenValidationError(
            "exact-step host prestate was not captured before all runs"
        )
    nonces = {host_pre_snapshot.session_nonce}
    nonces.update(snapshot.session_nonce for snapshot in post_snapshots)
    if len(nonces) != len(post_snapshots) + 1:
        raise PcGoldenValidationError(
            "exact-step state snapshots reuse a session nonce"
        )

    recomputed_comparison = compare_formal_exact_step_runs(loaded_runs)
    if (
        recomputed_comparison.get("schema") != EXACT_STEP_COMPARISON_SCHEMA
        or recomputed_comparison.get("version")
        != EXACT_STEP_COMPARISON_VERSION
        or recomputed_comparison.get("status") != "PASS"
    ):
        raise PcGoldenValidationError(
            "formal exact-step independent replay comparison failed"
        )
    stored_comparison = read_exact_step_canonical_json(
        verified_artifacts[contract.comparison_artifact]
    )
    if stored_comparison != recomputed_comparison:
        raise PcGoldenValidationError(
            "stored exact-step comparison differs from independent recomputation"
        )

    reference_trace = _normalized_replay_trace(traces[0])
    if any(
        _normalized_replay_trace(trace) != reference_trace
        for trace in traces[1:]
    ):
        raise PcGoldenValidationError(
            "independent exact-step packaged trace semantics differ"
        )
    return {
        "machine_verifiable_evidence": True,
        "transport": "formal_exact_step_external_lossless",
        "run_count": len(loaded_runs),
        "native_tick_count": len(loaded_runs[0].visual_frames),
        "fresh_holdout_preregistered": True,
        "independent_processes": True,
        "normalized_gameplay_and_rng_exact": True,
        "render_driving_state_exact": True,
        "full_800x600_bgra_bmp_per_tick_matched": True,
        "full_800x600_rgb24_video_per_tick_bound": True,
        "full_viewport_rgb24_per_tick_matched": True,
        "normalized_trace_matched": True,
        "dmo_binding_matched": True,
        "host_state_restored_after_every_run": True,
        "end_state_roots_matched": True,
        "runs": run_summaries,
    }


def _validate_framework_update_map(
    manifest: PcGoldenManifest,
    *,
    run: ReplayRunContract,
    update_map: PcFrameworkUpdateMap,
    capture_metadata: PcDxgiCaptureMetadata,
    capture_metadata_sha256: str,
    inspection: PcVideoInspection,
    tick_map: PcTickMap,
    resolved_updates: Sequence[int] | None = None,
) -> None:
    """Recompute the DMO-update to native-tick/video-PTS phase binding."""

    if (
        update_map.capture_metadata_sha256 != capture_metadata_sha256
        or update_map.process_instance != capture_metadata.process_instance
        or update_map.executable_sha256
        != manifest.pc_environment.executable_sha256
    ):
        raise PcGoldenValidationError(
            "framework update map identity differs from DXGI/process evidence"
        )
    if (
        len(update_map.records) != inspection.frame_count
        or len(inspection.dxgi_source_present_ticks)
        != inspection.frame_count
    ):
        raise _ProtocolIncomparable(
            "framework update map or video lacks a complete frame timeline"
        )
    if (
        resolved_updates is not None
        and len(resolved_updates) != len(update_map.records)
    ):
        raise PcGoldenValidationError(
            "render-settled update count differs from the frame timeline"
        )
    observed_present_ticks = tuple(
        record.present_ticks for record in update_map.records
    )
    if observed_present_ticks != inspection.dxgi_source_present_ticks:
        raise PcGoldenValidationError(
            "framework update map presentation ticks differ from video "
            "provenance"
        )
    if any(
        not (
            capture_metadata.capture_start_perf_counter_ns
            < record.host_perf_counter_ns
            <= capture_metadata.capture_end_perf_counter_ns
        )
        for record in update_map.records
    ):
        raise PcGoldenValidationError(
            "framework update samples fall outside the DXGI capture interval"
        )

    stable_pts_by_tick: dict[int, set[int]] = {}
    for index, (record, pts) in enumerate(
        zip(
            update_map.records,
            inspection.frame_pts,
            strict=True,
        )
    ):
        update = (
            record.stable_update
            if resolved_updates is None
            else resolved_updates[index]
        )
        if update is None:
            continue
        native_tick = (
            update + manifest.input_timeline.native_tick_offset
        )
        if 0 <= native_tick <= run.clock.tick_end:
            stable_pts_by_tick.setdefault(native_tick, set()).add(pts)
    expected_ticks = set(range(run.clock.tick_end + 1))
    if set(stable_pts_by_tick) != expected_ticks:
        raise _ProtocolIncomparable(
            "capture has no stable presentation for every native tick"
        )
    declared_pts = tuple(record.pts for record in tick_map.records)
    if any(
        declared_pts[tick] not in stable_pts_by_tick[tick]
        for tick in range(run.clock.tick_end + 1)
    ):
        raise PcGoldenValidationError(
            "tick-map PTS are not stable framework-update/video candidates"
        )


def _verify_replay_determinism_evidence(
    manifest: PcGoldenManifest,
    verified_artifacts: Mapping[str, Path],
    *,
    demo: PopCapDemo,
    primary_trace: PcGoldenTrace,
    primary_tick_map: PcTickMap,
    primary_video_inspection: PcVideoInspection | None,
    save_context: _SaveProtocolContext,
) -> Mapping[str, Any]:
    contract = manifest.replay_determinism
    save_contract = manifest.save_transaction
    if contract is None or save_contract is None:
        raise _ProtocolIncomparable(
            "manifest does not declare v4 replay determinism evidence"
        )
    save_runs = {run.run_id: run for run in save_contract.runs}
    pixel_contract = contract.pixel_comparison
    excluded_bottom_rows = (
        pixel_contract.excluded_bottom_rows
        if pixel_contract is not None
        else 0
    )
    maximum_excluded_edge_mismatches = (
        pixel_contract.maximum_excluded_edge_mismatches
        if pixel_contract is not None
        else 0
    )
    reference_pixels: tuple[str, ...] | None = None
    reference_edge_pixels: tuple[bytes, ...] | None = None
    reference_trace: tuple[Mapping[str, Any], ...] | None = None
    run_summaries: list[Mapping[str, Any]] = []
    maximum_observed_edge_mismatches = 0

    for index, run in enumerate(contract.runs):
        if index == 0:
            trace = primary_trace
            tick_map = primary_tick_map
            inspection = primary_video_inspection
        else:
            trace = PcGoldenTrace.read_ndjson(
                verified_artifacts[run.trace_artifact]
            )
            trace.validate_against_run(
                manifest,
                trace_artifact=run.trace_artifact,
                video=run.video,
                clock=run.clock,
            )
            tick_map = PcTickMap.read(
                verified_artifacts[run.clock.tick_map_artifact]
            )
            tick_map.validate_against_run(
                manifest,
                trace,
                tick_map_artifact=run.clock.tick_map_artifact,
                video=run.video,
                clock=run.clock,
            )
            inspection = inspect_pc_video_artifact(
                verified_artifacts[run.video.artifact],
                metadata=run.video,
                artifact=manifest.artifacts[run.video.artifact],
            )
        if inspection is None:
            raise _ProtocolIncomparable(
                "video decoder is unavailable for a deterministic replay"
            )
        input_check = _input_sequence_check(manifest, trace, demo)
        if input_check.status is not VerificationCheckStatus.PASS:
            raise _ProtocolIncomparable(
                "replay trace lacks exact DMO sequence/update binding"
            )

        capture_metadata = PcDxgiCaptureMetadata.read(
            verified_artifacts[run.capture_metadata_artifact]
        )
        capture_spec = manifest.artifacts[run.capture_metadata_artifact]
        if (
            capture_metadata.sha256 != capture_spec.sha256
            or len(capture_metadata.canonical_bytes) != capture_spec.bytes
        ):
            raise PcGoldenValidationError(
                "canonical DXGI metadata identity differs from its artifact"
            )
        save_run = save_runs[run.run_id]
        start_event, stop_event = save_context.process_pairs[run.run_id]
        start_snapshot = save_context.snapshots[f"{run.run_id}-start"]
        end_snapshot = save_context.snapshots[f"{run.run_id}-end"]
        if (
            capture_metadata.process_instance
            != save_run.process_instance
            or capture_metadata.executable_sha256
            != manifest.pc_environment.executable_sha256
        ):
            raise PcGoldenValidationError(
                "DXGI target process differs from the replay process contract"
            )
        if not (
            start_snapshot.captured_perf_counter_ns
            < start_event.perf_counter_ns
            <= capture_metadata.capture_start_perf_counter_ns
            < capture_metadata.capture_end_perf_counter_ns
            <= stop_event.perf_counter_ns
            < end_snapshot.captured_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "DXGI capture interval is outside the native replay lifetime"
            )
        if (
            capture_metadata.width != run.video.width
            or capture_metadata.height != run.video.height
            or capture_metadata.frame_count != run.video.frame_count
        ):
            raise PcGoldenValidationError(
                "DXGI metadata geometry/frame count differs from video"
            )
        if inspection.dxgi_source_metadata_sha256 is None:
            raise _ProtocolIncomparable(
                "video lacks embedded raw DXGI provenance"
            )
        if (
            inspection.dxgi_source_metadata_sha256
            != capture_spec.sha256
            or inspection.dxgi_source_raw_sha256
            != capture_metadata.raw_sha256
            or inspection.dxgi_source_frames_csv_sha256
            != capture_metadata.frames_csv_sha256
        ):
            raise PcGoldenValidationError(
                "video embedded provenance differs from raw DXGI metadata"
            )

        update_map = PcFrameworkUpdateMap.read(
            verified_artifacts[run.framework_update_artifact]
        )
        resolved_updates: tuple[int, ...] | None = None
        render_settled_summary: Mapping[str, Any] | None = None
        if run.render_settled_evidence is not None:
            settled_evidence = run.render_settled_evidence
            settled_map = PcRenderSettledUpdateMap.read(
                verified_artifacts[settled_evidence.update_map_artifact]
            )
            recomputed = verify_render_settled_update_map(
                settled_map,
                capture_metadata_path=verified_artifacts[
                    run.capture_metadata_artifact
                ],
                frames_csv_path=verified_artifacts[
                    settled_evidence.frames_csv_artifact
                ],
                framework_update_map_path=verified_artifacts[
                    run.framework_update_artifact
                ],
                framework_state_sidecar_path=verified_artifacts[
                    settled_evidence.framework_state_artifact
                ],
                framework_poll_path=verified_artifacts[
                    settled_evidence.framework_poll_artifact
                ],
                calibration_preregistration_path=verified_artifacts[
                    settled_evidence.calibration_preregistration_artifact
                ],
                calibration_execution_binding_path=verified_artifacts[
                    settled_evidence.calibration_execution_binding_artifact
                ],
                calibration_holdout_report_path=verified_artifacts[
                    settled_evidence.calibration_holdout_report_artifact
                ],
            )
            resolved_updates = tuple(
                row.assigned_framework_update
                for row in recomputed.records
            )
            render_settled_summary = {
                "source_recomputed": True,
                "render_settle_delay_ns": (
                    recomputed.render_settle_delay_ns
                ),
                "first_assigned_framework_update": resolved_updates[0],
                "last_assigned_framework_update": resolved_updates[-1],
            }
        _validate_framework_update_map(
            manifest,
            run=run,
            update_map=update_map,
            capture_metadata=capture_metadata,
            capture_metadata_sha256=capture_spec.sha256,
            inspection=inspection,
            tick_map=tick_map,
            resolved_updates=resolved_updates,
        )
        tick_pixels = _per_tick_pixel_hashes(
            inspection,
            tick_map,
            excluded_bottom_rows=excluded_bottom_rows,
        )
        tick_edge_pixels = (
            _per_tick_last_row_pixels(inspection, tick_map)
            if excluded_bottom_rows == 1
            else None
        )
        normalized_trace = _normalized_replay_trace(trace)
        if reference_pixels is None:
            reference_pixels = tick_pixels
            reference_edge_pixels = tick_edge_pixels
            reference_trace = normalized_trace
        else:
            if tick_pixels != reference_pixels:
                first_difference = next(
                    tick
                    for tick, (left, right) in enumerate(
                        zip(
                            reference_pixels,
                            tick_pixels,
                            strict=True,
                        )
                    )
                    if left != right
                )
                raise PcGoldenValidationError(
                    "independent replay gameplay viewport pixels differ at native "
                    f"tick {first_difference}"
                )
            if excluded_bottom_rows == 1:
                assert reference_edge_pixels is not None
                assert tick_edge_pixels is not None
                expected_row_bytes = inspection.width * 3
                for tick, (left_row, right_row) in enumerate(
                    zip(
                        reference_edge_pixels,
                        tick_edge_pixels,
                        strict=True,
                    )
                ):
                    if (
                        len(left_row) != expected_row_bytes
                        or len(right_row) != expected_row_bytes
                    ):
                        raise _ProtocolIncomparable(
                            "video decoder exposed an invalid final raster row"
                        )
                    left_pixels = np.frombuffer(
                        left_row,
                        dtype=np.uint8,
                    ).reshape((inspection.width, 3))
                    right_pixels = np.frombuffer(
                        right_row,
                        dtype=np.uint8,
                    ).reshape((inspection.width, 3))
                    mismatches = int(
                        np.any(left_pixels != right_pixels, axis=1).sum()
                    )
                    maximum_observed_edge_mismatches = max(
                        maximum_observed_edge_mismatches,
                        mismatches,
                    )
                    if mismatches > maximum_excluded_edge_mismatches:
                        raise PcGoldenValidationError(
                            "independent replay bottom raster edge exceeds "
                            f"its mismatch budget at native tick {tick}"
                        )
            assert reference_trace is not None
            if normalized_trace != reference_trace:
                first_difference = next(
                    tick
                    for tick, (left, right) in enumerate(
                        zip(
                            reference_trace,
                            normalized_trace,
                            strict=True,
                        )
                    )
                    if left != right
                )
                raise PcGoldenValidationError(
                    "independent replay trace semantics differ at native "
                    f"tick {first_difference}"
                )
        run_summaries.append(
            {
                "run_id": run.run_id,
                "process_id": save_run.process_id,
                "tick_count": len(tick_map.records),
                "frame_count": inspection.frame_count,
                "normal_exit": stop_event.exit_code == 0,
                "capture_inside_process_lifetime": True,
                "dmo_binding": "exact",
                "gameplay_viewport_excluded_bottom_rows": (
                    excluded_bottom_rows
                ),
                "render_settled_update_evidence": render_settled_summary,
            }
        )

    assert reference_pixels is not None
    pixel_fingerprint = hashlib.sha256()
    for digest in reference_pixels:
        pixel_fingerprint.update(bytes.fromhex(digest.removeprefix("sha256:")))
    return {
        "machine_verifiable_evidence": True,
        "run_count": len(contract.runs),
        "native_tick_count": len(reference_pixels),
        "full_viewport_rgb24_per_tick_matched": (
            excluded_bottom_rows == 0
        ),
        "gameplay_viewport_rgb24_per_tick_matched": True,
        "excluded_bottom_rows": excluded_bottom_rows,
        "maximum_excluded_edge_mismatches": (
            maximum_excluded_edge_mismatches
        ),
        "maximum_observed_excluded_edge_mismatches": (
            maximum_observed_edge_mismatches
        ),
        "normalized_trace_matched": True,
        "dmo_binding_matched": True,
        "independent_processes": True,
        "end_state_roots_matched": True,
        "per_tick_pixel_set_sha256": (
            f"sha256:{pixel_fingerprint.hexdigest()}"
        ),
        "runs": run_summaries,
    }


def _verify_memory_transition_evidence(
    manifest: PcGoldenManifest,
    verified_artifacts: Mapping[str, Path],
    *,
    trace: PcGoldenTrace,
    tick_map: PcTickMap,
    video_inspection: PcVideoInspection,
    original_root: str | Path | None,
) -> Mapping[str, Any]:
    contract_path = verified_artifacts[
        MEMORY_TRANSITION_CONTRACT_ARTIFACT
    ]
    contract = PcMemoryTransitionContract.from_dict(
        read_canonical_report(contract_path)
    )
    if contract.case_id != manifest.case_id:
        raise PcMemoryEvidenceError(
            "memory_transition_contract_case_id_mismatch"
        )
    missing = contract.referenced_artifacts.difference(
        verified_artifacts
    )
    if missing:
        raise PcMemoryEvidenceError(
            "memory_transition_contract_artifact_missing"
        )
    if MEMORY_CURVE_GEOMETRY_ARTIFACT not in verified_artifacts:
        raise PcMemoryEvidenceError(
            "memory_curve_geometry_binding_artifact_missing"
        )
    allowed_paths = frozenset(
        path.resolve() for path in verified_artifacts.values()
    )
    before_probe_path = verified_artifacts[
        contract.before_probe_artifact
    ]
    after_probe_path = verified_artifacts[
        contract.after_probe_artifact
    ]
    before_pixel_path = verified_artifacts[
        contract.before_pixel_validation_artifact
    ]
    after_pixel_path = verified_artifacts[
        contract.after_pixel_validation_artifact
    ]
    transition_path = verified_artifacts[
        contract.transition_validation_artifact
    ]
    binding_path = verified_artifacts[
        contract.video_binding_validation_artifact
    ]
    curve_geometry_path = verified_artifacts[
        MEMORY_CURVE_GEOMETRY_ARTIFACT
    ]
    for path in (
        before_pixel_path,
        after_pixel_path,
        transition_path,
        binding_path,
        curve_geometry_path,
    ):
        read_canonical_report(path)

    selected_first_update = -manifest.input_timeline.native_tick_offset
    native_ticks = (
        contract.expected_before_update - selected_first_update,
        contract.expected_after_update - selected_first_update,
    )
    if (
        native_ticks[0] < 0
        or native_ticks[1] <= native_ticks[0]
        or native_ticks[1] > manifest.clock.tick_end
    ):
        raise PcMemoryEvidenceError(
            "memory_transition_native_tick_mapping_invalid"
        )
    for tick in native_ticks:
        measurement = trace.records[tick].measurements.get("score")
        if (
            measurement is None
            or measurement.status is not MeasurementStatus.OBSERVED
            or measurement.value != contract.expected_score
        ):
            raise PcMemoryEvidenceError(
                "memory_transition_score_trace_binding_invalid"
            )

    recomputed_transition = validate_shot_transition(
        before_probe_path,
        after_probe_path,
        before_pixel_path,
        after_pixel_path,
        expected_before_update=contract.expected_before_update,
        expected_after_update=contract.expected_after_update,
        expected_score=contract.expected_score,
        expected_chain_distance_delta=(
            contract.expected_chain_distance_delta
        ),
        distance_tolerance=contract.distance_tolerance,
        minimum_chain_count=contract.minimum_chain_count,
        minimum_visible=contract.minimum_visible,
        maximum_mismatches=contract.maximum_mismatches,
        minimum_visible_after_fired_bullets=(
            contract.minimum_visible_after_fired_bullets
        ),
        maximum_fired_bullet_mismatches=(
            contract.maximum_fired_bullet_mismatches
        ),
        expected_runtime_sha256=(
            manifest.pc_environment.executable_sha256
        ),
        expected_dmo_sha256=manifest.artifacts[
            manifest.input_timeline.artifact
        ].sha256,
        allowed_paths=allowed_paths,
    )
    if read_canonical_report(transition_path) != recomputed_transition:
        raise PcMemoryEvidenceError(
            "memory_transition_report_not_canonical"
        )

    before_raw = validate_probe_payloads(
        before_probe_path,
        expected_runtime_sha256=(
            manifest.pc_environment.executable_sha256
        ),
        expected_dmo_sha256=manifest.artifacts[
            manifest.input_timeline.artifact
        ].sha256,
        expected_update=contract.expected_before_update,
        expected_score=contract.expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    after_raw = validate_probe_payloads(
        after_probe_path,
        expected_runtime_sha256=(
            manifest.pc_environment.executable_sha256
        ),
        expected_dmo_sha256=manifest.artifacts[
            manifest.input_timeline.artifact
        ].sha256,
        expected_update=contract.expected_after_update,
        expected_score=contract.expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    curve_geometry_binding: Mapping[str, Any] | None = None
    if original_root is not None:
        catalog = OriginalGameCatalog(original_root)
        loaded = catalog.load_level(
            manifest.scenario.level_id,
            hard=manifest.scenario.hard,
        )
        if not 0 <= manifest.scenario.curve_index < len(loaded.curves):
            raise PcMemoryEvidenceError(
                "memory_curve_geometry_curve_index_invalid"
            )
        curve_geometry_binding = build_memory_curve_geometry_binding(
            case_id=manifest.case_id,
            level_id=loaded.definition.id,
            hard=manifest.scenario.hard,
            curve_index=manifest.scenario.curve_index,
            curve=loaded.curves[manifest.scenario.curve_index],
            phases=(
                (
                    "before",
                    contract.expected_before_update,
                    before_raw["active_chain"],
                ),
                (
                    "after",
                    contract.expected_after_update,
                    after_raw["active_chain"],
                ),
            ),
        )
        if (
            read_canonical_report(curve_geometry_path)
            != curve_geometry_binding
        ):
            raise PcMemoryEvidenceError(
                "memory_curve_geometry_report_not_canonical"
            )
    requested_pts = tuple(
        tick_map.records[tick].pts for tick in native_ticks
    )
    video_frames = decode_pc_video_rgb24_at_pts(
        verified_artifacts[manifest.video.artifact],
        metadata=manifest.video,
        artifact=manifest.artifacts[manifest.video.artifact],
        requested_pts=requested_pts,
    )
    try:
        from PIL import Image
    except ImportError as error:
        raise PcMemoryImageDecoderUnavailable(
            "Pillow is required for memory/video pixel binding"
        ) from error
    binding_rows: list[dict[str, Any]] = []
    for phase, update, tick, raw, pts in zip(
        ("before", "after"),
        (
            contract.expected_before_update,
            contract.expected_after_update,
        ),
        native_ticks,
        (before_raw, after_raw),
        requested_pts,
        strict=True,
    ):
        video_payload = video_frames[pts]
        if len(video_payload) != 800 * 600 * 3:
            raise PcMemoryEvidenceError(
                "memory_video_frame_size_invalid"
            )
        video_rgb = np.frombuffer(
            video_payload,
            dtype=np.uint8,
        ).reshape((600, 800, 3))
        try:
            with Image.open(raw["frozen_frame_path"]) as source:
                if source.format != "BMP" or source.size != (800, 600):
                    raise PcMemoryEvidenceError(
                        "memory_video_probe_frame_invalid"
                    )
                probe_rgb = np.asarray(source.convert("RGB"))
        except PcMemoryEvidenceError:
            raise
        except Exception as error:
            raise PcMemoryEvidenceError(
                "memory_video_probe_frame_decode_failed"
            ) from error
        difference = np.any(probe_rgb != video_rgb, axis=2)
        interior_end = (
            600 - contract.video_binding_excluded_bottom_rows
        )
        interior_mismatches = int(difference[:interior_end].sum())
        edge_mismatches = int(difference[interior_end:].sum())
        if interior_mismatches:
            raise PcMemoryEvidenceError(
                "memory_video_gameplay_pixels_differ"
            )
        if (
            edge_mismatches
            > contract.maximum_excluded_edge_mismatches
        ):
            raise PcMemoryEvidenceError(
                "memory_video_edge_mismatch_budget_exceeded"
            )
        try:
            frame_sequence = video_inspection.frame_pts.index(pts)
        except ValueError as error:
            raise PcMemoryEvidenceError(
                "memory_video_pts_not_in_inspection"
            ) from error
        binding_rows.append(
            {
                "phase": phase,
                "framework_update": update,
                "native_tick": tick,
                "video_frame_sequence": frame_sequence,
                "video_pts": pts,
                "interior_height": interior_end,
                "interior_mismatch_count": interior_mismatches,
                "excluded_bottom_rows": (
                    contract.video_binding_excluded_bottom_rows
                ),
                "excluded_edge_mismatch_count": edge_mismatches,
                "probe_rgb24_sha256": _sha256_bytes(
                    np.ascontiguousarray(probe_rgb).tobytes()
                ),
                "video_rgb24_sha256": _sha256_bytes(video_payload),
                "interior_rgb24_sha256": _sha256_bytes(
                    np.ascontiguousarray(
                        probe_rgb[:interior_end]
                    ).tobytes()
                ),
            }
        )
    recomputed_binding = {
        "schema": "zuma-rl.pc-memory-video-binding",
        "version": 1,
        "status": "PASS",
        "case_id": manifest.case_id,
        "primary_video_artifact": manifest.video.artifact,
        "selected_first_update": selected_first_update,
        "excluded_bottom_rows": (
            contract.video_binding_excluded_bottom_rows
        ),
        "maximum_excluded_edge_mismatches": (
            contract.maximum_excluded_edge_mismatches
        ),
        "bindings": binding_rows,
    }
    if read_canonical_report(binding_path) != recomputed_binding:
        raise PcMemoryEvidenceError(
            "memory_video_binding_report_not_canonical"
        )

    shot = recomputed_transition["shot_transition"][
        "transferred_fired_bullet"
    ]
    after_pixels = recomputed_transition["inputs"][
        "after_pixel_validation"
    ]
    after_pixel_report = read_canonical_report(after_pixel_path)
    active_color_ids = sorted(
        int(color_id)
        for color_id, row in after_pixel_report["per_color"].items()
        if row["visible"] > 0 and row["matches"] == row["visible"]
    )
    fired_color_ids = sorted(
        {
            int(row["color_id"])
            for row in after_pixel_report["fired_bullets"]["results"]
            if row["match"] is True
        }
    )
    return {
        "machine_verifiable_evidence": True,
        "before_update": contract.expected_before_update,
        "after_update": contract.expected_after_update,
        "native_ticks": list(native_ticks),
        "active_chain_count": recomputed_transition["active_chain"][
            "count"
        ],
        "active_visible_count": after_pixels["active_visible_count"],
        "active_pixel_match_count": after_pixels["active_match_count"],
        "active_color_ids": active_color_ids,
        "fired_visible_count": after_pixels["fired_visible_count"],
        "fired_pixel_match_count": after_pixels["fired_match_count"],
        "fired_color_ids": fired_color_ids,
        "fired_ball_id": shot["ball_id"],
        "fired_color_id": shot["color_id"],
        "fired_speed": shot["speed"],
        "gameplay_viewport_pixels_matched": True,
        "curve_geometry_binding": {
            "resolved_against_original_data": (
                curve_geometry_binding is not None
            ),
            "level_id": manifest.scenario.level_id,
            "hard": manifest.scenario.hard,
            "curve_index": manifest.scenario.curve_index,
            "phases": (
                curve_geometry_binding["phases"]
                if curve_geometry_binding is not None
                else ()
            ),
        },
        "video_bindings": binding_rows,
        "raw_artifact_count": len(
            before_raw["used_artifact_paths"]
            | after_raw["used_artifact_paths"]
        ),
    }


def verify_pc_golden_case(
    manifest_path: str | Path,
    *,
    case_root: str | Path | None = None,
    original_root: str | Path | None = None,
    require_certifying_dmo_provenance: bool = False,
) -> PcGoldenVerificationReport:
    """Read and strictly accept one immutable PC golden case.

    Structural, artifact, trace, DMO, or explicit input-binding errors produce
    ``FAIL``.  A structurally valid case whose required evidence is incomplete
    produces ``INCOMPARABLE``.  When ``require_certifying_dmo_provenance`` is
    true, the raw retail recording, acquisition report, sole permitted startup
    normalization, and collector plan must also be machine-verifiable.  The
    function performs no filesystem writes.
    """

    manifest_source = Path(manifest_path)
    root = manifest_source.parent if case_root is None else Path(case_root)
    checks: list[VerificationCheck] = []
    summary: dict[str, Any] = {}

    try:
        manifest = PcGoldenManifest.read_json(manifest_source)
    except (
        OSError,
        UnicodeError,
        PcGoldenValidationError,
        RecursionError,
    ) as error:
        return _failure_report(
            case_id=None,
            checks=checks,
            stage="manifest",
            message="manifest could not be loaded and validated",
            error=error,
        )
    checks.append(
        VerificationCheck(
            name="manifest",
            status=VerificationCheckStatus.PASS,
            message="manifest schema and invariants are valid",
            details={"schema_version": manifest.manifest_version},
        )
    )
    summary["artifact_count"] = len(manifest.artifacts)

    required_provenance_artifacts = {
        RAW_RECORDING_DMO_ARTIFACT,
        RECORDING_REPORT_ARTIFACT,
        COLLECTOR_PLAN_ARTIFACT,
        PROVENANCE_ARTIFACT,
        PLAYBACK_DMO_ARTIFACT,
    }
    if require_certifying_dmo_provenance and (
        manifest.input_timeline.artifact != PLAYBACK_DMO_ARTIFACT
        or required_provenance_artifacts.difference(manifest.artifacts)
    ):
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="dmo_source_provenance",
            message=(
                "certification requires a provenance-bound raw retail "
                "recording and exact startup transport normalization"
            ),
            summary=summary,
        )

    if manifest.exact_step_replay is not None:
        video_declarations = tuple(
            (run.video, run.video.artifact)
            for run in manifest.exact_step_replay.runs
        )
    elif manifest.replay_determinism is None:
        video_declarations = ((manifest.video, manifest.video.artifact),)
    else:
        video_declarations = tuple(
            (run.video, run.video.artifact)
            for run in manifest.replay_determinism.runs
        )
    video_artifact_names = {
        artifact_name for _, artifact_name in video_declarations
    }
    try:
        normalized_video_bytes = sum(
            validate_pc_video_artifact_declaration(
                metadata,
                manifest.artifacts[artifact_name],
            )
            for metadata, artifact_name in video_declarations
        )
    except VideoResourceLimitError:
        reason = (
            "declared video evidence exceeds the verifier resource budget"
        )
        checks.append(
            VerificationCheck(
                name="artifact_budget",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=reason,
                details={"artifact_content_read": False},
            )
        )
        return PcGoldenVerificationReport(
            status=ComparisonStatus.INCOMPARABLE,
            case_id=manifest.case_id,
            checks=tuple(checks),
            reasons=(reason,),
            summary=summary,
        )
    except PcVideoError as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="artifact_budget",
            message="video resource declarations are invalid",
            error=error,
            summary=summary,
        )
    nonvideo_artifacts = tuple(
        artifact
        for name, artifact in manifest.artifacts.items()
        if name not in video_artifact_names
    )
    nonvideo_bytes = sum(
        artifact.bytes for artifact in nonvideo_artifacts
    )
    video_bytes = sum(
        manifest.artifacts[name].bytes for name in video_artifact_names
    )
    if (
        len(manifest.artifacts) > MAX_PC_ARTIFACT_COUNT
        or any(
            artifact.bytes > MAX_NONVIDEO_ARTIFACT_BYTES
            for artifact in nonvideo_artifacts
        )
        or nonvideo_bytes > MAX_TOTAL_NONVIDEO_ARTIFACT_BYTES
        or video_bytes > MAX_TOTAL_VIDEO_ARTIFACT_BYTES
        or normalized_video_bytes > MAX_TOTAL_NORMALIZED_VIDEO_BYTES
    ):
        reason = (
            "declared PC evidence exceeds the verifier resource budget"
        )
        checks.append(
            VerificationCheck(
                name="artifact_budget",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=reason,
                details={"artifact_content_read": False},
            )
        )
        return PcGoldenVerificationReport(
            status=ComparisonStatus.INCOMPARABLE,
            case_id=manifest.case_id,
            checks=tuple(checks),
            reasons=(reason,),
            summary=summary,
        )
    budget_summary = {
        "artifact_count": len(manifest.artifacts),
        "nonvideo_artifact_bytes": nonvideo_bytes,
        "video_artifact_bytes": video_bytes,
        "video_artifact_count": len(video_artifact_names),
        "normalized_video_bytes": normalized_video_bytes,
    }
    summary["artifact_budget"] = budget_summary
    checks.append(
        VerificationCheck(
            name="artifact_budget",
            status=VerificationCheckStatus.PASS,
            message=(
                "declared artifacts fit the verifier resource budget"
            ),
            details=budget_summary,
        )
    )

    try:
        verified_artifacts = manifest.verify_artifacts(root)
    except (
        OSError,
        MemoryError,
        OverflowError,
        PcGoldenArtifactError,
        RecursionError,
    ) as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="artifacts",
            message="artifact inventory verification failed",
            error=error,
            summary=summary,
        )
    checks.append(
        VerificationCheck(
            name="artifacts",
            status=VerificationCheckStatus.PASS,
            message="all artifact byte counts and SHA-256 digests match",
            details={"verified_count": len(verified_artifacts)},
        )
    )

    if require_certifying_dmo_provenance:
        try:
            provenance_summary = read_certifying_provenance(
                source_path=verified_artifacts[
                    RAW_RECORDING_DMO_ARTIFACT
                ],
                output_path=verified_artifacts[PLAYBACK_DMO_ARTIFACT],
                recording_report_path=verified_artifacts[
                    RECORDING_REPORT_ARTIFACT
                ],
                collector_plan_path=verified_artifacts[
                    COLLECTOR_PLAN_ARTIFACT
                ],
                provenance_path=verified_artifacts[PROVENANCE_ARTIFACT],
                expected_runtime_sha256=(
                    manifest.pc_environment.executable_sha256
                ),
            )
        except (
            MemoryError,
            OverflowError,
            RecursionError,
            RetailDmoProvenanceError,
        ) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="dmo_source_provenance",
                message=(
                    "raw retail DMO provenance or startup normalization "
                    "could not be certified"
                ),
                error=error,
                summary=summary,
            )
        summary["dmo_source_provenance"] = provenance_summary
        checks.append(
            VerificationCheck(
                name="dmo_source_provenance",
                status=VerificationCheckStatus.PASS,
                message=(
                    "raw retail recording and the sole permitted startup "
                    "transport normalization were independently recomputed"
                ),
                details=provenance_summary,
            )
        )

    try:
        trace = PcGoldenTrace.read_ndjson(
            verified_artifacts[manifest.trace_artifact]
        )
        trace.validate_against_manifest(manifest)
    except (
        OSError,
        UnicodeError,
        PcGoldenValidationError,
        RecursionError,
    ) as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="trace",
            message="trace could not be loaded or matched to the manifest",
            error=error,
            summary=summary,
        )
    trace_input_count = sum(
        len(record.inputs) for record in trace.records
    )
    summary.update(
        {
            "trace_tick_count": len(trace.records),
            "trace_input_count": trace_input_count,
        }
    )
    checks.append(
        VerificationCheck(
            name="trace",
            status=VerificationCheckStatus.PASS,
            message="canonical trace matches the manifest",
            details={
                "tick_count": len(trace.records),
                "input_count": trace_input_count,
            },
        )
    )

    try:
        tick_map = PcTickMap.read(
            verified_artifacts[manifest.clock.tick_map_artifact]
        )
        tick_map.validate_against(manifest, trace)
    except (OSError, UnicodeError, PcGoldenValidationError) as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="tick_map",
            message="tick-map could not be loaded or matched to the trace",
            error=error,
            summary=summary,
        )
    checks.append(
        VerificationCheck(
            name="tick_map",
            status=VerificationCheckStatus.PASS,
            message="canonical tick-map matches trace frames and input PTS",
            details={"tick_count": len(tick_map.records)},
        )
    )

    try:
        calibration = PcCalibrationSidecar.read(
            verified_artifacts[
                manifest.coordinates.calibration_artifact
            ]
        )
        calibration_fit = calibration.validate_against(manifest)
    except (OSError, UnicodeError, PcGoldenValidationError) as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="calibration",
            message=(
                "coordinate calibration could not be independently "
                "recomputed and matched to the manifest"
            ),
            error=error,
            summary=summary,
        )
    summary["calibration"] = {
        "control_point_count": len(calibration.control_points),
        "rms_error_px": calibration_fit.rms_error_px,
        "max_error_px": calibration_fit.max_error_px,
    }
    checks.append(
        VerificationCheck(
            name="calibration",
            status=VerificationCheckStatus.PASS,
            message=(
                "coordinate transform and errors were recomputed from "
                "control points"
            ),
            details=summary["calibration"],
        )
    )

    try:
        demo = PopCapDemo.read(
            verified_artifacts[manifest.input_timeline.artifact]
        )
    except (OSError, PopCapDemoError) as error:
        # Deliberately omit PopCapDemoError text: future parsers must not make
        # an embedded DMO value observable through this report.
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="dmo",
            message="DMO parsing failed",
            error=error,
            summary=summary,
        )

    mismatches = _dmo_header_mismatches(manifest, demo)
    if mismatches:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="dmo",
            message=(
                "DMO header does not match InputTimeline fields: "
                + ", ".join(mismatches)
            ),
            summary=summary,
        )
    dmo_summary = {
        "format": DMO_FORMAT,
        "file_id": f"0x{DEMO_FILE_ID:08X}",
        "dmo_version": demo.version,
        "product_version_matched": True,
        "product_version_bytes": len(
            demo.product_version.encode("latin-1")
        ),
        "random_seed": demo.random_seed,
        "length_updates": demo.length_updates,
        "artifact_bytes": demo.artifact_bytes,
        "artifact_sha256": demo.artifact_sha256,
        "marker_count": len(demo.markers),
        "command_count": len(demo.commands),
        "input_command_count": len(demo.input_commands),
    }
    summary["dmo"] = dmo_summary
    checks.append(
        VerificationCheck(
            name="dmo",
            status=VerificationCheckStatus.PASS,
            message="DMO header and artifact identity match InputTimeline",
            details=dmo_summary,
        )
    )

    try:
        input_check = _input_sequence_check(manifest, trace, demo)
    except _InputSequenceMismatch as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="input_sequence",
            message=str(error),
            error=error,
            summary=summary,
        )
    checks.append(input_check)

    readiness = manifest.coverage_readiness()
    readiness_status = VerificationCheckStatus(readiness.status.value)
    checks.append(
        VerificationCheck(
            name="coverage",
            status=readiness_status,
            message=(
                "required evidence coverage is complete"
                if readiness.status is ComparisonStatus.PASS
                else "required evidence is not comparable"
            ),
            details={
                "range_count": len(manifest.coverage),
                "required_range_count": sum(
                    int(item.required) for item in manifest.coverage
                ),
            },
        )
    )
    trace_readiness = _trace_coverage_readiness(manifest, trace)
    checks.append(
        VerificationCheck(
            name="trace_coverage",
            status=VerificationCheckStatus(
                trace_readiness.status.value
            ),
            message=(
                "required trace evidence is present"
                if trace_readiness.status is ComparisonStatus.PASS
                else "required trace evidence is not comparable"
            ),
            details=trace_readiness.metrics,
        )
    )
    summary["coverage_status"] = readiness.status.value
    summary["trace_coverage_status"] = trace_readiness.status.value
    summary["input_sequence_status"] = input_check.status.value
    summary["semantic_limits"] = {
        "video_stream_decoded": False,
        "calibration_sidecar_parsed": True,
        "scenario_resolved_against_original_data": False,
        "save_restoration_attested": False,
        "deterministic_replay_attested": False,
        "memory_measurement_provenance_attested": False,
    }
    reasons = list(readiness.reasons)
    final_status = readiness.status
    if trace_readiness.status is ComparisonStatus.INCOMPARABLE:
        reasons.extend(trace_readiness.reasons)
        final_status = ComparisonStatus.INCOMPARABLE
    if input_check.status is VerificationCheckStatus.INCOMPARABLE:
        reasons.append(
            "trace and DMO input evidence lacks an explicit sequence/update "
            "binding"
        )
        if final_status is ComparisonStatus.PASS:
            final_status = ComparisonStatus.INCOMPARABLE
    semantic_reasons: list[str] = []
    video_inspection: PcVideoInspection | None = None
    try:
        video_inspection = inspect_pc_video(
            verified_artifacts[manifest.video.artifact],
            manifest,
        )
    except VideoDecoderUnavailable:
        video_reason = (
            "video decoder is unavailable, so the video stream cannot be "
            "independently matched to VideoMetadata"
        )
        semantic_reasons.append(video_reason)
        checks.append(
            VerificationCheck(
                name="video_semantics",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=video_reason,
                details={"decoder_available": False},
            )
        )
    except PcVideoError as error:
        return _failure_report(
            case_id=manifest.case_id,
            checks=checks,
            stage="video_semantics",
            message=(
                "video stream could not be fully decoded and matched to "
                "VideoMetadata"
            ),
            error=error,
            summary=summary,
        )
    else:
        frame_binding_mismatch_count = 0
        referenced_frame_indices: set[int] = set()
        if (
            len(video_inspection.frame_pts)
            != manifest.video.frame_count
        ):
            frame_binding_mismatch_count = 1
        else:
            for record in trace.records:
                for frame in record.frames:
                    referenced_frame_indices.add(frame.frame_index)
                    if (
                        video_inspection.frame_pts[frame.frame_index]
                        != frame.pts
                    ):
                        frame_binding_mismatch_count += 1
        if frame_binding_mismatch_count:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="video_semantics",
                message=(
                    "decoded video frame PTS do not match trace/tick-map "
                    "frame references"
                ),
                summary=summary,
            )
        video_summary = {
            "width": video_inspection.width,
            "height": video_inspection.height,
            "codec": video_inspection.codec,
            "pixel_format": video_inspection.pixel_format,
            "time_base": list(video_inspection.time_base),
            "nominal_fps": list(video_inspection.nominal_fps),
            "frame_count": video_inspection.frame_count,
            "first_pts": video_inspection.first_pts,
            "last_pts": video_inspection.last_pts,
            "cfr": video_inspection.cfr,
            "dropped_frames": video_inspection.dropped_frames,
            "duplicate_frames": video_inspection.duplicate_frames,
            "normalized_pixel_format": (
                video_inspection.normalized_pixel_format
            ),
            "normalized_decoded_bytes": (
                video_inspection.normalized_decoded_bytes
            ),
            "decoded_pixel_sha256": (
                video_inspection.decoded_pixel_sha256
            ),
            "referenced_frame_pts_bound_to_trace": True,
            "referenced_frame_count": len(referenced_frame_indices),
            "unreferenced_frame_count": (
                video_inspection.frame_count
                - len(referenced_frame_indices)
            ),
        }
        summary["video"] = video_summary
        summary["semantic_limits"]["video_stream_decoded"] = True
        checks.append(
            VerificationCheck(
                name="video_semantics",
                status=VerificationCheckStatus.PASS,
                message=(
                    "the unique video stream was fully decoded and matched "
                    "to VideoMetadata"
                ),
                details=video_summary,
            )
        )

    if (
        MEMORY_TRANSITION_CONTRACT_ARTIFACT in manifest.artifacts
        and video_inspection is None
    ):
        provenance_reason = (
            "retail memory evidence cannot be bound to the primary video "
            "without a working video decoder"
        )
        semantic_reasons.append(provenance_reason)
        checks.append(
            VerificationCheck(
                name="measurement_provenance",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=provenance_reason,
                details={"machine_verifiable_evidence": False},
            )
        )
    elif MEMORY_TRANSITION_CONTRACT_ARTIFACT in manifest.artifacts:
        try:
            memory_details = _verify_memory_transition_evidence(
                manifest,
                verified_artifacts,
                trace=trace,
                tick_map=tick_map,
                video_inspection=video_inspection,
                original_root=original_root,
            )
        except PcMemoryImageDecoderUnavailable as error:
            provenance_reason = str(error)
            semantic_reasons.append(provenance_reason)
            checks.append(
                VerificationCheck(
                    name="measurement_provenance",
                    status=VerificationCheckStatus.INCOMPARABLE,
                    message=provenance_reason,
                    details={"machine_verifiable_evidence": False},
                )
            )
        except (
            KeyError,
            OSError,
            UnicodeError,
            PcMemoryEvidenceError,
            PcVideoError,
            RecursionError,
            ValueError,
        ) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="measurement_provenance",
                message=(
                    "retail memory bytes, frozen pixels, shot transition, "
                    "or primary-video binding failed independent "
                    "verification"
                ),
                error=error,
                summary=summary,
            )
        else:
            summary["measurement_provenance"] = memory_details
            summary["semantic_limits"][
                "memory_measurement_provenance_attested"
            ] = True
            checks.append(
                VerificationCheck(
                    name="measurement_provenance",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "retail Board/Ball/Bullet bytes, frozen BMP pixels, "
                        "shot transition, and primary video agree"
                    ),
                    details=memory_details,
                )
            )

    if original_root is None:
        scenario_reason = (
            "scenario level/curve/gun identifiers have not been resolved "
            "against the installed original data"
        )
        semantic_reasons.append(scenario_reason)
        checks.append(
            VerificationCheck(
                name="scenario_semantics",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=scenario_reason,
                details={"original_root_supplied": False},
            )
        )
    else:
        try:
            scenario_mismatches = _scenario_semantics_mismatches(
                manifest,
                original_root,
            )
        except (OSError, OriginalDataError, ValueError) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="scenario_semantics",
                message=(
                    "original scenario data could not be loaded and "
                    "validated"
                ),
                error=error,
                summary=summary,
            )
        if scenario_mismatches:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="scenario_semantics",
                message=(
                    "original scenario identity differs in fields: "
                    + ", ".join(scenario_mismatches)
                ),
                summary=summary,
            )
        checks.append(
            VerificationCheck(
                name="scenario_semantics",
                status=VerificationCheckStatus.PASS,
                message=(
                    "scenario identities match the caller-supplied original "
                    "game directory"
                ),
                details={"original_root_supplied": True},
            )
        )
        summary["semantic_limits"][
            "scenario_resolved_against_original_data"
        ] = True

    exact_step_details: Mapping[str, Any] | None = None
    if manifest.exact_step_replay is not None:
        try:
            exact_step_details = _verify_exact_step_replay_evidence(
                manifest,
                verified_artifacts,
                demo=demo,
                primary_trace=trace,
                primary_tick_map=tick_map,
                primary_video_inspection=video_inspection,
            )
        except (_ProtocolIncomparable, VideoDecoderUnavailable) as error:
            exact_reason = str(error)
            semantic_reasons.append(exact_reason)
            checks.append(
                VerificationCheck(
                    name="exact_step_replay",
                    status=VerificationCheckStatus.INCOMPARABLE,
                    message=exact_reason,
                    details={"machine_verifiable_evidence": False},
                )
            )
        except (
            KeyError,
            OSError,
            UnicodeError,
            PcExactStepEvidenceError,
            PcGoldenValidationError,
            PcRngTrajectoryError,
            PcVideoError,
            RecursionError,
            ValueError,
        ) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="exact_step_replay",
                message=(
                    "formal exact-step source, independent pixels, trace "
                    "semantics, process identity, or restored state differ"
                ),
                error=error,
                summary=summary,
            )
        else:
            summary["exact_step_replay"] = exact_step_details
            summary["replay_determinism"] = exact_step_details
            summary["save_transaction"] = {
                "machine_verifiable_evidence": True,
                "transport": "formal_exact_step_post_run_snapshots",
                "run_count": exact_step_details["run_count"],
                "state_roots_matched": True,
            }
            summary["semantic_limits"]["save_restoration_attested"] = True
            summary["semantic_limits"][
                "deterministic_replay_attested"
            ] = True
            checks.append(
                VerificationCheck(
                    name="save_transaction",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "pre-capture and per-run post-capture snapshots prove "
                        "host state restoration"
                    ),
                    details=summary["save_transaction"],
                )
            )
            checks.append(
                VerificationCheck(
                    name="exact_step_replay",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "fresh independent exact-step runs match at every "
                        "tick in native state and full external pixels"
                    ),
                    details=exact_step_details,
                )
            )
            checks.append(
                VerificationCheck(
                    name="replay_determinism",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "formal exact-step evidence proves deterministic "
                        "native replay under the frozen transport"
                    ),
                    details=exact_step_details,
                )
            )

    save_context: _SaveProtocolContext | None = None
    if manifest.exact_step_replay is not None:
        pass
    elif (
        manifest.manifest_version == LEGACY_MANIFEST_VERSION
        or manifest.save_transaction is None
    ):
        save_reason = (
            "legacy v3 manifest has no v4 save-transaction contract"
            if manifest.manifest_version == LEGACY_MANIFEST_VERSION
            else (
                "pre-capture backup and post-capture save/registry "
                "restoration are not represented by machine-verifiable "
                "evidence"
            )
        )
        semantic_reasons.append(save_reason)
        checks.append(
            VerificationCheck(
                name="save_transaction",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=save_reason,
                details={"machine_verifiable_evidence": False},
            )
        )
    else:
        try:
            save_context, save_details = _load_save_protocol_evidence(
                manifest,
                verified_artifacts,
            )
        except _ProtocolIncomparable as error:
            save_reason = str(error)
            semantic_reasons.append(save_reason)
            checks.append(
                VerificationCheck(
                    name="save_transaction",
                    status=VerificationCheckStatus.INCOMPARABLE,
                    message=save_reason,
                    details={"machine_verifiable_evidence": False},
                )
            )
        except (
            OSError,
            UnicodeError,
            PcGoldenValidationError,
            RecursionError,
        ) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="save_transaction",
                message=(
                    "raw save-transaction evidence failed independent "
                    "verification"
                ),
                error=error,
                summary=summary,
            )
        else:
            summary["save_transaction"] = save_details
            summary["semantic_limits"][
                "save_restoration_attested"
            ] = True
            checks.append(
                VerificationCheck(
                    name="save_transaction",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "raw state bundles, hash-chain journal, and native "
                        "process timeline prove save restoration"
                    ),
                    details=save_details,
                )
            )

    if manifest.exact_step_replay is not None:
        pass
    elif (
        manifest.manifest_version == LEGACY_MANIFEST_VERSION
        or manifest.replay_determinism is None
    ):
        replay_reason = (
            "legacy v3 manifest has no v4 replay-determinism contract"
            if manifest.manifest_version == LEGACY_MANIFEST_VERSION
            else (
                "two independent replays of the same DMO are not "
                "represented by machine-verifiable evidence"
            )
        )
        semantic_reasons.append(replay_reason)
        checks.append(
            VerificationCheck(
                name="replay_determinism",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=replay_reason,
                details={"machine_verifiable_evidence": False},
            )
        )
    elif save_context is None:
        replay_reason = (
            "replay determinism cannot be established without a verified "
            "save transaction"
        )
        semantic_reasons.append(replay_reason)
        checks.append(
            VerificationCheck(
                name="replay_determinism",
                status=VerificationCheckStatus.INCOMPARABLE,
                message=replay_reason,
                details={"machine_verifiable_evidence": False},
            )
        )
    else:
        try:
            replay_details = _verify_replay_determinism_evidence(
                manifest,
                verified_artifacts,
                demo=demo,
                primary_trace=trace,
                primary_tick_map=tick_map,
                primary_video_inspection=video_inspection,
                save_context=save_context,
            )
        except (_ProtocolIncomparable, VideoDecoderUnavailable) as error:
            replay_reason = str(error)
            semantic_reasons.append(replay_reason)
            checks.append(
                VerificationCheck(
                    name="replay_determinism",
                    status=VerificationCheckStatus.INCOMPARABLE,
                    message=replay_reason,
                    details={"machine_verifiable_evidence": False},
                )
            )
        except (
            OSError,
            UnicodeError,
            PcGoldenValidationError,
            PcVideoError,
            RecursionError,
        ) as error:
            return _failure_report(
                case_id=manifest.case_id,
                checks=checks,
                stage="replay_determinism",
                message=(
                    "independent replay pixels, trace semantics, process "
                    "identity, or end state differ"
                ),
                error=error,
                summary=summary,
            )
        else:
            summary["replay_determinism"] = replay_details
            summary["semantic_limits"][
                "deterministic_replay_attested"
            ] = True
            checks.append(
                VerificationCheck(
                    name="replay_determinism",
                    status=VerificationCheckStatus.PASS,
                    message=(
                        "two or more independent native replays match at "
                        "every tick in pixels, trace semantics, and end state"
                    ),
                    details=replay_details,
                )
            )
    reasons.extend(semantic_reasons)
    if semantic_reasons and final_status is ComparisonStatus.PASS:
        final_status = ComparisonStatus.INCOMPARABLE
    return PcGoldenVerificationReport(
        status=final_status,
        case_id=manifest.case_id,
        checks=tuple(checks),
        reasons=tuple(reasons),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verification of a PC golden manifest, artifacts, "
            "trace, calibration, PopCap DMO, and full video stream."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--case-root",
        type=Path,
        help="Artifact root; defaults to the manifest directory.",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        help=(
            "Read-only original game installation used to resolve the "
            "declared level, curve, gun, executable, and data hashes."
        ),
    )
    parser.add_argument(
        "--require-certifying-dmo-provenance",
        action="store_true",
        help=(
            "require a raw retail recording, non-mutating acquisition report, "
            "and exact duplicate-startup-read normalization receipt"
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_pc_golden_case(
        args.manifest,
        case_root=args.case_root,
        original_root=args.original_root,
        require_certifying_dmo_provenance=(
            args.require_certifying_dmo_provenance
        ),
    )
    print(report.to_json(indent=None if args.compact else 2))
    return report.exit_code


__all__ = [
    "REPORT_SCHEMA",
    "REPORT_VERSION",
    "PcGoldenVerificationReport",
    "VerificationCheck",
    "VerificationCheckStatus",
    "build_parser",
    "main",
    "verify_pc_golden_case",
]


if __name__ == "__main__":
    raise SystemExit(main())
