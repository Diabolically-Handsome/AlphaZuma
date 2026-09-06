"""Scope-specific, fail-closed simulator-fidelity gate.

The project has many useful PC diagnostics, but a diagnostic ``PASS`` is not
the same thing as an unmodified retail-game oracle.  This module keeps those
evidence classes separate and evaluates a fixed policy that cannot be weakened
by omitting requirements from a suite file.

Version 2 separates four evidence lanes: authentic retail sources, per-
mechanism simulator differentials, stochastic-distribution calibration, and
the state-actor interface contract.  Cross-process pixel/RNG identity remains
valuable evidence, but is no longer confused with physical fidelity.  A
separate training gate binds any actual large run to an exact recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from zuma_rl.pc_ambient_rng_boundary import (
    AMBIENT_RNG_BOUNDARY_CLASSIFICATION,
    AMBIENT_RNG_BOUNDARY_SCHEMA,
    AMBIENT_RNG_BOUNDARY_VERSION,
)
from zuma_rl.pc_golden import (
    EXACT_STEP_MANIFEST_VERSION,
    MANIFEST_VERSION,
    ComparisonStatus,
    PcGoldenManifest,
    PcGoldenValidationError,
    pc_golden_native_source_fingerprint,
)
from zuma_rl.pc_mtrand_reconciliation import (
    MTRAND_RECONCILIATION_PROOF_CLASSIFICATION,
    MTRAND_RECONCILIATION_PROOF_SCHEMA,
    MTRAND_RECONCILIATION_PROOF_VERSION,
    MTRandReconciliationProofError,
    reconciliation_transcript_sha256,
    verify_mtrand_reconciliation_proof_binding,
)
from zuma_rl.verify_pc_golden import verify_pc_golden_case

SUITE_SCHEMA = "zuma-rl.fidelity-gate-suite"
SUITE_VERSION = 1
REPORT_SCHEMA = "zuma-rl.fidelity-gate-report"
REPORT_VERSION = 2
AUDIT_SCHEMA = "zuma-rl.fidelity-audit"
AUDIT_VERSION = 1
AUDIT_REPORT_VERSIONS = frozenset({1, 2})

MAX_SUITE_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_COUNT = 256
MAX_JSON_DEPTH = 128

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_FEATURE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_EXACT_MTRAND_POLICY = "exact_no_reconciliation"
_LEGACY_EXTERNAL_MTRAND_POLICY = (
    "conditional_external_visual_draw_reconciliation_v1"
)
_BOUND_EXTERNAL_MTRAND_POLICY = (
    "conditional_external_mtrand_reconciliation_v2"
)
_MAX_UNPROVEN_MTRAND_DRAWS_PER_TICK = 64
_MAX_INDEPENDENTLY_PROVEN_MTRAND_DRAWS_PER_TICK = 2048
_BOUND_EXTERNAL_MTRAND_CLASSIFICATION = (
    "unclassified_until_bound_call_trace_proof"
)
_MTRAND_STATE_BINDING = "sha256_of_624_words_plus_index_le_u32"
_MTRAND_RECONCILIATION_PROOF_SCHEMA = MTRAND_RECONCILIATION_PROOF_SCHEMA

_SOURCE_BOUND_PC_SOURCE_TRANSPORT = (
    "single_retail_process_source_bound_full_state_exact_step_source"
)
_SOURCE_BOUND_REPLAY_SCHEMA = (
    "zuma-rl.pc-source-bound-original-command-replay-binding"
)
_SOURCE_BOUND_REPLAY_VERSIONS = frozenset({1, 2})
_SOURCE_BOUND_NATURAL_RECORDING_CLASSIFICATIONS = frozenset(
    {
        "source_bound_original_natural_retail_recording",
        "source_bound_original_natural_retail_outcome",
    }
)
_PC_GOLDEN_AUTHENTICATION_BASIS = (
    "verified_pc_golden_recording_provenance"
)
_SAME_DMO_PC_GOLDEN_AUTHENTICATION_BASIS = (
    "same_dmo_verified_pc_golden_recording_provenance"
)
_SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS = (
    "verified_source_bound_natural_recording_provenance"
)


class FidelityGateValidationError(ValueError):
    """The suite or one of its content-addressed inputs is malformed."""


class FidelityGateStatus(str, Enum):
    """Top-level gate outcome."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


class EvidenceKind(str, Enum):
    """Trust and semantic class of one suite entry."""

    PC_GOLDEN = "pc_golden"
    PC_SOURCE = "pc_source"
    SIMULATOR_DIFF = "simulator_diff"
    DISTRIBUTION_AUDIT = "distribution_audit"
    AUDIT = "audit"
    DIAGNOSTIC = "diagnostic"


_SIMULATOR_REPORT_SCHEMAS = MappingProxyType(
    {
        "zuma-rl.pc-gameplay-simulator-diff": frozenset({1, 2, 3, 4, 5}),
        "zuma-rl.pc-fruit-expiry-simulator-diff": frozenset({1}),
        "zuma-rl.pc-fruit-scheduler-simulator-diff": frozenset({1}),
        "zuma-rl.pc-merge-simulator-diff": frozenset({1}),
        "zuma-rl.pc-loss-simulator-diff": frozenset({1, 2}),
        "zuma-rl.pc-natural-win-simulator-diff": frozenset({1, 2}),
        "zuma-rl.pc-tunnel-collision-simulator-diff": frozenset({1}),
        "zuma-rl.pc-clear-simulator-transition": frozenset({1}),
        "zuma-rl.pc-powerup-lifecycle-core-diff": frozenset({1}),
        "zuma-rl.pc-powerup-spawn-core-diff": frozenset({1}),
    }
)

_DIAGNOSTIC_REPORT_SCHEMAS = MappingProxyType(
    {
        "zuma-rl.pc-clear-sequence-verification": frozenset({1}),
        "zuma-rl.pc-click-cadence-verification": frozenset({1}),
        "zuma-rl.pc-loss-sequence-verification": frozenset({1}),
        "zuma-rl.pc-natural-win-sequence": frozenset({1}),
        "zuma-rl.pc-memory-trajectory-analysis": frozenset({1}),
        "zuma-rl.pc-pending-rng-verification": frozenset({1}),
        "zuma-rl.pc-powerup-lifecycle-verification": frozenset({1}),
        "zuma-rl.pc-powerup-spawn-verification": frozenset({1}),
        "zuma-rl.pc-powerup-trigger-verification": frozenset({1}),
        "zuma-rl.pc-rng-trajectory-analysis": frozenset({1, 2}),
    }
)


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FidelityGateValidationError(f"{name} must be an integer")
    if value < minimum:
        raise FidelityGateValidationError(
            f"{name} must be at least {minimum}"
        )
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FidelityGateValidationError(
            f"{name} must be a non-empty string"
        )
    return value


def _identifier(value: Any, name: str) -> str:
    result = _nonempty(value, name)
    if _IDENTIFIER_PATTERN.fullmatch(result) is None:
        raise FidelityGateValidationError(
            f"{name} must use lowercase letters, digits, '.', '_' or '-'"
        )
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise FidelityGateValidationError(
            f"{name} must be 'sha256:' plus 64 lowercase hex digits"
        )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FidelityGateValidationError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise FidelityGateValidationError(f"{name} keys must be strings")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FidelityGateValidationError(f"{name} must be a JSON array")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise FidelityGateValidationError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    unknown = set(value).difference(required, optional)
    if unknown:
        raise FidelityGateValidationError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FidelityGateValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise FidelityGateValidationError(
        f"forbidden non-finite JSON constant: {value}"
    )


def _validate_json_value(
    value: Any,
    name: str,
    *,
    depth: int = 0,
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise FidelityGateValidationError(
            f"{name} exceeds maximum JSON nesting depth"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FidelityGateValidationError(f"{name} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FidelityGateValidationError(
                    f"{name} keys must be strings"
                )
            _validate_json_value(
                item,
                f"{name}.{key}",
                depth=depth + 1,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                f"{name}[{index}]",
                depth=depth + 1,
            )
        return
    raise FidelityGateValidationError(f"{name} is not JSON-safe")


def _loads_strict(text: str, name: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except FidelityGateValidationError:
        raise
    except (RecursionError, json.JSONDecodeError) as error:
        raise FidelityGateValidationError(f"invalid {name} JSON") from error
    _validate_json_value(value, name)
    return value


def _read_json(path: Path, name: str, *, maximum_bytes: int) -> Any:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise FidelityGateValidationError(
            f"{name} is missing or unreadable"
        ) from error
    if size > maximum_bytes:
        raise FidelityGateValidationError(
            f"{name} exceeds the {maximum_bytes}-byte limit"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FidelityGateValidationError(
            f"{name} is not readable UTF-8"
        ) from error
    return _loads_strict(text, name)


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as error:
        raise FidelityGateValidationError(
            "evidence is missing or unreadable"
        ) from error
    return f"sha256:{hasher.hexdigest()}"


def _relative_path(value: Any, name: str) -> PurePosixPath:
    raw = _nonempty(value, name)
    if "\\" in raw:
        raise FidelityGateValidationError(
            f"{name} must use relative POSIX separators"
        )
    result = PurePosixPath(raw)
    if (
        result.is_absolute()
        or not result.parts
        or any(part in {"", ".", ".."} for part in result.parts)
    ):
        raise FidelityGateValidationError(
            f"{name} must be a normalized relative POSIX path"
        )
    return result


def _resolve_evidence_path(root: Path, relative: PurePosixPath) -> Path:
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FidelityGateValidationError(
            "evidence path does not exist"
        ) from error
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise FidelityGateValidationError(
            "evidence path escapes the suite root"
        ) from error
    if not resolved.is_file():
        raise FidelityGateValidationError(
            "evidence path must identify a regular file"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    """One content-addressed evidence input."""

    id: str
    kind: EvidenceKind
    path: PurePosixPath
    sha256: str
    features: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceSpec":
        data = _mapping(value, "evidence entry")
        _check_keys(
            data,
            required={"id", "kind", "path", "sha256", "features"},
            name="evidence entry",
        )
        try:
            kind = EvidenceKind(data["kind"])
        except (TypeError, ValueError) as error:
            allowed = ", ".join(item.value for item in EvidenceKind)
            raise FidelityGateValidationError(
                f"evidence.kind must be one of: {allowed}"
            ) from error
        feature_values = _sequence(data["features"], "evidence.features")
        features: list[str] = []
        for index, feature in enumerate(feature_values):
            normalized = _nonempty(
                feature,
                f"evidence.features[{index}]",
            )
            if _FEATURE_PATTERN.fullmatch(normalized) is None:
                raise FidelityGateValidationError(
                    "evidence feature names must be lowercase snake_case"
                )
            features.append(normalized)
        if not features:
            raise FidelityGateValidationError(
                "evidence.features must not be empty"
            )
        if features != sorted(set(features)):
            raise FidelityGateValidationError(
                "evidence.features must be unique and sorted"
            )
        return cls(
            id=_identifier(data["id"], "evidence.id"),
            kind=kind,
            path=_relative_path(data["path"], "evidence.path"),
            sha256=_sha256(data["sha256"], "evidence.sha256"),
            features=tuple(features),
        )


@dataclass(frozen=True, slots=True)
class FidelitySuite:
    """Strict suite input; requirements come from its named fixed policy."""

    policy: str
    evidence: tuple[EvidenceSpec, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FidelitySuite":
        data = _mapping(value, "suite")
        _check_keys(
            data,
            required={"schema", "version", "policy", "evidence"},
            name="suite",
        )
        if data["schema"] != SUITE_SCHEMA:
            raise FidelityGateValidationError(
                f"unsupported suite schema: {data['schema']!r}"
            )
        version = _strict_int(data["version"], "suite.version")
        if version != SUITE_VERSION:
            raise FidelityGateValidationError(
                f"unsupported suite version: {version}"
            )
        policy = _identifier(data["policy"], "suite.policy")
        if policy not in GATE_POLICIES:
            raise FidelityGateValidationError(
                f"unsupported fidelity policy: {policy!r}"
            )
        raw_evidence = _sequence(data["evidence"], "suite.evidence")
        if len(raw_evidence) > MAX_EVIDENCE_COUNT:
            raise FidelityGateValidationError(
                f"suite.evidence exceeds {MAX_EVIDENCE_COUNT} entries"
            )
        evidence = tuple(
            EvidenceSpec.from_dict(
                _mapping(item, f"suite.evidence[{index}]")
            )
            for index, item in enumerate(raw_evidence)
        )
        ids = [item.id for item in evidence]
        if ids != sorted(ids):
            raise FidelityGateValidationError(
                "suite.evidence must be sorted by id"
            )
        if len(ids) != len(set(ids)):
            raise FidelityGateValidationError(
                "suite evidence ids must be unique"
            )
        paths = [item.path.as_posix() for item in evidence]
        if len(paths) != len(set(paths)):
            raise FidelityGateValidationError(
                "suite evidence paths must be unique"
            )
        return cls(policy=policy, evidence=evidence)

    @classmethod
    def read_json(cls, path: str | Path) -> "FidelitySuite":
        source = Path(path)
        return cls.from_dict(
            _mapping(
                _read_json(
                    source,
                    "fidelity suite",
                    maximum_bytes=MAX_SUITE_BYTES,
                ),
                "suite",
            )
        )


@dataclass(frozen=True, slots=True)
class GateRequirement:
    """Evidence counts needed for one behavior."""

    feature: str
    lane: str = "dynamics"
    pc_golden: int = 0
    pc_source: int = 0
    original_source: int = 0
    simulator_diff: int = 0
    distribution_audit: int = 0
    audit: int = 0
    minimum_ticks: int = 0


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Immutable approval policy for one narrowly defined training scope."""

    id: str
    environment_id: str
    levels: tuple[str, ...]
    hard: bool
    profile_mode: str
    observation_mode: str
    curve_count: int
    minimum_pc_golden_cases: int
    minimum_simulator_diff_cases: int
    minimum_random_seeds: int
    requirements: tuple[GateRequirement, ...]
    generation: int = 1
    transfer_authorizing: bool = False
    minimum_original_source_cases: int = 0
    minimum_distribution_audits: int = 0
    compatible_audit_policies: tuple[str, ...] = ()
    compatible_distribution_policies: tuple[str, ...] = ()


_JUNGLE2_REQUIREMENTS = tuple(
    GateRequirement(feature, pc_golden=1, simulator_diff=1)
    for feature in (
        "back_insertion",
        "front_insertion",
        "gap_shot",
        "input_cadence",
        "match3",
        "match4",
        "natural_loss",
        "natural_win",
        "powerup_proximity_bomb",
        "powerup_reverse",
        "powerup_slow",
        "powerup_spawn",
        "projectile_collision",
        "rng_pending",
        "rng_rejection",
        "rollback_chain",
        "shot_release",
        "swap",
        "tunnel_collision",
        "zuma_transition",
    )
) + (
    GateRequirement(
        "deterministic_replay",
        pc_golden=1,
    ),
    GateRequirement(
        "long_horizon_drift",
        pc_golden=1,
        simulator_diff=1,
        minimum_ticks=500,
    ),
    GateRequirement(
        "actor_no_hidden_state",
        audit=1,
    ),
    GateRequirement(
        "actor_visual_derivability",
        pc_golden=1,
        audit=1,
    ),
)

_JUNGLE2_V2_REQUIREMENTS = tuple(
    GateRequirement(
        feature,
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    )
    for feature in (
        "back_insertion",
        "front_insertion",
        "gap_shot",
        "input_cadence",
        "match3",
        "match4",
        "natural_loss",
        "natural_win",
        "powerup_proximity_bomb",
        "powerup_reverse",
        "powerup_slow",
        "powerup_spawn",
        "projectile_collision",
        "rng_pending",
        "rng_rejection",
        "rollback_chain",
        "shot_release",
        "swap",
        "tunnel_collision",
        "zuma_transition",
    )
) + (
    GateRequirement(
        "long_horizon_drift",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
        minimum_ticks=500,
    ),
    GateRequirement(
        "startup_actor_distribution",
        lane="distribution",
        distribution_audit=1,
    ),
    GateRequirement(
        "actor_no_hidden_state",
        lane="interface",
        audit=1,
    ),
)

_JUNGLE2_V3_REQUIREMENTS = _JUNGLE2_V2_REQUIREMENTS + (
    GateRequirement(
        "fruit_asset_geometry",
        lane="dynamics",
        audit=1,
    ),
    GateRequirement(
        "fruit_visual_oscillator",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
        minimum_ticks=500,
    ),
    GateRequirement(
        "fruit_scheduler_spawn",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_expiry",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_projectile_collision",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_powerup_collision",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_collection_score",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_collection_animation",
        lane="dynamics",
        original_source=1,
        simulator_diff=1,
    ),
    GateRequirement(
        "fruit_actor_observation",
        lane="interface",
        audit=1,
    ),
)

# The installed retail Jungle2 CURV has positive weights only for power-up
# types 0 (proximity bomb) and 3 (reverse):
# ``((100, ...), (0, ...), (0, ...), (100, ...), ...)``.  Requiring a
# naturally sourced slow power-up on this level made v3 impossible to open.
# Keep v3 immutable for historical reproduction and correct the transfer
# policy in a new generation.
_JUNGLE2_V4_REQUIREMENTS = tuple(
    requirement
    for requirement in _JUNGLE2_V3_REQUIREMENTS
    if requirement.feature != "powerup_slow"
)

GATE_POLICIES: Mapping[str, GatePolicy] = MappingProxyType(
    {
        "original-transfer-jungle2-v1": GatePolicy(
            id="original-transfer-jungle2-v1",
            environment_id="ZumaRevenge-v0",
            levels=("Jungle2",),
            hard=False,
            profile_mode="tutorials_completed",
            observation_mode="actor",
            curve_count=1,
            minimum_pc_golden_cases=8,
            minimum_simulator_diff_cases=8,
            minimum_random_seeds=3,
            requirements=_JUNGLE2_REQUIREMENTS,
            generation=1,
            transfer_authorizing=False,
            compatible_audit_policies=("original-transfer-jungle2-v1",),
        ),
        "original-transfer-jungle2-v2": GatePolicy(
            id="original-transfer-jungle2-v2",
            environment_id="ZumaRevenge-v0",
            levels=("Jungle2",),
            hard=False,
            profile_mode="tutorials_completed",
            observation_mode="actor",
            curve_count=1,
            minimum_pc_golden_cases=0,
            minimum_simulator_diff_cases=8,
            minimum_random_seeds=3,
            requirements=_JUNGLE2_V2_REQUIREMENTS,
            generation=2,
            transfer_authorizing=False,
            minimum_original_source_cases=8,
            minimum_distribution_audits=1,
            compatible_audit_policies=(
                "original-transfer-jungle2-v1",
                "original-transfer-jungle2-v2",
            ),
        ),
        "original-transfer-jungle2-v3": GatePolicy(
            id="original-transfer-jungle2-v3",
            environment_id="ZumaRevenge-v0",
            levels=("Jungle2",),
            hard=False,
            profile_mode="tutorials_completed",
            observation_mode="actor",
            curve_count=1,
            minimum_pc_golden_cases=0,
            minimum_simulator_diff_cases=8,
            minimum_random_seeds=3,
            requirements=_JUNGLE2_V3_REQUIREMENTS,
            generation=3,
            transfer_authorizing=True,
            minimum_original_source_cases=8,
            minimum_distribution_audits=1,
            compatible_audit_policies=(
                "original-transfer-jungle2-v2",
                "original-transfer-jungle2-v3",
            ),
            compatible_distribution_policies=(
                "original-transfer-jungle2-v2",
                "original-transfer-jungle2-v3",
            ),
        ),
        "original-transfer-jungle2-v4": GatePolicy(
            id="original-transfer-jungle2-v4",
            environment_id="ZumaRevenge-v0",
            levels=("Jungle2",),
            hard=False,
            profile_mode="tutorials_completed",
            observation_mode="actor",
            curve_count=1,
            minimum_pc_golden_cases=0,
            minimum_simulator_diff_cases=8,
            minimum_random_seeds=3,
            requirements=_JUNGLE2_V4_REQUIREMENTS,
            generation=4,
            transfer_authorizing=True,
            minimum_original_source_cases=8,
            minimum_distribution_audits=1,
            compatible_audit_policies=(
                "original-transfer-jungle2-v2",
                "original-transfer-jungle2-v3",
                "original-transfer-jungle2-v4",
            ),
            compatible_distribution_policies=(
                "original-transfer-jungle2-v2",
                "original-transfer-jungle2-v3",
                "original-transfer-jungle2-v4",
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class _AcceptedEvidence:
    spec: EvidenceSpec
    accepted: bool
    schema: str
    version: int
    status: str
    level_id: str | None
    hard: bool | None
    profile_mode: str | None
    random_seed: int | None
    tick_count: int
    source_fingerprint: str
    authorized_features: tuple[str, ...]
    unsupported_features: tuple[str, ...]
    source_dmo_sha256: str | None = None
    original_source_authenticated: bool = False
    original_source_authentication_basis: str | None = None
    supporting_source_fingerprints: tuple[str, ...] = ()
    supporting_source_features: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "kind": self.spec.kind.value,
            "schema": self.schema,
            "version": self.version,
            "status": self.status,
            "accepted": self.accepted,
            "features": list(self.spec.features),
            "declared_features": list(self.spec.features),
            "authorized_features": list(self.authorized_features),
            "unsupported_features": list(self.unsupported_features),
            "supporting_source_fingerprints": list(
                self.supporting_source_fingerprints
            ),
            "supporting_source_features": [
                {
                    "source_fingerprint": source,
                    "authorized_features": list(features),
                }
                for source, features in self.supporting_source_features
            ],
            "level_id": self.level_id,
            "hard": self.hard,
            "profile_mode": self.profile_mode,
            "random_seed": self.random_seed,
            "tick_count": self.tick_count,
            "source_fingerprint": self.source_fingerprint,
            "source_dmo_sha256": self.source_dmo_sha256,
            "original_source_authenticated": (
                self.original_source_authenticated
            ),
            "original_source_authentication_basis": (
                self.original_source_authentication_basis
            ),
            "reason": self.reason,
        }


def _pc_source_self_authentication_basis(
    report: Mapping[str, Any],
) -> str | None:
    """Return the independently verified provenance class, if present.

    Generic single-process replay proves that retail consumed a DMO, but it
    does not prove where those DMO bytes came from.  The source-bound v3
    verifier is stronger: before returning PASS it recomputes the raw natural
    recording, permitted DMO normalization, zero-write recording controls,
    original executable identity, and exact host restoration.  That class can
    therefore authenticate its own original source without requiring a second
    process to enter the same asynchronous RNG branch.
    """

    replay = report.get("strict_command_replay")
    if not isinstance(replay, Mapping):
        return None
    if (
        report.get("status") == "PASS"
        and report.get("transport") == _SOURCE_BOUND_PC_SOURCE_TRANSPORT
        and report.get("cross_process_exact_replay_required") is False
        and replay.get("schema") == _SOURCE_BOUND_REPLAY_SCHEMA
        and replay.get("version") in _SOURCE_BOUND_REPLAY_VERSIONS
        and replay.get("status") == "PASS"
        and replay.get("classification")
        in _SOURCE_BOUND_NATURAL_RECORDING_CLASSIFICATIONS
    ):
        return _SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS
    return None


def _authenticate_original_sources(
    evidence: Sequence[_AcceptedEvidence],
) -> tuple[_AcceptedEvidence, ...]:
    """Authenticate original sources without conflating provenance and RNG.

    PC Golden remains intrinsically certifying.  A generic PC source still
    needs an accepted same-DMO PC Golden because replay alone cannot establish
    the DMO's origin.  A source-bound natural-recording source is different:
    its live verifier already recomputes the raw zero-write retail recording
    and the sole permitted transport normalization, so demanding a second
    process with identical pixels would test asynchronous RNG scheduling, not
    original-source authenticity.
    """

    certifying_dmo_sha256 = {
        item.source_dmo_sha256
        for item in evidence
        if item.accepted
        and item.spec.kind is EvidenceKind.PC_GOLDEN
        and item.source_dmo_sha256 is not None
    }
    authenticated: list[_AcceptedEvidence] = []
    for item in evidence:
        basis: str | None = None
        if item.accepted and item.spec.kind is EvidenceKind.PC_GOLDEN:
            basis = _PC_GOLDEN_AUTHENTICATION_BASIS
        elif item.accepted and item.spec.kind is EvidenceKind.PC_SOURCE:
            if (
                item.original_source_authentication_basis
                == _SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS
            ):
                basis = item.original_source_authentication_basis
            elif (
                item.source_dmo_sha256 is not None
                and item.source_dmo_sha256 in certifying_dmo_sha256
            ):
                basis = _SAME_DMO_PC_GOLDEN_AUTHENTICATION_BASIS
        authenticated.append(
            replace(
                item,
                original_source_authenticated=basis is not None,
                original_source_authentication_basis=basis,
            )
        )
    return tuple(authenticated)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _nonnegative_int(value: Any) -> int | None:
    result = _optional_int(value)
    if result is None or result < 0:
        return None
    return result


def _positive_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _mapping_rows(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    rows = tuple(value)
    if any(not isinstance(row, Mapping) for row in rows):
        return None
    return rows


def _status_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else None


def _pc_golden_authorized_features(verification: Any) -> frozenset[str]:
    """Derive PC feature coverage from the independent verifier output.

    Suite labels are requests, not proof.  A feature is available only when a
    verifier stage exposes the corresponding machine-recomputed semantics.
    """

    if (
        verification is None
        or verification.status is not ComparisonStatus.PASS
    ):
        return frozenset()
    summary = getattr(verification, "summary", None)
    if not isinstance(summary, Mapping):
        return frozenset()

    available: set[str] = set()
    semantic_limits = summary.get("semantic_limits")
    if not isinstance(semantic_limits, Mapping):
        semantic_limits = {}
    replay = summary.get("replay_determinism")
    if (
        semantic_limits.get("deterministic_replay_attested") is True
        and isinstance(replay, Mapping)
        and replay.get("machine_verifiable_evidence") is True
        and _nonnegative_int(replay.get("run_count")) is not None
        and int(replay["run_count"]) >= 2
        and _nonnegative_int(replay.get("native_tick_count")) is not None
        and int(replay["native_tick_count"]) > 0
        and replay.get("full_viewport_rgb24_per_tick_matched") is True
        and replay.get("normalized_trace_matched") is True
        and replay.get("dmo_binding_matched") is True
        and replay.get("independent_processes") is True
        and replay.get("end_state_roots_matched") is True
    ):
        available.add("deterministic_replay")
        if int(replay["native_tick_count"]) >= 500:
            available.add("long_horizon_drift")

    input_check = None
    checks = getattr(verification, "checks", ())
    if (
        isinstance(checks, Sequence)
        and not isinstance(checks, (str, bytes))
    ):
        input_check = next(
            (
                check
                for check in checks
                if getattr(check, "name", None) == "input_sequence"
            ),
            None,
        )
    input_details = getattr(input_check, "details", None)
    if (
        summary.get("input_sequence_status") == "PASS"
        and _status_value(getattr(input_check, "status", None)) == "PASS"
        and isinstance(input_details, Mapping)
        and input_details.get("mode") == "explicit_dmo_binding"
    ):
        trace_count = _nonnegative_int(
            input_details.get("trace_input_count")
        )
        dmo_count = _nonnegative_int(input_details.get("dmo_input_count"))
        if (
            trace_count is not None
            and trace_count > 0
            and dmo_count == trace_count
        ):
            available.add("input_cadence")

    measurement = summary.get("measurement_provenance")
    if (
        semantic_limits.get("memory_measurement_provenance_attested") is True
        and isinstance(measurement, Mapping)
        and measurement.get("machine_verifiable_evidence") is True
        and _nonnegative_int(measurement.get("fired_ball_id")) is not None
        and _nonnegative_int(measurement.get("fired_color_id")) is not None
        and _positive_finite_number(measurement.get("fired_speed"))
        and measurement.get("gameplay_viewport_pixels_matched") is True
    ):
        bindings = _mapping_rows(measurement.get("video_bindings"))
        if (
            bindings is not None
            and len(bindings) >= 2
            and all(
                _nonnegative_int(row.get("framework_update")) is not None
                and _nonnegative_int(row.get("native_tick")) is not None
                and _nonnegative_int(row.get("video_frame_sequence"))
                is not None
                and _nonnegative_int(row.get("interior_mismatch_count")) == 0
                for row in bindings
            )
        ):
            available.add("shot_release")
    return frozenset(available)


def _input_results_prove_cadence(report: Mapping[str, Any]) -> bool:
    rows = _mapping_rows(report.get("input_results"))
    if not rows:
        return False
    previous_sequence = -1
    previous_update = -1
    accepted_nonidle = False
    for row in rows:
        sequence = _nonnegative_int(row.get("sequence"))
        update = _nonnegative_int(row.get("framework_update"))
        kind = row.get("kind")
        accepted = row.get("accepted")
        if (
            sequence is None
            or update is None
            or sequence <= previous_sequence
            or update < previous_update
            or not isinstance(kind, str)
            or not kind
            or (accepted is not None and not isinstance(accepted, bool))
        ):
            return False
        previous_sequence = sequence
        previous_update = update
        if accepted is True and kind != "idle":
            accepted_nonidle = True
    return accepted_nonidle


def _pass_ticks(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = _mapping_rows(report.get("ticks"))
    if not rows or any(row.get("status") != "PASS" for row in rows):
        return ()
    return rows


def _event_count(tick: Mapping[str, Any], name: str) -> int | None:
    events = tick.get("events")
    if not isinstance(events, Mapping):
        return None
    return _nonnegative_int(events.get(name))


def _source_feature_proofs(
    report: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    raw_features = report.get("source_authorized_features")
    raw_proofs = _mapping_rows(report.get("source_feature_proofs"))
    if (
        isinstance(raw_features, (str, bytes))
        or not isinstance(raw_features, Sequence)
        or raw_proofs is None
        or any(not isinstance(item, str) for item in raw_features)
        or list(raw_features) != sorted(set(raw_features))
    ):
        return {}
    parsed: dict[str, Mapping[str, Any]] = {}
    for proof in raw_proofs:
        feature = proof.get("feature")
        if (
            not isinstance(feature, str)
            or feature not in raw_features
            or feature in parsed
            or proof.get("status") != "PASS"
        ):
            return {}
        parsed[feature] = proof
    if set(parsed) != set(raw_features):
        return {}
    return parsed


def _swap_proof_has_exact_simulator_outcome(
    report: Mapping[str, Any],
    proof: Mapping[str, Any],
    by_update: Mapping[int | None, Mapping[str, Any]],
) -> bool:
    """Cross-check a retail swap proof against exact simulator tick state."""

    update = _nonnegative_int(proof.get("framework_update"))
    current_before = _nonnegative_int(proof.get("current_ball_id_before"))
    next_before = _nonnegative_int(proof.get("next_ball_id_before"))
    current_after = _nonnegative_int(proof.get("current_ball_id_after"))
    next_after = _nonnegative_int(proof.get("next_ball_id_after"))
    if (
        update is None
        or update == 0
        or None in {
            current_before,
            next_before,
            current_after,
            next_after,
        }
        or current_before == next_before
        or current_after != next_before
        or next_after != current_before
        or proof.get("qrand_unchanged") is not True
        or proof.get("thread_crt_rand_unchanged") is not True
        or proof.get("score_and_curve_lists_unchanged") is not True
    ):
        return False

    previous = by_update.get(update - 1)
    tick = by_update.get(update)
    if previous is None or tick is None:
        return False

    inputs = _mapping_rows(report.get("input_results"))
    if inputs is None or not any(
        row.get("framework_update") == update - 1
        and row.get("kind") == "mouse_button"
        and row.get("accepted") is True
        for row in inputs
    ):
        return False

    def _exact_integer_pair(
        row: Mapping[str, Any],
        pc_name: str,
        simulator_name: str,
    ) -> int | None:
        pc_value = _nonnegative_int(row.get(pc_name))
        simulator_value = _nonnegative_int(row.get(simulator_name))
        return pc_value if pc_value is not None and simulator_value == pc_value else None

    for row in (previous, tick):
        latent = row.get("latent_mismatches")
        if (
            row.get("active_identity_match") is not True
            or row.get("active_color_match") is not True
            or row.get("shooter_observed_match") is not True
            or row.get("fired_identity_match") is not True
            or row.get("staging_identity_match") is not True
            or not isinstance(latent, Mapping)
            or latent
            or _exact_integer_pair(
                row, "active_count_pc", "active_count_simulator"
            )
            is None
            or _exact_integer_pair(
                row, "fired_count_pc", "fired_count_simulator"
            )
            != 0
            or _exact_integer_pair(
                row, "staging_count_pc", "staging_count_simulator"
            )
            != 0
            or row.get("qrand_observed_match") is not True
            or row.get("thread_crt_rand_observed_match") is not True
            or row.get("global_mtrand_observed_match") is not True
        ):
            return False

    before_current_id = _nonnegative_int(
        previous.get("current_ball_id_pc")
    )
    before_next_id = _nonnegative_int(previous.get("next_ball_id_pc"))
    after_current_id = _nonnegative_int(tick.get("current_ball_id_pc"))
    after_next_id = _nonnegative_int(tick.get("next_ball_id_pc"))
    if (
        before_current_id != current_before
        or before_next_id != next_before
        or after_current_id != current_after
        or after_next_id != next_after
    ):
        return False

    before_current_color = _exact_integer_pair(
        previous,
        "current_color_id_pc",
        "current_color_id_simulator",
    )
    before_next_color = _exact_integer_pair(
        previous,
        "next_color_id_pc",
        "next_color_id_simulator",
    )
    after_current_color = _exact_integer_pair(
        tick,
        "current_color_id_pc",
        "current_color_id_simulator",
    )
    after_next_color = _exact_integer_pair(
        tick,
        "next_color_id_pc",
        "next_color_id_simulator",
    )
    if (
        None in {
            before_current_color,
            before_next_color,
            after_current_color,
            after_next_color,
        }
        or before_current_color == before_next_color
        or after_current_color != before_next_color
        or after_next_color != before_current_color
    ):
        return False

    for name in (
        "fired",
        "hits",
        "inserted",
        "matches",
        "balls_exploded",
        "balls_removed",
        "score_delta",
    ):
        if _event_count(tick, name) != 0:
            return False

    exact_state_pairs = (
        ("score_pc", "score_simulator"),
        ("qrand_update_count_pc", "qrand_update_count_simulator"),
        ("qrand_selected_index_pc", "qrand_selected_index_simulator"),
        ("thread_crt_rand_state_pc", "thread_crt_rand_state_simulator"),
        ("global_mtrand_index_pc", "global_mtrand_index_simulator"),
    )
    for pc_name, simulator_name in exact_state_pairs:
        before = _exact_integer_pair(previous, pc_name, simulator_name)
        after = _exact_integer_pair(tick, pc_name, simulator_name)
        if before is None or after != before:
            return False
    return True


def _exact_v5_gameplay_tick(
    report: Mapping[str, Any],
    tick: Mapping[str, Any],
    *,
    require_free_projectile: bool,
) -> bool:
    """Require every state family used by a v5 mechanism transition."""

    if report.get("version") != 5 or tick.get("status") != "PASS":
        return False
    mismatch_fields = (
        "latent_mismatches",
        "fired_latent_mismatches",
        "staging_latent_mismatches",
        "curve_state_mismatches",
        "fruit_state_mismatches",
    )
    if any(
        not isinstance(tick.get(field), Mapping) or bool(tick.get(field))
        for field in mismatch_fields
    ):
        return False

    def exact_count(pc_name: str, simulator_name: str) -> int | None:
        pc_count = _nonnegative_int(tick.get(pc_name))
        simulator_count = _nonnegative_int(tick.get(simulator_name))
        return pc_count if pc_count is not None and simulator_count == pc_count else None

    fired_count = exact_count("fired_count_pc", "fired_count_simulator")
    if (
        exact_count("active_count_pc", "active_count_simulator") is None
        or fired_count is None
        or (require_free_projectile and fired_count < 1)
        or exact_count("staging_count_pc", "staging_count_simulator") is None
        or tick.get("active_identity_match") is not True
        or tick.get("active_color_match") is not True
        or tick.get("pending_colors_match") is not True
        or tick.get("fired_identity_match") is not True
        or tick.get("staging_identity_match") is not True
        or tick.get("shooter_observed_match") is not True
        or tick.get("qrand_observed_match") is not True
        or tick.get("thread_crt_rand_observed_match") is not True
        or tick.get("global_mtrand_observed_match") is not True
        or tick.get("gameplay_state_matched_before_global_rng") is not True
        or tick.get("score_pc") != tick.get("score_simulator")
        or tick.get("score_delta_pc") != tick.get("score_delta_simulator")
    ):
        return False
    pending_pc = tick.get("pending_colors_pc")
    pending_simulator = tick.get("pending_colors_simulator")
    if (
        isinstance(pending_pc, (str, bytes))
        or not isinstance(pending_pc, Sequence)
        or isinstance(pending_simulator, (str, bytes))
        or not isinstance(pending_simulator, Sequence)
        or list(pending_pc) != list(pending_simulator)
        or any(_nonnegative_int(color) is None for color in pending_pc)
    ):
        return False

    tolerance_pairs = (
        ("maximum_position_error_px", "position_tolerance_px"),
        ("maximum_waypoint_error", "waypoint_tolerance"),
        ("fired_position_error_px", "position_tolerance_px"),
        ("fired_waypoint_error", "waypoint_tolerance"),
        ("fired_progress_error", "progress_tolerance"),
        ("staging_position_error_px", "position_tolerance_px"),
        ("staging_progress_error", "progress_tolerance"),
    )
    for error_name, tolerance_name in tolerance_pairs:
        error = tick.get(error_name)
        tolerance = report.get(tolerance_name)
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or float(error) < 0.0
            or not _positive_finite_number(tolerance)
            or float(error) > float(tolerance)
        ):
            return False
    return True


def _gap_shot_proof_has_exact_simulator_outcome(
    report: Mapping[str, Any],
    proof: Mapping[str, Any],
    by_update: Mapping[int | None, Mapping[str, Any]],
) -> bool:
    start = _nonnegative_int(proof.get("from_update"))
    update = _nonnegative_int(proof.get("framework_update"))
    projectile_id = _nonnegative_int(proof.get("projectile_ball_id"))
    curve_index = _nonnegative_int(proof.get("curve_index"))
    before_point = _nonnegative_int(proof.get("curve_point_before"))
    after_point = _nonnegative_int(proof.get("curve_point_after"))
    gap = proof.get("new_gap_entry")
    simulations = _mapping_rows(proof.get("simulations"))
    if (
        start is None
        or update != start + 1
        or projectile_id in {None, 0}
        or curve_index is None
        or before_point is None
        or after_point is None
        or after_point <= 0
        or after_point == before_point
        or proof.get("projectile_remained_free") is not True
        or not _positive_finite_number(proof.get("projectile_radius"))
        or not isinstance(gap, Mapping)
        or simulations is None
        or len(simulations) != 2
        or report.get("initial_free_projectile_state_restored") is not True
    ):
        return False
    gap_distance = _nonnegative_int(gap.get("gap_distance"))
    boundary_id = _nonnegative_int(gap.get("boundary_ball_id"))
    if (
        gap.get("curve_index") != curve_index
        or gap_distance in {None, 0}
        or boundary_id in {None, 0}
        or any(
            simulation.get("final_curve_point") != after_point
            or _mapping_rows(simulation.get("new_entries")) != (gap,)
            for simulation in simulations
        )
    ):
        return False
    previous = by_update.get(start)
    tick = by_update.get(update)
    if previous is None or tick is None:
        return False
    for row in (previous, tick):
        if not _exact_v5_gameplay_tick(
            report,
            row,
            require_free_projectile=True,
        ):
            return False
    return (
        _event_count(tick, "hits") == 0
        and _nonnegative_int(previous.get("fired_count_pc"))
        == _nonnegative_int(tick.get("fired_count_pc"))
    )


def _rng_rejection_proof_has_exact_simulator_outcome(
    report: Mapping[str, Any],
    proof: Mapping[str, Any],
    by_update: Mapping[int | None, Mapping[str, Any]],
) -> bool:
    start = _nonnegative_int(proof.get("from_update"))
    update = _nonnegative_int(proof.get("framework_update"))
    previous_color = _nonnegative_int(proof.get("previous_color_id"))
    generated_color = _nonnegative_int(proof.get("generated_color_id"))
    generated_id = _nonnegative_int(proof.get("generated_ball_id"))
    rejection_count = _nonnegative_int(proof.get("rejection_count"))
    draw_count = _nonnegative_int(proof.get("mtrand_draw_count"))
    rejected_outputs = proof.get("rejected_candidate_outputs")
    rejected_colors = proof.get("rejected_candidate_colors")
    outputs = proof.get("mtrand_outputs")
    sequence_values = (rejected_outputs, rejected_colors, outputs)
    if (
        start is None
        or update != start + 1
        or previous_color is None
        or generated_color is None
        or generated_color == previous_color
        or generated_id in {None, 0}
        or rejection_count in {None, 0}
        or draw_count is None
        or any(
            isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            for value in sequence_values
        )
        or proof.get("accepted_candidate_color") != generated_color
        or proof.get("mtrand_state_exact") is not True
        or proof.get("neighbouring_ambient_draw_count") != 0
        or proof.get("ambient_before_output") is not None
        or proof.get("ambient_after_output") is not None
        or proof.get("rng_sequence_layout")
        != "repeat,candidate_rejection_loop,ball_visual_frame"
        or report.get("global_mtrand_state_restored") is not True
    ):
        return False
    assert isinstance(rejected_outputs, Sequence)
    assert isinstance(rejected_colors, Sequence)
    assert isinstance(outputs, Sequence)
    if (
        len(rejected_outputs) != rejection_count
        or len(rejected_colors) != rejection_count
        or len(outputs) != draw_count
        or draw_count != rejection_count + 3
        or any(_nonnegative_int(value) is None for value in outputs)
        or any(color != previous_color for color in rejected_colors)
        or list(outputs)
        != [
            proof.get("repeat_roll_output"),
            *rejected_outputs,
            proof.get("accepted_candidate_output"),
            proof.get("ball_visual_frame_output"),
        ]
    ):
        return False
    previous = by_update.get(start)
    tick = by_update.get(update)
    if previous is None or tick is None:
        return False
    for row in (previous, tick):
        if not _exact_v5_gameplay_tick(
            report,
            row,
            require_free_projectile=False,
        ):
            return False
    pending_before = previous.get("pending_colors_pc")
    pending_after = tick.get("pending_colors_pc")
    before_index = _nonnegative_int(previous.get("global_mtrand_index_pc"))
    after_index = _nonnegative_int(tick.get("global_mtrand_index_pc"))
    if (
        isinstance(pending_before, (str, bytes))
        or not isinstance(pending_before, Sequence)
        or isinstance(pending_after, (str, bytes))
        or not isinstance(pending_after, Sequence)
        or list(pending_after) != [*pending_before, generated_color]
        or before_index is None
        or after_index is None
        or before_index > 624
        or after_index > 624
        or (after_index - before_index) % 624 != draw_count % 624
        or previous.get("active_count_pc") != tick.get("active_count_pc")
    ):
        return False
    return all(
        _event_count(tick, name) == 0
        for name in (
            "fired",
            "hits",
            "inserted",
            "matches",
            "balls_exploded",
            "balls_removed",
            "score_delta",
        )
    )


def _powerup_proof_has_exact_simulator_outcome(
    report: Mapping[str, Any],
    proof: Mapping[str, Any],
    by_update: Mapping[int | None, Mapping[str, Any]],
    *,
    feature: str,
    powerup_type: int,
) -> bool:
    """Bind a recomputed retail power-up proof to exact simulator ticks."""

    update = _nonnegative_int(proof.get("framework_update"))
    native_game_time = _nonnegative_int(proof.get("native_game_time"))
    curve_index = _nonnegative_int(proof.get("curve_index"))
    trigger_ball_id = _nonnegative_int(proof.get("trigger_ball_id"))
    trigger_color_id = _nonnegative_int(proof.get("trigger_color_id"))
    score_delta = _nonnegative_int(proof.get("score_delta"))
    raw_ids = proof.get("newly_exploding_ball_ids")
    if (
        proof.get("feature") != feature
        or proof.get("powerup_type") != powerup_type
        or proof.get("manager_transition_exact") is not True
        or update is None
        or native_game_time is None
        or curve_index is None
        or curve_index != _nonnegative_int(report.get("curve_index", 0))
        or trigger_ball_id in {None, 0}
        or trigger_color_id is None
        or trigger_color_id >= 6
        or score_delta in {None, 0}
        or isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_ids, Sequence)
    ):
        return False
    exploding_ids = tuple(_nonnegative_int(value) for value in raw_ids)
    if (
        any(value in {None, 0} for value in exploding_ids)
        or list(exploding_ids) != sorted(set(exploding_ids))
        or trigger_ball_id not in exploding_ids
    ):
        return False

    effect = proof.get("effect_verification")
    target = effect.get("target") if isinstance(effect, Mapping) else None
    trigger = effect.get("trigger") if isinstance(effect, Mapping) else None
    trajectory = (
        effect.get("trajectory") if isinstance(effect, Mapping) else None
    )
    last_seen = (
        effect.get("last_seen_update_by_exploding_ball")
        if isinstance(effect, Mapping)
        else None
    )
    if (
        not isinstance(effect, Mapping)
        or effect.get("schema")
        != "zuma-rl.pc-powerup-trigger-verification"
        or effect.get("version") != 1
        or effect.get("status") != "PASS"
        or not isinstance(target, Mapping)
        or not isinstance(trigger, Mapping)
        or not isinstance(trajectory, Mapping)
        or not isinstance(last_seen, Mapping)
    ):
        return False
    trigger_ids = trigger.get("newly_exploding_ball_ids")
    if (
        isinstance(trigger_ids, (str, bytes))
        or not isinstance(trigger_ids, Sequence)
    ):
        return False
    effect_start = _nonnegative_int(trajectory.get("start_update"))
    effect_end = _nonnegative_int(trajectory.get("end_update"))
    if (
        effect_start != update - 2
        or effect_end != update + 2
        or trajectory.get("tick_count") != 5
        or effect.get("trigger_flag_persists_through_update") != effect_end
        or set(last_seen) != {str(value) for value in exploding_ids}
        or any(
            _nonnegative_int(last_seen.get(str(value))) != effect_end
            for value in exploding_ids
        )
        or target.get("curve_index") != curve_index
        or target.get("ball_id") != trigger_ball_id
        or target.get("color_id") != trigger_color_id
        or target.get("powerup_type") != powerup_type
        or trigger.get("update") != update
        or trigger.get("native_game_time") != native_game_time
        or list(trigger_ids) != list(exploding_ids)
        or trigger.get("score_delta") != score_delta
    ):
        return False

    score_before = _nonnegative_int(trigger.get("score_before"))
    score_after = _nonnegative_int(trigger.get("score_after"))
    counter_before = _nonnegative_int(trigger.get("counter_before"))
    counter_after = _nonnegative_int(trigger.get("counter_after"))
    cooldown_before = _optional_int(trigger.get("cooldown_before"))
    cooldown_after = _nonnegative_int(trigger.get("cooldown_after"))
    active_before = _nonnegative_int(
        trigger.get("active_color_count_before")
    )
    active_after = _nonnegative_int(
        trigger.get("active_color_count_after")
    )
    lifetime_before = _nonnegative_int(trigger.get("target_lifetime_before"))
    lifetime_after = _nonnegative_int(trigger.get("target_lifetime_after"))
    if (
        score_before is None
        or score_after is None
        or score_after - score_before != score_delta
        or counter_before is None
        or counter_after != counter_before + 1
        or cooldown_before is None
        or cooldown_after != native_game_time
        or cooldown_before >= cooldown_after
        or active_before in {None, 0}
        or active_after != active_before - 1
        or lifetime_before in {None, 0}
        or lifetime_after != lifetime_before - 1
        or trigger.get("target_explode_frame_at_trigger") != 1
    ):
        return False

    tick = by_update.get(update)
    previous = by_update.get(update - 1)
    if (
        tick is None
        or previous is None
        or not _exact_v5_gameplay_tick(
            report,
            tick,
            require_free_projectile=False,
        )
        or not _exact_v5_gameplay_tick(
            report,
            previous,
            require_free_projectile=False,
        )
        or previous.get("score_pc") != score_before
        or previous.get("score_simulator") != score_before
        or tick.get("score_pc") != score_after
        or tick.get("score_simulator") != score_after
        or tick.get("score_delta_pc") != score_delta
        or tick.get("score_delta_simulator") != score_delta
        or _event_count(tick, "powerups_triggered") != 1
        or (_event_count(tick, "matches") or 0) < 1
        or _event_count(tick, "balls_exploded") != len(exploding_ids)
        or _event_count(tick, "score_delta") != score_delta
    ):
        return False

    def finite_number(value: Any) -> float | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    if feature == "powerup_proximity_bomb":
        geometry = effect.get("bomb_geometry")
        spatial_ids = (
            geometry.get("spatially_selected_ball_ids")
            if isinstance(geometry, Mapping)
            else None
        )
        if (
            not isinstance(geometry, Mapping)
            or isinstance(spatial_ids, (str, bytes))
            or not isinstance(spatial_ids, Sequence)
            or effect.get("reverse") is not None
            or effect.get("slow") is not None
            or proof.get("slow_physical_movement") is not None
            or geometry.get("collision_pad_per_ball") != 56
            or geometry.get("retail_radius_for_18px_balls") != 148
            or geometry.get("strict_less_than_threshold") is not True
            or list(spatial_ids) != list(exploding_ids)
        ):
            return False
        direct = geometry.get("direct_match_ball_ids")
        if (
            isinstance(direct, (str, bytes))
            or not isinstance(direct, Sequence)
            or len(direct) < 3
            or any(_nonnegative_int(value) in {None, 0} for value in direct)
            or len(set(int(value) for value in direct)) != len(direct)
            or trigger_ball_id not in direct
            or not set(direct).issubset(exploding_ids)
        ):
            return False
        farthest = geometry.get("farthest_selected")
        rejected = geometry.get("nearest_rejected")
        if not isinstance(farthest, Mapping) or not isinstance(
            rejected, Mapping
        ):
            return False
        farthest_distance = finite_number(farthest.get("distance"))
        farthest_threshold = finite_number(farthest.get("threshold"))
        rejected_distance = finite_number(rejected.get("distance"))
        rejected_threshold = finite_number(rejected.get("threshold"))
        return bool(
            _nonnegative_int(farthest.get("ball_id")) in exploding_ids
            and _nonnegative_int(rejected.get("ball_id"))
            not in exploding_ids
            and farthest_distance is not None
            and farthest_threshold == 148.0
            and farthest_distance < farthest_threshold
            and rejected_distance is not None
            and rejected_threshold == farthest_threshold
            and rejected_distance >= rejected_threshold
        )

    if feature == "powerup_reverse":
        reverse = effect.get("reverse")
        movement = (
            reverse.get("movement") if isinstance(reverse, Mapping) else None
        )
        if (
            not isinstance(reverse, Mapping)
            or not isinstance(movement, Mapping)
            or effect.get("bomb_geometry") is not None
            or effect.get("slow") is not None
            or proof.get("slow_physical_movement") is not None
            or reverse.get("ticks_at_trigger") != 300
            or reverse.get("ticks_next_update") != 299
            or movement.get("sample_phase")
            != "two_complete_post_trigger_intervals"
            or _nonnegative_int(movement.get("reference_ball_id"))
            in {None, 0, *exploding_ids}
        ):
            return False
        speed = finite_number(reverse.get("speed"))
        distance_per_tick = finite_number(movement.get("distance_per_tick"))
        trigger_distance = finite_number(movement.get("trigger_distance"))
        next_distance = finite_number(movement.get("next_distance"))
        post_next_distance = finite_number(
            movement.get("post_next_distance")
        )
        if (
            speed != 1.0
            or distance_per_tick != speed
            or trigger_distance is None
            or next_distance is None
            or post_next_distance is None
            or not math.isclose(
                trigger_distance - next_distance,
                speed,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            or not math.isclose(
                next_distance - post_next_distance,
                speed,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            return False
        return all(
            (row := by_update.get(candidate)) is not None
            and _exact_v5_gameplay_tick(
                report,
                row,
                require_free_projectile=False,
            )
            for candidate in range(update, update + 3)
        )

    if feature == "powerup_slow":
        slow = effect.get("slow")
        movement = proof.get("slow_physical_movement")
        if (
            not isinstance(slow, Mapping)
            or not isinstance(movement, Mapping)
            or effect.get("bomb_geometry") is not None
            or effect.get("reverse") is not None
            or slow.get("ticks_at_trigger") != 800
            or slow.get("ticks_next_update") != 799
            or movement.get("strictly_slower") is not True
            or _nonnegative_int(movement.get("reference_ball_id"))
            in {None, 0, *exploding_ids}
        ):
            return False
        movement_before = finite_number(
            movement.get("movement_before_trigger")
        )
        movement_at = finite_number(movement.get("movement_at_trigger"))
        movement_after = finite_number(
            movement.get("movement_after_trigger")
        )
        if (
            movement_before is None
            or movement_at is None
            or movement_after is None
            or movement_before <= 0.0
            or movement_at < 0.0
            or movement_after < 0.0
            or movement_at >= movement_before
            or movement_after >= movement_before
        ):
            return False
        return all(
            (row := by_update.get(candidate)) is not None
            and _exact_v5_gameplay_tick(
                report,
                row,
                require_free_projectile=False,
            )
            for candidate in range(update - 2, update + 2)
        )
    return False


def _mechanism_features_from_exact_ticks(
    report: Mapping[str, Any],
    ticks: tuple[Mapping[str, Any], ...],
) -> frozenset[str]:
    """Require source-derived proofs and matching simulator tick outcomes."""

    proofs = _source_feature_proofs(report)
    if not proofs:
        return frozenset()
    by_update = {
        _nonnegative_int(row.get("framework_update")): row for row in ticks
    }
    if None in by_update or len(by_update) != len(ticks):
        return frozenset()
    available: set[str] = set()
    for feature, expected_count in (("match3", 3), ("match4", 4)):
        proof = proofs.get(feature)
        if proof is None:
            continue
        update = _nonnegative_int(proof.get("framework_update"))
        score_delta = _nonnegative_int(proof.get("score_delta"))
        ids = proof.get("exploding_ball_ids")
        ids_valid = (
            not isinstance(ids, (str, bytes))
            and isinstance(ids, Sequence)
            and len(ids) == expected_count
            and all(_nonnegative_int(value) is not None for value in ids)
        )
        if ids_valid:
            ids_valid = len(set(ids)) == expected_count
        tick = by_update.get(update)
        previous = by_update.get(None if update is None else update - 1)
        if (
            update is None
            or score_delta is None
            or score_delta <= 0
            or not ids_valid
            or tick is None
            or tick.get("active_identity_match") is not True
            or tick.get("active_color_match") is not True
            or (_event_count(tick, "matches") or 0) < 1
            or _event_count(tick, "balls_exploded") != expected_count
            or _event_count(tick, "score_delta") != score_delta
        ):
            continue
        pc_delta = _nonnegative_int(tick.get("score_delta_pc"))
        simulator_delta = _nonnegative_int(
            tick.get("score_delta_simulator")
        )
        if pc_delta is not None or simulator_delta is not None:
            if pc_delta != score_delta or simulator_delta != score_delta:
                continue
        else:
            if previous is None:
                continue
            current_pc = _nonnegative_int(tick.get("score_pc"))
            current_sim = _nonnegative_int(tick.get("score_simulator"))
            previous_pc = _nonnegative_int(previous.get("score_pc"))
            previous_sim = _nonnegative_int(
                previous.get("score_simulator")
            )
            if (
                None in {current_pc, current_sim, previous_pc, previous_sim}
                or current_pc != current_sim
                or previous_pc != previous_sim
                or current_pc - previous_pc != score_delta
            ):
                continue
        latent = tick.get("latent_mismatches")
        if isinstance(latent, Mapping) and latent:
            continue
        available.add(feature)

    rollback = proofs.get("rollback_chain")
    if rollback is not None:
        start = _nonnegative_int(rollback.get("start_update"))
        stop = _nonnegative_int(rollback.get("stop_update"))
        score_delta = _nonnegative_int(rollback.get("score_delta"))
        stop_tick = by_update.get(stop)
        stop_pc_delta = (
            _nonnegative_int(stop_tick.get("score_delta_pc"))
            if stop_tick is not None
            else None
        )
        stop_simulator_delta = (
            _nonnegative_int(stop_tick.get("score_delta_simulator"))
            if stop_tick is not None
            else None
        )
        covered_updates = sorted(
            update for update in by_update if update is not None
        )
        rollback_ticks = (
            [by_update[update] for update in range(start, stop + 1)]
            if start is not None
            and stop is not None
            and all(update in by_update for update in range(start, stop + 1))
            else []
        )
        if (
            start is not None
            and stop is not None
            and start < stop
            and score_delta is not None
            and score_delta > 0
            and stop_tick is not None
            and (_event_count(stop_tick, "matches") or 0) >= 1
            and (_event_count(stop_tick, "balls_exploded") or 0) >= 3
            and _event_count(stop_tick, "score_delta") == score_delta
            and stop_pc_delta == score_delta
            and stop_simulator_delta == score_delta
            and rollback_ticks
            and all(
                tick.get("active_identity_match") is True
                and tick.get("active_color_match") is True
                and isinstance(tick.get("latent_mismatches"), Mapping)
                and not tick["latent_mismatches"]
                for tick in rollback_ticks
            )
            and covered_updates == list(
                range(covered_updates[0], covered_updates[-1] + 1)
            )
        ):
            available.add("rollback_chain")

    gap_shot = proofs.get("gap_shot")
    if gap_shot is not None and _gap_shot_proof_has_exact_simulator_outcome(
        report,
        gap_shot,
        by_update,
    ):
        available.add("gap_shot")

    rng_rejection = proofs.get("rng_rejection")
    if (
        rng_rejection is not None
        and _rng_rejection_proof_has_exact_simulator_outcome(
            report,
            rng_rejection,
            by_update,
        )
    ):
        available.add("rng_rejection")

    for feature, powerup_type in (
        ("powerup_proximity_bomb", 0),
        ("powerup_slow", 1),
        ("powerup_reverse", 3),
    ):
        powerup = proofs.get(feature)
        if (
            powerup is not None
            and _powerup_proof_has_exact_simulator_outcome(
                report,
                powerup,
                by_update,
                feature=feature,
                powerup_type=powerup_type,
            )
        ):
            available.add(feature)

    if (
        report.get("version") == 5
        and report.get("fruit_runtime_state_restored") is True
    ):
        def exact_fruit_tick(update: int | None) -> Mapping[str, Any] | None:
            tick = by_update.get(update)
            if (
                update is None
                or tick is None
                or tick.get("status") != "PASS"
                or not isinstance(
                    tick.get("fruit_state_mismatches"), Mapping
                )
                or bool(tick.get("fruit_state_mismatches"))
                or tick.get("score_pc") != tick.get("score_simulator")
                or tick.get("score_delta_pc")
                != tick.get("score_delta_simulator")
            ):
                return None
            return tick

        projectile = proofs.get("fruit_projectile_collision")
        projectile_update = (
            _nonnegative_int(projectile.get("framework_update"))
            if projectile is not None
            else None
        )
        projectile_tick = exact_fruit_tick(projectile_update)
        projectile_distance = (
            projectile.get("distance_squared")
            if projectile is not None
            else None
        )
        projectile_radius = (
            projectile.get("collision_radius_squared")
            if projectile is not None
            else None
        )
        if (
            projectile is not None
            and projectile_tick is not None
            and projectile.get("collision_cause") == "free_projectile"
            and projectile.get("active_pointer_transition")
            == "same_nonzero_to_collecting"
            and _nonnegative_int(projectile.get("projectile_ball_id"))
            is not None
            and isinstance(projectile_distance, (int, float))
            and not isinstance(projectile_distance, bool)
            and isinstance(projectile_radius, (int, float))
            and not isinstance(projectile_radius, bool)
            and math.isfinite(float(projectile_distance))
            and math.isfinite(float(projectile_radius))
            and 0.0 <= float(projectile_distance)
            <= float(projectile_radius)
            and _event_count(projectile_tick, "fruits_collected") == 1
            and _event_count(projectile_tick, "powerups_triggered") == 0
            and _event_count(projectile_tick, "hits") == 0
            and projectile_tick.get("fired_identity_match") is True
            and isinstance(
                projectile_tick.get("fired_latent_mismatches"), Mapping
            )
            and not projectile_tick["fired_latent_mismatches"]
        ):
            available.add("fruit_projectile_collision")

        score = proofs.get("fruit_collection_score")
        score_update = (
            _nonnegative_int(score.get("framework_update"))
            if score is not None
            else None
        )
        score_tick = exact_fruit_tick(score_update)
        if score is not None and score_tick is not None:
            score_before = _nonnegative_int(score.get("score_before"))
            score_after = _nonnegative_int(score.get("score_after"))
            score_delta = _nonnegative_int(score.get("score_delta"))
            score_start = _nonnegative_int(
                score.get("score_at_level_start")
            )
            expected_delta = (
                max(500, ((score_before - score_start) // 600) * 100)
                if score_before is not None
                and score_start is not None
                and score_before >= score_start
                else None
            )
            if (
                None
                not in {
                    score_before,
                    score_after,
                    score_delta,
                    score_start,
                }
                and score_after - score_before == score_delta
                and score_delta == expected_delta
                and score.get("tier_divisor") == 600
                and score.get("tier_multiplier") == 100
                and score.get("minimum_points") == 500
                and score.get("isolated_projectile_collection") is True
                and report.get("score_at_level_start") == score_start
                and score_tick.get("score_pc") == score_after
                and score_tick.get("score_delta_pc") == score_delta
                and _event_count(score_tick, "score_delta") == score_delta
                and _event_count(score_tick, "fruits_collected") == 1
                and _event_count(score_tick, "hits") == 0
                and _event_count(score_tick, "matches") == 0
                and _event_count(score_tick, "balls_exploded") == 0
                and _event_count(score_tick, "powerups_triggered") == 0
            ):
                available.add("fruit_collection_score")

        powerup = proofs.get("fruit_powerup_collision")
        powerup_update = (
            _nonnegative_int(powerup.get("framework_update"))
            if powerup is not None
            else None
        )
        powerup_tick = exact_fruit_tick(powerup_update)
        powerup_distance = (
            powerup.get("distance_squared")
            if powerup is not None
            else None
        )
        powerup_radius = (
            powerup.get("strict_radius_squared")
            if powerup is not None
            else None
        )
        if (
            powerup is not None
            and powerup_tick is not None
            and powerup.get("collision_cause") == "proximity_bomb"
            and powerup.get("powerup_type") == 0
            and powerup.get("collision_radius") == 108.0
            and powerup.get("active_pointer_transition")
            == "same_nonzero_to_collecting"
            and powerup.get("linked_powerup_manager_transition_exact")
            is True
            and _nonnegative_int(powerup.get("trigger_ball_id"))
            is not None
            and isinstance(powerup_distance, (int, float))
            and not isinstance(powerup_distance, bool)
            and isinstance(powerup_radius, (int, float))
            and not isinstance(powerup_radius, bool)
            and math.isfinite(float(powerup_distance))
            and math.isfinite(float(powerup_radius))
            and 0.0 <= float(powerup_distance) < float(powerup_radius)
            and _event_count(powerup_tick, "fruits_collected") == 1
            and (_event_count(powerup_tick, "powerups_triggered") or 0)
            >= 1
            and (_event_count(powerup_tick, "balls_exploded") or 0) >= 1
            and powerup_tick.get("active_identity_match") is True
            and isinstance(powerup_tick.get("latent_mismatches"), Mapping)
            and not powerup_tick["latent_mismatches"]
        ):
            available.add("fruit_powerup_collision")

        animation = proofs.get("fruit_collection_animation")
        if animation is not None:
            start = _nonnegative_int(animation.get("start_update"))
            clear = _nonnegative_int(animation.get("clear_update"))
            collection_ticks = _nonnegative_int(
                animation.get("collection_ticks")
            )
            animation_rows = (
                [by_update.get(update) for update in range(start, clear + 1)]
                if start is not None
                and clear is not None
                and collection_ticks is not None
                and clear - start == collection_ticks
                else []
            )
            if (
                animation_rows
                and all(row is not None for row in animation_rows)
                and animation.get("framework_update") == start
                and animation.get("tick_hz") == 100
                and animation.get("active_pointer_clear_offset")
                == collection_ticks
                and animation.get("alpha_step") == -8
                and animation.get("glow_recurrence_exact") is True
                and animation.get("cell_recurrence_exact") is True
                and animation.get("float32_state_retention_exact") is True
                and all(
                    exact_fruit_tick(update) is not None
                    for update in range(start, clear + 1)
                )
                and _event_count(animation_rows[0], "fruits_collected")
                == 1
                and all(
                    _event_count(row, "fruits_collected") == 0
                    and _event_count(row, "fruits_expired") == 0
                    for row in animation_rows[1:]
                )
            ):
                available.add("fruit_collection_animation")

    swap = proofs.get("swap")
    if swap is not None and _swap_proof_has_exact_simulator_outcome(
        report,
        swap,
        by_update,
    ):
        available.add("swap")
    return frozenset(available)


def _gameplay_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    ticks = _pass_ticks(report)
    if not ticks:
        return frozenset()
    available: set[str] = set()

    if _input_results_prove_cadence(report):
        available.add("input_cadence")

    available.update(_mechanism_features_from_exact_ticks(report, ticks))

    if any(
        (_event_count(tick, "fired") or 0) > 0
        and tick.get("fired_identity_match") is True
        and _nonnegative_int(tick.get("fired_count_pc"))
        == _nonnegative_int(tick.get("fired_count_simulator"))
        and _nonnegative_int(tick.get("fired_count_pc")) is not None
        for tick in ticks
    ):
        available.add("shot_release")

    if any(
        (_event_count(tick, "hits") or 0) > 0
        and tick.get("staging_identity_match") is True
        and _nonnegative_int(tick.get("staging_count_pc"))
        == _nonnegative_int(tick.get("staging_count_simulator"))
        and _nonnegative_int(tick.get("staging_count_pc")) is not None
        for tick in ticks
    ):
        available.add("projectile_collision")

    qrand_counts: list[int] = []
    rng_exact = report.get("shooter_rng_state_restored") is True
    for tick in ticks:
        pc_count = _nonnegative_int(tick.get("qrand_update_count_pc"))
        simulator_count = _nonnegative_int(
            tick.get("qrand_update_count_simulator")
        )
        if (
            pc_count is None
            or simulator_count != pc_count
            or tick.get("qrand_observed_match") is not True
            or tick.get("thread_crt_rand_observed_match") is not True
            or tick.get("qrand_selected_index_pc")
            != tick.get("qrand_selected_index_simulator")
            or tick.get("thread_crt_rand_state_pc")
            != tick.get("thread_crt_rand_state_simulator")
        ):
            rng_exact = False
            break
        qrand_counts.append(pc_count)
    if rng_exact and any(
        left != right
        for left, right in zip(qrand_counts, qrand_counts[1:])
    ):
        available.add("rng_pending")

    if (
        len(ticks) >= 500
        and _extract_tick_count(report) >= 500
        and all(
            tick.get("active_identity_match") is True
            and tick.get("active_color_match") is True
            and tick.get("fired_identity_match") is True
            and tick.get("staging_identity_match") is True
            for tick in ticks
        )
    ):
        available.add("long_horizon_drift")
    if (
        report.get("version") in {4, 5}
        and report.get("fruit_runtime_state_restored") is True
        and len(ticks) >= 500
        and all(
            isinstance(tick.get("fruit_state_mismatches"), Mapping)
            and not tick["fruit_state_mismatches"]
            for tick in ticks
        )
    ):
        available.add("fruit_visual_oscillator")
    return frozenset(available)


def _merge_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    ticks = _pass_ticks(report)
    hit_in_front = report.get("initial_hit_in_front")
    pc_update = _nonnegative_int(report.get("pc_insertion_update"))
    simulator_update = _nonnegative_int(
        report.get("simulator_insertion_update")
    )
    if (
        not ticks
        or not isinstance(hit_in_front, bool)
        or pc_update is None
        or simulator_update != pc_update
        or _nonnegative_int(report.get("initial_hit_ball_id")) is None
        or _nonnegative_int(report.get("initial_hit_index")) is None
    ):
        return frozenset()
    insertion = next(
        (
            tick
            for tick in ticks
            if _nonnegative_int(tick.get("framework_update")) == pc_update
            and (_event_count(tick, "inserted") or 0) > 0
            and tick.get("active_identity_match") is True
            and tick.get("staging_identity_match") is True
        ),
        None,
    )
    if insertion is None:
        return frozenset()
    available = {"front_insertion" if hit_in_front else "back_insertion"}
    available.update(_mechanism_features_from_exact_ticks(report, ticks))
    return frozenset(available)


def _powerup_spawn_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    selected_powerup_type = _nonnegative_int(
        report.get("selected_powerup_type")
    )
    if (
        report.get("exact_ball_and_manager_state") is True
        and _nonnegative_int(report.get("selected_ball_id")) is not None
        and _nonnegative_int(report.get("selected_color_id")) is not None
        and selected_powerup_type is not None
        and selected_powerup_type < 14
        and _nonnegative_int(report.get("mtrand_index_before")) is not None
        and _nonnegative_int(report.get("mtrand_index_after")) is not None
    ):
        return frozenset({"powerup_spawn"})
    return frozenset()


def _natural_loss_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    start = _nonnegative_int(report.get("start_update"))
    reset = _nonnegative_int(report.get("reset_update"))
    trigger = _nonnegative_int(report.get("trigger_update"))
    empty = _nonnegative_int(report.get("empty_update"))
    end = _nonnegative_int(report.get("end_update"))
    oracle = report.get("oracle")
    removals = report.get("removal_updates")
    if (
        None in {start, reset, trigger, empty, end}
        or not (start <= reset <= trigger <= empty <= end)
        or not isinstance(oracle, Mapping)
        or oracle.get("schema")
        != "zuma-rl.pc-loss-sequence-verification"
        or oracle.get("status") != "PASS"
        or _nonnegative_int(oracle.get("trigger_update")) != trigger
        or _nonnegative_int(oracle.get("empty_update")) != empty
        or not isinstance(removals, Mapping)
        or not removals
        or (_nonnegative_int(report.get("initial_chain_count")) or 0) <= 0
        or (_nonnegative_int(report.get("ticks_compared")) or 0) <= 0
    ):
        return frozenset()
    parsed_updates: list[int] = []
    for key, ball_ids in removals.items():
        try:
            update = int(key)
        except (TypeError, ValueError):
            return frozenset()
        rows = _mapping_rows(ball_ids)
        if rows is not None:
            return frozenset()
        if (
            update < trigger
            or update > empty
            or isinstance(ball_ids, (str, bytes))
            or not isinstance(ball_ids, Sequence)
            or not ball_ids
            or any(_nonnegative_int(ball_id) is None for ball_id in ball_ids)
        ):
            return frozenset()
        parsed_updates.append(update)
    if empty not in parsed_updates:
        return frozenset()
    return frozenset({"natural_loss"})


def _natural_win_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    version = _nonnegative_int(report.get("version"))
    start = _nonnegative_int(report.get("start_update"))
    empty = _nonnegative_int(report.get("empty_update"))
    formal = _nonnegative_int(report.get("formal_transition_update"))
    end = _nonnegative_int(report.get("end_update"))
    ticks = _nonnegative_int(report.get("ticks_compared"))
    initial_count = _nonnegative_int(report.get("initial_chain_count"))
    removed = _nonnegative_int(report.get("balls_removed"))
    oracle = report.get("oracle")
    scenario = report.get("scenario")
    if (
        version not in {1, 2}
        or None in {start, empty, formal, end, ticks, initial_count, removed}
        or start + 1 != empty
        or empty + 1 != formal
        or end != formal
        or ticks != 2
        or report.get("score_target_achieved") is not True
        or report.get("plan_exhausted") is not True
        or report.get("stop_adding_transplanted") is not True
        or report.get("pc_empty_state_matched") is not True
        or report.get("pc_formal_state_matched") is not True
        or report.get("maximum_waypoint_error") != 0.0
        or not isinstance(oracle, Mapping)
        or oracle.get("schema") != "zuma-rl.pc-natural-win-sequence"
        or oracle.get("version") != 1
        or oracle.get("status") != "PASS"
        or _nonnegative_int(oracle.get("empty_update")) != empty
        or _nonnegative_int(oracle.get("formal_transition_update"))
        != formal
        or not isinstance(scenario, Mapping)
        or scenario.get("shooter_rng_state_restored") is not True
        or scenario.get("global_mtrand_state_restored") is not True
    ):
        return frozenset()
    if version == 1:
        if initial_count <= 0 or removed != initial_count:
            return frozenset()
    else:
        initial_free = _nonnegative_int(
            report.get("initial_free_projectile_count")
        )
        initial_gameplay = _nonnegative_int(
            report.get("initial_gameplay_entity_count")
        )
        gameplay_removed = _nonnegative_int(
            report.get("gameplay_entities_removed")
        )
        terminal_kind = report.get("terminal_object_kind")
        chain_path = (
            terminal_kind == "chain"
            and initial_count > 0
            and initial_free == 0
            and initial_gameplay == initial_count
            and gameplay_removed == initial_gameplay
            and removed == initial_count
        )
        projectile_path = (
            terminal_kind == "free_projectile"
            and initial_count == 0
            and initial_free == 1
            and initial_gameplay == 1
            and gameplay_removed == 1
            and removed == 0
        )
        if (
            not (chain_path or projectile_path)
            or report.get("pc_terminal_chamber_cleared") is not True
        ):
            return frozenset()
    score_target = _nonnegative_int(oracle.get("score_target"))
    score_cross = _nonnegative_int(oracle.get("score_cross_update"))
    feed_exhaustion = _nonnegative_int(
        oracle.get("feed_exhaustion_update")
    )
    if (
        score_target is None
        or score_target <= 0
        or score_cross is None
        or score_cross > empty
    ):
        return frozenset()
    if version == 1:
        if feed_exhaustion is None or feed_exhaustion > empty:
            return frozenset()
    else:
        feed_closed_at_start = oracle.get("feed_closed_at_capture_start")
        if (
            feed_closed_at_start is True
            and oracle.get("feed_exhaustion_update") is not None
        ) or (
            feed_closed_at_start is False
            and (feed_exhaustion is None or feed_exhaustion > empty)
        ) or feed_closed_at_start not in {True, False}:
            return frozenset()
    return frozenset({"natural_win", "zuma_transition"})


def _tunnel_collision_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    """Authorize an exact free-flight transition through a retail tunnel."""

    proofs = _source_feature_proofs(report)
    proof = proofs.get("tunnel_collision")
    source = report.get("source_projectile")
    transition = report.get("simulator_transition")
    scenario = report.get("scenario")
    if (
        set(proofs) != {"tunnel_collision"}
        or proof is None
        or not isinstance(source, Mapping)
        or not isinstance(transition, Mapping)
        or not isinstance(scenario, Mapping)
        or _nonnegative_int(report.get("compared_tick_count")) != 1
    ):
        return frozenset()
    start = _nonnegative_int(report.get("start_update"))
    end = _nonnegative_int(report.get("end_update"))
    overlaps = proof.get("tunnel_overlaps")
    events = transition.get("events")
    rng = transition.get("rng_state_matches")
    active_position_error = transition.get("active_position_error_px")
    active_waypoint_error = transition.get("active_waypoint_error")
    if (
        start is None
        or end != start + 1
        or proof.get("from_update") != start
        or proof.get("framework_update") != end
        or _nonnegative_int(proof.get("projectile_ball_id")) is None
        or proof.get("projectile_ball_id") != source.get("ball_id")
        or proof.get("projectile_color_id") != source.get("color_id")
        or proof.get("projectile_position_before")
        != source.get("position_before")
        or proof.get("projectile_position_after")
        != source.get("position_after")
        or proof.get("projectile_velocity") != source.get("velocity")
        or not isinstance(overlaps, Sequence)
        or isinstance(overlaps, (str, bytes))
        or not overlaps
        or any(not isinstance(item, Mapping) for item in overlaps)
        or proof.get("non_tunnel_overlap_count") != 0
        or proof.get("staging_list_empty") is not True
        or proof.get("projectile_remained_free") is not True
        or source.get("remained_free") is not True
        or source.get("staging_count_before") != 0
        or source.get("staging_count_after") != 0
        or transition.get("projectile_count_after") != 1
        or transition.get("staging_count_after") != 0
        or transition.get("projectile_identity_match") is not True
        or transition.get("projectile_position_error_px") != 0.0
        or transition.get("projectile_waypoint_error") != 0.0
        or transition.get("projectile_progress_error") != 0.0
        or transition.get("active_identity_match") is not True
        or transition.get("active_color_match") is not True
        or not isinstance(active_position_error, (int, float))
        or not math.isfinite(float(active_position_error))
        or float(active_position_error) > 1e-5
        or not isinstance(active_waypoint_error, (int, float))
        or not math.isfinite(float(active_waypoint_error))
        or float(active_waypoint_error) > 1e-6
        or any(
            not isinstance(transition.get(name), Mapping)
            or bool(transition[name])
            for name in (
                "projectile_latent_mismatches",
                "active_latent_mismatches",
                "curve_state_mismatches",
                "fruit_state_mismatches",
            )
        )
        or transition.get("pending_colors_match") is not True
        or transition.get("shooter_state_match") is not True
        or not isinstance(rng, Mapping)
        or rng != {
            "qrand": True,
            "thread_crt_rand": True,
            "global_mtrand": True,
        }
        or transition.get("score_source")
        != transition.get("score_simulator")
        or not isinstance(events, Mapping)
        or any(
            _nonnegative_int(events.get(name)) != 0
            for name in (
                "fired",
                "hits",
                "inserted",
                "matches",
                "balls_exploded",
                "balls_removed",
                "score_delta",
            )
        )
        or transition.get("collision_suppressed") is not True
        or transition.get("exact_native_state_match") is not True
        or scenario.get("shooter_rng_state_restored") is not True
        or scenario.get("global_mtrand_state_restored") is not True
    ):
        return frozenset()
    return frozenset({"tunnel_collision"})


def _fruit_scheduler_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    """Authorize only the source-bound, exact natural spawn transition."""

    proofs = _source_feature_proofs(report)
    proof = proofs.get("fruit_scheduler_spawn")
    transition = report.get("transition")
    mtrand = report.get("mtrand")
    if (
        set(proofs) != {"fruit_scheduler_spawn"}
        or proof is None
        or not isinstance(transition, Mapping)
        or not isinstance(mtrand, Mapping)
        or report.get("fruit_runtime_state_restored") is not True
        or _nonnegative_int(report.get("compared_tick_count")) != 1
    ):
        return frozenset()
    update = _nonnegative_int(proof.get("framework_update"))
    selected = _nonnegative_int(proof.get("selected_point_index"))
    expiry = _nonnegative_int(proof.get("expiry_time"))
    total_draws = _nonnegative_int(proof.get("mtrand_total_draw_count"))
    scheduler_draws = _nonnegative_int(
        proof.get("mtrand_scheduler_draw_count")
    )
    proof_outputs = proof.get("mtrand_outputs")
    report_outputs = mtrand.get("source_outputs")
    phase_mismatches = transition.get(
        "scheduler_phase_fruit_state_mismatches"
    )
    full_mismatches = transition.get(
        "full_tick_fruit_state_mismatches"
    )
    if (
        update is None
        or update != report.get("end_update")
        or report.get("start_update") != update - 1
        or selected is None
        or expiry is None
        or total_draws is None
        or total_draws < 2
        or scheduler_draws != 2
        or proof.get("mtrand_state_exact") is not True
        or proof.get("active_pointer_transition") != "zero_to_nonzero"
        or proof.get("fruit_reset_state_exact") is not True
        or not isinstance(proof_outputs, Sequence)
        or isinstance(proof_outputs, (str, bytes))
        or list(proof_outputs) != report_outputs
        or mtrand.get("source_total_draw_count") != total_draws
        or mtrand.get("scheduler_draw_count") != 2
        or mtrand.get("scheduler_prefix_state_exact") is not True
        or mtrand.get("full_tick_final_state_exact") is not True
        or transition.get("source_before_active") is not False
        or transition.get("source_after_active") is not True
        or transition.get("source_selected_point_index") != selected
        or transition.get("simulator_selected_point_index") != selected
        or transition.get("source_expiry_time") != expiry
        or transition.get("simulator_expiry_time") != expiry
        or not isinstance(phase_mismatches, Mapping)
        or bool(phase_mismatches)
        or not isinstance(full_mismatches, Mapping)
        or bool(full_mismatches)
        or transition.get("scheduler_phase_fruit_chance_draws") != 1
        or transition.get("scheduler_phase_fruits_spawned") != 1
        or transition.get("full_tick_fruit_chance_draws") != 1
        or transition.get("full_tick_fruits_spawned") != 1
    ):
        return frozenset()
    return frozenset({"fruit_scheduler_spawn"})


def _fruit_expiry_authorized_features(
    report: Mapping[str, Any],
) -> frozenset[str]:
    """Authorize only an exact, source-bound natural expiry transition."""

    proofs = _source_feature_proofs(report)
    proof = proofs.get("fruit_expiry")
    transition = report.get("transition")
    mtrand = report.get("mtrand")
    if (
        set(proofs) != {"fruit_expiry"}
        or proof is None
        or not isinstance(transition, Mapping)
        or not isinstance(mtrand, Mapping)
        or report.get("fruit_runtime_state_restored") is not True
        or _nonnegative_int(report.get("compared_tick_count")) != 1
    ):
        return frozenset()
    start = _nonnegative_int(proof.get("start_update"))
    update = _nonnegative_int(proof.get("framework_update"))
    expiry = _nonnegative_int(proof.get("expiry_time"))
    native_before = _nonnegative_int(proof.get("native_game_time_before"))
    native_after = _nonnegative_int(proof.get("native_game_time_after"))
    phase_mismatches = transition.get(
        "scheduler_phase_fruit_state_mismatches"
    )
    full_mismatches = transition.get(
        "full_tick_fruit_state_mismatches"
    )
    if (
        start is None
        or update is None
        or update != start + 1
        or report.get("start_update") != start
        or report.get("end_update") != update
        or expiry is None
        or native_before is None
        or native_after != native_before + 1
        or native_after != expiry
        or proof.get("active_pointer_transition") != "nonzero_to_zero"
        or proof.get("collecting_before") is not False
        or proof.get("collecting_after") is not False
        or proof.get("score_delta") != 0
        or transition.get("source_before_active") is not True
        or transition.get("source_after_active") is not False
        or transition.get("source_collecting_before") is not False
        or transition.get("source_collecting_after") is not False
        or transition.get("source_native_game_time_before") != native_before
        or transition.get("source_native_game_time_after") != native_after
        or transition.get("source_expiry_time") != expiry
        or transition.get("simulator_active_after_scheduler") is not False
        or transition.get("simulator_expiry_time") != expiry
        or transition.get("source_selected_point_index_after") != 0
        or transition.get("source_score_delta") != 0
        or not isinstance(phase_mismatches, Mapping)
        or bool(phase_mismatches)
        or not isinstance(full_mismatches, Mapping)
        or bool(full_mismatches)
        or transition.get("scheduler_phase_fruit_chance_draws") != 0
        or transition.get("scheduler_phase_fruits_spawned") != 0
        or transition.get("scheduler_phase_fruits_expired") != 1
        or transition.get("scheduler_phase_fruits_collected") != 0
        or transition.get("full_tick_fruit_chance_draws") != 0
        or transition.get("full_tick_fruits_spawned") != 0
        or transition.get("full_tick_fruits_expired") != 1
        or transition.get("full_tick_fruits_collected") != 0
        or mtrand.get("authorization_dependency") != "none"
    ):
        return frozenset()
    return frozenset({"fruit_expiry"})


def _simulator_authorized_features(
    report: Mapping[str, Any],
    *,
    schema: str,
) -> frozenset[str]:
    if schema == "zuma-rl.pc-gameplay-simulator-diff":
        return _gameplay_authorized_features(report)
    if schema == "zuma-rl.pc-fruit-expiry-simulator-diff":
        return _fruit_expiry_authorized_features(report)
    if schema == "zuma-rl.pc-fruit-scheduler-simulator-diff":
        return _fruit_scheduler_authorized_features(report)
    if schema == "zuma-rl.pc-merge-simulator-diff":
        return _merge_authorized_features(report)
    if schema == "zuma-rl.pc-powerup-spawn-core-diff":
        return _powerup_spawn_authorized_features(report)
    if schema == "zuma-rl.pc-loss-simulator-diff":
        return _natural_loss_authorized_features(report)
    if schema == "zuma-rl.pc-natural-win-simulator-diff":
        return _natural_win_authorized_features(report)
    if schema == "zuma-rl.pc-tunnel-collision-simulator-diff":
        return _tunnel_collision_authorized_features(report)
    # Lifecycle-only reports do not prove a power-up's physical effect.  The
    # current clear-transition report is generated from an intervened board.
    return frozenset()


def _declared_feature_authorization(
    spec: EvidenceSpec,
    available: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    authorized = tuple(
        feature for feature in spec.features if feature in available
    )
    unsupported = tuple(
        feature for feature in spec.features if feature not in available
    )
    return authorized, unsupported


def _extract_scenario(
    report: Mapping[str, Any],
) -> tuple[str | None, bool | None, str | None]:
    level = report.get("level_id")
    hard = report.get("hard")
    profile = report.get("profile_mode")
    scenario = report.get("scenario")
    if isinstance(scenario, Mapping):
        level = scenario.get("level_id", level)
        hard = scenario.get("hard", hard)
        profile = scenario.get("profile_mode", profile)
    normalized_level = level if isinstance(level, str) else None
    normalized_hard = hard if isinstance(hard, bool) else None
    normalized_profile = profile if isinstance(profile, str) else None
    return normalized_level, normalized_hard, normalized_profile


def _extract_tick_count(report: Mapping[str, Any]) -> int:
    for key in (
        "compared_tick_count",
        "ticks_compared",
        "transition_count",
        "tick_count",
    ):
        value = _optional_int(report.get(key))
        if value is not None and value >= 0:
            return value
    ticks = report.get("ticks")
    if isinstance(ticks, Sequence) and not isinstance(ticks, (str, bytes)):
        return len(ticks)
    start = _optional_int(report.get("start_update"))
    end = _optional_int(report.get("end_update"))
    if start is not None and end is not None and end >= start:
        return end - start
    return 0


def _extract_source_fingerprint(
    report: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    """Prefer the immutable raw trajectory identity over a derived report."""

    candidates: list[Any] = []
    trajectory = report.get("trajectory")
    if isinstance(trajectory, Mapping):
        candidates.extend(
            (
                trajectory.get("artifact_sha256"),
                trajectory.get("sha256"),
            )
        )
    candidates.extend(
        (
            report.get("trajectory_sha256"),
            report.get("source_trajectory_sha256"),
        )
    )
    for candidate in candidates:
        if (
            isinstance(candidate, str)
            and _SHA256_PATTERN.fullmatch(candidate) is not None
        ):
            return candidate
    return fallback


def _gameplay_mtrand_contract_reason(
    report: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
) -> str | None:
    """Validate the report-local half of the shared-MTRand contract.

    Version 1 predates global-state comparison.  Version 2 introduced an
    experimental suffix search but incorrectly labelled every skipped caller
    as rendering-only without binding a native call trace.  Version 3 records
    complete state hashes and can certify only when a separate, content-
    addressed proof classifies every reconciled call.
    """

    version = _optional_int(report.get("version"))
    policy = report.get("global_mtrand_policy")
    reconciliation = report.get("global_mtrand_reconciliation")

    if version in {None, 1}:
        if policy is None and reconciliation is None:
            return None
        return "legacy gameplay differential has an invalid MTRand policy"
    if version not in {2, 3, 4, 5}:
        return "gameplay differential MTRand policy version is unsupported"
    if not isinstance(reconciliation, Mapping):
        return "gameplay differential MTRand reconciliation is missing"

    rows_value = reconciliation.get("rows")
    if (
        isinstance(rows_value, (str, bytes))
        or not isinstance(rows_value, Sequence)
    ):
        return "gameplay differential MTRand reconciliation rows are invalid"
    rows = list(rows_value)
    reconciled_count = _nonnegative_int(
        reconciliation.get("reconciled_tick_count")
    )
    total_draws = _nonnegative_int(
        reconciliation.get("total_reconciled_draw_count")
    )
    maximum_draws = _nonnegative_int(
        reconciliation.get("maximum_draws_per_tick")
    )
    if (
        reconciled_count is None
        or reconciled_count != len(rows)
        or total_draws is None
        or maximum_draws is None
        or not 1
        <= maximum_draws
        <= _MAX_INDEPENDENTLY_PROVEN_MTRAND_DRAWS_PER_TICK
    ):
        return "gameplay differential MTRand reconciliation totals are invalid"

    ticks_value = report.get("ticks")
    if (
        isinstance(ticks_value, (str, bytes))
        or not isinstance(ticks_value, Sequence)
    ):
        return "gameplay differential MTRand tick transcript is missing"
    ticks = [tick for tick in ticks_value if isinstance(tick, Mapping)]
    if len(ticks) != len(ticks_value):
        return "gameplay differential MTRand tick transcript is invalid"

    if policy == _EXACT_MTRAND_POLICY:
        if (
            maximum_draws > _MAX_UNPROVEN_MTRAND_DRAWS_PER_TICK
            or reconciliation.get("enabled") is not False
            or rows
            or total_draws != 0
            or reconciliation.get("classification")
            not in {
                "no_reconciliation",
                # Preserve exact version-2 reports emitted before the label
                # was corrected.  Their zero-row contract is still exact.
                "excluded_rendering_only_shared_rng_consumers",
            }
            or report.get("global_mtrand_state_restored") is not True
            or any(
                tick.get("status") != "PASS"
                or tick.get("global_mtrand_observed_match") is not True
                or (_nonnegative_int(
                    tick.get("global_mtrand_reconciled_draws")
                ) or 0) != 0
                or (_nonnegative_int(
                    tick.get("global_mtrand_leading_reconciled_draws")
                ) or 0) != 0
                for tick in ticks
            )
        ):
            return "exact gameplay MTRand policy has a non-exact transcript"
        return None

    if policy == _LEGACY_EXTERNAL_MTRAND_POLICY:
        return (
            "legacy gameplay MTRand reconciliation is unbound and may "
            "conceal gameplay-relevant RNG consumers"
        )
    if policy != _BOUND_EXTERNAL_MTRAND_POLICY or version not in {3, 4, 5}:
        return "gameplay differential MTRand policy is unknown"
    if (
        reconciliation.get("enabled") is not True
        or reconciliation.get("classification")
        != _BOUND_EXTERNAL_MTRAND_CLASSIFICATION
        or reconciliation.get("state_binding") != _MTRAND_STATE_BINDING
        or not rows
        or report.get("global_mtrand_state_restored") is not True
    ):
        return "bound gameplay MTRand reconciliation header is invalid"

    by_update: dict[int, Mapping[str, Any]] = {}
    for tick in ticks:
        update = _nonnegative_int(tick.get("framework_update"))
        if update is None or update in by_update:
            return "gameplay differential MTRand tick updates are invalid"
        by_update[update] = tick

    seen: set[tuple[int, str]] = set()
    computed_total = 0
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            return "gameplay differential MTRand reconciliation row is invalid"
        update = _nonnegative_int(raw_row.get("framework_update"))
        phase = raw_row.get("phase")
        draw_count = _nonnegative_int(raw_row.get("draw_count"))
        if (
            update is None
            or phase not in {
                "after_gameplay_state_match",
                "before_gameplay_tick",
            }
            or draw_count is None
            or not 1 <= draw_count <= maximum_draws
            or (update, str(phase)) in seen
            or update not in by_update
        ):
            return "gameplay differential MTRand reconciliation row is invalid"
        seen.add((update, str(phase)))
        computed_total += draw_count
        tick = by_update[update]
        mismatch_fields = (
            "latent_mismatches",
            "fired_latent_mismatches",
            "staging_latent_mismatches",
            "curve_state_mismatches",
        ) + (("fruit_state_mismatches",) if version in {4, 5} else ())
        if (
            tick.get("status") != "PASS"
            or tick.get("gameplay_state_matched_before_global_rng") is not True
            or tick.get("global_mtrand_observed_match") is not True
            or tick.get("active_identity_match") is not True
            or tick.get("active_color_match") is not True
            or tick.get("pending_colors_match") is not True
            or tick.get("fired_identity_match") is not True
            or tick.get("staging_identity_match") is not True
            or tick.get("qrand_observed_match") is not True
            or tick.get("thread_crt_rand_observed_match") is not True
            or tick.get("score_pc") != tick.get("score_simulator")
            or tick.get("score_delta_pc")
            != tick.get("score_delta_simulator")
            or any(
                not isinstance(tick.get(field), Mapping)
                or bool(tick.get(field))
                for field in mismatch_fields
            )
        ):
            return "reconciled MTRand tick does not have exact gameplay state"
        required_hashes = (
            "simulator_state_sha256_before",
            "simulator_state_sha256_after",
        )
        if any(
            not isinstance(raw_row.get(field), str)
            or _SHA256_PATTERN.fullmatch(str(raw_row.get(field))) is None
            for field in required_hashes
        ):
            return "reconciled MTRand row lacks complete state hashes"
        if phase == "after_gameplay_state_match":
            if (
                raw_row.get("gameplay_state_matched_before_reconciliation")
                is not True
                or raw_row.get("full_state_reached_exactly") is not True
                or raw_row.get("simulator_state_sha256_after")
                != raw_row.get("pc_state_sha256")
                or _nonnegative_int(
                    tick.get("global_mtrand_reconciled_draws")
                ) != draw_count
                or (_nonnegative_int(
                    tick.get("global_mtrand_leading_reconciled_draws")
                ) or 0) != 0
            ):
                return "post-gameplay MTRand reconciliation row is inconsistent"
        else:
            if (
                raw_row.get("pc_final_state_forward_reachable") is not True
                or raw_row.get(
                    "source_transition_selected_uniquely_by_full_gameplay_state"
                ) is not True
                or _nonnegative_int(
                    raw_row.get("post_gameplay_draw_count")
                ) is None
                or _SHA256_PATTERN.fullmatch(
                    str(raw_row.get("post_gameplay_state_sha256"))
                ) is None
                or _SHA256_PATTERN.fullmatch(
                    str(raw_row.get("pc_final_state_sha256"))
                ) is None
                or _nonnegative_int(
                    tick.get("global_mtrand_leading_reconciled_draws")
                ) != draw_count
            ):
                return "pre-gameplay MTRand reconciliation row is inconsistent"
    if computed_total != total_draws:
        return "gameplay differential MTRand draw total is inconsistent"

    proof = report.get("global_mtrand_reconciliation_proof")
    if (
        not isinstance(proof, Mapping)
        or proof.get("schema") != _MTRAND_RECONCILIATION_PROOF_SCHEMA
        or proof.get("version") != MTRAND_RECONCILIATION_PROOF_VERSION
        or proof.get("status") != "PASS"
        or proof.get("classification")
        != MTRAND_RECONCILIATION_PROOF_CLASSIFICATION
        or not isinstance(proof.get("artifact"), str)
        or _SHA256_PATTERN.fullmatch(
            str(proof.get("artifact_sha256"))
        ) is None
        or _SHA256_PATTERN.fullmatch(
            str(proof.get("reconciliation_transcript_sha256"))
        ) is None
    ):
        return (
            "gameplay MTRand reconciliation lacks an independent bound "
            "call-trace proof"
        )
    if evidence_root is None:
        return (
            "gameplay MTRand reconciliation proof cannot be verified "
            "without the evidence root"
        )
    try:
        proof_relative = _relative_path(
            proof.get("artifact"),
            "gameplay MTRand reconciliation proof artifact",
        )
        proof_path = _resolve_evidence_path(evidence_root, proof_relative)
        if _sha256_path(proof_path) != proof.get("artifact_sha256"):
            return "gameplay MTRand reconciliation proof hash mismatch"
        verified_proof = verify_mtrand_reconciliation_proof_binding(
            proof_path,
            report=report,
            evidence_root=evidence_root,
        )
        binding = verified_proof.get("gameplay_diff_binding")
        reconciliation_summary = verified_proof.get("reconciliation")
        if (
            not isinstance(binding, Mapping)
            or not isinstance(reconciliation_summary, Mapping)
            or binding.get("reconciliation_transcript_sha256")
            != reconciliation_transcript_sha256(report)
            or proof.get("reconciliation_transcript_sha256")
            != binding.get("reconciliation_transcript_sha256")
            or reconciliation_summary.get("reconciled_tick_count")
            != reconciled_count
            or reconciliation_summary.get("reconciled_draw_count")
            != total_draws
        ):
            return (
                "gameplay MTRand reconciliation proof is not bound to "
                "the complete differential transcript"
            )
    except (
        FidelityGateValidationError,
        MTRandReconciliationProofError,
        OSError,
        ValueError,
    ) as error:
        return (
            "gameplay MTRand reconciliation proof failed independent "
            f"verification: {error}"
        )
    return None


def _gameplay_fruit_contract_reason(
    report: Mapping[str, Any],
    *,
    evidence_root: Path | None,
) -> str | None:
    """Require exact fruit rows and, for long windows, isolated replay."""

    version = _optional_int(report.get("version"))
    if version not in {4, 5}:
        return None
    ticks = _mapping_rows(report.get("ticks"))
    if (
        report.get("fruit_runtime_state_restored") is not True
        or not ticks
        or any(
            not isinstance(tick.get("fruit_state_mismatches"), Mapping)
            or bool(tick.get("fruit_state_mismatches"))
            for tick in ticks
        )
    ):
        return "v4/v5 gameplay differential lacks exact retail fruit state"
    source_features = report.get("source_authorized_features")
    requires_visual_proof = (
        version == 4
        or len(ticks) >= 500
        or (
            isinstance(source_features, Sequence)
            and not isinstance(source_features, (str, bytes))
            and "fruit_visual_oscillator" in source_features
        )
    )
    if not requires_visual_proof:
        # A short v5 collision/collection window is fully checked against the
        # exact Board fruit fields in every gameplay tick.  It cannot claim
        # the separate 500-tick oscillator feature below.
        return None
    binding = report.get("fruit_visual_diff")
    if not isinstance(binding, Mapping) or evidence_root is None:
        return "long v4/v5 gameplay differential lacks a bound fruit visual proof"
    try:
        relative = _relative_path(
            binding.get("artifact"),
            "fruit visual proof artifact",
        )
        path = _resolve_evidence_path(evidence_root, relative)
        expected_sha = _sha256(
            binding.get("artifact_sha256"),
            "fruit visual proof SHA-256",
        )
        if _sha256_path(path) != expected_sha:
            return "fruit visual proof hash mismatch"
        proof = _mapping(
            _read_json(
                path,
                "fruit visual proof",
                maximum_bytes=MAX_REPORT_BYTES,
            ),
            "fruit visual proof",
        )
    except FidelityGateValidationError:
        return "fruit visual proof cannot be verified"
    comparison = proof.get("comparison_contract")
    coverage = proof.get("coverage")
    provenance = proof.get("provenance")
    trajectory = report.get("trajectory")
    runtime = report.get("runtime_executable")

    def nested(value: Any, *keys: str) -> Any:
        current = value
        for key in keys:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    if (
        proof.get("schema") != "zuma-rl.pc-fruit-visual-simulator-diff"
        or proof.get("version") not in {1, 2}
        or proof.get("status") != "PASS"
        or proof.get("level_id") != report.get("level_id")
        or proof.get("hard") is not report.get("hard")
        or _nonnegative_int(proof.get("transition_count")) is None
        or proof.get("transition_count", 0) < 500
        or proof.get("exact_transition_count") != proof.get("transition_count")
        or proof.get("mismatch_transition_count") != 0
        or not isinstance(comparison, Mapping)
        or comparison.get("float32_policy")
        != "little-endian IEEE-754 bit exact"
        or not isinstance(coverage, Mapping)
        or not isinstance(coverage.get("vertical_offset_min"), (int, float))
        or not isinstance(coverage.get("vertical_offset_max"), (int, float))
        or float(coverage["vertical_offset_min"]) >= 0.0
        or float(coverage["vertical_offset_max"]) <= 0.0
        or coverage.get("cell_index_min") != 0
        or coverage.get("cell_index_max") != 59
        or not isinstance(provenance, Mapping)
        or not isinstance(trajectory, Mapping)
        or not isinstance(runtime, Mapping)
        or nested(
            provenance,
            "trajectory_index",
            "sha256",
        )
        != trajectory.get("artifact_sha256")
        or nested(
            provenance,
            "runtime_executable",
            "sha256",
        )
        != runtime.get("artifact_sha256")
    ):
        return "fruit visual proof semantics differ from the v4/v5 report"
    return None


def _pc_golden_source_fingerprint(manifest: PcGoldenManifest) -> str:
    """Identify the native capture source rather than its packaging.

    ``evidence_set_fingerprint`` intentionally changes when provenance or
    annotations are added to a v4 case.  It therefore cannot be used to count
    independent PC captures: relabelling or enriching the same two videos
    would otherwise increase the Gate count.  Bind independence to the raw
    input DMO and every deterministic replay video instead.

    The fallback keeps read-only legacy/test doubles diagnosable.  A real v4
    manifest always has ``artifacts`` and ``replay_determinism`` and therefore
    takes the capture-bound path.
    """

    artifacts = getattr(manifest, "artifacts", None)
    replay_determinism = getattr(manifest, "replay_determinism", None)
    exact_step_replay = getattr(manifest, "exact_step_replay", None)
    input_timeline = getattr(manifest, "input_timeline", None)
    if (
        not isinstance(artifacts, Mapping)
        or (
            replay_determinism is None
            and exact_step_replay is None
        )
        or input_timeline is None
    ):
        return manifest.evidence_set_fingerprint

    return pc_golden_native_source_fingerprint(manifest)


def _noncertifying_report_reason(
    report: Mapping[str, Any],
    *,
    schema: str,
    evidence_root: Path | None = None,
    original_root: str | Path | None = None,
) -> str | None:
    """Return why a PASS report is still only a diagnostic.

    Several early PC differential tools intentionally supported shooter-state
    synchronization or injected terminal states.  Those modes are excellent
    debugging oracles, but accepting them here would let a corrected simulator
    hide a naturally occurring divergence.
    """

    if (
        schema == "zuma-rl.pc-loss-simulator-diff"
        and report.get("version") == 2
    ):
        if evidence_root is None or original_root is None:
            return (
                "source-bound natural-loss differential cannot be "
                "recomputed without both evidence and original roots"
            )
        from zuma_rl.pc_loss_sequence import (
            validate_loss_simulator_diff_report,
        )

        loss_reason = validate_loss_simulator_diff_report(
            report,
            evidence_root=evidence_root,
            original_root=original_root,
        )
        if loss_reason is not None:
            return (
                "source-bound natural-loss differential failed live "
                f"recomputation: {loss_reason}"
            )

    if schema == "zuma-rl.pc-gameplay-simulator-diff":
        mtrand_reason = _gameplay_mtrand_contract_reason(
            report,
            evidence_root=evidence_root,
        )
        if mtrand_reason is not None:
            return mtrand_reason
        fruit_reason = _gameplay_fruit_contract_reason(
            report,
            evidence_root=evidence_root,
        )
        if fruit_reason is not None:
            return fruit_reason
        if report.get("synchronize_shooter") is not False:
            return (
                "gameplay differential used or did not rule out shooter "
                "synchronization"
            )
        mismatches = report.get("shooter_mismatch_updates")
        if (
            isinstance(mismatches, Sequence)
            and not isinstance(mismatches, (str, bytes))
            and len(mismatches) > 0
        ):
            return "gameplay differential observed shooter-state mismatches"
        boundary = report.get("scope_boundary")
        if boundary is not None:
            end_update = _optional_int(report.get("end_update"))
            if (
                not isinstance(boundary, Mapping)
                or boundary.get("schema") != AMBIENT_RNG_BOUNDARY_SCHEMA
                or boundary.get("version")
                != AMBIENT_RNG_BOUNDARY_VERSION
                or boundary.get("status") != "PASS"
                or boundary.get("classification")
                != AMBIENT_RNG_BOUNDARY_CLASSIFICATION
                or _SHA256_PATTERN.fullmatch(
                    str(boundary.get("artifact_sha256"))
                )
                is None
                or end_update is None
                or boundary.get("last_included_update") != end_update
                or boundary.get("first_excluded_update")
                != end_update + 1
                or boundary.get("excluded_scope")
                != "shadow_canopy_ambient_visual_rng_consumers"
            ):
                return (
                    "gameplay differential scope-boundary proof is "
                    "malformed or not bound to its final tick"
                )

    stack: list[Any] = [report]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = key.lower()
                if lowered == "classification" and isinstance(item, str):
                    classification = item.lower()
                    if (
                        "diagnostic-not-unmodified" in classification
                        or "synthetic" in classification
                        or "injected" in classification
                    ):
                        return (
                            "report provenance is diagnostic rather than "
                            "unmodified PC evidence"
                        )
                if (
                    (
                        lowered.startswith("injected_")
                        or lowered in {
                            "diagnostic_mutation",
                            "memory_mutation",
                            "synthetic_state",
                        }
                    )
                    and item is not None
                    and item is not False
                    and item != 0
                    and item != ""
                ):
                    return (
                        "simulator differential depends on injected or "
                        "synthetic state"
                    )
                stack.append(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            stack.extend(value)
    return None


def _evaluate_pc_golden(
    spec: EvidenceSpec,
    path: Path,
    *,
    policy: GatePolicy,
    original_root: str | Path | None,
) -> _AcceptedEvidence:
    try:
        manifest = PcGoldenManifest.read_json(path)
    except (OSError, UnicodeError, PcGoldenValidationError) as error:
        raise FidelityGateValidationError(
            "pc_golden evidence is not a valid manifest"
        ) from error
    scenario = manifest.scenario
    tick_count = int(manifest.clock.tick_end) + 1
    reason: str | None = None
    if manifest.manifest_version not in {
        MANIFEST_VERSION,
        EXACT_STEP_MANIFEST_VERSION,
    }:
        reason = (
            "certification requires a current evidence-v4 or formal "
            "exact-step evidence-v5 manifest"
        )
    elif scenario.level_id not in policy.levels:
        reason = "PC Golden level is outside the policy scope"
    elif scenario.hard is not policy.hard:
        reason = "PC Golden difficulty differs from the policy scope"
    elif scenario.profile_mode != policy.profile_mode:
        reason = "PC Golden profile differs from the policy scope"
    elif scenario.mode != "adventure":
        reason = "PC Golden mode differs from the policy scope"
    elif original_root is None:
        reason = "original_root is required for certification"

    verification = None
    if reason is None:
        verification = verify_pc_golden_case(
            path,
            case_root=path.parent,
            original_root=original_root,
            require_certifying_dmo_provenance=True,
        )
        if verification.status is not ComparisonStatus.PASS:
            verifier_reason = (
                verification.reasons[0]
                if getattr(verification, "reasons", ())
                else None
            )
            reason = (
                "PC Golden verifier returned "
                f"{verification.status.value}"
                + (
                    f": {verifier_reason}"
                    if verifier_reason is not None
                    else ""
                )
            )
    available_features = (
        _pc_golden_authorized_features(verification)
        if reason is None
        else frozenset()
    )
    authorized_features, unsupported_features = (
        _declared_feature_authorization(spec, available_features)
        if reason is None
        else ((), ())
    )
    return _AcceptedEvidence(
        spec=spec,
        accepted=reason is None,
        schema="zuma-rl.pc-golden-manifest",
        version=int(manifest.manifest_version),
        status=(
            verification.status.value
            if verification is not None
            else "INCOMPARABLE"
        ),
        level_id=scenario.level_id,
        hard=scenario.hard,
        profile_mode=scenario.profile_mode,
        random_seed=int(manifest.input_timeline.random_seed),
        tick_count=tick_count,
        source_fingerprint=_pc_golden_source_fingerprint(manifest),
        authorized_features=authorized_features,
        unsupported_features=unsupported_features,
        source_dmo_sha256=manifest.artifacts[
            manifest.input_timeline.artifact
        ].sha256,
        supporting_source_fingerprints=(),
        reason=reason,
    )


def _evaluate_pc_source(
    spec: EvidenceSpec,
    path: Path,
    *,
    policy: GatePolicy,
    evidence_root: Path,
    original_root: str | Path | None,
) -> _AcceptedEvidence:
    """Validate one retail trajectory without demanding a twin replay."""

    from zuma_rl.pc_source import (
        PcSourceValidationError,
        verify_pc_source_manifest,
    )

    try:
        report = verify_pc_source_manifest(
            path,
            evidence_root=evidence_root,
            original_root=original_root,
        )
    except (OSError, PcSourceValidationError) as error:
        raise FidelityGateValidationError(
            "pc_source evidence is invalid: " + str(error)
        ) from error
    scope = _mapping(report.get("scope"), "pc_source scope")
    reason: str | None = None
    if scope.get("level_id") not in policy.levels:
        reason = "PC source level is outside the policy scope"
    elif scope.get("hard") is not policy.hard:
        reason = "PC source difficulty differs from the policy scope"
    elif scope.get("profile_mode") != policy.profile_mode:
        reason = "PC source profile differs from the policy scope"
    elif scope.get("mode") != "adventure":
        reason = "PC source mode differs from the policy scope"
    raw_features = _sequence(
        report.get("authorized_features"),
        "pc_source authorized_features",
    )
    if any(not isinstance(item, str) for item in raw_features):
        raise FidelityGateValidationError(
            "pc_source authorized_features are invalid"
        )
    available = frozenset(raw_features)
    authorized_features, unsupported_features = (
        _declared_feature_authorization(spec, available)
        if reason is None
        else ((), ())
    )
    return _AcceptedEvidence(
        spec=spec,
        accepted=reason is None,
        schema=_nonempty(report.get("schema"), "pc_source schema"),
        version=_strict_int(report.get("version"), "pc_source version"),
        status=_nonempty(report.get("status"), "pc_source status"),
        level_id=str(scope["level_id"]),
        hard=bool(scope["hard"]),
        profile_mode=str(scope["profile_mode"]),
        random_seed=_strict_int(
            report.get("random_seed"),
            "pc_source random_seed",
        ),
        tick_count=_strict_int(
            report.get("tick_count"),
            "pc_source tick_count",
            minimum=1,
        ),
        source_fingerprint=_sha256(
            report.get("source_fingerprint"),
            "pc_source source_fingerprint",
        ),
        authorized_features=authorized_features,
        unsupported_features=unsupported_features,
        source_dmo_sha256=_sha256(
            report.get("dmo_sha256"),
            "pc_source dmo_sha256",
        ),
        original_source_authentication_basis=(
            _pc_source_self_authentication_basis(report)
        ),
        reason=reason,
    )


def _evaluate_distribution_audit(
    spec: EvidenceSpec,
    path: Path,
    *,
    policy: GatePolicy,
    evidence_root: Path,
    original_root: str | Path | None,
) -> _AcceptedEvidence:
    """Recompute a preregistered PC-versus-simulator distribution audit."""

    from zuma_rl.distribution_fidelity import (
        DistributionFidelityError,
        verify_distribution_audit,
    )

    if tuple(spec.features) != ("startup_actor_distribution",):
        raise FidelityGateValidationError(
            "distribution_audit may only declare startup_actor_distribution"
        )
    report: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    compatible_policies = (
        policy.compatible_distribution_policies or (policy.id,)
    )
    for policy_id in compatible_policies:
        try:
            report = verify_distribution_audit(
                path,
                evidence_root=evidence_root,
                original_root=original_root,
                policy_id=policy_id,
            )
            break
        except (OSError, DistributionFidelityError) as error:
            last_error = error
    if report is None:
        raise FidelityGateValidationError(
            "distribution_audit evidence is invalid: " + str(last_error)
        ) from last_error
    accepted = report.get("status") == "PASS"
    raw_sources = _sequence(
        report.get("source_fingerprints"),
        "distribution source_fingerprints",
    )
    if any(
        not isinstance(item, str)
        or _SHA256_PATTERN.fullmatch(item) is None
        for item in raw_sources
    ):
        raise FidelityGateValidationError(
            "distribution source_fingerprints are invalid"
        )
    sources = tuple(sorted(set(raw_sources)))
    if len(sources) != len(raw_sources):
        raise FidelityGateValidationError(
            "distribution source_fingerprints must be unique and sorted"
        )
    return _AcceptedEvidence(
        spec=spec,
        accepted=accepted,
        schema=_nonempty(report.get("schema"), "distribution schema"),
        version=_strict_int(report.get("version"), "distribution version"),
        status=_nonempty(report.get("status"), "distribution status"),
        level_id=policy.levels[0],
        hard=policy.hard,
        profile_mode=policy.profile_mode,
        random_seed=None,
        tick_count=0,
        source_fingerprint=spec.sha256,
        authorized_features=(
            ("startup_actor_distribution",) if accepted else ()
        ),
        unsupported_features=(),
        supporting_source_fingerprints=sources,
        reason=(
            None
            if accepted
            else "PC/simulator startup RNG distributions exceed the frozen limit"
        ),
    )


def _evaluate_report(
    spec: EvidenceSpec,
    path: Path,
    *,
    policy: GatePolicy,
    evidence_root: Path,
    original_root: str | Path | None,
) -> _AcceptedEvidence:
    report = _mapping(
        _read_json(path, "evidence report", maximum_bytes=MAX_REPORT_BYTES),
        "evidence report",
    )
    schema = _nonempty(report.get("schema"), "report.schema")
    version = _strict_int(report.get("version"), "report.version")
    status = _nonempty(report.get("status"), "report.status")

    if spec.kind is EvidenceKind.SIMULATOR_DIFF:
        versions = _SIMULATOR_REPORT_SCHEMAS.get(schema)
    elif spec.kind is EvidenceKind.AUDIT:
        versions = (
            AUDIT_REPORT_VERSIONS
            if schema == AUDIT_SCHEMA
            else None
        )
    else:
        versions = {
            **_SIMULATOR_REPORT_SCHEMAS,
            **_DIAGNOSTIC_REPORT_SCHEMAS,
        }.get(schema)
    if versions is None or version not in versions:
        raise FidelityGateValidationError(
            f"report schema/version is not valid for {spec.kind.value}"
        )

    failure_reasons = report.get("failure_reasons")
    if (
        failure_reasons is not None
        and (
            isinstance(failure_reasons, (str, bytes))
            or not isinstance(failure_reasons, Sequence)
        )
    ):
        raise FidelityGateValidationError(
            "report.failure_reasons must be an array"
        )
    accepted = status == "PASS" and not failure_reasons
    reason = None if accepted else f"evidence report status is {status}"
    if accepted and spec.kind is EvidenceKind.SIMULATOR_DIFF:
        reason = _noncertifying_report_reason(
            report,
            schema=schema,
            evidence_root=evidence_root,
            original_root=original_root,
        )
        accepted = reason is None
    if (
        accepted
        and spec.kind is EvidenceKind.SIMULATOR_DIFF
        and (
            report.get("source_authorized_features") is not None
            or report.get("source_feature_proofs") is not None
        )
    ):
        from zuma_rl.pc_mechanism_audit import (
            validate_simulator_source_feature_proofs,
        )

        reason = validate_simulator_source_feature_proofs(
            report,
            evidence_root=evidence_root,
            original_root=original_root,
            level_id=policy.levels[0],
            hard=policy.hard,
        )
        accepted = reason is None

    level_id, hard, profile_mode = _extract_scenario(report)
    if spec.kind is EvidenceKind.AUDIT:
        scope = report.get("scope")
        if not isinstance(scope, Mapping):
            raise FidelityGateValidationError(
                "audit report requires a scope object"
            )
        expected_scope = {
            "environment_id": policy.environment_id,
            "profile_mode": policy.profile_mode,
            "observation_mode": policy.observation_mode,
        }
        audit_type = report.get("audit_type")
        compatible_audit_policies = (
            policy.compatible_audit_policies or (policy.id,)
        )
        # A mechanism audit is bound to the native source fingerprint and is
        # independently recomputed by its current validator.  Its historical
        # v1 policy label describes the suite that first consumed it, not a
        # weaker actor interface.  Actor/interface audits do not receive this
        # compatibility exception.
        if audit_type == "pc_mechanism_coverage":
            compatible_audit_policies = tuple(
                sorted(
                    {
                        *compatible_audit_policies,
                        "original-transfer-jungle2-v1",
                    }
                )
            )
        if (
            scope.get("policy") not in compatible_audit_policies
            or any(
                scope.get(key) != value
                for key, value in expected_scope.items()
            )
        ):
            accepted = False
            reason = "audit report scope differs from the gate policy"
        allowed_features = {
            "actor_no_hidden_state": frozenset(
                {"actor_no_hidden_state", "fruit_actor_observation"}
            ),
            "actor_visual_derivability": frozenset(
                {"actor_visual_derivability"}
            ),
            "pc_mechanism_coverage": frozenset(
                {"pc_mechanism_coverage"}
            ),
            "original_asset_fidelity": frozenset(
                {"fruit_asset_geometry"}
            ),
        }.get(audit_type)
        if (
            allowed_features is None
            or not set(spec.features).issubset(allowed_features)
        ):
            raise FidelityGateValidationError(
                "audit type does not authorize its declared features"
            )
        if accepted:
            if audit_type == "actor_no_hidden_state":
                from zuma_rl.actor_audit import (
                    validate_actor_audit_report,
                )

                reason = validate_actor_audit_report(
                    report,
                    root=original_root,
                    level_id=policy.levels[0],
                    hard=policy.hard,
                    profile_mode=policy.profile_mode,
                )
            elif audit_type == "actor_visual_derivability":
                from zuma_rl.pc_visual_audit import (
                    validate_actor_visual_audit_report,
                )

                reason = validate_actor_visual_audit_report(
                    report,
                    evidence_root=evidence_root,
                    original_root=original_root,
                    level_id=policy.levels[0],
                    hard=policy.hard,
                    profile_mode=policy.profile_mode,
                )
            elif audit_type == "original_asset_fidelity":
                from zuma_rl.original_asset_audit import (
                    validate_original_asset_audit_report,
                )

                reason = validate_original_asset_audit_report(
                    report,
                    original_root=original_root,
                    level_id=policy.levels[0],
                    hard=policy.hard,
                    profile_mode=policy.profile_mode,
                )
            else:
                from zuma_rl.pc_mechanism_audit import (
                    validate_pc_mechanism_audit_report,
                )

                reason = validate_pc_mechanism_audit_report(
                    report,
                    evidence_root=evidence_root,
                    original_root=original_root,
                    level_id=policy.levels[0],
                    hard=policy.hard,
                    profile_mode=policy.profile_mode,
                )
            accepted = reason is None
    else:
        if level_id is None:
            accepted = False
            reason = "simulator/diagnostic report does not identify its level"
        elif level_id not in policy.levels:
            accepted = False
            reason = "evidence report level is outside the policy scope"
        if hard is not None and hard is not policy.hard:
            accepted = False
            reason = "evidence report difficulty differs from the policy"
        if (
            profile_mode is not None
            and profile_mode != policy.profile_mode
        ):
            accepted = False
            reason = "evidence report profile differs from the policy"

    if spec.kind is EvidenceKind.DIAGNOSTIC:
        accepted = status == "PASS" and not failure_reasons
        reason = (
            "diagnostic evidence is validated but cannot certify "
            "transferability"
        )
    if spec.kind is EvidenceKind.AUDIT and accepted:
        authorized_features = spec.features
        unsupported_features: tuple[str, ...] = ()
    elif spec.kind is EvidenceKind.SIMULATOR_DIFF and accepted:
        authorized_features, unsupported_features = (
            _declared_feature_authorization(
                spec,
                _simulator_authorized_features(report, schema=schema),
            )
        )
    else:
        authorized_features = ()
        unsupported_features = ()
    supporting_source_fingerprints: tuple[str, ...] = ()
    supporting_source_features: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    if (
        spec.kind is EvidenceKind.AUDIT
        and accepted
        and report.get("audit_type") == "actor_visual_derivability"
    ):
        summary = report.get("summary")
        raw_sources = (
            summary.get("pc_source_fingerprints")
            if isinstance(summary, Mapping)
            else None
        )
        if (
            isinstance(raw_sources, Sequence)
            and not isinstance(raw_sources, (str, bytes))
            and all(
                isinstance(item, str)
                and _SHA256_PATTERN.fullmatch(item) is not None
                for item in raw_sources
            )
        ):
            supporting_source_fingerprints = tuple(sorted(set(raw_sources)))
            supporting_source_features = tuple(
                (
                    source,
                    ("actor_visual_derivability",),
                )
                for source in supporting_source_fingerprints
            )
    if (
        spec.kind is EvidenceKind.AUDIT
        and accepted
        and report.get("audit_type") == "pc_mechanism_coverage"
    ):
        summary = report.get("summary")
        raw_authorizations = (
            summary.get("source_feature_authorizations")
            if isinstance(summary, Mapping)
            else None
        )
        parsed: dict[str, tuple[str, ...]] = {}
        if (
            isinstance(raw_authorizations, Sequence)
            and not isinstance(raw_authorizations, (str, bytes))
        ):
            for row in raw_authorizations:
                if not isinstance(row, Mapping):
                    parsed = {}
                    break
                source = row.get("source_fingerprint")
                raw_features = row.get("authorized_features")
                if (
                    not isinstance(source, str)
                    or _SHA256_PATTERN.fullmatch(source) is None
                    or isinstance(raw_features, (str, bytes))
                    or not isinstance(raw_features, Sequence)
                    or not raw_features
                    or any(
                        not isinstance(feature, str)
                        or not feature
                        for feature in raw_features
                    )
                ):
                    parsed = {}
                    break
                features = tuple(sorted(set(raw_features)))
                if list(raw_features) != list(features) or source in parsed:
                    parsed = {}
                    break
                parsed[source] = features
        if not parsed:
            accepted = False
            reason = "PC mechanism audit source authorizations are invalid"
            authorized_features = ()
            unsupported_features = ()
        else:
            supporting_source_features = tuple(sorted(parsed.items()))
            supporting_source_fingerprints = tuple(
                source for source, _ in supporting_source_features
            )
    return _AcceptedEvidence(
        spec=spec,
        accepted=accepted,
        schema=schema,
        version=version,
        status=status,
        level_id=level_id,
        hard=hard,
        profile_mode=profile_mode,
        random_seed=None,
        tick_count=_extract_tick_count(report),
        source_fingerprint=_extract_source_fingerprint(
            report,
            fallback=spec.sha256,
        ),
        authorized_features=authorized_features,
        unsupported_features=unsupported_features,
        supporting_source_fingerprints=supporting_source_fingerprints,
        supporting_source_features=supporting_source_features,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class FidelityGateReport:
    """Machine-readable outcome of one complete suite evaluation."""

    status: FidelityGateStatus
    policy: str | None
    evidence: tuple[Mapping[str, Any], ...] = ()
    requirements: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return {
            FidelityGateStatus.OPEN: 0,
            FidelityGateStatus.CLOSED: 1,
            FidelityGateStatus.INVALID: 2,
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "status": self.status.value,
            "policy": self.policy,
            "evidence": [dict(item) for item in self.evidence],
            "requirements": [dict(item) for item in self.requirements],
            "reasons": list(self.reasons),
            "summary": dict(self.summary),
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


def _invalid_report(
    message: str,
    *,
    policy: str | None = None,
) -> FidelityGateReport:
    return FidelityGateReport(
        status=FidelityGateStatus.INVALID,
        policy=policy,
        reasons=(message,),
        summary={"gate_open": False},
    )


def verify_fidelity_suite(
    suite_path: str | Path,
    *,
    suite_root: str | Path | None = None,
    original_root: str | Path | None = None,
) -> FidelityGateReport:
    """Validate all evidence and evaluate the suite's immutable policy."""

    source = Path(suite_path)
    try:
        suite = FidelitySuite.read_json(source)
    except FidelityGateValidationError as error:
        return _invalid_report(str(error))
    policy = GATE_POLICIES[suite.policy]
    root = source.parent if suite_root is None else Path(suite_root)

    accepted: list[_AcceptedEvidence] = []
    try:
        for spec in suite.evidence:
            path = _resolve_evidence_path(root, spec.path)
            if _sha256_path(path) != spec.sha256:
                raise FidelityGateValidationError(
                    f"evidence {spec.id!r} SHA-256 differs"
                )
            if spec.kind is EvidenceKind.PC_GOLDEN:
                item = _evaluate_pc_golden(
                    spec,
                    path,
                    policy=policy,
                    original_root=original_root,
                )
            elif spec.kind is EvidenceKind.PC_SOURCE:
                item = _evaluate_pc_source(
                    spec,
                    path,
                    policy=policy,
                    evidence_root=root.resolve(),
                    original_root=original_root,
                )
            elif spec.kind is EvidenceKind.DISTRIBUTION_AUDIT:
                item = _evaluate_distribution_audit(
                    spec,
                    path,
                    policy=policy,
                    evidence_root=root.resolve(),
                    original_root=original_root,
                )
            else:
                item = _evaluate_report(
                    spec,
                    path,
                    policy=policy,
                    evidence_root=root.resolve(),
                    original_root=original_root,
                )
            accepted.append(item)
    except FidelityGateValidationError as error:
        return _invalid_report(str(error), policy=policy.id)

    accepted = list(_authenticate_original_sources(accepted))

    source_feature_authorizations: dict[str, set[str]] = {}
    for item in accepted:
        if not item.accepted or item.spec.kind is not EvidenceKind.AUDIT:
            continue
        for source_fingerprint, features in item.supporting_source_features:
            source_feature_authorizations.setdefault(
                source_fingerprint,
                set(),
            ).update(features)
    if source_feature_authorizations:
        for index, item in enumerate(accepted):
            if (
                not item.accepted
                or item.spec.kind
                not in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
            ):
                continue
            cross_authorized = set(item.spec.features).intersection(
                source_feature_authorizations.get(
                    item.source_fingerprint,
                    set(),
                )
            )
            if not cross_authorized:
                continue
            accepted[index] = replace(
                item,
                authorized_features=tuple(
                    sorted(
                        {
                            *item.authorized_features,
                            *cross_authorized,
                        }
                    )
                ),
                unsupported_features=tuple(
                    feature
                    for feature in item.unsupported_features
                    if feature not in cross_authorized
                ),
            )

    # A distribution result is meaningful only when every PC sample points to
    # a source that this same suite authenticated.  This prevents a detached
    # JSON sample list from standing in for raw retail evidence.
    authenticated_source_fingerprints = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.spec.kind
        in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
    }
    for index, item in enumerate(accepted):
        if (
            not item.accepted
            or item.spec.kind is not EvidenceKind.DISTRIBUTION_AUDIT
        ):
            continue
        missing_sources = set(item.supporting_source_fingerprints).difference(
            authenticated_source_fingerprints
        )
        if missing_sources:
            accepted[index] = replace(
                item,
                accepted=False,
                authorized_features=(),
                reason=(
                    "distribution audit references unauthenticated PC "
                    f"sources ({len(missing_sources)} missing)"
                ),
            )

    reasons: list[str] = []
    requirement_rows: list[dict[str, Any]] = []
    for requirement in policy.requirements:
        matching = [
            item
            for item in accepted
            if item.accepted
            and requirement.feature in item.authorized_features
            and (
                requirement.minimum_ticks == 0
                or item.tick_count >= requirement.minimum_ticks
            )
        ]
        counts = {
            kind: sum(item.spec.kind is kind for item in matching)
            for kind in (
                EvidenceKind.PC_GOLDEN,
                EvidenceKind.PC_SOURCE,
                EvidenceKind.SIMULATOR_DIFF,
                EvidenceKind.DISTRIBUTION_AUDIT,
                EvidenceKind.AUDIT,
            )
        }
        original_source_count = len(
            {
                item.source_fingerprint
                for item in matching
                if item.spec.kind
                in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
                and item.original_source_authenticated
            }
        )
        passed = (
            counts[EvidenceKind.PC_GOLDEN] >= requirement.pc_golden
            and counts[EvidenceKind.PC_SOURCE] >= requirement.pc_source
            and original_source_count >= requirement.original_source
            and counts[EvidenceKind.SIMULATOR_DIFF]
            >= requirement.simulator_diff
            and counts[EvidenceKind.DISTRIBUTION_AUDIT]
            >= requirement.distribution_audit
            and counts[EvidenceKind.AUDIT] >= requirement.audit
        )
        if not passed:
            reasons.append(f"missing certifying evidence: {requirement.feature}")
        requirement_rows.append(
            {
                "feature": requirement.feature,
                "lane": requirement.lane,
                "status": "PASS" if passed else "MISSING",
                "minimum_ticks": requirement.minimum_ticks,
                "required": {
                    "pc_golden": requirement.pc_golden,
                    "pc_source": requirement.pc_source,
                    "original_source": requirement.original_source,
                    "simulator_diff": requirement.simulator_diff,
                    "distribution_audit": requirement.distribution_audit,
                    "audit": requirement.audit,
                },
                "accepted": {
                    "pc_golden": counts[EvidenceKind.PC_GOLDEN],
                    "pc_source": counts[EvidenceKind.PC_SOURCE],
                    "original_source": original_source_count,
                    "simulator_diff": counts[EvidenceKind.SIMULATOR_DIFF],
                    "distribution_audit": counts[
                        EvidenceKind.DISTRIBUTION_AUDIT
                    ],
                    "audit": counts[EvidenceKind.AUDIT],
                },
            }
        )

    pc_cases = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.spec.kind is EvidenceKind.PC_GOLDEN
    }
    pc_source_cases = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.spec.kind is EvidenceKind.PC_SOURCE
    }
    original_source_cases = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.original_source_authenticated
        and item.spec.kind
        in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
    }
    original_source_authentication_basis_counts = {
        basis: sum(
            1
            for item in accepted
            if item.accepted
            and item.original_source_authenticated
            and item.original_source_authentication_basis == basis
            and item.spec.kind
            in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
        )
        for basis in sorted(
            {
                item.original_source_authentication_basis
                for item in accepted
                if item.accepted
                and item.original_source_authenticated
                and item.original_source_authentication_basis is not None
                and item.spec.kind
                in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
            }
        )
    }
    simulator_cases = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.spec.kind is EvidenceKind.SIMULATOR_DIFF
    }
    random_seeds = {
        item.random_seed
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.spec.kind
        in {EvidenceKind.PC_GOLDEN, EvidenceKind.PC_SOURCE}
        and item.random_seed is not None
    }
    distribution_cases = {
        item.source_fingerprint
        for item in accepted
        if item.accepted
        and item.authorized_features
        and item.spec.kind is EvidenceKind.DISTRIBUTION_AUDIT
    }
    if len(pc_cases) < policy.minimum_pc_golden_cases:
        reasons.append(
            "insufficient independent PC Golden cases: "
            f"{len(pc_cases)}/{policy.minimum_pc_golden_cases}"
        )
    if len(simulator_cases) < policy.minimum_simulator_diff_cases:
        reasons.append(
            "insufficient simulator differential cases: "
            f"{len(simulator_cases)}/{policy.minimum_simulator_diff_cases}"
        )
    if len(original_source_cases) < policy.minimum_original_source_cases:
        reasons.append(
            "insufficient authenticated original source cases: "
            f"{len(original_source_cases)}/"
            f"{policy.minimum_original_source_cases}"
        )
    if len(distribution_cases) < policy.minimum_distribution_audits:
        reasons.append(
            "insufficient distribution fidelity audits: "
            f"{len(distribution_cases)}/"
            f"{policy.minimum_distribution_audits}"
        )
    if len(random_seeds) < policy.minimum_random_seeds:
        reasons.append(
            "insufficient independent PC random seeds: "
            f"{len(random_seeds)}/{policy.minimum_random_seeds}"
        )
    for item in accepted:
        if (
            item.spec.kind is not EvidenceKind.DIAGNOSTIC
            and not item.accepted
            and item.reason is not None
        ):
            reasons.append(f"{item.spec.id}: {item.reason}")
        if item.unsupported_features:
            reasons.append(
                f"{item.spec.id}: unsupported evidence feature claim(s): "
                + ", ".join(item.unsupported_features)
            )

    status = (
        FidelityGateStatus.OPEN
        if not reasons
        else FidelityGateStatus.CLOSED
    )
    requirement_status_by_lane: dict[str, bool] = {}
    for row in requirement_rows:
        lane = str(row["lane"])
        requirement_status_by_lane[lane] = (
            requirement_status_by_lane.get(lane, True)
            and row["status"] == "PASS"
        )
    if policy.generation >= 2:
        lane_rows = {
            "source_authenticity": {
                "status": (
                    "PASS"
                    if len(original_source_cases)
                    >= policy.minimum_original_source_cases
                    and len(random_seeds) >= policy.minimum_random_seeds
                    else "MISSING"
                ),
                "authenticated_original_sources": len(original_source_cases),
                "independent_random_seeds": len(random_seeds),
            },
            "dynamics": {
                "status": (
                    "PASS"
                    if requirement_status_by_lane.get("dynamics", True)
                    and len(simulator_cases)
                    >= policy.minimum_simulator_diff_cases
                    else "MISSING"
                )
            },
            "distribution": {
                "status": (
                    "PASS"
                    if requirement_status_by_lane.get("distribution", True)
                    and len(distribution_cases)
                    >= policy.minimum_distribution_audits
                    else "MISSING"
                )
            },
            "interface": {
                "status": (
                    "PASS"
                    if requirement_status_by_lane.get("interface", True)
                    else "MISSING"
                )
            },
        }
    else:
        lane_rows = {
            "legacy_v1": {
                "status": "PASS" if status is FidelityGateStatus.OPEN else "MISSING"
            }
        }
    return FidelityGateReport(
        status=status,
        policy=policy.id,
        evidence=tuple(item.to_dict() for item in accepted),
        requirements=tuple(requirement_rows),
        reasons=tuple(reasons),
        summary={
            "gate_open": status is FidelityGateStatus.OPEN,
            "policy_generation": policy.generation,
            "transfer_authorizing_policy": policy.transfer_authorizing,
            "lanes": lane_rows,
            "scope": {
                "environment_id": policy.environment_id,
                "levels": list(policy.levels),
                "hard": policy.hard,
                "profile_mode": policy.profile_mode,
                "observation_mode": policy.observation_mode,
                "curve_count": policy.curve_count,
            },
            "accepted_pc_golden_cases": len(pc_cases),
            "accepted_pc_source_cases": len(pc_source_cases),
            "accepted_original_source_cases": len(original_source_cases),
            "original_source_authentication_basis_counts": (
                original_source_authentication_basis_counts
            ),
            "accepted_simulator_diff_cases": len(simulator_cases),
            "accepted_distribution_audits": len(distribution_cases),
            "accepted_random_seed_count": len(random_seeds),
            "certifying_retail_dmo_count": len(
                {
                    item.source_dmo_sha256
                    for item in accepted
                    if item.accepted
                    and item.spec.kind is EvidenceKind.PC_GOLDEN
                    and item.source_dmo_sha256 is not None
                }
            ),
            "diagnostic_only_cases": sum(
                item.spec.kind is EvidenceKind.DIAGNOSTIC
                for item in accepted
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a content-addressed, scope-specific fidelity suite. "
            "Diagnostic PASS reports never substitute for unmodified PC "
            "Golden evidence."
        )
    )
    parser.add_argument("suite", type=Path)
    parser.add_argument(
        "--suite-root",
        type=Path,
        help="Evidence root; defaults to the suite directory.",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        help=(
            "Original Zuma's Revenge installation. Required before the gate "
            "can open."
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create the recomputed gate report at this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_fidelity_suite(
        args.suite,
        suite_root=args.suite_root,
        original_root=args.original_root,
    )
    payload = report.to_json(indent=None if args.compact else 2)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() or not output.parent.is_dir():
            raise FidelityGateValidationError(
                "gate report output path must be absent"
            )
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload + "\n")
        except OSError as error:
            raise FidelityGateValidationError(
                "gate report output could not be created"
            ) from error
    print(payload)
    return report.exit_code


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "EvidenceKind",
    "EvidenceSpec",
    "FidelityGateReport",
    "FidelityGateStatus",
    "FidelityGateValidationError",
    "FidelitySuite",
    "GATE_POLICIES",
    "GatePolicy",
    "GateRequirement",
    "REPORT_SCHEMA",
    "REPORT_VERSION",
    "SUITE_SCHEMA",
    "SUITE_VERSION",
    "build_parser",
    "main",
    "verify_fidelity_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
