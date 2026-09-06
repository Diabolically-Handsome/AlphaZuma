"""Fail-closed recipe gate for bounded state-policy training.

Fidelity evidence answers whether the simulator is suitable for a narrow
scope.  This independent gate answers whether one *training request* stays
inside the only recipe and scale envelope supported by prior behavioral
evidence.  Both gates are required by the transfer-training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SUITE_SCHEMA = "zuma-rl.training-gate-suite"
SUITE_VERSION = 1
REPORT_SCHEMA = "zuma-rl.training-gate-report"
REPORT_VERSION = 2

# Historical policy retained for read-only reproduction.  It no longer
# authorizes transfer because it predates the fruit actor channels and binds a
# five-times-larger horizon than its supporting stability run.
POLICY_ID = "jungle2-teacher-anchor-v1"
STAGE_ID = "state-policy-calibration-491520-v1"
REQUIRED_FIDELITY_POLICY = "original-transfer-jungle2-v2"

LEGACY_TRANSFER_POLICY_ID = "jungle2-transfer-bootstrap-v2"
LEGACY_TRANSFER_STAGE_ID = "state-policy-bootstrap-98304-v2"
TRANSFER_POLICY_ID = "jungle2-transfer-bootstrap-v3"
TRANSFER_STAGE_ID = "state-policy-bootstrap-98304-v3"
TRANSFER_REQUIRED_FIDELITY_POLICY = "original-transfer-jungle2-v4"

EXPECTED_OUTCOME_SHA256 = (
    "sha256:2db48f6ff73b72e84705e715d4d9431cf24afeb3e4135dc5c83114e7ae4a2449"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "sha256:e51c651e5f17604f59c7c3a057a5d3c9e95ff32428c689077db4ef2993fc9c50"
)
EXPECTED_INITIAL_MODEL_SHA256 = (
    "sha256:f03dee97cb5a3e2ce3abcf4324137fe6ed9b68e827765b4a7b6adfa63cafb8f7"
)
VALIDATION_EFFECTIVE_STEPS = 98_304
AUTHORIZED_EFFECTIVE_STEPS = 491_520
TRANSFER_AUTHORIZED_EFFECTIVE_STEPS = VALIDATION_EFFECTIVE_STEPS
AUTHORIZED_TRAINING_SEED = 20_260_950

_SUPPORTED_POLICY_STAGES = frozenset(
    {
        (POLICY_ID, STAGE_ID),
        (LEGACY_TRANSFER_POLICY_ID, LEGACY_TRANSFER_STAGE_ID),
        (TRANSFER_POLICY_ID, TRANSFER_STAGE_ID),
    }
)

_VALIDATED_SOURCE_HASHES = {
    "train.py": "sha256:94d14cb9cf5da4e46989cab00ef501d0852bc8e08488d1e7fd955bd7185f0f84",
    "teacher_anchor.py": "sha256:52611e4833422f81a148281dad8982d1d0908410f4c52842bd11f15c58d5a601",
    "boundary_evidence.py": "sha256:27c1a103c53d57e0f3983714fce58ca144ab927a84dfd3770c4273c2590826f6",
    "compare_ppo_boundaries.py": "sha256:c3d6d3bc02d0de6901d5d265db966dd015bda87473a10154a92c9439a17efeca",
    "audit_maskable_policy_drift.py": "sha256:2c54c32a3752b3c27ebeb0b38724143b52ddaa02610b19d2be00b25c2098cd26",
}
_CURRENT_SOURCE_HASHES = {
    "src/zuma_rl/train.py": "sha256:2691c66a9f8ef32d781dec799a1a6d5002e457f7126f1851d444ef7e1c11be94",
    "src/zuma_rl/teacher_anchor.py": _VALIDATED_SOURCE_HASHES[
        "teacher_anchor.py"
    ],
    "src/zuma_rl/boundary_evidence.py": _VALIDATED_SOURCE_HASHES[
        "boundary_evidence.py"
    ],
    "tools/compare_ppo_boundaries.py": _VALIDATED_SOURCE_HASHES[
        "compare_ppo_boundaries.py"
    ],
    "tools/audit_maskable_policy_drift.py": _VALIDATED_SOURCE_HASHES[
        "audit_maskable_policy_drift.py"
    ],
}

MAX_SUITE_BYTES = 256 * 1024
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 128

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class TrainingGateValidationError(ValueError):
    """The suite or a content-addressed prerequisite is malformed."""


class TrainingGateStatus(str, Enum):
    """Top-level training-recipe gate outcome."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class TrainingEvidenceSpec:
    """One immutable prerequisite supplied by a suite."""

    id: str
    kind: str
    path: PurePosixPath
    sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingEvidenceSpec":
        data = _mapping(value, "evidence entry")
        _check_keys(
            data,
            required={"id", "kind", "path", "sha256"},
            name="evidence entry",
        )
        identifier = _identifier(data["id"], "evidence.id")
        kind = _identifier(data["kind"], "evidence.kind")
        if kind not in {
            "environment_contract",
            "fidelity_report",
            "fidelity_suite",
            "model_migration",
            "preregistration",
            "scale_stability_outcome",
        }:
            raise TrainingGateValidationError(
                "unsupported training evidence kind"
            )
        return cls(
            id=identifier,
            kind=kind,
            path=_relative_path(data["path"], "evidence.path"),
            sha256=_digest(data["sha256"], "evidence.sha256"),
        )


@dataclass(frozen=True, slots=True)
class TrainingSuite:
    """Strict suite input; policy and stage requirements live in code."""

    policy: str
    stage: str
    evidence: tuple[TrainingEvidenceSpec, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingSuite":
        data = _mapping(value, "training suite")
        _check_keys(
            data,
            required={"schema", "version", "policy", "stage", "evidence"},
            name="training suite",
        )
        if data["schema"] != SUITE_SCHEMA or data["version"] != SUITE_VERSION:
            raise TrainingGateValidationError(
                "unsupported training suite schema/version"
            )
        if (data["policy"], data["stage"]) not in _SUPPORTED_POLICY_STAGES:
            raise TrainingGateValidationError(
                "unsupported training policy or stage"
            )
        raw_evidence = _sequence(data["evidence"], "training suite evidence")
        evidence = tuple(
            TrainingEvidenceSpec.from_dict(_mapping(item, "evidence entry"))
            for item in raw_evidence
        )
        ids = [item.id for item in evidence]
        paths = [item.path.as_posix() for item in evidence]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise TrainingGateValidationError(
                "training evidence ids must be unique and sorted"
            )
        if len(paths) != len(set(paths)):
            raise TrainingGateValidationError(
                "training evidence paths must be unique"
            )
        return cls(
            policy=str(data["policy"]),
            stage=str(data["stage"]),
            evidence=evidence,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "TrainingSuite":
        return cls.from_dict(
            _mapping(
                _read_json(
                    Path(path),
                    "training suite",
                    maximum_bytes=MAX_SUITE_BYTES,
                ),
                "training suite",
            )
        )


@dataclass(frozen=True, slots=True)
class TrainingGateReport:
    """Machine-readable result of prerequisite and artifact verification."""

    status: TrainingGateStatus
    policy: str | None
    stage: str | None
    reasons: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    recipe: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return {
            TrainingGateStatus.OPEN: 0,
            TrainingGateStatus.CLOSED: 1,
            TrainingGateStatus.INVALID: 2,
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "status": self.status.value,
            "policy": self.policy,
            "stage": self.stage,
            "reasons": list(self.reasons),
            "evidence": [dict(item) for item in self.evidence],
            "recipe": dict(self.recipe),
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingGateValidationError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TrainingGateValidationError(f"{name} keys must be strings")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrainingGateValidationError(f"{name} must be a JSON array")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise TrainingGateValidationError(
            f"{name} fields are invalid: {'; '.join(details)}"
        )


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise TrainingGateValidationError(f"{name} is not a valid identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise TrainingGateValidationError(
            f"{name} must be 'sha256:' plus 64 lowercase hex digits"
        )
    return value


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TrainingGateValidationError(
            f"{name} must be a relative POSIX path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainingGateValidationError(f"{name} escapes the suite root")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingGateValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise TrainingGateValidationError(
        f"forbidden non-finite JSON constant: {value}"
    )


def _validate_json(value: Any, name: str, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise TrainingGateValidationError(f"{name} is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingGateValidationError(f"{name} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrainingGateValidationError(
                    f"{name} keys must be strings"
                )
            _validate_json(item, f"{name}.{key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_json(item, f"{name}[{index}]", depth=depth + 1)
        return
    raise TrainingGateValidationError(f"{name} is not JSON-safe")


def _read_json(path: Path, name: str, *, maximum_bytes: int) -> Any:
    try:
        if path.stat().st_size > maximum_bytes:
            raise TrainingGateValidationError(f"{name} is too large")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise TrainingGateValidationError(f"{name} cannot be read") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except TrainingGateValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise TrainingGateValidationError(f"{name} is invalid JSON") from error
    _validate_json(value, name)
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TrainingGateValidationError("a bound artifact cannot be read") from error
    return "sha256:" + digest.hexdigest()


def _resolve(root: Path, relative: PurePosixPath) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        path = resolved_root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise TrainingGateValidationError(
            "training evidence escapes the suite root"
        ) from error
    if not path.is_file():
        raise TrainingGateValidationError("training evidence is not a file")
    return path


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _portable_host_path(value: str) -> Path:
    """Resolve frozen Windows paths when verification runs inside WSL."""

    match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", value)
    if match is not None and os.name != "nt":
        drive, suffix = match.groups()
        return Path("/mnt") / drive.lower() / Path(
            suffix.replace("\\", "/")
        )
    return Path(value)


def expected_training_request(
    *,
    policy: str = POLICY_ID,
    initial_model_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the exact request authorized by one immutable stage."""

    if policy == POLICY_ID:
        model_sha256 = EXPECTED_INITIAL_MODEL_SHA256
        effective_steps = AUTHORIZED_EFFECTIVE_STEPS
    elif policy in {LEGACY_TRANSFER_POLICY_ID, TRANSFER_POLICY_ID}:
        if (
            not isinstance(initial_model_sha256, str)
            or _SHA256_PATTERN.fullmatch(initial_model_sha256) is None
        ):
            raise TrainingGateValidationError(
                "transfer recipe requires the verified migrated-model SHA-256"
            )
        model_sha256 = initial_model_sha256
        effective_steps = TRANSFER_AUTHORIZED_EFFECTIVE_STEPS
    else:
        raise TrainingGateValidationError("unsupported training policy")

    return {
        "environment": "revenge",
        "level": "Jungle2",
        "hard": False,
        "curve_index": 0,
        "aim_bins": 180,
        "flat_actions": False,
        "mask_rejected_actions": True,
        "frame_skip": 1,
        "max_ticks": 12_000,
        "training_time_limit_steps": 0,
        "max_balls": 768,
        "initial_model_sha256": model_sha256,
        "freeze_aim_path": False,
        "reset_optimizer_state": True,
        "teacher_kl_coef": 1_000.0,
        "total_steps": effective_steps,
        "planned_effective_steps": effective_steps,
        "num_envs": 4,
        "seed": AUTHORIZED_TRAINING_SEED,
        "device": "cuda",
        "rollout_steps": 512,
        "batch_size": 512,
        "ppo_epochs": 1,
        "learning_rate": 0.000005,
        "reward_scale": 0.01,
        "entropy_coef": 0.0,
        "checkpoint_every": 16_384,
        "eval_every": 1_000_000,
        "eval_episodes": 8,
        "boundary_eval_episodes": 32,
        "eval_num_envs": 8,
        "also_evaluate_stochastic_boundaries": True,
        "single_process": False,
    }


def training_request_mismatches(
    request: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Compare a live request with the immutable stage recipe."""

    expected_request = (
        expected_training_request() if expected is None else dict(expected)
    )
    mismatches: list[str] = []
    if set(request) != set(expected_request):
        missing = sorted(set(expected_request).difference(request))
        unknown = sorted(set(request).difference(expected_request))
        if missing:
            mismatches.append("missing request fields: " + ", ".join(missing))
        if unknown:
            mismatches.append("unknown request fields: " + ", ".join(unknown))
    for name, expected_value in expected_request.items():
        if name in request and request[name] != expected_value:
            mismatches.append(
                f"{name}={request[name]!r}, required {expected_value!r}"
            )
    return tuple(mismatches)


def _prior_recipe() -> dict[str, Any]:
    return {
        "environment": "ZumaRevenge-v0",
        "level": "Jungle2",
        "profile_mode": "tutorials_completed",
        "aim_bins": 180,
        "factorized_actions": True,
        "actor_observable_retail_rejection_mask": True,
        "frame_skip": 1,
        "max_ticks": 12_000,
        "training_time_limit_steps": 0,
        "max_balls": 768,
        "algorithm": "TeacherAnchoredMaskablePPO",
        "total_effective_steps": VALIDATION_EFFECTIVE_STEPS,
        "num_envs": 4,
        "rollout_steps_per_env": 512,
        "batch_size": 512,
        "ppo_epochs": 1,
        "learning_rate": 0.000005,
        "entropy_coef": 0.0,
        "reset_optimizer_state": True,
        "teacher_forward_kl_coefficient": 1_000.0,
        "teacher_forward_kl_state_source": "current_rollout_observations",
        "teacher_and_current_action_masks_identical": True,
        "all_current_policy_and_value_parameters_trainable": True,
        "checkpoint_every_effective_steps": 16_384,
        "expected_checkpoint_steps": [
            16_384,
            32_768,
            49_152,
            65_536,
            81_920,
            98_304,
        ],
    }


def _evaluate_prerequisite_documents(
    preregistration: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    preregistration_path: Path,
) -> list[str]:
    reasons: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            reasons.append(message)

    require(
        preregistration.get("schema")
        == "zuma-rl.teacher-policy-anchor-scale-stability-preregistration"
        and preregistration.get("version") == 2
        and preregistration.get("status") == "FROZEN_BEFORE_RUN_OUTPUT",
        "scale-stability preregistration identity differs",
    )
    require(
        _nested(preregistration, "design", "input_model", "sha256")
        == EXPECTED_INITIAL_MODEL_SHA256,
        "validated input-model identity differs",
    )
    require(
        _nested(preregistration, "design", "recipe") == _prior_recipe(),
        "validated 98,304-step recipe differs",
    )
    require(
        _nested(
            preregistration,
            "reserved_after_pass_only",
            "large_scale_training_base_seed",
        )
        == AUTHORIZED_TRAINING_SEED,
        "reserved next-stage training seed differs",
    )
    require(
        _nested(
            preregistration,
            "reserved_after_pass_only",
            "gate_open_required_before_use",
        )
        is True,
        "reserved seed is not conditioned on an open fidelity gate",
    )
    require(
        _nested(
            preregistration,
            "regression_anchor",
            "source_hashes",
        )
        == _VALIDATED_SOURCE_HASHES,
        "validated training source registry differs",
    )

    require(
        outcome.get("schema")
        == "zuma-rl.teacher-policy-anchor-scale-stability-outcome"
        and outcome.get("version") == 2
        and outcome.get("status") == "BEHAVIORAL_SCALE_STABILITY_PASS",
        "behavioral scale-stability outcome did not pass",
    )
    require(
        outcome.get("decision")
        == "AUTHORIZE_EIGHTH_PC_GOLDEN_AND_FIDELITY_GATE_RECERTIFICATION",
        "scale-stability decision differs",
    )
    prereg_binding = _nested(outcome, "preregistration")
    require(
        isinstance(prereg_binding, Mapping)
        and prereg_binding.get("sha256") == EXPECTED_PREREGISTRATION_SHA256
        and prereg_binding.get("status") == "FROZEN_BEFORE_RUN_OUTPUT"
        and Path(str(prereg_binding.get("path", ""))).name
        == preregistration_path.name,
        "outcome does not bind the frozen preregistration",
    )
    require(
        _nested(outcome, "independence", "seed_family") == 20_260_940
        and _nested(outcome, "independence", "json_exact_hits_before_freeze")
        == 0
        and _nested(
            outcome,
            "independence",
            "prior_run_artifacts_using_seed_family",
        )
        == 0
        and _nested(
            outcome,
            "independence",
            "replacement_seed_selection_after_baseline",
        )
        is False,
        "scale-stability seed-independence checks differ",
    )
    for section in (
        "pipeline",
        "capability_precondition",
        "deterministic_behavior",
        "stochastic_behavior",
        "frozen_dataset_actor_safety",
        "overall",
    ):
        require(
            _nested(outcome, "acceptance_results", section, "pass") is True,
            f"scale-stability acceptance section {section!r} did not pass",
        )
    require(
        _nested(
            outcome,
            "acceptance_results",
            "overall",
            "post_hoc_override_applied",
        )
        is False,
        "scale-stability outcome used a post-hoc override",
    )
    require(
        _nested(
            outcome,
            "acceptance_results",
            "pipeline",
            "effective_steps",
        )
        == VALIDATION_EFFECTIVE_STEPS,
        "scale-stability validation horizon differs",
    )
    require(
        _nested(outcome, "next_action", "reserved_large_scale_seed_family")
        == AUTHORIZED_TRAINING_SEED
        and _nested(
            outcome,
            "next_action",
            "large_scale_training_requires_gate_open",
        )
        is True,
        "outcome does not preserve the next-stage gate condition",
    )
    return reasons


_RUN_ARTIFACT_FILES = {
    "config_sha256": "config.json",
    "completion_sha256": "completion.json",
    "optimizer_reset_sha256": "optimizer_state_reset.json",
    "teacher_anchor_installation_sha256": "teacher_policy_anchor.json",
    "teacher_anchor_verification_sha256": (
        "teacher_policy_anchor_verification.json"
    ),
    "pretrain_deterministic_sha256": "pretrain_evaluation.json",
    "posttrain_deterministic_sha256": "posttrain_evaluation.json",
    "pretrain_stochastic_sha256": "pretrain_evaluation_stochastic.json",
    "posttrain_stochastic_sha256": "posttrain_evaluation_stochastic.json",
    "deterministic_comparison_sha256": "deterministic_comparison.json",
    "stochastic_comparison_sha256": "stochastic_comparison.json",
    "final_model_sha256": "final_model.zip",
}


def _verify_prior_run_artifacts(
    outcome: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    artifacts = _nested(outcome, "artifacts")
    if not isinstance(artifacts, Mapping):
        return ["outcome artifact registry is missing"], rows
    run_dir_value = artifacts.get("run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        return ["validated run directory is missing"], rows
    run_dir = _portable_host_path(run_dir_value)
    for digest_name, filename in _RUN_ARTIFACT_FILES.items():
        expected = artifacts.get(digest_name)
        path = run_dir / filename
        actual: str | None = None
        if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
            reasons.append(f"validated artifact binding {digest_name!r} is invalid")
        elif not path.is_file():
            reasons.append(f"validated artifact is missing: {filename}")
        else:
            actual = _sha256_path(path)
            if actual != expected:
                reasons.append(f"validated artifact hash differs: {filename}")
        rows.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual is not None and actual == expected else "FAIL",
            }
        )

    checkpoints = artifacts.get("checkpoint_sha256_by_step")
    if not isinstance(checkpoints, Mapping):
        reasons.append("validated checkpoint registry is missing")
    else:
        for step in (16_384, 32_768, 49_152, 65_536, 81_920, 98_304):
            expected = checkpoints.get(str(step))
            path = (
                run_dir
                / "checkpoints"
                / (
                    "zuma_teacher_anchored_maskable_ppo_diagnostic_"
                    f"{step}_steps.zip"
                )
            )
            actual = _sha256_path(path) if path.is_file() else None
            if actual != expected:
                reasons.append(f"validated checkpoint hash differs: step {step}")
            rows.append(
                {
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "status": (
                        "PASS"
                        if actual is not None and actual == expected
                        else "FAIL"
                    ),
                }
            )
    if (run_dir / "failure.json").exists():
        reasons.append("validated run contains a failure receipt")

    try:
        config = _mapping(
            _read_json(
                run_dir / "config.json",
                "validated run config",
                maximum_bytes=MAX_EVIDENCE_BYTES,
            ),
            "validated run config",
        )
        completion = _mapping(
            _read_json(
                run_dir / "completion.json",
                "validated completion receipt",
                maximum_bytes=MAX_EVIDENCE_BYTES,
            ),
            "validated completion receipt",
        )
    except TrainingGateValidationError as error:
        reasons.append(str(error))
        return reasons, rows
    if (
        config.get("algorithm") != "teacher_anchored_maskable_ppo"
        or config.get("diagnostic_non_transferable") is not True
        or config.get("planned_effective_steps") != VALIDATION_EFFECTIVE_STEPS
        or _nested(config, "initial_model", "sha256")
        != EXPECTED_INITIAL_MODEL_SHA256
        or _nested(config, "environment", "id") != "ZumaRevenge-v0"
        or _nested(config, "environment", "level") != "Jungle2"
        or _nested(config, "environment", "profile_mode")
        != "tutorials_completed"
    ):
        reasons.append("validated run config semantics differ")
    if (
        completion.get("schema") != "zuma-rl.ppo-training-completion"
        or completion.get("version") != 1
        or completion.get("status") != "COMPLETE"
        or completion.get("algorithm")
        != "teacher_anchored_maskable_ppo"
        or completion.get("effective_steps") != VALIDATION_EFFECTIVE_STEPS
        or _nested(
            completion,
            "process_contract",
            "vector_env_close_status",
        )
        != "PASS"
    ):
        reasons.append("validated completion receipt semantics differ")
    return reasons, rows


def _verify_input_model(preregistration: Mapping[str, Any]) -> tuple[str | None, str | None]:
    path_value = _nested(preregistration, "design", "input_model", "path")
    if not isinstance(path_value, str) or not path_value:
        return None, "validated input-model path is missing"
    path = _portable_host_path(path_value)
    if not path.is_file():
        return str(path), "validated input model is missing"
    if _sha256_path(path) != EXPECTED_INITIAL_MODEL_SHA256:
        return str(path), "validated input-model hash differs"
    return str(path), None


def _verify_current_training_sources() -> tuple[list[str], list[dict[str, Any]]]:
    """Bind the executable entry to the reviewed post-gate source set."""

    project_root = Path(__file__).resolve().parents[2]
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    for relative, expected in _CURRENT_SOURCE_HASHES.items():
        path = project_root.joinpath(*PurePosixPath(relative).parts)
        actual = _sha256_path(path) if path.is_file() else None
        if actual != expected:
            reasons.append(f"current training source hash differs: {relative}")
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": (
                    "PASS" if actual is not None and actual == expected else "FAIL"
                ),
            }
        )
    return reasons, rows


def _read_bound_document(
    resolved: Mapping[str, tuple[TrainingEvidenceSpec, Path]],
    kind: str,
    name: str,
) -> Mapping[str, Any]:
    entry = resolved.get(kind)
    if entry is None:
        return {}
    return _mapping(
        _read_json(
            entry[1],
            name,
            maximum_bytes=MAX_EVIDENCE_BYTES,
        ),
        name,
    )


def _verify_transfer_suite(
    *,
    suite: TrainingSuite,
    root: Path,
    resolved: Mapping[str, tuple[TrainingEvidenceSpec, Path]],
    evidence_rows: Sequence[Mapping[str, Any]],
    original_root: str | Path | None,
) -> TrainingGateReport:
    """Verify current environment, migrated model, and exact Fidelity suite."""

    expected_kinds = {
        "environment_contract",
        "fidelity_report",
        "fidelity_suite",
        "model_migration",
        "preregistration",
        "scale_stability_outcome",
    }
    reasons: list[str] = []
    is_current_policy = suite.policy == TRANSFER_POLICY_ID
    if not is_current_policy:
        reasons.append(
            "v2 transfer policy is historical after the runtime-source boundary repair"
        )
    if set(resolved) != expected_kinds or len(suite.evidence) != len(
        expected_kinds
    ):
        generation = "v3" if is_current_policy else "v2"
        reasons.append(
            f"the six fixed {generation} prerequisite documents are required"
        )

    preregistration = _read_bound_document(
        resolved,
        "preregistration",
        "scale-stability preregistration",
    )
    outcome = _read_bound_document(
        resolved,
        "scale_stability_outcome",
        "scale-stability outcome",
    )
    environment_contract = _read_bound_document(
        resolved,
        "environment_contract",
        "training environment contract",
    )
    migration = _read_bound_document(
        resolved,
        "model_migration",
        "model migration report",
    )
    frozen_fidelity_report = _read_bound_document(
        resolved,
        "fidelity_report",
        "fidelity gate report",
    )

    outcome_entry = resolved.get("scale_stability_outcome")
    prereg_entry = resolved.get("preregistration")
    if outcome_entry is not None and outcome_entry[0].sha256 != EXPECTED_OUTCOME_SHA256:
        reasons.append("scale-stability outcome identity is not approved")
    if (
        prereg_entry is not None
        and prereg_entry[0].sha256 != EXPECTED_PREREGISTRATION_SHA256
    ):
        reasons.append("scale-stability preregistration identity is not approved")
    artifact_rows: list[dict[str, Any]] = []
    legacy_model_path: str | None = None
    if outcome_entry is not None and prereg_entry is not None:
        reasons.extend(
            _evaluate_prerequisite_documents(
                preregistration,
                outcome,
                preregistration_path=prereg_entry[1],
            )
        )
        artifact_reasons, artifact_rows = _verify_prior_run_artifacts(outcome)
        reasons.extend(artifact_reasons)
        legacy_model_path, model_reason = _verify_input_model(preregistration)
        if model_reason is not None:
            reasons.append(model_reason)

    environment_entry = resolved.get("environment_contract")
    environment_reason: str | None = None
    if environment_entry is None:
        environment_reason = "training environment contract is missing"
    else:
        from zuma_rl.environment_contract import (
            validate_training_environment_contract,
        )

        environment_reason = validate_training_environment_contract(
            environment_contract,
            original_root=original_root,
        )
    if environment_reason is not None:
        reasons.append(environment_reason)

    migration_entry = resolved.get("model_migration")
    migration_reason: str | None = None
    if migration_entry is None:
        migration_reason = "model migration report is missing"
    else:
        from zuma_rl.model_migration import validate_model_migration_report

        migration_reason = validate_model_migration_report(
            migration,
            evidence_root=root,
            original_root=original_root,
        )
    if migration_reason is not None:
        reasons.append(migration_reason)

    source_model = migration.get("source_model")
    migrated_model = migration.get("migrated_model")
    migration_contract = migration.get("environment_contract")
    migrated_sha256 = (
        migrated_model.get("sha256")
        if isinstance(migrated_model, Mapping)
        else None
    )
    if (
        not isinstance(source_model, Mapping)
        or source_model.get("sha256") != EXPECTED_INITIAL_MODEL_SHA256
    ):
        reasons.append("model migration does not start from the validated teacher")
    if (
        not isinstance(migrated_sha256, str)
        or _SHA256_PATTERN.fullmatch(migrated_sha256) is None
    ):
        reasons.append("migrated model identity is invalid")
        migrated_sha256 = EXPECTED_INITIAL_MODEL_SHA256
    if (
        environment_entry is not None
        and (
            not isinstance(migration_contract, Mapping)
            or migration_contract.get("sha256") != environment_entry[0].sha256
        )
    ):
        reasons.append("model migration does not bind the environment contract")

    fidelity_entry = resolved.get("fidelity_suite")
    fidelity_report_entry = resolved.get("fidelity_report")
    live_fidelity_report: Mapping[str, Any] = {}
    if fidelity_entry is None:
        reasons.append("fidelity suite is missing")
    else:
        from zuma_rl.fidelity_gate import (
            FidelityGateStatus,
            GATE_POLICIES,
        )
        from zuma_rl.fidelity_runtime import (
            verify_fidelity_suite_for_training,
        )

        live = verify_fidelity_suite_for_training(
            fidelity_entry[1],
            suite_root=root,
            original_root=original_root,
        )
        live_fidelity_report = live.to_dict()
        policy = GATE_POLICIES.get(live.policy or "")
        lanes = live.summary.get("lanes")
        all_lanes_pass = (
            isinstance(lanes, Mapping)
            and set(lanes) == {
                "source_authenticity",
                "dynamics",
                "distribution",
                "interface",
            }
            and all(
                isinstance(row, Mapping) and row.get("status") == "PASS"
                for row in lanes.values()
            )
        )
        if (
            live.status is not FidelityGateStatus.OPEN
            or live.policy != TRANSFER_REQUIRED_FIDELITY_POLICY
            or policy is None
            or not policy.transfer_authorizing
            or not all_lanes_pass
        ):
            reasons.append(
                "bound Fidelity Gate is not OPEN on all v4 evidence lanes"
            )
    if fidelity_report_entry is None:
        reasons.append("fidelity report is missing")
    elif dict(frozen_fidelity_report) != dict(live_fidelity_report):
        reasons.append("frozen Fidelity report differs from fresh recomputation")

    recipe = expected_training_request(
        policy=suite.policy,
        initial_model_sha256=migrated_sha256,
    )
    recipe_sha256 = _canonical_digest(recipe)
    status = TrainingGateStatus.OPEN if not reasons else TrainingGateStatus.CLOSED
    migrated_model_path = (
        migrated_model.get("path")
        if isinstance(migrated_model, Mapping)
        else None
    )
    return TrainingGateReport(
        status=status,
        policy=suite.policy,
        stage=suite.stage,
        reasons=tuple(reasons),
        evidence=tuple(dict(row) for row in evidence_rows),
        recipe=recipe,
        summary={
            "training_gate_open": status is TrainingGateStatus.OPEN,
            "transfer_authorizing_policy": is_current_policy,
            "required_fidelity_policy": TRANSFER_REQUIRED_FIDELITY_POLICY,
            "required_fidelity_suite_sha256": (
                fidelity_entry[0].sha256 if fidelity_entry is not None else None
            ),
            "required_fidelity_report_sha256": (
                fidelity_report_entry[0].sha256
                if fidelity_report_entry is not None
                else None
            ),
            "environment_contract_sha256": (
                environment_entry[0].sha256
                if environment_entry is not None
                else None
            ),
            "environment_contract_fingerprint": environment_contract.get(
                "contract_fingerprint"
            ),
            "validated_legacy_input_model_path": legacy_model_path,
            "validated_input_model_path": migrated_model_path,
            "validated_input_model_sha256": migrated_sha256,
            "validated_effective_steps": VALIDATION_EFFECTIVE_STEPS,
            "authorized_effective_steps": TRANSFER_AUTHORIZED_EFFECTIVE_STEPS,
            "authorized_training_seed": AUTHORIZED_TRAINING_SEED,
            "recipe_sha256": recipe_sha256,
            "validated_run_artifacts": artifact_rows,
            "state_policy_training_only": True,
            "visual_policy_training_authorized": False,
            "original_game_deployment_authorized": False,
            "scale_extrapolation_factor": 1.0,
        },
    )


def _invalid_report(message: str) -> TrainingGateReport:
    return TrainingGateReport(
        status=TrainingGateStatus.INVALID,
        policy=None,
        stage=None,
        reasons=(message,),
        summary={"training_gate_open": False},
    )


def verify_training_suite(
    suite_path: str | Path,
    *,
    suite_root: str | Path | None = None,
    original_root: str | Path | None = None,
) -> TrainingGateReport:
    """Recompute the fixed scale-stability prerequisite and recipe gate."""

    source = Path(suite_path)
    try:
        suite = TrainingSuite.read_json(source)
        root = source.parent if suite_root is None else Path(suite_root)
        resolved: dict[str, tuple[TrainingEvidenceSpec, Path]] = {}
        evidence_rows: list[dict[str, Any]] = []
        for spec in suite.evidence:
            if spec.kind in resolved:
                raise TrainingGateValidationError(
                    f"duplicate training evidence kind: {spec.kind}"
                )
            path = _resolve(root, spec.path)
            actual = _sha256_path(path)
            if actual != spec.sha256:
                raise TrainingGateValidationError(
                    f"training evidence {spec.id!r} SHA-256 differs"
                )
            resolved[spec.kind] = (spec, path)
            evidence_rows.append(
                {
                    "id": spec.id,
                    "kind": spec.kind,
                    "path": spec.path.as_posix(),
                    "sha256": actual,
                    "status": "CONTENT_ADDRESS_PASS",
                }
            )
    except TrainingGateValidationError as error:
        return _invalid_report(str(error))

    if suite.policy in {LEGACY_TRANSFER_POLICY_ID, TRANSFER_POLICY_ID}:
        try:
            return _verify_transfer_suite(
                suite=suite,
                root=root.resolve(strict=True),
                resolved=resolved,
                evidence_rows=evidence_rows,
                original_root=original_root,
            )
        except TrainingGateValidationError as error:
            return _invalid_report(str(error))

    reasons: list[str] = []
    expected_kinds = {"scale_stability_outcome", "preregistration"}
    if set(resolved) != expected_kinds or len(suite.evidence) != 2:
        reasons.append("the two fixed prerequisite documents are required")
    outcome_entry = resolved.get("scale_stability_outcome")
    prereg_entry = resolved.get("preregistration")
    preregistration: Mapping[str, Any] = {}
    outcome: Mapping[str, Any] = {}
    try:
        if outcome_entry is not None:
            if outcome_entry[0].sha256 != EXPECTED_OUTCOME_SHA256:
                reasons.append("scale-stability outcome identity is not approved")
            outcome = _mapping(
                _read_json(
                    outcome_entry[1],
                    "scale-stability outcome",
                    maximum_bytes=MAX_EVIDENCE_BYTES,
                ),
                "scale-stability outcome",
            )
        if prereg_entry is not None:
            if prereg_entry[0].sha256 != EXPECTED_PREREGISTRATION_SHA256:
                reasons.append(
                    "scale-stability preregistration identity is not approved"
                )
            preregistration = _mapping(
                _read_json(
                    prereg_entry[1],
                    "scale-stability preregistration",
                    maximum_bytes=MAX_EVIDENCE_BYTES,
                ),
                "scale-stability preregistration",
            )
        if outcome_entry is not None and prereg_entry is not None:
            reasons.extend(
                _evaluate_prerequisite_documents(
                    preregistration,
                    outcome,
                    preregistration_path=prereg_entry[1],
                )
            )
            artifact_reasons, artifact_rows = _verify_prior_run_artifacts(
                outcome
            )
            reasons.extend(artifact_reasons)
            model_path, model_reason = _verify_input_model(preregistration)
            if model_reason is not None:
                reasons.append(model_reason)
            source_reasons, source_rows = _verify_current_training_sources()
            reasons.extend(source_reasons)
        else:
            artifact_rows = []
            model_path = None
            source_rows = []
    except TrainingGateValidationError as error:
        return _invalid_report(str(error))

    recipe = expected_training_request(policy=POLICY_ID)
    recipe_sha256 = _canonical_digest(recipe)
    status = TrainingGateStatus.OPEN if not reasons else TrainingGateStatus.CLOSED
    return TrainingGateReport(
        status=status,
        policy=suite.policy,
        stage=suite.stage,
        reasons=tuple(reasons),
        evidence=tuple(evidence_rows),
        recipe=recipe,
        summary={
            "training_gate_open": status is TrainingGateStatus.OPEN,
            "transfer_authorizing_policy": False,
            "required_fidelity_policy": REQUIRED_FIDELITY_POLICY,
            "validated_effective_steps": VALIDATION_EFFECTIVE_STEPS,
            "authorized_effective_steps": AUTHORIZED_EFFECTIVE_STEPS,
            "authorized_training_seed": AUTHORIZED_TRAINING_SEED,
            "recipe_sha256": recipe_sha256,
            "validated_input_model_path": model_path,
            "validated_run_artifacts": artifact_rows,
            "current_training_sources": source_rows,
            "state_policy_training_only": True,
            "visual_policy_training_authorized": False,
            "original_game_deployment_authorized": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument(
        "--suite-root",
        type=Path,
        help="Evidence root; defaults to the suite directory.",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        help="Installed retail game required by the current environment contract.",
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create the recomputed Training Gate report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_training_suite(
        args.suite,
        suite_root=args.suite_root,
        original_root=args.original_root,
    )
    payload = report.to_json(indent=None if args.compact else 2)
    if args.output is not None:
        output = args.output.resolve(strict=False)
        if output.exists() or not output.parent.is_dir():
            raise TrainingGateValidationError(
                "training report output must be absent with an existing parent"
            )
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
    print(payload)
    return report.exit_code


__all__ = [
    "AUTHORIZED_EFFECTIVE_STEPS",
    "AUTHORIZED_TRAINING_SEED",
    "EXPECTED_INITIAL_MODEL_SHA256",
    "LEGACY_TRANSFER_POLICY_ID",
    "LEGACY_TRANSFER_STAGE_ID",
    "POLICY_ID",
    "REPORT_SCHEMA",
    "REPORT_VERSION",
    "REQUIRED_FIDELITY_POLICY",
    "STAGE_ID",
    "SUITE_SCHEMA",
    "SUITE_VERSION",
    "TrainingGateReport",
    "TrainingGateStatus",
    "TrainingGateValidationError",
    "TrainingSuite",
    "TRANSFER_AUTHORIZED_EFFECTIVE_STEPS",
    "TRANSFER_POLICY_ID",
    "TRANSFER_REQUIRED_FIDELITY_POLICY",
    "TRANSFER_STAGE_ID",
    "build_parser",
    "expected_training_request",
    "main",
    "training_request_mismatches",
    "verify_training_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
