"""Independently audit the frozen action-head-only DAgger preregistration."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
BASE = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081629-preregistration-v1.json"
)
BASE_SHA256 = (
    "sha256:11dd10f8537dfa12deb94c41ad3be3998a4204efec93a53fa97b8a04f21d0076"
)
CANDIDATE = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-preregistration-v1.json"
)
CANDIDATE_SHA256 = (
    "sha256:18b12280221b59588a65349812303ab58c115e859fdfb8e2eb674d222cd2ea72"
)
EXPECTED_INCIDENT_SHA256 = (
    "sha256:e957e188226f1b2b6f402e68c1da40a18d2c6c85c505973db885d7759bbc2bb5"
)
EXPECTED_GATE_AUDIT_SHA256 = (
    "sha256:e2f73963847962ca7de39e1bcf3d7ad6eed604145caf57e78d77d32223818ff6"
)
EXPECTED_TRAINER_SHA256 = (
    "sha256:ceb6e87d388f9ab6d03ba2c1f0194da4c7dd2d62344d70bc6a6f1c1ab30744ba"
)
EXPECTED_BUILDER_SHA256 = (
    "sha256:728d38dd8ec9dda5f42778b9e4bcef00e75f3aef15937e124badd2f198521c1b"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-independent-prereg-audit-v1.json"
)
EXPECTED_RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
)
EXPECTED_RANGE = (1_546_000_000, 1_546_000_219)
EXPECTED_TRAINABLE = ("action_net.weight", "action_net.bias")


def _read_strict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(
            stream,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _verify_reference(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference is not an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} hash differs: {path}")
    return path


def _normalize_candidate_to_base(
    *, base: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(candidate)
    for name in (
        "campaign_id",
        "created_utc",
        "trainer",
        "builder",
        "seed_registry",
        "outputs",
    ):
        value[name] = copy.deepcopy(base[name])
    value.pop("experimental_lineage", None)
    value.pop("predecessor_gate_audit", None)
    value["hypothesis"] = copy.deepcopy(base["hypothesis"])
    value["recovery_lineage"] = copy.deepcopy(base["recovery_lineage"])
    value["implementation"].pop("head_only_optimizer_adapter", None)

    for name in (
        "id",
        "run_dir",
        "device",
        "model_seed",
        "training_seed_base",
        "training_seed_last_consumed",
    ):
        value["run"][name] = copy.deepcopy(base["run"][name])
    for name in (
        "route_variant",
        "optimizer_parameter_scope",
        "trainable_parameter_names",
        "frozen_backbone",
        "optimizer_state_reset_before_first_update",
        "frozen_parameter_invariant_checked_after_each_round",
    ):
        value["run"].pop(name, None)
    return value


def audit(*, output: Path) -> dict[str, Any]:
    base_path = BASE.resolve(strict=True)
    candidate_path = CANDIDATE.resolve(strict=True)
    if legacy._sha256(base_path) != BASE_SHA256:
        raise ValueError("base preregistration hash differs")
    if legacy._sha256(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("candidate preregistration hash differs")
    base = _read_strict(base_path)
    candidate = _read_strict(candidate_path)

    trainer_path = _verify_reference(candidate.get("trainer"), "trainer")
    builder_path = _verify_reference(candidate.get("builder"), "builder")
    _verify_reference(candidate.get("master_preregistration"), "master")
    gate_path = _verify_reference(
        candidate.get("predecessor_gate_audit"), "predecessor gate audit"
    )
    for name, reference in candidate.get("implementation", {}).items():
        _verify_reference(reference, f"implementation {name}")
    lineage = candidate.get("experimental_lineage", {})
    incident_path = _verify_reference(
        lineage.get("training_seed_overlap_incident"),
        "training-seed overlap incident",
    )
    for name in (
        "fire_only_predecessor_preregistration",
        "fire_only_predecessor_trainer",
        "parallel_all_actions_route",
    ):
        _verify_reference(lineage.get(name), f"experimental lineage {name}")

    if legacy._sha256(trainer_path) != EXPECTED_TRAINER_SHA256:
        raise ValueError("head-only trainer hash differs")
    if legacy._sha256(builder_path) != EXPECTED_BUILDER_SHA256:
        raise ValueError("head-only builder hash differs")
    if legacy._sha256(incident_path) != EXPECTED_INCIDENT_SHA256:
        raise ValueError("seed-overlap incident hash differs")
    if legacy._sha256(gate_path) != EXPECTED_GATE_AUDIT_SHA256:
        raise ValueError("predecessor Gate audit hash differs")

    run = candidate.get("run", {})
    seed_registry = candidate.get("seed_registry", {})
    authority = candidate.get("authority_boundary", {})
    reuse = seed_registry.get("reuse_disclosure", {})
    if not (
        candidate.get("status") == "FROZEN_BEFORE_TRAINING"
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("route_variant") == "action_head_only_fire_aim_v1"
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == EXPECTED_TRAINABLE
        and run.get("frozen_backbone") is True
        and run.get("optimizer_state_reset_before_first_update") is True
        and run.get("frozen_parameter_invariant_checked_after_each_round")
        is True
        and run.get("device") == "cuda:0"
        and tuple(run.get("teacher_execution_probabilities", ()))
        == (1.0, 0.9, 0.7, 0.5)
        and int(run.get("training_seed_base", -1)) == EXPECTED_RANGE[0]
        and int(run.get("training_seed_last_consumed", -1))
        == EXPECTED_RANGE[1]
        and tuple(seed_registry.get("range", ())) == EXPECTED_RANGE
        and seed_registry.get("classification")
        == "engineering_training_seed_reuse_for_registered_ablation"
        and seed_registry.get("fresh_training_randomness_claim") is False
        and reuse.get("sha256") == EXPECTED_INCIDENT_SHA256
        and seed_registry.get("formal_selection_seed_consumption") == "NONE"
        and seed_registry.get("formal_final_blind_seed_consumption") == "NONE"
        and seed_registry.get("continuous_campaign_seed_consumption") == "NONE"
        and Path(str(run.get("run_dir", ""))) == EXPECTED_RUN_DIR
        and candidate.get("outputs", {}).get("run_dir")
        == str(EXPECTED_RUN_DIR)
        and authority == base.get("authority_boundary")
        and all(value is False for value in authority.values())
    ):
        raise ValueError("head-only substantive contract differs")
    if EXPECTED_RUN_DIR.exists():
        raise ValueError("head-only training directory already exists")

    normalized = _normalize_candidate_to_base(base=base, candidate=candidate)
    if normalized != base:
        raise ValueError("candidate contains an unregistered recipe difference")

    result = {
        "schema": "zuma-rl.alphazuma-55-headonly-prereg-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "auditor": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "base": {"path": str(base_path), "sha256": BASE_SHA256},
        "candidate": {
            "path": str(candidate_path),
            "sha256": CANDIDATE_SHA256,
        },
        "verified_runtime_bindings": {
            "trainer": {
                "path": str(trainer_path),
                "sha256": EXPECTED_TRAINER_SHA256,
            },
            "builder": {
                "path": str(builder_path),
                "sha256": EXPECTED_BUILDER_SHA256,
            },
            "training_seed_overlap_incident": {
                "path": str(incident_path),
                "sha256": EXPECTED_INCIDENT_SHA256,
            },
            "predecessor_gate_audit": {
                "path": str(gate_path),
                "sha256": EXPECTED_GATE_AUDIT_SHA256,
            },
        },
        "allowed_recipe_delta": {
            "optimizer_parameter_scope": ["full_policy", "action_net_only"],
            "optimizer_state_reset_required_by_scope_change": True,
            "training_seed_range_reused_with_disclosure": list(EXPECTED_RANGE),
            "run_identity_device_and_output_paths": True,
        },
        "all_other_recipe_fields_byte_equivalent_after_normalization": True,
        "run_directory_absent": True,
        "formal_seed_consumption": "NONE",
        "formal_candidate_authority": False,
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"audit receipt already exists: {output}")
    dagger._write_atomic(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(output=args.output)
    print(f"{result['status']} {legacy._sha256(args.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
