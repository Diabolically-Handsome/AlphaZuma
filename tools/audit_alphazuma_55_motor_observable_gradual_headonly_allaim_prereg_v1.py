"""Independently audit the head-only plus all-actions preregistration."""

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
BASE = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)
BASE_SHA256 = (
    "sha256:502c0ab5566a66fa33ff33df80b10b101c5ea1b75dc4e21ff07f5996965730e1"
)
CANDIDATE = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
    "s99081646-preregistration-v1.json"
)
CANDIDATE_SHA256 = (
    "sha256:bcede95c539b97ea9a440ccee5f68e8715ce6b3dd3c9ad3eac2b7651d28c3470"
)
EXPECTED_TRAINER_SHA256 = (
    "sha256:a1087452f49bb1fdc516394e1e9f07a6e8955490ddfd5025798dbdd2710ec372"
)
EXPECTED_BUILDER_SHA256 = (
    "sha256:33547d3149a64aec6972e28910455a3a388cb410f05d3522510b771a9f554a4a"
)
EXPECTED_HEADONLY_PARENT_SHA256 = (
    "sha256:18b12280221b59588a65349812303ab58c115e859fdfb8e2eb674d222cd2ea72"
)
EXPECTED_HEADONLY_AUDIT_SHA256 = (
    "sha256:9efd8a02d81df0cce8cd9f339d3f16faf1012b8ccca358194bdeafef5bd5be09"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
    "s99081646-independent-prereg-audit-v1.json"
)
EXPECTED_RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
)
EXPECTED_RANGE = (1_546_001_000, 1_546_001_219)
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


def _normalize_to_allaim_base(
    *, base: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(candidate)
    for name in (
        "campaign_id",
        "created_utc",
        "trainer",
        "builder",
        "hypothesis",
        "experimental_lineage",
        "seed_registry",
        "outputs",
    ):
        value[name] = copy.deepcopy(base[name])
    value["implementation"].pop("headonly_allaim_factorial_adapter", None)
    for name in (
        "id",
        "run_dir",
        "device",
        "model_seed",
        "training_seed_base",
        "training_seed_last_consumed",
        "route_variant",
    ):
        value["run"][name] = copy.deepcopy(base["run"][name])
    for name in (
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
        raise ValueError("all-actions parent preregistration hash differs")
    if legacy._sha256(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("factorial candidate preregistration hash differs")
    base = _read_strict(base_path)
    candidate = _read_strict(candidate_path)

    trainer_path = _verify_reference(candidate.get("trainer"), "trainer")
    builder_path = _verify_reference(candidate.get("builder"), "builder")
    _verify_reference(candidate.get("master_preregistration"), "master")
    for name, reference in candidate.get("implementation", {}).items():
        _verify_reference(reference, f"implementation {name}")
    lineage = candidate.get("experimental_lineage", {})
    allaim_path = _verify_reference(lineage.get("all_actions_parent"), "all-actions parent")
    headonly_path = _verify_reference(lineage.get("head_only_parent"), "head-only parent")
    headonly_audit_path = _verify_reference(
        lineage.get("head_only_parent_independent_prereg_audit"),
        "head-only parent independent prereg audit",
    )

    if legacy._sha256(trainer_path) != EXPECTED_TRAINER_SHA256:
        raise ValueError("factorial trainer hash differs")
    if legacy._sha256(builder_path) != EXPECTED_BUILDER_SHA256:
        raise ValueError("factorial builder hash differs")
    if legacy._sha256(allaim_path) != BASE_SHA256:
        raise ValueError("all-actions lineage hash differs")
    if legacy._sha256(headonly_path) != EXPECTED_HEADONLY_PARENT_SHA256:
        raise ValueError("head-only lineage hash differs")
    if legacy._sha256(headonly_audit_path) != EXPECTED_HEADONLY_AUDIT_SHA256:
        raise ValueError("head-only independent audit hash differs")

    run = candidate.get("run", {})
    seed_registry = candidate.get("seed_registry", {})
    authority = candidate.get("authority_boundary", {})
    proof = seed_registry.get("subrange_proof", {})
    if not (
        candidate.get("status") == "FROZEN_BEFORE_TRAINING"
        and run.get("aim_loss_scope") == "all_actions"
        and run.get("all_action_aim_supervision") is True
        and run.get("aim_loss_sample_semantics")
        == "raw_teacher_intent_on_every_retained_action"
        and run.get("route_variant")
        == "action_head_only_all_actions_aim_v1"
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == EXPECTED_TRAINABLE
        and run.get("frozen_backbone") is True
        and run.get("optimizer_state_reset_before_first_update") is True
        and run.get("frozen_parameter_invariant_checked_after_each_round")
        is True
        and run.get("device") == "cuda:1"
        and tuple(run.get("teacher_execution_probabilities", ()))
        == (1.0, 0.9, 0.7, 0.5)
        and int(run.get("training_seed_base", -1)) == EXPECTED_RANGE[0]
        and int(run.get("training_seed_last_consumed", -1))
        == EXPECTED_RANGE[1]
        and tuple(seed_registry.get("range", ())) == EXPECTED_RANGE
        and seed_registry.get("classification")
        == "fresh_engineering_training_subrange"
        and proof.get("master_training_range")
        == [1_500_000_000, 1_549_999_999]
        and int(proof.get("previous_current_route_last", -1))
        == 1_546_000_219
        and proof.get("strictly_after_current_routes") is True
        and seed_registry.get("formal_selection_seed_consumption") == "NONE"
        and seed_registry.get("formal_final_blind_seed_consumption") == "NONE"
        and seed_registry.get("continuous_campaign_seed_consumption") == "NONE"
        and Path(str(run.get("run_dir", ""))) == EXPECTED_RUN_DIR
        and candidate.get("outputs", {}).get("run_dir")
        == str(EXPECTED_RUN_DIR)
        and authority == base.get("authority_boundary")
        and all(value is False for value in authority.values())
    ):
        raise ValueError("factorial substantive contract differs")
    if EXPECTED_RUN_DIR.exists():
        raise ValueError("factorial training directory already exists")

    normalized = _normalize_to_allaim_base(base=base, candidate=candidate)
    if normalized != base:
        raise ValueError("factorial candidate contains an unregistered delta")

    result = {
        "schema": "zuma-rl.alphazuma-55-headonly-allaim-prereg-independent-audit",
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
        "allowed_recipe_delta": {
            "optimizer_parameter_scope": ["full_policy", "action_net_only"],
            "fresh_training_seed_range": list(EXPECTED_RANGE),
            "run_identity_device_and_output_paths": True,
        },
        "factorial_lineage_verified": True,
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
