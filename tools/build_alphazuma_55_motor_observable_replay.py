"""Freeze the first actor-visible motor-state replay route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.observable_human_speedrun import MOTOR_OBSERVATION_PROFILE


SCRIPT_PATH = Path(__file__).resolve()
CAMPAIGN_ID = "alphazuma-55-motor-observable-replay-s99081613-v1"
ANCHOR_SEED_BASE = 1_543_000_000
ANCHOR_SEED_LAST = 1_543_000_164
STUDENT_SEED_BASE = 1_543_000_165
STUDENT_SEED_LAST = 1_543_000_329
MODEL_SEED = 1_543_000_500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"preregistration already exists: {path}")
    os.replace(temporary, path)


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    master_path = (
        root
        / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
    )
    master = legacy._read_json(master_path)
    source_model = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-wide-"
        "s99081540-v1/epoch_10_model.zip"
    )
    source_audit = (
        root
        / "diagnostics/alphazuma-55-polar-intent-wide-validation-"
        "s99081544-audit-v1.json"
    )
    source_audit_value = legacy._read_json(source_audit)
    source_hash = legacy._sha256(source_model.resolve(strict=True))
    selected = next(
        row
        for row in source_audit_value["model_summaries"]
        if row["model_id"] == "polar-intent-wide-epoch-10"
    )
    if not (
        source_audit_value.get("status") == "PASS"
        and selected.get("model_sha256") == source_hash
        and int(selected.get("wins", -1)) == 6
        and int(selected.get("cleared_levels", -1)) == 6
    ):
        raise ValueError("frozen epoch-10 source evidence changed")

    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= ANCHOR_SEED_BASE
        <= STUDENT_SEED_LAST
        <= int(training["last"])
        and ANCHOR_SEED_LAST
        == ANCHOR_SEED_BASE + 3 * len(INCLUDED_LEVELS) - 1
        and STUDENT_SEED_BASE == ANCHOR_SEED_LAST + 1
        and STUDENT_SEED_LAST
        == STUDENT_SEED_BASE + 3 * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("motor-observable training seed interval is invalid")

    run_dir = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    preregistration = {
        "schema": (
            "zuma-rl.alphazuma-55-motor-observable-replay-preregistration"
        ),
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "objective": (
            "Test whether exposing the unchanged elite-human actuator's "
            "deterministic motor state repairs the offline-to-closed-loop gap."
        ),
        "decision_evidence_available_before_freeze": {
            "paired_epoch_curve": {
                "epoch_10_wins": 6,
                "epoch_20_wins": 3,
                "epoch_30_wins": 1,
                "attempts_per_checkpoint": 55,
                "classification": "consumed training-validation evidence",
            },
            "pure_student_autonomy": {
                "wins": 0,
                "attempts": 220,
                "classification": "consumed engineering evidence",
            },
            "replay_anchor_round_02_partial": {
                "wins": 0,
                "attempts_observed": 48,
                "teacher_execution_probability": 0.0,
                "classification": "consumed engineering evidence",
            },
            "interface_finding": (
                "HumanSpeedrunWrapper leaves aim velocity, delayed aim, aim "
                "queue, button queue, cooldown, and previous desired verb "
                "outside the actor observation although they affect the next "
                "executed action."
            ),
        },
        "hypotheses": {
            "alternative": (
                "actor-visible deterministic motor state materially improves "
                "autonomous closed-loop coverage"
            ),
            "null": (
                "the remaining failure is dominated by representation, "
                "objective, or capacity rather than partial observability"
            ),
        },
        "master_preregistration": _reference(master_path),
        "builder": _reference(SCRIPT_PATH),
        "trainer": _reference(
            root
            / "tools/distill_alphazuma_55_motor_observable_replay_v1.py"
        ),
        "implementation": {
            "motor_observation_wrapper": _reference(
                root / "src/zuma_rl/observable_human_speedrun.py"
            ),
            "migration": _reference(
                root / "src/zuma_rl/motor_observable_migration.py"
            ),
            "human_input_wrapper": _reference(
                root / "src/zuma_rl/human_speedrun.py"
            ),
            "legacy_distillation_runtime": _reference(
                root / "tools/distill_alphazuma_55.py"
            ),
            "settled_v3_runtime": _reference(
                root / "tools/distill_alphazuma_55_settled_v3.py"
            ),
            "polar_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_v1.py"
            ),
            "polar_wide_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_wide_v1.py"
            ),
            "raw_intent_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_intent_wide_v1.py"
            ),
            "raw_intent_dagger_runtime": _reference(
                root
                / "tools/distill_alphazuma_55_polar_intent_dagger_v1.py"
            ),
            "features": _reference(root / "src/zuma_rl/revenge_features.py"),
            "environment": _reference(root / "src/zuma_rl/revenge_env.py"),
            "core": _reference(root / "src/zuma_rl/revenge_core.py"),
            "scope": _reference(root / "src/zuma_rl/alphazuma_55.py"),
        },
        "source": {
            "model_path": str(source_model),
            "model_sha256": source_hash,
            "selection_audit_path": str(source_audit.resolve(strict=True)),
            "selection_audit_sha256": legacy._sha256(source_audit),
            "model_id": "polar-intent-wide-epoch-10",
            "selection_rule": (
                "best wins and coverage in the already-consumed paired "
                "epoch-10/20/30 engineering matrix; fixed before this route"
            ),
        },
        "levels": list(INCLUDED_LEVELS),
        "run": {
            "id": "alphazuma55-motor-observable-replay-5080",
            "run_dir": str(run_dir),
            "device": "cuda:1",
            "policy_architecture": (
                "entity_polar_intent_motor_observable_replay"
            ),
            "motor_observation_profile": MOTOR_OBSERVATION_PROFILE,
            "learned_features_dim": 1024,
            "model_seed": MODEL_SEED,
            "anchor_seed_base": ANCHOR_SEED_BASE,
            "anchor_seed_last_consumed": ANCHOR_SEED_LAST,
            "student_seed_base": STUDENT_SEED_BASE,
            "student_seed_last_consumed": STUDENT_SEED_LAST,
            "anchor_teacher_passes": 3,
            "student_rounds": 3,
            "teacher_execution_probabilities": [0.5, 0.1, 0.0],
            "parallel_envs": 24,
            "max_ticks": 30000,
            "reservoir_capacity_by_verb": {
                "wait": 256,
                "fire": 256,
                "swap": 192,
                "hop": 192,
            },
            "sample_stride_by_verb": {
                "wait": 4,
                "fire": 1,
                "swap": 1,
                "hop": 1,
            },
            "target_batch_fraction_by_verb": {
                "wait": 0.45,
                "fire": 0.35,
                "swap": 0.15,
                "hop": 0.05,
            },
            "batch_size": 512,
            "capacity_eval_batch_size": 128,
            "epochs_per_round": 4,
            "checkpoint_interval_epochs": 1,
            "learning_rate": 0.00001,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "aim_loss_scope": "fire_only",
            "aim_loss_mode": "circular_smoothed",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.5,
            "aim_head_identity_scale": 5.0,
            "teacher_anchor_replayed_every_round": True,
            "aggregate_student_states_across_rounds": True,
            "raw_teacher_intent_labels": True,
            "training_mask_relaxation": (
                "unmask_only_the_teacher_target_verb"
            ),
            "exact_action_masks_for_environment_execution": True,
            "exact_action_masks_for_online_policy_inference": True,
            "label_aim_must_remain_exact_mask_legal": True,
            "final_runtime_teacher_calls_forbidden": True,
        },
        "frozen_engineering_validation": {
            "registry": "training_validation",
            "checkpoint_screen": {
                "base_seed": 1_550_002_400,
                "last_seed": 1_550_002_411,
                "levels": 12,
                "candidate_rule": (
                    "source plus every per-epoch checkpoint; rank without "
                    "using the full55 gate seeds"
                ),
            },
            "full55_gate": {
                "base_seed": 1_550_002_500,
                "last_seed": 1_550_002_554,
                "levels": 55,
                "maximum_candidates": 5,
                "minimum_wins": 35,
                "minimum_cleared_levels": 35,
            },
            "formal_seed_consumption": False,
        },
        "authority_boundary": {
            "classification": (
                "motor_observable_engineering_training_only"
            ),
            "current_campaign_candidate_authority": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }
    _write_new(output, preregistration)
    return preregistration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    preregistration = build(
        project_root=args.project_root,
        output=args.output,
    )
    print(
        json.dumps(
            preregistration,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
