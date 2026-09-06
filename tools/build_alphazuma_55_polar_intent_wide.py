"""Freeze wide-polar distillation from raw teacher intent labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_polar_wide as effective_builder


SCRIPT_PATH = Path(__file__).resolve()
CHECKPOINT_EPOCHS = (10, 20, 30)


def build(
    *,
    master_path: Path,
    teacher_result_path: Path,
    run_dir: Path,
    device: str,
    training_seed_base: int,
    model_seed: int,
    validation_seed_base: int,
) -> dict[str, Any]:
    value = effective_builder.build(
        master_path=master_path,
        teacher_result_path=teacher_result_path,
        run_dir=run_dir,
        device=device,
        training_seed_base=training_seed_base,
        model_seed=model_seed,
        validation_seed_base=validation_seed_base,
    )
    root = SCRIPT_PATH.parents[1]
    trainer = root / "tools/distill_alphazuma_55_polar_intent_wide_v1.py"
    effective_runtime = root / "tools/distill_alphazuma_55_polar_wide_v1.py"
    value.update(
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-preregistration"
            ),
            "objective": (
                "Test whether learning the teacher's persistent raw verb intent, "
                "while preserving exact masks for environment execution and all "
                "online evaluation, removes the narrow legal-action-window "
                "failure of effective-label cloning."
            ),
            "decision_evidence": {
                "effective_label_clone_online_failure": {
                    "dagger_round_01_offline_verb_accuracy": 0.815717,
                    "dagger_round_01_fire_aim_within_three_accuracy": 0.619508,
                    "dagger_round_01_online_wins": 3,
                    "dagger_round_01_online_attempts": 55,
                },
                "teacher_intent_sparsification_example": {
                    "level_id": "volcano9",
                    "seed": 1542001253,
                    "raw_fire_requests": 1721,
                    "effective_fire_actions": 156,
                    "mask_forced_waits": 2622,
                    "classification": (
                        "adaptive engineering evidence observed before this "
                        "preregistration"
                    ),
                },
            },
            "hypotheses": {
                "alternative": (
                    "raw-intent supervision produces persistent action logits; "
                    "the unchanged exact online mask then executes fire, swap, or "
                    "hop on the first legal frame and improves full55 coverage"
                ),
                "null": (
                    "the online failure is caused by representation or recovery "
                    "state distribution rather than sparse effective labels, so "
                    "raw-intent supervision does not improve coverage"
                ),
            },
            "builder": effective_builder.base._artifact(SCRIPT_PATH),
            "trainer": effective_builder.base._artifact(trainer),
        }
    )
    value["implementation"]["effective_wide_builder"] = (
        effective_builder.base._artifact(
            root / "tools/build_alphazuma_55_polar_wide.py"
        )
    )
    value["implementation"]["effective_wide_runtime"] = (
        effective_builder.base._artifact(effective_runtime)
    )
    run = value["run"]
    run.update(
        {
            "id": "alphazuma55-polar-intent-wide-5090",
            "policy_architecture": "entity_polar_intent_wide",
            "epochs_per_round": 30,
            "raw_teacher_intent_labels": True,
            "training_mask_relaxation": "unmask_only_the_teacher_target_verb",
            "exact_action_masks_for_environment_execution": True,
            "exact_action_masks_for_online_policy_inference": True,
            "invalid_intent_execution": "unchanged_wait_fallback",
            "label_aim_must_remain_exact_mask_legal": True,
        }
    )
    frozen = value["frozen_engineering_validation"]
    frozen["checkpoint_epochs"] = list(CHECKPOINT_EPOCHS)
    frozen["comparison_model_id"] = "polar-source-final"
    frozen["candidate_model_ids"] = [
        "polar-source-final",
        *[f"polar-intent-wide-epoch-{epoch:02d}" for epoch in CHECKPOINT_EPOCHS],
    ]
    value["authority_boundary"]["classification"] = (
        "isolated_adaptive_intent_label_engineering_training_only"
    )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--teacher-result", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--training-seed-base", required=True, type=int)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--validation-seed-base", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite intent-wide prereg: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser(),
        teacher_result_path=args.teacher_result.expanduser(),
        run_dir=args.run_dir.expanduser(),
        device=args.device,
        training_seed_base=args.training_seed_base,
        model_seed=args.model_seed,
        validation_seed_base=args.validation_seed_base,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {
                    "path": str(output),
                    "sha256": effective_builder.base._sha256(output),
                },
                "run": value["run"],
                "validation": value["frozen_engineering_validation"],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
