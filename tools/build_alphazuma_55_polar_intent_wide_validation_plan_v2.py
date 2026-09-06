"""Freeze a race-free successor to the intent-wide validation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_polar_intent_wide_validation_plan as legacy
from tools.build_alphazuma_55_eval_contract import (
    _sha256,
    _write_json_exclusive,
)


SCRIPT_PATH = Path(__file__).resolve()
WATCHER = SCRIPT_PATH.with_name(
    "watch_alphazuma_55_polar_intent_wide_validation_v2.py"
)


def build(args: argparse.Namespace) -> dict[str, object]:
    supersedes = args.supersedes_plan.expanduser().resolve(strict=True)
    if _sha256(supersedes) != str(args.expected_supersedes_sha256):
        raise ValueError("superseded validation plan hash differs")
    original_script = legacy.SCRIPT_PATH
    original_watcher = legacy.WATCHER
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.WATCHER = WATCHER
        value = legacy.build(
            master_path=args.master_preregistration.expanduser(),
            training_preregistration_path=(
                args.training_preregistration.expanduser()
            ),
            launch_plan_path=args.launch_plan.expanduser(),
            source_training_preregistration_path=(
                args.source_training_preregistration.expanduser()
            ),
            source_completion_path=args.source_completion.expanduser(),
            original_root=args.original_root.expanduser(),
            output_root=args.output_root.expanduser(),
            manifest_output=args.models_manifest.expanduser(),
            evaluation_preregistration_output=(
                args.evaluation_preregistration.expanduser()
            ),
            audit_output=args.audit_output.expanduser(),
            prior_wide_audit_path=args.prior_wide_audit.expanduser(),
            status_root=args.status_root.expanduser(),
            device=args.device,
            parallel_envs=args.parallel_envs,
        )
    finally:
        legacy.SCRIPT_PATH = original_script
        legacy.WATCHER = original_watcher
    value["supersedes"] = {
        "path": str(supersedes),
        "sha256": _sha256(supersedes),
        "reason": (
            "Wait for the intent trainer launch controller to become terminal "
            "before reading the wrapper-rewritten completion receipt."
        ),
        "superseded_plan_formal_seed_consumption": "NONE",
    }
    value["race_fix"] = {
        "wait_for_launch_controller_terminal_before_completion_read": True,
        "training_recipe_changed": False,
        "model_candidates_changed": False,
        "seed_matrix_changed": False,
    }
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = legacy.build_parser()
    parser.add_argument("--supersedes-plan", required=True, type=Path)
    parser.add_argument("--expected-supersedes-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 validation plan: {output}")
    value = build(args)
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "supersedes": value["supersedes"],
                "race_fix": value["race_fix"],
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
