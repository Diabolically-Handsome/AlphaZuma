"""Transfer the no-inference intent to the race-free validation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_polar_intent_wide_successor_intent as legacy
from tools.build_alphazuma_55_eval_contract import (
    _read_json,
    _sha256,
    _write_json_exclusive,
)


SCRIPT_PATH = Path(__file__).resolve()


def build(
    *,
    source_master_path: Path,
    validation_plan_path: Path,
    sibling_wide_intent_path: Path,
    superseded_intent_path: Path,
    expected_superseded_sha256: str,
    output_root: Path,
    independent_audit_receipt: Path,
    campaign_id: str,
) -> dict[str, Any]:
    superseded_intent_path = superseded_intent_path.resolve(strict=True)
    if _sha256(superseded_intent_path) != expected_superseded_sha256:
        raise ValueError("superseded successor intent hash differs")
    superseded = _read_json(superseded_intent_path)
    if (
        superseded.get("schema")
        != "zuma-rl.alphazuma-55-polar-intent-wide-successor-intent"
        or superseded.get("status") != "FROZEN_BEFORE_ENGINEERING_RESULT"
        or superseded.get("implementation_boundary", {}).get(
            "this_intent_does_not_authorize_policy_inference"
        )
        is not True
    ):
        raise ValueError("unexpected superseded successor intent")
    old_output_root = Path(str(superseded["outputs"]["root"])).resolve()
    old_audit = Path(
        str(superseded["outputs"]["independent_audit_receipt"])
    ).resolve()
    if old_output_root.exists() or old_audit.exists():
        raise ValueError("superseded intent has result artifacts")
    original_script = legacy.SCRIPT_PATH
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        value = legacy.build(
            source_master_path=source_master_path,
            validation_plan_path=validation_plan_path,
            sibling_wide_intent_path=sibling_wide_intent_path,
            output_root=output_root,
            independent_audit_receipt=independent_audit_receipt,
            campaign_id=campaign_id,
        )
    finally:
        legacy.SCRIPT_PATH = original_script
    old_formal = superseded["formal_campaign_if_promoted"]
    new_formal = value["formal_campaign_if_promoted"]
    if (
        old_formal["final_blind"] != new_formal["final_blind"]
        or old_formal["continuous_campaign_challenge"]
        != new_formal["continuous_campaign_challenge"]
    ):
        raise ValueError("V2 intent changed the reserved formal matrices")
    value["supersedes"] = {
        "path": str(superseded_intent_path),
        "sha256": expected_superseded_sha256,
        "reason": "Bind the race-free V2 engineering validation plan.",
        "superseded_intent_policy_inference": "NONE",
        "superseded_intent_formal_seed_consumption": "NONE",
    }
    value["seed_reservation_transfer"] = {
        "final_blind_range_unchanged": True,
        "continuous_range_unchanged": True,
        "old_output_root_absent": str(old_output_root),
        "old_independent_audit_absent": str(old_audit),
    }
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = legacy.build_parser()
    parser.add_argument("--superseded-intent", required=True, type=Path)
    parser.add_argument("--expected-superseded-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 successor intent: {output}")
    value = build(
        source_master_path=args.source_master.expanduser(),
        validation_plan_path=args.intent_validation_plan.expanduser(),
        sibling_wide_intent_path=args.sibling_wide_intent.expanduser(),
        superseded_intent_path=args.superseded_intent.expanduser(),
        expected_superseded_sha256=str(args.expected_superseded_sha256),
        output_root=args.output_root.expanduser(),
        independent_audit_receipt=args.independent_audit_receipt.expanduser(),
        campaign_id=str(args.campaign_id),
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "supersedes": value["supersedes"],
                "seed_reservation_transfer": value["seed_reservation_transfer"],
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
