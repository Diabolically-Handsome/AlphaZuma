"""Freeze the corrected V3 raw-intent DAgger certification controller."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_configured_successor_master_v2 as v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SUPERSEDED_MASTER = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-polar-intent-dagger-successor-"
    "s99081551-preregistration-v1.json"
)


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": v2.base._sha256(path)}


def build_master(
    *, successor_intent_path: Path, expected_intent_sha256: str
) -> dict[str, Any]:
    master = v2.build_master(
        successor_intent_path=successor_intent_path,
        expected_intent_sha256=expected_intent_sha256,
    )
    master["implementation"].update(
        {
            "master_builder": _artifact(SCRIPT_PATH),
            "controller": _artifact(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_configured_successor_certification_v3.py"
            ),
            "watcher": _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_certification_v3.py"
            ),
            "independent_auditor": _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result_v3.py"
            ),
        }
    )
    master["preflight_correction"] = {
        "supersedes_preflight_only_master": _artifact(SUPERSEDED_MASTER),
        "detected_before_controller_launch": True,
        "formal_seed_consumption_before_correction": False,
        "policy_inference_before_correction": False,
        "output_root_created_before_correction": False,
        "issue": "V2 wrapper omitted the required base.run poll_seconds argument",
        "correction": "forward poll_seconds to base.run",
        "candidate_order_changed": False,
        "ranking_changed": False,
        "seed_matrix_changed": False,
    }
    return master


def main(argv: list[str] | None = None) -> int:
    args = v2.base.build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 successor master: {output}")
    master = build_master(
        successor_intent_path=args.successor_intent.expanduser(),
        expected_intent_sha256=str(args.expected_intent_sha256),
    )
    v2.base._write_json_exclusive(output, master)
    print(
        json.dumps(
            {
                "status": master["status"],
                "output": {
                    "path": str(output),
                    "sha256": v2.base._sha256(output),
                },
                "supersedes": master["preflight_correction"][
                    "supersedes_preflight_only_master"
                ],
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
