#!/usr/bin/env python3
"""Migrate one frozen AlphaZuma model to the full55-v1 actor interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from zuma_rl.full55_migration import migrate_model_to_full55, sha256_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1_400_000_000)
    return parser


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")
    preregistration = json.loads(
        args.master_preregistration.read_text(encoding="utf-8")
    )
    if preregistration.get("campaign_id") != "alphazuma-55-weekend-s81081401-v1":
        raise ValueError("master preregistration campaign_id is invalid")
    report = migrate_model_to_full55(
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        original_root=args.original_root,
        source_id=args.source_id,
        seed=args.seed,
    )
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["master_preregistration"] = {
        "path": str(args.master_preregistration.resolve(strict=True)),
        "sha256": sha256_path(args.master_preregistration),
    }
    report["migration_tool"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": sha256_path(Path(__file__).resolve(strict=True)),
    }
    _write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_id": args.source_id,
                "migrated_model": report["migrated_model"],
                "proof": report["proof"],
                "report": str(args.report.resolve(strict=True)),
                "report_sha256": sha256_path(args.report),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
