"""Create and audit a current-interface model from the pre-fruit teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zuma_rl.model_migration import migrate_model_for_fruit_actor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--environment-contract", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args(argv)
    report_path = args.output_report.resolve(strict=False)
    if report_path.exists() or not report_path.parent.is_dir():
        raise ValueError("output report must be absent with an existing parent")
    report = migrate_model_for_fruit_actor(
        evidence_root=args.evidence_root,
        original_root=args.original_root,
        source_model=args.source_model,
        output_model=args.output_model,
        environment_contract=args.environment_contract,
    )
    data = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with report_path.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "report_sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "model": str(args.output_model.resolve()),
                "model_sha256": report["migrated_model"]["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
