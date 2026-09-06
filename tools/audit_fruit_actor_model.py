"""Independently audit an existing pre-fruit-to-current model migration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zuma_rl.model_migration import audit_model_migration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--migrated-model", required=True, type=Path)
    parser.add_argument("--environment-contract", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_report.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output report must be absent with an existing parent")
    report = audit_model_migration(
        evidence_root=args.evidence_root,
        original_root=args.original_root,
        source_model=args.source_model,
        migrated_model=args.migrated_model,
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
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": report["status"],
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "model_sha256": report["migrated_model"]["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
