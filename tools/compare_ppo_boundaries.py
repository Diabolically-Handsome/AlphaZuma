#!/usr/bin/env python3
"""Write an identity-aware comparison of two PPO boundary receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from zuma_rl.boundary_evidence import compare_boundary_receipts


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pretrain", type=Path)
    parser.add_argument("posttrain", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, required=True)
    parser.add_argument("--severe-regression-threshold", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    pretrain = json.loads(args.pretrain.read_text(encoding="utf-8"))
    posttrain = json.loads(args.posttrain.read_text(encoding="utf-8"))
    comparison = compare_boundary_receipts(
        pretrain,
        posttrain,
        max_ticks=args.max_ticks,
        severe_regression_threshold=args.severe_regression_threshold,
    )
    comparison["created_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    comparison["inputs"] = {
        "pretrain": {
            "path": str(args.pretrain.resolve()),
            "sha256": _sha256(args.pretrain),
        },
        "posttrain": {
            "path": str(args.posttrain.resolve()),
            "sha256": _sha256(args.posttrain),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
