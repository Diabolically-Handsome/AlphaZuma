"""Recompute and exclusively freeze the installed original-asset audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zuma_rl.original_asset_audit import audit_original_assets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--profile-mode", default="tutorials_completed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent with an existing parent")
    report = audit_original_assets(
        original_root=args.original_root,
        level_id=args.level,
        hard=args.hard,
        profile_mode=args.profile_mode,
    )
    data = (
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": report["status"],
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
