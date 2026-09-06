"""Exclusively create the current recomputable training-environment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zuma_rl.environment_contract import build_training_environment_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent with an existing parent")
    report = build_training_environment_contract(
        original_root=args.original_root,
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
                "status": "CREATED",
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "contract_fingerprint": report["contract_fingerprint"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
