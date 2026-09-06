"""Derive a fail-closed native proof for gameplay MTRand reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_mtrand_reconciliation import (
    derive_mtrand_reconciliation_proof,
    sha256_path,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameplay-diff", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--trace-window-root",
        action="append",
        required=True,
        dest="trace_window_roots",
        help=(
            "Normalized POSIX path relative to --evidence-root; repeat once "
            "per immutable trace window."
        ),
    )
    parser.add_argument(
        "--runtime-artifact",
        default="tools/direct-runtime/popcapgame1.exe",
        help="Normalized POSIX path relative to --evidence-root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    try:
        raw = json.loads(args.gameplay_diff.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("gameplay diff JSON is invalid") from error
    if not isinstance(raw, dict):
        raise ValueError("gameplay diff JSON root must be an object")
    proof = derive_mtrand_reconciliation_proof(
        raw,
        evidence_root=args.evidence_root,
        trace_window_roots=args.trace_window_roots,
        runtime_artifact=args.runtime_artifact,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)
    print(
        json.dumps(
            {
                "sha256": sha256_path(output),
                "reconciled_tick_count": proof["reconciliation"][
                    "reconciled_tick_count"
                ],
                "reconciled_draw_count": proof["reconciliation"][
                    "reconciled_draw_count"
                ],
                "native_call_count": proof["reconciliation"][
                    "native_call_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
