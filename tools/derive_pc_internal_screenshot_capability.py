"""Create the fail-closed retail F11 capability report."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_internal_screenshot_capability import (
    derive_internal_screenshot_capability,
    write_capability_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_executable", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--post-message-probe", type=Path)
    parser.add_argument("--send-input-probe", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = derive_internal_screenshot_capability(
        runtime_executable=args.runtime_executable,
        post_message_probe=args.post_message_probe,
        send_input_probe=args.send_input_probe,
    )
    write_capability_report(report, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(args.output.resolve())
    print(f"sha256:{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
