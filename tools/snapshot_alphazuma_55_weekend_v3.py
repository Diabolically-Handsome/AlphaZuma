"""Run the AlphaZuma 55 snapshot with V2/V3 postprocess process matching."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import snapshot_alphazuma_55_weekend as legacy


def _matching_processes(
    inventory: list[dict[str, Any]], *needles: str
) -> list[dict[str, Any]]:
    alternatives = [
        (needle, needle.replace("_v2.py", "_v3.py"))
        if needle.endswith("_v2.py")
        else (needle,)
        for needle in needles
    ]
    return [
        row
        for row in inventory
        if all(
            any(candidate in str(row["command"]) for candidate in candidates)
            for candidates in alternatives
        )
    ]


def main(argv: list[str] | None = None) -> int:
    legacy._matching_processes = _matching_processes
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
