"""Verify pinned CircleShootApp guidance against the retail Revenge runtime."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.source_guidance import main


if __name__ == "__main__":
    raise SystemExit(main())
