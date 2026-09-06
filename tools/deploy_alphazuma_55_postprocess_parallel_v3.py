"""Deploy the capacity-safe AlphaZuma 55 postprocess contract.

This versioned shim preserves the mature V2 deployment lifecycle while
binding validation to this executable.  The frozen deployment document still
selects the builder, controller, auditor, roots, and seed semantics.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import deploy_alphazuma_55_postprocess_parallel_v2 as legacy


SCRIPT_PATH = Path(__file__).resolve()


def main(argv: list[str] | None = None) -> int:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
