"""Derive a hash-bound retail ShadowCanopy shared-RNG scope boundary."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_ambient_rng_boundary import (
    AmbientRngBoundaryError,
    derive_ambient_rng_boundary,
    sha256_path,
)
from zuma_rl.pc_memory_trajectory import PcMemoryTrajectoryError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--memory-probe", required=True, type=Path)
    parser.add_argument("--pc-golden-manifest", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--last-included-update", required=True, type=int)
    parser.add_argument("--first-excluded-update", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.resolve()
    try:
        if output.exists() or not output.parent.is_dir():
            raise AmbientRngBoundaryError("ambient_boundary_output_invalid")
        report = derive_ambient_rng_boundary(
            trajectory_path=arguments.trajectory,
            memory_probe_path=arguments.memory_probe,
            pc_golden_manifest_path=arguments.pc_golden_manifest,
            trace_path=arguments.trace,
            runtime_executable_path=arguments.runtime,
            last_included_update=arguments.last_included_update,
            first_excluded_update=arguments.first_excluded_update,
        )
        payload = (
            json.dumps(
                report,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        output.write_bytes(payload)
        print(output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "last_included_update": report[
                        "last_included_update"
                    ],
                    "first_excluded_update": report[
                        "first_excluded_update"
                    ],
                    "sha256": sha256_path(output),
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        OSError,
        AmbientRngBoundaryError,
        PcMemoryTrajectoryError,
    ) as error:
        print(f"ambient RNG boundary error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
