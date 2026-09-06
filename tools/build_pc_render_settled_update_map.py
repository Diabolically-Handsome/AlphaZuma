"""Build a canonical render-settled framework-update map for one PC run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from zuma_rl.pc_golden import PcGoldenValidationError
from zuma_rl.pc_render_settle import recompute_render_settled_update_map


class BuildError(RuntimeError):
    """The source run cannot support a render-settled update map."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build(
    run_root: Path,
    *,
    output: Path,
    calibration_preregistration: Path,
    calibration_execution_binding: Path,
    calibration_holdout_report: Path,
) -> dict[str, object]:
    run_root = run_root.resolve()
    output = output.resolve()
    if not run_root.is_dir():
        raise BuildError("run root is missing")
    if output.exists() or output.parent != run_root:
        raise BuildError(
            "output must be a new file directly inside the run root"
        )
    try:
        update_map = recompute_render_settled_update_map(
            capture_metadata_path=run_root / "capture" / "metadata.json",
            frames_csv_path=run_root / "capture" / "frames.csv",
            framework_update_map_path=run_root / "framework-updates.json",
            framework_state_sidecar_path=(
                run_root / "framework-state-diagnostic.json"
            ),
            framework_poll_path=(
                run_root / "framework-state-poll-diagnostic.json"
            ),
            calibration_preregistration_path=(
                calibration_preregistration.resolve()
            ),
            calibration_execution_binding_path=(
                calibration_execution_binding.resolve()
            ),
            calibration_holdout_report_path=(
                calibration_holdout_report.resolve()
            ),
        )
        payload = update_map.to_json().encode("ascii")
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except (OSError, PcGoldenValidationError) as error:
        raise BuildError(str(error)) from error
    return {
        "status": "complete",
        "output": str(output),
        "sha256": _sha256_path(output),
        "frame_count": len(update_map.records),
        "first_assigned_framework_update": (
            update_map.records[0].assigned_framework_update
        ),
        "last_assigned_framework_update": (
            update_map.records[-1].assigned_framework_update
        ),
        "render_settle_delay_ns": update_map.render_settle_delay_ns,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a fixed-delay framework-update map from DXGI and "
            "high-rate draw evidence."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "new output path; defaults to "
            "RUN_ROOT/render-settled-updates.json"
        ),
    )
    parser.add_argument(
        "--calibration-preregistration",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--calibration-execution-binding",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--calibration-holdout-report",
        required=True,
        type=Path,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output = args.output or args.run_root / "render-settled-updates.json"
    try:
        report = build(
            args.run_root,
            output=output,
            calibration_preregistration=args.calibration_preregistration,
            calibration_execution_binding=(
                args.calibration_execution_binding
            ),
            calibration_holdout_report=args.calibration_holdout_report,
        )
    except BuildError as error:
        parser.error(str(error))
    for key, value in report.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
