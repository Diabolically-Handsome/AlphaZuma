"""Freeze a finite source-bound full-state replay plan and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
from typing import Any, Mapping

from zuma_rl.pc_exact_step_evidence import canonical_json_bytes
from zuma_rl.pc_protocol_evidence import PcStateSnapshot
from zuma_rl.retail_dmo_provenance import (
    build_certifying_provenance,
    canonical_provenance_bytes,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _copy_file(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_file() or target.exists():
        raise ValueError("source-bound plan copy contract failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if _sha256_path(source) != _sha256_path(target):
        raise ValueError("source-bound plan copy verification failed")


def prepare_plan(
    *,
    base_plan_path: Path,
    host_pre_path: Path,
    output_root: Path,
    stop_after_update: int,
) -> dict[str, Any]:
    base_plan_path = base_plan_path.resolve(strict=True)
    host_pre_path = host_pre_path.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if (
        output_root.exists()
        or not output_root.parent.is_dir()
        or stop_after_update <= 0
    ):
        raise ValueError("source-bound output contract failed")
    base_root = base_plan_path.parent
    base_plan = _mapping(
        json.loads(base_plan_path.read_text(encoding="utf-8")),
        "base plan",
    )
    if (
        base_plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan"
        or base_plan.get("version") != 5
    ):
        raise ValueError("base plan is not a v5 PC Golden plan")
    plan = json.loads(json.dumps(base_plan))
    runtime = _mapping(plan.get("runtime"), "runtime")
    dmo = _mapping(plan.get("dmo"), "dmo")
    prestate = _mapping(plan.get("prestate"), "prestate")
    trace = _mapping(plan.get("trace"), "trace")
    anchor = _mapping(
        trace.get("source_bound_board_anchor"),
        "source-bound board anchor",
    )
    precall = _mapping(
        trace.get("source_bound_board_precall_global_restore"),
        "source-bound pre-call restore",
    )
    anchor_update = anchor.get("framework_update")
    if (
        runtime.get("launch_mode") != "direct_byte_identical_fixed_seed"
        or not isinstance(anchor_update, int)
        or isinstance(anchor_update, bool)
        or stop_after_update <= anchor_update
        or trace.get("allow_source_bound_board_global_correction") is not False
        or trace.get("seed_board_before_attach") is not False
        or not isinstance(precall.get("global_state"), Mapping)
    ):
        raise ValueError("base source-bound replay contract is invalid")

    staging = output_root.with_name(
        f".{output_root.name}.{secrets.token_hex(8)}.tmp"
    )
    if staging.exists():
        raise ValueError("source-bound staging path already exists")
    staging.mkdir()
    try:
        normalized_source = (
            base_root / str(dmo.get("artifact"))
        ).resolve(strict=True)
        normalized_target = staging / "inputs" / "input.dmo"
        _copy_file(normalized_source, normalized_target)
        if (
            normalized_target.stat().st_size != dmo.get("bytes")
            or _sha256_path(normalized_target) != dmo.get("sha256")
        ):
            raise ValueError("normalized DMO differs from base plan")

        template_source = (
            base_root / str(prestate.get("template_artifact"))
        ).resolve(strict=True)
        template_target = staging / "protocol" / "pre-template.json"
        _copy_file(template_source, template_target)
        plan["prestate"]["template_artifact"] = (
            "protocol/pre-template.json"
        )
        for overlay in plan["prestate"].get("overlays", []):
            source = (base_root / str(overlay["artifact"])).resolve(
                strict=True
            )
            target = staging / "inputs" / "prestate" / source.name
            _copy_file(source, target)
            overlay["artifact"] = f"inputs/prestate/{source.name}"
            provenance_name = overlay.get("provenance_artifact")
            if provenance_name is not None:
                provenance_source = (
                    base_root / str(provenance_name)
                ).resolve(strict=True)
                provenance_target = (
                    staging
                    / "inputs"
                    / "prestate"
                    / provenance_source.name
                )
                _copy_file(provenance_source, provenance_target)
                overlay["provenance_artifact"] = (
                    f"inputs/prestate/{provenance_source.name}"
                )

        host_pre_target = staging / "safety" / "host-pre.json"
        _copy_file(host_pre_path, host_pre_target)
        host_pre = PcStateSnapshot.read(host_pre_target)
        plan["safety"] = {
            "host_pre_artifact": "safety/host-pre.json",
            "host_pre_state_root": host_pre.state_root,
        }
        plan["dmo"]["artifact"] = "inputs/input.dmo"
        plan["dmo"]["provenance_artifact"] = None
        plan["dmo"]["provenance_sha256"] = None
        plan["session_nonce"] = secrets.token_hex(16)

        plan_trace = plan["trace"]
        plan_trace["attach_at_update"] = 0
        plan_trace["detach_at_update"] = None
        plan_trace["reattach_at_update"] = None
        plan_trace["source_bound_command_broker_stop_after_update"] = (
            stop_after_update
        )
        provenance_target = (
            staging / "inputs" / "input.transport.normalization.json"
        )
        plan_trace["source_bound_dmo_provenance_path"] = str(
            output_root / "inputs" / "input.transport.normalization.json"
        )
        plan_trace["allow_blackout_file_write_order_rebase"] = False
        plan_trace["allow_post_blackout_attach_stabilization"] = False
        plan_trace["allow_post_blackout_overdue_idle_reentry"] = False
        plan_trace["blackout_allowed_offset_pairs"] = []
        plan_trace["blackout_expected_command_order_offset"] = None
        plan_trace["blackout_expected_native_timeline_offset"] = None
        plan_trace["blackout_file_write_order_rebase_rows"] = []
        plan_trace["post_blackout_overdue_idle_reentry_candidates"] = []

        plan_path = staging / "plan.json"
        plan_data = canonical_json_bytes(plan)
        plan_path.write_bytes(plan_data)

        raw_dmo_path = Path(str(anchor["source_dmo_path"])).resolve(
            strict=True
        )
        recording_report_path = Path(
            str(anchor["recording_report_path"])
        ).resolve(strict=True)
        runtime_path = Path(str(runtime["runtime_executable"])).resolve(
            strict=True
        )
        if (
            _sha256_path(raw_dmo_path) != anchor["source_dmo_sha256"]
            or _sha256_path(recording_report_path)
            != anchor["recording_report_sha256"]
            or _sha256_path(runtime_path)
            != runtime["runtime_executable_sha256"]
        ):
            raise ValueError("base source artifacts differ")
        provenance = build_certifying_provenance(
            source_data=raw_dmo_path.read_bytes(),
            output_data=normalized_target.read_bytes(),
            recording_report_data=recording_report_path.read_bytes(),
            collector_plan_data=plan_data,
            expected_runtime_sha256=runtime["runtime_executable_sha256"],
        )
        provenance_target.write_bytes(
            canonical_provenance_bytes(provenance)
        )
        report = {
            "schema": "zuma-rl.pc-source-bound-plan-preparation",
            "version": 1,
            "status": "PASS",
            "base_plan": {
                "path": str(base_plan_path),
                "sha256": _sha256_path(base_plan_path),
            },
            "plan": {
                "path": str(output_root / "plan.json"),
                "sha256": _sha256_path(plan_path),
            },
            "normalized_dmo": {
                "path": str(output_root / "inputs" / "input.dmo"),
                "sha256": _sha256_path(normalized_target),
            },
            "retail_dmo_provenance": {
                "path": str(
                    output_root
                    / "inputs"
                    / "input.transport.normalization.json"
                ),
                "sha256": _sha256_path(provenance_target),
            },
            "host_pre": {
                "path": str(output_root / "safety" / "host-pre.json"),
                "sha256": _sha256_path(host_pre_target),
                "state_root": host_pre.state_root,
            },
            "source_bound_command_broker_stop_after_update": (
                stop_after_update
            ),
            "source_bound_framework_update": anchor_update,
            "global_precall_state_sha256": precall["global_state"][
                "state_sha256"
            ],
        }
        (staging / "preparation-report.json").write_bytes(
            canonical_json_bytes(report)
        )
        staging.rename(output_root)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--host-pre", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stop-after-update", type=int, default=3151)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = prepare_plan(
        base_plan_path=args.base_plan,
        host_pre_path=args.host_pre,
        output_root=args.output_root,
        stop_after_update=args.stop_after_update,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
