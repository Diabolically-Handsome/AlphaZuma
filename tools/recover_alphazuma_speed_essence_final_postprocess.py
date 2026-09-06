"""Recover the frozen speed-essence final postprocess after a CLI-label failure.

The frozen final-blind inference shards are not rerun.  The frozen summarizer
module is loaded byte-for-byte and only its argparse result is supplied
programmatically so that the already-requested ``final-blind`` purpose reaches
the unchanged aggregation logic.  All remaining artifacts are then produced
with the frozen controller helpers and independently auditable receipts.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import run_alphazuma_speed_essence_postprocess as controller
except ModuleNotFoundError:  # Imported as tools.<module>.
    from tools import run_alphazuma_speed_essence_postprocess as controller


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recovery artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_frozen_summarizer(
    *, summarizer_path: Path, preregistration: Path, manifest: Path, run_directory: Path
) -> str:
    spec = importlib.util.spec_from_file_location(
        "alphazuma_frozen_summarizer_recovery", summarizer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen summarizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_parser = lambda: SimpleNamespace(
        parse_args=lambda: argparse.Namespace(
            preregistration=preregistration,
            models_manifest=manifest,
            run_directory=run_directory,
            purpose="final-blind",
        )
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        module.main()
    return stdout.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=True)
    execution_receipt_path = args.execution_receipt.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    status_path = (output_root / "controller_status.json").resolve(strict=True)
    state = _read_json(status_path)
    if state.get("status") != "FAILED" or state.get("phase") != "FAILED":
        raise ValueError("recovery requires the preserved FAILED controller state")

    contract = controller._load_contract(master_path)
    master = contract["master"]
    execution = controller._validate_execution_receipt(
        execution_receipt_path,
        master_path=master_path,
        output_root=output_root,
    )
    final_root = Path(str(master["execution"]["final_output_root"])).resolve(strict=True)
    final_prereg = (output_root / "final-blind-preregistration.json").resolve(strict=True)
    final_manifest = (output_root / "final-blind-models-manifest.json").resolve(strict=True)
    selection_path = (output_root / "selection-decision.json").resolve(strict=True)
    summarizer_path = Path(str(execution["summarizer"]["path"])).resolve(strict=True)
    stderr_path = (final_root / "summarizer.stderr.log").resolve(strict=True)
    stdout_path = (final_root / "summarizer.stdout.json").resolve(strict=True)

    shard_receipts = []
    for index in range(2):
        path = (final_root / f"matrix-shard-{index:02d}-of-02.json").resolve(strict=True)
        shard = _read_json(path)
        if (
            shard.get("status") != "COMPLETE"
            or shard.get("error") is not None
            or int(shard.get("completed_attempts", -1))
            != int(shard.get("expected_attempts", -2))
        ):
            raise ValueError(f"final shard is not complete and clean: {path}")
        shard_receipts.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "completed_attempts": int(shard["completed_attempts"]),
                "expected_attempts": int(shard["expected_attempts"]),
            }
        )

    evidence_path = output_root / "final-blind-summarizer-failure-evidence.json"
    evidence = {
        "schema": "zuma-rl.alphazuma-speed-essence-recovery-evidence",
        "version": 1,
        "status": "FROZEN_BEFORE_RECOVERY",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "failure": state.get("error"),
        "controller_status_before_recovery": {
            "path": str(status_path),
            "sha256": _sha256(status_path),
        },
        "summarizer": {
            "path": str(summarizer_path),
            "sha256": _sha256(summarizer_path),
            "frozen_sha256": execution["summarizer"]["sha256"],
            "stderr": {"path": str(stderr_path), "sha256": _sha256(stderr_path)},
            "stdout": {"path": str(stdout_path), "sha256": _sha256(stdout_path)},
        },
        "final_shards": shard_receipts,
        "recovery_boundary": {
            "rerun_inference": False,
            "change_attempts_or_seeds": False,
            "change_aggregation_logic": False,
            "parser_label_injection_only": "final-blind",
        },
    }
    _write_new_json(evidence_path, evidence)

    recovery_stdout = _run_frozen_summarizer(
        summarizer_path=summarizer_path,
        preregistration=final_prereg,
        manifest=final_manifest,
        run_directory=final_root,
    )
    recovery_stdout_path = final_root / "summarizer.recovery.stdout.json"
    if recovery_stdout_path.exists():
        raise FileExistsError(f"refusing to overwrite recovery stdout: {recovery_stdout_path}")
    recovery_stdout_path.write_text(recovery_stdout, encoding="utf-8", newline="\n")

    aggregate_path = (final_root / "aggregate.json").resolve(strict=True)
    aggregate = _read_json(aggregate_path)
    if (
        aggregate.get("status") != "COMPLETE"
        or aggregate.get("purpose") != "final-blind"
        or not all(aggregate.get("validation", {}).values())
    ):
        raise ValueError("recovered aggregate failed validation")

    final_manifest_value = _read_json(final_manifest)
    roles = final_manifest_value.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("final manifest roles are absent")
    selection = _read_json(selection_path)
    final_audit = controller._strict_audit_evaluation(
        root=final_root,
        preregistration_path=final_prereg,
        manifest_path=final_manifest,
        aggregate_path=aggregate_path,
        expected_level_ids=[str(level["id"]) for level in contract["levels"]],
        expected_attempts_per_level=8,
        expected_seed_base=1_300_000_000,
        expected_model_ids=[str(model["id"]) for model in final_manifest_value["models"]],
    )

    state["phase"] = "RECORD_VIDEOS"
    state["updated_utc"] = controller._utc_now()
    state["error"] = None
    state["recovery"] = {
        "reason": "frozen summarizer argparse omitted the controller's final-blind label",
        "evidence": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
        "summarizer_reexecuted_without_byte_change": True,
        "inference_rerun": False,
        "recovery_stdout": {
            "path": str(recovery_stdout_path),
            "sha256": _sha256(recovery_stdout_path),
        },
    }
    controller._replace_json(status_path, state)

    new_bests = controller._new_level_bests(aggregate, roles)
    video_manifest_value = controller._render_new_bests(
        new_bests=new_bests,
        aggregate_path=aggregate_path,
        preregistration_path=final_prereg,
        manifest_path=final_manifest,
        original_root=original_root,
        output_root=output_root,
    )
    video_manifest = output_root / "record-videos-manifest.json"
    controller._write_new_json(video_manifest, video_manifest_value)

    state["phase"] = "FINAL_REPORT"
    state["updated_utc"] = controller._utc_now()
    controller._replace_json(status_path, state)
    report = controller._final_report(
        master=master,
        selection=selection,
        aggregate=aggregate,
        evaluation_root=final_root,
        roles=roles,
        new_bests=new_bests,
        video_manifest={
            "path": str(video_manifest),
            "sha256": _sha256(video_manifest),
            "artifacts": video_manifest_value["artifacts"],
        },
    )
    report["strict_final_matrix_audit"] = {
        key: final_audit[key]
        for key in (
            "attempts",
            "seed_range",
            "preregistration_sha256",
            "manifest_sha256",
            "aggregate_sha256",
            "shard_sha256",
        )
    }
    report_path = output_root / "final_report.json"
    chinese_path = output_root / "FINAL_BRIEF.zh-CN.md"
    english_path = output_root / "FINAL_BRIEF.en.md"
    controller._write_new_json(report_path, report)
    controller._write_new_text(chinese_path, controller._brief(report, chinese=True))
    controller._write_new_text(english_path, controller._brief(report, chinese=False))
    state.update(
        {
            "status": "COMPLETE",
            "phase": "COMPLETE",
            "updated_utc": controller._utc_now(),
            "error": None,
            "selection_decision": {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
            },
            "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "brief_zh_cn": {"path": str(chinese_path), "sha256": _sha256(chinese_path)},
            "brief_en": {"path": str(english_path), "sha256": _sha256(english_path)},
        }
    )
    controller._replace_json(status_path, state)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "aggregate": str(aggregate_path),
                "aggregate_sha256": _sha256(aggregate_path),
                "new_level_bests": len(new_bests),
                "final_report": str(report_path),
                "final_report_sha256": _sha256(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
