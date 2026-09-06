"""Independently audit the frozen AlphaZuma speed-essence campaign result."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping

try:
    from audit_alphazuma_v11_frontier_result import _audit_evaluation
except ModuleNotFoundError:
    from tools.audit_alphazuma_v11_frontier_result import _audit_evaluation


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_MASTER_SHA256 = (
    "sha256:5de3a8e4205d5a665caa840f1f3f8115da792ee22a3dd6c29d5691cb64cf204c"
)
EXPECTED_TRAINING_PREREGISTRATIONS = (
    (
        "alphazuma-speed-essence-v1-lowdrift-s71081401-preregistration-v1.json",
        "sha256:b40b0bfda1fbe492b5cd66354342e852a129069f828b8ef74a34564f16a47fca",
    ),
    (
        "alphazuma-speed-essence-v1-speed-s72081401-preregistration-v1.json",
        "sha256:d0014e7d253d31bd0d3e003410cacdfdbcaaaa811f88785a361d0a7bee82edf3",
    ),
    (
        "alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json",
        "sha256:bbb84c51ed600664b61d18222d6136e7914f793ad05adf977bfcdde3cf103ba9",
    ),
)
HARD_LEVEL_IDS = ("Jungle7", "village6", "village10")
SPEED_LEVEL_IDS = ("Jungle2", "Jungle4", "Jungle9", "village7")


class AuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(receipt: Mapping[str, Any], name: str) -> Path:
    raw = receipt.get(name)
    _require(isinstance(raw, dict), f"execution receipt lacks {name}")
    path = Path(str(raw.get("path", ""))).resolve(strict=True)
    _require(raw.get("sha256") == _sha256(path), f"execution artifact hash changed: {name}")
    return path


def _validate_execution_receipt(
    receipt_path: Path, master_path: Path, output_root: Path
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    _require(
        receipt.get("schema") == "zuma-rl.alphazuma-speed-essence-postprocess-execution-receipt",
        "unexpected execution receipt schema",
    )
    _require(receipt.get("version") == 1 and receipt.get("status") == "FROZEN_BEFORE_SELECTION", "execution receipt is not frozen")
    master = receipt.get("master_preregistration")
    _require(isinstance(master, dict), "execution receipt lacks master")
    _require(Path(str(master.get("path", ""))).resolve(strict=True) == master_path, "execution master path differs")
    _require(master.get("sha256") == EXPECTED_MASTER_SHA256 == _sha256(master_path), "execution master hash differs")
    _require(Path(str(receipt.get("output_root", ""))).resolve() == output_root, "execution output root differs")
    auditor = _artifact(receipt, "independent_auditor")
    _require(auditor == SCRIPT_PATH and _sha256(auditor) == receipt["independent_auditor"]["sha256"], "auditor identity differs")
    for name in (
        "controller",
        "evaluator",
        "summarizer",
        "matrix_auditor",
        "record_renderer",
        "postprocess_helper",
    ):
        _artifact(receipt, name)
    return receipt


def _training_contracts(master_path: Path, master: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references = master.get("training_preregistrations")
    _require(isinstance(references, list) and len(references) == 3, "training receipt count differs")
    specs = []
    runs = []
    for index, (filename, expected_hash) in enumerate(EXPECTED_TRAINING_PREREGISTRATIONS):
        reference = references[index]
        _require(isinstance(reference, dict) and reference.get("sha256") == expected_hash, "training receipt differs")
        path = (master_path.parent / filename).resolve(strict=True)
        _require(_sha256(path) == expected_hash, f"training preregistration changed: {filename}")
        spec = _read_json(path)
        specs.append(spec)
        raw_runs = spec.get("runs")
        _require(isinstance(raw_runs, list) and len(raw_runs) == 2, "training run count differs")
        for raw in raw_runs:
            run = dict(raw)
            run["preregistration_path"] = str(path)
            run["preregistration_sha256"] = expected_hash
            run["initial_training_steps"] = int(spec["initial_model"]["training_steps"])
            runs.append(run)
    _require(len(runs) == 6 and len({str(run["id"]) for run in runs}) == 6, "six-route contract differs")
    return specs, runs


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    _require(match is not None, f"unparseable checkpoint path: {path}")
    return int(match.group(1))


def _audit_candidates(
    master: Mapping[str, Any], runs: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(manifest.get("schema") == "zuma-rl.zero-shot-models-manifest" and manifest.get("status") == "FROZEN", "selection manifest is not frozen")
    models = manifest.get("models")
    receipts = manifest.get("route_receipts")
    _require(isinstance(models, list) and isinstance(receipts, list), "selection candidates or route receipts absent")
    _require(len(receipts) == 6, "route receipt count differs")
    expected_models = [dict(master["baselines"]["v1"]), dict(master["baselines"]["v11"])]
    for model in expected_models:
        model["path"] = str(Path(str(model["path"])).resolve(strict=True))
        model["source"] = "frozen_v1_baseline" if model["id"] == master["baselines"]["v1"]["id"] else "frozen_v11_baseline"
    for run, receipt in zip(runs, receipts, strict=True):
        run_id = str(run["id"])
        _require(receipt.get("route_id") == run_id, f"route receipt order differs: {run_id}")
        status = receipt.get("status")
        if status == "FAILED":
            failure = receipt.get("failure")
            _require(isinstance(failure, dict), f"failed route lacks failure receipt: {run_id}")
            failure_path = Path(str(failure.get("path", ""))).resolve(strict=True)
            _require(failure.get("sha256") == _sha256(failure_path), f"failure receipt changed: {run_id}")
            continue
        if status == "NO_ELIGIBLE_CHECKPOINT":
            continue
        _require(status == "COMPLETE", f"unknown route terminal status: {run_id}")
        config_path = Path(str(receipt["config"]["path"])).resolve(strict=True)
        completion_path = Path(str(receipt["completion"]["path"])).resolve(strict=True)
        _require(receipt["config"]["sha256"] == _sha256(config_path), f"config changed: {run_id}")
        _require(receipt["completion"]["sha256"] == _sha256(completion_path), f"completion changed: {run_id}")
        config = _read_json(config_path)
        completion = _read_json(completion_path)
        _require(config.get("trainer_sha256") == "sha256:2886a72a7348e492f4d34f4ef9733eee1668df24e17cc896ace0094267d371ba", f"trainer changed: {run_id}")
        _require(config.get("preregistration", {}).get("sha256") == run["preregistration_sha256"], f"route preregistration changed: {run_id}")
        source = int(config.get("training_source_model", {}).get("timesteps", 0))
        final_counter = int(completion["actual_steps"])
        new_steps = int(completion["actual_steps_this_run"])
        _require(final_counter - source == new_steps > 0, f"route step accounting differs: {run_id}")
        midpoint_target = source + new_steps / 2.0
        eligible = [
            path
            for path in (Path(str(run["run_dir"])) / "checkpoints").glob("*_steps.zip")
            if source < _checkpoint_steps(path) <= final_counter
        ]
        _require(bool(eligible), f"completed route lacks eligible checkpoint: {run_id}")
        midpoint = min(eligible, key=lambda path: (abs(_checkpoint_steps(path) - midpoint_target), _checkpoint_steps(path)))
        midpoint_counter = _checkpoint_steps(midpoint)
        final_path = Path(str(completion["final_model"]["path"])).resolve(strict=True)
        _require(completion["final_model"]["sha256"] == _sha256(final_path), f"final model changed: {run_id}")
        _require(receipt["midpoint_checkpoint"]["sha256"] == _sha256(midpoint), f"midpoint hash differs: {run_id}")
        _require(Path(str(receipt["midpoint_checkpoint"]["path"])).resolve(strict=True) == midpoint.resolve(strict=True), f"midpoint path differs: {run_id}")
        _require(int(receipt["midpoint_counter_timesteps"]) == midpoint_counter, f"midpoint counter differs: {run_id}")
        for stage, path, counter in (("mid", midpoint, midpoint_counter), ("final", final_path, final_counter)):
            expected_models.append(
                {
                    "id": f"{run_id}-{stage}-{counter}",
                    "training_steps": int(run["initial_training_steps"]) + counter - source,
                    "campaign_steps": counter - source,
                    "path": str(path.resolve(strict=True)),
                    "sha256": _sha256(path),
                    "source": f"{run_id}:{stage}",
                    "route_family": receipt["family"],
                    "device": run["device"],
                }
            )
    _require(models == expected_models, "selection candidate manifest differs from independently reconstructed rule")
    for model in models:
        _require(model["sha256"] == _sha256(Path(str(model["path"])).resolve(strict=True)), f"candidate model changed: {model['id']}")
    return models, receipts


def _level(summary: Mapping[str, Any], level_id: str) -> dict[str, Any]:
    matches = [row for row in summary["level_summaries"] if str(row["level_id"]).casefold() == level_id.casefold()]
    _require(len(matches) == 1, f"missing level summary: {level_id}")
    return matches[0]


def _summary(aggregate: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    matches = [row for row in aggregate["models"] if str(row["model"]["id"]) == model_id]
    _require(len(matches) == 1, f"missing model summary: {model_id}")
    return matches[0]


def _independent_rankings(
    master: Mapping[str, Any], aggregate: Mapping[str, Any], models: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    references = master["selection"]["frozen_v1_best_tick_references"]
    robust = []
    speed = []
    for index, model in enumerate(models):
        summary = _summary(aggregate, str(model["id"]))
        levels = list(summary["level_summaries"])
        anchor = _level(summary, "Jungle2")
        best_ticks = [int(row["best_attempt"]["ticks"]) for row in levels if isinstance(row.get("best_attempt"), dict)]
        median = statistics.median(best_ticks) if best_ticks else None
        robust_key = (
            int(int(anchor["wins"]) >= 3),
            sum(int(_level(summary, level_id)["wins"]) > 0 for level_id in HARD_LEVEL_IDS),
            sum(int(row["wins"]) > 0 for row in levels),
            sum(int(row["wins"]) for row in levels),
            -float(median) if median is not None else -1_000_000_000.0,
            -index,
        )
        speed_levels = [_level(summary, level_id) for level_id in SPEED_LEVEL_IDS]
        ratios = [
            int(row["best_attempt"]["ticks"]) / int(references[level_id])
            for level_id, row in zip(SPEED_LEVEL_IDS, speed_levels, strict=True)
            if isinstance(row.get("best_attempt"), dict)
        ]
        geometric = math.prod(ratios) ** (1 / 4) if len(ratios) == 4 else None
        speed_key = (
            int(int(anchor["wins"]) >= 3),
            sum(int(row["wins"]) > 0 for row in speed_levels),
            sum(int(row["wins"]) for row in levels),
            -float(geometric) if geometric is not None else -1_000_000_000.0,
            -index,
        )
        robust.append((robust_key, str(model["id"])))
        speed.append((speed_key, str(model["id"])))
    return (
        [model_id for _, model_id in sorted(robust, reverse=True)],
        [model_id for _, model_id in sorted(speed, reverse=True)],
    )


def _all_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for index in range(2):
        rows.extend(_read_json(root / f"matrix-shard-{index:02d}-of-02.json")["attempts"])
    return rows


def _paired(rows: Iterable[dict[str, Any]], first_id: str, second_id: str) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if str(row["model_id"]) in {first_id, second_id}:
            by_key[(str(row["level_id"]), int(row["attempt_index"]))][str(row["model_id"])] = row
    counts = {"both_win": 0, "first_only_win": 0, "second_only_win": 0, "both_fail": 0}
    for values in by_key.values():
        if first_id == second_id:
            first = second = values[first_id]
        else:
            _require(set(values) == {first_id, second_id}, "final paired matrix is incomplete")
            first, second = values[first_id], values[second_id]
        a = first.get("outcome") == "win"
        b = second.get("outcome") == "win"
        key = "both_win" if a and b else "first_only_win" if a else "second_only_win" if b else "both_fail"
        counts[key] += 1
    counts["pairs"] = len(by_key)
    counts["same_model"] = first_id == second_id
    return counts


def _new_bests(aggregate: Mapping[str, Any], roles: Mapping[str, str]) -> list[dict[str, Any]]:
    summaries = {str(row["model"]["id"]): row for row in aggregate["models"]}
    baseline_ids = {roles["v1_baseline"], roles["v11_baseline"]}
    selected_ids = {roles["robust_selected"], roles["speed_selected"]} - baseline_ids
    if not selected_ids:
        return []
    level_ids = [str(row["level_id"]) for row in next(iter(summaries.values()))["level_summaries"]]
    result = []
    for level_id in level_ids:
        baseline = [
            _level(summaries[model_id], level_id).get("best_attempt") for model_id in baseline_ids
        ]
        selected = [
            _level(summaries[model_id], level_id).get("best_attempt") for model_id in selected_ids
        ]
        baseline = [row for row in baseline if isinstance(row, dict)]
        selected = [row for row in selected if isinstance(row, dict)]
        if not selected:
            continue
        selected_best = min(selected, key=lambda row: (int(row["ticks"]), str(row["model_id"])))
        baseline_best = min(baseline, key=lambda row: (int(row["ticks"]), str(row["model_id"]))) if baseline else None
        if baseline_best is None or int(selected_best["ticks"]) < int(baseline_best["ticks"]):
            result.append(
                {
                    "level_id": level_id,
                    "selected_attempt": selected_best,
                    "best_baseline_attempt": baseline_best,
                    "improvement_ticks": None if baseline_best is None else int(baseline_best["ticks"]) - int(selected_best["ticks"]),
                }
            )
    return result


def audit_campaign(
    *, master_path: Path, output_root: Path, execution_receipt_path: Path
) -> dict[str, Any]:
    _require(_sha256(master_path) == EXPECTED_MASTER_SHA256, "master preregistration hash changed")
    master = _read_json(master_path)
    execution = _validate_execution_receipt(execution_receipt_path, master_path, output_root)
    _, runs = _training_contracts(master_path, master)

    status_path = (output_root / "controller_status.json").resolve(strict=True)
    status = _read_json(status_path)
    _require(status.get("status") == "COMPLETE" and status.get("phase") == "COMPLETE", "controller is incomplete")
    _require(status.get("master_preregistration", {}).get("sha256") == EXPECTED_MASTER_SHA256, "controller master receipt differs")
    _require(status.get("controller", {}).get("sha256") == execution["controller"]["sha256"], "controller execution identity differs")

    selection_manifest_path = (output_root / "selection-models-manifest.json").resolve(strict=True)
    selection_prereg_path = (output_root / "selection-preregistration.json").resolve(strict=True)
    selection_decision_path = (output_root / "selection-decision.json").resolve(strict=True)
    selection_manifest = _read_json(selection_manifest_path)
    models, route_receipts = _audit_candidates(master, runs, selection_manifest)
    selection_root = Path(str(master["execution"]["selection_output_root"])).resolve(strict=True)
    selection_eval = _audit_evaluation(
        root=selection_root,
        preregistration_path=selection_prereg_path,
        manifest_path=selection_manifest_path,
        aggregate_path=selection_root / "aggregate.json",
        expected_level_ids=[str(value) for value in master["selection"]["level_ids"]],
        expected_attempts_per_level=4,
        expected_seed_base=1_200_000_000,
        expected_model_ids=[str(model["id"]) for model in models],
    )
    decision = _read_json(selection_decision_path)
    _require(decision.get("status") == "COMPLETE", "selection decision is incomplete")
    robust_order, speed_order = _independent_rankings(master, selection_eval["aggregate"], models)
    _require([row["model"]["id"] for row in decision["robust_ranking"]] == robust_order, "robust ranking differs")
    _require([row["model"]["id"] for row in decision["speed_ranking"]] == speed_order, "speed ranking differs")
    _require(decision["robust_selected_model"]["id"] == robust_order[0], "robust selected model differs")
    _require(decision["speed_selected_model"]["id"] == speed_order[0], "speed selected model differs")

    final_manifest_path = (output_root / "final-blind-models-manifest.json").resolve(strict=True)
    final_prereg_path = (output_root / "final-blind-preregistration.json").resolve(strict=True)
    final_manifest = _read_json(final_manifest_path)
    roles = final_manifest.get("roles")
    _require(isinstance(roles, dict), "final role map is absent")
    source_roles = {
        "v1_baseline": str(master["baselines"]["v1"]["id"]),
        "v11_baseline": str(master["baselines"]["v11"]["id"]),
        "robust_selected": robust_order[0],
        "speed_selected": speed_order[0],
    }
    expected_final = []
    retained_by_hash: dict[str, str] = {}
    expected_roles: dict[str, str] = {}
    by_id = {str(model["id"]): model for model in models}
    for role in ("v1_baseline", "v11_baseline", "robust_selected", "speed_selected"):
        model = by_id[source_roles[role]]
        model_hash = str(model["sha256"])
        if model_hash not in retained_by_hash:
            retained_by_hash[model_hash] = str(model["id"])
            expected_final.append(model)
        expected_roles[role] = retained_by_hash[model_hash]
    _require(roles == expected_roles, "final role map differs")
    _require(final_manifest.get("models") == expected_final, "final manifest deduplication differs")

    final_root = Path(str(master["execution"]["final_output_root"])).resolve(strict=True)
    final_eval = _audit_evaluation(
        root=final_root,
        preregistration_path=final_prereg_path,
        manifest_path=final_manifest_path,
        aggregate_path=final_root / "aggregate.json",
        expected_level_ids=[str(level["id"]) for level in _read_json(final_prereg_path)["levels"]],
        expected_attempts_per_level=8,
        expected_seed_base=1_300_000_000,
        expected_model_ids=[str(model["id"]) for model in expected_final],
    )
    _require(len(final_eval["preregistration"]["levels"]) == 17, "final level inventory differs")

    report_path = (output_root / "final_report.json").resolve(strict=True)
    report = _read_json(report_path)
    _require(report.get("status") == "COMPLETE" and report.get("scientifically_valid") is True, "final report is incomplete or invalid")
    _require(report.get("roles") == expected_roles, "final report roles differ")
    summaries = {str(row["model"]["id"]): row for row in final_eval["aggregate"]["models"]}
    for role, model_id in expected_roles.items():
        metrics = report["final_blind"][role]
        summary = summaries[model_id]
        for field in ("levels", "levels_cleared", "attempts", "wins", "losses", "truncations", "win_rate", "median_win_ticks", "median_win_seconds", "best_attempt", "level_summaries"):
            _require(metrics.get(field) == summary.get(field), f"report metric differs: {role}.{field}")
    rows = _all_rows(final_root)
    for selected_role in ("robust_selected", "speed_selected"):
        for baseline_role in ("v1_baseline", "v11_baseline"):
            key = f"{selected_role}_vs_{baseline_role}"
            _require(report["paired_results"][key] == _paired(rows, expected_roles[selected_role], expected_roles[baseline_role]), f"paired result differs: {key}")

    new_bests = _new_bests(final_eval["aggregate"], expected_roles)
    _require(report.get("new_level_bests_against_both_paired_baselines") == new_bests, "new-best inventory differs")
    videos_path = (output_root / "record-videos-manifest.json").resolve(strict=True)
    videos = _read_json(videos_path)
    artifacts = videos.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == len(new_bests), "record video count differs")
    for best, artifact in zip(new_bests, artifacts, strict=True):
        _require(artifact.get("level_id") == best["level_id"], "record level differs")
        _require(artifact.get("model_id") == best["selected_attempt"]["model_id"], "record model differs")
        _require(artifact.get("trajectory_sha256") == best["selected_attempt"]["trajectory_sha256"], "record trajectory differs")
        for name in ("video", "poster", "sidecar"):
            path = Path(str(artifact[name]["path"])).resolve(strict=True)
            _require(artifact[name]["sha256"] == _sha256(path), f"record artifact hash differs: {name}")
        sidecar = _read_json(Path(str(artifact["sidecar"]["path"])))
        _require(sidecar.get("status") == "COMPLETE", "record sidecar is incomplete")
        _require(sidecar.get("candidate") == best["selected_attempt"], "record sidecar attempt differs")
        _require(sidecar.get("verification", {}).get("pre_render_matches_formal_attempt") is True, "pre-render identity failed")
        _require(sidecar.get("verification", {}).get("rendered_matches_formal_attempt") is True, "rendered identity failed")

    for field, filename in (
        ("brief_zh_cn", "FINAL_BRIEF.zh-CN.md"),
        ("brief_en", "FINAL_BRIEF.en.md"),
    ):
        path = (output_root / filename).resolve(strict=True)
        _require(status.get(field, {}).get("sha256") == _sha256(path), f"brief receipt differs: {field}")
    _require(status.get("final_report", {}).get("sha256") == _sha256(report_path), "controller final report receipt differs")

    return {
        "schema": "zuma-rl.alphazuma-speed-essence-independent-final-audit",
        "version": 1,
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "execution_receipt": {"path": str(execution_receipt_path), "sha256": _sha256(execution_receipt_path)},
        "checks": [
            "execution_artifact_hashes",
            "six_route_candidate_reconstruction",
            "selection_attempt_and_seed_matrix",
            "independent_robust_and_speed_rankings",
            "final_manifest_role_deduplication",
            "final_blind_attempt_and_seed_matrix",
            "report_metrics_and_paired_results",
            "record_video_poster_sidecar_and_trajectory_hashes",
        ],
        "selection": {
            "candidate_count": len(models),
            "robust_selected": robust_order[0],
            "speed_selected": speed_order[0],
            "attempt_rows": selection_eval["attempts"],
            "shard_sha256": selection_eval["shard_sha256"],
        },
        "final_blind": {
            "model_count": len(expected_final),
            "attempt_rows": final_eval["attempts"],
            "shard_sha256": final_eval["shard_sha256"],
            "new_level_bests": len(new_bests),
        },
        "artifacts": {
            "controller_status": {"path": str(status_path), "sha256": _sha256(status_path)},
            "selection_decision": {"path": str(selection_decision_path), "sha256": _sha256(selection_decision_path)},
            "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "record_videos_manifest": {"path": str(videos_path), "sha256": _sha256(videos_path)},
        },
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists():
        existing = _read_json(path)
        _require(existing.get("status") == "PASS", "existing audit receipt is not PASS")
        _require(
            existing.get("master_preregistration") == value.get("master_preregistration"),
            "existing audit master receipt differs",
        )
        _require(
            existing.get("execution_receipt") == value.get("execution_receipt"),
            "existing audit execution receipt differs",
        )
        _require(
            existing.get("artifacts") == value.get("artifacts"),
            "existing audited artifact hashes differ",
        )
        return
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_campaign(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(strict=True),
        execution_receipt_path=args.execution_receipt.expanduser().resolve(strict=True),
    )
    if args.receipt is not None:
        receipt = args.receipt.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        _write_new_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
