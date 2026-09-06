"""Wait for V1.1 overnight runs, select a candidate, and run morning blind gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "tools" / "evaluate_zero_shot_multilevel.py"
SUMMARIZER = PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must contain an offset")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _replace_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_logged(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite log: {path}")
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}; "
            f"stderr={stderr_path}"
        )


def _run_shards(
    *,
    preregistration: Path,
    manifest: Path,
    original_root: Path,
    output_root: Path,
    shard_count: int,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    processes: list[tuple[subprocess.Popen[str], Any, Any, Path]] = []
    for index in range(shard_count):
        stdout_path = output_root / f"shard-{index:02d}.stdout.log"
        stderr_path = output_root / f"shard-{index:02d}.stderr.log"
        stdout = stdout_path.open("x", encoding="utf-8", newline="\n")
        stderr = stderr_path.open("x", encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            str(EVALUATOR),
            "--preregistration",
            str(preregistration),
            "--models-manifest",
            str(manifest),
            "--original-root",
            str(original_root),
            "--shard-index",
            str(index),
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        processes.append((process, stdout, stderr, stderr_path))
    failures = []
    for process, stdout, stderr, stderr_path in processes:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        if return_code:
            failures.append((process.pid, return_code, stderr_path))
    if failures:
        raise RuntimeError(f"one or more evaluation shards failed: {failures}")


def _evaluation_preregistration(
    *,
    baseline_model: dict[str, Any],
    levels: list[dict[str, Any]],
    attempts_per_level: int,
    base_seed: int,
    output_root: Path,
    shard_count: int,
    devices: list[str],
    environment: dict[str, Any],
    purpose: str,
) -> dict[str, Any]:
    return {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "frozen_utc": _utc_now(),
        "purpose": purpose,
        "model": baseline_model,
        "levels": levels,
        "attempts_per_level": attempts_per_level,
        "total_attempts": len(levels) * attempts_per_level,
        "seed_plan": {
            "base_seed": base_seed,
            "last_seed": base_seed + len(levels) * attempts_per_level - 1,
            "all_seeds_frozen_before_policy_inference": True,
        },
        "environment": environment,
        "execution": {
            "output_root": str(output_root),
            "shard_count": shard_count,
            "devices": devices,
            "parallel_envs_per_shard": 24,
        },
        "reporting": {"all_attempts_must_be_reported": True},
        "evaluator": {
            "path": str(EVALUATOR),
            "sha256": _sha256(EVALUATOR),
        },
    }


def _evaluate_and_summarize(
    *,
    phase: str,
    preregistration: Path,
    manifest: Path,
    original_root: Path,
    output_root: Path,
    shard_count: int,
    purpose: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=False)
    _run_logged(
        [
            sys.executable,
            str(EVALUATOR),
            "--preregistration",
            str(preregistration),
            "--models-manifest",
            str(manifest),
            "--original-root",
            str(original_root),
            "--validate-only",
        ],
        stdout_path=output_root / "validation.stdout.json",
        stderr_path=output_root / "validation.stderr.log",
    )
    _run_shards(
        preregistration=preregistration,
        manifest=manifest,
        original_root=original_root,
        output_root=output_root,
        shard_count=shard_count,
    )
    _run_logged(
        [
            sys.executable,
            str(SUMMARIZER),
            "--preregistration",
            str(preregistration),
            "--models-manifest",
            str(manifest),
            "--run-directory",
            str(output_root),
            "--purpose",
            purpose,
        ],
        stdout_path=output_root / "summarizer.stdout.json",
        stderr_path=output_root / "summarizer.stderr.log",
    )
    aggregate = _read_json(output_root / "aggregate.json")
    if aggregate.get("status") != "COMPLETE" or not all(aggregate["validation"].values()):
        raise RuntimeError(f"{phase} aggregate failed validation")
    return aggregate


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint steps: {path}")
    return int(match.group(1))


def _candidate_models(master: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = dict(master["initial_model"])
    candidates = [
        {
            "id": "baseline-v1",
            "training_steps": int(source["training_steps"]),
            "overnight_steps": 0,
            "path": str(Path(str(source["path"])).resolve(strict=True)),
            "sha256": source["sha256"],
            "source": "unchanged_initial_model",
        }
    ]
    completions = []
    for run in master["runs"]:
        run_dir = Path(str(run["run_dir"])).resolve(strict=True)
        completion_path = run_dir / "completion.json"
        completion = _read_json(completion_path)
        if completion.get("status") != "COMPLETE":
            raise RuntimeError(f"run did not complete: {run['id']}")
        actual_steps = int(completion["actual_steps"])
        final = Path(str(completion["final_model"]["path"])).resolve(strict=True)
        if completion["final_model"]["sha256"] != _sha256(final):
            raise RuntimeError(f"final model hash changed: {run['id']}")
        checkpoints = sorted((run_dir / "checkpoints").glob("*_steps.zip"))
        if not checkpoints:
            raise RuntimeError(f"run has no checkpoints: {run['id']}")
        midpoint = min(checkpoints, key=lambda path: abs(_checkpoint_steps(path) - actual_steps / 2.0))
        midpoint_steps = _checkpoint_steps(midpoint)
        prefix = "adaptive" if str(run["id"]).startswith("adaptive") else "uniform"
        for stage, path, overnight_steps in (
            ("mid", midpoint, midpoint_steps),
            ("final", final, actual_steps),
        ):
            candidates.append(
                {
                    "id": f"{prefix}-{stage}-{overnight_steps}",
                    "training_steps": int(source["training_steps"]) + overnight_steps,
                    "overnight_steps": overnight_steps,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "source": f"{run['id']}:{stage}",
                }
            )
        completions.append(
            {
                "run_id": run["id"],
                "completion": str(completion_path),
                "completion_sha256": _sha256(completion_path),
                "actual_steps": actual_steps,
                "midpoint_checkpoint": str(midpoint),
                "midpoint_steps": midpoint_steps,
            }
        )
    if len(candidates) != 5:
        raise RuntimeError(f"frozen selection rule expected five candidates, got {len(candidates)}")
    return candidates, completions


def _model_summary(aggregate: dict[str, Any], model_id: str) -> dict[str, Any]:
    matches = [row for row in aggregate["models"] if row["model"]["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"aggregate has no unique model summary: {model_id}")
    return matches[0]


def _level(summary: dict[str, Any], level_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in summary["level_summaries"]
        if str(row["level_id"]).casefold() == level_id.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"summary has no unique level: {level_id}")
    return matches[0]


def _paired_counts(
    *,
    output_root: Path,
    shard_count: int,
    baseline_id: str,
    selected_id: str,
) -> dict[str, Any]:
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for index in range(shard_count):
        shard = _read_json(output_root / f"matrix-shard-{index:02d}-of-{shard_count:02d}.json")
        for row in shard["attempts"]:
            paired[(str(row["level_id"]), int(row["attempt_index"]))][str(row["model_id"])] = (
                row["outcome"] == "win"
            )
    if not paired or any(set(value) != {baseline_id, selected_id} for value in paired.values()):
        raise ValueError("paired attempt matrix is incomplete")
    selected_only = sum(value[selected_id] and not value[baseline_id] for value in paired.values())
    baseline_only = sum(value[baseline_id] and not value[selected_id] for value in paired.values())
    both_win = sum(value[baseline_id] and value[selected_id] for value in paired.values())
    both_fail = sum(not value[baseline_id] and not value[selected_id] for value in paired.values())
    discordant = selected_only + baseline_only
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(0, min(selected_only, baseline_only) + 1))
        exact_two_sided = min(1.0, 2.0 * tail / (2**discordant))
    else:
        exact_two_sided = 1.0
    return {
        "both_win": both_win,
        "selected_only_win": selected_only,
        "baseline_only_win": baseline_only,
        "both_fail": both_fail,
        "mcnemar_exact_two_sided_p": exact_two_sided,
    }


def _final_report(
    *,
    master: dict[str, Any],
    selection: dict[str, Any],
    target: dict[str, Any],
    anchor: dict[str, Any],
    target_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    selected_id = "selected-overnight"
    baseline_id = "baseline-v1"
    selected_target = _model_summary(target, selected_id)
    baseline_target = _model_summary(target, baseline_id)
    selected_anchor = _model_summary(anchor, selected_id)
    baseline_anchor = _model_summary(anchor, baseline_id)
    gates = master["morning_blind"]["success_gates"]
    village = selected_target["regions"].get("village", {"wins": 0, "win_rate": 0.0})
    jungle9 = _level(selected_target, "Jungle9")
    village6 = _level(selected_target, "village6")
    anchor_level = _level(selected_anchor, "Jungle2")
    checks = {
        "target_levels_cleared": {
            "actual": selected_target["levels_cleared"],
            "required": gates["target_levels_cleared_at_least"],
        },
        "target_wins": {
            "actual": selected_target["wins"],
            "required": gates["target_wins_at_least"],
        },
        "target_win_rate": {
            "actual": selected_target["win_rate"],
            "required": gates["target_win_rate_at_least"],
        },
        "village_wins": {
            "actual": village["wins"],
            "required": gates["village_wins_at_least"],
        },
        "village_win_rate": {
            "actual": village["win_rate"],
            "required": gates["village_win_rate_at_least"],
        },
        "jungle2_wins": {
            "actual": anchor_level["wins"],
            "required": gates["jungle2_wins_at_least"],
        },
        "Jungle9_wins": {
            "actual": jungle9["wins"],
            "required": gates["Jungle9_wins_at_least"],
        },
        "village6_wins": {
            "actual": village6["wins"],
            "required": gates["village6_wins_at_least"],
        },
    }
    for check in checks.values():
        check["passed"] = check["actual"] >= check["required"]
    paired = _paired_counts(
        output_root=target_root,
        shard_count=2,
        baseline_id=baseline_id,
        selected_id=selected_id,
    )
    return {
        "schema": "zuma-rl.overnight-v11-final-report",
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "deadline_utc": master["schedule"]["goal_end_utc"],
        "within_deadline": datetime.now(timezone.utc) <= _parse_utc(master["schedule"]["goal_end_utc"]),
        "selected_source_model": selection["selection"]["selected_model"],
        "all_success_gates_passed": all(check["passed"] for check in checks.values()),
        "success_gates": checks,
        "target_blind": {
            "selected": selected_target,
            "baseline": baseline_target,
            "delta_wins": selected_target["wins"] - baseline_target["wins"],
            "delta_levels_cleared": selected_target["levels_cleared"] - baseline_target["levels_cleared"],
            "paired_comparison": paired,
        },
        "jungle2_blind": {
            "selected": selected_anchor,
            "baseline": baseline_anchor,
            "delta_wins": selected_anchor["wins"] - baseline_anchor["wins"],
        },
    }


def _report_markdown(report: dict[str, Any]) -> str:
    selected = report["target_blind"]["selected"]
    baseline = report["target_blind"]["baseline"]
    selected_anchor = report["jungle2_blind"]["selected"]
    baseline_anchor = report["jungle2_blind"]["baseline"]
    lines = [
        "# AlphaZuma V1.1 overnight result",
        "",
        f"- Completed: {report['completed_utc']}",
        f"- Within 11:30 deadline: {report['within_deadline']}",
        f"- All preregistered success gates passed: {report['all_success_gates_passed']}",
        f"- Selected source: `{report['selected_source_model']['id']}`",
        "",
        "## Morning blind comparison",
        "",
        "| Model | Target levels cleared | Target wins | Target win rate | Jungle2 wins |",
        "|---|---:|---:|---:|---:|",
        f"| Selected | {selected['levels_cleared']}/{selected['levels']} | {selected['wins']}/{selected['attempts']} | {100*selected['win_rate']:.1f}% | {selected_anchor['wins']}/{selected_anchor['attempts']} |",
        f"| Baseline | {baseline['levels_cleared']}/{baseline['levels']} | {baseline['wins']}/{baseline['attempts']} | {100*baseline['win_rate']:.1f}% | {baseline_anchor['wins']}/{baseline_anchor['attempts']} |",
        "",
        "## Frozen gates",
        "",
        "| Gate | Actual | Required | Result |",
        "|---|---:|---:|---|",
    ]
    for name, check in report["success_gates"].items():
        lines.append(
            f"| {name} | {check['actual']} | {check['required']} | {'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse postprocess output root: {output_root}")
    output_root.mkdir(parents=True)
    master = _read_json(master_path)
    status_path = output_root / "controller_status.json"
    state = {
        "schema": "zuma-rl.overnight-v11-controller-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_TRAINING",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "evaluator": {"path": str(EVALUATOR), "sha256": _sha256(EVALUATOR)},
        "summarizer": {"path": str(SUMMARIZER), "sha256": _sha256(SUMMARIZER)},
        "training": {},
        "error": None,
    }
    _replace_json(status_path, state)
    try:
        latest_allowed = _parse_utc(master["schedule"]["training_stop_utc"]) + timedelta(minutes=20)
        while True:
            complete = True
            snapshots = {}
            for run in master["runs"]:
                run_dir = Path(str(run["run_dir"])).resolve()
                failure_path = run_dir / "failure.json"
                if failure_path.exists():
                    raise RuntimeError(f"training run failed: {failure_path}")
                completion_path = run_dir / "completion.json"
                training_status_path = run_dir / "training_status.json"
                snapshots[run["id"]] = (
                    _read_json(completion_path)
                    if completion_path.exists()
                    else _read_json(training_status_path)
                    if training_status_path.exists()
                    else {"status": "STARTING"}
                )
                complete = complete and completion_path.exists()
            state["training"] = snapshots
            state["updated_utc"] = _utc_now()
            _replace_json(status_path, state)
            if complete:
                break
            if datetime.now(timezone.utc) > latest_allowed:
                raise TimeoutError("training did not finalize within 20 minutes of its deadline")
            time.sleep(args.poll_seconds)

        state["phase"] = "BUILDING_SELECTION_CANDIDATES"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        candidates, completions = _candidate_models(master)
        selection_manifest = output_root / "selection-models-manifest.json"
        _write_new_json(
            selection_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "overnight candidate selection on isolated frozen seeds",
                "models": candidates,
                "training_completions": completions,
            },
        )
        baseline = candidates[0]
        selection_root = output_root / "selection-evaluation"
        selection_prereg = output_root / "selection-preregistration.json"
        selection_seed_base = int(master["seed_isolation"]["selection_seed_base"])
        selection_attempts = int(master["seed_isolation"]["selection_seeds_per_level"])
        _write_new_json(
            selection_prereg,
            _evaluation_preregistration(
                baseline_model=baseline,
                levels=master["levels"],
                attempts_per_level=selection_attempts,
                base_seed=selection_seed_base,
                output_root=selection_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=master["environment"],
                purpose="candidate selection; not morning blind evidence",
            ),
        )
        state["phase"] = "SELECTION_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        selection = _evaluate_and_summarize(
            phase="selection",
            preregistration=selection_prereg,
            manifest=selection_manifest,
            original_root=original_root,
            output_root=selection_root,
            shard_count=2,
            purpose="selection",
        )
        selected = selection["selection"]["selected_model"]

        morning_manifest = output_root / "morning-blind-models-manifest.json"
        _write_new_json(
            morning_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "paired morning blind comparison",
                "models": [
                    dict(baseline),
                    {
                        **selected,
                        "id": "selected-overnight",
                        "selected_source_id": selected["id"],
                    },
                ],
                "selection_aggregate": {
                    "path": str(selection_root / "aggregate.json"),
                    "sha256": _sha256(selection_root / "aggregate.json"),
                },
            },
        )

        target_levels = [
            level for level in master["levels"] if str(level["id"]).casefold() != "jungle2"
        ]
        target_root = output_root / "morning-target-evaluation"
        target_prereg = output_root / "morning-target-preregistration.json"
        _write_new_json(
            target_prereg,
            _evaluation_preregistration(
                baseline_model=baseline,
                levels=target_levels,
                attempts_per_level=int(master["seed_isolation"]["morning_blind_target_seeds_per_level"]),
                base_seed=int(master["seed_isolation"]["morning_blind_target_seed_base"]),
                output_root=target_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=master["environment"],
                purpose="paired morning blind target-level evaluation",
            ),
        )
        state["phase"] = "MORNING_TARGET_BLIND"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        target = _evaluate_and_summarize(
            phase="morning-target",
            preregistration=target_prereg,
            manifest=morning_manifest,
            original_root=original_root,
            output_root=target_root,
            shard_count=2,
            purpose="morning-target",
        )

        anchor_root = output_root / "morning-anchor-evaluation"
        anchor_prereg = output_root / "morning-anchor-preregistration.json"
        anchor_level = [
            level for level in master["levels"] if str(level["id"]).casefold() == "jungle2"
        ]
        _write_new_json(
            anchor_prereg,
            _evaluation_preregistration(
                baseline_model=baseline,
                levels=anchor_level,
                attempts_per_level=int(master["seed_isolation"]["morning_blind_jungle2_seeds"]),
                base_seed=int(master["seed_isolation"]["morning_blind_jungle2_seed_base"]),
                output_root=anchor_root,
                shard_count=1,
                devices=["cuda:0"],
                environment=master["environment"],
                purpose="paired morning blind Jungle2 retention evaluation",
            ),
        )
        state["phase"] = "MORNING_JUNGLE2_BLIND"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        anchor = _evaluate_and_summarize(
            phase="morning-anchor",
            preregistration=anchor_prereg,
            manifest=morning_manifest,
            original_root=original_root,
            output_root=anchor_root,
            shard_count=1,
            purpose="morning-anchor",
        )

        state["phase"] = "FINAL_REPORT"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        report = _final_report(
            master=master,
            selection=selection,
            target=target,
            anchor=anchor,
            target_root=target_root,
            output_root=output_root,
        )
        report_path = output_root / "final_report.json"
        summary_path = output_root / "FINAL_REPORT.md"
        _write_new_json(report_path, report)
        _write_new_text(summary_path, _report_markdown(report))
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
                "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            }
        )
        _replace_json(status_path, state)
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _replace_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
