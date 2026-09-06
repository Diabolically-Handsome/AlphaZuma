"""Orchestrate engineering teacher selection, distillation, and PPO follow-up."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_receipt(value: str) -> tuple[str, Path]:
    try:
        model_id, path_text = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "source receipt must be MODEL_ID=PATH"
        ) from error
    path = Path(path_text).expanduser().resolve(strict=True)
    if not model_id:
        raise argparse.ArgumentTypeError("source receipt model id is empty")
    return model_id, path


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).resolve().parents[1]
        self.output_root = args.output_root.expanduser().resolve()
        if self.output_root.exists():
            raise FileExistsError(
                f"refusing to reuse teacher pipeline: {self.output_root}"
            )
        self.output_root.mkdir(parents=True)
        self.status_path = self.output_root / "pipeline_status.json"
        self.started_utc = _utc_now()
        self.started_monotonic = time.monotonic()
        self.receipts = {
            model_id: path for model_id, path in args.source_receipt
        }
        if len(self.receipts) != len(args.source_receipt):
            raise ValueError("source receipt model ids are duplicated")

    def publish(self, stage: str, **extra: Any) -> None:
        payload = {
            "schema": "zuma-rl.alphazuma-55-teacher-pipeline-status",
            "version": 1,
            "status": "RUNNING",
            "stage": stage,
            "started_utc": self.started_utc,
            "updated_utc": _utc_now(),
            "wall_seconds": time.monotonic() - self.started_monotonic,
            **extra,
        }
        _write_json_atomic(self.status_path, payload)

    def wait_for_engineering(self) -> list[Path]:
        shard_count = int(
            _read_json(self.args.engineering_preregistration)["execution"][
                "shard_count"
            ]
        )
        shard_paths = [
            self.args.engineering_root
            / f"matrix-shard-{index:02d}-of-{shard_count:02d}.json"
            for index in range(shard_count)
        ]
        deadline = time.monotonic() + 4 * 60 * 60
        last_completed = -1
        while True:
            snapshots: list[dict[str, Any]] = []
            completed = 0
            all_complete = True
            for path in shard_paths:
                if not path.is_file():
                    snapshots.append({"path": str(path), "status": "MISSING"})
                    all_complete = False
                    continue
                shard = _read_json(path)
                snapshots.append(
                    {
                        "path": str(path),
                        "status": shard.get("status"),
                        "completed_attempts": shard.get("completed_attempts"),
                        "expected_attempts": shard.get("expected_attempts"),
                        "wall_seconds": shard.get("runtime", {}).get(
                            "wall_seconds"
                        ),
                        "error": shard.get("error"),
                    }
                )
                completed += int(shard.get("completed_attempts", 0))
                if shard.get("status") == "ERROR" or shard.get("error") is not None:
                    raise RuntimeError(f"engineering shard failed: {path}")
                all_complete &= shard.get("status") == "COMPLETE"
            if completed != last_completed or all_complete:
                self.publish(
                    "WAITING_FOR_ENGINEERING",
                    completed_attempts=completed,
                    shards=snapshots,
                )
                last_completed = completed
            if all_complete:
                return shard_paths
            if time.monotonic() >= deadline:
                raise TimeoutError("engineering evaluation exceeded four hours")
            time.sleep(30)

    def choose_student(
        self,
        shard_paths: list[Path],
    ) -> dict[str, Any]:
        manifest = _read_json(self.args.models_manifest)
        attempts = [
            row
            for path in shard_paths
            for row in _read_json(path)["attempts"]
        ]
        rankings: list[dict[str, Any]] = []
        for model in manifest["models"]:
            rows = [row for row in attempts if row["model_id"] == model["id"]]
            win_rows = [row for row in rows if row["outcome"] == "win"]
            coverage = len({row["level_id"].casefold() for row in win_rows})
            rankings.append(
                {
                    "model": model,
                    "coverage": coverage,
                    "wins": len(win_rows),
                    "total_score": sum(int(row.get("score", 0)) for row in rows),
                    "median_winning_ticks": (
                        statistics.median(int(row["ticks"]) for row in win_rows)
                        if win_rows
                        else None
                    ),
                }
            )
        winner = min(
            rankings,
            key=lambda row: (
                -row["coverage"],
                -row["wins"],
                -row["total_score"],
                (
                    row["median_winning_ticks"]
                    if row["median_winning_ticks"] is not None
                    else float("inf")
                ),
                row["model"]["sha256"],
            ),
        )
        return {"winner": winner, "ranking": rankings}

    def run_checked(self, command: list[str], *, name: str) -> None:
        stdout_path = self.output_root / f"{name}.stdout.log"
        stderr_path = self.output_root / f"{name}.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} exited {result.returncode}; see {stderr_path}"
            )

    def run_monitored(
        self,
        command: list[str],
        *,
        name: str,
        stage: str,
        watch_path: Path,
    ) -> None:
        stdout_path = self.output_root / f"{name}.stdout.log"
        stderr_path = self.output_root / f"{name}.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                watch = _read_json(watch_path) if watch_path.is_file() else None
                self.publish(
                    stage,
                    child_pid=process.pid,
                    watched_receipt=(
                        {"path": str(watch_path), "value": watch}
                        if watch is not None
                        else {"path": str(watch_path), "value": None}
                    ),
                )
                time.sleep(30)
        if process.returncode != 0:
            raise RuntimeError(
                f"{name} exited {process.returncode}; see {stderr_path}"
            )

    def execute(self) -> dict[str, Any]:
        shard_paths = self.wait_for_engineering()
        teacher_map = self.args.teacher_map_output.expanduser().resolve()
        selector = self.project_root / "tools" / "select_alphazuma_55_teachers.py"
        command = [
            sys.executable,
            str(selector),
            "--preregistration",
            str(self.args.engineering_preregistration),
            "--models-manifest",
            str(self.args.models_manifest),
        ]
        for path in shard_paths:
            command.extend(("--shard", str(path)))
        command.extend(("--output", str(teacher_map)))
        self.publish("FREEZING_TEACHER_MAP")
        self.run_checked(command, name="select-teachers")

        student_selection = self.choose_student(shard_paths)
        source = student_selection["winner"]["model"]
        receipt = self.receipts.get(str(source["id"]))
        if receipt is None:
            raise ValueError(f"source receipt is missing for {source['id']}")
        distill_prereg = self.args.distillation_preregistration_output.resolve()
        distill_run_dir = self.args.distillation_run_dir.resolve()
        builder = self.project_root / "tools" / "build_alphazuma_55_distillation.py"
        self.publish(
            "FREEZING_DISTILLATION",
            student_selection=student_selection,
        )
        self.run_checked(
            [
                sys.executable,
                str(builder),
                "--master-preregistration",
                str(self.args.master_preregistration),
                "--teacher-map",
                str(teacher_map),
                "--source-model",
                str(source["path"]),
                "--source-id",
                str(source["id"]),
                "--source-training-steps",
                str(source["training_steps"]),
                "--source-migration-receipt",
                str(receipt),
                "--run-dir",
                str(distill_run_dir),
                "--device",
                "cuda:0",
                "--training-seed-base",
                "1520000000",
                "--model-seed",
                "85081501",
                "--output",
                str(distill_prereg),
            ],
            name="build-distillation",
        )
        distiller = self.project_root / "tools" / "distill_alphazuma_55.py"
        self.run_checked(
            [
                sys.executable,
                str(distiller),
                "--preregistration",
                str(distill_prereg),
                "--original-root",
                str(self.args.original_root),
                "--validate-only",
            ],
            name="validate-distillation",
        )
        self.run_monitored(
            [
                sys.executable,
                str(distiller),
                "--preregistration",
                str(distill_prereg),
                "--original-root",
                str(self.args.original_root),
            ],
            name="run-distillation",
            stage="DISTILLING",
            watch_path=distill_run_dir / "training_status.json",
        )

        completion_path = distill_run_dir / "completion.json"
        completion = _read_json(completion_path)
        if completion.get("status") != "COMPLETE":
            raise RuntimeError("distillation completion receipt is not COMPLETE")
        distilled_model = completion["final_model"]
        ppo_prereg = self.args.ppo_preregistration_output.resolve()
        ppo_run_dir = self.args.ppo_run_dir.resolve()
        ppo_builder = (
            self.project_root / "tools" / "build_alphazuma_55_training_route.py"
        )
        self.publish("FREEZING_DISTILLED_PPO", distillation=completion)
        self.run_checked(
            [
                sys.executable,
                str(ppo_builder),
                "--master-preregistration",
                str(self.args.master_preregistration),
                "--calibration",
                str(self.args.calibration),
                "--original-root",
                str(self.args.original_root),
                "--source-model",
                str(distilled_model["path"]),
                "--source-id",
                "alphazuma55-specialist-distilled",
                "--source-training-steps",
                str(distilled_model["num_timesteps"]),
                "--migration-receipt",
                str(completion_path),
                "--route-id",
                "alphazuma55-distilled-frontier-5090",
                "--run-dir",
                str(ppo_run_dir),
                "--device",
                "cuda:0",
                "--model-seed",
                "86081501",
                "--episode-seed-base",
                "1530000000",
                "--num-envs",
                "16",
                "--curriculum-mode",
                "frontier",
                "--learning-rate",
                "0.000002",
                "--entropy-coef",
                "0.0002",
                "--ppo-epochs",
                "5",
                "--output",
                str(ppo_prereg),
            ],
            name="build-distilled-ppo",
        )
        trainer = self.project_root / "tools" / "train_alphazuma_55.py"
        self.run_checked(
            [
                sys.executable,
                str(trainer),
                "--preregistration",
                str(ppo_prereg),
                "--original-root",
                str(self.args.original_root),
                "--run-id",
                "alphazuma55-distilled-frontier-5090",
                "--validate-only",
            ],
            name="validate-distilled-ppo",
        )
        self.run_monitored(
            [
                sys.executable,
                str(trainer),
                "--preregistration",
                str(ppo_prereg),
                "--original-root",
                str(self.args.original_root),
                "--run-id",
                "alphazuma55-distilled-frontier-5090",
            ],
            name="run-distilled-ppo",
            stage="TRAINING_DISTILLED_PPO",
            watch_path=ppo_run_dir / "training_status.json",
        )
        return {
            "schema": "zuma-rl.alphazuma-55-teacher-pipeline-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.monotonic() - self.started_monotonic,
            "teacher_map": {
                "path": str(teacher_map),
                "sha256": _sha256(teacher_map),
            },
            "student_selection": student_selection,
            "distillation_completion": {
                "path": str(completion_path),
                "sha256": _sha256(completion_path),
            },
            "distilled_ppo_completion": {
                "path": str(ppo_run_dir / "completion.json"),
                "sha256": _sha256(ppo_run_dir / "completion.json"),
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--engineering-preregistration", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--engineering-root", required=True, type=Path)
    parser.add_argument("--source-receipt", action="append", type=_source_receipt, required=True)
    parser.add_argument("--teacher-map-output", required=True, type=Path)
    parser.add_argument("--distillation-preregistration-output", required=True, type=Path)
    parser.add_argument("--distillation-run-dir", required=True, type=Path)
    parser.add_argument("--ppo-preregistration-output", required=True, type=Path)
    parser.add_argument("--ppo-run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "master_preregistration",
        "calibration",
        "original_root",
        "engineering_preregistration",
        "models_manifest",
        "engineering_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    pipeline = Pipeline(args)
    try:
        completion = pipeline.execute()
        _write_json_atomic(pipeline.output_root / "completion.json", completion)
        _write_json_atomic(
            pipeline.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-teacher-pipeline-status",
                "version": 1,
                "status": "COMPLETE",
                "stage": "COMPLETE",
                "updated_utc": _utc_now(),
                "completion": completion,
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-teacher-pipeline-failure",
            "version": 1,
            "status": "FAILED",
            "failed_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json_atomic(pipeline.output_root / "failure.json", failure)
        _write_json_atomic(
            pipeline.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-teacher-pipeline-status",
                "version": 1,
                "status": "FAILED",
                "stage": "FAILED",
                "updated_utc": _utc_now(),
                "failure": failure,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
