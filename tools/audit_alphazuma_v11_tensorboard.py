"""Audit live AlphaZuma V1.1 PPO TensorBoard streams without changing training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXPECTED_TAGS = (
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "time/fps",
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
)


class AuditError(RuntimeError):
    """Raised when a live optimization stream fails a structural invariant."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected a JSON object: {path}")
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AuditError(f"timestamp has no timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _deduplicate_samples(samples: Iterable[Any]) -> list[Any]:
    """Keep the latest wall-time record for every TensorBoard step."""
    by_step: dict[int, Any] = {}
    for sample in samples:
        step = int(sample.step)
        current = by_step.get(step)
        if current is None or float(sample.wall_time) >= float(current.wall_time):
            by_step[step] = sample
    return [by_step[step] for step in sorted(by_step)]


def _summarize_samples(samples: Iterable[Any], recent_count: int) -> dict[str, Any]:
    ordered = _deduplicate_samples(samples)
    if not ordered:
        raise AuditError("scalar stream is empty")
    values = [float(sample.value) for sample in ordered]
    finite = [math.isfinite(value) for value in values]
    recent = values[-recent_count:]
    recent_finite = [value for value in recent if math.isfinite(value)]
    return {
        "count": len(ordered),
        "first_step": int(ordered[0].step),
        "last_step": int(ordered[-1].step),
        "last_wall_time_utc": datetime.fromtimestamp(
            float(ordered[-1].wall_time), tz=timezone.utc
        ).isoformat(),
        "latest": values[-1] if math.isfinite(values[-1]) else None,
        "all_finite": all(finite),
        "nonfinite_count": finite.count(False),
        "recent_count": len(recent),
        "recent_finite_count": len(recent_finite),
        "recent_min": min(recent_finite) if recent_finite else None,
        "recent_max": max(recent_finite) if recent_finite else None,
        "recent_mean": statistics.fmean(recent_finite) if recent_finite else None,
        "recent_median": statistics.median(recent_finite) if recent_finite else None,
    }


def _diagnostic_warnings(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Return advisory warnings; these are intentionally not certification gates."""
    warnings = []
    approx_kl = summaries.get("train/approx_kl")
    if approx_kl and approx_kl["recent_max"] is not None and approx_kl["recent_max"] > 0.2:
        warnings.append("recent_approx_kl_above_0.2")
    clip = summaries.get("train/clip_fraction")
    if clip and clip["recent_mean"] is not None and clip["recent_mean"] > 0.8:
        warnings.append("recent_clip_fraction_mean_above_0.8")
    explained = summaries.get("train/explained_variance")
    if explained and explained["recent_mean"] is not None and explained["recent_mean"] < -1.0:
        warnings.append("recent_explained_variance_mean_below_minus_1")
    entropy = summaries.get("train/entropy_loss")
    if entropy and entropy["recent_mean"] is not None and abs(entropy["recent_mean"]) < 0.001:
        warnings.append("recent_entropy_magnitude_below_0.001")
    return warnings


def audit_run(
    run_dir: Path,
    *,
    recent_count: int,
    maximum_event_age_seconds: float,
    maximum_status_age_seconds: float,
    maximum_step_lag: int,
    now: datetime,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    status_path = run_dir / "training_status.json"
    status = _read_json(status_path)
    event_paths = sorted((run_dir / "tensorboard").glob("**/events.out.tfevents.*"))
    if not event_paths:
        raise AuditError(f"no TensorBoard event files: {run_dir}")
    by_tag: dict[str, list[Any]] = {tag: [] for tag in EXPECTED_TAGS}
    observed_tags: set[str] = set()
    for event_path in event_paths:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        observed_tags.update(tags)
        for tag in EXPECTED_TAGS:
            if tag in tags:
                by_tag[tag].extend(accumulator.Scalars(tag))
    missing = [tag for tag in EXPECTED_TAGS if not by_tag[tag]]
    summaries = {
        tag: _summarize_samples(samples, recent_count)
        for tag, samples in by_tag.items()
        if samples
    }
    nonfinite_tags = [tag for tag, summary in summaries.items() if not summary["all_finite"]]
    latest_wall_time = max(
        _parse_utc(summary["last_wall_time_utc"]) for summary in summaries.values()
    )
    event_age = (now - latest_wall_time).total_seconds()
    status_age = (now - _parse_utc(str(status["updated_utc"]))).total_seconds()
    train_steps = [
        int(summary["last_step"])
        for tag, summary in summaries.items()
        if tag.startswith("train/")
    ]
    if not train_steps:
        raise AuditError(f"no train/* scalar steps: {run_dir}")
    latest_train_step = max(train_steps)
    status_timesteps = int(status["timesteps"])
    step_lag = status_timesteps - latest_train_step
    structural_errors = []
    if status.get("status") != "RUNNING" or status.get("error") is not None:
        structural_errors.append("training_status_not_running_or_errored")
    if missing:
        structural_errors.append("missing_expected_scalar_tags")
    if nonfinite_tags:
        structural_errors.append("nonfinite_scalar_values")
    if event_age < -5 or event_age > maximum_event_age_seconds:
        structural_errors.append("event_stream_stale_or_future_dated")
    if status_age < -5 or status_age > maximum_status_age_seconds:
        structural_errors.append("training_status_stale_or_future_dated")
    if step_lag < -maximum_step_lag or step_lag > maximum_step_lag:
        structural_errors.append("tensorboard_step_lag_out_of_bounds")
    learning_rate = summaries.get("train/learning_rate")
    if learning_rate and learning_rate["recent_min"] is not None and learning_rate["recent_min"] <= 0:
        structural_errors.append("nonpositive_learning_rate")
    clip = summaries.get("train/clip_fraction")
    if (
        clip
        and clip["recent_min"] is not None
        and clip["recent_max"] is not None
        and (clip["recent_min"] < 0 or clip["recent_max"] > 1)
    ):
        structural_errors.append("clip_fraction_outside_zero_one")
    fps = summaries.get("time/fps")
    if fps and fps["recent_min"] is not None and fps["recent_min"] <= 0:
        structural_errors.append("nonpositive_fps")
    return {
        "run_dir": str(run_dir),
        "run_id": status.get("run_id"),
        "status": "PASS" if not structural_errors else "FAIL",
        "training_status": {
            "path": str(status_path),
            "sha256": _sha256(status_path),
            "updated_utc": status["updated_utc"],
            "age_seconds": status_age,
            "timesteps": status_timesteps,
            "steps_this_run": int(status["steps_this_run"]),
        },
        "event_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in event_paths
        ],
        "observed_scalar_tags": sorted(observed_tags),
        "missing_expected_tags": missing,
        "nonfinite_tags": nonfinite_tags,
        "latest_event_age_seconds": event_age,
        "latest_train_step": latest_train_step,
        "tensorboard_step_lag": step_lag,
        "structural_errors": structural_errors,
        "diagnostic_warnings": _diagnostic_warnings(summaries),
        "scalars": summaries,
    }


def audit_portfolio(
    run_dirs: Iterable[Path],
    *,
    recent_count: int = 20,
    maximum_event_age_seconds: float = 900.0,
    maximum_status_age_seconds: float = 180.0,
    maximum_step_lag: int = 1_048_576,
) -> dict[str, Any]:
    if recent_count <= 0:
        raise ValueError("recent_count must be positive")
    now = datetime.now(timezone.utc)
    runs = [
        audit_run(
            run_dir,
            recent_count=recent_count,
            maximum_event_age_seconds=maximum_event_age_seconds,
            maximum_status_age_seconds=maximum_status_age_seconds,
            maximum_step_lag=maximum_step_lag,
            now=now,
        )
        for run_dir in run_dirs
    ]
    if not runs:
        raise ValueError("at least one run directory is required")
    return {
        "schema": "zuma-rl.alphazuma-v11-live-tensorboard-audit",
        "version": 1,
        "status": "PASS" if all(run["status"] == "PASS" for run in runs) else "FAIL",
        "created_utc": now.isoformat(),
        "classification": "live_training_diagnostic_only",
        "formal_gate_effect": "none",
        "recent_scalar_count": recent_count,
        "limits": {
            "maximum_event_age_seconds": maximum_event_age_seconds,
            "maximum_status_age_seconds": maximum_status_age_seconds,
            "maximum_step_lag": maximum_step_lag,
        },
        "runs": runs,
        "summary": {
            "runs": len(runs),
            "passed": sum(run["status"] == "PASS" for run in runs),
            "failed": sum(run["status"] != "PASS" for run in runs),
            "warnings": sum(len(run["diagnostic_warnings"]) for run in runs),
            "nonfinite_tags": sum(len(run["nonfinite_tags"]) for run in runs),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--recent-count", type=int, default=20)
    parser.add_argument("--maximum-event-age-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-status-age-seconds", type=float, default=180.0)
    parser.add_argument("--maximum-step-lag", type=int, default=1_048_576)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_portfolio(
        args.run_dir,
        recent_count=args.recent_count,
        maximum_event_age_seconds=args.maximum_event_age_seconds,
        maximum_status_age_seconds=args.maximum_status_age_seconds,
        maximum_step_lag=args.maximum_step_lag,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
