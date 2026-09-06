from __future__ import annotations

from dataclasses import dataclass
import math

from tools.audit_alphazuma_v11_tensorboard import (
    _diagnostic_warnings,
    _summarize_samples,
)


@dataclass(frozen=True)
class _Sample:
    wall_time: float
    step: int
    value: float


def _summary(value: float, *, minimum: float | None = None, maximum: float | None = None):
    minimum = value if minimum is None else minimum
    maximum = value if maximum is None else maximum
    return {"recent_mean": value, "recent_min": minimum, "recent_max": maximum}


def test_scalar_summary_deduplicates_by_step_and_keeps_latest_wall_time() -> None:
    samples = [
        _Sample(wall_time=10.0, step=2, value=2.0),
        _Sample(wall_time=5.0, step=1, value=1.0),
        _Sample(wall_time=20.0, step=2, value=4.0),
        _Sample(wall_time=30.0, step=3, value=8.0),
    ]

    summary = _summarize_samples(samples, recent_count=2)

    assert summary["count"] == 3
    assert summary["first_step"] == 1
    assert summary["last_step"] == 3
    assert summary["latest"] == 8.0
    assert summary["recent_mean"] == 6.0
    assert summary["all_finite"] is True


def test_scalar_summary_reports_nonfinite_values_without_serializing_nan_as_metric() -> None:
    summary = _summarize_samples(
        [_Sample(wall_time=1.0, step=1, value=math.inf)], recent_count=1
    )

    assert summary["all_finite"] is False
    assert summary["nonfinite_count"] == 1
    assert summary["latest"] is None
    assert summary["recent_mean"] is None


def test_diagnostic_thresholds_are_advisory_and_explicit() -> None:
    warnings = _diagnostic_warnings(
        {
            "train/approx_kl": _summary(0.1, maximum=0.3),
            "train/clip_fraction": _summary(0.9),
            "train/explained_variance": _summary(-2.0),
            "train/entropy_loss": _summary(-0.0001),
        }
    )

    assert warnings == [
        "recent_approx_kl_above_0.2",
        "recent_clip_fraction_mean_above_0.8",
        "recent_explained_variance_mean_below_minus_1",
        "recent_entropy_magnitude_below_0.001",
    ]
