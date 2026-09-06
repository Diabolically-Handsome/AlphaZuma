from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.audit_alphazuma_v11_checkpoints import (
    _checkpoint_timesteps,
    _select_latest_stable_checkpoint,
)


def test_checkpoint_timesteps_requires_exact_suffix() -> None:
    assert _checkpoint_timesteps(Path("route_524288_steps.zip")) == 524288
    with pytest.raises(ValueError):
        _checkpoint_timesteps(Path("route_latest.zip"))


def test_selects_latest_stable_checkpoint(tmp_path: Path) -> None:
    old = tmp_path / "route_524288_steps.zip"
    new = tmp_path / "route_1048576_steps.zip"
    writing = tmp_path / "route_1572864_steps.zip"
    for path in (old, new, writing):
        path.write_bytes(b"checkpoint")
    now_ns = 2_000_000_000_000
    os.utime(old, ns=(now_ns - 120_000_000_000,) * 2)
    os.utime(new, ns=(now_ns - 60_000_000_000,) * 2)
    os.utime(writing, ns=(now_ns - 5_000_000_000,) * 2)

    path, steps, age = _select_latest_stable_checkpoint(
        tmp_path,
        now_ns=now_ns,
        stable_age_seconds=30,
    )
    assert path == new
    assert steps == 1048576
    assert age == 60


def test_rejects_absent_stable_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "route_524288_steps.zip"
    path.write_bytes(b"checkpoint")
    now_ns = 2_000_000_000_000
    os.utime(path, ns=(now_ns - 5_000_000_000,) * 2)
    with pytest.raises(FileNotFoundError):
        _select_latest_stable_checkpoint(
            tmp_path,
            now_ns=now_ns,
            stable_age_seconds=30,
        )
