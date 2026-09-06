from __future__ import annotations

from pathlib import Path

import pytest

from tools.materialize_alphazuma_55_polar_validation import _bound_path


def test_bound_path_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _bound_path("not-a-reference", "fixture")


def test_bound_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _bound_path(
            {"path": str(tmp_path / "missing.json"), "sha256": "sha256:bad"},
            "fixture",
        )
