from __future__ import annotations

from pathlib import Path

import pytest

from tools.materialize_alphazuma_55_polar_dagger import _bound_path


def test_bound_path_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _bound_path("bad", "fixture")


def test_bound_path_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        _bound_path(
            {"path": str(artifact), "sha256": "sha256:not-the-hash"},
            "fixture",
        )
