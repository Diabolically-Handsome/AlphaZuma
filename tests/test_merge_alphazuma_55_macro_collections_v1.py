"""Unit tests for the macro-collection merge tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools import collect_alphazuma_55_macro_decisions_v1 as c1
from tools import distill_alphazuma_55 as legacy
from tools import merge_alphazuma_55_macro_collections_v1 as merger


def _make_collection(
    root: Path,
    name: str,
    *,
    seed_base: int,
    seeds: int,
    hold_ticks: int = 8,
) -> Path:
    run_dir = root / name
    episodes = run_dir / c1.EPISODES_DIRNAME
    episodes.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for offset in range(seeds):
        seed = seed_base + offset
        file_name = f"Jungle9-{seed}.npz"
        path = episodes / file_name
        np.savez_compressed(
            path,
            observations=np.zeros((3, 6), dtype=np.float32),
            macro_actions=np.zeros((3, 2), dtype=np.int64),
        )
        entries.append(
            {
                "level_id": "Jungle9",
                "seed": seed,
                "path": f"{c1.EPISODES_DIRNAME}/{file_name}",
                "sha256": legacy._sha256(path),
                "decision_count": 3,
                "outcome": "loss",
            }
        )
    manifest = {
        "schema": c1.MANIFEST_SCHEMA,
        "version": c1.VERSION,
        "teacher_policy_id": "t",
        "environment_stack": "stack",
        "decision_interface": {"hold_ticks": hold_ticks},
        "observation_stack": {"stack_lags": [0, 4, 8]},
        "levels": ["Jungle9"],
        "seeds_per_level": seeds,
        "seed_base": seed_base,
        "max_ticks": 30000,
        "hold_ticks": hold_ticks,
        "stack_lags": [0, 4, 8],
        "episodes": entries,
    }
    legacy._write_json_atomic(run_dir / "episodes_manifest.json", manifest)
    return run_dir


def test_merge_links_and_concatenates(tmp_path: Path) -> None:
    a = _make_collection(tmp_path, "teacher", seed_base=1000, seeds=3)
    b = _make_collection(tmp_path, "dagger", seed_base=2000, seeds=2)
    out = tmp_path / "merged"
    summary = merger.merge(source_dirs=[a, b], out_dir=out)
    assert summary["episodes"] == 5
    manifest = json.loads(
        (out / "episodes_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == c1.MANIFEST_SCHEMA
    assert len(manifest["episodes"]) == 5
    for entry in manifest["episodes"]:
        npz = out / str(entry["path"])
        assert npz.is_file()
        assert legacy._sha256(npz) == entry["sha256"]


def test_merge_refuses_duplicates_and_contract_mismatch(
    tmp_path: Path,
) -> None:
    a = _make_collection(tmp_path, "one", seed_base=1000, seeds=2)
    b = _make_collection(tmp_path, "two", seed_base=1001, seeds=2)
    with pytest.raises(ValueError, match="duplicate"):
        merger.merge(source_dirs=[a, b], out_dir=tmp_path / "dup")
    c = _make_collection(
        tmp_path, "three", seed_base=3000, seeds=1, hold_ticks=4
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        merger.merge(source_dirs=[a, c], out_dir=tmp_path / "mismatch")
