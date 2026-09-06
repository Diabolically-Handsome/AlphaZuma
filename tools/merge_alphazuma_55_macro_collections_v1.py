"""Merge macro-decision collections into one distill-ready aggregate.

The DAgger recipe trains on the AGGREGATE of the teacher-driven
collection and the student-driven (teacher-labeled) DAgger rounds.
``distill_alphazuma_55_macro_decisions_v1`` reads exactly one
collection dir, so this tool builds one: a new run dir whose
``episodes/`` holds HARDLINKS (copy fallback) to every source episode
NPZ and whose manifest is the concatenation of the source manifests'
episode entries, with per-entry sha256 receipts carried verbatim.

Refusals (fail-closed):
* a source manifest whose schema is not the collector-v1 manifest
  schema (the DAgger collector emits the same schema by design);
* mismatched decision-interface / observation-stack / hold-ticks /
  stack-lags contracts between sources;
* duplicate (level, seed) keys across sources (seed-plan discipline
  guarantees disjoint blocks; a collision means a planning error);
* an entry whose NPZ is missing or fails its sha256 receipt.

The merged manifest's ``seed_base``/``seeds_per_level`` are those of
the FIRST source (metadata only - the distill tool reads per-entry
receipts); ``merged_from`` records every source manifest path + sha.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import collect_alphazuma_55_macro_decisions_v1 as c1
from tools import distill_alphazuma_55 as legacy

SCRIPT_PATH = Path(__file__).resolve()
VERSION = 1
CONTRACT_KEYS = (
    "decision_interface",
    "observation_stack",
    "hold_ticks",
    "stack_lags",
)


def merge(
    *, source_dirs: Sequence[Path], out_dir: Path
) -> dict[str, Any]:
    sources = [Path(d).expanduser().resolve(strict=True) for d in source_dirs]
    if len(sources) < 2:
        raise ValueError("need at least two source collections to merge")
    out_dir = Path(out_dir).expanduser().resolve()
    episodes_out = out_dir / c1.EPISODES_DIRNAME
    if (out_dir / "episodes_manifest.json").exists():
        raise ValueError(f"refusing to overwrite existing merge: {out_dir}")
    episodes_out.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for source in sources:
        manifest_path = source / "episodes_manifest.json"
        manifest = legacy._read_json(manifest_path)
        if manifest.get("schema") != c1.MANIFEST_SCHEMA:
            raise ValueError(
                f"not a macro-decisions manifest: {manifest_path} "
                f"({manifest.get('schema')})"
            )
        manifests.append(manifest)
        provenance.append(
            {
                "run_dir": str(source),
                "manifest": str(manifest_path),
                "manifest_sha256": legacy._sha256(manifest_path),
            }
        )
    first = manifests[0]
    for manifest, source in zip(manifests[1:], sources[1:]):
        for key in CONTRACT_KEYS:
            if manifest.get(key) != first.get(key):
                raise ValueError(
                    f"contract mismatch on {key!r} between {sources[0]} "
                    f"and {source}"
                )

    merged_entries: dict[tuple[str, int], dict[str, Any]] = {}
    levels: list[str] = []
    for manifest, source in zip(manifests, sources):
        for level in manifest.get("levels", []):
            if level not in levels:
                levels.append(str(level))
        for entry in manifest.get("episodes", []):
            key = (str(entry["level_id"]), int(entry["seed"]))
            if key in merged_entries:
                raise ValueError(
                    f"duplicate (level, seed) across sources: {key}"
                )
            npz_path = source / str(entry["path"])
            if not npz_path.is_file():
                raise ValueError(f"missing episode file: {npz_path}")
            if legacy._sha256(npz_path) != entry.get("sha256"):
                raise ValueError(f"sha256 receipt mismatch: {npz_path}")
            target = episodes_out / npz_path.name
            if target.exists():
                raise ValueError(
                    f"episode filename collision in merge: {target.name}"
                )
            try:
                os.link(npz_path, target)
            except OSError:
                shutil.copy2(npz_path, target)
            entry = dict(entry)
            entry["path"] = f"{c1.EPISODES_DIRNAME}/{target.name}"
            entry["merged_from"] = str(source)
            merged_entries[key] = entry

    ordered = sorted(
        merged_entries.values(),
        key=lambda entry: (str(entry["level_id"]), int(entry["seed"])),
    )
    manifest = {
        "schema": c1.MANIFEST_SCHEMA,
        "version": c1.VERSION,
        "purpose": (
            "MERGED macro-decision aggregate (DAgger recipe) for "
            "tools/distill_alphazuma_55_macro_decisions_v1.py"
        ),
        "merged_by": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "merged_from": provenance,
        "teacher_policy_id": first.get("teacher_policy_id"),
        "environment_stack": first.get("environment_stack"),
        "decision_interface": first.get("decision_interface"),
        "observation_stack": first.get("observation_stack"),
        "levels": levels,
        "seeds_per_level": first.get("seeds_per_level"),
        "seed_base": first.get("seed_base"),
        "max_ticks": first.get("max_ticks"),
        "hold_ticks": first.get("hold_ticks"),
        "stack_lags": first.get("stack_lags"),
        "teacher_losses_are_recorded_and_flagged": True,
        "formal_seed_consumption": False,
        "episodes": ordered,
    }
    legacy._write_json_atomic(out_dir / "episodes_manifest.json", manifest)
    summary = {
        "out_dir": str(out_dir),
        "sources": len(sources),
        "episodes": len(ordered),
        "decisions": sum(int(e["decision_count"]) for e in ordered),
        "levels": levels,
    }
    legacy._write_json_atomic(
        out_dir / "merge_summary.json",
        {
            "schema": "zuma-rl.alphazuma-55-macro-collections-merge-v1",
            "version": VERSION,
            **summary,
            "merged_from": provenance,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dirs", nargs="+", type=Path, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(merge(source_dirs=args.source_dirs, out_dir=args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
