"""Merge N park-settle episode collections into one runner-ready manifest.

Purpose
-------
``tools/run_alphazuma_55_coverage_replay_build_v1.py`` consumes ONE
collection run dir (``<run-dir>/episodes_manifest.json``).  Tonight's
pipeline needs the teacher-driven v1 collection and the DAgger
collection built by ``tools/collect_alphazuma_55_park_settle_dagger_v2
.py`` in a SINGLE coverage build, without copying ~73 GB of episode
NPZs.  This tool reads N collection run dirs and writes a merged
``episodes_manifest.json`` (plus a ``merge_receipt.json``) into a new
output run dir that the runner accepts unchanged.

Path resolution (provably supported, no data copied)
----------------------------------------------------
The runner resolves every entry through ``_entry_path``::

    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = manifest_dir / path

i.e. ABSOLUTE paths pass through untouched and only relative paths are
anchored at the manifest's own directory.  The v1 hydrators
(``collect_alphazuma_55_park_settle_episodes_v1._hydrate_episode``) and
the park-settle trainer loader
(``distill_alphazuma_55_park_settle_v1._entry_npz_path`` equivalent)
apply the identical absolute-passthrough rule.  Merged entries
therefore carry each episode's RESOLVED ABSOLUTE path (existence
checked at merge time) -- no symlinks, no relative traversal, no copies
-- and the sha256 receipts pass through byte-identical for the runner's
own ``verify_episode_receipts`` to enforce before any use.

Refusals (mixing incompatible collections would poison training)
----------------------------------------------------------------
* input manifest schema or entry structure not accepted by the
  runner's own ``load_collection_manifest`` (reused here, so this tool
  can never be more permissive than the runner);
* ``mask_semantics`` mismatch between inputs (exact/relaxed key
  semantics must be identical);
* ``teacher_policy_id``, ``training_label_semantics``, or
  ``environment_stack`` mismatch between inputs;
* duplicate (level_id, seed) keys across inputs (merging would
  double-count an episode);
* a level outside the frozen 55 scope;
* an output dir already holding an ``episodes_manifest.json`` or
  ``merge_receipt.json`` (frozen-output discipline).

Heterogeneous ``seed_base`` / ``seeds_per_level`` / ``max_ticks`` /
``collection_mode`` across inputs are LEGITIMATE (that is the point of
merging teacher and DAgger collections) and are recorded per source in
the merged manifest's ``sources`` list.  ``formal_seed_consumption``
stays ``false``: this tool reshapes receipts and never touches an
environment or a seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import run_alphazuma_55_coverage_replay_build_v1 as runner
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
MERGE_RECEIPT_SCHEMA = (
    "zuma-rl.alphazuma-55-park-settle-episode-manifests-merge-receipt"
)
VERSION = 1
MANIFEST_NAME = "episodes_manifest.json"
RECEIPT_NAME = "merge_receipt.json"
# Manifest keys every input must agree on; a mismatch means the inputs
# were collected under incompatible label/mask/stack contracts.
_CONSISTENCY_KEYS = (
    "mask_semantics",
    "teacher_policy_id",
    "training_label_semantics",
    "environment_stack",
)


def _load_source(run_dir: Path) -> dict[str, Any]:
    """One input run dir through the RUNNER'S OWN manifest validation."""

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    manifest_path = (run_dir / MANIFEST_NAME).resolve(strict=True)
    manifest, entries = runner.load_collection_manifest(manifest_path)
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": legacy._sha256(manifest_path),
        "manifest": manifest,
        "entries": entries,
    }


def _refuse_inconsistent(sources: Sequence[dict[str, Any]]) -> None:
    reference = sources[0]
    for source in sources[1:]:
        for key in _CONSISTENCY_KEYS:
            left = reference["manifest"].get(key)
            right = source["manifest"].get(key)
            if left != right:
                raise ValueError(
                    f"input collections disagree on {key!r}: "
                    f"{reference['manifest_path']} declares {left!r}, "
                    f"{source['manifest_path']} declares {right!r}; "
                    f"refusing to merge incompatible collections"
                )


def merge_manifests(
    *,
    input_run_dirs: Sequence[Path],
    output_run_dir: Path,
) -> dict[str, Any]:
    """Merge the collections and write the runner-ready manifest."""

    if len(input_run_dirs) < 2:
        raise ValueError(
            "at least two collection run dirs are required to merge"
        )
    output_run_dir = Path(output_run_dir).expanduser().resolve()
    merged_manifest_path = output_run_dir / MANIFEST_NAME
    receipt_path = output_run_dir / RECEIPT_NAME
    for planned in (merged_manifest_path, receipt_path):
        if planned.exists():
            raise FileExistsError(f"output already exists: {planned}")

    sources = [_load_source(run_dir) for run_dir in input_run_dirs]
    _refuse_inconsistent(sources)
    reference = sources[0]["manifest"]

    level_positions = {
        level: index for index, level in enumerate(INCLUDED_LEVELS)
    }
    seen: dict[tuple[str, int], Path] = {}
    merged_entries: list[dict[str, Any]] = []
    for source in sources:
        manifest_path = source["manifest_path"]
        for entry in source["entries"]:
            level_id = str(entry["level_id"])
            if level_id not in level_positions:
                raise ValueError(
                    f"episode level {level_id!r} is outside the frozen 55 "
                    f"scope: {manifest_path}"
                )
            key = (level_id, int(entry["seed"]))
            if key in seen:
                raise ValueError(
                    f"duplicate episode {key} in {manifest_path} and "
                    f"{seen[key]}; merging would double-count it "
                    f"(collect with disjoint --seed-base blocks)"
                )
            seen[key] = manifest_path
            # Existence check only; the sha256 RECEIPT passes through
            # unchanged for the runner's verify_episode_receipts.
            resolved = runner._entry_path(
                manifest_path.parent, entry
            ).resolve(strict=True)
            merged = dict(entry)
            merged["path"] = str(resolved)
            merged["source_manifest"] = str(manifest_path)
            merged_entries.append(merged)
    merged_entries.sort(
        key=lambda entry: (
            level_positions[str(entry["level_id"])],
            int(entry["seed"]),
        )
    )

    merged_levels = [
        level
        for level in INCLUDED_LEVELS
        if any(str(entry["level_id"]) == level for entry in merged_entries)
    ]
    source_blocks = [
        {
            "run_dir": str(source["run_dir"]),
            "episodes_manifest": {
                "path": str(source["manifest_path"]),
                "sha256": source["manifest_sha256"],
            },
            "episodes": len(source["entries"]),
            "ticks": sum(
                int(entry.get("tick_count", 0))
                for entry in source["entries"]
            ),
            "collection_mode": source["manifest"].get(
                "collection_mode", "teacher_driven_v1"
            ),
            "student_policy": source["manifest"].get("student_policy"),
            "seed_base": source["manifest"].get("seed_base"),
            "seeds_per_level": source["manifest"].get("seeds_per_level"),
            "max_ticks": source["manifest"].get("max_ticks"),
        }
        for source in sources
    ]

    script_sha256 = legacy._sha256(SCRIPT_PATH)
    merged_manifest = {
        # The runner requires this exact schema string; the merged
        # document IS a park-settle episodes manifest.
        "schema": runner.COLLECTION_MANIFEST_SCHEMA,
        "version": VERSION,
        "merged": True,
        "merged_utc": legacy._utc_now(),
        "merge_tool": {"path": str(SCRIPT_PATH), "sha256": script_sha256},
        "path_semantics": "absolute_paths_no_data_copy",
        "builder_contract": reference.get("builder_contract"),
        "teacher_policy_id": reference.get("teacher_policy_id"),
        "environment_stack": reference.get("environment_stack"),
        "training_label_semantics": reference.get(
            "training_label_semantics"
        ),
        "mask_semantics": reference.get("mask_semantics"),
        "levels": merged_levels,
        "sources": source_blocks,
        "formal_seed_consumption": False,
        "episodes": merged_entries,
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    legacy._write_json_atomic(merged_manifest_path, merged_manifest)

    receipt = {
        "schema": MERGE_RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "MERGED",
        "merged_utc": merged_manifest["merged_utc"],
        "merge_tool": {"path": str(SCRIPT_PATH), "sha256": script_sha256},
        "output_run_dir": str(output_run_dir),
        "merged_manifest": {
            "path": str(merged_manifest_path),
            "sha256": legacy._sha256(merged_manifest_path),
        },
        "sources": source_blocks,
        "episodes": len(merged_entries),
        "total_ticks": sum(
            int(entry.get("tick_count", 0)) for entry in merged_entries
        ),
        "levels": merged_levels,
        "formal_seed_consumption": False,
    }
    legacy._write_json_atomic(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-run-dirs",
        nargs="+",
        required=True,
        type=Path,
        help="collection run dirs, each holding episodes_manifest.json",
    )
    parser.add_argument(
        "--output-run-dir",
        required=True,
        type=Path,
        help=(
            "new run dir for the merged episodes_manifest.json; pass it "
            "to run_alphazuma_55_coverage_replay_build_v1 as "
            "--collection-run-dir"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = merge_manifests(
        input_run_dirs=list(args.input_run_dirs),
        output_run_dir=args.output_run_dir,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "episodes": receipt["episodes"],
                "total_ticks": receipt["total_ticks"],
                "sources": [
                    {
                        "run_dir": block["run_dir"],
                        "episodes": block["episodes"],
                        "collection_mode": block["collection_mode"],
                    }
                    for block in receipt["sources"]
                ],
                "merged_manifest": receipt["merged_manifest"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
