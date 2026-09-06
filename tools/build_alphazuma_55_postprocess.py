"""Freeze the AlphaZuma 55 postprocess execution before selection seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MASTER_SHA256 = (
    "sha256:f43e10c299332b3e4b3497ff057596775c1bf3011e1ce08fc23f832a6652f4cb"
)


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


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen preregistration: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _migration(value: str) -> dict[str, Any]:
    try:
        role, model_id, path_text = value.split("=", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("migration must be ROLE=ID=RECEIPT") from error
    path = Path(path_text).expanduser().resolve(strict=True)
    return {
        "role": role,
        "id": model_id,
        "receipt": {"path": str(path), "sha256": _sha256(path)},
    }


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def build(
    *,
    master_path: Path,
    original_root: Path,
    migrations: list[dict[str, Any]],
    route_paths: list[Path],
    controller_root: Path,
    selection_root: Path,
    final_root: Path,
    continuous_root: Path,
) -> dict[str, Any]:
    if _sha256(master_path) != EXPECTED_MASTER_SHA256:
        raise ValueError("master preregistration hash differs")
    master = _read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    if len(migrations) != 3 or len({row["role"] for row in migrations}) != 3:
        raise ValueError("exactly three uniquely named migrations are required")
    if len({row["id"] for row in migrations}) != 3:
        raise ValueError("migration model ids are duplicated")
    expected_roles = {"v1_baseline", "v11_baseline", "speed_essence_baseline"}
    if {row["role"] for row in migrations} != expected_roles:
        raise ValueError("migration roles differ from the frozen baseline inventory")

    frozen_migrations = []
    for row in migrations:
        receipt_path = Path(row["receipt"]["path"])
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema") != "zuma-rl.alphazuma-55-model-migration"
            or receipt.get("status") != "PASS"
        ):
            raise ValueError(f"migration is not PASS: {receipt_path}")
        migrated = receipt.get("migrated_model", {})
        model_path = Path(str(migrated.get("path", ""))).resolve(strict=True)
        if migrated.get("sha256") != _sha256(model_path):
            raise ValueError(f"migrated model changed: {model_path}")
        frozen_migrations.append(row)

    if not 1 <= len(route_paths) <= int(master["candidate_and_selection_rules"]["maximum_training_routes"]):
        raise ValueError("training route count is outside the frozen maximum")
    frozen_routes = []
    route_ids = []
    run_dirs = []
    seed_ranges = []
    frozen_levels = master["scope"]["included_levels_in_adventure_order"]
    training_registry = master["seed_registry"]["training"]
    for path in route_paths:
        spec = _read_json(path)
        if (
            spec.get("schema") != "zuma-rl.overnight-multilevel-preregistration"
            or spec.get("status") != "FROZEN_BEFORE_TRAINING"
        ):
            raise ValueError(f"training route is not frozen: {path}")
        if spec.get("campaign_id") != master["campaign_id"]:
            raise ValueError(f"training route campaign differs: {path}")
        if [row["id"] for row in spec.get("levels", [])] != frozen_levels:
            raise ValueError(f"training route level inventory differs: {path}")
        route_implementation = spec.get("implementation", {})
        if not isinstance(route_implementation, dict):
            raise ValueError(f"training route implementation is absent: {path}")
        for name, reference in route_implementation.items():
            if not isinstance(reference, dict) or "path" not in reference or "sha256" not in reference:
                raise ValueError(f"invalid route implementation receipt {name}: {path}")
            implementation_path = Path(str(reference["path"])).resolve(strict=True)
            if reference["sha256"] != _sha256(implementation_path):
                raise ValueError(f"training route implementation changed ({name}): {path}")
        runs = spec.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise ValueError(f"training route must contain one run: {path}")
        run = runs[0]
        route_ids.append(str(run["id"]))
        run_dirs.append(str(run["run_dir"]))
        first = int(run["episode_seed_base"])
        last = int(run["episode_seed_last"])
        if not int(training_registry["first"]) <= first <= last <= int(training_registry["last"]):
            raise ValueError(f"training route seed range escapes registry: {path}")
        seed_ranges.append((first, last, str(run["id"])))
        frozen_routes.append(_artifact(path))
    if len(set(route_ids)) != len(route_ids) or len(set(run_dirs)) != len(run_dirs):
        raise ValueError("training route ids or directories are duplicated")
    for index, (first, last, route_id) in enumerate(seed_ranges):
        for other_first, other_last, other_id in seed_ranges[index + 1 :]:
            if max(first, other_first) <= min(last, other_last):
                raise ValueError(f"training seed namespaces overlap: {route_id}, {other_id}")

    outputs = {
        "controller": str(controller_root.resolve()),
        "selection": str(selection_root.resolve()),
        "final_blind": str(final_root.resolve()),
        "continuous": str(continuous_root.resolve()),
    }
    if len(set(outputs.values())) != 4:
        raise ValueError("formal output roots overlap")
    if any(Path(value).exists() for value in outputs.values()):
        raise FileExistsError("a formal output root already exists")

    implementations = {
        "controller": PROJECT_ROOT / "tools" / "run_alphazuma_55_postprocess.py",
        "independent_auditor": PROJECT_ROOT / "tools" / "audit_alphazuma_55_result.py",
        "evaluator": PROJECT_ROOT / "tools" / "evaluate_zero_shot_multilevel.py",
        "summarizer": PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py",
        "matrix_auditor": PROJECT_ROOT / "tools" / "audit_alphazuma_v11_frontier_result.py",
        "evaluation_builder": PROJECT_ROOT / "tools" / "build_alphazuma_55_eval_contract.py",
        "postprocess_helper": PROJECT_ROOT / "tools" / "run_overnight_v11_postprocess.py",
        "candidate_helper": PROJECT_ROOT / "tools" / "run_alphazuma_speed_essence_postprocess.py",
        "gpu_power_guard": PROJECT_ROOT / "tools" / "manage_alphazuma_55_power.ps1",
        "windows_power_plan_guard": PROJECT_ROOT / "tools" / "manage_alphazuma_55_power_plan.ps1",
    }
    return {
        "schema": "zuma-rl.alphazuma-55-postprocess-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_SELECTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "original_root": str(original_root),
        "migrations": frozen_migrations,
        "training_routes": frozen_routes,
        "selection_rules": {
            "capability_ranking": master["candidate_and_selection_rules"]["selection_ranking_lexicographic"],
            "speed_ranking": [
                "number of levels with at least one win out of four",
                "total wins",
                "paired median winning ticks",
                "model sha256 ascending as deterministic tie break",
            ],
            "single_policy_finalist": "rank 1 under capability_ranking",
            "final_blind_may_not_change_finalist": True,
        },
        "seed_embargo": {
            "selection": [1_600_000_000, 1_600_000_219],
            "final_blind": [1_700_000_000, 1_700_000_439],
            "continuous": [1_800_000_000, 1_800_000_219],
            "all_candidate_hashes_frozen_before_selection": True,
            "selection_report_frozen_before_final_blind": True,
            "finalist_hash_frozen_before_continuous": True,
        },
        "implementation": {name: _artifact(path) for name, path in implementations.items()},
        "outputs": outputs,
        "schedule": master["schedule"],
        "restore_contract": {
            "gpu_limits_after_campaign_watts": {"0": 600, "1": 400},
            "windows_power_plan_after_campaign": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "restore_only_after_independent_audit_or_deadline": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--migration", action="append", type=_migration, required=True)
    parser.add_argument("--training-route", action="append", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--final-blind-root", type=Path, required=True)
    parser.add_argument("--continuous-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen preregistration: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        migrations=args.migration,
        route_paths=[path.expanduser().resolve(strict=True) for path in args.training_route],
        controller_root=args.controller_root.expanduser().resolve(),
        selection_root=args.selection_root.expanduser().resolve(),
        final_root=args.final_blind_root.expanduser().resolve(),
        continuous_root=args.continuous_root.expanduser().resolve(),
    )
    if args.validate_only:
        print(json.dumps({**value, "status": "VALID_PREVIEW"}, ensure_ascii=False, indent=2))
        return 0
    _write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "routes": len(value["training_routes"]),
                "migrations": len(value["migrations"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
