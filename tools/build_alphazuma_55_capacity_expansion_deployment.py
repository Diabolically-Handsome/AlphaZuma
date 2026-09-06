"""Freeze an expanded AlphaZuma 55 deployment after a clean withdrawal."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
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
        raise FileExistsError(f"refusing to overwrite frozen deployment: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


def _route_range(path: Path) -> tuple[str, int, int]:
    route = _read_json(path)
    _require(
        route.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
        and route.get("status") == "FROZEN_BEFORE_TRAINING",
        f"route is not frozen: {path}",
    )
    runs = route.get("runs")
    _require(isinstance(runs, list) and len(runs) == 1, f"route has wrong run count: {path}")
    run = runs[0]
    return (
        str(run["id"]),
        int(run["episode_seed_base"]),
        int(run["episode_seed_last"]),
    )


def _expanded_route_references(
    expansion: dict[str, Any],
    *,
    current_fixed_routes: list[dict[str, Any]],
    expected_distilled_route_id: str,
) -> list[dict[str, str]]:
    if expansion.get("version") == 1:
        path = _bound_path(expansion["replicate_route"], "replicate route")
        return [{"path": str(path), "sha256": _sha256(path)}]

    registry = expansion.get("route_registry")
    _require(isinstance(registry, list), "version 2 route registry is absent")
    expected_count = int(expansion["postprocess_supersession"]["registered_route_count"])
    _require(len(registry) == expected_count, "version 2 route registry count differs")
    current_ids = {
        _route_range(Path(str(reference["path"])).resolve(strict=True))[0]
        for reference in current_fixed_routes
    }
    current_ids.add(expected_distilled_route_id)
    additional: list[dict[str, str]] = []
    registry_ids: set[str] = set()
    for index, reference in enumerate(registry):
        path = _bound_path(reference, f"route registry {index}")
        route_id, first, last = _route_range(path)
        _require(reference.get("run_id") == route_id, "route registry id differs")
        _require(
            int(reference.get("episode_seed_first")) == first
            and int(reference.get("episode_seed_last")) == last,
            "route registry seed namespace differs",
        )
        _require(route_id not in registry_ids, "route registry id is duplicated")
        registry_ids.add(route_id)
        if route_id not in current_ids:
            additional.append({"path": str(path), "sha256": _sha256(path)})
    _require(len(registry_ids) == expected_count, "version 2 route registry is incomplete")
    _require(
        len(current_ids | { _route_range(Path(row["path"]))[0] for row in additional })
        == expected_count,
        "expanded route registry does not match the deployment",
    )
    return additional


def _assert_disjoint(ranges: Iterable[tuple[str, int, int]]) -> None:
    rows = list(ranges)
    _require(len({row[0] for row in rows}) == len(rows), "route ids are duplicated")
    for index, (left_id, left_first, left_last) in enumerate(rows):
        _require(left_first <= left_last, f"invalid route seed range: {left_id}")
        for right_id, right_first, right_last in rows[index + 1 :]:
            _require(
                left_last < right_first or right_last < left_first,
                f"route seed ranges overlap: {left_id} and {right_id}",
            )


def _validate_withdrawal(
    withdrawal: dict[str, Any],
    *,
    expansion_sha256: str,
    current_deployment_sha256: str,
) -> None:
    _require(
        withdrawal.get("schema")
        == "zuma-rl.alphazuma-55-waiting-postprocess-withdrawal"
        and withdrawal.get("version") == 1
        and withdrawal.get("status") == "PASS",
        "unexpected V3 withdrawal receipt",
    )
    _require(
        withdrawal["capacity_expansion"]["sha256"] == expansion_sha256,
        "withdrawal does not bind capacity expansion",
    )
    _require(
        withdrawal["withdrawn_deployment"]["sha256"]
        == current_deployment_sha256,
        "withdrawal does not bind current deployment",
    )
    preconditions = withdrawal["preconditions"]
    _require(
        preconditions["old_controller_phase"] == "WAITING_FOR_TRAINING"
        and preconditions["old_formal_roots_absent"] is True
        and preconditions["formal_seed_consumption"] is False,
        "withdrawal formal boundary is invalid",
    )
    processes = withdrawal["processes"]
    _require(
        processes["old_controller_absent"] is True
        and processes["old_deployer_absent"] is True,
        "old postprocess processes remain alive",
    )
    _require(
        withdrawal["milestone_controller"]["status"] in {"COMPLETE", "ERROR"},
        "milestone controller is not terminal",
    )
    _require(
        withdrawal["power_state"]["restoration_requested"] is False,
        "withdrawal unexpectedly requested early power restoration",
    )


def _live_process_inventory() -> list[str]:
    commands: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command:
            commands.append(command)
    return commands


def build_deployment(
    *,
    expansion_path: Path,
    expected_expansion_sha256: str,
    withdrawal_path: Path,
    current_deployment_path: Path,
    process_inventory: list[str] | None = None,
) -> dict[str, Any]:
    _require(
        _sha256(expansion_path) == expected_expansion_sha256,
        "capacity expansion hash mismatch",
    )
    expansion = _read_json(expansion_path)
    expansion_version = expansion.get("version")
    expansion_status = expansion.get("status")
    _require(
        expansion.get("schema")
        == "zuma-rl.alphazuma-55-capacity-expansion-preregistration"
        and (
            (expansion_version == 1 and expansion_status == "FROZEN_BEFORE_REPLICATE_TRAINING")
            or (
                expansion_version == 2
                and expansion_status == "FROZEN_BEFORE_HARD_FRONTIER_TRAINING"
            )
        ),
        "unexpected capacity expansion preregistration",
    )
    current = _read_json(current_deployment_path)
    current_sha256 = _sha256(current_deployment_path)
    _require(
        current.get("schema")
        == "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        and current.get("status") == "FROZEN_BEFORE_FORMAL_SEEDS",
        "unexpected current deployment preregistration",
    )
    _require(
        expansion["postprocess_supersession"]["current_deployment"]["sha256"]
        == current_sha256,
        "capacity expansion does not bind current deployment",
    )
    _require(
        Path(str(expansion["postprocess_supersession"]["current_deployment"]["path"])).resolve(strict=True)
        == current_deployment_path,
        "capacity expansion binds a different current deployment path",
    )
    withdrawal = _read_json(withdrawal_path)
    _validate_withdrawal(
        withdrawal,
        expansion_sha256=expected_expansion_sha256,
        current_deployment_sha256=current_sha256,
    )
    _bound_path(withdrawal["withdrawal_builder"], "withdrawal builder")
    milestone_path = Path(str(expansion["safe_switch_gate"]["wait_for_milestone_controller_to_reach_terminal_complete_or_error"])).resolve(strict=True)
    milestone = _read_json(milestone_path)
    _require(
        withdrawal["milestone_controller"]["sha256"] == _sha256(milestone_path),
        "withdrawal milestone status hash mismatch",
    )
    _require(
        Path(str(withdrawal["milestone_controller"]["path"])).resolve(strict=True)
        == milestone_path
        and withdrawal["milestone_controller"]["status"] == milestone.get("status"),
        "withdrawal milestone status differs from the actual terminal artifact",
    )

    inventory = _live_process_inventory() if process_inventory is None else process_inventory
    current_name = current_deployment_path.name
    current_postprocess_name = Path(str(current["postprocess_preregistration"])).name
    _require(
        not any(
            (
                current_name in command
                and "deploy_alphazuma_55_postprocess_parallel_v2.py" in command
            )
            or (
                current_postprocess_name in command
                and "run_alphazuma_55_postprocess_parallel_v2.py" in command
            )
            for command in inventory
        ),
        "old deployment or controller process is still alive",
    )

    old_outputs = current["formal_outputs"]
    _require(
        not any(Path(str(path)).exists() for name, path in old_outputs.items() if name != "controller"),
        "old formal evaluation roots are no longer absent",
    )
    fixed_routes = [copy.deepcopy(item) for item in current["fixed_routes"]]
    for index, reference in enumerate(fixed_routes):
        _bound_path(reference, f"fixed route {index}")
    additional = _expanded_route_references(
        expansion,
        current_fixed_routes=fixed_routes,
        expected_distilled_route_id=str(current["expected_distilled_route"]["route_id"]),
    )
    fixed_routes.extend(additional)
    registered_route_count = int(
        expansion["postprocess_supersession"].get(
            "registered_route_count",
            len(fixed_routes) + 1,
        )
    )
    _require(
        len(fixed_routes) + 1 == registered_route_count,
        "expanded fixed route count differs from the preregistration",
    )

    ranges = [_route_range(Path(str(item["path"])).resolve(strict=True)) for item in fixed_routes]
    expected = current["expected_distilled_route"]
    ranges.append(
        (
            str(expected["route_id"]),
            int(expected["episode_seed_first"]),
            int(expected["episode_seed_last"]),
        )
    )
    _assert_disjoint(ranges)

    supersession = expansion["postprocess_supersession"]
    new_postprocess = Path(str(supersession["new_postprocess_preregistration"])).resolve()
    new_deployment_path = Path(str(supersession["new_deployment_preregistration"])).resolve()
    new_audit = Path(str(supersession["new_independent_audit_receipt"])).resolve()
    new_deployment_root = Path(str(supersession["new_deployment_output_root"])).resolve()
    new_outputs = {name: Path(str(path)).resolve() for name, path in supersession["new_formal_outputs"].items()}
    _require(
        not any(path.exists() for path in [new_postprocess, new_deployment_path, new_audit, new_deployment_root, *new_outputs.values()]),
        "one or more expanded output paths already exist",
    )
    _require(
        len(set(new_outputs.values())) == 4
        and set(new_outputs) == {"controller", "selection", "final_blind", "continuous"},
        "expanded formal output roots are invalid",
    )

    result = copy.deepcopy(current)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["supersedes"] = {
        "path": str(withdrawal_path),
        "sha256": _sha256(withdrawal_path),
    }
    result["capacity_expansion"] = {
        "path": str(expansion_path),
        "sha256": expected_expansion_sha256,
    }
    result["capacity_expansion_deployment_builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    result["fixed_routes"] = fixed_routes
    result["postprocess_preregistration"] = str(new_postprocess)
    result["formal_outputs"] = {name: str(path) for name, path in new_outputs.items()}
    result["independent_audit_receipt"] = str(new_audit)
    result["deployment_output_root"] = str(new_deployment_root)
    _require(
        result["parallelism_decision"] == current["parallelism_decision"]
        and result["migrations"] == current["migrations"]
        and result["expected_distilled_route"] == current["expected_distilled_route"]
        and result["implementation"] == current["implementation"]
        and result["restore"] == current["restore"],
        "an immutable V3 semantic changed while building V4",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-preregistration", required=True, type=Path)
    parser.add_argument("--expected-expansion-sha256", required=True)
    parser.add_argument("--withdrawal-receipt", required=True, type=Path)
    parser.add_argument("--current-deployment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen deployment: {output}")
    value = build_deployment(
        expansion_path=args.expansion_preregistration.expanduser().resolve(strict=True),
        expected_expansion_sha256=args.expected_expansion_sha256,
        withdrawal_path=args.withdrawal_receipt.expanduser().resolve(strict=True),
        current_deployment_path=args.current_deployment.expanduser().resolve(strict=True),
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "fixed_routes": len(value["fixed_routes"]),
                "registered_routes": len(value["fixed_routes"]) + 1,
                "formal_outputs": value["formal_outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
