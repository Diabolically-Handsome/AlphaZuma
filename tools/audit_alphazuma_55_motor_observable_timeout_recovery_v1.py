"""Audit a completed motor-observable matrix after the V1 timeout-reader bug.

The V1 evaluator canonically emits ``outcome=null`` together with
``time_limit_truncated=true`` when an episode reaches ``max_ticks``.  The V1
postprocessor accidentally accepted only the strings ``win`` and ``loss``.
This auditor consumes the already-complete immutable shards, treats the
canonical timeout representation as a non-win, and never performs inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import (
    run_alphazuma_55_motor_observable_gradual_postprocess_v1 as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_FAILURE_ERROR = (
    "evaluation attempt contract changed: "
    "('gradual-r00-e03', 'jungle6')"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_result(row: dict[str, Any], *, max_ticks: int) -> str:
    outcome = row.get("outcome")
    truncated = row.get("time_limit_truncated")
    ticks = int(row.get("ticks", -1))
    if outcome in {"win", "loss"} and truncated is False:
        return str(outcome)
    if outcome is None and truncated is True and ticks == max_ticks:
        return "truncation"
    raise ValueError(
        "noncanonical outcome/truncation tuple: "
        f"outcome={outcome!r}, truncated={truncated!r}, ticks={ticks}"
    )


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    median = row.get("median_winning_ticks")
    return (
        -int(row["wins"]),
        -int(row["cleared_levels"]),
        -int(row["total_score"]),
        float(median) if median is not None else math.inf,
        int(row["candidate_order"]),
    )


def audit_matrix_timeout_compatible(
    *, prereg_path: Path, manifest_path: Path
) -> dict[str, Any]:
    prereg_path = prereg_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    prereg = _read(prereg_path)
    manifest = _read(manifest_path)
    models = list(manifest.get("models", []))
    levels = list(prereg.get("levels", []))
    if not models or not levels:
        raise ValueError("empty model or level inventory")
    if len({str(row.get("id")) for row in models}) != len(models):
        raise ValueError("duplicated model id")
    if len({str(row.get("id")) for row in levels}) != len(levels):
        raise ValueError("duplicated level id")

    evaluator_path = Path(str(prereg["evaluator"]["path"])).resolve(strict=True)
    if prereg["evaluator"].get("sha256") != _sha256(evaluator_path):
        raise ValueError("evaluation preregistration evaluator changed")
    shard_count = int(prereg["execution"]["shard_count"])
    output_root = Path(str(prereg["execution"]["output_root"])).resolve(
        strict=True
    )
    max_ticks = int(prereg["environment"]["base_config"]["max_ticks"])
    seed_base = int(prereg["seed_plan"]["base_seed"])
    model_hashes = {
        str(model["id"]): str(model["sha256"]) for model in models
    }
    expected = {
        (str(model["id"]), str(level["id"])): (
            seed_base + level_index,
            str(model["sha256"]),
        )
        for level_index, level in enumerate(levels)
        for model in models
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    canonical: dict[tuple[str, str], str] = {}
    shard_receipts: list[dict[str, Any]] = []

    for shard_index in range(shard_count):
        shard_path = output_root / (
            f"matrix-shard-{shard_index:02d}-of-{shard_count:02d}.json"
        )
        shard = _read(shard_path)
        start, end = controller._shard_bounds(
            len(levels), shard_index, shard_count
        )
        expected_attempts = (end - start) * len(models)
        expected_level_ids = [str(row["id"]) for row in levels[start:end]]
        shard_level_ids = [str(row["id"]) for row in shard.get("levels", [])]
        if not (
            shard.get("status") == "COMPLETE"
            and shard.get("error") is None
            and int(shard.get("shard_index", -1)) == shard_index
            and int(shard.get("shard_count", -1)) == shard_count
            and int(shard.get("completed_attempts", -1)) == expected_attempts
            and int(shard.get("expected_attempts", -1)) == expected_attempts
            and len(shard.get("attempts", [])) == expected_attempts
            and shard_level_ids == expected_level_ids
            and shard.get("preregistration", {}).get("sha256")
            == _sha256(prereg_path)
            and shard.get("models_manifest", {}).get("sha256")
            == _sha256(manifest_path)
            and shard.get("evaluator_sha256") == _sha256(evaluator_path)
        ):
            raise ValueError(f"evaluation shard failed validation: {shard_path}")
        shard_models = shard.get("models", [])
        if [str(row.get("id")) for row in shard_models] != [
            str(row["id"]) for row in models
        ]:
            raise ValueError(f"model order changed in shard: {shard_path}")
        for model in shard_models:
            if model.get("sha256") != model_hashes.get(str(model.get("id"))):
                raise ValueError(f"model hash changed in shard: {shard_path}")
        for row in shard["attempts"]:
            key = (str(row.get("model_id")), str(row.get("level_id")))
            if key in observed or key not in expected:
                raise ValueError(f"unexpected or duplicated evaluation attempt: {key}")
            expected_seed, expected_hash = expected[key]
            if not (
                int(row.get("seed", -1)) == expected_seed
                and row.get("model_sha256") == expected_hash
                and int(row.get("attempt_index", -1)) == 0
            ):
                raise ValueError(f"evaluation attempt identity changed: {key}")
            canonical[key] = _canonical_result(row, max_ticks=max_ticks)
            observed[key] = row
        shard_receipts.append(
            {"path": str(shard_path), "sha256": _sha256(shard_path)}
        )

    if set(observed) != set(expected):
        raise ValueError("evaluation matrix is incomplete")

    summaries: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        keys = [(model_id, str(level["id"])) for level in levels]
        rows = [observed[key] for key in keys]
        results = [canonical[key] for key in keys]
        wins = [
            row for row, result in zip(rows, results, strict=True)
            if result == "win"
        ]
        summaries.append(
            {
                "model_id": model_id,
                "model_sha256": model["sha256"],
                "observation_profile": model["observation_profile"],
                "promotable": bool(model.get("promotable", False)),
                "candidate_order": int(model["candidate_order"]),
                "attempts": len(rows),
                "wins": len(wins),
                "losses": sum(result == "loss" for result in results),
                "truncations": sum(
                    result == "truncation" for result in results
                ),
                "non_wins": sum(result != "win" for result in results),
                "cleared_levels": len({str(row["level_id"]) for row in wins}),
                "total_score": sum(int(row.get("score", 0)) for row in rows),
                "median_winning_ticks": (
                    statistics.median(int(row["ticks"]) for row in wins)
                    if wins
                    else None
                ),
                "capacity_overflows": sum(
                    bool(row.get("observation_capacity_overflow"))
                    for row in rows
                ),
            }
        )
    ranking = sorted(summaries, key=_rank_key)
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    return {
        "schema": "zuma-rl.alphazuma-55-timeout-compatible-matrix-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": _utc_now(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "models_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "shards": shard_receipts,
        "attempts": len(observed),
        "timeout_semantics": {
            "canonical_tuple": {
                "outcome": None,
                "time_limit_truncated": True,
                "ticks": max_ticks,
            },
            "ranking_classification": "non_win",
        },
        "model_summaries": summaries,
        "ranking": ranking,
    }


def audit_recovery(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    output_root: Path,
    expected_failure_sha256: str,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    if _sha256(plan_path) != expected_plan_sha256:
        raise ValueError("frozen postprocess plan hash differs")
    plan = controller.validate_plan(plan_path, expected_plan_sha256)
    failure_path = output_root / "failure.json"
    status_path = output_root / "controller_status.json"
    failure = _read(failure_path)
    status = _read(status_path)
    if not (
        _sha256(failure_path) == expected_failure_sha256
        and failure.get("status") == "ERROR"
        and failure.get("error_type") == "ValueError"
        and failure.get("error") == EXPECTED_FAILURE_ERROR
        and failure.get("formal_seed_consumption") == "NONE"
        and status.get("status") == "ERROR"
        and status.get("error") == EXPECTED_FAILURE_ERROR
        and status.get("formal_seed_consumption") == "NONE"
    ):
        raise ValueError("predecessor timeout-reader incident differs")

    screen_prereg = output_root / "screen-preregistration.json"
    screen_manifest = output_root / "screen-models-manifest.json"
    screen = audit_matrix_timeout_compatible(
        prereg_path=screen_prereg,
        manifest_path=screen_manifest,
    )
    original_screen_path = output_root / "screen-audit.json"
    original_screen = _read(original_screen_path)
    selected = [
        str(row["model_id"])
        for row in screen["ranking"][: int(plan["screen"]["select_top"])]
    ]
    if not (
        original_screen.get("status") == "PASS"
        and original_screen.get("attempts") == screen["attempts"]
        and original_screen.get("selected_for_full55_gate") == selected
    ):
        raise ValueError("predecessor screen selection differs")

    gate_prereg = output_root / "full55-preregistration.json"
    gate_manifest = output_root / "full55-models-manifest.json"
    gate_manifest_value = _read(gate_manifest)
    gate_ids = [str(row["id"]) for row in gate_manifest_value["models"]]
    if gate_ids != selected:
        raise ValueError("full55 manifest differs from frozen screen selection")
    gate = audit_matrix_timeout_compatible(
        prereg_path=gate_prereg,
        manifest_path=gate_manifest,
    )
    minimum_wins = int(plan["full55_gate"]["minimum_wins"])
    minimum_levels = int(plan["full55_gate"]["minimum_cleared_levels"])
    passing = [
        row for row in gate["ranking"]
        if row["promotable"]
        and int(row["wins"]) >= minimum_wins
        and int(row["cleared_levels"]) >= minimum_levels
    ]
    if passing:
        raise ValueError("timeout-compatible recomputation unexpectedly passed Gate")
    if any(
        int(row.get("capacity_overflows", -1)) != 0
        for row in gate["model_summaries"]
    ):
        raise ValueError("capacity overflow invalidates recovery audit")

    return {
        "schema": "zuma-rl.alphazuma-55-timeout-reader-recovery-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": _utc_now(),
        "classification": "POST_HOC_READER_CORRECTION_FROM_IMMUTABLE_MATRIX",
        "controller_result": "COMPLETE_NO_PROMOTION",
        "gate_passed": False,
        "formal_seed_consumption": "NONE",
        "formal_candidate_authority": False,
        "inference_performed": False,
        "seed_reuse": False,
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "auditor": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "predecessor_failure": {
            "path": str(failure_path),
            "sha256": _sha256(failure_path),
            "error": EXPECTED_FAILURE_ERROR,
        },
        "predecessor_status": {
            "path": str(status_path),
            "sha256": _sha256(status_path),
        },
        "screen": {
            "recomputed": screen,
            "predecessor_audit": {
                "path": str(original_screen_path),
                "sha256": _sha256(original_screen_path),
            },
            "selected_for_full55_gate": selected,
        },
        "full55_gate": {
            "recomputed": gate,
            "thresholds": {
                "minimum_wins": minimum_wins,
                "minimum_cleared_levels": minimum_levels,
            },
            "passing_promotable_models": [],
        },
        "correction_boundary": {
            "changed": ["reader_accepts_canonical_timeout_tuple"],
            "unchanged": [
                "models",
                "model_hashes",
                "levels",
                "seeds",
                "attempts",
                "trajectories",
                "ranking_order",
                "promotion_thresholds",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-failure-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit_recovery(
        plan_path=args.plan,
        expected_plan_sha256=str(args.expected_plan_sha256),
        output_root=args.output_root,
        expected_failure_sha256=str(args.expected_failure_sha256),
    )
    _write_new(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "controller_result": receipt["controller_result"],
                "gate_passed": receipt["gate_passed"],
                "receipt": {
                    "path": str(args.output.resolve()),
                    "sha256": _sha256(args.output.resolve(strict=True)),
                },
                "formal_seed_consumption": "NONE",
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
