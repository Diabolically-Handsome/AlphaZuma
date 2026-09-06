"""Audit the capacity-safe recovery of the AlphaZuma horizon probe."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREG = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-recovery-s99081502-preregistration-v1.json"
REMEDIATION = PROJECT_ROOT / "diagnostics/alphazuma-55-evaluation-capacity-remediation-s99081503-preregistration-v1.json"
RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "horizon-probe-recovery-s99081502-v1/shard-00-of-01.json"
)
OUTCOME = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-recovery-s99081502-outcome-v1.json"
EXPECTED_PREREG = "sha256:e1877c86366f58f4a9852ad67d4f98bd7693ba888b59fcea6684b16d3b328d4f"
EXPECTED_REMEDIATION = "sha256:9b79bb2b1a4fa6ddc089ba6c121292026d302163b7f30c620f16fa8a30b73237"
EXPECTED_EVALUATOR = "sha256:811de7b530e6703858ae218aa9a0309724c49824d817e2f55733646ec89fd6c3"
EXPECTED_MODEL = "sha256:4324ccf39b822ad946359e34c73a1afe62c7f00d73d4a2d1375a5b9e64a955bc"
CAPACITY_ERROR_PREFIX = "actor observation capacity exceeded:"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    prereg_path = PREREG.resolve(strict=True)
    remediation_path = REMEDIATION.resolve(strict=True)
    result_path = RESULT.resolve(strict=True)
    _require(_sha256(prereg_path) == EXPECTED_PREREG, "recovery prereg hash mismatch")
    _require(
        _sha256(remediation_path) == EXPECTED_REMEDIATION,
        "remediation prereg hash mismatch",
    )
    prereg = _load(prereg_path)
    result = _load(result_path)
    _require(
        result.get("schema") == "zuma-rl.zero-shot-multilevel-shard",
        "unexpected recovery result schema",
    )
    _require(
        result.get("status") == "COMPLETE" and result.get("error") is None,
        "recovery shard is not clean COMPLETE",
    )
    _require(
        result.get("preregistration", {}).get("sha256") == EXPECTED_PREREG,
        "result preregistration hash mismatch",
    )
    _require(
        result.get("evaluator_sha256") == EXPECTED_EVALUATOR,
        "result evaluator hash mismatch",
    )

    levels = [str(level["id"]) for level in prereg["levels"]]
    base_seed = int(prereg["seed_plan"]["base_seed"])
    expected = {
        level_id: base_seed + index for index, level_id in enumerate(levels)
    }
    attempts = result.get("attempts")
    _require(isinstance(attempts, list), "attempts are not an array")
    _require(
        len(attempts) == len(levels) == 17
        and result.get("completed_attempts") == 17
        and result.get("expected_attempts") == 17,
        "recovery did not retain all 17 sibling attempts",
    )
    seen: set[str] = set()
    overflow_rows: list[dict[str, Any]] = []
    for row in attempts:
        _require(isinstance(row, dict), "attempt is not an object")
        level_id = str(row.get("level_id"))
        _require(level_id in expected and level_id not in seen, "unexpected duplicate level")
        seen.add(level_id)
        _require(int(row.get("seed", -1)) == expected[level_id], "attempt seed mismatch")
        _require(int(row.get("attempt_index", -1)) == 0, "attempt index mismatch")
        _require(row.get("model_sha256") == EXPECTED_MODEL, "attempt model hash mismatch")
        outcome = row.get("outcome")
        truncated = bool(row.get("time_limit_truncated"))
        _require(outcome in {"win", "loss", None}, "unknown attempt outcome")
        _require(truncated == (outcome is None), "outcome/truncation semantics mismatch")
        _require(0 < int(row.get("ticks", -1)) <= 30_000, "attempt ticks out of range")
        if bool(row.get("observation_capacity_overflow")):
            _require(outcome == "loss" and not truncated, "overflow was not fail-closed loss")
            _require(
                row.get("failure_reason") == "actor_observation_capacity_overflow",
                "overflow failure reason mismatch",
            )
            _require(int(row.get("capacity_limit_balls", -1)) == 768, "capacity changed")
            _require(
                int(row.get("visible_balls_at_failure", -1)) > 768,
                "overflow did not exceed capacity",
            )
            _require(
                str(row.get("capacity_error", "")).startswith(CAPACITY_ERROR_PREFIX),
                "overflow did not retain exact error",
            )
            _require(row.get("evaluation_fail_closed") is True, "fail-closed marker missing")
            overflow_rows.append(row)
    _require(seen == set(levels), "recovery level matrix mismatch")
    _require(overflow_rows, "recovery did not reproduce any capacity overflow")
    _require(len(attempts) - len(overflow_rows) >= 1, "no sibling attempts remained")

    wins = [row for row in attempts if row["outcome"] == "win"]
    late_wins = [row for row in wins if int(row["ticks"]) > 12_000]
    payload = {
        "schema": "zuma-rl.alphazuma-55-horizon-probe-recovery-outcome",
        "version": 1,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "engineering_recovery_not_fresh_or_formal_evidence",
        "preregistration": {"path": str(prereg_path), "sha256": EXPECTED_PREREG},
        "remediation_preregistration": {
            "path": str(remediation_path),
            "sha256": EXPECTED_REMEDIATION,
        },
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "matrix_integrity": {
            "levels": 17,
            "attempts": 17,
            "seed_range": [base_seed, base_seed + 16],
            "all_siblings_completed": True,
        },
        "capacity_recovery": {
            "overflow_attempts": len(overflow_rows),
            "non_overflow_sibling_attempts": len(attempts) - len(overflow_rows),
            "all_overflows_are_explicit_losses": True,
            "overflow_records": [
                {
                    "level_id": row["level_id"],
                    "seed": row["seed"],
                    "ticks": row["ticks"],
                    "visible_balls_at_failure": row["visible_balls_at_failure"],
                    "capacity_limit_balls": row["capacity_limit_balls"],
                    "trajectory_sha256": row["trajectory_sha256"],
                }
                for row in overflow_rows
            ],
        },
        "horizon_probe": {
            "wins": len(wins),
            "wins_after_12000_ticks": len(late_wins),
            "late_win_records": [
                {
                    "level_id": row["level_id"],
                    "seed": row["seed"],
                    "ticks": row["ticks"],
                    "seconds": row["seconds"],
                    "score": row["score"],
                    "trajectory_sha256": row["trajectory_sha256"],
                }
                for row in late_wins
            ],
        },
        "promotion": {
            "capacity_safe_postprocess_authorized": True,
            "training_recipe_change_authorized": False,
            "formal_seed_consumption_authorized_only_after_waiting_deployment_supersession": True,
        },
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    with OUTCOME.resolve().open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
