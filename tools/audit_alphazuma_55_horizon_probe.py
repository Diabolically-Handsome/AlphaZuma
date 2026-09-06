"""Independently validate and summarize the engineering horizon probe."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREG = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-preregistration-v1.json"
RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "horizon-probe-s99081501-v1/shard-00-of-01.json"
)
OUTCOME = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-outcome-v1.json"
EXPECTED_PREREG_SHA256 = "sha256:86c4aa5c0ebaed53b022d53fde7e006639ce57dfb3318e03194caa72d4b1aac7"
EXPECTED_EVALUATOR_SHA256 = "sha256:d0011f9ad5c6837315007a7fb31e128fb315e4baddb45959c5f3b98d11a6a7e3"
EXPECTED_MODEL_SHA256 = "sha256:4324ccf39b822ad946359e34c73a1afe62c7f00d73d4a2d1375a5b9e64a955bc"
TRAINING_HORIZON = 12_000
PROBE_HORIZON = 30_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    prereg_path = PREREG.resolve(strict=True)
    result_path = RESULT.resolve(strict=True)
    if _sha256(prereg_path) != EXPECTED_PREREG_SHA256:
        raise RuntimeError("horizon-probe preregistration hash mismatch")

    prereg = _load(prereg_path)
    result = _load(result_path)
    if result.get("schema") != "zuma-rl.zero-shot-multilevel-shard":
        raise RuntimeError("unexpected probe result schema")
    if result.get("status") != "COMPLETE" or result.get("error") is not None:
        raise RuntimeError("probe result is not a clean COMPLETE shard")
    if result.get("preregistration", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        raise RuntimeError("result points to a different preregistration")
    if result.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("result evaluator hash mismatch")

    expected_levels = [str(level["id"]) for level in prereg["levels"]]
    expected_seeds = {
        level_id: int(prereg["seed_plan"]["base_seed"]) + index
        for index, level_id in enumerate(expected_levels)
    }
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(expected_levels):
        raise RuntimeError("probe attempt cardinality mismatch")
    if result.get("completed_attempts") != len(attempts):
        raise RuntimeError("completed_attempts mismatch")
    if result.get("expected_attempts") != len(expected_levels):
        raise RuntimeError("expected_attempts mismatch")

    by_level: dict[str, dict[str, Any]] = {}
    for row in attempts:
        if not isinstance(row, dict):
            raise RuntimeError("probe attempt is not an object")
        level_id = str(row.get("level_id"))
        if level_id not in expected_seeds or level_id in by_level:
            raise RuntimeError(f"unexpected or duplicate probe level: {level_id}")
        if int(row.get("seed", -1)) != expected_seeds[level_id]:
            raise RuntimeError(f"seed mismatch for {level_id}")
        if int(row.get("attempt_index", -1)) != 0:
            raise RuntimeError(f"attempt index mismatch for {level_id}")
        if row.get("model_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"model hash mismatch for {level_id}")
        ticks = int(row.get("ticks", -1))
        if not 0 < ticks <= PROBE_HORIZON:
            raise RuntimeError(f"tick count outside probe horizon for {level_id}")
        truncated = bool(row.get("time_limit_truncated"))
        outcome = row.get("outcome")
        if outcome not in {None, "win", "loss"}:
            raise RuntimeError(f"unexpected outcome for {level_id}: {outcome}")
        if truncated != (ticks == PROBE_HORIZON and outcome is None):
            raise RuntimeError(f"truncation semantics mismatch for {level_id}")
        by_level[level_id] = row
    if set(by_level) != set(expected_levels):
        raise RuntimeError("probe level set mismatch")

    wins = [row for row in attempts if row["outcome"] == "win"]
    wins_after_training_horizon = [
        row for row in wins if int(row["ticks"]) > TRAINING_HORIZON
    ]
    wins_within_training_horizon = [
        row for row in wins if int(row["ticks"]) <= TRAINING_HORIZON
    ]
    losses = [row for row in attempts if row["outcome"] == "loss"]
    truncations = [row for row in attempts if row["time_limit_truncated"]]
    decision = (
        "SUPPORT_SEPARATE_LONG_HORIZON_ROUTE"
        if wins_after_training_horizon
        else "NO_DIRECT_LONG_HORIZON_WIN_IN_THIS_FROZEN_PROBE"
    )
    payload = {
        "schema": "zuma-rl.alphazuma-55-horizon-probe-outcome",
        "version": 1,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "engineering_only_not_formal_selection_evidence",
        "preregistration": {
            "path": str(prereg_path),
            "sha256": EXPECTED_PREREG_SHA256,
        },
        "result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
        },
        "matrix_integrity": {
            "levels": len(expected_levels),
            "attempts": len(attempts),
            "seed_range": [min(expected_seeds.values()), max(expected_seeds.values())],
            "training_horizon_ticks": TRAINING_HORIZON,
            "probe_horizon_ticks": PROBE_HORIZON,
        },
        "outcomes": {
            "wins": len(wins),
            "wins_within_training_horizon": len(wins_within_training_horizon),
            "wins_after_training_horizon": len(wins_after_training_horizon),
            "losses": len(losses),
            "truncations": len(truncations),
            "wins_after_training_horizon_records": [
                {
                    "level_id": row["level_id"],
                    "seed": row["seed"],
                    "ticks": row["ticks"],
                    "seconds": row["seconds"],
                    "score": row["score"],
                    "trajectory_sha256": row["trajectory_sha256"],
                }
                for row in wins_after_training_horizon
            ],
        },
        "decision": decision,
        "bounded_inference": (
            "A positive record supports a separately preregistered long-horizon "
            "training route but does not alter candidate ranking or consume formal "
            "seeds. A negative result applies only to this checkpoint and matrix."
        ),
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_new(OUTCOME.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
