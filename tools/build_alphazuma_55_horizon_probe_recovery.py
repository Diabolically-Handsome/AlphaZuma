"""Freeze the capacity-safe recovery of the failed horizon probe."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-preregistration-v1.json"
INCIDENT = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-incident-v1.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
OUTPUT = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-recovery-s99081502-preregistration-v1.json"
RUN_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "horizon-probe-recovery-s99081502-v1"
)
EXPECTED_SOURCE = "sha256:86c4aa5c0ebaed53b022d53fde7e006639ce57dfb3318e03194caa72d4b1aac7"
EXPECTED_INCIDENT = "sha256:f4724400052048a9658cb3fd04a2186cd969ecf1a1496f66697477aa031f1702"
EXPECTED_EVALUATOR = "sha256:811de7b530e6703858ae218aa9a0309724c49824d817e2f55733646ec89fd6c3"


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


def main() -> int:
    if _sha256(SOURCE.resolve(strict=True)) != EXPECTED_SOURCE:
        raise RuntimeError("source horizon probe hash mismatch")
    if _sha256(INCIDENT.resolve(strict=True)) != EXPECTED_INCIDENT:
        raise RuntimeError("horizon incident hash mismatch")
    if _sha256(EVALUATOR.resolve(strict=True)) != EXPECTED_EVALUATOR:
        raise RuntimeError("capacity-safe evaluator hash mismatch")
    if OUTPUT.exists() or RUN_ROOT.exists():
        raise FileExistsError("recovery preregistration or output root already exists")

    value = copy.deepcopy(_load(SOURCE))
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["stage"] = "engineering_horizon_probe_capacity_recovery"
    value["classification"] = (
        "engineering_recovery_reusing_consumed_seeds_not_fresh_or_formal_evidence"
    )
    value["execution"]["output_root"] = str(RUN_ROOT)
    value["evaluator"] = {
        "path": str(EVALUATOR.resolve()),
        "sha256": EXPECTED_EVALUATOR,
    }
    value["source_incident"] = {
        "path": str(INCIDENT.resolve()),
        "sha256": EXPECTED_INCIDENT,
    }
    value["recovery_contract"] = {
        "same_model_levels_and_engineering_seeds": True,
        "seeds_are_already_consumed": True,
        "formal_authority": False,
        "only_exact_actor_observation_capacity_overflow_is_fail_closed": True,
        "overflow_attempt_outcome": "loss",
        "sibling_attempts_must_complete": True,
        "all_other_exceptions_must_fail_the_shard": True,
        "source_probe_output_must_not_be_overwritten": True,
    }
    value["builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "path": str(OUTPUT.resolve()),
                "sha256": _sha256(OUTPUT.resolve()),
                "output_root_absent": not RUN_ROOT.exists(),
                "seed_reuse": [
                    value["seed_plan"]["base_seed"],
                    value["seed_plan"]["last_seed"],
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
