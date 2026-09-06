"""Freeze the post-V1.1 speed/robustness experiment before training.

The freezer reuses the independently audited V1.1 environment contract, binds
all mutable inputs by SHA-256, allocates disjoint training/selection/blind seed
namespaces, and refuses to overwrite either preregistrations or run roots.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = PROJECT_ROOT / "diagnostics"
SOURCE_MASTER = (
    DIAGNOSTICS
    / "alphazuma-v1.1-frontier-postprocess-s51081301-preregistration-v1.json"
)
SOURCE_MASTER_SHA256 = (
    "sha256:3dc8a51563c763ef46b3c3415a6e4da9b5430ab3a19b586e5e0b95b98b0ffc01"
)
V1_MODEL = Path(
    "/mnt/d/ZumaTraining/overnight-v11-adaptive-recovery-s43081301-v1/"
    "checkpoints/adaptive-5090-recovery_13631488_steps.zip"
)
V11_MODEL = Path(
    "/mnt/d/ZumaTraining/alphazuma-v1.1-frontier-fixed-oom-recovery-"
    "s47081302-v1/final_model.zip"
)
TRAINER = PROJECT_ROOT / "tools" / "train_overnight_multilevel.py"
EVALUATOR = PROJECT_ROOT / "tools" / "evaluate_zero_shot_multilevel.py"
TRAINING_STOP_UTC = "2026-08-14T21:45:00Z"
GOAL_END_UTC = "2026-08-15T00:00:00Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _run(
    *,
    run_id: str,
    run_dir: str,
    device: str,
    seed: int,
    episode_seed_base: int,
    weights: dict[str, float],
    ppo_epochs: int,
    learning_rate: float,
    adaptive: bool,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "run_dir": run_dir,
        "device": device,
        "seed": seed,
        "episode_seed_base": episode_seed_base,
        "episode_seed_stride": 250_000,
        "total_steps": 60_000_000,
        "num_envs": 16,
        "rollout_steps": 256,
        "batch_size": 512,
        "ppo_epochs": ppo_epochs,
        "learning_rate": learning_rate,
        "entropy_coef": 0.0,
        "checkpoint_every": 524_288,
        "initial_weights": weights,
        "adaptive": {
            "enabled": adaptive,
            "update_every_steps": 262_144,
            "rolling_episodes": 64,
            "minimum_episodes": 4,
            "minimum_weight": 0.5,
            "maximum_weight": 8.0,
            "anchor_level": "Jungle2",
        },
    }


def _training_preregistration(
    *,
    prereg_id: str,
    objective: str,
    initial_model: dict[str, Any],
    levels: list[dict[str, Any]],
    environment: dict[str, Any],
    runs: list[dict[str, Any]],
    source_master: dict[str, Any],
) -> dict[str, Any]:
    implementation_paths = {
        "revenge_core": PROJECT_ROOT / "src" / "zuma_rl" / "revenge_core.py",
        "revenge_env": PROJECT_ROOT / "src" / "zuma_rl" / "revenge_env.py",
        "human_speedrun": PROJECT_ROOT / "src" / "zuma_rl" / "human_speedrun.py",
        "original_data": PROJECT_ROOT / "src" / "zuma_rl" / "original_data.py",
    }
    return {
        "schema": "zuma-rl.overnight-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "id": prereg_id,
        "objective": objective,
        "scope": {
            "state_policy_training_only": True,
            "visual_policy_training_authorized": False,
            "environment_semantics": "unchanged_from_independently_audited_v1.1",
            "human_input_limits": "elite-human-v1 unchanged",
        },
        "trainer": {"path": str(TRAINER), "sha256": _sha256(TRAINER)},
        "initial_model": initial_model,
        "levels": levels,
        "environment": environment,
        "runs": runs,
        "seed_isolation": {
            "training_ranges": [
                [
                    int(run["episode_seed_base"]),
                    int(run["episode_seed_base"])
                    + int(run["num_envs"]) * int(run["episode_seed_stride"])
                    - 1,
                ]
                for run in runs
            ],
            "selection_range_reserved": [1_200_000_000, 1_200_000_035],
            "final_blind_range_reserved": [1_300_000_000, 1_300_000_135],
            "prior_v1_and_v11_ranges_excluded": True,
            "no_overlap": True,
        },
        "schedule": {
            "training_stop_utc": TRAINING_STOP_UTC,
            "training_stop_local": "2026-08-14T17:45:00-04:00",
            "reserved_for_selection_blind_video_seconds": 8_100,
            "goal_end_utc": GOAL_END_UTC,
            "goal_end_local": "2026-08-14T20:00:00-04:00",
        },
        "evidence_anchor": {
            "source_master_path": str(SOURCE_MASTER),
            "source_master_sha256": SOURCE_MASTER_SHA256,
            "source_v11_gate_result": source_master["promotion_gates"],
        },
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in implementation_paths.items()
        },
    }


def main() -> int:
    if _sha256(SOURCE_MASTER) != SOURCE_MASTER_SHA256:
        raise RuntimeError("audited V1.1 source preregistration changed")
    source_master = _read_json(SOURCE_MASTER)
    levels = copy.deepcopy(source_master["levels"])
    base_environment = copy.deepcopy(source_master["environment"])

    model_specs = {
        "v1": {
            "id": "alphazuma-v1-published-champion-adaptive-mid-13631488",
            "training_steps": 13_631_488,
            "path": str(V1_MODEL),
            "sha256": _sha256(V1_MODEL),
        },
        "v11": {
            "id": "alphazuma-v1.1-selected-fixed-final-33772736",
            "training_steps": 33_772_736,
            "path": str(V11_MODEL),
            "sha256": _sha256(V11_MODEL),
        },
    }

    lowdrift_weights = {
        "Jungle1": 0.5,
        "Jungle2": 2.5,
        "Jungle3": 0.75,
        "Jungle4": 0.75,
        "jungle6": 0.75,
        "Jungle7": 5.0,
        "Jungle8": 0.75,
        "Jungle9": 3.0,
        "Jungle10": 1.0,
        "village1": 1.0,
        "village2": 1.0,
        "village4": 2.5,
        "village5": 4.5,
        "village6": 6.0,
        "village7": 2.5,
        "village8": 3.5,
        "village10": 6.0,
    }
    speed_weights = {
        "Jungle1": 1.0,
        "Jungle2": 2.5,
        "Jungle3": 1.0,
        "Jungle4": 1.5,
        "jungle6": 1.0,
        "Jungle7": 3.0,
        "Jungle8": 1.0,
        "Jungle9": 3.0,
        "Jungle10": 2.0,
        "village1": 1.0,
        "village2": 1.0,
        "village4": 2.0,
        "village5": 2.5,
        "village6": 3.5,
        "village7": 4.0,
        "village8": 2.5,
        "village10": 3.5,
    }
    repair_weights = {
        "Jungle1": 0.75,
        "Jungle2": 2.5,
        "Jungle3": 2.0,
        "Jungle4": 0.75,
        "jungle6": 2.0,
        "Jungle7": 5.0,
        "Jungle8": 2.0,
        "Jungle9": 4.0,
        "Jungle10": 2.0,
        "village1": 1.0,
        "village2": 1.0,
        "village4": 3.0,
        "village5": 5.0,
        "village6": 6.0,
        "village7": 4.0,
        "village8": 3.0,
        "village10": 6.0,
    }

    lowdrift_runs = [
        _run(
            run_id="essence-v1-lowdrift-fixed-5090",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v1-lowdrift-fixed-s71081401-v1",
            device="cuda:0",
            seed=71_081_401,
            episode_seed_base=1_100_000_000,
            weights=lowdrift_weights,
            ppo_epochs=1,
            learning_rate=2.5e-6,
            adaptive=False,
        ),
        _run(
            run_id="essence-v1-lowdrift-adaptive-5080",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v1-lowdrift-adaptive-s71081402-v1",
            device="cuda:1",
            seed=71_081_402,
            episode_seed_base=1_104_000_000,
            weights=lowdrift_weights,
            ppo_epochs=1,
            learning_rate=5.0e-6,
            adaptive=True,
        ),
    ]

    speed_environment = copy.deepcopy(base_environment)
    speed_environment["reward_profile"] = {
        "profile_id": "win-time-score-v1-speed-essence-v1",
        "win_reward": 10.0,
        "failure_reward": -10.0,
        "time_penalty_per_native_tick": -0.0002,
        "score_progress_reward_cap": 0.01,
    }
    speed_runs = [
        _run(
            run_id="essence-v1-speed-fixed-5090",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v1-speed-fixed-s72081401-v1",
            device="cuda:0",
            seed=72_081_401,
            episode_seed_base=1_108_000_000,
            weights=speed_weights,
            ppo_epochs=5,
            learning_rate=1.0e-6,
            adaptive=False,
        ),
        _run(
            run_id="essence-v1-speed-adaptive-5080",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v1-speed-adaptive-s72081402-v1",
            device="cuda:1",
            seed=72_081_402,
            episode_seed_base=1_112_000_000,
            weights=speed_weights,
            ppo_epochs=5,
            learning_rate=2.0e-6,
            adaptive=True,
        ),
    ]

    repair_environment = copy.deepcopy(base_environment)
    repair_environment["reward_profile"] = {
        "profile_id": "win-time-score-v1-repair-v1",
        "win_reward": 10.0,
        "failure_reward": -10.0,
        "time_penalty_per_native_tick": -0.00015,
        "score_progress_reward_cap": 0.01,
    }
    repair_runs = [
        _run(
            run_id="essence-v11-repair-fixed-5090",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v11-repair-fixed-s73081401-v1",
            device="cuda:0",
            seed=73_081_401,
            episode_seed_base=1_116_000_000,
            weights=repair_weights,
            ppo_epochs=5,
            learning_rate=1.0e-6,
            adaptive=False,
        ),
        _run(
            run_id="essence-v11-repair-adaptive-5080",
            run_dir="/mnt/d/ZumaTraining/alphazuma-speed-essence-v11-repair-adaptive-s73081402-v1",
            device="cuda:1",
            seed=73_081_402,
            episode_seed_base=1_120_000_000,
            weights=repair_weights,
            ppo_epochs=5,
            learning_rate=2.0e-6,
            adaptive=True,
        ),
    ]

    specs = [
        (
            DIAGNOSTICS / "alphazuma-speed-essence-v1-lowdrift-s71081401-preregistration-v1.json",
            _training_preregistration(
                prereg_id="alphazuma-speed-essence-v1-lowdrift-s71081401-v1",
                objective=(
                    "Preserve the V1 champion with very low PPO update intensity while "
                    "rehearsing all seventeen levels and oversampling the unresolved tail."
                ),
                initial_model=model_specs["v1"],
                levels=levels,
                environment=base_environment,
                runs=lowdrift_runs,
                source_master=source_master,
            ),
        ),
        (
            DIAGNOSTICS / "alphazuma-speed-essence-v1-speed-s72081401-preregistration-v1.json",
            _training_preregistration(
                prereg_id="alphazuma-speed-essence-v1-speed-s72081401-v1",
                objective=(
                    "Retain V1's gun-limited cadence and colour swaps while doubling only "
                    "the bounded native-time penalty and keeping victory dominant."
                ),
                initial_model=model_specs["v1"],
                levels=levels,
                environment=speed_environment,
                runs=speed_runs,
                source_master=source_master,
            ),
        ),
        (
            DIAGNOSTICS / "alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json",
            _training_preregistration(
                prereg_id="alphazuma-speed-essence-v11-repair-s73081401-v1",
                objective=(
                    "Start from the selected V1.1 policy, rehearse levels where it became "
                    "slower than V1, and retain pressure on the three unresolved levels."
                ),
                initial_model=model_specs["v11"],
                levels=levels,
                environment=repair_environment,
                runs=repair_runs,
                source_master=source_master,
            ),
        ),
    ]

    master_path = (
        DIAGNOSTICS
        / "alphazuma-speed-essence-campaign-s70081401-preregistration-v1.json"
    )
    for path, _ in specs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite preregistration: {path}")
    if master_path.exists():
        raise FileExistsError(f"refusing to overwrite campaign: {master_path}")
    for _, prereg in specs:
        for run in prereg["runs"]:
            run_dir = Path(run["run_dir"])
            if run_dir.exists():
                raise FileExistsError(f"refusing to reuse run directory: {run_dir}")

    for path, prereg in specs:
        _write_exclusive(path, prereg)

    campaign = {
        "schema": "zuma-rl.alphazuma-speed-essence-campaign-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "id": "alphazuma-speed-essence-campaign-s70081401-v1",
        "objective": (
            "Preserve V1's rare fast colour-management behaviour, recover V1.1's "
            "breadth gains, and search for an independently held-out speed surprise."
        ),
        "hypotheses": [
            "A low-update V1 continuation retains speed better than the long V1.1 frontier.",
            "A doubled but bounded time penalty improves winning times without replacing victory as the dominant objective.",
            "A mild V1.1 repair curriculum can recover its speed regressions without losing its win-count gain.",
        ],
        "training_preregistrations": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in specs
        ],
        "baselines": model_specs,
        "candidate_rule": (
            "The two frozen baselines plus, for each of six routes, the checkpoint "
            "nearest half of that route's completed new steps and its final model; "
            "exactly fourteen candidates unless a route produces no eligible checkpoint, "
            "which is retained as a recorded route failure rather than substituted."
        ),
        "selection": {
            "level_ids": [
                "Jungle2",
                "Jungle4",
                "Jungle7",
                "Jungle9",
                "village4",
                "village5",
                "village6",
                "village7",
                "village10",
            ],
            "attempts_per_level": 4,
            "seed_base": 1_200_000_000,
            "last_seed": 1_200_000_035,
            "robust_ranking": [
                "Jungle2 wins at least 3 of 4",
                "Jungle7 village6 village10 levels cleared descending",
                "all selection levels cleared descending",
                "all selection wins descending",
                "median best winning ticks ascending",
                "earlier manifest position as deterministic final tie-break",
            ],
            "speed_ranking": [
                "Jungle2 wins at least 3 of 4",
                "Jungle2 Jungle4 Jungle9 village7 levels cleared descending",
                "all selection wins descending",
                "geometric mean ratio of best winning ticks to frozen V1 references ascending",
                "earlier manifest position as deterministic final tie-break",
            ],
            "frozen_v1_best_tick_references": {
                "Jungle2": 1567,
                "Jungle4": 1865,
                "Jungle9": 5578,
                "village7": 3179,
            },
            "selection_results_are_not_final_blind_claims": True,
        },
        "final_blind": {
            "models": (
                "V1 baseline, V1.1 baseline, robust-selected candidate, and "
                "speed-selected candidate; deduplicate identical selected candidates"
            ),
            "level_ids": "all seventeen frozen levels",
            "attempts_per_level": 8,
            "seed_base": 1_300_000_000,
            "last_seed": 1_300_000_135,
            "paired_same_seed": True,
            "all_attempts_must_be_reported": True,
        },
        "reporting": {
            "separate_best_seed_speed_from_robust_win_rate": True,
            "no_original_client_world_record_claim": True,
            "render_video_for_any_new_final_blind_level_best": True,
            "report_negative_result_without_recipe_rewrite": True,
        },
        "schedule": {
            "training_stop_utc": TRAINING_STOP_UTC,
            "training_stop_local": "2026-08-14T17:45:00-04:00",
            "goal_end_utc": GOAL_END_UTC,
            "goal_end_local": "2026-08-14T20:00:00-04:00",
        },
        "execution": {
            "selection_output_root": "/mnt/d/ZumaTraining/alphazuma-speed-essence-selection-s70081401-v1",
            "final_output_root": "/mnt/d/ZumaTraining/alphazuma-speed-essence-final-blind-s70081401-v1",
            "selection_shards": 2,
            "final_shards": 2,
            "devices": ["cuda:0", "cuda:1"],
            "parallel_envs_per_shard": 24,
        },
        "evaluator": {"path": str(EVALUATOR), "sha256": _sha256(EVALUATOR)},
        "power_plan": {
            "temporary": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "restore_at_finish_or_deadline": "381b4222-f694-41f0-9685-ff5bb260df2e",
        },
    }
    _write_exclusive(master_path, campaign)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "campaign": {"path": str(master_path), "sha256": _sha256(master_path)},
                "training_preregistrations": campaign["training_preregistrations"],
                "run_count": 6,
                "training_stop_utc": TRAINING_STOP_UTC,
                "goal_end_utc": GOAL_END_UTC,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
