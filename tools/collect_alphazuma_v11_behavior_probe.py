"""Collect a frozen actor-visible dataset for V1.1 policy-drift diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    return parser


def _make_environment(
    *,
    original_root: Path,
    level_id: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> HumanSpeedrunWrapper:
    base = RevengeEnv(
        config=base_config,
        level_id=level_id,
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    return HumanSpeedrunWrapper(
        base,
        input_config=input_config,
        reward_config=reward_config,
    )


def main() -> int:
    args = _parser().parse_args()
    prereg_path = args.preregistration.expanduser().resolve()
    original_root = args.original_root.expanduser().resolve()
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_COLLECTION":
        raise ValueError("probe preregistration is not frozen")
    collector_row = prereg["collector"]
    if _sha256(Path(__file__).resolve()) != collector_row["sha256"]:
        raise ValueError("collector implementation hash changed")
    source_path = prereg_path.parent.parent / prereg["environment_source"][
        "preregistration"
    ]
    if _sha256(source_path) != prereg["environment_source"]["sha256"]:
        raise ValueError("environment-source preregistration hash changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output_root = Path(prereg["output_root"])
    working_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or working_root.exists():
        raise FileExistsError(
            f"refusing to overwrite {output_root} or {working_root}"
        )
    working_root.mkdir(parents=True)

    environment = source["environment"]
    base_config = RevengeEnvConfig(**environment["base_config"])
    input_row = environment["input_profile"]
    reward_row = environment["reward_profile"]
    input_config = EliteHumanInputConfig(
        profile_id=str(input_row["profile_id"]),
        reaction_delay_ticks=int(input_row["reaction_delay_ticks"]),
        max_aim_speed_degrees_per_second=float(
            input_row["max_aim_speed_degrees_per_second"]
        ),
        max_aim_acceleration_degrees_per_second_squared=float(
            input_row["max_aim_acceleration_degrees_per_second_squared"]
        ),
        min_button_interval_ticks=int(input_row["min_button_interval_ticks"]),
    )
    reward_config = WinFirstRewardConfig(
        profile_id=str(reward_row["profile_id"]),
        win_reward=float(reward_row["win_reward"]),
        failure_reward=float(reward_row["failure_reward"]),
        time_penalty_per_native_tick=float(
            reward_row["time_penalty_per_native_tick"]
        ),
        score_progress_reward_cap=float(reward_row["score_progress_reward_cap"]),
    )

    from sb3_contrib import MaskablePPO

    collector_path = Path(prereg["collector_policy"]["path"])
    if _sha256(collector_path) != prereg["collector_policy"]["sha256"]:
        raise ValueError("collector policy hash changed")
    model = MaskablePPO.load(collector_path, device=prereg["collection"]["device"])
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    sample_stride = int(prereg["collection"]["sample_stride"])
    maximum_samples = int(prereg["collection"]["maximum_samples"])
    expected_width = int(prereg["environment_source"]["observation_width"])
    expected_nvec = np.asarray(
        prereg["environment_source"]["action_nvec"], dtype=np.int64
    )

    for level_id in prereg["environment_source"]["levels"]:
        seeds = prereg["seed_isolation"]["probe_seed_ranges"][level_id]
        if len(seeds) != int(prereg["collection"]["episodes_per_level"]):
            raise ValueError(f"seed count differs for {level_id}")
        env = _make_environment(
            original_root=original_root,
            level_id=level_id,
            base_config=base_config,
            input_config=input_config,
            reward_config=reward_config,
        )
        try:
            if env.observation_space.shape != (expected_width,):
                raise ValueError(f"observation width differs for {level_id}")
            if not np.array_equal(env.action_space.nvec, expected_nvec):
                raise ValueError(f"action space differs for {level_id}")
            for seed in seeds:
                observation, _ = env.reset(seed=int(seed))
                terminated = truncated = False
                decision = 0
                sampled = 0
                final_info: dict[str, Any] = {}
                while not (terminated or truncated):
                    action_mask = np.asarray(env.action_masks(), dtype=np.bool_)
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                        action_masks=action_mask,
                    )
                    action = np.asarray(action, dtype=np.int64)
                    if decision % sample_stride == 0 and len(observations) < maximum_samples:
                        observations.append(
                            np.asarray(observation, dtype=np.float32).copy()
                        )
                        actions.append(action.copy())
                        masks.append(action_mask.copy())
                        sampled += 1
                    observation, _, terminated, truncated, info = env.step(action)
                    final_info = dict(info)
                    decision += 1
                episodes.append(
                    {
                        "level_id": level_id,
                        "seed": int(seed),
                        "decisions": decision,
                        "samples": sampled,
                        "outcome": final_info.get("outcome"),
                        "ticks": int(final_info.get("ticks", 0)),
                        "score": int(final_info.get("score", 0)),
                    }
                )
        finally:
            env.close()

    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    mask_array = np.asarray(masks, dtype=np.bool_)
    if observation_array.ndim != 2 or observation_array.shape[1] != expected_width:
        raise RuntimeError("collected observation contract differs")
    if action_array.shape != (len(observation_array), 2):
        raise RuntimeError("collected action contract differs")
    if mask_array.shape != (len(observation_array), int(np.sum(expected_nvec))):
        raise RuntimeError("collected mask contract differs")
    working_dataset_path = working_root / "frozen_probe_dataset.npz"
    final_dataset_path = output_root / "frozen_probe_dataset.npz"
    np.savez_compressed(
        working_dataset_path,
        observations=observation_array,
        actions=action_array,
        action_masks=mask_array,
    )
    receipt = {
        "schema": "zuma-rl.alphazuma-v11-behavior-probe-dataset",
        "version": 1,
        "created_utc": _utc_now(),
        "status": "COMPLETE",
        "classification": prereg["classification"],
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "collector_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "collector": {
            "path": str(collector_path),
            "sha256": _sha256(collector_path),
            "deterministic": True,
        },
        "dataset": {
            "path": str(final_dataset_path),
            "sha256": _sha256(working_dataset_path),
            "sample_count": len(observation_array),
            "observation_shape": list(observation_array.shape),
            "action_shape": list(action_array.shape),
            "mask_shape": list(mask_array.shape),
            "dtypes": {
                "observations": str(observation_array.dtype),
                "actions": str(action_array.dtype),
                "action_masks": str(mask_array.dtype),
            },
        },
        "episodes": episodes,
        "formal_gate_exclusion": True,
    }
    _write_json(working_root / "dataset_receipt.json", receipt)
    working_root.rename(output_root)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
