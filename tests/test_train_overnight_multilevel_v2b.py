from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from tools import train_overnight_multilevel_v2 as v2
from tools import train_overnight_multilevel_v2b as v2b
from tools.train_overnight_multilevel import _sha256
from zuma_rl.alphazuma_55 import full55_environment_config
from zuma_rl.human_speedrun import EliteHumanInputConfig
from zuma_rl.observation_stack_wrapper import ObservationStackWrapper
from zuma_rl.park_settle_action_wrapper import ParkSettleActionWrapper


STACK_LAGS = (0, 4, 8)


def _base_kwargs() -> dict[str, Any]:
    return {
        "level_id": "Jungle1",
        "base_config": full55_environment_config(),
        "input_config": EliteHumanInputConfig(),
    }


def _wrapper_chain(env: gym.Env) -> list[type]:
    chain: list[type] = []
    current: Any = env
    while current is not None:
        chain.append(type(current))
        current = getattr(current, "env", None)
    return chain


# ---------------------------------------------------------------------------
# Why v2b exists: v2 cannot compose stacking via configuration alone.
# ---------------------------------------------------------------------------


def test_v2_cannot_compose_stacking_via_config_alone() -> None:
    fields = {field.name for field in dataclasses.fields(v2.TrainerV2Config)}
    assert not any("stack" in name for name in fields)
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert "ObservationStackWrapper" not in source
    assert "observation_stack_wrapper" not in source
    env = v2._make_human_env_v2(
        trainer_config=v2.build_trainer_config(
            None,
            {
                "use_park_settle_wrapper": True,
                "use_motor_observable": True,
                "max_ticks_overrides": {"jungle1": 100},
            },
        ),
        **_base_kwargs(),
    )
    try:
        assert ObservationStackWrapper not in _wrapper_chain(env)
    finally:
        env.close()


def test_v2b_is_a_delegate_not_a_copy() -> None:
    source = Path(v2b.__file__).read_text(encoding="utf-8")
    # The rollout/learn loop, receipts, and checkpointing stay v2's.
    assert "model.learn" not in source
    assert "CheckpointCallback" not in source
    assert "load_state_dict" not in source
    assert issubclass(
        v2b.StackedMultiLevelSpeedrunEnvV2, v2.MultiLevelSpeedrunEnvV2
    )
    assert v2b.v2 is v2


# ---------------------------------------------------------------------------
# Composition order: stack UNDER the macro wrapper.
# ---------------------------------------------------------------------------


def test_make_stacked_macro_env_composition_and_spaces() -> None:
    trainer_config = v2.build_trainer_config(
        None,
        {
            "use_park_settle_wrapper": True,
            "use_motor_observable": True,
            "max_ticks_overrides": {"jungle1": 300},
        },
    )
    env = v2b.make_stacked_macro_env(
        trainer_config=trainer_config,
        stack_lags=STACK_LAGS,
        **_base_kwargs(),
    )
    try:
        chain = _wrapper_chain(env)
        macro_at = chain.index(ParkSettleActionWrapper)
        stack_at = chain.index(ObservationStackWrapper)
        truncation_at = chain.index(v2.TruncationNeutralRewardWrapper)
        dense_at = chain.index(
            v2.DenseShapingObservableHumanSpeedrunWrapper
        )
        # Macro on top, stack directly under it, v2's stack below that.
        assert macro_at == 0
        assert stack_at == 1
        assert macro_at < stack_at < truncation_at < dense_at
        assert env.action_space.nvec.tolist() == [3, 180]
        stacked_width = int(env.observation_space.shape[0])
        raw_width = int(
            env.env.env.observation_space.shape[0]  # under the stack
        )
        assert stacked_width == raw_width * len(STACK_LAGS)
        observation, info = env.reset(seed=21)
        assert observation.shape == (stacked_width,)
        assert env.action_masks().shape == (3 + 180,)
        assert env.contract()["schema"] == (
            "zuma-rl.park-settle-macro-contract"
        )
        assert info["park_settle"]["profile"] == "park-settle-v1"
        # max_ticks override plumbed through v2's env maker.
        assert env.unwrapped.config.max_ticks == 300
    finally:
        env.close()


class _FrameTap(gym.ObservationWrapper):
    """Record every raw frame the wrapped env emits, one per native tick."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.frames: list[np.ndarray] = []

    def observation(self, observation: Any) -> Any:
        self.frames.append(
            np.array(observation, dtype=np.float32, copy=True).reshape(-1)
        )
        return observation


def test_stack_under_macro_advances_every_native_tick_within_one_macro() -> None:
    """The trainer-composition smoke: tiny level, one macro, exact stack.

    The obs the policy receives after ONE macro must be the feature
    concatenation of the LAST NATIVE TICK's frame with the frames 4 and
    8 NATIVE TICKS earlier -- proving the ring buffer sits under the
    macro and advanced on every native tick the macro consumed (stacking
    over the macro would lag by whole macros instead).
    """

    inner = v2._make_human_env_v2(
        trainer_config=v2.build_trainer_config(
            None,
            {
                "use_motor_observable": True,
                "max_ticks_overrides": {"jungle1": 800},
            },
        ),
        **_base_kwargs(),
    )
    tap = _FrameTap(inner)
    stacked = ObservationStackWrapper(tap, stack_lags=STACK_LAGS)
    macro = ParkSettleActionWrapper(stacked)
    try:
        raw_width = int(inner.observation_space.shape[0])
        observation, _ = macro.reset(seed=17)
        assert len(tap.frames) == 1
        # Park the cursor far away so the executed aim moves every tick.
        observation, _, terminated, truncated, info = macro.step(
            np.asarray((0, 90), dtype=np.int64)
        )
        assert not terminated and not truncated
        telemetry = info["park_settle"]
        ticks = int(telemetry["ticks_consumed"])
        assert telemetry["macro_verb"] == "wait_hold"
        assert ticks >= 8
        # One raw frame per native tick plus the reset frame.
        assert len(tap.frames) == 1 + ticks
        expected = np.concatenate(
            (tap.frames[-1], tap.frames[-1 - 4], tap.frames[-1 - 8])
        )
        np.testing.assert_array_equal(observation, expected)
        # The obs the policy receives changed ACROSS native ticks within
        # this single macro: consecutive raw frames differ, and the lag
        # slices of the stacked obs are not duplicates of the current
        # frame.
        assert any(
            not np.array_equal(tap.frames[i], tap.frames[i + 1])
            for i in range(len(tap.frames) - 1)
        )
        assert not np.array_equal(
            observation[:raw_width], observation[raw_width : 2 * raw_width]
        )
    finally:
        macro.close()


def test_stacked_multilevel_env_steps_macros_with_stacked_obs() -> None:
    environment = v2b.StackedMultiLevelSpeedrunEnvV2(
        stack_lags=STACK_LAGS,
        original_root=None,
        levels=("Jungle1",),
        weights={"Jungle1": 1.0},
        base_config=full55_environment_config(),
        input_config=EliteHumanInputConfig(),
        trainer_config=v2.build_trainer_config(
            None,
            {
                "use_park_settle_wrapper": True,
                "use_motor_observable": True,
                "max_ticks_overrides": {"jungle1": 400},
            },
        ),
        reward_profile=None,
        selector_seed=1,
        episode_seed_base=1_450_000_000,
        episode_seed_stride=1_000,
        worker_rank=0,
    )
    try:
        stacked_width = int(environment.observation_space.shape[0])
        assert stacked_width % len(STACK_LAGS) == 0
        assert environment.action_space.nvec.tolist() == [3, 180]
        observation, info = environment.reset()
        assert observation.shape == (stacked_width,)
        assert info["training_level_id"] == "Jungle1"
        assert environment.action_masks().shape == (3 + 180,)
        observation, reward, terminated, truncated, info = environment.step(
            np.asarray((0, 45), dtype=np.int64)
        )
        assert observation.shape == (stacked_width,)
        assert info["park_settle"]["ticks_consumed"] >= 8
        assert info["training_level_id"] == "Jungle1"
    finally:
        environment.close()


# ---------------------------------------------------------------------------
# --init-model injection.
# ---------------------------------------------------------------------------


def test_inject_init_model_binds_path_and_sha256(tmp_path: Path) -> None:
    model_path = tmp_path / "macro_model.zip"
    model_path.write_bytes(b"stub-model-bytes")
    prereg: dict[str, Any] = {}
    injected = v2b.inject_init_model(prereg, model_path)
    assert prereg["initial_model"] is injected
    assert injected["path"] == str(model_path.resolve())
    assert injected["sha256"] == _sha256(model_path)
    assert injected["replaced_preregistration_initial_model"] is None
    # A CLI init model wins over a preregistered one, recording it.
    old = {"path": "/somewhere/else.zip", "sha256": "sha256:old"}
    prereg = {"initial_model": dict(old)}
    injected = v2b.inject_init_model(prereg, model_path)
    assert injected["replaced_preregistration_initial_model"] == old
    with pytest.raises(FileNotFoundError):
        v2b.inject_init_model({}, tmp_path / "missing.zip")


# ---------------------------------------------------------------------------
# End-to-end: v2's whole training run through the v2b rebinding.
# ---------------------------------------------------------------------------


def _smoke_prereg(run_dir: Path) -> dict[str, Any]:
    return {
        "levels": [{"id": "Jungle1"}],
        "environment": {
            "base_config": {
                "aim_bins": 180,
                "action_mode": "factorized",
                "frame_skip": 1,
                "max_ticks": 12_000,
                "max_balls": 768,
                "max_projectiles": 32,
                "max_curves": 2,
                "actor_interface": "full55-v1",
            },
            "input_profile": {
                "profile_id": "elite-human-v1",
                "reaction_delay_ticks": 12,
                "max_aim_speed_degrees_per_second": 1080.0,
                "max_aim_acceleration_degrees_per_second_squared": 18000.0,
                "min_button_interval_ticks": 5,
            },
        },
        "schedule": {
            "training_stop_utc": "2027-01-01T00:00:00Z",
            "goal_end_utc": "2027-01-02T00:00:00Z",
        },
        "runs": [],
    }


def test_prevalidate_spaces_v2b_one_level_reports_stacked_width(
    tmp_path: Path,
) -> None:
    """Regression: prevalidation must not recurse into its own rebinding.

    ``prevalidate_spaces_v2b`` rebinds ``v2._make_human_env_v2`` to the
    stacked maker for the duration of the call; the stacked maker must
    build its inner stack from the import-time capture of the pristine
    original, or the first prevalidated level calls itself forever
    (RecursionError) and every v2b CLI invocation dies at validation.
    This drives the real rebinding path on a 1-level prereg and checks
    the reported stacked width is exactly ``len(lags) x`` the raw width.
    """

    prereg = _smoke_prereg(tmp_path / "run")
    trainer_config = v2.build_trainer_config(
        None,
        {
            "use_park_settle_wrapper": True,
            "use_motor_observable": True,
            "max_ticks_overrides": {"jungle1": 300},
        },
    )
    # Ground truth raw width from v2's own unstacked, macro-off stack.
    inner = v2._make_human_env_v2(
        trainer_config=dataclasses.replace(
            trainer_config, use_park_settle_wrapper=False
        ),
        **_base_kwargs(),
    )
    try:
        raw_width = int(inner.observation_space.shape[0])
    finally:
        inner.close()
    original = v2._make_human_env_v2
    validation = v2b.prevalidate_spaces_v2b(
        prereg=prereg,
        original_root=None,
        base_config=full55_environment_config(),
        input_config=EliteHumanInputConfig(),
        reward_profile=None,
        trainer_config=trainer_config,
        stack_lags=STACK_LAGS,
    )
    # The rebinding is restored after the call.
    assert v2._make_human_env_v2 is original
    assert validation["levels_verified"] == 1
    assert validation["action_nvec"] == [3, 180]
    assert validation["observation_shape"] == [raw_width * len(STACK_LAGS)]
    stack = validation["observation_stack"]
    assert stack["stack_lags"] == list(STACK_LAGS)
    assert stack["frames"] == len(STACK_LAGS)
    assert stack["raw_observation_dim"] == raw_width
    assert stack["stacked_observation_dim"] == raw_width * len(STACK_LAGS)
    assert stack["position"] == "under_park_settle_macro_wrapper"


def test_run_training_v2b_smoke_trains_macro_steps_on_cpu(
    tmp_path: Path,
) -> None:
    from sb3_contrib import MaskablePPO

    run_dir = tmp_path / "run"
    prereg = _smoke_prereg(run_dir)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_spec = {
        "id": "v2b-smoke",
        "run_dir": str(run_dir),
        "device": "cpu",
        "seed": 7,
        "episode_seed_base": 1_450_000_000,
        "episode_seed_stride": 1_000,
        "total_steps": 4,
        "num_envs": 1,
        "rollout_steps": 4,
        "batch_size": 4,
        "ppo_epochs": 1,
        "checkpoint_every": 4,
        "initial_weights": {"Jungle1": 1.0},
        "adaptive": {"enabled": False},
    }
    trainer_config = v2.build_trainer_config(
        None,
        {
            "use_park_settle_wrapper": True,
            "use_motor_observable": True,
            "max_ticks_overrides": {"jungle1": 3_000},
        },
    )
    original_worker = v2._make_worker_v2
    completion = v2b.run_training_v2b(
        prereg_path=prereg_path,
        prereg=prereg,
        run_spec=run_spec,
        original_root=None,
        trainer_config=trainer_config,
        validation={"smoke": True},
        stack_lags=STACK_LAGS,
    )
    # The rebinding is always restored.
    assert v2._make_worker_v2 is original_worker
    assert completion["status"] == "COMPLETE"
    assert completion["actual_steps"] >= 4
    final_path = Path(completion["final_model"]["path"])
    assert final_path.exists()
    # v2's receipts landed untouched; v2b added its delegation receipt.
    assert (run_dir / "config.json").exists()
    assert (run_dir / "initialization.json").exists()
    delegate = json.loads(
        (run_dir / "delegate_v2b.json").read_text(encoding="utf-8")
    )
    assert delegate["schema"] == v2b.DELEGATE_SCHEMA
    assert delegate["stack_lags"] == list(STACK_LAGS)
    assert delegate["stack_position"] == "under_park_settle_macro_wrapper"
    assert delegate["ring_buffer_advances_every_native_tick"] is True
    checkpoints = list((run_dir / "checkpoints").glob("*.zip"))
    assert checkpoints
    # The trained policy consumed STACKED macro observations: 4 macro
    # steps of experience, macro action space, stacked obs width.
    model = MaskablePPO.load(str(final_path), device="cpu")
    assert model.action_space.nvec.tolist() == [3, 180]
    assert int(model.observation_space.shape[0]) % len(STACK_LAGS) == 0
    status = json.loads(
        (run_dir / "training_status.json").read_text(encoding="utf-8")
    )
    assert "Jungle1" in status["per_level"]
