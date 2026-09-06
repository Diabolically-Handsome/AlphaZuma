from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from tools import distill_alphazuma_55_park_settle_v1 as distill_v1
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_macro_native_eval_v1 as probe
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
import zuma_rl.park_settle_action_wrapper as wrapper_module


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")
STACK_LAGS = (0, 4, 8)
STACKED_TOY_DIM = 18  # toy raw width 6 x 3 frames


def _macro_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array((3, 180), dtype=np.int64))


class _StubMacroNativeModel:
    """Scripted macro-native policy: fires bin 40 every third decision."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0
        self.mask_widths: list[int] = []
        self.masks: list[np.ndarray] = []
        self.emitted: list[np.ndarray] = []
        self.action_space = _macro_space()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STACKED_TOY_DIM,), dtype=np.float32
        )

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        masks = np.asarray(action_masks)
        self.mask_widths.append(int(masks.shape[-1]))
        self.masks.append(np.array(masks, copy=True))
        verb = (
            wrapper_module.MACRO_FIRE
            if self.calls % 3 == 0
            else wrapper_module.MACRO_WAIT_HOLD
        )
        self.calls += 1
        actions = np.tile(
            np.array([verb, 40], dtype=np.int64), (self.count, 1)
        )
        self.emitted.append(actions.copy())
        return actions, None


class _RecordingMacroVector:
    """Macro-wrapped vec-env stand-in that records every action batch."""

    def __init__(self, count: int, episode_macros: int) -> None:
        self.count = count
        self.episode_macros = episode_macros
        self.received: list[np.ndarray] = []
        self._step = 0

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._step = 0
        return np.zeros((self.count, STACKED_TOY_DIM), dtype=np.float32)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        self.received.append(actions.copy())
        step = self._step
        self._step += 1
        done = step == self.episode_macros - 1
        infos = []
        for index in range(self.count):
            verb = int(actions[index][0])
            target = int(actions[index][1])
            released = verb == wrapper_module.MACRO_FIRE
            outcome = ("win" if index == 0 else "loss") if done else None
            infos.append(
                {
                    "score": (step + 1) * 10 * (index + 1),
                    "ticks": (step + 1) * 20,
                    "outcome": outcome,
                    "native_outcome": outcome,
                    "TimeLimit.truncated": False,
                    "park_settle": {
                        "profile": "park-settle-v1",
                        "macro_verb": wrapper_module.MACRO_VERB_NAMES[verb],
                        "target_aim_bin": target,
                        "ticks_consumed": 20,
                        "settle_ticks": 3,
                        "settled": True,
                        "button_executed": released,
                        "released": released,
                        "release_aim_bin": (
                            (target + 1) % 180 if released else None
                        ),
                        "decision_point": True,
                        "terminal": done,
                    },
                }
            )
        observations = np.zeros(
            (self.count, STACKED_TOY_DIM), dtype=np.float32
        )
        rewards = np.zeros(self.count, dtype=np.float64)
        dones = np.full(self.count, done, dtype=np.bool_)
        return observations, rewards, dones, infos

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# (a) Direct macro action passthrough: no adapter path, wrapper masks
# reach the model bit for bit, the model's output reaches the vector
# verbatim.
# ---------------------------------------------------------------------------


def test_native_predictions_pass_through_as_macro_actions() -> None:
    level_ids = ("Jungle1", "Jungle9")
    model = _StubMacroNativeModel(len(level_ids))
    vector = _RecordingMacroVector(len(level_ids), 9)
    macro_masks = np.ones((len(level_ids), 183), dtype=np.bool_)
    # A masked verb must survive the widen/narrow round trip unchanged.
    macro_masks[1, wrapper_module.MACRO_SWAP] = False
    result = probe._run_macro_native(
        model=model,
        vector=vector,
        level_ids=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        masks_provider=lambda _: macro_masks,
    )
    # The native model always sees the wrapper's own (3 + 180) mask.
    assert set(model.mask_widths) == {183}
    for received in model.masks:
        np.testing.assert_array_equal(received, macro_masks)
    # Every action the vector executed is the model's output, verbatim.
    assert len(vector.received) == len(model.emitted) == 9
    for executed, emitted in zip(vector.received, model.emitted):
        np.testing.assert_array_equal(executed, emitted)
    # No adapter path: zero fallbacks anywhere.
    summary = result["summary"]
    assert summary["mask_fallbacks"] == {}
    assert all(row["mask_fallbacks"] == {} for row in result["episodes"])
    # macro_eval's telemetry is intact through the delegation.
    assert summary["attempts"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_macro_decisions"] == 18
    assert summary["total_shots"] == 6
    assert summary["shots_on_target"] == 6
    assert summary["macro_verb_counts"] == {"fire": 6, "wait_hold": 12}
    assert summary["total_score"] == 90 + 180
    assert summary["mean_native_ticks_per_macro"] == 20.0


def test_shim_rejects_masks_not_derived_from_macro_masks() -> None:
    shim = probe.MacroNativePredictShim(_StubMacroNativeModel(1))
    observations = np.zeros((1, STACKED_TOY_DIM), dtype=np.float32)
    hop_set = np.ones((1, 184), dtype=np.bool_)
    with pytest.raises(ValueError, match="hop"):
        shim.predict(observations, action_masks=hop_set)
    with pytest.raises(ValueError, match="mask width"):
        shim.predict(
            observations, action_masks=np.ones((1, 183), dtype=np.bool_)
        )


def test_run_macro_native_fails_loudly_if_a_fallback_triggers() -> None:
    """A prediction violating the macro mask can never be silently
    rewritten to wait_hold on the native path."""

    class _MisbehavingModel:
        def predict(
            self,
            observations: Any,
            deterministic: bool = True,
            action_masks: Any = None,
        ) -> tuple[np.ndarray, None]:
            return (
                np.array(
                    [[wrapper_module.MACRO_SWAP, 5]], dtype=np.int64
                ),
                None,
            )

    masks = np.ones((1, 183), dtype=np.bool_)
    masks[0, wrapper_module.MACRO_SWAP] = False
    with pytest.raises(RuntimeError, match="fallback"):
        probe._run_macro_native(
            model=_MisbehavingModel(),
            vector=_RecordingMacroVector(1, 3),
            level_ids=("Jungle1",),
            seed_base=probe.DEFAULT_SEED_BASE,
            max_ticks=50,
            masks_provider=lambda _: masks,
        )


# ---------------------------------------------------------------------------
# (b) Refusal of a per-tick 4-verb model, pointing to macro_eval_v1.
# ---------------------------------------------------------------------------


def test_per_tick_model_is_refused_with_pointer_to_macro_eval_v1() -> None:
    class _PerTickModel:
        action_space = spaces.MultiDiscrete(
            np.array((4, 180), dtype=np.int64)
        )

    with pytest.raises(
        ValueError, match="probe_alphazuma_55_macro_eval_v1"
    ):
        probe.require_macro_native_model(_PerTickModel())

    class _AlienModel:
        action_space = spaces.MultiDiscrete(
            np.array((5, 90), dtype=np.int64)
        )

    with pytest.raises(ValueError, match="MultiDiscrete"):
        probe.require_macro_native_model(_AlienModel())

    class _NoSpaceModel:
        pass

    with pytest.raises(ValueError, match="MultiDiscrete"):
        probe.require_macro_native_model(_NoSpaceModel())

    class _NativeModel:
        action_space = _macro_space()

    assert probe.require_macro_native_model(_NativeModel()) == (3, 180)


# ---------------------------------------------------------------------------
# Completion contract: schema bump, nulled adapter, stack lags, and the
# model's space signature.
# ---------------------------------------------------------------------------


def test_completion_schema_bumps_and_records_stack_and_spaces() -> None:
    level_ids = ("Jungle1", "Jungle9")
    model = _StubMacroNativeModel(len(level_ids))
    result = probe._run_macro_native(
        model=model,
        vector=_RecordingMacroVector(len(level_ids), 9),
        level_ids=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        masks_provider=lambda _: np.ones(
            (len(level_ids), 183), dtype=np.bool_
        ),
    )
    completion = probe._build_completion(
        result=result,
        source={"model_path": "stub://model", "model_sha256": "sha256:stub"},
        wrapper_contract={"schema": "zuma-rl.park-settle-macro-contract"},
        runtime={"python": "3.12", "device": "cpu"},
        wall_seconds=0.01,
        levels=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        raw_observation_dim=STACKED_TOY_DIM // len(STACK_LAGS),
        model_spaces=probe.model_space_signature(model),
    )
    assert completion["schema"] == (
        "zuma-rl.alphazuma-55-macro-native-eval-v1"
    )
    assert completion["mode"] == "student_only"
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False
    # No per-tick adapter block: replaced by the native contract.
    assert completion["adapter"] is None
    assert completion["native_interface"]["action_mapping"].startswith(
        "none"
    )
    assert completion["base_probe"]["module"] == (
        "tools.probe_alphazuma_55_macro_eval_v1"
    )
    assert completion["base_probe"]["sha256"].startswith("sha256:")
    stack = completion["observation_stack"]
    assert stack["stack_lags"] == list(STACK_LAGS)
    assert stack["frames"] == len(STACK_LAGS)
    assert stack["raw_observation_dim"] == 6
    assert stack["stacked_observation_dim"] == STACKED_TOY_DIM
    assert stack["position"] == "under_park_settle_macro_wrapper"
    assert stack["ring_buffer_advances_every_native_tick"] is True
    model_spaces = completion["model_spaces"]
    assert model_spaces["action_space"]["nvec"] == [3, 180]
    assert model_spaces["observation_space"]["shape"] == [STACKED_TOY_DIM]
    assert model_spaces["observation_space"]["dtype"] == "float32"
    # Telemetry keys inherited from macro_eval, seeds included.
    assert [row["seed"] for row in completion["episodes"]] == [
        probe.DEFAULT_SEED_BASE,
        probe.DEFAULT_SEED_BASE + 1,
    ]
    json.dumps(completion, allow_nan=False)


def test_thin_delegate_binds_frozen_dependencies_and_defaults() -> None:
    assert probe.macro_eval is macro_eval
    assert probe.v5 is v5
    assert probe.DEFAULT_LEVELS == macro_eval.DEFAULT_LEVELS
    assert len(probe.DEFAULT_LEVELS) == 12
    assert probe.DEFAULT_SEED_BASE == 1_400_920_000
    assert probe.DEFAULT_MAX_TICKS == 30_000
    assert probe.DEFAULT_HOLD_TICKS == 8
    # The delegate never re-implements the macro expansion or run loop:
    # telemetry parsing stays in macro_eval, delegated rather than copied.
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "def step(" not in source
    assert "_run_student_only" in source
    assert 'info.get("park_settle")' not in source
    parser = probe.build_parser()
    args = parser.parse_args(
        [
            "--model-path", "model.zip",
            "--original-root", "root",
            "--run-dir", "run",
        ]
    )
    assert args.seed_base == 1_400_920_000
    assert args.max_ticks == 30_000
    assert args.hold_ticks == 8
    assert args.device == "cpu"
    assert args.stack_lags == "0,4,8"
    assert args.levels == list(macro_eval.DEFAULT_LEVELS)


# ---------------------------------------------------------------------------
# (c) Real-engine CPU smoke: Jungle1, one seed, a tiny macro-native
# stacked model built in-test on the real stacked interface.
# ---------------------------------------------------------------------------


class _RecordingPredict:
    """Delegate predict, recording observation and mask widths."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.action_space = model.action_space
        self.observation_space = model.observation_space
        self.observation_widths: list[int] = []
        self.mask_widths: list[int] = []

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[Any, Any]:
        self.observation_widths.append(
            int(np.asarray(observations).shape[-1])
        )
        self.mask_widths.append(int(np.asarray(action_masks).shape[-1]))
        return self._model.predict(
            observations,
            deterministic=deterministic,
            action_masks=action_masks,
        )


def _tiny_macro_native_stacked_model(vector: Any) -> Any:
    """Small macro-native stacked MaskablePPO on the REAL interface.

    Mirrors the construction style of
    tests/test_build_alphazuma_55_macro_policy_bootstrap_v1.py: the v3
    stacked extractor over the real env's entity_polar layout, an
    interface-only env carrying the stacked Box and the macro
    MultiDiscrete([3, 180]).
    """

    from sb3_contrib import MaskablePPO

    # The layout prototype is the motor-observable wrapper: it folds the
    # appended motor tail into its globals layout, so the per-frame
    # entity slices cover the FULL raw frame the stack wrapper buffers.
    prototype = wrapper_module.resolve_human_speedrun_wrapper(
        vector.envs[0]
    )
    policy_kwargs = trainer_v3.stacked_entity_polar_policy_kwargs(
        prototype, learned_features_dim=16, stack_lags=STACK_LAGS
    )
    return MaskablePPO(
        "MlpPolicy",
        distill_v1._SpaceOnlyMaskableEnv(
            vector.observation_space, _macro_space()
        ),
        policy_kwargs=policy_kwargs,
        n_steps=8,
        batch_size=8,
        device="cpu",
        verbose=0,
        seed=99_081_680,
    )


def test_real_engine_smoke_jungle1_macro_native_stacked_model() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 400
    factory = probe._make_stacked_macro_env_factory(
        original_root=RETAIL_ROOT,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
    )
    vector = DummyVecEnv([factory])
    try:
        wrapper_contract = vector.env_method("contract")[0]
        assert wrapper_contract["schema"] == (
            "zuma-rl.park-settle-macro-contract"
        )
        assert wrapper_contract["config"]["hold_ticks"] == 8
        stacked_width = int(vector.observation_space.shape[0])
        assert stacked_width % len(STACK_LAGS) == 0
        raw_width = stacked_width // len(STACK_LAGS)
        model = _tiny_macro_native_stacked_model(vector)
        assert probe.require_macro_native_model(model) == (3, 180)
        extractor = model.policy.features_extractor
        assert type(extractor) is (
            trainer_v3.StackedRevengeEntityFeatureExtractor
        )
        assert extractor.stacked_frames == len(STACK_LAGS)
        assert extractor.frame_width == raw_width
        recording = _RecordingPredict(model)
        result = probe._run_macro_native(
            model=recording,
            vector=vector,
            level_ids=("Jungle1",),
            seed_base=probe.DEFAULT_SEED_BASE,
            max_ticks=max_ticks,
        )
    finally:
        vector.close()
    # The STACKED observation reached the policy at every decision point
    # and the model was masked with the wrapper's own (3 + 180) mask.
    assert recording.observation_widths
    assert set(recording.observation_widths) == {stacked_width}
    assert set(recording.mask_widths) == {183}
    completion = probe._build_completion(
        result=result,
        source={"model_path": "stub://tiny", "model_sha256": "sha256:stub"},
        wrapper_contract=wrapper_contract,
        runtime={"python": "3", "device": "cpu"},
        wall_seconds=result["runtime"]["wall_seconds"],
        levels=("Jungle1",),
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        raw_observation_dim=raw_width,
        model_spaces=probe.model_space_signature(model),
    )
    assert completion["schema"] == probe.COMPLETION_SCHEMA
    json.dumps(completion, allow_nan=False)
    entry = completion["episodes"][0]
    assert entry["level_id"] == "Jungle1"
    assert entry["seed"] == probe.DEFAULT_SEED_BASE
    # At least one macro executed and the episode reached an end state.
    assert entry["macro_decisions"] >= 1
    assert 0 < entry["native_ticks_consumed"] <= max_ticks
    assert entry["time_limit_truncated"] or entry["outcome"] in {
        "win",
        "loss",
    }
    # Native path: the adapter fallback machinery never fired.
    assert entry["mask_fallbacks"] == {}
    stack = completion["observation_stack"]
    assert stack["raw_observation_dim"] == raw_width
    assert stack["stacked_observation_dim"] == stacked_width
    assert completion["model_spaces"]["action_space"]["nvec"] == [3, 180]
    assert completion["model_spaces"]["observation_space"]["shape"] == [
        stacked_width
    ]
