from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import probe_alphazuma_55_macro_eval_v1 as probe
from tools import probe_alphazuma_55_recovery_intervention_v1 as v1
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
import zuma_rl.park_settle_action_wrapper as wrapper_module


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


def _mask(*, wait: bool = True, fire: bool = True, swap: bool = True):
    verbs = np.array([wait, fire, swap], dtype=np.bool_)
    return np.concatenate((verbs, np.ones(180, dtype=np.bool_)))


# ---------------------------------------------------------------------------
# (a) Adapter mapping, including mask fallback, with stubbed inputs.
# ---------------------------------------------------------------------------


def test_adapter_maps_per_tick_verbs_to_macro_actions() -> None:
    action, fallback = probe.adapt_per_tick_action((0, 37), _mask())
    assert action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 37]
    assert fallback is None
    action, fallback = probe.adapt_per_tick_action((1, 90), _mask())
    assert action.tolist() == [wrapper_module.MACRO_FIRE, 90]
    assert fallback is None
    action, fallback = probe.adapt_per_tick_action((2, 0), _mask())
    assert action.tolist() == [wrapper_module.MACRO_SWAP, 0]
    assert fallback is None
    assert action.dtype == np.int64


def test_adapter_hop_and_masked_verbs_fall_back_to_wait_hold() -> None:
    # hop (per-tick verb 3) has no macro: parked wait_hold on its aim bin.
    action, fallback = probe.adapt_per_tick_action((3, 12), _mask())
    assert action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 12]
    assert fallback == probe.FALLBACK_UNSUPPORTED
    # A masked macro verb falls back but keeps the model's aim target.
    action, fallback = probe.adapt_per_tick_action((1, 55), _mask(fire=False))
    assert action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 55]
    assert fallback == probe.FALLBACK_MASKED
    action, fallback = probe.adapt_per_tick_action((2, 7), _mask(swap=False))
    assert action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 7]
    assert fallback == probe.FALLBACK_MASKED
    # wait_hold itself can never be masked at a decision point.
    with pytest.raises(RuntimeError, match="wait_hold is masked"):
        probe.adapt_per_tick_action((0, 5), _mask(wait=False))
    with pytest.raises(ValueError):
        probe.adapt_per_tick_action((4, 5), _mask())
    with pytest.raises(ValueError):
        probe.adapt_per_tick_action((0, 180), _mask())


def test_per_tick_masks_derived_from_macro_masks() -> None:
    macro = np.stack([_mask(), _mask(fire=False)])
    per_tick = probe.per_tick_masks_from_macro(macro)
    assert per_tick.shape == (2, 4 + 180)
    assert per_tick.dtype == np.bool_
    # hop is always masked off: no macro exposes it.
    assert per_tick[:, 3].tolist() == [False, False]
    assert per_tick[0, :3].tolist() == [True, True, True]
    assert per_tick[1, :3].tolist() == [True, False, True]
    assert bool(np.all(per_tick[:, 4:]))
    single = probe.per_tick_masks_from_macro(_mask(swap=False))
    assert single.shape == (184,)
    assert single[:4].tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="macro mask width"):
        probe.per_tick_masks_from_macro(np.ones((2, 184), dtype=np.bool_))


# ---------------------------------------------------------------------------
# Stubbed run loop: completion contract without an engine.
# ---------------------------------------------------------------------------


class _StubMacroVector:
    """Macro-wrapped SubprocVecEnv stand-in emitting park_settle info."""

    def __init__(self, count: int, episode_macros: int) -> None:
        self.count = count
        self.episode_macros = episode_macros
        self._step = 0

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._step = 0
        return np.zeros((self.count, 8), dtype=np.float32)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
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
        observations = np.zeros((self.count, 8), dtype=np.float32)
        rewards = np.zeros(self.count, dtype=np.float64)
        dones = np.full(self.count, done, dtype=np.bool_)
        return observations, rewards, dones, infos

    def close(self) -> None:
        pass


class _StubPerTickModel:
    """Scripted per-tick student: fires bin 40 every third decision."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0
        self.mask_widths: list[int] = []

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        masks = np.asarray(action_masks)
        self.mask_widths.append(int(masks.shape[-1]))
        verb = 1 if self.calls % 3 == 0 else 0
        self.calls += 1
        return (
            np.tile(np.array([verb, 40], dtype=np.int64), (self.count, 1)),
            None,
        )


def test_stub_run_produces_macro_eval_completion_schema() -> None:
    level_ids = ("Jungle1", "Jungle9")
    model = _StubPerTickModel(len(level_ids))
    result = probe._run_student_only(
        model=model,
        vector=_StubMacroVector(len(level_ids), 9),
        level_ids=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        masks_provider=lambda vector: np.ones(
            (len(level_ids), 183), dtype=np.bool_
        ),
    )
    # The per-tick model always receives the derived 4+180 mask.
    assert set(model.mask_widths) == {184}
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
    )
    assert completion["schema"] == "zuma-rl.alphazuma-55-macro-eval-v1"
    assert completion["mode"] == "student_only"
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False
    json.dumps(completion, allow_nan=False)

    episodes = completion["episodes"]
    assert [row["level_id"] for row in episodes] == list(level_ids)
    assert [row["seed"] for row in episodes] == [
        probe.DEFAULT_SEED_BASE,
        probe.DEFAULT_SEED_BASE + 1,
    ]
    summary = completion["summary"]
    assert summary["attempts"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["truncations"] == 0
    # 9 macro decisions per env, fire on decisions 0, 3, 6.
    assert summary["total_macro_decisions"] == 18
    assert summary["total_shots"] == 6
    # Stub releases land 1 bin off target: inside the 2-bin tolerance.
    assert summary["shots_on_target"] == 6
    assert summary["shots_on_target_rate"] == 1.0
    assert summary["mean_release_error_bins"] == 1.0
    assert summary["macro_verb_counts"] == {"fire": 6, "wait_hold": 12}
    assert summary["mask_fallbacks"] == {}
    assert summary["total_score"] == 90 + 180
    assert summary["mean_ticks"] == 180.0
    assert summary["mean_native_ticks_per_macro"] == 20.0


# ---------------------------------------------------------------------------
# (b) Real-engine smoke: Jungle1, one seed, scripted stub policy through
# the real macro-wrapped stack.
# ---------------------------------------------------------------------------


def test_real_engine_smoke_jungle1_macro_stack() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 400
    factory = probe._make_macro_env_factory(
        original_root=RETAIL_ROOT,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
    )
    vector = DummyVecEnv([factory])
    try:
        wrapper_contract = vector.env_method("contract")[0]
        assert wrapper_contract["schema"] == (
            "zuma-rl.park-settle-macro-contract"
        )
        assert wrapper_contract["config"]["hold_ticks"] == 8
        model = _StubPerTickModel(1)
        # Fire every third decision so the very first (gun still loading,
        # macro fire masked) exercises the wait_hold fallback for real.
        result = probe._run_student_only(
            model=model,
            vector=vector,
            level_ids=("Jungle1",),
            seed_base=probe.DEFAULT_SEED_BASE,
            max_ticks=max_ticks,
        )
    finally:
        vector.close()
    completion = probe._build_completion(
        result=result,
        source={"model_path": "stub://model", "model_sha256": "sha256:stub"},
        wrapper_contract=wrapper_contract,
        runtime={"python": "3", "device": "cpu"},
        wall_seconds=result["runtime"]["wall_seconds"],
        levels=("Jungle1",),
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=max_ticks,
        hold_ticks=8,
    )
    assert completion["schema"] == probe.COMPLETION_SCHEMA
    json.dumps(completion, allow_nan=False)
    entry = completion["episodes"][0]
    assert entry["level_id"] == "Jungle1"
    assert entry["seed"] == probe.DEFAULT_SEED_BASE
    # At least one macro executed and the episode reached its end state.
    assert entry["macro_decisions"] >= 2
    assert 0 < entry["native_ticks_consumed"] <= max_ticks
    assert entry["time_limit_truncated"] or entry["outcome"] in {
        "win",
        "loss",
    }
    # The wrapper executed the motor protocol: at least one real release,
    # each within the settle tolerance of its commissioning target.
    summary = completion["summary"]
    assert summary["total_shots"] >= 1
    assert summary["shots_on_target"] == summary["total_shots"]
    assert summary["shots_on_target_rate"] == 1.0
    # The first fire request arrived while the gun was still loading, so
    # the adapter's masked-verb fallback fired at least once for real.
    assert entry["mask_fallbacks"].get(probe.FALLBACK_MASKED, 0) >= 1
    assert entry["macro_verb_counts"].get("fire", 0) >= 1


# ---------------------------------------------------------------------------
# (c) New files only: the probe imports, never copies or modifies, the
# frozen wrapper and the probe-v4 machinery.
# ---------------------------------------------------------------------------


def test_new_files_only_probe_imports_frozen_dependencies() -> None:
    # Identity, not copies: the probe binds the exact frozen objects.
    assert probe.ParkSettleActionWrapper is (
        wrapper_module.ParkSettleActionWrapper
    )
    assert probe.ParkSettleActionConfig is (
        wrapper_module.ParkSettleActionConfig
    )
    assert probe.MACRO_WAIT_HOLD is wrapper_module.MACRO_WAIT_HOLD
    assert probe.base is v1
    assert probe.v4 is v4
    assert probe.motor is motor
    # Panel parity with probe v4: same levels, same engineering seed base.
    assert probe.DEFAULT_LEVELS == v4.DEFAULT_LEVELS
    assert probe.DEFAULT_SEED_BASE == v4.DEFAULT_SEED_BASE == 1_400_920_000
    assert probe.DEFAULT_MAX_TICKS == v4.DEFAULT_MAX_TICKS == 30_000
    assert probe.SHOT_TOLERANCE_BINS == (
        wrapper_module.ParkSettleActionConfig().settle_tolerance_bins
    )
    # The probe module never re-implements the macro expansion or the
    # motor env stack; it holds no class definitions at all.
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "\nclass " not in source
    assert "def step(" not in source
    # The frozen dependency files exist exactly where the receipts bind.
    assert probe.WRAPPER_PATH.exists()
    assert probe.SCRIPT_PATH.name == "probe_alphazuma_55_macro_eval_v1.py"
