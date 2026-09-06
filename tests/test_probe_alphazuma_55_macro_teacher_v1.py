from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_macro_teacher_v1 as probe
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
import zuma_rl.park_settle_action_wrapper as wrapper_module
from zuma_rl.alphazuma_55 import DUAL_POSITION_LEVELS


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


def _mask(*, wait: bool = True, fire: bool = True, swap: bool = True):
    verbs = np.array([wait, fire, swap], dtype=np.bool_)
    return np.concatenate((verbs, np.ones(180, dtype=np.bool_)))


class _ScriptedTeacher:
    """Stub teacher replaying a fixed per-tick proposal script."""

    def __init__(self, script: list[tuple[int, int]]) -> None:
        self.script = list(script)
        self.calls = 0
        self.seen_widths: list[int] = []

    def act(self, observation: Any) -> np.ndarray:
        frame = np.asarray(observation).reshape(-1)
        self.seen_widths.append(int(frame.shape[0]))
        verb, aim = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return np.asarray((verb, aim), dtype=np.int64)


# ---------------------------------------------------------------------------
# (a) Teacher-to-macro adapter mapping with stubbed teachers, including
# the parked-target passthrough on wait proposals.
# ---------------------------------------------------------------------------


def test_teacher_wait_carries_parked_target_through_the_adapter() -> None:
    # The settle-gated teacher's wait proposal carries its parked target
    # bin; the shared adapter must park exactly that bin in wait_hold.
    teacher = _ScriptedTeacher([(0, 137)])
    policy = probe.TeacherPanelDecisionPolicy([teacher])
    actions, state = policy.predict(np.zeros((1, 16), dtype=np.float32))
    assert state is None
    assert actions.tolist() == [[0, 137]]
    macro_action, fallback = macro_eval.adapt_per_tick_action(
        actions[0], _mask()
    )
    assert macro_action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 137]
    assert fallback is None


def test_teacher_fire_swap_and_hop_map_like_the_student_adapter() -> None:
    for verb, aim, expected in (
        (1, 90, wrapper_module.MACRO_FIRE),
        (2, 55, wrapper_module.MACRO_SWAP),
    ):
        teacher = _ScriptedTeacher([(verb, aim)])
        actions, _ = probe.TeacherPanelDecisionPolicy([teacher]).predict(
            np.zeros((1, 16), dtype=np.float32)
        )
        macro_action, fallback = macro_eval.adapt_per_tick_action(
            actions[0], _mask()
        )
        assert macro_action.tolist() == [expected, aim]
        assert fallback is None
    # hop has no macro: parked wait_hold on the teacher's aim bin.
    teacher = _ScriptedTeacher([(3, 12)])
    actions, _ = probe.TeacherPanelDecisionPolicy([teacher]).predict(
        np.zeros((1, 16), dtype=np.float32)
    )
    macro_action, fallback = macro_eval.adapt_per_tick_action(
        actions[0], _mask()
    )
    assert macro_action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 12]
    assert fallback == macro_eval.FALLBACK_UNSUPPORTED
    # A macro-mask-illegal teacher verb also falls back, keeping the aim.
    macro_action, fallback = macro_eval.adapt_per_tick_action(
        (1, 44), _mask(fire=False)
    )
    assert macro_action.tolist() == [wrapper_module.MACRO_WAIT_HOLD, 44]
    assert fallback == macro_eval.FALLBACK_MASKED


def test_teacher_panel_facade_contract() -> None:
    teachers = [_ScriptedTeacher([(0, 1)]), _ScriptedTeacher([(1, 2)])]
    policy = probe.TeacherPanelDecisionPolicy(teachers)
    observations = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    actions, _ = policy.predict(observations, action_masks=np.ones((2, 184)))
    assert actions.shape == (2, 2)
    assert actions.dtype == np.int64
    assert actions.tolist() == [[0, 1], [1, 2]]
    # Each teacher saw its own raw frame, full width.
    assert teachers[0].seen_widths == [6]
    assert teachers[1].seen_widths == [6]
    with pytest.raises(ValueError, match="deterministic"):
        policy.predict(observations, deterministic=False)
    with pytest.raises(ValueError, match="raw frames"):
        policy.predict(np.zeros((3, 6), dtype=np.float32))
    with pytest.raises(ValueError):
        probe.TeacherPanelDecisionPolicy([])


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
                        "release_aim_bin": (target if released else None),
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


def test_stub_run_produces_macro_teacher_completion_schema() -> None:
    level_ids = ("Jungle1", "village3")
    # Teacher 0: park 70, park 70, fire 70, then waits.  Teacher 1 keeps
    # proposing hop (dual-position level): every proposal falls back.
    teachers = [
        _ScriptedTeacher([(0, 70), (0, 70), (1, 70), (0, 70)]),
        _ScriptedTeacher([(3, 20)]),
    ]
    result = probe._run_teacher_only(
        teachers=teachers,
        vector=_StubMacroVector(len(level_ids), 5),
        level_ids=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        masks_provider=lambda vector: np.ones(
            (len(level_ids), 183), dtype=np.bool_
        ),
    )
    assert result["mode"] == "teacher_only"
    completion = probe._build_completion(
        result=result,
        wrapper_contract={"schema": "zuma-rl.park-settle-macro-contract"},
        runtime={"python": "3.12", "device": "cpu"},
        wall_seconds=0.01,
        levels=level_ids,
        seed_base=probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        hold_ticks=8,
    )
    assert completion["schema"] == "zuma-rl.alphazuma-55-macro-teacher-v1"
    assert completion["mode"] == "teacher_only"
    assert completion["teacher"]["id"] == v4.TEACHER_ID
    assert completion["teacher"]["observation"] == "raw_current_frame"
    assert completion["teacher_wins_are_not_autonomy"] is True
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False
    json.dumps(completion, allow_nan=False)

    episodes = completion["episodes"]
    assert [row["level_id"] for row in episodes] == list(level_ids)
    # Same outcome/shot telemetry as the macro-eval schema.
    for row in episodes:
        for key in (
            "outcome",
            "native_outcome",
            "time_limit_truncated",
            "ticks",
            "score",
            "macro_decisions",
            "native_ticks_consumed",
            "macro_verb_counts",
            "mask_fallbacks",
            "shots",
            "shots_on_target",
            "shots_on_target_rate",
            "mean_release_error_bins",
            "unsettled_fire_macros",
        ):
            assert key in row
    # Per-level outcomes are surfaced so interface-blocked levels are
    # identifiable at a glance.
    assert completion["per_level_outcomes"] == {
        "Jungle1": "win",
        "village3": "loss",
    }
    assert completion["dual_position_levels_in_panel"] == ["village3"]
    # The hop-only teacher fell back on every one of its 5 decisions.
    hop_row = episodes[1]
    assert hop_row["mask_fallbacks"] == {
        macro_eval.FALLBACK_UNSUPPORTED: 5
    }
    assert hop_row["macro_verb_counts"] == {"wait_hold": 5}
    # Teacher 0's parked target passed through to the fire macro.
    fire_row = episodes[0]
    assert fire_row["shots"] == 1
    assert fire_row["shots_on_target"] == 1
    summary = completion["summary"]
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_macro_decisions"] == 10
    assert summary["mask_fallbacks"] == {
        macro_eval.FALLBACK_UNSUPPORTED: 5
    }


def test_adapter_contract_documents_hop_and_parked_target() -> None:
    contract = probe._adapter_contract(8)
    assert "CARRIES its parked target bin" in contract["per_tick_interface"]
    assert probe.TEACHER_ID in contract["decision_maker"]
    for level in DUAL_POSITION_LEVELS:
        assert level in contract["hop_expectation"]
    assert macro_eval.FALLBACK_UNSUPPORTED in contract["hop_expectation"]
    # Wait semantics are inherited verbatim from the shared macro-eval
    # adapter (hold_ticks plumbs through).
    assert "hold_ticks=8" in contract["wait_semantics"]


# ---------------------------------------------------------------------------
# (b) Real-engine smoke: Jungle1, one seed, small max_ticks, REAL teacher
# built from the environment's teacher_spec() through the macro stack.
# ---------------------------------------------------------------------------


def test_real_engine_smoke_jungle1_teacher_through_macro() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 2_000
    factory = macro_eval._make_macro_env_factory(
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
        teachers = v4._build_teachers(vector)
        assert len(teachers) == 1
        result = probe._run_teacher_only(
            teachers=teachers,
            vector=vector,
            level_ids=("Jungle1",),
            seed_base=probe.DEFAULT_SEED_BASE,
            max_ticks=max_ticks,
        )
    finally:
        vector.close()
    completion = probe._build_completion(
        result=result,
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
    assert entry["macro_decisions"] >= 2
    assert 0 < entry["native_ticks_consumed"] <= max_ticks
    assert entry["time_limit_truncated"] or entry["outcome"] in {
        "win",
        "loss",
    }
    # The teacher actually fired through the macro protocol, and the
    # wrapper executed every release within its settle tolerance.
    summary = completion["summary"]
    assert summary["total_shots"] >= 1
    assert summary["shots_on_target"] == summary["total_shots"]
    assert summary["shots_on_target_rate"] == 1.0
    assert entry["macro_verb_counts"].get("fire", 0) >= 1


# ---------------------------------------------------------------------------
# (c) New files only: the probe imports, never copies, the frozen pieces.
# ---------------------------------------------------------------------------


def test_new_files_only_probe_imports_frozen_dependencies() -> None:
    assert probe.macro_eval is macro_eval
    assert probe.v4 is v4
    assert probe.DEFAULT_LEVELS == macro_eval.DEFAULT_LEVELS
    assert probe.DEFAULT_SEED_BASE == v4.DEFAULT_SEED_BASE == 1_400_920_000
    assert probe.DEFAULT_MAX_TICKS == v4.DEFAULT_MAX_TICKS == 30_000
    assert probe.DEFAULT_HOLD_TICKS == macro_eval.DEFAULT_HOLD_TICKS == 8
    assert probe.TEACHER_ID == v4.TEACHER_ID
    assert probe.MODE == "teacher_only"
    # The adapter itself is macro-eval's; this module only adds the
    # teacher facade (the sole class) and the receipts.
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert source.count("\nclass ") == 1
    assert "def adapt_per_tick_action" not in source
    assert "def per_tick_masks_from_macro" not in source
    assert probe.WRAPPER_PATH.exists()
    assert probe.SCRIPT_PATH.name == "probe_alphazuma_55_macro_teacher_v1.py"
