"""Tests for the lag-stacked recovery-intervention probe (v5).

Smoke-tests the v5 completion contract through the stubbed-env path the
v4 probe tests use: stacked observations reach the policy, the
deterministic teacher sees ONLY the raw current frame (the leading
slice), the intervention logic is v4's untouched, and the completion
schema bumps to v5 while recording stack_lags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
from tools.alphazuma_55_recovery_intervention_v4 import (
    DecisionGateConfigV4,
    RecoveryInterventionConfigV4,
)


RAW_DIM = 6
FRAMES = 3
STACKED_DIM = RAW_DIM * FRAMES
LAG_SENTINEL = -55.0


def _stacked_observation(count: int, tick: int) -> np.ndarray:
    """Current frame = tick marker; lagged tail = sentinel the teacher
    must never see."""

    observations = np.full(
        (count, STACKED_DIM), LAG_SENTINEL, dtype=np.float32
    )
    observations[:, :RAW_DIM] = float(tick)
    return observations


class _StubStackedVector:
    """SubprocVecEnv stand-in emitting STACKED observations."""

    def __init__(self, count: int, episode_ticks: int) -> None:
        self.count = count
        self.episode_ticks = episode_ticks
        self._tick = 0

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._tick = 0
        return _stacked_observation(self.count, 0)

    def step(self, actions: np.ndarray) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        tick = self._tick
        self._tick += 1
        done = tick == self.episode_ticks - 1
        infos = []
        for index in range(self.count):
            verb = int(actions[index][0])
            aim = int(actions[index][1])
            outcome = ("win" if index == 0 else "loss") if done else None
            infos.append(
                {
                    "score": (tick + 1) * (index + 1),
                    "chain_length": 5,
                    "ticks": tick + 1,
                    "outcome": outcome,
                    "native_outcome": outcome,
                    "TimeLimit.truncated": False,
                    "human_speedrun": {
                        "executed_verb": verb,
                        "executed_aim_bin": aim,
                        "desired_verb": verb,
                        "button_enqueued": verb == 1,
                    },
                }
            )
        observations = _stacked_observation(self.count, self._tick)
        rewards = np.zeros(self.count, dtype=np.float64)
        dones = np.full(self.count, done, dtype=np.bool_)
        return observations, rewards, dones, infos

    def close(self) -> None:
        pass


class _StubStackedModel:
    """Dead student that ASSERTS it receives the full stacked width."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.observed_widths: set[int] = set()

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        observations = np.asarray(observations)
        assert observations.shape == (self.count, STACKED_DIM)
        # The stacked tail must actually reach the policy.
        assert bool(
            np.all(observations[:, RAW_DIM:] == LAG_SENTINEL)
        )
        self.observed_widths.add(int(observations.shape[1]))
        return (
            np.tile(np.array([0, 10], dtype=np.int64), (self.count, 1)),
            None,
        )


class _RecordingRawTeacher:
    """Fires bin 40 every fifteenth decision; ASSERTS raw-frame input."""

    def __init__(self) -> None:
        self.calls = 0
        self.observed_widths: set[int] = set()

    def act(self, observation: Any) -> tuple[int, int]:
        observation = np.asarray(observation)
        assert observation.shape == (RAW_DIM,)
        # The sentinel lag region must have been stripped.
        assert bool(np.all(observation != LAG_SENTINEL))
        self.observed_widths.add(int(observation.shape[0]))
        verb = 1 if self.calls % 15 == 0 else 0
        self.calls += 1
        return (verb, 40)


def test_raw_frame_teacher_strips_the_lagged_tail() -> None:
    class _Inner:
        def act(self, observation: Any) -> tuple[int, int]:
            self.seen = np.asarray(observation)
            return (0, 1)

    inner = _Inner()
    teacher = v5.RawFrameTeacher(inner, raw_observation_dim=RAW_DIM)
    stacked = np.arange(STACKED_DIM, dtype=np.float32)
    assert teacher.act(stacked) == (0, 1)
    assert np.array_equal(inner.seen, stacked[:RAW_DIM])
    with pytest.raises(ValueError, match="narrower"):
        teacher.act(np.zeros(RAW_DIM - 1, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        v5.RawFrameTeacher(inner, raw_observation_dim=0)


def test_probe_v5_completion_schema_with_stubbed_stacked_env() -> None:
    level_ids = ("Jungle1", "Jungle9")
    config = RecoveryInterventionConfigV4()
    gate = DecisionGateConfigV4()
    teachers_by_mode: dict[str, list[_RecordingRawTeacher]] = {}
    models_by_mode: dict[str, _StubStackedModel] = {}
    modes = []
    for mode in v5.EXPECTED_MODES:
        teachers = [_RecordingRawTeacher() for _ in level_ids]
        model = _StubStackedModel(len(level_ids))
        teachers_by_mode[mode] = teachers
        models_by_mode[mode] = model
        result = v5._run_stacked_mode(
            mode=mode,
            model=model,
            vector=_StubStackedVector(len(level_ids), 60),
            teachers=teachers,
            level_ids=level_ids,
            seed_base=v5.DEFAULT_SEED_BASE,
            max_ticks=100,
            controller_config=config,
            raw_observation_dim=RAW_DIM,
            masks_provider=lambda vector: np.ones(
                (len(level_ids), 184), dtype=np.bool_
            ),
        )
        modes.append(result)

    # Stacked observations reached the policy; raw frames reached the
    # teachers -- in BOTH modes.
    for mode in v5.EXPECTED_MODES:
        assert models_by_mode[mode].observed_widths == {STACKED_DIM}
        for teacher in teachers_by_mode[mode]:
            assert teacher.calls > 0
            assert teacher.observed_widths == {RAW_DIM}

    decision = v4._decision(gate=gate, level_ids=level_ids, modes=modes)
    completion = v5._build_completion(
        modes=modes,
        decision=decision,
        source={
            "model_path": "stub://model",
            "model_sha256": "sha256:stub",
        },
        runtime={"python": "3.12", "device": "cpu"},
        wall_seconds=0.01,
        controller_config=config,
        gate=gate,
        preregistration=None,
        stack_lags=(0, 4, 8),
        raw_observation_dim=RAW_DIM,
    )
    assert completion["schema"] == (
        "zuma-rl.alphazuma-55-recovery-intervention-v5-probe-completion"
    )
    json.dumps(completion, allow_nan=False)

    stack = completion["observation_stack"]
    assert stack["stack_lags"] == [0, 4, 8]
    assert stack["frames"] == FRAMES
    assert stack["raw_observation_dim"] == RAW_DIM
    assert stack["stacked_observation_dim"] == STACKED_DIM
    assert stack["teacher_observation"] == (
        "raw_current_frame_leading_slice"
    )
    assert completion["base_probe"]["module"] == (
        "tools.probe_alphazuma_55_recovery_intervention_v4"
    )

    # v4's closed-loop accounting and gate survive the wrapper untouched.
    by_mode = {row["mode"]: row for row in completion["modes"]}
    assert set(by_mode) == {"student_only", "recovery_intervention"}
    for row in by_mode.values():
        summary = row["summary"]
        for key in (
            "shots_on_target_rate",
            "decision_relevant_teacher_fraction",
            "teacher_initiated_shot_fraction",
            "aim_intent_error",
        ):
            assert key in summary
    intervention = by_mode["recovery_intervention"]["summary"]
    assert intervention["interventions"] >= 2
    assert intervention["teacher_initiated_shot_fraction"] == (
        pytest.approx(1.0)
    )
    assert completion["decision"]["status"] in {"PASS", "FAIL"}
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False


def test_parser_and_validate_only_report_stack_lags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = v5.build_parser()
    args = parser.parse_args(
        [
            "--model-path",
            "model.zip",
            "--original-root",
            "root",
            "--run-dir",
            "run",
        ]
    )
    assert args.stack_lags == "0,4,8"

    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"stub-model-bytes")
    assert (
        v5.main(
            [
                "--model-path",
                str(model_path),
                "--original-root",
                str(tmp_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--stack-lags",
                "0,4,8",
                "--validate-only",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "VALID"
    assert printed["stack_lags"] == [0, 4, 8]
    assert printed["modes"] == ["student_only", "recovery_intervention"]
    assert printed["training_authority"] is False
    assert not (tmp_path / "run").exists()

    with pytest.raises(ValueError, match="first stack lag"):
        v5.main(
            [
                "--model-path",
                str(model_path),
                "--original-root",
                str(tmp_path),
                "--run-dir",
                str(tmp_path / "run2"),
                "--stack-lags",
                "4,0",
                "--validate-only",
            ]
        )
