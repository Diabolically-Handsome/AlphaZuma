"""Unit tests for the macro-decision DAgger collector.

Mirrors the collector-v1 test seams: a stub stacked macro vector, a
scripted raw-frame teacher, and (new) a scripted student model.  The
contract under test:

* the EXECUTED action reaching the vector is the STUDENT's masked
  deterministic prediction;
* the RECORDED NPZ label stream is the TEACHER's adapter-mapped
  proposal, byte-identical to collector v1;
* the ``.executed.json`` sidecar holds the executed actions, their
  authority, and label-agreement telemetry;
* ``--teacher-drive-prob 1.0`` hands every decision to the teacher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools import collect_alphazuma_55_macro_decisions_v1 as collector
from tools import collect_alphazuma_55_macro_dagger_decisions_v1 as dagger
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
from zuma_rl import park_settle_action_wrapper as wrapper_module

STACK_LAGS = (0, 4, 8)
RAW_DIM = 40
STACKED_DIM = RAW_DIM * len(STACK_LAGS)
TICKS_PER_MACRO = 30


def _macro_info(verb: int, aim: int) -> dict[str, Any]:
    released = verb == wrapper_module.MACRO_FIRE
    return {
        "profile": "park-settle-v1",
        "macro_verb": wrapper_module.MACRO_VERB_NAMES[verb],
        "target_aim_bin": aim,
        "ticks_consumed": TICKS_PER_MACRO,
        "settle_ticks": 3,
        "settled": True,
        "button_executed": released,
        "released": released,
        "release_aim_bin": aim if released else None,
        "decision_point": True,
        "terminal": False,
    }


class _ScriptedTeacher:
    def __init__(self, script: list[tuple[int, int]]) -> None:
        self.script = list(script)
        self.calls = 0

    def act(self, observation: Any) -> np.ndarray:
        verb, aim = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return np.asarray((verb, aim), dtype=np.int64)


class _ScriptedStudent:
    """MaskablePPO stand-in: fixed macro actions, records predict calls."""

    def __init__(self, script: list[tuple[int, int]]) -> None:
        self.script = list(script)
        self.calls = 0
        self.observed_widths: list[int] = []
        self.mask_widths: list[int] = []

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        assert deterministic is True
        observations = np.asarray(observations)
        masks = np.asarray(action_masks)
        self.observed_widths.append(int(observations.shape[-1]))
        self.mask_widths.append(int(masks.shape[-1]))
        verb, aim = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        batch = observations.shape[0] if observations.ndim == 2 else 1
        actions = np.tile(
            np.asarray((verb, aim), dtype=np.int64), (batch, 1)
        )
        return actions, None


class _StubStackedMacroVector:
    def __init__(self, count: int, episode_macros: int) -> None:
        self.count = count
        self.episode_macros = episode_macros
        self.received: list[np.ndarray] = []
        self._step = 0

    def _observe(self, marker: float) -> np.ndarray:
        observations = np.full(
            (self.count, STACKED_DIM), -1.0, dtype=np.float32
        )
        observations[:, :RAW_DIM] = marker
        return observations

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._step = 0
        return self._observe(100.0)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        self.received.append(actions.copy())
        step = self._step
        self._step += 1
        done = step == self.episode_macros - 1
        infos = []
        for index in range(self.count):
            infos.append(
                {
                    "score": (step + 1) * 10,
                    "ticks": (step + 1) * TICKS_PER_MACRO,
                    "outcome": ("loss" if done else None),
                    "TimeLimit.truncated": False,
                    "park_settle": _macro_info(
                        int(actions[index][0]), int(actions[index][1])
                    ),
                }
            )
        return (
            self._observe(float(step + 1)),
            np.zeros(self.count, dtype=np.float64),
            np.full(self.count, done, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        pass


def _run_batch(
    tmp_path: Path, *, teacher_drive_prob: float
) -> tuple[list[dict[str, Any]], _StubStackedMacroVector, _ScriptedStudent, Path]:
    tasks = collector.plan_tasks(
        levels=("Jungle1",),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    episodes_dir = tmp_path / collector.EPISODES_DIRNAME
    spill_dir = tmp_path / collector.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True)
    spill_dir.mkdir(parents=True)
    teacher = _ScriptedTeacher([(0, 137), (1, 70), (0, 12), (1, 90)])
    student = _ScriptedStudent([(0, 20), (0, 21), (1, 22), (0, 23)])
    masks = np.ones((1, collector.MACRO_MASK_WIDTH), dtype=np.bool_)
    entries: list[dict[str, Any]] = []
    vector = _StubStackedMacroVector(1, 4)

    def teachers_factory(vector: Any, *, raw_observation_dim: int) -> list:
        return v5.wrap_raw_frame_teachers(
            [teacher], raw_observation_dim=raw_observation_dim
        )

    dagger._collect_batch_dagger(
        original_root=tmp_path,
        tasks=tasks,
        record_flags=[True],
        max_ticks=200,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        episodes_dir=episodes_dir,
        spill_dir=spill_dir,
        on_episode=entries.append,
        student_model=student,
        teacher_drive_prob=teacher_drive_prob,
        vector_factory=lambda **kwargs: vector,
        teachers_factory=teachers_factory,
        masks_provider=lambda vector: masks.copy(),
    )
    return entries, vector, student, episodes_dir


def test_student_drives_and_teacher_labels(tmp_path: Path) -> None:
    entries, vector, student, episodes_dir = _run_batch(
        tmp_path, teacher_drive_prob=0.0
    )
    assert len(entries) == 1
    entry = entries[0]

    # The vector executed the STUDENT's script, not the teacher's.
    executed = np.stack([row[0] for row in vector.received])
    np.testing.assert_array_equal(
        executed, np.asarray([(0, 20), (0, 21), (1, 22), (0, 23)])
    )
    # The student predicted on STACKED observations with MACRO masks.
    assert set(student.observed_widths) == {STACKED_DIM}
    assert set(student.mask_widths) == {collector.MACRO_MASK_WIDTH}

    # The NPZ label stream is the TEACHER's adapter-mapped proposals.
    npz = np.load(tmp_path / str(entry["path"]))
    np.testing.assert_array_equal(
        npz["macro_actions"],
        np.asarray([(0, 137), (1, 70), (0, 12), (1, 90)]),
    )

    # Sidecar: executed actions, authority, agreement.
    sidecar_path = episodes_dir / str(entry["executed_sidecar"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["outcome_semantics"] == "student_driven"
    assert sidecar["decisions"] == 4
    assert sidecar["teacher_driven_decisions"] == 0
    assert [row["executed"] for row in sidecar["executed_actions"]] == [
        [0, 20],
        [0, 21],
        [1, 22],
        [0, 23],
    ]
    assert [
        row["teacher_label"] for row in sidecar["executed_actions"]
    ] == [[0, 137], [1, 70], [0, 12], [1, 90]]
    assert sidecar["label_agreement"]["exact"] == 0
    assert entry["executed_sidecar_sha256"].startswith("sha256:")


def test_teacher_drive_prob_one_executes_teacher(tmp_path: Path) -> None:
    entries, vector, student, episodes_dir = _run_batch(
        tmp_path, teacher_drive_prob=1.0
    )
    executed = np.stack([row[0] for row in vector.received])
    np.testing.assert_array_equal(
        executed, np.asarray([(0, 137), (1, 70), (0, 12), (1, 90)])
    )
    sidecar = json.loads(
        (episodes_dir / str(entries[0]["executed_sidecar"])).read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["teacher_driven_decisions"] == 4
    assert sidecar["label_agreement"]["exact"] == 4


def test_cli_defaults_and_bounds() -> None:
    parser = dagger.build_parser()
    args = parser.parse_args(
        [
            "--original-root", ".",
            "--run-dir", "out",
            "--student-model", "model.zip",
            "--levels", "Jungle9",
            "--seeds-per-level", "160",
            "--seed-base", "1537000000",
        ]
    )
    assert args.teacher_drive_prob == 0.0
    assert args.parallel_envs == collector.DEFAULT_PARALLEL_ENVS
    assert args.hold_ticks == collector.DEFAULT_HOLD_TICKS
    with pytest.raises(ValueError, match="teacher_drive_prob"):
        dagger._collect_batch_dagger(
            original_root=Path("."),
            tasks=collector.plan_tasks(
                levels=("Jungle1",),
                seeds_per_level=1,
                seed_base=collector.DEFAULT_SEED_BASE,
            ),
            record_flags=[True],
            max_ticks=10,
            hold_ticks=8,
            stack_lags=STACK_LAGS,
            episodes_dir=Path("."),
            spill_dir=Path("."),
            on_episode=lambda entry: None,
            student_model=object(),
            teacher_drive_prob=1.5,
        )
