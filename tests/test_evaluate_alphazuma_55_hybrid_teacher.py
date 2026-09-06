from __future__ import annotations

import json

from tools.build_alphazuma_55_hybrid_teacher_probe import (
    GEOMETRIC_POLICY,
    STRATEGIC_POLICY,
    _select_teacher,
)
from tools.evaluate_alphazuma_55_hybrid_teacher import _validate_preregistration


def _row(*, outcome, truncated=False, ticks=100, score=0):
    return {
        "outcome": outcome,
        "time_limit_truncated": truncated,
        "ticks": ticks,
        "score": score,
    }


def test_teacher_selection_prefers_win_then_speed():
    geometric = _row(outcome="win", ticks=500)
    strategic = _row(outcome="loss", ticks=900, score=9999)
    assert _select_teacher(geometric, strategic)[0] == GEOMETRIC_POLICY

    strategic = _row(outcome="win", ticks=400)
    assert _select_teacher(geometric, strategic)[0] == STRATEGIC_POLICY


def test_teacher_selection_nonwin_rule_is_deterministic():
    geometric = _row(outcome="loss", ticks=500, score=100)
    strategic = _row(outcome=None, truncated=True, ticks=1200, score=0)
    assert _select_teacher(geometric, strategic)[0] == STRATEGIC_POLICY

    strategic = _row(outcome="loss", ticks=700, score=101)
    assert _select_teacher(geometric, strategic)[0] == STRATEGIC_POLICY


def test_frozen_hybrid_probe_contract_validates(tmp_path):
    from tools import build_alphazuma_55_hybrid_teacher_probe as builder

    value = builder.build(output_root=tmp_path / "probe", workers=3)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    prereg, source = _validate_preregistration(path)

    assert len(prereg["tasks"]) == 55
    assert prereg["seed_range"] == [1_400_600_000, 1_400_600_054]
    assert len(source["levels"]) == 55
    assert {row["teacher_policy_id"] for row in prereg["teacher_policy_map"]} == {
        GEOMETRIC_POLICY,
        STRATEGIC_POLICY,
    }
