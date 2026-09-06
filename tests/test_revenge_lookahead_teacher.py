from __future__ import annotations

from zuma_rl.revenge_lookahead_teacher import LookaheadOutcome


def _outcome(**overrides):
    values = {
        "aim_bin": 10,
        "fire_executed": True,
        "outcome": None,
        "score_gain": 0,
        "ball_reduction": 0,
        "danger_gain": 0.0,
        "simulated_ticks": 100,
        "static_rank": (1.0,),
    }
    values.update(overrides)
    return LookaheadOutcome(**values)


def test_lookahead_ranking_prefers_win_then_scoring_clear():
    idle = _outcome()
    clear = _outcome(score_gain=300, ball_reduction=3, danger_gain=2.0)
    win = _outcome(outcome="win", score_gain=100)

    assert win.rank() < clear.rank() < idle.rank()


def test_lookahead_ranking_prefers_skull_relief_over_points():
    points = _outcome(score_gain=500, ball_reduction=1, danger_gain=-1.0)
    relief = _outcome(score_gain=100, ball_reduction=1, danger_gain=2.0)

    assert relief.rank() < points.rank()


def test_lookahead_ranking_rejects_unexecuted_or_losing_fire():
    idle = _outcome()
    unexecuted = _outcome(fire_executed=False, score_gain=999)
    loss = _outcome(outcome="loss", score_gain=100)

    assert idle.rank() < loss.rank() < unexecuted.rank()
