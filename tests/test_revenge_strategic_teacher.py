from __future__ import annotations

from zuma_rl.revenge_strategic_teacher import _StrategicCandidate


def test_candidate_prefers_immediate_chain_powerup_clear():
    plain = _StrategicCandidate(0, 2, 0, 0, 0.0, 1.0, 1.0)
    chain = _StrategicCandidate(1, 2, 1, 4, 0.2, 2.0, 2.0)
    setup = _StrategicCandidate(2, 1, 3, 8, 1.0, 0.1, 0.1)

    assert chain.rank() < plain.rank() < setup.rank()
