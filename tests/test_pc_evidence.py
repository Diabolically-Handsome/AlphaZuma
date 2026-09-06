from __future__ import annotations

import pytest

from zuma_rl.pc_evidence import PcTickMap, TickPts
from zuma_rl.pc_golden import PcGoldenValidationError


def test_tick_map_round_trip_is_canonical_and_strictly_increasing() -> None:
    tick_map = PcTickMap(
        records=(
            TickPts(0, -10),
            TickPts(1, 0),
            TickPts(2, 15),
        )
    )

    text = "tick,pts\n0,-10\n1,0\n2,15\n"
    assert tick_map.to_csv() == text
    assert PcTickMap.from_csv(text) == tick_map


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("tick,pts\r\n0,0\r\n", "canonical"),
        ("tick,pts\n0,+1\n", "canonical"),
        ("tick,pts\n00,1\n", "canonical"),
        ("tick,pts\n0,-0\n", "canonical"),
        ("tick,pts\n0,0\n2,1\n", "contiguous"),
        ("tick,pts\n0,0\n1,0\n", "strictly increasing"),
        ("pts,tick\n0,0\n", "exact header"),
    ],
)
def test_tick_map_rejects_noncanonical_or_ambiguous_rows(
    text: str,
    message: str,
) -> None:
    with pytest.raises(PcGoldenValidationError, match=message):
        PcTickMap.from_csv(text)


def test_tick_map_wraps_runtime_integer_digit_limits() -> None:
    with pytest.raises(PcGoldenValidationError, match="parser limit"):
        PcTickMap.from_csv("tick,pts\n0," + "9" * 5000 + "\n")
