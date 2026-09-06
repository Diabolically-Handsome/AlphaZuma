from __future__ import annotations

import pytest

from zuma_rl.windows_display_mode import (
    DisplayMode,
    DisplayModeError,
    select_exact_display_mode,
)


def _mode(
    width: int,
    height: int,
    refresh_rate_hz: int,
    *,
    bits_per_pixel: int = 32,
    display_flags: int = 0,
) -> DisplayMode:
    return DisplayMode(
        bits_per_pixel=bits_per_pixel,
        width=width,
        height=height,
        display_flags=display_flags,
        refresh_rate_hz=refresh_rate_hz,
    )


def test_select_exact_display_mode_prefers_current_format() -> None:
    current = _mode(3840, 2160, 165, display_flags=7)
    selected = select_exact_display_mode(
        (
            _mode(3840, 2160, 100, bits_per_pixel=24),
            _mode(3840, 2160, 100, display_flags=7),
            _mode(800, 600, 100, display_flags=7),
        ),
        width=3840,
        height=2160,
        refresh_rate_hz=100,
        current=current,
    )
    assert selected == _mode(3840, 2160, 100, display_flags=7)
    assert selected.to_dict() == {
        "bits_per_pixel": 32,
        "width": 3840,
        "height": 2160,
        "display_flags": 7,
        "refresh_rate_hz": 100,
    }


def test_select_exact_display_mode_rejects_nearest_rate() -> None:
    with pytest.raises(
        DisplayModeError,
        match="requested_display_mode_unavailable",
    ):
        select_exact_display_mode(
            (_mode(3840, 2160, 99), _mode(3840, 2160, 101)),
            width=3840,
            height=2160,
            refresh_rate_hz=100,
        )


@pytest.mark.parametrize(
    ("width", "height", "rate"),
    ((0, 600, 100), (800, 0, 100), (800, 600, 1)),
)
def test_select_exact_display_mode_rejects_invalid_request(
    width: int,
    height: int,
    rate: int,
) -> None:
    with pytest.raises(ValueError, match="requested display mode is invalid"):
        select_exact_display_mode(
            (_mode(800, 600, 100),),
            width=width,
            height=height,
            refresh_rate_hz=rate,
        )
