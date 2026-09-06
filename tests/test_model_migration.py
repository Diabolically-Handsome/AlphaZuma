"""Pure layout tests for the fruit-channel actor migration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zuma_rl import model_migration


def _fake_environment() -> SimpleNamespace:
    base = tuple(f"base_{index}" for index in range(23)) + (
        *model_migration.FRUIT_GLOBAL_NAMES,
    )
    names = base + tuple(f"current_color_{index}" for index in range(4)) + tuple(
        f"next_color_{index}" for index in range(4)
    )
    return SimpleNamespace(
        _GLOBAL_BASE_FEATURES=base,
        global_feature_names=names,
        global_feature_size=len(names),
        num_colors=4,
        observation_layout={"globals": slice(10, 46)},
    )


def test_global_column_mapping_inserts_only_five_fruit_channels() -> None:
    old = SimpleNamespace(
        global_feature_size=31,
        global_current_color_start=23,
        global_next_color_start=27,
        num_colors=4,
    )
    mapping = model_migration._global_column_mapping(old, _fake_environment())

    assert mapping == {
        **{index: index for index in range(23)},
        **{23 + index: 28 + index for index in range(4)},
        **{27 + index: 32 + index for index in range(4)},
    }
    assert set(range(23, 28)).isdisjoint(mapping.values())


def test_old_observation_projection_removes_only_fruit_channels() -> None:
    env = _fake_environment()
    current = np.arange(46, dtype=np.float32).reshape(1, -1)

    projected = model_migration._old_observations(
        current,
        env=env,
        old_global_size=31,
    )

    assert projected.shape == (1, 41)
    assert projected[0, :33].tolist() == current[0, :33].tolist()
    assert projected[0, 33:].tolist() == current[0, 38:].tolist()
