from __future__ import annotations

import pytest


def test_full55_column_mapping_preserves_named_legacy_features() -> None:
    pytest.importorskip("torch")
    from zuma_rl.full55_migration import Full55MigrationError, _column_mapping

    assert _column_mapping(
        ("present", "color_0", "color_1"),
        ("present", "color_0", "color_1", "color_2", "color_3"),
    ) == {0: 0, 1: 1, 2: 2}
    with pytest.raises(Full55MigrationError, match="lost columns"):
        _column_mapping(("present", "color_0"), ("present",))


def test_full55_linear_expansion_zeros_new_columns_and_keeps_relations() -> None:
    torch = pytest.importorskip("torch")
    from zuma_rl.full55_migration import _expand_linear_columns

    source = torch.asarray(
        [[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]]
    )
    target = torch.full((2, 6), 99.0)
    expanded = _expand_linear_columns(
        source,
        target,
        {0: 0, 1: 2, 2: 3},
        source_raw_width=3,
        target_raw_width=5,
    )
    torch.testing.assert_close(
        expanded,
        torch.asarray(
            [[1.0, 0.0, 2.0, 3.0, 0.0, 10.0], [4.0, 0.0, 5.0, 6.0, 0.0, 20.0]]
        ),
    )
