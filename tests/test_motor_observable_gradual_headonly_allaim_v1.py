from tools import (
    distill_alphazuma_55_motor_observable_gradual_headonly_allaim_v1
    as combined,
)
from tools import (
    distill_alphazuma_55_motor_observable_gradual_headonly_v1 as headonly,
)


def test_combined_route_uses_fresh_engineering_seed_subrange() -> None:
    assert combined.EXPECTED_SEED_BASE == 1_546_001_000
    assert combined.EXPECTED_SEED_BASE > 1_546_000_219


def test_combined_route_reuses_exact_head_only_parameter_scope() -> None:
    assert combined.TRAINABLE_PARAMETER_NAMES == (
        "action_net.weight",
        "action_net.bias",
    )
    assert combined.TRAINABLE_PARAMETER_NAMES == headonly.TRAINABLE_PARAMETER_NAMES


def test_combined_route_has_distinct_factorial_identity() -> None:
    assert combined.EXPECTED_VARIANT == "action_head_only_all_actions_aim_v1"
