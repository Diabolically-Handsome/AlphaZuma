from tools import (
    build_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_master_v1
    as master_builder,
)
from tools import (
    build_alphazuma_55_formal_seed_registry_v10 as registry_builder,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_v1
    as controller,
)


def test_factorial_successor_uses_route_unique_uint32_seed_ranges() -> None:
    assert controller.FINAL_SEED_BASE == 4_100_000_000
    assert controller.CONTINUOUS_SEED_BASE == 4_200_000_000
    assert controller.FINAL_SEED_BASE + 439 <= 0xFFFFFFFF
    assert controller.CONTINUOUS_SEED_BASE + 219 <= 0xFFFFFFFF


def test_factorial_successor_campaigns_are_bound() -> None:
    assert controller.EXPECTED_CAMPAIGN == master_builder.CAMPAIGN_ID
    assert registry_builder.CAMPAIGN_ID in controller.EXPECTED_CAMPAIGN


def test_factorial_successor_deadline_is_sunday_four_pm() -> None:
    source = master_builder.SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"2026-08-16T20:00:00Z"' in source


def test_factorial_successor_registry_supersedes_v9() -> None:
    assert registry_builder.PREDECESSOR.name.endswith("s99081645-v9.json")
