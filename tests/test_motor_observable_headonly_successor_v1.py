from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as engineering,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_successor_v1
    as successor,
)


def test_headonly_successor_uses_route_unique_formal_ranges() -> None:
    assert successor.FINAL_SEED_BASE == 3_900_000_000
    assert successor.CONTINUOUS_SEED_BASE == 4_000_000_000
    assert successor.LEVEL_COUNT * successor.FINAL_ATTEMPTS == 440
    assert successor.LEVEL_COUNT * successor.CONTINUOUS_CAMPAIGNS == 220


def test_headonly_successor_binds_frozen_backbone_engineering() -> None:
    assert successor.engineering is engineering
    assert successor.EXPECTED_ENGINEERING_CAMPAIGN.endswith("s99081643-v2")


def test_headonly_patch_restores_legacy_globals() -> None:
    original_final = successor.legacy.FINAL_SEED_BASE
    original_continuous = successor.legacy.CONTINUOUS_SEED_BASE
    with successor._patched_legacy():
        assert successor.legacy.FINAL_SEED_BASE == successor.FINAL_SEED_BASE
        assert (
            successor.legacy.CONTINUOUS_SEED_BASE
            == successor.CONTINUOUS_SEED_BASE
        )
    assert successor.legacy.FINAL_SEED_BASE == original_final
    assert successor.legacy.CONTINUOUS_SEED_BASE == original_continuous
