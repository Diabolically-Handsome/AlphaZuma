from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2 as engineering,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_successor_v2 as successor,
)


def test_v2_successor_preserves_formal_matrix() -> None:
    assert successor.FINAL_SEED_BASE == 3_700_000_000
    assert successor.CONTINUOUS_SEED_BASE == 3_800_000_000
    assert successor.LEVEL_COUNT * successor.FINAL_ATTEMPTS == 440
    assert successor.LEVEL_COUNT * successor.CONTINUOUS_CAMPAIGNS == 220


def test_v2_successor_binds_v2_engineering() -> None:
    assert successor.engineering is engineering
    assert successor.EXPECTED_ENGINEERING_CAMPAIGN.endswith("s99081634-v2")
