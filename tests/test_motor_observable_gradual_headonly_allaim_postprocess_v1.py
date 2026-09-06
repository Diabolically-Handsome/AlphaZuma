from tools import (
    audit_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_prereg_v1
    as prereg_audit,
)
from tools import (
    build_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_plan_v1
    as plan_builder,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1
    as postprocess,
)


def test_factorial_postprocess_uses_distinct_engineering_seeds() -> None:
    assert postprocess.EXPECTED_SCREEN_SEED == 1_550_005_000
    assert postprocess.EXPECTED_GATE_SEED == 1_550_005_100


def test_factorial_postprocess_keeps_five_candidate_gate_cap() -> None:
    assert postprocess.MAXIMUM_GATE_CANDIDATES == 5


def test_factorial_postprocess_uses_distinct_audit_paths() -> None:
    assert plan_builder.PREREG_AUDIT_OUTPUT != plan_builder.FINAL_AUDIT_OUTPUT


def test_factorial_postprocess_campaign_is_bound_across_components() -> None:
    assert plan_builder.CAMPAIGN_ID == prereg_audit.EXPECTED_CAMPAIGN


def test_factorial_postprocess_plan_is_bound_to_launch_receipt() -> None:
    assert plan_builder.TRAINING_LAUNCH_RECEIPT.name.endswith(
        "s99081646-launch-receipt-v1.json"
    )
