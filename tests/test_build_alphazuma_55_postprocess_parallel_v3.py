from __future__ import annotations

from tools import build_alphazuma_55_postprocess_parallel_v2 as legacy
from tools import build_alphazuma_55_postprocess_parallel_v3 as builder_v3


def test_v3_preserves_v2_contract_and_rebinds_capacity_safe_tools(monkeypatch) -> None:
    base = {
        "implementation": {
            "controller": {"path": "old-controller", "sha256": "old"},
            "evaluator": {"path": "old-evaluator", "sha256": "old"},
        },
        "training_routes": [1, 2],
        "migrations": [1, 2, 3],
    }
    monkeypatch.setattr(legacy, "build", lambda **kwargs: base | {"kwargs": kwargs})
    monkeypatch.setattr(
        legacy,
        "_artifact",
        lambda path: {"path": str(path), "sha256": "sha256:test"},
    )
    value = builder_v3.build(marker=123)
    assert value["kwargs"] == {"marker": 123}
    assert value["implementation"]["controller"]["path"] == str(
        builder_v3.CONTROLLER
    )
    assert value["implementation"]["evaluator"]["path"] == str(
        builder_v3.EVALUATOR
    )
    contract = value["capacity_fail_closed_contract"]
    assert contract["actor_observation_capacity_balls"] == 768
    assert contract["observation_shape_unchanged"] is True
    assert contract["silent_state_truncation_forbidden"] is True
    assert contract["exact_capacity_overflow_outcome"] == "loss"
    assert contract["all_other_exceptions_fail_the_shard"] is True
