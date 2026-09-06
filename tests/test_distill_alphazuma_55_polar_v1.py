from __future__ import annotations

import numpy as np

from tools.distill_alphazuma_55_polar_v1 import _evaluate_full55_dataset


class _ActionSpace:
    nvec = np.asarray((4, 180), dtype=np.int64)


def test_polar_dataset_evaluator_requires_full55_interface() -> None:
    class Model:
        action_space = _ActionSpace()

    Model.action_space.nvec = np.asarray((3, 180), dtype=np.int64)
    try:
        _evaluate_full55_dataset(model=Model(), dataset=[], batch_size=1)
    except ValueError as error:
        assert "full55" in str(error)
    else:
        raise AssertionError("non-full55 action interface was accepted")


def test_polar_dataset_evaluator_rejects_empty_non_fire_contract_late() -> None:
    # The contract check must occur before touching policy state. This keeps
    # corrupted manifests fail-closed even when no dataset was materialized.
    class Model:
        action_space = _ActionSpace()

    Model.action_space.nvec = np.asarray((2, 90), dtype=np.int64)
    try:
        _evaluate_full55_dataset(model=Model(), dataset=[], batch_size=1)
    except ValueError as error:
        assert "full55" in str(error)
    else:
        raise AssertionError("corrupted action interface was accepted")
