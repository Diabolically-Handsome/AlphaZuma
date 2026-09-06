from __future__ import annotations

from tools import run_alphazuma_55_postprocess_parallel_v2 as legacy
from tools import run_alphazuma_55_postprocess_parallel_v3 as controller_v3


def test_v3_rebinds_only_controller_and_evaluator(monkeypatch) -> None:
    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["script"] = legacy.SCRIPT_PATH
        observed["evaluator"] = legacy.EVALUATOR
        observed["helper_evaluator"] = (
            legacy._evaluate_and_summarize.__globals__["EVALUATOR"]
        )
        return 11

    monkeypatch.setattr(legacy, "main", fake_main)
    assert controller_v3.main(["--validate-only"]) == 11
    assert observed == {
        "argv": ["--validate-only"],
        "script": controller_v3.SCRIPT_PATH,
        "evaluator": controller_v3.EVALUATOR,
        "helper_evaluator": controller_v3.EVALUATOR,
    }
