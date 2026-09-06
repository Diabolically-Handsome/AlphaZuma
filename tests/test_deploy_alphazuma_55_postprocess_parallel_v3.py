from __future__ import annotations

from tools import deploy_alphazuma_55_postprocess_parallel_v2 as legacy
from tools import deploy_alphazuma_55_postprocess_parallel_v3 as subject


def test_main_rebinds_script_identity(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        captured["script_path"] = legacy.SCRIPT_PATH
        return 17

    monkeypatch.setattr(legacy, "main", fake_main)
    result = subject.main(["--validate-only"])

    assert result == 17
    assert captured == {
        "argv": ["--validate-only"],
        "script_path": subject.SCRIPT_PATH,
    }
