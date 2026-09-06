from __future__ import annotations

from tools.evaluate_alphazuma_55_geometric_teacher import _validate_preregistration


def test_frozen_probe_contract_validates(tmp_path):
    from tools import build_alphazuma_55_geometric_teacher_probe as builder

    value = builder.build(output_root=tmp_path / "probe", workers=3)
    path = tmp_path / "probe.json"
    path.write_text(__import__("json").dumps(value), encoding="utf-8")

    prereg, source = _validate_preregistration(path)

    assert len(prereg["tasks"]) == 55
    assert prereg["seed_range"] == [1_400_500_000, 1_400_500_054]
    assert len(source["levels"]) == 55
