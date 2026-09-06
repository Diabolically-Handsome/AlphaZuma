"""Settled-V3 clone with only the inherited 180-bin aim head reset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55_settled_v3 as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    prereg = legacy._validate_preregistration(path)
    reset = prereg.get("run", {}).get("aim_head_reset", {})
    if reset != {
        "enabled": True,
        "rows": "aim_component_only",
        "weight": "zero",
        "bias": "zero",
        "verb_head_unchanged": True,
        "feature_extractor_unchanged_before_training": True,
    }:
        raise ValueError("unexpected aim-head reset contract")
    return prereg


def _reset_aim_head(model: Any) -> dict[str, Any]:
    import torch

    nvec = tuple(int(value) for value in model.action_space.nvec)
    if nvec != (4, 180):
        raise ValueError("aim reset requires the full55-v1 action interface")
    layer = model.policy.action_net
    if tuple(int(value) for value in layer.weight.shape[:1]) != (184,):
        raise ValueError("unexpected action-head row count")
    with torch.no_grad():
        weight_before = float(torch.linalg.vector_norm(layer.weight[4:]).cpu())
        bias_before = float(torch.linalg.vector_norm(layer.bias[4:]).cpu())
        verb_weight_hash = torch.sum(layer.weight[:4].detach().cpu()).item()
        verb_bias_hash = torch.sum(layer.bias[:4].detach().cpu()).item()
        layer.weight[4:].zero_()
        layer.bias[4:].zero_()
        weight_after = float(torch.linalg.vector_norm(layer.weight[4:]).cpu())
        bias_after = float(torch.linalg.vector_norm(layer.bias[4:]).cpu())
        if not math.isclose(
            verb_weight_hash,
            torch.sum(layer.weight[:4].detach().cpu()).item(),
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            verb_bias_hash,
            torch.sum(layer.bias[:4].detach().cpu()).item(),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("aim reset altered the verb head")
    if weight_after != 0.0 or bias_after != 0.0:
        raise RuntimeError("aim-head zero initialization failed")
    return {
        "status": "APPLIED_BEFORE_OPTIMIZER_REPLACEMENT_AND_COLLECTION",
        "aim_rows": [4, 184],
        "aim_weight_l2_before": weight_before,
        "aim_bias_l2_before": bias_before,
        "aim_weight_l2_after": weight_after,
        "aim_bias_l2_after": bias_after,
        "verb_head_unchanged": True,
    }


def run_distillation(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    original_root: Path,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    original_descriptor = MaskablePPO.__dict__["load"]
    original_load = MaskablePPO.load
    reset_receipts: list[dict[str, Any]] = []

    def load_and_reset(cls: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        del cls
        model = original_load(path, *args, **kwargs)
        reset_receipts.append(_reset_aim_head(model))
        return model

    MaskablePPO.load = classmethod(load_and_reset)
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy.__file__ = str(SCRIPT_PATH)
    try:
        result = legacy.run_distillation(
            prereg_path=prereg_path,
            prereg=prereg,
            original_root=original_root,
        )
    finally:
        MaskablePPO.load = original_descriptor
    if len(reset_receipts) != 1:
        raise RuntimeError("aim reset was not applied exactly once")
    result["aim_head_reset"] = reset_receipts[0]
    run_dir = Path(str(prereg["run"]["run_dir"])).resolve()
    legacy.legacy._write_json_atomic(run_dir / "completion.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": str(prereg_path),
                    "sha256": legacy.legacy._sha256(prereg_path),
                    "run": prereg["run"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = run_distillation(
        prereg_path=prereg_path,
        prereg=prereg,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "teacher_wins": result["collection"]["wins"],
                "aim_head_reset": result["aim_head_reset"],
                "final_model": result["final_model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
