"""Recovered settled-V3 clone with only the inherited aim head reset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55_settled_v3 as legacy
from tools.distill_alphazuma_55_settled_v3_reset_aim import _reset_aim_head


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
    recovery = prereg.get("startup_recovery", {})
    if (
        recovery.get("failed_before_run_initialization") is not True
        or recovery.get("failed_seed_consumption") is not False
        or recovery.get("reuse_failed_training_seed_range") is not False
    ):
        raise ValueError("unexpected startup-recovery contract")
    return prereg


def _install_reset_loader(model_class: Any) -> tuple[list[dict[str, Any]], Callable[[], None]]:
    """Install a temporary loader and correctly restore inherited descriptors."""

    had_local_descriptor = "load" in model_class.__dict__
    local_descriptor = model_class.__dict__.get("load")
    original_load = model_class.load
    receipts: list[dict[str, Any]] = []

    def load_and_reset(cls: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        del cls
        model = original_load(path, *args, **kwargs)
        receipts.append(_reset_aim_head(model))
        return model

    model_class.load = classmethod(load_and_reset)

    def restore() -> None:
        if had_local_descriptor:
            model_class.load = local_descriptor
        else:
            delattr(model_class, "load")

    return receipts, restore


def run_distillation(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    original_root: Path,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    reset_receipts, restore_loader = _install_reset_loader(MaskablePPO)
    legacy.SCRIPT_PATH = SCRIPT_PATH
    try:
        result = legacy.run_distillation(
            prereg_path=prereg_path,
            prereg=prereg,
            original_root=original_root,
        )
    finally:
        restore_loader()
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
                    "startup_recovery": prereg["startup_recovery"],
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
