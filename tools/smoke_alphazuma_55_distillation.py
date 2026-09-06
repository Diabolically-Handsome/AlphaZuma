"""Bounded one-level smoke test for the AlphaZuma 55 distillation loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.distill_alphazuma_55 import _collect_round, _optimize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    from sb3_contrib import MaskablePPO

    args = build_parser().parse_args(argv)
    model_path = args.model.expanduser().resolve(strict=True)
    root = args.original_root.expanduser().resolve(strict=True)
    teacher = MaskablePPO.load(model_path, device=args.device)
    student = MaskablePPO.load(model_path, device=args.device)
    row = {
        "level_id": "Jungle1",
        "engineering_wins": 0,
        "teacher_model": {
            "id": "smoke-teacher",
            "path": str(model_path),
        },
    }
    dataset, collection = _collect_round(
        original_root=root,
        teacher_rows=[row],
        teacher_models={"smoke-teacher": teacher},
        student=student,
        device=args.device,
        round_index=0,
        seed_base=1_400_300_000,
        parallel_envs=1,
        max_ticks=200,
        samples_per_level=8,
        sample_stride=8,
        teacher_execution_probability=1.0,
        model_seed=8_508_150,
    )
    if len(dataset) != 8 or collection["retained_samples"] != 8:
        raise RuntimeError("bounded collection did not retain eight samples")
    optimization = _optimize(
        model=student,
        dataset=dataset,
        epochs=1,
        batch_size=8,
        max_grad_norm=0.5,
        verb_loss_weight=2.0,
        model_seed=8_508_150,
        round_index=0,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "collection": collection,
                "optimization": optimization,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
