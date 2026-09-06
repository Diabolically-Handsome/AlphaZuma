"""Create frozen inference variants from a polar-basis AlphaZuma policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _head_layout(model: Any) -> tuple[int, int, int]:
    import torch

    action_net = model.policy.action_net
    if not isinstance(action_net, torch.nn.Linear):
        raise TypeError("polar re-anchor requires a linear action head")
    action_sizes = tuple(int(value) for value in model.action_space.nvec)
    if len(action_sizes) != 2:
        raise ValueError("polar re-anchor requires a factorized action space")
    verb_bins, aim_bins = action_sizes
    extractor = model.policy.features_extractor
    learned = int(getattr(extractor, "learned_features_dim", 0))
    polar = int(getattr(extractor, "polar_aim_bins", 0))
    if polar != aim_bins or action_net.in_features != learned + aim_bins:
        raise ValueError("model does not expose the expected polar basis")
    return verb_bins, aim_bins, learned


def apply_transform(model: Any, *, mode: str, scale: float) -> dict[str, Any]:
    """Apply one isolated aim-row transform and return invariant evidence."""

    import torch

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    verb_bins, aim_bins, learned = _head_layout(model)
    action_net = model.policy.action_net
    before_weight = action_net.weight.detach().clone()
    before_bias = action_net.bias.detach().clone()
    with torch.no_grad():
        if mode == "reset_identity":
            action_net.weight[verb_bins:, :].zero_()
            action_net.bias[verb_bins:].zero_()
            indices = torch.arange(aim_bins, device=action_net.weight.device)
            action_net.weight[
                verb_bins + indices, learned + indices
            ] = float(scale)
        elif mode == "additive_identity":
            indices = torch.arange(aim_bins, device=action_net.weight.device)
            action_net.weight[
                verb_bins + indices, learned + indices
            ] += float(scale)
        else:
            raise ValueError(f"unsupported polar re-anchor mode: {mode!r}")

    after_weight = action_net.weight.detach()
    after_bias = action_net.bias.detach()
    if not torch.equal(after_weight[:verb_bins], before_weight[:verb_bins]):
        raise RuntimeError("verb action rows changed during aim-only transform")
    if not torch.equal(after_bias[:verb_bins], before_bias[:verb_bins]):
        raise RuntimeError("verb action biases changed during aim-only transform")
    return {
        "mode": mode,
        "scale": float(scale),
        "verb_rows_bitwise_unchanged": True,
        "verb_biases_bitwise_unchanged": True,
        "aim_weight_l2_before": float(
            torch.linalg.vector_norm(before_weight[verb_bins:]).cpu()
        ),
        "aim_weight_l2_after": float(
            torch.linalg.vector_norm(after_weight[verb_bins:]).cpu()
        ),
        "aim_bias_l2_before": float(
            torch.linalg.vector_norm(before_bias[verb_bins:]).cpu()
        ),
        "aim_bias_l2_after": float(
            torch.linalg.vector_norm(after_bias[verb_bins:]).cpu()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    if _sha256(plan_path) != args.expected_plan_sha256:
        raise ValueError("plan hash differs from the expected frozen hash")
    plan = _read_json(plan_path)
    if (
        plan.get("schema") != "zuma-rl.alphazuma-55-polar-reanchor-probe-plan"
        or plan.get("version") != 1
        or plan.get("status") != "FROZEN_BEFORE_VARIANT_CREATION"
    ):
        raise ValueError("unexpected polar re-anchor plan")
    runner = plan["implementation"]["runner"]
    if Path(str(runner["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("plan binds another variant runner")
    if runner["sha256"] != _sha256(Path(__file__).resolve()):
        raise ValueError("variant runner bytes differ from the plan")

    source = plan["source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != _sha256(source_path):
        raise ValueError("source model hash differs from the plan")
    output_root = Path(str(plan["variant_output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse variant output: {output_root}")
    output_root.mkdir(parents=True)
    completion_path = output_root / "completion.json"
    failure_path = output_root / "failure.json"

    try:
        import torch
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(source_path, device="cpu")
        _head_layout(model)
        source_weight = model.policy.action_net.weight.detach().clone()
        source_bias = model.policy.action_net.bias.detach().clone()
        source_dynamic_mix = getattr(
            model.policy.features_extractor, "dynamic_shooter_mix", None
        )
        source_dynamic_mix_value = (
            float(source_dynamic_mix.detach().cpu())
            if source_dynamic_mix is not None
            else None
        )
        variants: list[dict[str, Any]] = []
        for spec in plan["variants"]:
            with torch.no_grad():
                model.policy.action_net.weight.copy_(source_weight)
                model.policy.action_net.bias.copy_(source_bias)
            evidence = apply_transform(
                model,
                mode=str(spec["mode"]),
                scale=float(spec["scale"]),
            )
            dynamic_mix = getattr(
                model.policy.features_extractor, "dynamic_shooter_mix", None
            )
            dynamic_mix_value = (
                float(dynamic_mix.detach().cpu()) if dynamic_mix is not None else None
            )
            if dynamic_mix_value != source_dynamic_mix_value:
                raise RuntimeError("dynamic shooter parameter changed during transform")
            model_path = output_root / f"{spec['id']}.zip"
            model.save(model_path)
            if not model_path.is_file():
                raise RuntimeError(f"variant save did not create {model_path}")
            variants.append(
                {
                    "id": str(spec["id"]),
                    "path": str(model_path),
                    "sha256": _sha256(model_path),
                    "bytes": model_path.stat().st_size,
                    "training_steps": int(source["training_steps"]),
                    "transform": evidence,
                    "dynamic_shooter_parameter_unchanged": True,
                }
            )
        completion = {
            "schema": "zuma-rl.alphazuma-55-polar-reanchor-variants-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "source_model": source,
            "variants": variants,
            "formal_seed_consumption": False,
            "environment_steps_consumed": 0,
        }
        _write_exclusive(completion_path, completion)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "completion": str(completion_path),
                    "completion_sha256": _sha256(completion_path),
                    "variants": [
                        {"id": row["id"], "sha256": row["sha256"]}
                        for row in variants
                    ],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        if not failure_path.exists():
            _write_exclusive(
                failure_path,
                {
                    "schema": "zuma-rl.alphazuma-55-polar-reanchor-variants-failure",
                    "version": 1,
                    "status": "ERROR",
                    "failed_utc": _utc_now(),
                    "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "formal_seed_consumption": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
