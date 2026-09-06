"""Transplant a stacked per-tick BC student onto the macro interface.

The best BC init (park-settle distill v3, frame-stacked, MaskablePPO
per-tick ``MultiDiscrete([4, 180])``) cannot be loaded by a macro-
interface PPO run: the park-settle wrapper's action space is
``MultiDiscrete([3, 180])`` (wait_hold/fire/swap; hop dropped).  This
tool builds a FRESH MaskablePPO on the macro interface and transplants
the per-tick student's decision knowledge into it:

* feature extractor (StackedRevengeEntityFeatureExtractor, imported
  from ``tools.distill_alphazuma_55_park_settle_v3`` -- the module the
  pickled model resolves it from): weights VERBATIM;
* actor pathway (``mlp_extractor.policy_net``, empty under the v1
  ``net_arch {"pi": [], "vf": [32]}`` but copied generically so the
  transplanted heads always see identical latents): VERBATIM;
* aim head (``action_net`` rows ``[verb_count:]``): VERBATIM;
* macro verb head: initialized from the per-tick verb head rows
  ``[wait, fire, swap]`` in order; the hop row (per-tick verb 3) is
  DROPPED -- the macro interface does not expose hop;
* value pathway (``mlp_extractor.value_net`` + ``value_net``): FRESH --
  the BC student was never trained with a critic and the macro reward
  is per-macro summed per-tick reward, so its value scale is new.
  Fresh means the target model's own initialization: SB3's orthogonal
  init (gain sqrt(2) on hidden layers, 1.0 on the value output) seeded
  by ``--model-seed``.

The output zip is loadable by plain ``MaskablePPO.load`` with only the
repository on ``sys.path`` (its pickled ``policy_kwargs`` reference the
extractor class by module path, exactly like the source model).  The
stored optimisation fields (lr/n_steps/gamma/...) are inert: the v2/v2b
trainer constructs a fresh algorithm from its own run receipt and only
copies ``policy_kwargs`` plus the policy state dict bitwise.

A transplant receipt (``transplant.json``) records source/output
sha256, the layer-by-layer transplant map, the parameter-count check
(target = source minus exactly one verb row), a logits-equivalence
check (macro verb/aim logits bitwise-equal the per-tick logits on random
observations), and a masked-predict smoke.

Frozen modules are imported, never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55_park_settle_v1 as v1
# Imported for the side effect of making the pickled feature-extractor
# class importable before MaskablePPO.load runs, and to bind its bytes
# into the receipt.
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from tools import probe_alphazuma_55_recovery_intervention_v1 as base


SCRIPT_PATH = Path(__file__).resolve()
RECEIPT_SCHEMA = "zuma-rl.alphazuma-55-macro-policy-bootstrap-v1"
OUTPUT_MODEL_NAME = "macro_model.zip"
RECEIPT_NAME = "transplant.json"

PER_TICK_VERB_COUNT = 4  # full55-v1: wait / fire / swap / hop
MACRO_VERB_COUNT = 3  # park-settle-v1: wait_hold / fire / swap
# Macro verb row i is initialized from per-tick verb row KEPT_VERB_ROWS[i].
KEPT_VERB_ROWS = (0, 1, 2)
DROPPED_VERB_ROW = 3  # hop: no macro exposes it
VERB_ROW_MAP = {
    "wait_hold": "per-tick wait (row 0)",
    "fire": "per-tick fire (row 1)",
    "swap": "per-tick swap (row 2)",
    "dropped": "per-tick hop (row 3)",
}
VALUE_HEAD_INIT = (
    "fresh: the target model's own SB3 orthogonal initialization "
    "(hidden gain sqrt(2), value output gain 1.0) seeded by model_seed; "
    "never copied from the source"
)
# Stored optimisation fields of the saved zip.  Inert for the v2/v2b
# trainer (it rebuilds the algorithm from its own receipt), set to the
# documented warm-start values so a bare load is still sensible.
STORED_HYPERPARAMETERS = {
    "learning_rate": 3e-5,
    "n_steps": 64,
    "batch_size": 64,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "ent_coef": 2e-3,
}

_ACTOR_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
    "mlp_extractor.policy_net",
)
_VALUE_PREFIXES = ("mlp_extractor.value_net", "value_net.")


def transplant_state_dict(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    *,
    aim_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the target policy state dict plus its transplant map.

    Every target key must classify as actor-verbatim, action-head
    row-remap, or value-fresh; an unrecognized key raises so an SB3
    layout change can never silently produce a half-transplant.
    """

    import torch

    new_state: dict[str, Any] = {}
    transplant_map: list[dict[str, Any]] = []
    for key, target_tensor in target_state.items():
        if key.startswith(_ACTOR_PREFIXES):
            source_tensor = source_state[key]
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                raise ValueError(
                    f"actor layer shape changed across interfaces: {key} "
                    f"{tuple(source_tensor.shape)} vs "
                    f"{tuple(target_tensor.shape)}"
                )
            new_state[key] = source_tensor.clone()
            entry = {
                "target": key,
                "source": key,
                "mode": "verbatim",
            }
        elif key in ("action_net.weight", "action_net.bias"):
            source_tensor = source_state[key]
            expected_source = PER_TICK_VERB_COUNT + aim_bins
            expected_target = MACRO_VERB_COUNT + aim_bins
            if int(source_tensor.shape[0]) != expected_source:
                raise ValueError(
                    f"per-tick action head has {source_tensor.shape[0]} "
                    f"rows, expected {expected_source}"
                )
            if int(target_tensor.shape[0]) != expected_target:
                raise ValueError(
                    f"macro action head has {target_tensor.shape[0]} "
                    f"rows, expected {expected_target}"
                )
            kept = torch.as_tensor(
                np.asarray(KEPT_VERB_ROWS), dtype=torch.long
            )
            new_state[key] = torch.cat(
                (
                    source_tensor.index_select(0, kept).clone(),
                    source_tensor[PER_TICK_VERB_COUNT:].clone(),
                ),
                dim=0,
            )
            entry = {
                "target": key,
                "source": key,
                "mode": "verb_rows_0_1_2_then_aim_rows_verbatim",
                "dropped_source_row": DROPPED_VERB_ROW,
            }
        elif key.startswith(_VALUE_PREFIXES):
            new_state[key] = target_tensor.clone()
            entry = {
                "target": key,
                "source": None,
                "mode": "fresh_target_init",
            }
        else:
            raise ValueError(
                f"unrecognized policy parameter cannot be transplanted "
                f"safely: {key}"
            )
        entry["shape"] = list(target_tensor.shape)
        entry["elements"] = int(target_tensor.numel())
        transplant_map.append(entry)
    return new_state, transplant_map


def _verify_logits_equivalence(
    source: Any, target: Any, *, aim_bins: int, seed: int
) -> dict[str, Any]:
    """Macro logits must equal the per-tick logits minus the hop row."""

    import torch

    source.policy.set_training_mode(False)
    target.policy.set_training_mode(False)
    rng = np.random.default_rng(seed)
    observations = rng.uniform(
        -1.0, 1.0, size=(4, int(source.observation_space.shape[0]))
    ).astype(np.float32)
    with torch.no_grad():
        source_tensor, _ = source.policy.obs_to_tensor(observations)
        target_tensor, _ = target.policy.obs_to_tensor(observations)
        source_verbs, source_aims = v1._policy_logits(source, source_tensor)
        target_verbs, target_aims = v1._policy_logits(target, target_tensor)
    kept = torch.as_tensor(np.asarray(KEPT_VERB_ROWS), dtype=torch.long)
    verbs_equal = bool(
        torch.equal(source_verbs.index_select(1, kept), target_verbs)
    )
    aims_equal = bool(torch.equal(source_aims, target_aims))
    if not (verbs_equal and aims_equal):
        raise RuntimeError(
            "transplanted logits diverge from the per-tick student: "
            f"verbs_equal={verbs_equal} aims_equal={aims_equal}"
        )
    return {
        "observations": int(observations.shape[0]),
        "observation_rng_seed": int(seed),
        "verb_logits_bitwise_equal_rows_0_1_2": verbs_equal,
        "aim_logits_bitwise_equal": aims_equal,
    }


def _masked_predict_smoke(model: Any, *, aim_bins: int) -> dict[str, Any]:
    """Deterministic masked predicts must respect a restrictive mask."""

    observation = np.zeros(
        (1, int(model.observation_space.shape[0])), dtype=np.float32
    )
    results: dict[str, Any] = {}
    for verb in range(MACRO_VERB_COUNT):
        mask = np.zeros(MACRO_VERB_COUNT + aim_bins, dtype=np.bool_)
        mask[verb] = True
        mask[MACRO_VERB_COUNT:] = True
        action, _ = model.predict(
            observation, deterministic=True, action_masks=mask
        )
        chosen = int(np.asarray(action).reshape(-1)[0])
        if chosen != verb:
            raise RuntimeError(
                f"masked predict escaped the mask: wanted verb {verb}, "
                f"got {chosen}"
            )
        results[f"only_verb_{verb}_legal_predicts_{verb}"] = True
    return results


def build_macro_model_from_per_tick(
    source: Any, *, model_seed: int, device: str = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Return (macro MaskablePPO, transplant evidence) for one source."""

    import torch
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO

    nvec = tuple(int(value) for value in source.action_space.nvec)
    if len(nvec) != 2 or nvec[0] != PER_TICK_VERB_COUNT:
        raise ValueError(
            f"source must use the per-tick MultiDiscrete([4, aim_bins]) "
            f"interface, got {nvec}"
        )
    aim_bins = nvec[1]
    macro_space = spaces.MultiDiscrete(
        np.array((MACRO_VERB_COUNT, aim_bins), dtype=np.int64)
    )
    environment = v1._SpaceOnlyMaskableEnv(
        source.observation_space, macro_space
    )
    target = MaskablePPO(
        "MlpPolicy",
        environment,
        policy_kwargs=copy.deepcopy(source.policy_kwargs),
        seed=int(model_seed),
        device=str(device),
        verbose=0,
        **STORED_HYPERPARAMETERS,
    )
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    fresh_value_state = {
        key: value.clone()
        for key, value in target_state.items()
        if key.startswith(_VALUE_PREFIXES)
    }
    new_state, transplant_map = transplant_state_dict(
        source_state, target_state, aim_bins=aim_bins
    )
    target.policy.load_state_dict(new_state, strict=True)

    # Post-load verification: every claim in the map is re-checked
    # against the LIVE policy, not the intermediate dict.
    live = target.policy.state_dict()
    for entry in transplant_map:
        key = entry["target"]
        if entry["mode"] == "verbatim":
            if not torch.equal(
                live[key].cpu(), source_state[key].cpu()
            ):
                raise RuntimeError(f"verbatim transplant drifted: {key}")
        elif entry["mode"].startswith("verb_rows"):
            kept = torch.as_tensor(
                np.asarray(KEPT_VERB_ROWS), dtype=torch.long
            )
            expected = torch.cat(
                (
                    source_state[key].index_select(0, kept),
                    source_state[key][PER_TICK_VERB_COUNT:],
                ),
                dim=0,
            )
            if not torch.equal(live[key].cpu(), expected.cpu()):
                raise RuntimeError(f"row remap drifted: {key}")
        else:
            if not torch.equal(
                live[key].cpu(), fresh_value_state[key].cpu()
            ):
                raise RuntimeError(f"fresh value init drifted: {key}")

    source_parameters = int(
        sum(parameter.numel() for parameter in source.policy.parameters())
    )
    target_parameters = int(
        sum(parameter.numel() for parameter in target.policy.parameters())
    )
    features_in = int(source.policy.action_net.in_features)
    expected_difference = features_in + 1  # one dropped verb row + bias
    if source_parameters - target_parameters != expected_difference:
        raise RuntimeError(
            "parameter-count check failed: source "
            f"{source_parameters} minus target {target_parameters} is "
            f"{source_parameters - target_parameters}, expected "
            f"{expected_difference} (one hop row of width {features_in} "
            "plus its bias)"
        )
    evidence = {
        "aim_bins": int(aim_bins),
        "action_nvec": {
            "source": list(nvec),
            "target": [MACRO_VERB_COUNT, int(aim_bins)],
        },
        "observation_width": int(source.observation_space.shape[0]),
        "verb_row_map": dict(VERB_ROW_MAP),
        "value_head_init": VALUE_HEAD_INIT,
        "model_seed": int(model_seed),
        "transplant_map": transplant_map,
        "parameter_counts": {
            "source_total": source_parameters,
            "target_total": target_parameters,
            "difference": source_parameters - target_parameters,
            "expected_difference": expected_difference,
            "difference_ok": True,
        },
        "logits_equivalence": _verify_logits_equivalence(
            source, target, aim_bins=aim_bins, seed=int(model_seed)
        ),
        "stored_hyperparameters": {
            key: float(value) if isinstance(value, float) else int(value)
            for key, value in STORED_HYPERPARAMETERS.items()
        },
        "stored_hyperparameters_are_inert_for_trainer_v2b": True,
    }
    return target, evidence


def run(
    *,
    source_path: Path,
    output_dir: Path,
    device: str = "cpu",
    model_seed: int = 99_081_679,
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    source_path = source_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"bootstrap output already exists: {output_dir}"
        )
    started = time.perf_counter()
    source_sha = base._sha256(source_path)
    source = MaskablePPO.load(str(source_path), device=str(device))
    target, evidence = build_macro_model_from_per_tick(
        source, model_seed=int(model_seed), device=str(device)
    )
    output_dir.mkdir(parents=True)
    model_path = output_dir / OUTPUT_MODEL_NAME
    target.save(str(model_path))

    # Round-trip: the saved zip must load into a bitwise-equal policy
    # without any trainer module in play (the pickled policy_kwargs
    # resolve the extractor class from the distill-v3 module path).
    reloaded = MaskablePPO.load(str(model_path), device="cpu")
    live = {
        key: value.detach().cpu()
        for key, value in target.policy.state_dict().items()
    }
    for key, value in reloaded.policy.state_dict().items():
        if not torch.equal(value.detach().cpu(), live[key]):
            raise RuntimeError(f"round-trip drifted: {key}")
    if reloaded.action_space != target.action_space:
        raise RuntimeError("round-trip changed the action space")
    if reloaded.observation_space != target.observation_space:
        raise RuntimeError("round-trip changed the observation space")
    masked_predict = _masked_predict_smoke(
        reloaded, aim_bins=int(evidence["aim_bins"])
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": base._utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "builder": {
            "path": str(SCRIPT_PATH),
            "sha256": base._sha256(SCRIPT_PATH),
        },
        "extractor_module": {
            "module": "tools.distill_alphazuma_55_park_settle_v3",
            "path": str(Path(trainer_v3.__file__).resolve()),
            "sha256": base._sha256(Path(trainer_v3.__file__).resolve()),
            "class": "StackedRevengeEntityFeatureExtractor",
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha,
            "interface": "per-tick MultiDiscrete([4, aim_bins])",
        },
        "output": {
            "path": str(model_path),
            "sha256": base._sha256(model_path),
            "bytes": int(model_path.stat().st_size),
            "interface": "park-settle macro MultiDiscrete([3, aim_bins])",
        },
        "round_trip_bitwise_equal": True,
        "masked_predict_smoke": masked_predict,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        **evidence,
        "formal_seed_consumption": False,
        "training_authority": False,
    }
    base._write_atomic(output_dir / RECEIPT_NAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-seed", type=int, default=99_081_679)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(
        source_path=args.source_model.expanduser().resolve(strict=True),
        output_dir=args.output_dir.expanduser(),
        device=str(args.device),
        model_seed=int(args.model_seed),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": receipt["output"],
                "parameter_counts": receipt["parameter_counts"],
                "logits_equivalence": receipt["logits_equivalence"],
                "round_trip_bitwise_equal": receipt[
                    "round_trip_bitwise_equal"
                ],
                "formal_seed_consumption": False,
                "training_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
