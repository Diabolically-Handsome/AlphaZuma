"""Freeze a fixed-focus PPO route for empirically identified near-miss levels.

This wrapper reuses the audited hard-rescue inventory and seed checks, but it
freezes a deliberately different curriculum hypothesis: levels that repeatedly
reach the 12,000-tick limit without a win receive most of the sampling mass,
and adaptive reweighting is disabled so that focus cannot immediately decay.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_distilled_hard_rescue_v1 as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _focused_weight(
    *,
    level_id: str,
    source: dict[str, Any],
    aggregate: dict[str, int],
) -> float:
    del level_id
    source_wins = int(source["wins"])
    source_recent = float(source["recent_win_rate"])
    aggregate_wins = int(aggregate["wins"])
    episodes = int(aggregate["episodes"])
    truncations = int(aggregate["truncations"])

    # Retain the source checkpoint's solved repertoire with a small but real
    # rehearsal floor.  Fragile solved levels receive extra rehearsal.
    if source_wins > 0:
        return 0.75 if source_recent >= 0.25 else 1.25

    # Sibling-only wins are useful transfer tasks, but are not the central
    # hypothesis of this route.
    if aggregate_wins > 0:
        return 1.5

    truncation_rate = truncations / episodes if episodes else 0.0
    if episodes >= 128 and truncation_rate >= 0.85:
        return 12.0
    if episodes >= 128 and truncation_rate >= 0.70:
        return 8.0
    if episodes >= 128 and truncation_rate >= 0.40:
        return 4.0
    return 1.0


def build(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    original_weight = legacy._rescue_weight
    original_write = legacy.base._write_json_exclusive

    def _write_with_focus(path: Path, value: dict[str, Any]) -> None:
        schema = value.get("schema")
        if schema == "zuma-rl.alphazuma-55-distilled-hard-frontier-analysis":
            value["schema"] = "zuma-rl.alphazuma-55-distilled-nearmiss-focus-analysis"
            rows = list(value["per_level"])
            value["summary"].update(
                {
                    "fixed_focus_levels": sum(
                        float(row["rescue_initial_weight"]) >= 8.0 for row in rows
                    ),
                    "maximum_fixed_weight": max(
                        float(row["rescue_initial_weight"]) for row in rows
                    ),
                }
            )
            value["rescue_contract"] = {
                "single_policy": True,
                "environment_network_reward_and_ppo_hyperparameters_unchanged": True,
                "focus_signal": "zero aggregate wins and high aggregate time-limit truncation rate",
                "primary_focus_threshold": {
                    "minimum_aggregate_episodes": 128,
                    "minimum_truncation_rate": 0.85,
                    "weight": 12.0,
                },
                "secondary_focus_threshold": {
                    "minimum_aggregate_episodes": 128,
                    "minimum_truncation_rate": 0.70,
                    "weight": 8.0,
                },
                "solved_rehearsal_weights": [0.75, 1.25],
                "sibling_only_transfer_weight": 1.5,
                "adaptive_reweighting": False,
            }
            value["authority_boundary"].update(
                {
                    "classification": "engineering_training_only_fixed_nearmiss_focus",
                    "formal_selection_authority": False,
                    "promotion_requires_a_separate_frozen_successor_contract": True,
                }
            )
            value["builder"] = {
                "path": str(SCRIPT_PATH),
                "sha256": legacy.base._sha256(SCRIPT_PATH),
                "base_builder": {
                    "path": str(Path(legacy.__file__).resolve()),
                    "sha256": legacy.base._sha256(Path(legacy.__file__).resolve()),
                },
            }
        elif schema == "zuma-rl.overnight-multilevel-preregistration":
            value["route_family"] = "alphazuma-55-single-policy-distilled-nearmiss-fixed-focus"
            implementation = value["implementation"]
            analysis_reference = implementation.pop("distilled_hard_frontier_analysis")
            base_builder_reference = implementation.pop("distilled_hard_frontier_builder")
            implementation["distilled_nearmiss_focus_analysis"] = analysis_reference
            implementation["distilled_nearmiss_focus_builder"] = {
                "path": str(SCRIPT_PATH),
                "sha256": legacy.base._sha256(SCRIPT_PATH),
            }
            implementation["distilled_nearmiss_focus_base_builder"] = base_builder_reference
            run = value["runs"][0]
            run["description"] = (
                "Engineering-only fixed near-miss focus initialized from frozen "
                f"distilled checkpoint {value['evidence_anchor']['source_checkpoint_steps']}."
            )
            adaptive = copy.deepcopy(run["adaptive"])
            adaptive["enabled"] = False
            run["adaptive"] = adaptive
            value["evidence_anchor"]["curriculum_hypothesis"] = (
                "fixed high sampling mass on zero-win levels that repeatedly reach the time limit"
            )
        original_write(path, value)

    try:
        legacy._rescue_weight = _focused_weight
        legacy.base._write_json_exclusive = _write_with_focus
        return legacy.build(**kwargs)
    finally:
        legacy._rescue_weight = original_weight
        legacy.base._write_json_exclusive = original_write


def main(argv: list[str] | None = None) -> int:
    args = legacy.build_parser().parse_args(argv)
    analysis_output = args.analysis_output.expanduser().resolve()
    route_output = args.route_output.expanduser().resolve()
    analysis, route = build(
        master_path=args.master.expanduser().resolve(strict=True),
        route_paths=[path.expanduser().resolve(strict=True) for path in args.training_route],
        source_route_path=args.source_route.expanduser().resolve(strict=True),
        checkpoint_path=args.checkpoint.expanduser().resolve(strict=True),
        checkpoint_steps=args.checkpoint_steps,
        route_id=args.route_id,
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        model_seed=args.model_seed,
        episode_seed_base=args.episode_seed_base,
        num_envs=args.num_envs,
        analysis_output=analysis_output,
        route_output=route_output,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_NEARMISS_FOCUS_TRAINING",
                "analysis": {
                    "path": str(analysis_output),
                    "sha256": legacy.base._sha256(analysis_output),
                },
                "route": {
                    "path": str(route_output),
                    "sha256": legacy.base._sha256(route_output),
                },
                "summary": analysis["summary"],
                "run": route["runs"][0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
