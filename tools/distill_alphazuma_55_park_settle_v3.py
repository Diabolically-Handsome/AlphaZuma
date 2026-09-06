"""Park-and-settle distillation v3: v1 recipe on lag-stacked observations.

THIN wrapper over ``tools.distill_alphazuma_55_park_settle_v1`` (recipe:
losses, annotations, training loop, calibration, receipts, checkpoints)
and ``tools.distill_alphazuma_55_park_settle_v2`` (memmap observation
backing).  Neither is re-implemented; v3 exists for exactly one reason:
a SINGLE frame makes chain velocity unobservable -- projectile flight is
20-60 ticks, the chain moves meanwhile, and both park-settle student
lineages plateau at ~10% shots-on-target with ~39-bin release error
while the teacher leads targets from internally derived velocity.  v3
gives the student temporal context by concatenating the current row
with lagged rows of the SAME episode along the feature axis
(``obs_dim x n_lags``) BEFORE the entity_polar feature extractor.

What v3 adds
------------
1. ``--stack-lags "0,4,8"`` (source-tick lags; the first must be 0 so
   the current frame leads the stack).  For every dataset row the rows
   of the same episode at ``tick_index - lag`` are gathered through a
   per-row index table precomputed ONCE from ``episode_indices`` /
   ``tick_indices`` (int32, rows x n_lags, ~12 MB at 1.06M rows), so
   training-time cost is only the extra memmap row reads.  Fallbacks
   when the lagged tick is not retained: reuse the NEAREST EARLIER
   retained row of that episode; when the target precedes the episode's
   earliest retained row, clamp to that earliest row (which duplicates
   the current row on an episode's first row).  On a CONTIGUOUS tick
   stream this matches ``zuma_rl.observation_stack_wrapper`` exactly;
   across retention GAPS it deliberately DIVERGES from runtime (the
   wrapper holds the true ``t - lag`` frame, the trainer only retained
   rows), so ``build_stack_gather`` quantifies that divergence and
   every receipt carries it: per-type/per-lag fallback counts plus a
   ``train_runtime_divergence`` block -- reads whose gathered source
   tick differs from the runtime wrapper's ``max(t - lag, 0)``, with
   tick-gap percentiles per lag, and episodes-retaining-tick-0 counts
   for the undertrained eval reset window.

Known train/runtime divergence (record in any preregistration)
--------------------------------------------------------------
Measured 2026-08-16 on the coverage-v4 merged dataset (s99081667,
1,060,587 rows, lags 0,4,8): lag-4 reads diverge from the runtime
wrapper on 7.65% of rows (staleness p50/p90/p99 = 26/45/104 ticks, max
1615) and lag-8 on 14.90% (p50 24, max 1615); those rows train verb and
low-weight aim supervision on frames older than labeled.  On FIRE rows
divergence is only 1.58% (lag 4) / 0.29% (lag 8) and on
pre-fire-window(<=10) rows 0.69% / 0.13%, so the aim supervision this
fix targets is ~99% runtime-consistent.  Only 16/275 episodes retain
tick 0 (median earliest retained tick 86), so at eval every episode's
first ~max(lag) ticks emit clamp/duplicate stacks the training set
barely contains (550 duplicate-current-row stacks).  The receipts
recompute all of this per dataset; preregistrations quoting this
trainer MUST carry the receipted numbers.  Cheapest future hardening:
down-weight or drop lagged reads whose receipted tick gap exceeds a
threshold, or have the dataset builder retain the exact ``t - lag``
frames for retained rows.
2. ``StackedRevengeEntityFeatureExtractor``: consumes the wider input
   through v1's untouched single-frame constructor path (the base
   ``RevengeEntityFeatureExtractor`` is built for one frame width and
   applied with SHARED weights to every frame); the output is
   ``[learned(frame_0), ..., learned(frame_{n-1}), polar_basis(frame_0)]``
   so ``initialize_polar_aim_head`` and the v1 aim-head identity anchor
   work unchanged.  With ``--stack-lags 0`` the extractor and the whole
   training run are bit-identical to v1/v2.
3. Training entry: dataset loading delegates to v2's memmap loader,
   optimization/receipts to ``v1.execute_training`` on a dataset whose
   ``observations`` member is a gather-backed ``StackedObservationView``
   (supports exactly the row fancy-indexing the v1 loop performs).
   ``completion.json`` additionally records ``observation_stack``
   (stack_lags plus the gather-fallback statistics); everything else
   keeps the v1 format.

Real-dataset launch (1,060,587 rows, cuda:0)::

    .venv/bin/python tools/distill_alphazuma_55_park_settle_v3.py \
        --dataset-manifest /mnt/d/ZumaTraining/\
alphazuma-55-coverage-replay-v4-merged-s99081667/replay.dataset.json \
        --original-root "/mnt/d/SteamLibrary/steamapps/common/\
Zuma's Revenge" \
        --run-dir /mnt/d/ZumaTraining/\
alphazuma-55-park-settle-distill-stack-s99081671-v1 \
        --device cuda:0 --epochs 40 --checkpoint-interval-epochs 4 \
        --learning-rate 3e-5 --learned-features-dim 2048 \
        --verb-class-weights 1.0 5.0 7.5 47.5 --stack-lags 0,4,8

``formal_seed_consumption`` stays ``false`` throughout (inherited from
v1's receipts): this wrapper never steps an environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import torch
from gymnasium import spaces

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v2 as v2
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    parse_stack_lags,
    stacked_box,
)
from zuma_rl.revenge_features import RevengeEntityFeatureExtractor


SCRIPT_PATH = Path(__file__).resolve()
GATHER_DTYPE = np.int32
FALLBACK_EXACT = "exact"
FALLBACK_NEAREST_EARLIER = "nearest_earlier"
FALLBACK_EPISODE_START = "episode_start_clamp"
DIVERGENCE_DEFINITION = (
    "lagged reads whose gathered source tick differs from the tick the "
    "runtime ObservationStackWrapper would supply on a live contiguous "
    "episode, max(t - lag, 0); divergence_ticks is the absolute tick gap "
    "between the two"
)
DIVERGENCE_DISCLOSURE = (
    "Across retention gaps the nearest-earlier fallback trains lagged "
    "slots on frames OLDER than the runtime wrapper's true t-lag frame, "
    "and episode-start clamps train on the episode's earliest RETAINED "
    "row while at eval every episode's first max(lag) ticks emit "
    "clamp/duplicate stacks the training distribution may barely contain "
    "(see episodes_retaining_tick_zero); preregistrations quoting this "
    "receipt must carry these numbers."
)


def _divergence_tick_summary(
    gaps: list[np.ndarray],
) -> dict[str, Any] | None:
    """Percentile summary of runtime-vs-gather tick gaps (None if empty)."""

    if not gaps:
        return None
    merged = np.concatenate(gaps)
    return {
        "p50": float(np.percentile(merged, 50)),
        "p90": float(np.percentile(merged, 90)),
        "p99": float(np.percentile(merged, 99)),
        "max": int(merged.max()),
    }


def build_stack_gather(
    *,
    episode_indices: np.ndarray,
    tick_indices: np.ndarray,
    stack_lags: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Precompute per-row lagged-gather indices plus honest fallback stats.

    For row ``i`` (episode ``e``, source tick ``t``) and lag ``l`` the
    gathered row is the retained row of episode ``e`` whose tick is the
    LARGEST tick ``<= t - l``:

    * tick ``t - l`` retained  -> that row (``exact``);
    * gap in the retained set  -> nearest earlier retained row
      (``nearest_earlier``);
    * ``t - l`` precedes the episode's earliest retained tick -> the
      earliest retained row (``episode_start_clamp``; on the episode's
      first row this duplicates the current row) -- the same clamp the
      runtime ``ObservationStackWrapper`` applies pre-episode.

    Both fallback kinds can DIVERGE from the runtime wrapper, which
    always holds the true ``t - l`` frame (clamped at the episode's
    true tick 0): the stats therefore include a
    ``train_runtime_divergence`` block counting, per lag, the reads
    whose gathered source tick differs from the runtime
    ``max(t - l, 0)`` plus percentiles of the absolute tick gap, and
    ``episodes_retaining_tick_zero`` for the eval reset window.

    Episode boundaries are never crossed.  Returns ``(gather, stats)``
    with ``gather`` shaped ``[rows, n_lags]`` int32.
    """

    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    tick_indices = np.asarray(tick_indices, dtype=np.int64)
    lags = parse_stack_lags(stack_lags)
    count = int(episode_indices.shape[0])
    if count < 1:
        raise ValueError("cannot build a stack gather over zero rows")
    if tick_indices.shape != (count,):
        raise ValueError("episode_indices and tick_indices must align")
    if count >= np.iinfo(GATHER_DTYPE).max:
        raise ValueError("row count exceeds the int32 gather index range")
    gather = np.empty((count, len(lags)), dtype=GATHER_DTYPE)
    per_lag = {
        int(lag): {
            FALLBACK_EXACT: 0,
            FALLBACK_NEAREST_EARLIER: 0,
            FALLBACK_EPISODE_START: 0,
        }
        for lag in lags
    }
    duplicate_current = 0
    divergent_counts = {int(lag): 0 for lag in lags}
    divergence_gaps: dict[int, list[np.ndarray]] = {
        int(lag): [] for lag in lags
    }
    episodes_total = 0
    episodes_retaining_tick_zero = 0

    order = np.argsort(episode_indices, kind="stable")
    sorted_episodes = episode_indices[order]
    boundaries = np.flatnonzero(np.diff(sorted_episodes) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [count]))
    for start, stop in zip(starts, stops):
        rows = order[start:stop]
        tick_order = np.argsort(tick_indices[rows], kind="stable")
        rows = rows[tick_order]
        ticks = tick_indices[rows]
        if rows.size > 1 and bool(np.any(np.diff(ticks) <= 0)):
            raise ValueError(
                "episode tick_indices must be unique to gather lagged rows"
            )
        episodes_total += 1
        if int(ticks[0]) == 0:
            episodes_retaining_tick_zero += 1
        for column, lag in enumerate(lags):
            targets = ticks - int(lag)
            position = np.searchsorted(ticks, targets, side="right") - 1
            before_start = position < 0
            clamped = np.where(before_start, 0, position)
            chosen = rows[clamped]
            chosen_ticks = ticks[clamped]
            gather[rows, column] = chosen.astype(GATHER_DTYPE)
            exact = (~before_start) & (chosen_ticks == targets)
            nearest = (~before_start) & ~exact
            per_lag[int(lag)][FALLBACK_EXACT] += int(np.sum(exact))
            per_lag[int(lag)][FALLBACK_NEAREST_EARLIER] += int(
                np.sum(nearest)
            )
            per_lag[int(lag)][FALLBACK_EPISODE_START] += int(
                np.sum(before_start)
            )
            duplicate_current += int(np.sum(before_start & (chosen == rows)))
            runtime_ticks = np.maximum(targets, 0)
            divergent = chosen_ticks != runtime_ticks
            if bool(np.any(divergent)):
                divergent_counts[int(lag)] += int(np.sum(divergent))
                divergence_gaps[int(lag)].append(
                    np.abs(
                        chosen_ticks[divergent] - runtime_ticks[divergent]
                    )
                )

    totals = {
        kind: sum(row[kind] for row in per_lag.values())
        for kind in (
            FALLBACK_EXACT,
            FALLBACK_NEAREST_EARLIER,
            FALLBACK_EPISODE_START,
        )
    }
    stats = {
        "rows": count,
        "stack_lags": [int(lag) for lag in lags],
        "lagged_reads": count * len(lags),
        FALLBACK_EXACT: totals[FALLBACK_EXACT],
        FALLBACK_NEAREST_EARLIER: totals[FALLBACK_NEAREST_EARLIER],
        FALLBACK_EPISODE_START: totals[FALLBACK_EPISODE_START],
        "episode_start_duplicate_current_row": duplicate_current,
        "fallback_reads": (
            totals[FALLBACK_NEAREST_EARLIER]
            + totals[FALLBACK_EPISODE_START]
        ),
        "per_lag": {
            str(lag): dict(row) for lag, row in sorted(per_lag.items())
        },
        "train_runtime_divergence": {
            "definition": DIVERGENCE_DEFINITION,
            "divergent_reads": sum(divergent_counts.values()),
            "divergent_fraction": (
                sum(divergent_counts.values()) / (count * len(lags))
            ),
            "per_lag": {
                str(lag): {
                    "divergent_reads": divergent_counts[lag],
                    "divergent_fraction": divergent_counts[lag] / count,
                    "divergence_ticks": _divergence_tick_summary(
                        divergence_gaps[lag]
                    ),
                }
                for lag in sorted(divergent_counts)
            },
            "episodes": episodes_total,
            "episodes_retaining_tick_zero": episodes_retaining_tick_zero,
            "disclosure": DIVERGENCE_DISCLOSURE,
        },
    }
    return gather, stats


class StackedObservationView:
    """Gather-backed 2-D view: row fancy-indexing concatenates lag frames.

    Supports exactly the access pattern v1's offline loop performs on
    ``dataset.observations`` -- 1-D integer-array row indexing plus the
    ``shape``/``ndim``/``dtype`` attributes ``validate()`` reads.  The
    base array (typically v2's read-only float32 memmap) is never
    materialized beyond the indexed rows.
    """

    def __init__(self, base: np.ndarray, gather: np.ndarray) -> None:
        gather = np.asarray(gather)
        if getattr(base, "ndim", None) != 2:
            raise ValueError("stacked view requires a 2-D base array")
        if gather.ndim != 2 or not np.issubdtype(gather.dtype, np.integer):
            raise ValueError("gather must be a 2-D integer index array")
        rows = int(base.shape[0])
        if gather.shape[0] != rows:
            raise ValueError("gather rows must align with the base rows")
        if rows and (
            int(gather.min()) < 0 or int(gather.max()) >= rows
        ):
            raise ValueError("gather indices escape the base array")
        self._base = base
        self._gather = gather
        self.shape = (rows, int(base.shape[1]) * int(gather.shape[1]))
        self.ndim = 2
        self.dtype = base.dtype

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def base_observations(self) -> np.ndarray:
        return self._base

    @property
    def gather_rows(self) -> np.ndarray:
        return self._gather

    def __getitem__(self, rows: Any) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise TypeError(
                "StackedObservationView supports 1-D integer row arrays "
                f"only, got {rows!r}"
            )
        picked = self._gather[rows]
        parts = [
            np.asarray(self._base[picked[:, column]])
            for column in range(picked.shape[1])
        ]
        return np.concatenate(parts, axis=1)


def stack_dataset(
    dataset: v1.ParkSettleDataset, *, stack_lags: Any
) -> tuple[v1.ParkSettleDataset, np.ndarray, dict[str, Any]]:
    """Wrap a loaded dataset's observations in the lag-stacked view."""

    lags = parse_stack_lags(stack_lags)
    if len(lags) > 1 and dataset.temporal_structure != "episode_ticks":
        raise ValueError(
            "observation stacking requires episode_ticks temporal "
            f"structure; this dataset is {dataset.temporal_structure!r}"
        )
    gather, stats = build_stack_gather(
        episode_indices=dataset.episode_indices,
        tick_indices=dataset.tick_indices,
        stack_lags=lags,
    )
    view = StackedObservationView(dataset.observations, gather)
    stacked = v1.ParkSettleDataset(
        observations=view,  # type: ignore[arg-type]
        raw_actions=dataset.raw_actions,
        masks=dataset.masks,
        episode_indices=dataset.episode_indices,
        tick_indices=dataset.tick_indices,
        temporal_structure=dataset.temporal_structure,
    )
    stacked.validate()
    return stacked, gather, stats


class StackedRevengeEntityFeatureExtractor(RevengeEntityFeatureExtractor):
    """Entity_polar extractor over ``stacked_frames`` feature-concat frames.

    Construction routes every layout kwarg through the UNTOUCHED v1
    single-frame constructor against the leading frame's slice of the
    stacked space; the shared-weight base pipeline is applied to each
    frame and the outputs are concatenated as
    ``[learned(frame_0), ..., learned(frame_{n-1}), polar(frame_0)]``.
    The polar basis stays the LAST ``polar_aim_bins`` features and
    ``learned_features_dim`` is widened to ``n * per_frame``, so
    ``initialize_polar_aim_head`` works unchanged.  With
    ``stacked_frames=1`` construction and forward are bit-identical to
    the base class.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        stacked_frames: int = 1,
        **base_kwargs: Any,
    ) -> None:
        stacked_frames = int(stacked_frames)
        if stacked_frames < 1:
            raise ValueError("stacked_frames must be positive")
        total = int(observation_space.shape[0])
        if total % stacked_frames:
            raise ValueError(
                f"stacked observation width {total} is not divisible by "
                f"{stacked_frames} frames"
            )
        frame_width = total // stacked_frames
        frame_space = spaces.Box(
            low=np.asarray(observation_space.low)[:frame_width],
            high=np.asarray(observation_space.high)[:frame_width],
            dtype=observation_space.dtype,
        )
        super().__init__(frame_space, **base_kwargs)
        self.stacked_frames = stacked_frames
        self.frame_width = frame_width
        self.per_frame_learned_features_dim = int(self.learned_features_dim)
        self.learned_features_dim = (
            stacked_frames * self.per_frame_learned_features_dim
        )
        self._features_dim = self.learned_features_dim + max(
            0, self.polar_aim_bins
        )
        self._observation_space = observation_space

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.stacked_frames == 1:
            return super().forward(observations)
        frames = observations.reshape(
            -1, self.stacked_frames, self.frame_width
        )
        per_frame = self.per_frame_learned_features_dim
        learned: list[torch.Tensor] = []
        polar: torch.Tensor | None = None
        for index in range(self.stacked_frames):
            output = super().forward(frames[:, index, :])
            learned.append(output[:, :per_frame])
            if index == 0:
                polar = output[:, per_frame:]
        assert polar is not None
        return torch.cat((*learned, polar), dim=1)


def stack_policy_kwargs(
    policy_kwargs: dict[str, Any], *, frames: int
) -> dict[str, Any]:
    """Swap a v1 entity_polar policy_kwargs dict to the stacked extractor."""

    kwargs = dict(policy_kwargs)
    extractor = dict(kwargs["features_extractor_kwargs"])
    extractor["stacked_frames"] = int(frames)
    kwargs["features_extractor_class"] = StackedRevengeEntityFeatureExtractor
    kwargs["features_extractor_kwargs"] = extractor
    return kwargs


def stacked_entity_polar_policy_kwargs(
    prototype: Any, *, learned_features_dim: int, stack_lags: Any
) -> dict[str, Any]:
    """v1's entity_polar kwargs widened to the stacked extractor."""

    lags = parse_stack_lags(stack_lags)
    return stack_policy_kwargs(
        v1.entity_polar_policy_kwargs(
            prototype, learned_features_dim=int(learned_features_dim)
        ),
        frames=len(lags),
    )


def execute_stacked_training(
    *,
    model: Any,
    dataset: v1.ParkSettleDataset,
    run_dir: Path,
    model_config: v1.ParkSettleModelConfig,
    train_config: v1.ParkSettleTrainConfig,
    dataset_receipt: dict[str, Any] | None,
    stack_lags: Any,
    gather_stats: dict[str, Any],
) -> dict[str, Any]:
    """v1's receipted training plus the observation_stack completion block."""

    lags = parse_stack_lags(stack_lags)
    run_dir = Path(run_dir)
    completion = v1.execute_training(
        model=model,
        dataset=dataset,
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=dataset_receipt,
    )
    completion["observation_stack"] = {
        "stack_lags": [int(lag) for lag in lags],
        "frames": len(lags),
        "gather_dtype": str(np.dtype(GATHER_DTYPE)),
        "gather_fallback": dict(gather_stats),
        "trainer": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
    }
    legacy._write_json_atomic(
        run_dir.resolve() / "completion.json", v1._jsonable(completion)
    )
    return completion


def build_parser() -> argparse.ArgumentParser:
    """v2's parser (v1 CLI + extraction flags) plus ``--stack-lags``."""

    parser = v2.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags concatenated along the "
            "feature axis; the first must be 0 (current frame)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lags = parse_stack_lags(args.stack_lags)
    if int(args.extraction_buffer_mib) < 1:
        raise SystemExit("--extraction-buffer-mib must be at least 1")
    buffer_bytes = int(args.extraction_buffer_mib) * 1024 * 1024
    if args.extract_observations_only:
        results = v2.extract_manifest_observations(
            args.dataset_manifest.expanduser(), buffer_bytes=buffer_bytes
        )
        print(
            json.dumps(
                v1._jsonable({"status": "EXTRACTED", "extractions": results}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    dataset, receipt = v2.load_replay_dataset_memmap(
        args.dataset_manifest.expanduser(),
        extraction_buffer_bytes=buffer_bytes,
    )
    base_width = int(dataset.observations.shape[1])
    stacked_dataset, _, gather_stats = stack_dataset(
        dataset, stack_lags=lags
    )
    receipt["observation_stack"] = {
        "stack_lags": [int(lag) for lag in lags],
        "frames": len(lags),
        "base_observation_width": base_width,
        "stacked_observation_width": base_width * len(lags),
        "gather_fallback": dict(gather_stats),
    }
    if args.validate_only:
        print(
            json.dumps(
                v1._jsonable({"status": "VALID", "dataset": receipt}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if args.original_root is None or args.run_dir is None:
        raise SystemExit(
            "--original-root and --run-dir are required for training"
        )

    # Config construction is v1.main's, verbatim.
    loss_config = v1.ParkSettleLossConfig(
        fire_window_ticks=int(args.fire_window_ticks),
        wait_aim_weight=float(args.wait_aim_weight),
        verb_loss_weight=float(args.verb_loss_weight),
        aim_loss_weight=float(args.aim_loss_weight),
        verb_class_weights=(
            tuple(float(value) for value in args.verb_class_weights)
            if args.verb_class_weights is not None
            else None
        ),
        max_grad_norm=float(args.max_grad_norm),
    )
    train_config = v1.ParkSettleTrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        holdout_fraction=float(args.holdout_fraction),
        shuffle_seed=int(args.shuffle_seed),
        device=str(args.device),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        loss=loss_config,
    )
    model_config = v1.ParkSettleModelConfig(
        learned_features_dim=int(args.learned_features_dim),
        aim_head_identity_scale=float(args.aim_head_identity_scale),
        model_seed=int(args.model_seed),
    )
    prototype = v1._launch_prototype(
        original_root=args.original_root.expanduser().resolve(strict=True),
        max_ticks=int(args.max_ticks),
        observation_width=base_width,
    )
    try:
        policy_kwargs = stacked_entity_polar_policy_kwargs(
            prototype,
            learned_features_dim=int(model_config.learned_features_dim),
            stack_lags=lags,
        )
        base_space = prototype.observation_space
        action_space = prototype.action_space
    finally:
        prototype.close()
    model = v1.build_student_model(
        observation_space=stacked_box(base_space, len(lags)),
        action_space=action_space,
        policy_kwargs=policy_kwargs,
        model_config=model_config,
        train_config=train_config,
    )
    completion = execute_stacked_training(
        model=model,
        dataset=stacked_dataset,
        run_dir=args.run_dir.expanduser(),
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=lags,
        gather_stats=gather_stats,
    )
    print(
        json.dumps(
            v1._jsonable(
                {
                    "status": completion["status"],
                    "final_model": completion["final_model"],
                    "observation_stack": completion["observation_stack"],
                    "illegal_teacher_verb_fraction": completion[
                        "illegal_teacher_verb_fraction"
                    ],
                    "calibration": completion["calibration"],
                    "formal_seed_consumption": False,
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
