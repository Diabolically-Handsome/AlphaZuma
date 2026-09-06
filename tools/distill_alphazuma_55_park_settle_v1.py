"""Park-and-settle distillation recipe v1 on the entity_polar basis.

Lineage: successor recipe to distill_alphazuma_55_motor_observable_replay_v2
(current best student) built on the entity_polar feature basis from
distill_alphazuma_55_polar_v1 / distill_alphazuma_55_polar_intent_wide_v1,
which passed the preregistered offline aim screen at 0.662 exact versus
0.17-0.26 for generic heads.  It corrects three verified recipe failures:

1. ``aim_loss_scope=fire_only`` supervised exactly the ticks that do NOT
   decide shots.  Under HumanSpeedrunWrapper the projectile release angle is
   the live cursor 6 ticks after the fire request, and the slew-limited
   cursor is steered by aim intents issued ~12-18 ticks earlier -- almost all
   on wait ticks.  The settled teacher parks the target bin during waits (the
   wait action CARRIES the target aim bin), so this recipe supervises the aim
   head with plain categorical cross-entropy on ALL ticks, weighting ticks
   inside a configurable pre-fire window (default 24 ticks before a teacher
   fire edge, edge tick included) at 1.0 and all remaining "parked" ticks at
   ``wait_aim_weight`` (default 0.25).  Ticks 1-6 after the fire request also
   influence the release; the teacher keeps the target parked there, so the
   labels stay consistent even though those ticks carry the parked weight.
2. The forced 45/35/15/5 verb batch rebalance made students fire ~10x too
   often at runtime.  Batches here are uniform shuffled samples with NO verb
   rebalancing; an optional per-verb ``verb_class_weights`` config (default
   None) is the only, explicitly-reported, reweighting hook.
3. ~26% of raw teacher intent labels are illegal under the exact runtime
   action mask.  The verb head is trained with cross-entropy over logits
   masked by the EXACT runtime valid-verb mask; ticks whose teacher verb is
   illegal under that mask are DROPPED from the verb loss (their fraction is
   reported in the completion receipt) while remaining fully supervised in
   the aim loss.  ``circular_smoothed`` aim losses were rejected by the
   preregistered offline screens; only plain categorical CE is implemented.

Dataset contract (offline, no environment stepping during optimization):
the trainer consumes a JSON manifest naming NPZ shards.  Two layouts are
accepted and auto-detected by manifest keys:

* replay-v2 style: ``{"shards": [{"path": ..., "sha256": ...}, ...]}``.
  Each NPZ holds ``observations`` [N, obs] float32, ``actions`` (alias
  ``raw_actions``) [N, 2] int64 raw teacher intent, and ``masks`` (alias
  ``runtime_masks``/``exact_masks``/``action_masks``) [N, 184] bool.
  Reservoir shards preserve no temporal adjacency, so the pre-fire window
  degrades to the fire-intent ticks themselves and the receipt records
  ``temporal_structure = "absent"``.
* coverage-v4 style (build_alphazuma_55_motor_observable_replay_v4, not
  imported here): the builder's ``<prefix>.dataset.json`` names ONE
  aggregate NPZ under ``{"shards": [{"path": ..., "sha256": ...}]}``.
  That NPZ carries ``observations``/``actions``/``training_masks`` plus
  ``exact_masks`` (the UNRELAXED runtime mask) and per-row
  ``episode_indices`` / ``tick_indices`` identity arrays, so episode
  structure survives aggregation and exact pre-fire windows are
  available (``temporal_structure = "episode_ticks"``).  A
  ``{"episodes": [{"path": ..., "sha256": ...}, ...]}`` layout with one
  NPZ per episode in tick order (optional ``tick_indices``) is also
  accepted; episode identity then comes from the manifest entry.

Either layout may instead provide explicit per-row ``episode_indices`` and
``tick_indices`` arrays inside every shard to opt into exact windows.

Mask semantics are enforced at load time: an NPZ shard that offers only
the relaxed ``training_masks`` (the replay-v2 key that unmasks the raw
teacher verb) is REFUSED, and when the manifest declares
``mask_semantics`` the loaded mask key must be marked as the exact
runtime mask or the load aborts.  The declared semantics are recorded in
the dataset receipt and therefore in the completion receipt, so a run
can never silently report ``illegal_teacher_verb_fraction == 0.0``
because relaxed masks were substituted for exact ones.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_v1 as polar
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.revenge_features import (
    RevengeEntityFeatureExtractor,
    initialize_polar_aim_head,
    revenge_polar_policy_kwargs,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ARCHITECTURE = "entity_polar_park_settle"
VERB_NAMES = settled.VERB_NAMES
VERB_COUNT = len(VERB_NAMES)
AIM_BINS = 180
MASK_WIDTH = VERB_COUNT + AIM_BINS
WAIT_VERB = 0
FIRE_VERB = 1
FIRE_CYCLE_TICKS = 21
VERB_MASK_FILL = -1.0e8
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-park-settle-distillation"

_OBSERVATION_KEYS = ("observations", "obs", "states")
_ACTION_KEYS = (
    "raw_actions",
    "raw_intent_actions",
    "teacher_actions",
    "actions",
    "labels",
)
_MASK_KEYS = ("runtime_masks", "exact_masks", "action_masks", "masks")
# Relaxed replay-v2 training masks unmask the raw teacher verb; loading
# them here would silently zero the illegal-teacher-verb accounting, so
# they are refused rather than aliased.
_RELAXED_MASK_KEYS = ("training_masks", "relaxed_masks")
_EXACT_MASK_SEMANTICS = frozenset(
    {"exact_runtime_valid_action_mask", "exact_runtime_mask"}
)
_EPISODE_KEYS = ("episode_indices", "episode_ids")
_TICK_KEYS = ("tick_indices", "ticks", "step_indices")


@dataclass(frozen=True)
class ParkSettleLossConfig:
    """Park-and-settle loss shape: what gets supervised, and how hard."""

    fire_window_ticks: int = 24
    wait_aim_weight: float = 0.25
    verb_loss_weight: float = 1.0
    aim_loss_weight: float = 1.0
    verb_class_weights: tuple[float, ...] | None = None
    max_grad_norm: float = 0.5

    def validate(self) -> None:
        if int(self.fire_window_ticks) < 0:
            raise ValueError("fire_window_ticks must be non-negative")
        if not 0.0 <= float(self.wait_aim_weight) <= 1.0:
            raise ValueError("wait_aim_weight must be within [0, 1]")
        if float(self.verb_loss_weight) < 0.0:
            raise ValueError("verb_loss_weight must be non-negative")
        if float(self.aim_loss_weight) <= 0.0:
            raise ValueError("aim_loss_weight must be positive")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.verb_class_weights is not None:
            weights = tuple(float(value) for value in self.verb_class_weights)
            if len(weights) != VERB_COUNT:
                raise ValueError("verb_class_weights must cover every verb")
            if any(value < 0.0 for value in weights):
                raise ValueError("verb_class_weights must be non-negative")


@dataclass(frozen=True)
class ParkSettleModelConfig:
    """Entity-polar student capacity and initialization."""

    learned_features_dim: int = 1024
    aim_head_identity_scale: float = 5.0
    model_seed: int = 1_543_003_100

    def validate(self) -> None:
        if int(self.learned_features_dim) < 1:
            raise ValueError("learned_features_dim must be positive")
        if float(self.aim_head_identity_scale) <= 0.0:
            raise ValueError("aim_head_identity_scale must be positive")


@dataclass(frozen=True)
class ParkSettleTrainConfig:
    """Optimization schedule, splits, and device for one offline run."""

    epochs: int = 4
    batch_size: int = 512
    learning_rate: float = 1.0e-5
    holdout_fraction: float = 0.2
    shuffle_seed: int = 1_543_003_101
    device: str = "cpu"
    checkpoint_interval_epochs: int = 1
    loss: ParkSettleLossConfig = field(default_factory=ParkSettleLossConfig)

    def validate(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError("epochs must be positive")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= float(self.holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be within [0, 1)")
        if int(self.checkpoint_interval_epochs) < 0:
            raise ValueError("checkpoint_interval_epochs cannot be negative")
        self.loss.validate()


@dataclass
class ParkSettleDataset:
    """Per-tick teacher intent rows with the exact runtime action masks.

    ``raw_actions[:, 1]`` is the teacher's commanded aim bin at every tick;
    on wait ticks that IS the parked target bin (the park-and-settle
    channel).  ``masks`` must be the exact runtime valid-action mask, NOT
    the relaxed training mask that unmasks the teacher verb.
    """

    observations: np.ndarray
    raw_actions: np.ndarray
    masks: np.ndarray
    episode_indices: np.ndarray
    tick_indices: np.ndarray
    temporal_structure: str

    def validate(self) -> None:
        count = int(self.observations.shape[0])
        if count < 1:
            raise ValueError("park-settle dataset is empty")
        if self.observations.ndim != 2:
            raise ValueError("observations must be [rows, features]")
        if self.raw_actions.shape != (count, 2):
            raise ValueError("raw_actions must be [rows, 2]")
        if self.masks.shape != (count, MASK_WIDTH):
            raise ValueError(
                f"masks must be [rows, {MASK_WIDTH}] exact runtime masks"
            )
        if self.episode_indices.shape != (count,):
            raise ValueError("episode_indices must align with rows")
        if self.tick_indices.shape != (count,):
            raise ValueError("tick_indices must align with rows")
        if self.temporal_structure not in {"episode_ticks", "absent"}:
            raise ValueError(
                f"unknown temporal structure: {self.temporal_structure!r}"
            )
        verbs = self.raw_actions[:, 0]
        aims = self.raw_actions[:, 1]
        if not (
            bool(np.all((verbs >= 0) & (verbs < VERB_COUNT)))
            and bool(np.all((aims >= 0) & (aims < AIM_BINS)))
        ):
            raise ValueError("teacher actions escape the actor interface")

    @property
    def sample_count(self) -> int:
        return int(self.observations.shape[0])

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> "ParkSettleDataset":
        """Adapt replay-v2 style in-memory (obs, action, mask) tuples."""

        count = len(rows)
        dataset = cls(
            observations=np.asarray(
                [row[0] for row in rows], dtype=np.float32
            ),
            raw_actions=np.asarray([row[1] for row in rows], dtype=np.int64),
            masks=np.asarray([row[2] for row in rows], dtype=np.bool_),
            episode_indices=np.arange(count, dtype=np.int64),
            tick_indices=np.zeros(count, dtype=np.int64),
            temporal_structure="absent",
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_episode_rows(
        cls,
        *,
        observations: np.ndarray,
        raw_actions: np.ndarray,
        masks: np.ndarray,
        episode_indices: np.ndarray,
        tick_indices: np.ndarray,
    ) -> "ParkSettleDataset":
        dataset = cls(
            observations=np.asarray(observations, dtype=np.float32),
            raw_actions=np.asarray(raw_actions, dtype=np.int64),
            masks=np.asarray(masks, dtype=np.bool_),
            episode_indices=np.asarray(episode_indices, dtype=np.int64),
            tick_indices=np.asarray(tick_indices, dtype=np.int64),
            temporal_structure="episode_ticks",
        )
        dataset.validate()
        return dataset


@dataclass
class ParkSettleAnnotations:
    """Derived per-row supervision channels (see annotate_park_settle)."""

    verb_legal: np.ndarray
    aim_label_legal: np.ndarray
    fire_intent: np.ndarray
    fire_commit: np.ndarray
    fire_edge: np.ndarray
    pre_fire_window: np.ndarray
    aim_weights: np.ndarray


def annotate_park_settle(
    dataset: ParkSettleDataset,
    loss: ParkSettleLossConfig,
) -> ParkSettleAnnotations:
    """Derive legality, fire edges, pre-fire windows, and aim weights.

    * ``verb_legal``: teacher verb is legal under the exact runtime mask.
    * ``fire_commit``: legal fire-intent ticks (the request can enter the
      gun on these ticks; illegal fire intents execute as waits).
    * ``fire_edge``: rising edges of legal fire intent (buttons act on
      rising edges only under the human input wrapper).
    * ``pre_fire_window``: ticks within ``fire_window_ticks`` BEFORE a fire
      edge (edge tick included) union all fire-commit ticks.  Without
      temporal structure this degrades to the raw fire-intent ticks.
    * ``aim_weights``: 1.0 inside the window, ``wait_aim_weight`` outside.
    """

    loss.validate()
    count = dataset.sample_count
    rows = np.arange(count)
    verbs = dataset.raw_actions[:, 0]
    aims = dataset.raw_actions[:, 1]
    verb_legal = dataset.masks[rows, verbs]
    aim_label_legal = dataset.masks[rows, VERB_COUNT + aims]
    fire_intent = verbs == FIRE_VERB
    fire_commit = fire_intent & verb_legal
    fire_edge = fire_commit.copy()
    pre_fire_window = np.zeros(count, dtype=np.bool_)
    if dataset.temporal_structure == "episode_ticks":
        window = int(loss.fire_window_ticks)
        for episode in np.unique(dataset.episode_indices):
            episode_rows = np.nonzero(dataset.episode_indices == episode)[0]
            order = np.argsort(
                dataset.tick_indices[episode_rows], kind="stable"
            )
            episode_rows = episode_rows[order]
            ticks = dataset.tick_indices[episode_rows]
            commits = fire_commit[episode_rows]
            if not bool(commits.any()):
                continue
            previous_commit = np.zeros(len(episode_rows), dtype=np.bool_)
            previous_commit[1:] = commits[:-1] & (
                ticks[1:] == ticks[:-1] + 1
            )
            fire_edge[episode_rows] = commits & ~previous_commit
            edge_ticks = ticks[fire_edge[episode_rows]]
            position = np.searchsorted(edge_ticks, ticks, side="left")
            has_next = position < edge_ticks.size
            next_edge = edge_ticks[np.minimum(position, edge_ticks.size - 1)]
            distance = np.where(has_next, next_edge - ticks, window + 1)
            pre_fire_window[episode_rows] = has_next & (distance <= window)
        pre_fire_window |= fire_commit
    else:
        pre_fire_window = fire_intent.copy()
    aim_weights = np.where(
        pre_fire_window, 1.0, float(loss.wait_aim_weight)
    ).astype(np.float32)
    return ParkSettleAnnotations(
        verb_legal=verb_legal,
        aim_label_legal=aim_label_legal,
        fire_intent=fire_intent,
        fire_commit=fire_commit,
        fire_edge=fire_edge,
        pre_fire_window=pre_fire_window,
        aim_weights=aim_weights,
    )


class _SpaceOnlyMaskableEnv(gym.Env):
    """Interface-only environment for building offline maskable students."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation = np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return observation, 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(int(np.sum(self.action_space.nvec)), dtype=np.bool_)


def tiny_entity_polar_interface(
    *, features_dim: int = 32, aim_bins: int = AIM_BINS
) -> tuple[spaces.Box, spaces.MultiDiscrete, dict[str, Any]]:
    """Synthetic entity_polar interface for CPU unit tests and smoke runs.

    Returns (observation_space, action_space, policy_kwargs) with the same
    packed layout family the real full55 environment exposes, at toy scale.
    """

    max_balls = 4
    num_colors = 3
    curve_feature_count = 1
    ball_feature_size = 3 + num_colors + 1 + curve_feature_count
    max_projectiles = 2
    projectile_feature_size = 3
    global_feature_size = 2 * num_colors + 2
    balls_stop = max_balls * ball_feature_size
    projectiles_stop = balls_stop + max_projectiles * projectile_feature_size
    observation_size = projectiles_stop + global_feature_size
    observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(observation_size,),
        dtype=np.float32,
    )
    action_space = spaces.MultiDiscrete([VERB_COUNT, int(aim_bins)])
    policy_kwargs = {
        "features_extractor_class": RevengeEntityFeatureExtractor,
        "features_extractor_kwargs": {
            "max_balls": max_balls,
            "ball_feature_size": ball_feature_size,
            "balls_start": 0,
            "balls_stop": balls_stop,
            "max_projectiles": max_projectiles,
            "projectile_feature_size": projectile_feature_size,
            "projectiles_start": balls_stop,
            "projectiles_stop": projectiles_stop,
            "global_feature_size": global_feature_size,
            "globals_start": projectiles_stop,
            "globals_stop": observation_size,
            "relational": True,
            "num_colors": num_colors,
            "ball_color_start": 3,
            "ball_exploding_index": 3 + num_colors,
            "ball_curve_start": 4 + num_colors,
            "curve_feature_count": curve_feature_count,
            "global_current_color_start": 0,
            "global_next_color_start": num_colors,
            "polar_aim_bins": int(aim_bins),
            "polar_sigma_bins": 0.75,
            "shooter_x_normalized": 0.0,
            "shooter_y_normalized": 0.0,
            "logical_width": 640.0,
            "logical_height": 480.0,
            "features_dim": int(features_dim),
        },
        "net_arch": {"pi": [], "vf": [32]},
    }
    return observation_space, action_space, policy_kwargs


def entity_polar_policy_kwargs(
    prototype: Any, *, learned_features_dim: int
) -> dict[str, Any]:
    """Entity-polar policy kwargs with configurable learned capacity."""

    kwargs = revenge_polar_policy_kwargs(prototype)
    extractor = dict(kwargs["features_extractor_kwargs"])
    extractor["features_dim"] = int(learned_features_dim)
    kwargs["features_extractor_kwargs"] = extractor
    return kwargs


def build_student_model(
    *,
    observation_space: spaces.Box,
    action_space: spaces.MultiDiscrete,
    policy_kwargs: dict[str, Any],
    model_config: ParkSettleModelConfig,
    train_config: ParkSettleTrainConfig,
) -> Any:
    """Build a fresh maskable entity_polar student for offline training."""

    from sb3_contrib import MaskablePPO

    model_config.validate()
    train_config.validate()
    nvec = tuple(int(value) for value in action_space.nvec)
    if len(nvec) != 2 or nvec[0] != VERB_COUNT:
        raise ValueError("park-settle students need factorized (verb, aim)")
    environment = _SpaceOnlyMaskableEnv(observation_space, action_space)
    model = MaskablePPO(
        "MlpPolicy",
        environment,
        learning_rate=float(train_config.learning_rate),
        n_steps=512,
        batch_size=int(train_config.batch_size),
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        seed=int(model_config.model_seed),
        device=str(train_config.device),
        verbose=0,
    )
    initialize_polar_aim_head(
        model, scale=float(model_config.aim_head_identity_scale)
    )
    return model


def _policy_logits(model: Any, observation_tensor: Any) -> tuple[Any, Any]:
    """Return (verb_logits, aim_logits) from one actor forward pass."""

    policy = model.policy
    features = policy.extract_features(
        observation_tensor, policy.pi_features_extractor
    )
    latent_pi = policy.mlp_extractor.forward_actor(features)
    flat_logits = policy.action_net(latent_pi)
    nvec = tuple(int(value) for value in model.action_space.nvec)
    if len(nvec) != 2:
        raise RuntimeError("expected factorized verb and aim heads")
    return flat_logits[:, : nvec[0]], flat_logits[:, nvec[0] :]


def _circular_bin_distance(
    predicted: np.ndarray, expected: np.ndarray, *, bins: int = AIM_BINS
) -> np.ndarray:
    distance = np.abs(
        predicted.astype(np.int64) - expected.astype(np.int64)
    )
    return np.minimum(distance, bins - distance)


def _train_park_settle_batch(
    *,
    model: Any,
    dataset: ParkSettleDataset,
    annotations: ParkSettleAnnotations,
    rows: np.ndarray,
    loss: ParkSettleLossConfig,
) -> dict[str, Any]:
    """One optimizer step of the park-and-settle objective.

    Aim: plain categorical CE over all rows, weighted by ``aim_weights``.
    Verb: CE over exact-runtime-mask logits on mask-legal teacher rows only.
    """

    import torch
    import torch.nn.functional as functional

    rows = np.asarray(rows, dtype=np.int64)
    observations = dataset.observations[rows]
    actions = dataset.raw_actions[rows]
    masks = dataset.masks[rows]
    weights = annotations.aim_weights[rows]
    legal = annotations.verb_legal[rows]
    window = annotations.pre_fire_window[rows]

    observation_tensor, _ = model.policy.obs_to_tensor(observations)
    action_tensor = torch.as_tensor(
        actions, dtype=torch.long, device=model.device
    )
    verb_mask_tensor = torch.as_tensor(
        masks[:, :VERB_COUNT], dtype=torch.bool, device=model.device
    )
    weight_tensor = torch.as_tensor(
        weights, dtype=torch.float32, device=model.device
    )
    legal_tensor = torch.as_tensor(
        legal, dtype=torch.bool, device=model.device
    )

    model.policy.set_training_mode(True)
    verb_logits, aim_logits = _policy_logits(model, observation_tensor)
    masked_verb_logits = verb_logits.masked_fill(
        ~verb_mask_tensor, VERB_MASK_FILL
    )
    legal_count = int(np.sum(legal))
    if legal_count:
        class_weight = None
        if loss.verb_class_weights is not None:
            class_weight = torch.as_tensor(
                loss.verb_class_weights,
                dtype=verb_logits.dtype,
                device=model.device,
            )
        verb_loss = functional.cross_entropy(
            masked_verb_logits[legal_tensor],
            action_tensor[legal_tensor, 0],
            weight=class_weight,
        )
    else:
        verb_loss = verb_logits.sum() * 0.0
    aim_nll = functional.cross_entropy(
        aim_logits, action_tensor[:, 1], reduction="none"
    )
    weight_total = weight_tensor.sum().clamp_min(1.0e-12)
    aim_loss = (weight_tensor * aim_nll).sum() / weight_total
    total_loss = (
        loss.verb_loss_weight * verb_loss + loss.aim_loss_weight * aim_loss
    )

    model.policy.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.policy.parameters(), float(loss.max_grad_norm)
    )
    model.policy.optimizer.step()
    model.policy.set_training_mode(False)

    with torch.no_grad():
        verb_after, aim_after = _policy_logits(model, observation_tensor)
        masked_after = verb_after.masked_fill(
            ~verb_mask_tensor, VERB_MASK_FILL
        )
        predicted_verbs = (
            masked_after.argmax(dim=1).detach().cpu().numpy()
        )
        predicted_aims = aim_after.argmax(dim=1).detach().cpu().numpy()
    all_legal = bool(
        np.all(masks[np.arange(len(rows)), predicted_verbs])
    )
    verb_accuracy = (
        float(np.mean(predicted_verbs[legal] == actions[legal, 0]))
        if legal_count
        else None
    )
    distance = _circular_bin_distance(predicted_aims, actions[:, 1])
    window_count = int(np.sum(window))
    return {
        "loss": float(total_loss.detach().cpu()),
        "verb_loss": float(verb_loss.detach().cpu()),
        "aim_loss": float(aim_loss.detach().cpu()),
        "gradient_norm": float(
            torch.as_tensor(gradient_norm).detach().cpu()
        ),
        "sample_count": int(len(rows)),
        "verb_loss_sample_count": legal_count,
        "dropped_illegal_verb_count": int(len(rows)) - legal_count,
        "aim_loss_sample_count": int(len(rows)),
        "aim_weight_sum": float(np.sum(weights)),
        "pre_fire_window_sample_count": window_count,
        "verb_accuracy": verb_accuracy,
        "aim_exact_accuracy": float(np.mean(distance == 0)),
        "aim_within_three_accuracy": float(np.mean(distance <= 3)),
        "pre_fire_window_aim_exact_accuracy": (
            float(np.mean(distance[window] == 0)) if window_count else None
        ),
        "masked_argmax_all_mask_legal": all_legal,
    }


def _mean_or_none(values: Sequence[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def optimize_park_settle(
    *,
    model: Any,
    dataset: ParkSettleDataset,
    annotations: ParkSettleAnnotations,
    indices: np.ndarray,
    train_config: ParkSettleTrainConfig,
    run_dir: Path | None = None,
    status_callback: (
        Callable[[int, list[dict[str, Any]], list[dict[str, Any]]], None]
        | None
    ) = None,
) -> dict[str, Any]:
    """Uniform shuffled minibatch epochs -- NO verb batch rebalancing."""

    train_config.validate()
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise RuntimeError("park-settle training split is empty")
    rng = np.random.default_rng(int(train_config.shuffle_seed) + 104_729)
    batch_size = int(train_config.batch_size)
    updates_per_epoch = math.ceil(indices.size / batch_size)
    epoch_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for epoch in range(int(train_config.epochs)):
        order = rng.permutation(indices)
        batches: list[dict[str, Any]] = []
        for start in range(0, order.size, batch_size):
            metrics = _train_park_settle_batch(
                model=model,
                dataset=dataset,
                annotations=annotations,
                rows=order[start : start + batch_size],
                loss=train_config.loss,
            )
            metrics = settled._finite_metrics(metrics)
            metrics["batch_index"] = start // batch_size
            batches.append(metrics)
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "updates": len(batches),
                "mean_loss": _mean_or_none(
                    [row["loss"] for row in batches]
                ),
                "mean_verb_loss": _mean_or_none(
                    [row["verb_loss"] for row in batches]
                ),
                "mean_aim_loss": _mean_or_none(
                    [row["aim_loss"] for row in batches]
                ),
                "mean_verb_accuracy": _mean_or_none(
                    [row["verb_accuracy"] for row in batches]
                ),
                "mean_aim_exact_accuracy": _mean_or_none(
                    [row["aim_exact_accuracy"] for row in batches]
                ),
                "mean_aim_within_three_accuracy": _mean_or_none(
                    [row["aim_within_three_accuracy"] for row in batches]
                ),
                "last_batch": batches[-1],
            }
        )
        interval = int(train_config.checkpoint_interval_epochs)
        if run_dir is not None and interval > 0 and (epoch + 1) % interval == 0:
            checkpoint = run_dir / f"epoch_{epoch + 1:02d}_model.zip"
            model.save(checkpoint)
            checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "path": str(checkpoint),
                    "sha256": legacy._sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                }
            )
        if status_callback is not None:
            status_callback(epoch + 1, epoch_rows, checkpoints)
    return {
        "updates": sum(int(row["updates"]) for row in epoch_rows),
        "samples": int(indices.size),
        "updates_per_epoch": updates_per_epoch,
        "epochs": epoch_rows,
        "checkpoints": checkpoints,
        "sampling": "uniform_shuffle_no_verb_rebalance",
        "aim_loss_scope": "all_ticks_park_and_settle",
        "aim_loss_mode": "categorical",
    }


def split_holdout(
    dataset: ParkSettleDataset,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split rows for calibration; by episode when temporal, else by row."""

    if not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be within [0, 1)")
    count = dataset.sample_count
    if dataset.temporal_structure == "episode_ticks":
        units = np.unique(dataset.episode_indices)
        unit_of_row = dataset.episode_indices
        mode = "episode"
    else:
        units = np.arange(count, dtype=np.int64)
        unit_of_row = np.arange(count, dtype=np.int64)
        mode = "row"
    if float(holdout_fraction) == 0.0 or units.size < 2:
        train = np.arange(count, dtype=np.int64)
        holdout = np.empty(0, dtype=np.int64)
        holdout_units = np.empty(0, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        permuted = rng.permutation(units)
        chosen = int(round(float(holdout_fraction) * units.size))
        chosen = min(max(chosen, 1), units.size - 1)
        holdout_units = permuted[:chosen]
        holdout_mask = np.isin(unit_of_row, holdout_units)
        holdout = np.nonzero(holdout_mask)[0].astype(np.int64)
        train = np.nonzero(~holdout_mask)[0].astype(np.int64)
    return train, holdout, {
        "mode": mode,
        "holdout_fraction": float(holdout_fraction),
        "seed": int(seed),
        "total_units": int(units.size),
        "holdout_units": int(holdout_units.size),
        "train_rows": int(train.size),
        "holdout_rows": int(holdout.size),
    }


def _aim_group_metrics(
    distance: np.ndarray, group: np.ndarray
) -> dict[str, Any]:
    count = int(np.sum(group))
    if not count:
        return {
            "sample_count": 0,
            "exact_accuracy": None,
            "within_three_accuracy": None,
        }
    grouped = distance[group]
    return {
        "sample_count": count,
        "exact_accuracy": float(np.mean(grouped == 0)),
        "within_three_accuracy": float(np.mean(grouped <= 3)),
    }


def calibrate_park_settle(
    *,
    model: Any,
    dataset: ParkSettleDataset,
    annotations: ParkSettleAnnotations,
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Deterministic-policy calibration on one slice.

    Reports the student's deterministic fire rate (masked verb argmax)
    against the teacher's executed fire rate, and plain-argmax aim accuracy
    on fire-commit ticks and on pre-fire-window ticks separately.
    """

    import torch

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("calibration slice is empty")
    predicted_verbs = np.empty(indices.size, dtype=np.int64)
    predicted_aims = np.empty(indices.size, dtype=np.int64)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, indices.size, int(batch_size)):
            rows = indices[start : start + int(batch_size)]
            observation_tensor, _ = model.policy.obs_to_tensor(
                dataset.observations[rows]
            )
            verb_logits, aim_logits = _policy_logits(
                model, observation_tensor
            )
            verb_mask_tensor = torch.as_tensor(
                dataset.masks[rows, :VERB_COUNT],
                dtype=torch.bool,
                device=model.device,
            )
            masked_verbs = verb_logits.masked_fill(
                ~verb_mask_tensor, VERB_MASK_FILL
            )
            stop = start + len(rows)
            predicted_verbs[start:stop] = (
                masked_verbs.argmax(dim=1).detach().cpu().numpy()
            )
            predicted_aims[start:stop] = (
                aim_logits.argmax(dim=1).detach().cpu().numpy()
            )
    actions = dataset.raw_actions[indices]
    masks = dataset.masks[indices]
    all_legal = bool(
        np.all(masks[np.arange(indices.size), predicted_verbs])
    )
    student_fire_rate = float(np.mean(predicted_verbs == FIRE_VERB))
    teacher_executed = float(np.mean(annotations.fire_commit[indices]))
    ratio = (
        student_fire_rate / teacher_executed if teacher_executed > 0.0
        else None
    )
    distance = _circular_bin_distance(predicted_aims, actions[:, 1])
    commit = annotations.fire_commit[indices]
    window_only = annotations.pre_fire_window[indices] & ~commit
    return {
        "sample_count": int(indices.size),
        "deterministic_policy": (
            "masked_verb_argmax_plus_plain_aim_argmax"
        ),
        "deterministic_verbs_all_mask_legal": all_legal,
        "student_deterministic_fire_rate": student_fire_rate,
        "teacher_executed_fire_rate": teacher_executed,
        "teacher_fire_intent_rate": float(
            np.mean(annotations.fire_intent[indices])
        ),
        "teacher_fire_edge_rate": float(
            np.mean(annotations.fire_edge[indices])
        ),
        "student_teacher_fire_rate_ratio": ratio,
        "fire_cycle_ticks": FIRE_CYCLE_TICKS,
        "fire_cycle_upper_bound_fire_rate": 1.0 / FIRE_CYCLE_TICKS,
        "fire_commit_aim": _aim_group_metrics(distance, commit),
        "pre_fire_window_aim": _aim_group_metrics(distance, window_only),
        "all_ticks_aim": _aim_group_metrics(
            distance, np.ones(indices.size, dtype=np.bool_)
        ),
    }


def _fraction(flags: np.ndarray, indices: np.ndarray) -> float | None:
    if indices.size == 0:
        return None
    return float(np.mean(~flags[indices]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def execute_training(
    *,
    model: Any,
    dataset: ParkSettleDataset,
    run_dir: Path,
    model_config: ParkSettleModelConfig,
    train_config: ParkSettleTrainConfig,
    dataset_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Offline park-and-settle training with a receipted completion."""

    model_config.validate()
    train_config.validate()
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"park-settle run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    if dataset_receipt is None:
        dataset_receipt = {
            "source": "in_memory",
            "sample_count": dataset.sample_count,
            "temporal_structure": dataset.temporal_structure,
        }
    legacy._write_json_atomic(
        run_dir / "config.json",
        _jsonable(
            {
                "schema": f"{SCHEMA_PREFIX}-config",
                "version": 1,
                "created_utc": legacy._utc_now(),
                "trainer": {
                    "path": str(SCRIPT_PATH),
                    "sha256": legacy._sha256(SCRIPT_PATH),
                },
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "dataset": dataset_receipt,
                "formal_seed_consumption": False,
                "formal_candidate_authority": False,
            }
        ),
    )
    try:
        annotations = annotate_park_settle(dataset, train_config.loss)
        train_indices, holdout_indices, split = split_holdout(
            dataset,
            holdout_fraction=float(train_config.holdout_fraction),
            seed=int(train_config.shuffle_seed),
        )

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            legacy._write_json_atomic(
                run_dir / "training_status.json",
                _jsonable(
                    {
                        "schema": f"{SCHEMA_PREFIX}-status",
                        "version": 1,
                        "status": "RUNNING",
                        "stage": "OPTIMIZING",
                        "updated_utc": legacy._utc_now(),
                        "completed_epochs": completed_epochs,
                        "expected_epochs": int(train_config.epochs),
                        "epochs": epoch_rows,
                        "checkpoints": checkpoints,
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
            )

        write_status(0, [], [])
        optimization = optimize_park_settle(
            model=model,
            dataset=dataset,
            annotations=annotations,
            indices=train_indices,
            train_config=train_config,
            run_dir=run_dir,
            status_callback=write_status,
        )
        calibration_indices = (
            holdout_indices if holdout_indices.size else train_indices
        )
        calibration = calibrate_park_settle(
            model=model,
            dataset=dataset,
            annotations=annotations,
            indices=calibration_indices,
            batch_size=int(train_config.batch_size),
        )
        calibration["slice"] = (
            "holdout" if holdout_indices.size else "train_no_holdout"
        )
        final_model = run_dir / "final_model.zip"
        model.save(final_model)
        completion = {
            "schema": f"{SCHEMA_PREFIX}-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": legacy._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": POLICY_ARCHITECTURE,
            "trainer": {
                "path": str(SCRIPT_PATH),
                "sha256": legacy._sha256(SCRIPT_PATH),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": str(train_config.device),
            },
            "policy_parameter_count": sum(
                parameter.numel()
                for parameter in model.policy.parameters()
            ),
            "dataset": dataset_receipt,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "recipe": {
                "aim_loss_scope": "all_ticks_park_and_settle",
                "aim_loss_mode": "categorical",
                "aim_label_semantics": (
                    "teacher_commanded_aim_bin_every_tick_"
                    "wait_carries_parked_target"
                ),
                "pre_fire_window_ticks": int(
                    train_config.loss.fire_window_ticks
                ),
                "wait_aim_weight": float(
                    train_config.loss.wait_aim_weight
                ),
                "verb_batch_rebalancing": "none",
                "verb_class_weights": train_config.loss.verb_class_weights,
                "verb_mask_semantics": "exact_runtime_mask_on_logits",
                "illegal_teacher_verb_rows": (
                    "dropped_from_verb_loss_kept_for_aim_loss"
                ),
            },
            "temporal_structure": dataset.temporal_structure,
            "split": split,
            "illegal_teacher_verb_fraction": {
                "overall": _fraction(
                    annotations.verb_legal,
                    np.arange(dataset.sample_count, dtype=np.int64),
                ),
                "train": _fraction(annotations.verb_legal, train_indices),
                "calibration_slice": _fraction(
                    annotations.verb_legal, calibration_indices
                ),
            },
            "masked_aim_label_rows": int(
                np.sum(~annotations.aim_label_legal)
            ),
            "optimization": optimization,
            "calibration": calibration,
            "final_model": {
                "path": str(final_model),
                "sha256": legacy._sha256(final_model),
                "bytes": final_model.stat().st_size,
            },
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        legacy._write_json_atomic(
            run_dir / "completion.json", _jsonable(completion)
        )
        return completion
    except BaseException as error:
        legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": f"{SCHEMA_PREFIX}-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": legacy._utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
            },
        )
        raise


def _first_key(
    archive: Any, keys: Sequence[str], *, required: bool
) -> str | None:
    for key in keys:
        if key in archive.files:
            return key
    if required:
        raise ValueError(
            f"NPZ shard offers none of {tuple(keys)}: {sorted(archive.files)}"
        )
    return None


def load_replay_dataset(
    manifest_path: Path,
) -> tuple[ParkSettleDataset, dict[str, Any]]:
    """Load a replay dataset manifest (replay-v2 or coverage-v4 layout)."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = legacy._read_json(manifest_path)
    if "shards" in manifest:
        entries = list(manifest["shards"])
        layout = "motor-observable-replay-v2-shards"
        entry_is_episode = False
    elif "episodes" in manifest:
        entries = list(manifest["episodes"])
        layout = "motor-observable-replay-v4-coverage-episodes"
        entry_is_episode = True
    elif "files" in manifest:
        entries = list(manifest["files"])
        layout = "generic-files"
        entry_is_episode = False
    else:
        raise ValueError(
            "replay manifest needs a 'shards', 'episodes', or 'files' list"
        )
    if not entries:
        raise ValueError(f"replay manifest lists no data: {manifest_path}")
    declared_semantics = manifest.get("mask_semantics")
    if declared_semantics is not None and not isinstance(
        declared_semantics, dict
    ):
        raise ValueError(
            "manifest mask_semantics must map NPZ keys to semantics strings"
        )

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    tick_indices: list[np.ndarray] = []
    entry_receipts: list[dict[str, Any]] = []
    temporal = True
    episode_offset = 0
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(
                f"manifest entry {ordinal} lacks a 'path': {entry!r}"
            )
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve(strict=True)
        digest = legacy._sha256(path)
        declared = entry.get("sha256")
        if declared is not None and declared != digest:
            raise ValueError(f"shard bytes differ from manifest: {path}")
        with np.load(path) as archive:
            observation_key = _first_key(
                archive, _OBSERVATION_KEYS, required=True
            )
            action_key = _first_key(archive, _ACTION_KEYS, required=True)
            mask_key = _first_key(archive, _MASK_KEYS, required=False)
            if mask_key is None:
                relaxed = [
                    key for key in _RELAXED_MASK_KEYS if key in archive.files
                ]
                if relaxed:
                    raise ValueError(
                        f"NPZ shard offers only relaxed training masks "
                        f"{relaxed}; the park-settle trainer requires the "
                        f"exact runtime valid-action mask (one of "
                        f"{_MASK_KEYS}): {path}"
                    )
                raise ValueError(
                    f"NPZ shard offers none of {_MASK_KEYS}: "
                    f"{sorted(archive.files)}"
                )
            if declared_semantics is not None:
                semantics = declared_semantics.get(mask_key)
                if semantics not in _EXACT_MASK_SEMANTICS:
                    raise ValueError(
                        f"manifest declares mask semantics {semantics!r} "
                        f"for NPZ key '{mask_key}'; the park-settle "
                        f"trainer requires one of "
                        f"{sorted(_EXACT_MASK_SEMANTICS)}"
                    )
            episode_key = _first_key(
                archive, _EPISODE_KEYS, required=False
            )
            tick_key = _first_key(archive, _TICK_KEYS, required=False)
            shard_observations = np.asarray(
                archive[observation_key], dtype=np.float32
            )
            shard_actions = np.asarray(archive[action_key], dtype=np.int64)
            shard_masks = np.asarray(archive[mask_key], dtype=np.bool_)
            rows = int(shard_observations.shape[0])
            if episode_key is not None:
                local = np.asarray(archive[episode_key], dtype=np.int64)
                _, remapped = np.unique(local, return_inverse=True)
                shard_episodes = episode_offset + remapped.astype(np.int64)
                episode_offset += int(remapped.max()) + 1 if rows else 0
            elif entry_is_episode:
                shard_episodes = np.full(rows, episode_offset, np.int64)
                episode_offset += 1
            else:
                shard_episodes = None
            if tick_key is not None:
                shard_ticks = np.asarray(archive[tick_key], dtype=np.int64)
            elif entry_is_episode:
                shard_ticks = np.arange(rows, dtype=np.int64)
            else:
                shard_ticks = None
        if shard_episodes is None or shard_ticks is None:
            temporal = False
            shard_episodes = np.zeros(rows, dtype=np.int64)
            shard_ticks = np.zeros(rows, dtype=np.int64)
        observations.append(shard_observations)
        actions.append(shard_actions)
        masks.append(shard_masks)
        episode_indices.append(shard_episodes)
        tick_indices.append(shard_ticks)
        entry_receipts.append(
            {
                "path": str(path),
                "sha256": digest,
                "rows": rows,
                "keys": {
                    "observations": observation_key,
                    "actions": action_key,
                    "masks": mask_key,
                    "episode_indices": episode_key,
                    "tick_indices": tick_key,
                },
            }
        )

    if temporal:
        dataset = ParkSettleDataset.from_episode_rows(
            observations=np.concatenate(observations, axis=0),
            raw_actions=np.concatenate(actions, axis=0),
            masks=np.concatenate(masks, axis=0),
            episode_indices=np.concatenate(episode_indices, axis=0),
            tick_indices=np.concatenate(tick_indices, axis=0),
        )
    else:
        count = int(sum(block.shape[0] for block in observations))
        dataset = ParkSettleDataset(
            observations=np.concatenate(observations, axis=0),
            raw_actions=np.concatenate(actions, axis=0),
            masks=np.concatenate(masks, axis=0),
            episode_indices=np.arange(count, dtype=np.int64),
            tick_indices=np.zeros(count, dtype=np.int64),
            temporal_structure="absent",
        )
        dataset.validate()
    receipt = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": legacy._sha256(manifest_path),
            "schema": manifest.get("schema"),
            "version": manifest.get("version"),
        },
        "layout": layout,
        "mask_semantics": (
            dict(declared_semantics)
            if declared_semantics is not None
            else None
        ),
        "entries": entry_receipts,
        "sample_count": dataset.sample_count,
        "temporal_structure": dataset.temporal_structure,
    }
    return dataset, receipt


def _launch_prototype(
    *, original_root: Path, max_ticks: int, observation_width: int
) -> Any:
    """Open the environment whose actor interface matches the dataset."""

    bare = polar._prototype_environment(
        original_root=original_root, max_ticks=max_ticks
    )
    if int(bare.observation_space.shape[0]) == int(observation_width):
        return bare
    bare.close()
    from tools import (
        distill_alphazuma_55_motor_observable_replay_v2 as replay_v2,
    )

    motor = replay_v2._prototype(
        original_root=original_root, max_ticks=max_ticks
    )
    if int(motor.observation_space.shape[0]) == int(observation_width):
        return motor
    motor.close()
    raise ValueError(
        "dataset observation width matches neither the bare full55 "
        f"interface nor the motor-observable interface: {observation_width}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--shuffle-seed", type=int, default=1_543_003_101)
    parser.add_argument("--model-seed", type=int, default=1_543_003_100)
    parser.add_argument(
        "--learned-features-dim", type=int, default=1024
    )
    parser.add_argument(
        "--aim-head-identity-scale", type=float, default=5.0
    )
    parser.add_argument("--fire-window-ticks", type=int, default=24)
    parser.add_argument("--wait-aim-weight", type=float, default=0.25)
    parser.add_argument("--verb-loss-weight", type=float, default=1.0)
    parser.add_argument("--aim-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--verb-class-weights",
        type=float,
        nargs=VERB_COUNT,
        default=None,
        metavar=tuple(name.upper() for name in VERB_NAMES),
    )
    parser.add_argument(
        "--checkpoint-interval-epochs", type=int, default=1
    )
    parser.add_argument("--max-ticks", type=int, default=30_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset, receipt = load_replay_dataset(
        args.dataset_manifest.expanduser()
    )
    if args.validate_only:
        print(
            json.dumps(
                _jsonable({"status": "VALID", "dataset": receipt}),
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
    loss_config = ParkSettleLossConfig(
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
    train_config = ParkSettleTrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        holdout_fraction=float(args.holdout_fraction),
        shuffle_seed=int(args.shuffle_seed),
        device=str(args.device),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        loss=loss_config,
    )
    model_config = ParkSettleModelConfig(
        learned_features_dim=int(args.learned_features_dim),
        aim_head_identity_scale=float(args.aim_head_identity_scale),
        model_seed=int(args.model_seed),
    )
    prototype = _launch_prototype(
        original_root=args.original_root.expanduser().resolve(strict=True),
        max_ticks=int(args.max_ticks),
        observation_width=int(dataset.observations.shape[1]),
    )
    try:
        policy_kwargs = entity_polar_policy_kwargs(
            prototype,
            learned_features_dim=int(model_config.learned_features_dim),
        )
        observation_space = prototype.observation_space
        action_space = prototype.action_space
    finally:
        prototype.close()
    model = build_student_model(
        observation_space=observation_space,
        action_space=action_space,
        policy_kwargs=policy_kwargs,
        model_config=model_config,
        train_config=train_config,
    )
    completion = execute_training(
        model=model,
        dataset=dataset,
        run_dir=args.run_dir.expanduser(),
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt,
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "status": completion["status"],
                    "final_model": completion["final_model"],
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
