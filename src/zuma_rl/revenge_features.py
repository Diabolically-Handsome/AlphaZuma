"""Entity-aware neural features for packed Zuma actor observations."""

from __future__ import annotations

import math
from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from zuma_rl.revenge_env import RevengeEnv


class RevengeEntityFeatureExtractor(BaseFeaturesExtractor):
    """Encode balls as an ordered entity sequence with conditional attention."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        max_balls: int,
        ball_feature_size: int,
        balls_start: int,
        balls_stop: int,
        max_projectiles: int,
        projectile_feature_size: int,
        projectiles_start: int,
        projectiles_stop: int,
        global_feature_size: int,
        globals_start: int,
        globals_stop: int,
        relational: bool = False,
        num_colors: int = 0,
        ball_color_start: int = 0,
        ball_exploding_index: int = 0,
        ball_curve_start: int = 0,
        curve_feature_count: int = 0,
        global_current_color_start: int = 0,
        global_next_color_start: int = 0,
        global_shooter_x_index: int = -1,
        global_shooter_y_index: int = -1,
        polar_aim_bins: int = 0,
        polar_sigma_bins: float = 0.75,
        shooter_x_normalized: float = 0.0,
        shooter_y_normalized: float = 0.0,
        logical_width: float = 1.0,
        logical_height: float = 1.0,
        features_dim: int = 256,
    ):
        polar_aim_bins = int(polar_aim_bins)
        super().__init__(
            observation_space,
            features_dim=features_dim + max(0, polar_aim_bins),
        )
        expected = int(observation_space.shape[0])
        if not (
            0 <= balls_start < balls_stop <= projectiles_start
            < projectiles_stop <= globals_start < globals_stop == expected
        ):
            raise ValueError("entity feature slices are inconsistent")
        if balls_stop - balls_start != max_balls * ball_feature_size:
            raise ValueError("ball slice does not match its declared layout")
        if (
            projectiles_stop - projectiles_start
            != max_projectiles * projectile_feature_size
        ):
            raise ValueError("projectile slice does not match its declared layout")
        if globals_stop - globals_start != global_feature_size:
            raise ValueError("global slice does not match its declared layout")
        self.max_balls = int(max_balls)
        self.ball_feature_size = int(ball_feature_size)
        self.balls_start = int(balls_start)
        self.balls_stop = int(balls_stop)
        self.max_projectiles = int(max_projectiles)
        self.projectile_feature_size = int(projectile_feature_size)
        self.projectiles_start = int(projectiles_start)
        self.projectiles_stop = int(projectiles_stop)
        self.global_feature_size = int(global_feature_size)
        self.globals_start = int(globals_start)
        self.globals_stop = int(globals_stop)
        self.relational = bool(relational)
        self.num_colors = int(num_colors)
        self.ball_color_start = int(ball_color_start)
        self.ball_exploding_index = int(ball_exploding_index)
        self.ball_curve_start = int(ball_curve_start)
        self.curve_feature_count = int(curve_feature_count)
        self.global_current_color_start = int(global_current_color_start)
        self.global_next_color_start = int(global_next_color_start)
        self.global_shooter_x_index = int(global_shooter_x_index)
        self.global_shooter_y_index = int(global_shooter_y_index)
        self.learned_features_dim = int(features_dim)
        self.polar_aim_bins = polar_aim_bins
        self.polar_sigma_bins = float(polar_sigma_bins)
        self.shooter_x_normalized = float(shooter_x_normalized)
        self.shooter_y_normalized = float(shooter_y_normalized)
        self.logical_width = float(logical_width)
        self.logical_height = float(logical_height)
        if self.polar_aim_bins < 0:
            raise ValueError("polar aim bins must be non-negative")
        if self.polar_aim_bins and not self.relational:
            raise ValueError("polar aim features require relational features")
        if self.polar_aim_bins and (
            not math.isfinite(self.polar_sigma_bins)
            or self.polar_sigma_bins <= 0.0
        ):
            raise ValueError("polar aim sigma must be positive")
        if not (
            math.isfinite(self.shooter_x_normalized)
            and math.isfinite(self.shooter_y_normalized)
        ):
            raise ValueError("normalized shooter coordinates must be finite")
        if self.polar_aim_bins and (
            not math.isfinite(self.logical_width)
            or not math.isfinite(self.logical_height)
            or self.logical_width <= 0.0
            or self.logical_height <= 0.0
        ):
            raise ValueError("logical canvas dimensions must be positive")
        dynamic_shooter = (
            self.global_shooter_x_index >= 0
            or self.global_shooter_y_index >= 0
        )
        if dynamic_shooter and not (
            0 <= self.global_shooter_x_index < self.global_feature_size
            and 0 <= self.global_shooter_y_index < self.global_feature_size
        ):
            raise ValueError(
                "dynamic shooter coordinates must be paired global indices"
            )
        if dynamic_shooter:
            # Zero is an exact legacy-function anchor.  Fine-tuning can move
            # tanh(mix) toward one and progressively use the actor-visible
            # active lily-pad coordinates without discarding the old policy.
            self.dynamic_shooter_mix = nn.Parameter(torch.zeros(()))
        if self.relational:
            if self.num_colors < 1 or self.curve_feature_count < 1:
                raise ValueError(
                    "relational features require colors and curve channels"
                )
            if not (
                0
                <= self.ball_color_start
                < self.ball_color_start + self.num_colors
                <= self.ball_feature_size
            ):
                raise ValueError("ball color slice is inconsistent")
            if not 0 <= self.ball_exploding_index < self.ball_feature_size:
                raise ValueError("ball exploding index is inconsistent")
            if not (
                0
                <= self.ball_curve_start
                < self.ball_curve_start + self.curve_feature_count
                <= self.ball_feature_size
            ):
                raise ValueError("ball curve slice is inconsistent")
            if not (
                0
                <= self.global_current_color_start
                < self.global_current_color_start + self.num_colors
                <= self.global_feature_size
            ):
                raise ValueError("current color slice is inconsistent")
            if not (
                0
                <= self.global_next_color_start
                < self.global_next_color_start + self.num_colors
                <= self.global_feature_size
            ):
                raise ValueError("next color slice is inconsistent")

        relational_ball_features = 6 if self.relational else 0
        self.ball_encoder = nn.Sequential(
            nn.Linear(ball_feature_size + relational_ball_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.ball_neighbour = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.ball_query = nn.Sequential(
            nn.Linear(global_feature_size, 64),
            nn.Tanh(),
        )
        self.projectile_encoder = nn.Sequential(
            nn.Linear(projectile_feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        relational_fusion_features = 64 + 64 + 8 if self.relational else 0
        self.fusion = nn.Sequential(
            nn.Linear(
                64 + 64 + 32 + 32 + 64 + relational_fusion_features,
                features_dim,
            ),
            nn.ReLU(),
        )

    @staticmethod
    def _masked_mean_and_max(
        entities: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = present.to(dtype=entities.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (entities * weights).sum(dim=1) / count
        masked = entities.masked_fill(~present.unsqueeze(-1), -torch.inf)
        maximum = masked.max(dim=1).values
        any_present = present.any(dim=1, keepdim=True)
        maximum = torch.where(any_present, maximum, torch.zeros_like(maximum))
        return mean, maximum

    @staticmethod
    def _masked_attention(
        entities: torch.Tensor,
        query: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            entities * query.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(entities.shape[-1])
        logits = logits.masked_fill(~present, -torch.inf)
        any_present = present.any(dim=1, keepdim=True)
        safe_logits = torch.where(
            any_present,
            logits,
            torch.zeros_like(logits),
        )
        attention = torch.softmax(safe_logits, dim=1) * present.to(
            dtype=entities.dtype
        )
        attention = attention / attention.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return (entities * attention.unsqueeze(-1)).sum(dim=1)

    def _ball_relations(
        self,
        balls: torch.Tensor,
        globals_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add explicit shooter-color and adjacent-run relations."""

        present = balls[:, :, 0] > 0.5
        eligible = present & (
            balls[:, :, self.ball_exploding_index] <= 0.5
        )
        colors = balls[
            :,
            :,
            self.ball_color_start : self.ball_color_start + self.num_colors,
        ]
        current_color = globals_[
            :,
            self.global_current_color_start : (
                self.global_current_color_start + self.num_colors
            ),
        ]
        next_color = globals_[
            :,
            self.global_next_color_start : (
                self.global_next_color_start + self.num_colors
            ),
        ]
        current_match = (
            (colors * current_color.unsqueeze(1)).sum(dim=-1) > 0.5
        ) & eligible
        next_match = (
            (colors * next_color.unsqueeze(1)).sum(dim=-1) > 0.5
        ) & eligible

        curves = balls[
            :,
            :,
            self.ball_curve_start : (
                self.ball_curve_start + self.curve_feature_count
            ),
        ]
        adjacent = (
            eligible[:, :-1]
            & eligible[:, 1:]
            & ((colors[:, :-1] * colors[:, 1:]).sum(dim=-1) > 0.5)
            & ((curves[:, :-1] * curves[:, 1:]).sum(dim=-1) > 0.5)
        )
        same_left = torch.zeros_like(eligible)
        same_right = torch.zeros_like(eligible)
        same_left[:, 1:] = adjacent
        same_right[:, :-1] = adjacent
        has_same_neighbour = same_left | same_right
        current_run = current_match & has_same_neighbour
        next_run = next_match & has_same_neighbour
        local_span = (
            1.0
            + same_left.to(dtype=balls.dtype)
            + same_right.to(dtype=balls.dtype)
        )
        relation_channels = torch.stack(
            (
                current_match,
                next_match,
                same_left,
                same_right,
                current_run,
                next_run,
            ),
            dim=-1,
        ).to(dtype=balls.dtype)
        augmented = torch.cat((balls, relation_channels), dim=-1)

        eligible_count = eligible.sum(dim=1).clamp_min(1).to(dtype=balls.dtype)
        summary = torch.stack(
            (
                current_match.any(dim=1).to(dtype=balls.dtype),
                next_match.any(dim=1).to(dtype=balls.dtype),
                current_run.any(dim=1).to(dtype=balls.dtype),
                next_run.any(dim=1).to(dtype=balls.dtype),
                (
                    local_span * current_match.to(dtype=balls.dtype)
                ).max(dim=1).values
                / 3.0,
                (
                    local_span * next_match.to(dtype=balls.dtype)
                ).max(dim=1).values
                / 3.0,
                current_match.sum(dim=1).to(dtype=balls.dtype)
                / eligible_count,
                next_match.sum(dim=1).to(dtype=balls.dtype)
                / eligible_count,
            ),
            dim=1,
        )
        return augmented, summary, current_match, next_match

    def _polar_aim_basis(
        self,
        balls: torch.Tensor,
        globals_: torch.Tensor,
        current_match: torch.Tensor,
        current_run: torch.Tensor,
    ) -> torch.Tensor:
        """Build a circular aim-bin map from actor-visible ball coordinates."""

        if not self.polar_aim_bins:
            raise RuntimeError("polar aim basis is disabled")
        eligible = (balls[:, :, 0] > 0.5) & (
            balls[:, :, self.ball_exploding_index] <= 0.5
        )
        use_current = current_match.any(dim=1, keepdim=True)
        target_mask = torch.where(use_current, current_match, eligible)
        if self.global_shooter_x_index >= 0:
            dynamic_x = globals_[
                :, self.global_shooter_x_index
            ].unsqueeze(1)
            dynamic_y = globals_[
                :, self.global_shooter_y_index
            ].unsqueeze(1)
            mix = torch.tanh(self.dynamic_shooter_mix)
            shooter_x = self.shooter_x_normalized + mix * (
                dynamic_x - self.shooter_x_normalized
            )
            shooter_y = self.shooter_y_normalized + mix * (
                dynamic_y - self.shooter_y_normalized
            )
        else:
            shooter_x = self.shooter_x_normalized
            shooter_y = self.shooter_y_normalized
        dx = (balls[:, :, 1] - shooter_x) * self.logical_width
        dy = (balls[:, :, 2] - shooter_y) * self.logical_height
        angle = torch.remainder(torch.atan2(dy, dx), math.tau)
        continuous_bin = (
            angle * self.polar_aim_bins / math.tau - 0.5
        )
        bin_indices = torch.arange(
            self.polar_aim_bins,
            device=balls.device,
            dtype=balls.dtype,
        ).reshape(1, 1, -1)
        distance = torch.abs(continuous_bin.unsqueeze(-1) - bin_indices)
        distance = torch.minimum(distance, self.polar_aim_bins - distance)
        basis = torch.exp(
            -0.5 * (distance / self.polar_sigma_bins) ** 2
        )
        priority = 1.0 + 2.0 * current_run.to(dtype=balls.dtype)
        weighted = basis * target_mask.to(dtype=balls.dtype).unsqueeze(-1)
        weighted = weighted * priority.unsqueeze(-1)
        return weighted.max(dim=1).values

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        balls = observations[:, self.balls_start : self.balls_stop].reshape(
            -1,
            self.max_balls,
            self.ball_feature_size,
        )
        projectiles = observations[
            :, self.projectiles_start : self.projectiles_stop
        ].reshape(
            -1,
            self.max_projectiles,
            self.projectile_feature_size,
        )
        globals_ = observations[:, self.globals_start : self.globals_stop]

        ball_present = balls[:, :, 0] > 0.5
        ball_weights = ball_present.to(dtype=observations.dtype).unsqueeze(-1)
        relation_summary: torch.Tensor | None = None
        current_match: torch.Tensor | None = None
        next_match: torch.Tensor | None = None
        raw_balls = balls
        if self.relational:
            balls, relation_summary, current_match, next_match = (
                self._ball_relations(balls, globals_)
            )
        ball_entities = self.ball_encoder(balls) * ball_weights
        neighbours = self.ball_neighbour(
            ball_entities.transpose(1, 2)
        ).transpose(1, 2)
        ball_entities = torch.relu(ball_entities + neighbours) * ball_weights
        query = self.ball_query(globals_)
        attended_balls = self._masked_attention(
            ball_entities,
            query,
            ball_present,
        )
        _, maximum_balls = self._masked_mean_and_max(
            ball_entities,
            ball_present,
        )

        projectile_present = projectiles[:, :, 0] > 0.5
        projectile_entities = self.projectile_encoder(projectiles)
        mean_projectiles, maximum_projectiles = self._masked_mean_and_max(
            projectile_entities,
            projectile_present,
        )
        global_features = self.global_encoder(globals_)
        relational_pieces: tuple[torch.Tensor, ...] = ()
        if self.relational:
            if (
                relation_summary is None
                or current_match is None
                or next_match is None
            ):
                raise RuntimeError("relational tensors were not constructed")
            relational_pieces = (
                self._masked_attention(ball_entities, query, current_match),
                self._masked_attention(ball_entities, query, next_match),
                relation_summary,
            )
        learned_features = self.fusion(
            torch.cat(
                (
                    attended_balls,
                    maximum_balls,
                    mean_projectiles,
                    maximum_projectiles,
                    global_features,
                    *relational_pieces,
                ),
                dim=1,
            )
        )
        if not self.polar_aim_bins:
            return learned_features
        if current_match is None:
            raise RuntimeError("polar aim basis requires current-color matches")
        current_run = balls[:, :, -2] > 0.5
        polar_basis = self._polar_aim_basis(
            raw_balls,
            globals_,
            current_match,
            current_run,
        )
        return torch.cat((learned_features, polar_basis), dim=1)


def revenge_entity_policy_kwargs(env: RevengeEnv) -> dict[str, Any]:
    """Return serializable layout kwargs plus the extractor class for SB3."""

    balls = env.observation_layout["balls"]
    projectiles = env.observation_layout["projectiles"]
    globals_ = env.observation_layout["globals"]
    return {
        "features_extractor_class": RevengeEntityFeatureExtractor,
        "features_extractor_kwargs": {
            "max_balls": int(env.config.max_balls),
            "ball_feature_size": int(env.ball_feature_size),
            "balls_start": int(balls.start or 0),
            "balls_stop": int(balls.stop or env.observation_size),
            "max_projectiles": int(env.config.max_projectiles),
            "projectile_feature_size": int(env.projectile_feature_size),
            "projectiles_start": int(projectiles.start or 0),
            "projectiles_stop": int(projectiles.stop or env.observation_size),
            "global_feature_size": int(env.global_feature_size),
            "globals_start": int(globals_.start or 0),
            "globals_stop": int(globals_.stop or env.observation_size),
            "features_dim": 256,
        },
        "net_arch": {"pi": [256], "vf": [256]},
    }


def revenge_relational_policy_kwargs(env: RevengeEnv) -> dict[str, Any]:
    """Return entity-policy kwargs with explicit color/run relations."""

    kwargs = revenge_entity_policy_kwargs(env)
    extractor_kwargs = dict(kwargs["features_extractor_kwargs"])
    ball_names = tuple(env.ball_feature_names)
    global_names = tuple(env.global_feature_names)
    extractor_kwargs.update(
        {
            "relational": True,
            "num_colors": int(env.color_feature_count),
            "ball_color_start": ball_names.index("color_0"),
            "ball_exploding_index": ball_names.index("exploding"),
            "ball_curve_start": ball_names.index("curve_0"),
            "curve_feature_count": len(env.curve_feature_names),
            "global_current_color_start": global_names.index(
                "current_color_0"
            ),
            "global_next_color_start": global_names.index("next_color_0"),
        }
    )
    kwargs["features_extractor_kwargs"] = extractor_kwargs
    return kwargs


def revenge_polar_policy_kwargs(env: RevengeEnv) -> dict[str, Any]:
    """Return relational features plus a direct actor-visible polar aim map."""

    kwargs = revenge_relational_policy_kwargs(env)
    extractor_kwargs = dict(kwargs["features_extractor_kwargs"])
    width = float(env.sim.config.logical_width)
    height = float(env.sim.config.logical_height)
    extractor_kwargs.update(
        {
            "polar_aim_bins": int(env.config.aim_bins),
            "polar_sigma_bins": 0.75,
            "shooter_x_normalized": 2.0 * float(env.sim.shooter[0]) / width - 1.0,
            "shooter_y_normalized": 2.0 * float(env.sim.shooter[1]) / height - 1.0,
            "logical_width": width,
            "logical_height": height,
        }
    )
    global_names = tuple(env.global_feature_names)
    if "shooter_x" in global_names and "shooter_y" in global_names:
        extractor_kwargs.update(
            {
                "global_shooter_x_index": global_names.index("shooter_x"),
                "global_shooter_y_index": global_names.index("shooter_y"),
            }
        )
    kwargs["features_extractor_kwargs"] = extractor_kwargs
    kwargs["net_arch"] = {"pi": [], "vf": [256]}
    return kwargs


def initialize_polar_aim_head(model: Any, *, scale: float = 5.0) -> None:
    """Initialize aim logits as a scaled identity over the polar feature map."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("polar aim-head scale must be positive")
    extractor = model.policy.features_extractor
    aim_bins = int(getattr(extractor, "polar_aim_bins", 0))
    learned = int(getattr(extractor, "learned_features_dim", 0))
    if aim_bins < 1:
        raise ValueError("model does not expose a polar aim basis")
    action_sizes = tuple(int(value) for value in model.action_space.nvec)
    if action_sizes not in {(3, aim_bins), (4, aim_bins)}:
        raise ValueError("polar aim head requires factorized verb and aim actions")
    verb_bins = action_sizes[0]
    action_net = model.policy.action_net
    if not isinstance(action_net, nn.Linear):
        raise TypeError("polar aim initialization requires a linear action head")
    if action_net.in_features != learned + aim_bins:
        raise ValueError("policy latent does not expose the direct polar basis")
    with torch.no_grad():
        action_net.weight[verb_bins:, :].zero_()
        action_net.bias[verb_bins:].zero_()
        indices = torch.arange(aim_bins, device=action_net.weight.device)
        action_net.weight[verb_bins + indices, learned + indices] = scale
