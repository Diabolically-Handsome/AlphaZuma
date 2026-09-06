"""Initial-policy anchoring for masked PPO diagnostics.

The factory keeps Stable-Baselines3 and PyTorch as optional runtime
dependencies.  Importing :mod:`zuma_rl.train` therefore remains possible in
an environment that only installed the base simulator dependencies.
"""

from __future__ import annotations

from typing import Any


def masked_distribution_forward_kl(
    teacher_distribution: Any,
    current_distribution: Any,
) -> Any:
    """Return per-sample ``KL(teacher || current)`` after action masking."""

    teacher_parts = getattr(teacher_distribution, "distributions", None)
    current_parts = getattr(current_distribution, "distributions", None)
    if teacher_parts is None and current_parts is None:
        teacher_parts = [teacher_distribution.distribution]
        current_parts = [current_distribution.distribution]
    elif teacher_parts is None or current_parts is None:
        raise RuntimeError("teacher and current action distributions differ")

    if len(teacher_parts) != len(current_parts) or not teacher_parts:
        raise RuntimeError("teacher and current action factors differ")

    factor_kls = []
    for teacher_part, current_part in zip(
        teacher_parts,
        current_parts,
        strict=True,
    ):
        teacher_probs = teacher_part.probs
        teacher_log_probs = teacher_part.logits
        current_log_probs = current_part.logits
        if (
            teacher_probs.shape != current_log_probs.shape
            or teacher_log_probs.shape != current_log_probs.shape
        ):
            raise RuntimeError("teacher and current categorical shapes differ")
        terms = teacher_probs * (teacher_log_probs - current_log_probs)
        terms = terms.where(teacher_probs > 0, terms.new_zeros(()))
        factor_kls.append(terms.sum(dim=-1))

    joint_kl = factor_kls[0]
    for factor_kl in factor_kls[1:]:
        joint_kl = joint_kl + factor_kl
    # Categorical KL is non-negative; clamp only floating-point roundoff.
    return joint_kl.clamp_min(0.0)


def make_teacher_anchored_maskable_ppo(base_class: type[Any]) -> type[Any]:
    """Create a MaskablePPO subclass with a frozen initial-policy KL term."""

    import numpy as np
    import torch as th
    from gymnasium import spaces
    from stable_baselines3.common.utils import explained_variance
    from torch.nn import functional as F

    class TeacherAnchoredMaskablePPO(base_class):
        """Pinned MaskablePPO train loop plus initial-policy forward KL."""

        _supports_teacher_policy_anchor = True

        def train(self) -> None:
            teacher_policy = getattr(self, "_teacher_policy", None)
            teacher_kl_coef = float(getattr(self, "teacher_kl_coef", 0.0))
            if teacher_policy is None or teacher_kl_coef <= 0.0:
                raise RuntimeError("teacher policy anchor is not installed")

            self.policy.set_training_mode(True)
            teacher_policy.set_training_mode(False)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(
                self._current_progress_remaining
            )
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(
                    self._current_progress_remaining
                )

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []
            anchor_kls = []
            continue_training = True
            loss = None
            approx_kl_divs = []

            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (
                            advantages - advantages.mean()
                        ) / (advantages.std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(
                        ratio,
                        1 - clip_range,
                        1 + clip_range,
                    )
                    policy_loss = -th.min(
                        policy_loss_1,
                        policy_loss_2,
                    ).mean()
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1) > clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = (
                            rollout_data.old_values
                            + th.clamp(
                                values - rollout_data.old_values,
                                -clip_range_vf,
                                clip_range_vf,
                            )
                        )
                    value_loss = F.mse_loss(
                        rollout_data.returns,
                        values_pred,
                    )
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    current_distribution = self.policy.get_distribution(
                        rollout_data.observations,
                        action_masks=rollout_data.action_masks,
                    )
                    with th.no_grad():
                        teacher_distribution = (
                            teacher_policy.get_distribution(
                                rollout_data.observations,
                                action_masks=rollout_data.action_masks,
                            )
                        )
                    anchor_kl_values = masked_distribution_forward_kl(
                        teacher_distribution,
                        current_distribution,
                    )
                    if not bool(th.isfinite(anchor_kl_values).all()):
                        raise RuntimeError(
                            "teacher policy anchor produced non-finite KL"
                        )
                    anchor_kl = anchor_kl_values.mean()
                    anchor_kls.append(float(anchor_kl.detach().cpu().item()))

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + teacher_kl_coef * anchor_kl
                    )

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean(
                            (th.exp(log_ratio) - 1) - log_ratio
                        ).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if (
                        self.target_kl is not None
                        and approx_kl_div > 1.5 * self.target_kl
                    ):
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                "Early stopping at step "
                                f"{epoch} due to reaching max kl: "
                                f"{approx_kl_div:.2f}"
                            )
                        break

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm,
                    )
                    self.policy.optimizer.step()

                    sample_count = int(anchor_kl_values.numel())
                    self._teacher_anchor_sample_count += sample_count
                    self._teacher_anchor_kl_sum += float(
                        anchor_kl_values.detach().sum().cpu().item()
                    )
                    self._teacher_anchor_kl_max = max(
                        self._teacher_anchor_kl_max,
                        float(anchor_kl_values.detach().max().cpu().item()),
                    )
                    self._teacher_anchor_minibatch_count += 1

                self._n_updates += 1
                if not continue_training:
                    break

            self._teacher_anchor_train_calls += 1
            if loss is None or not anchor_kls or not approx_kl_divs:
                raise RuntimeError("teacher-anchored PPO performed no update")
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record(
                "train/policy_gradient_loss",
                np.mean(pg_losses),
            )
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record(
                "train/clip_fraction",
                np.mean(clip_fractions),
            )
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record(
                "train/n_updates",
                self._n_updates,
                exclude="tensorboard",
            )
            self.logger.record("train/clip_range", clip_range)
            self.logger.record(
                "train/teacher_anchor_kl",
                np.mean(anchor_kls),
            )
            self.logger.record(
                "train/teacher_anchor_loss",
                teacher_kl_coef * np.mean(anchor_kls),
            )
            self.logger.record(
                "train/teacher_anchor_coef",
                teacher_kl_coef,
            )
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

    TeacherAnchoredMaskablePPO.__name__ = "TeacherAnchoredMaskablePPO"
    TeacherAnchoredMaskablePPO.__qualname__ = (
        "TeacherAnchoredMaskablePPO"
    )
    return TeacherAnchoredMaskablePPO
