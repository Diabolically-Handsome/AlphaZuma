"""Frozen level and actor-interface constants for the AlphaZuma 55 campaign."""

from __future__ import annotations

from zuma_rl.revenge_env import RevengeEnvConfig

CAMPAIGN_ID = "alphazuma-55-weekend-s81081401-v1"
ACTOR_INTERFACE = "full55-v1"

INCLUDED_LEVELS = (
    "Jungle1",
    "Jungle2",
    "Jungle3",
    "Jungle4",
    "jungle6",
    "Jungle7",
    "Jungle8",
    "Jungle9",
    "Jungle10",
    "village1",
    "village2",
    "village3",
    "village4",
    "village5",
    "village6",
    "village7",
    "village8",
    "village10",
    "city1",
    "city2",
    "city3",
    "city4",
    "city5",
    "city6",
    "city7",
    "city9",
    "city10",
    "Coast1",
    "Coast2",
    "Coast3",
    "Coast4",
    "Coast5",
    "Coast6",
    "Coast7",
    "Coast9",
    "Coast10",
    "grotto1",
    "grotto2",
    "grotto3",
    "grotto4",
    "grotto5",
    "grotto6",
    "grotto7",
    "grotto9",
    "grotto10",
    "volcano1",
    "volcano2",
    "volcano3",
    "volcano4",
    "volcano5",
    "volcano6",
    "volcano7",
    "volcano8",
    "volcano9",
    "volcano10",
)

EXCLUDED_MOVING_FROG_LEVELS = (
    "Jungle5",
    "village9",
    "city8",
    "Coast8",
    "grotto8",
)

DUAL_POSITION_LEVELS = (
    "village3",
    "city3",
    "Coast3",
    "Coast10",
    "grotto3",
    "volcano3",
)


def full55_environment_config(*, max_ticks: int = 180_000) -> RevengeEnvConfig:
    """Return the fixed six-colour, four-verb actor interface."""

    return RevengeEnvConfig(
        aim_bins=180,
        action_mode="factorized",
        frame_skip=1,
        max_ticks=max_ticks,
        max_balls=768,
        max_projectiles=32,
        max_curves=2,
        actor_interface=ACTOR_INTERFACE,
    )


def legacy_migration_environment_config(
    *,
    max_ticks: int = 180_000,
) -> RevengeEnvConfig:
    """Exact V1/V1.1 interface used only for function-preservation proofs."""

    return RevengeEnvConfig(
        aim_bins=180,
        action_mode="factorized",
        frame_skip=1,
        max_ticks=max_ticks,
        max_balls=768,
        max_projectiles=32,
        max_curves=2,
        actor_interface="legacy-v1",
    )
