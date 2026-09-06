"""Reference and fidelity-first Zuma reinforcement-learning environments."""

from gymnasium.envs.registration import register, registry

from zuma_rl.config import ZumaConfig
from zuma_rl.core import SpiralTrack, StepResult, ZumaSimulator, resolve_matches
from zuma_rl.env import ZumaEnv
from zuma_rl.fidelity_gate import (
    FidelityGateReport,
    FidelityGateStatus,
    FidelityGateValidationError,
    FidelitySuite,
    verify_fidelity_suite,
)
from zuma_rl.golden_diff import (
    SimulatorTick,
    compare_pc_golden_case,
    snapshot_revenge_simulator,
)
from zuma_rl.pc_calibration import PcCalibrationSidecar
from zuma_rl.pc_golden import (
    ComparisonResult,
    ComparisonStatus,
    PcGoldenManifest,
    PcGoldenTrace,
)
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError
from zuma_rl.replay import (
    ReplayAction,
    ReplayMismatchError,
    ReplayPlan,
    ReplayResult,
    ReplayValidationError,
    environment_fingerprint,
    run_replay,
    state_fingerprint,
    verify_replay,
)
from zuma_rl.revenge_core import (
    ChainBall,
    CurveRuntimeState,
    FruitSpawnCalibration,
    GunState,
    PowerupSpawnCalibration,
    PowerupType,
    Projectile,
    RevengePhysicsConfig,
    RevengeSimulator,
    SUPPORTED_PROFILE_MODE,
    TickEvents,
)
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.training_gate import (
    TrainingGateReport,
    TrainingGateStatus,
    TrainingGateValidationError,
    TrainingSuite,
    verify_training_suite,
)
from zuma_rl.verify_pc_golden import (
    PcGoldenVerificationReport,
    verify_pc_golden_case,
)

__all__ = [
    "ChainBall",
    "ComparisonResult",
    "ComparisonStatus",
    "CurveRuntimeState",
    "FruitSpawnCalibration",
    "FidelityGateReport",
    "FidelityGateStatus",
    "FidelityGateValidationError",
    "FidelitySuite",
    "GunState",
    "PcGoldenManifest",
    "PcGoldenTrace",
    "PcGoldenVerificationReport",
    "PcCalibrationSidecar",
    "PopCapDemo",
    "PopCapDemoError",
    "PowerupType",
    "PowerupSpawnCalibration",
    "Projectile",
    "ReplayAction",
    "ReplayMismatchError",
    "ReplayPlan",
    "ReplayResult",
    "ReplayValidationError",
    "RevengeEnv",
    "RevengeEnvConfig",
    "RevengePhysicsConfig",
    "RevengeSimulator",
    "SUPPORTED_PROFILE_MODE",
    "SimulatorTick",
    "SpiralTrack",
    "StepResult",
    "TickEvents",
    "TrainingGateReport",
    "TrainingGateStatus",
    "TrainingGateValidationError",
    "TrainingSuite",
    "ZumaConfig",
    "ZumaEnv",
    "ZumaSimulator",
    "compare_pc_golden_case",
    "environment_fingerprint",
    "resolve_matches",
    "run_replay",
    "state_fingerprint",
    "snapshot_revenge_simulator",
    "verify_pc_golden_case",
    "verify_fidelity_suite",
    "verify_replay",
    "verify_training_suite",
]

__version__ = "0.4.0"

if "ZumaSimple-v0" not in registry:
    register(
        id="ZumaSimple-v0",
        entry_point="zuma_rl.env:ZumaEnv",
    )

if "ZumaRevenge-v0" not in registry:
    register(
        id="ZumaRevenge-v0",
        entry_point="zuma_rl.revenge_env:RevengeEnv",
    )
