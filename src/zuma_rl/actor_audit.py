"""Headless audit of the state actor observation's hidden-information boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zuma_rl.pc_golden import canonical_sha256
from zuma_rl.replay import environment_fingerprint
from zuma_rl.revenge_core import PowerupType, SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig

AUDIT_SCHEMA = "zuma-rl.fidelity-audit"
AUDIT_VERSION = 2
AUDIT_TYPE = "actor_no_hidden_state"
POLICY_ID = "original-transfer-jungle2-v2"
AUDIT_SEED = 1729

_PRIVILEGED_GLOBALS = frozenset(
    {
        "advance_speed",
        "slow_timer",
        "backward_timer",
        "pending_fraction",
    }
)


def _implementation_fingerprint() -> str:
    """Bind the audit to the exact observation and simulator source."""

    from zuma_rl import revenge_core, revenge_env

    paths = (
        Path(__file__).resolve(),
        Path(revenge_core.__file__).resolve(),
        Path(revenge_env.__file__).resolve(),
    )
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        payload = path.read_bytes()
        encoded_name = path.name.encode("utf-8")
        hasher.update(len(encoded_name).to_bytes(4, "big"))
        hasher.update(encoded_name)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return f"sha256:{hasher.hexdigest()}"


def _visibility_class(section: str, feature: str) -> str | None:
    if section == "balls":
        if feature in {"present", "x", "y", "exploding"}:
            return "direct_visible"
        if feature == "in_tunnel":
            return "visibility_filter"
        if feature in {"waypoint", "contact_next_visible"}:
            return "geometry_derived"
        if feature.startswith(("color_", "powerup_")):
            return "direct_visible"
        if feature.startswith("curve_"):
            return "level_geometry"
        return None
    if section == "projectiles":
        if feature in {"present", "x", "y"}:
            return "direct_visible"
        if feature in {
            "velocity_x",
            "velocity_y",
            "merging",
            "merge_progress",
        }:
            return "history_derived"
        if feature == "waypoint":
            return "geometry_derived"
        if feature.startswith("color_"):
            return "direct_visible"
        if feature.startswith("curve_"):
            return "level_geometry"
        return None
    if section == "globals":
        if feature in _PRIVILEGED_GLOBALS:
            return "privileged_zero"
        if feature == "tick_progress":
            return "agent_clock"
        if feature == "outcome":
            return "terminal_api"
        if feature in {
            "gun_state_percent",
            "zuma_reached",
            "stop_adding",
        }:
            return "history_derived"
        if feature in {
            "visible_ball_fraction",
            "known_ball_fraction",
            "projectile_fraction",
        }:
            return "visible_count"
        if feature in {
            "aim_sin",
            "aim_cos",
            "current_missing",
            "next_missing",
            "bar_current",
            "bar_target",
            "score_progress",
            "score_tanh",
            "gun_normal",
            "gun_firing",
            "gun_reloading",
            "fruit_present",
            "fruit_collectable",
            "fruit_x",
            "fruit_y",
            "fruit_collecting",
        } or feature.startswith(("current_color_", "next_color_")):
            return "direct_visible"
        return None
    return None


def _feature_contract(env: RevengeEnv) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sections = (
        ("balls", env.ball_feature_names),
        ("projectiles", env.projectile_feature_names),
        ("globals", env.global_feature_names),
    )
    for section, names in sections:
        for name in names:
            visibility = _visibility_class(section, name)
            rows.append(
                {
                    "section": section,
                    "feature": name,
                    "visibility": (
                        "unmapped" if visibility is None else visibility
                    ),
                }
            )
    return rows


def _check_row(name: str, passed: bool, message: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "message": message,
    }


def _audit_payload(
    *,
    root: str | Path | None,
    level_id: str,
    hard: bool,
    profile_mode: str,
    seed: int,
) -> dict[str, Any]:
    actor = RevengeEnv(
        config=RevengeEnvConfig(privileged_debug=False),
        level_id=level_id,
        root=root,
        hard=hard,
        profile_mode=profile_mode,
        seed=seed,
    )
    try:
        tunnel_mask = np.asarray(
            actor.sim.curves[0].in_tunnel,
            dtype=np.bool_,
        )
        tunnel_indices = np.flatnonzero(tunnel_mask)
        visible_indices = np.flatnonzero(~tunnel_mask)
        if tunnel_indices.size < 1 or visible_indices.size < 2:
            raise RuntimeError(
                "level does not provide enough visible and tunnel samples"
            )
        tunnel_a = int(tunnel_indices[0])
        tunnel_b = int(tunnel_indices[-1])
        visible_a = int(visible_indices[len(visible_indices) // 3])
        visible_b = int(visible_indices[2 * len(visible_indices) // 3])
        colors = [0, 1 % actor.num_colors, 2 % actor.num_colors]
        actor.reset(
            seed=seed,
            options={
                "state": {
                    "colors": colors,
                    "waypoints": [
                        float(visible_a),
                        float(tunnel_a),
                        float(visible_b),
                    ],
                    "contacts": [False, False],
                    "pending_colors": [0, 1 % actor.num_colors],
                    "current_color": 0,
                    "next_color": 1 % actor.num_colors,
                    "score": 100,
                }
            },
        )
        privileged = RevengeEnv(
            config=RevengeEnvConfig(privileged_debug=True),
            simulator=actor.sim,
            profile_mode=profile_mode,
        )

        checks: list[dict[str, str]] = []
        contract = _feature_contract(actor)
        unmapped = [
            row for row in contract if row["visibility"] == "unmapped"
        ]
        checks.append(
            _check_row(
                "feature_contract_complete",
                not unmapped,
                "every actor feature has an explicit visibility class",
            )
        )

        visible_balls = actor._visible_balls()
        checks.append(
            _check_row(
                "tunnel_filter",
                (
                    len(actor.sim.balls) == 3
                    and len(visible_balls) == 2
                    and actor.sim.balls[1].id
                    not in {ball.id for _, ball in visible_balls}
                ),
                "a ball whose centre is in a tunnel is absent from actor slots",
            )
        )

        global_values = actor._global_observation(
            visible_count=len(visible_balls),
            projectile_count=0,
        )
        privileged_indices = [
            actor.global_feature_names.index(name)
            for name in sorted(_PRIVILEGED_GLOBALS)
        ]
        checks.append(
            _check_row(
                "privileged_global_slots_zero",
                bool(np.all(global_values[privileged_indices] == 0.0)),
                "pending and internal movement timers are zero in actor mode",
            )
        )

        actor_before_powerup = actor._observation()
        visible_ball = actor.sim.balls[0]
        visible_ball.powerup_primary_type = int(PowerupType.REVERSE)
        actor_after_powerup = actor._observation()
        balls = actor_after_powerup[
            actor.observation_layout["balls"]
        ].reshape(actor.config.max_balls, actor.ball_feature_size)
        powerup_index = actor.ball_feature_names.index(
            f"powerup_{int(PowerupType.REVERSE)}"
        )
        visible_powerup_ok = (
            balls[0, powerup_index] == 1.0
            and not np.array_equal(
                actor_before_powerup,
                actor_after_powerup,
            )
        )
        checks.append(
            _check_row(
                "visible_powerup_channel",
                bool(visible_powerup_ok),
                "a visible retail power-up icon changes its one-hot channel",
            )
        )
        visible_ball.powerup_primary_type = int(PowerupType.NONE)

        actor_before_fruit = actor._observation()
        actor.sim.fruit_active_point_index = 0
        actor.sim.fruit_collecting = False
        actor.sim.fruit_vertical_offset = np.float32(0.0)
        actor_after_fruit = actor._observation()
        fruit_globals = actor_after_fruit[
            actor.observation_layout["globals"]
        ]
        fruit_present_index = actor.global_feature_names.index(
            "fruit_present"
        )
        fruit_collectable_index = actor.global_feature_names.index(
            "fruit_collectable"
        )
        visible_fruit_ok = (
            fruit_globals[fruit_present_index] == 1.0
            and fruit_globals[fruit_collectable_index] == 1.0
            and not np.array_equal(actor_before_fruit, actor_after_fruit)
        )
        checks.append(
            _check_row(
                "visible_fruit_channels",
                bool(visible_fruit_ok),
                "a visible fruit changes presence, position, and state channels",
            )
        )

        actor_before_hidden = actor._observation()
        privileged_before_hidden = privileged._observation()
        state = actor.sim.curve_states[0]
        hidden_ball = state.balls[1]
        hidden_ball.color = (hidden_ball.color + 1) % actor.num_colors
        hidden_ball.waypoint = np.float32(tunnel_b)
        hidden_ball.powerup_primary_type = int(PowerupType.REVERSE)
        state.pending_colors[:] = [
            3 % actor.num_colors,
            2 % actor.num_colors,
            1 % actor.num_colors,
            0,
        ]
        state.advance_speed = np.float32(3.25)
        state.slow_count = 777
        state.backward_count = 123
        state.powerup_last_any_spawn_time = 9999
        state.powerup_last_spawn_times[:] = [811] * len(
            state.powerup_last_spawn_times
        )
        state.powerup_cooldown_times[:] = [733] * len(
            state.powerup_cooldown_times
        )
        actor.sim.fruit_expiry_time = 99_999
        actor.sim.fruit_spawn_count = 17
        actor.sim.fruit_collect_count = 9
        actor.sim.fruit_cell_index = 41
        actor.sim.fruit_glow_alpha = 211
        actor.sim.fruit_glow_step = -48
        actor.sim.fruit_collection_ticks_remaining = 13
        actor_after_hidden = actor._observation()
        privileged_after_hidden = privileged._observation()
        actor_equal = np.array_equal(
            actor_before_hidden,
            actor_after_hidden,
        )
        privileged_changed = not np.array_equal(
            privileged_before_hidden,
            privileged_after_hidden,
        )
        checks.append(
            _check_row(
                "hidden_state_invariance",
                bool(actor_equal),
                "tunnel-ball identity, pending colors, timers, and manager "
                "cooldowns do not alter actor output",
            )
        )
        checks.append(
            _check_row(
                "privileged_control_changed",
                bool(privileged_changed),
                "the same perturbation changes privileged_debug output",
            )
        )

        deterministic = np.array_equal(
            actor._observation(),
            actor._observation(),
        )
        checks.append(
            _check_row(
                "observation_deterministic",
                bool(deterministic),
                "repeated reads of one native state are bit-identical",
            )
        )

        failure_reasons = [
            f"actor audit check failed: {row['name']}"
            for row in checks
            if row["status"] != "PASS"
        ]
        payload: dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "version": AUDIT_VERSION,
            "status": "PASS" if not failure_reasons else "FAIL",
            "audit_type": AUDIT_TYPE,
            "scope": {
                "policy": POLICY_ID,
                "environment_id": "ZumaRevenge-v0",
                "level_id": level_id,
                "hard": hard,
                "profile_mode": profile_mode,
                "observation_mode": "actor",
            },
            "seed": seed,
            "environment_fingerprint": environment_fingerprint(actor.sim),
            "implementation_fingerprint": _implementation_fingerprint(),
            "checks": checks,
            "feature_contract": contract,
            "failure_reasons": failure_reasons,
            "summary": {
                "feature_count": len(contract),
                "unmapped_feature_count": len(unmapped),
                "hidden_perturbation_actor_equal": bool(actor_equal),
                "privileged_control_changed": bool(privileged_changed),
                "visible_powerup_channel_verified": bool(visible_powerup_ok),
                "visible_fruit_channels_verified": bool(visible_fruit_ok),
                "tunnel_hidden_ball_count": 1,
            },
        }
        return payload
    finally:
        actor.close()


def audit_actor_observation(
    *,
    root: str | Path | None = None,
    level_id: str = "Jungle2",
    hard: bool = False,
    profile_mode: str = SUPPORTED_PROFILE_MODE,
    seed: int = AUDIT_SEED,
) -> dict[str, Any]:
    """Return a deterministic report without exposing installation paths."""

    try:
        payload = _audit_payload(
            root=root,
            level_id=level_id,
            hard=hard,
            profile_mode=profile_mode,
            seed=seed,
        )
    except Exception as error:
        payload = {
            "schema": AUDIT_SCHEMA,
            "version": AUDIT_VERSION,
            "status": "INCOMPARABLE",
            "audit_type": AUDIT_TYPE,
            "scope": {
                "policy": POLICY_ID,
                "environment_id": "ZumaRevenge-v0",
                "level_id": level_id,
                "hard": hard,
                "profile_mode": profile_mode,
                "observation_mode": "actor",
            },
            "seed": seed,
            "implementation_fingerprint": _implementation_fingerprint(),
            "checks": [],
            "feature_contract": [],
            "failure_reasons": [
                "actor audit could not construct or probe the environment"
            ],
            "summary": {
                "error_type": type(error).__name__,
                "unmapped_feature_count": None,
                "hidden_perturbation_actor_equal": None,
                "privileged_control_changed": None,
                "visible_powerup_channel_verified": None,
                "visible_fruit_channels_verified": None,
            },
        }
    payload["audit_fingerprint"] = canonical_sha256(payload)
    return payload


def validate_actor_audit_report(
    report: Mapping[str, Any],
    *,
    root: str | Path | None,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> str | None:
    """Re-run the audit and return a safe rejection reason, if any."""

    if report.get("audit_type") != AUDIT_TYPE:
        return "unsupported actor audit type"
    if report.get("status") != "PASS":
        return "actor audit report is not PASS"
    live = audit_actor_observation(
        root=root,
        level_id=level_id,
        hard=hard,
        profile_mode=profile_mode,
    )
    if live.get("status") != "PASS":
        return "live actor audit is not comparable or did not pass"
    if report.get("audit_fingerprint") != live.get("audit_fingerprint"):
        return "actor audit fingerprint differs from the live implementation"
    checks = report.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(item, Mapping) or item.get("status") != "PASS"
            for item in checks
        )
    ):
        return "actor audit checks are missing or not all PASS"
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return "actor audit summary is missing"
    expected = {
        "unmapped_feature_count": 0,
        "hidden_perturbation_actor_equal": True,
        "privileged_control_changed": True,
        "visible_powerup_channel_verified": True,
        "visible_fruit_channels_verified": True,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return "actor audit summary does not prove the hidden-state boundary"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Headless audit that actor observations hide tunnel balls, "
            "pending colors, manager cooldowns, and internal movement timers."
        )
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create the recomputed audit report at this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_actor_observation(
        root=args.root,
        level_id=args.level,
        hard=args.hard,
        seed=args.seed,
    )
    payload = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() or not output.parent.is_dir():
            raise ValueError("actor audit output path must be absent")
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0 if report["status"] == "PASS" else 1


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_SEED",
    "AUDIT_TYPE",
    "AUDIT_VERSION",
    "audit_actor_observation",
    "build_parser",
    "main",
    "validate_actor_audit_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
