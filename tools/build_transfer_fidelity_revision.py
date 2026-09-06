"""Build an immutable revision of the transfer Fidelity candidate suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zuma_rl.fidelity_gate import FidelitySuite


POLICY_ID = "original-transfer-jungle2-v4"
BASE_POLICY_IDS = frozenset(
    {
        "original-transfer-jungle2-v3",
        "original-transfer-jungle2-v4",
    }
)
REMOVED_ENTRY_IDS = frozenset(
    {
        "a11-audit-c122-natural-drain",
        "p3-pc-source-c122-natural-drain",
    }
)

# Keep this registry explicit: every revision is a reviewable replacement or
# addition, while every unrelated entry remains byte-for-byte unchanged.
CURRENT_ENTRIES: tuple[dict[str, Any], ...] = (
    {
        "id": "l-audit-actor-hidden-state-current",
        "kind": "audit",
        "path": "diagnostics/actor-audit-jungle2-current-v12.json",
        "features": [
            "actor_no_hidden_state",
            "fruit_actor_observation",
        ],
    },
    {
        "id": "a7-audit-c35-natural-shot",
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c35-natural-shot-v2-current.json",
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a9-audit-c118-natural-full-state",
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c118-full-state-v3-live.json",
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a10-audit-c119-fruit-terminal",
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c119-first-fruit-terminal-v1.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a11-audit-c126-natural-win",
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c126-natural-win-live-v6-current.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a12-audit-c130a1-rng-gap-rollback",
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c130a1-rng-gap-rollback-live-v4.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a13-audit-c131b1-reverse",
        "kind": "audit",
        "path": (
            "diagnostics/pc-mechanism-c131b1-reverse-live-v4-current.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a14-audit-c132c1-proximity-fruit-expiry",
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c132c1-proximity-fruit-expiry-live-v1.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a15-audit-c133a1-fruit-spawn",
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c133a1-fruit-spawn-live-v1.json",
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a16-audit-c158-natural-loss",
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c158-natural-loss-live-v1.json",
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a17-audit-c164-fruit-collection",
        "kind": "audit",
        "path": "diagnostics/pc-mechanism-c164-fruit-collection-live-v1.json",
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "a18-audit-c185-fruit-powerup-collision",
        "kind": "audit",
        "path": (
            "diagnostics/"
            "pc-mechanism-c185-fruit-powerup-collision-live-v2.json"
        ),
        "features": ["pc_mechanism_coverage"],
    },
    {
        "id": "p-pc-source-c118-full-state",
        "kind": "pc_source",
        "path": "sources/c118-u9547-u11047-full-state-source-v1.json",
        "features": [
            "fruit_scheduler_spawn",
            "long_horizon_drift",
            "match4",
            "powerup_spawn",
            "projectile_collision",
            "rng_pending",
            "rollback_chain",
            "shot_release",
            "source_authenticity",
        ],
    },
    {
        "id": "q1-sim-c118-fruit-scheduler",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c118-u10582-u10583-fruit-scheduler-diff-v3-live.json"
        ),
        "features": ["fruit_scheduler_spawn"],
    },
    {
        "id": "p2-pc-source-c119-fruit-terminal",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c119-first-fruit-terminal-full-state-source-v1.json"
        ),
        "features": [
            "front_insertion",
            "fruit_expiry",
            "long_horizon_drift",
            "powerup_spawn",
            "projectile_collision",
            "rng_pending",
            "shot_release",
            "source_authenticity",
        ],
    },
    {
        "id": "p3-pc-source-c126-natural-win",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c126-u6950-u7520-source-bound-natural-win-v3.json"
        ),
        "features": [
            "back_insertion",
            "front_insertion",
            "fruit_visual_oscillator",
            "long_horizon_drift",
            "match3",
            "natural_win",
            "powerup_proximity_bomb",
            "projectile_collision",
            "rng_pending",
            "shot_release",
            "source_authenticity",
            "swap",
            "tunnel_collision",
            "zuma_transition",
        ],
    },
    {
        "id": "p5-pc-source-c131b1-reverse",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c131b1-u4955-u5160-source-bound-reverse-v1.json"
        ),
        "features": [
            "back_insertion",
            "gap_shot",
            "match3",
            "powerup_reverse",
            "projectile_collision",
            "rng_pending",
            "rollback_chain",
            "shot_release",
            "source_authenticity",
        ],
    },
    {
        "id": "p4-pc-source-c130a1-rng-gap-rollback",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c130a1-u3575-u3970-source-bound-rng-gap-rollback-v1.json"
        ),
        "features": [
            "front_insertion",
            "gap_shot",
            "projectile_collision",
            "rng_pending",
            "rng_rejection",
            "rollback_chain",
            "shot_release",
            "source_authenticity",
            "swap",
        ],
    },
    {
        "id": "p6-pc-source-c132c1-proximity-fruit-expiry",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c132c1-u5745-u5890-source-bound-proximity-fruit-expiry-v1.json"
        ),
        "features": [
            "front_insertion",
            "fruit_expiry",
            "match3",
            "powerup_spawn",
            "projectile_collision",
            "rng_pending",
            "shot_release",
            "source_authenticity",
            "swap",
        ],
    },
    {
        "id": "p7-pc-source-c133a1-fruit-spawn",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c133a1-u4745-u4760-source-bound-fruit-spawn-v1.json"
        ),
        "features": [
            "fruit_scheduler_spawn",
            "source_authenticity",
        ],
    },
    {
        "id": "p8-pc-source-c158-natural-loss",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c158-u11200-u11830-source-bound-natural-loss-v1.json"
        ),
        "features": [
            "natural_loss",
            "source_authenticity",
        ],
    },
    {
        "id": "p9-pc-source-c164-fruit-collection",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c164-u5220-u5310-source-bound-fruit-collection-v1.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_collection_score",
            "fruit_projectile_collision",
            "source_authenticity",
        ],
    },
    {
        "id": "p10-pc-source-c183-fruit-powerup-collision",
        "kind": "pc_source",
        "path": (
            "sources/"
            "c183-u8120-u8270-source-bound-fruit-powerup-collision-v1.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_powerup_collision",
            "powerup_proximity_bomb",
            "source_authenticity",
        ],
    },
    {
        "id": "q3-sim-c126-natural-win",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c126-u7491-u7493-natural-win-diff-v2.json"
        ),
        "features": ["natural_win", "zuma_transition"],
    },
    {
        "id": "q4-sim-c126-tunnel-collision",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c126-u6952-u6953-tunnel-collision-diff-v1.json"
        ),
        "features": ["tunnel_collision"],
    },
    {
        "id": "q5-sim-c130a1-rng-rejection",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c130a1-u3575-u3583-rng-rejection-diff-v3.json"
        ),
        "features": ["rng_rejection"],
    },
    {
        "id": "q6-sim-c130a1-gap-shot",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c130a1-u3739-u3772-gap-shot-diff-v2.json"
        ),
        "features": ["gap_shot"],
    },
    {
        "id": "q7-sim-c126-powerup-proximity-bomb",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c126-u7222-u7248-powerup-proximity-bomb-"
            "diff-v14-nosync-bound-current.json"
        ),
        "features": ["powerup_proximity_bomb"],
    },
    {
        "id": "q8-sim-c131b1-powerup-reverse",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c131b1-u5060-u5098-powerup-reverse-"
            "diff-v9-nosync-bound-current.json"
        ),
        "features": ["powerup_reverse"],
    },
    {
        "id": "q2-sim-c119-fruit-expiry",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c119-u10289-u10290-fruit-expiry-diff-v1.json"
        ),
        "features": ["fruit_expiry"],
    },
    {
        "id": "q9-sim-c132c1-fruit-expiry",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c132c1-u5749-u5750-fruit-expiry-diff-v1.json"
        ),
        "features": ["fruit_expiry"],
    },
    {
        "id": "q10-sim-c132c1-powerup-spawn",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c132c1-u5755-u5756-powerup-spawn-diff-v1.json"
        ),
        "features": ["powerup_spawn"],
    },
    {
        "id": "q11-sim-c133a1-fruit-scheduler",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c133a1-u4749-u4750-fruit-scheduler-diff-v1.json"
        ),
        "features": ["fruit_scheduler_spawn"],
    },
    {
        "id": "q12-sim-c158-natural-loss",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c158-u11341-u11519-natural-loss-"
            "diff-v4-source-bound.json"
        ),
        "features": ["natural_loss"],
    },
    {
        "id": "q13-sim-c164-fruit-collection",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c164-u5278-u5305-fruit-collection-"
            "diff-v3-reconciled-bound.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_collection_score",
            "fruit_projectile_collision",
        ],
    },
    {
        "id": "q14-sim-c183-fruit-powerup-collision",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c183-u8213-u8260-fruit-powerup-collision-"
            "diff-v6-reconciled-bound.json"
        ),
        "features": [
            "fruit_collection_animation",
            "fruit_powerup_collision",
            "powerup_proximity_bomb",
        ],
    },
    {
        "id": "q-sim-c111-full-gameplay",
        "kind": "simulator_diff",
        "path": (
            "diagnostics/"
            "jungle2-c111-u3620-u4240-full-gameplay-diff-v14.json"
        ),
        "features": [
            "fruit_visual_oscillator",
            "input_cadence",
            "long_horizon_drift",
            "match3",
            "match4",
            "projectile_collision",
            "rng_pending",
            "shot_release",
            "swap",
        ],
    },
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _resolve_evidence(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ValueError("revision evidence path is not normalized")
    resolved = root.joinpath(*pure.parts).resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError("revision evidence path is not a file")
    return resolved


def build_revision(
    *,
    base_suite: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    base = json.loads(Path(base_suite).read_text(encoding="utf-8"))
    FidelitySuite.from_dict(base)
    if base.get("policy") not in BASE_POLICY_IDS:
        raise ValueError("base suite is not a supported transfer policy")
    raw_entries = base.get("evidence")
    if not isinstance(raw_entries, list):
        raise ValueError("base suite evidence is invalid")
    by_id = {
        str(entry["id"]): dict(entry)
        for entry in raw_entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if len(by_id) != len(raw_entries):
        raise ValueError("base suite evidence identifiers are invalid")
    for evidence_id in REMOVED_ENTRY_IDS:
        by_id.pop(evidence_id, None)

    for raw in CURRENT_ENTRIES:
        entry = dict(raw)
        features = sorted(set(entry["features"]))
        if features != entry["features"]:
            raise ValueError("revision features must be unique and sorted")
        path = _resolve_evidence(root, entry["path"])
        entry["sha256"] = _sha256_path(path)
        by_id[entry["id"]] = entry

    revision = {
        "schema": base["schema"],
        "version": base["version"],
        "policy": POLICY_ID,
        "evidence": [by_id[key] for key in sorted(by_id)],
    }
    FidelitySuite.from_dict(revision)
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-suite", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    revision = build_revision(
        base_suite=args.base_suite,
        evidence_root=args.evidence_root,
    )
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("revision output must be absent with an existing parent")
    payload = _canonical_bytes(revision)
    with output.open("xb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "evidence_count": len(revision["evidence"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
