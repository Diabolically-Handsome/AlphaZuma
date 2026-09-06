"""Build the C111/v7 v3 candidate suite without mutating the live suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zuma_rl.fidelity_gate import FidelitySuite


REMOVED_BASE_IDS = frozenset(
    {
        "a8-audit-c109-natural-matches",
        "l-audit-actor-hidden-state",
        "o-pc-source-c109-long-horizon",
        "p-pc-source-c110-long-horizon",
    }
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
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


def _relative_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("candidate evidence path is not safe and relative")
    path = root.joinpath(*posix.parts).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file():
        raise ValueError("candidate evidence path is not a file")
    return path


def _entry(
    *,
    evidence_id: str,
    kind: str,
    relative: str,
    features: list[str],
    evidence_root: Path,
) -> dict[str, Any]:
    path = _relative_file(evidence_root, relative)
    return {
        "id": evidence_id,
        "kind": kind,
        "path": relative,
        "sha256": _sha256_path(path),
        "features": sorted(features),
    }


def build_candidate(
    *,
    base_suite: str | Path,
    evidence_root: str | Path,
    distribution_bundle: str | Path,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    base = json.loads(Path(base_suite).read_text(encoding="utf-8"))
    FidelitySuite.from_dict(base)
    if base.get("policy") != "original-transfer-jungle2-v2":
        raise ValueError("base suite is not the frozen v2 template")
    raw_entries = base.get("evidence")
    if not isinstance(raw_entries, list):
        raise ValueError("base suite evidence is invalid")
    entries = [
        dict(item)
        for item in raw_entries
        if isinstance(item, dict) and item.get("id") not in REMOVED_BASE_IDS
    ]
    if len(entries) != len(raw_entries) - len(REMOVED_BASE_IDS):
        raise ValueError("base suite replacement set is incomplete")

    entries.extend(
        [
            _entry(
                evidence_id="a8-audit-c111-natural-full-state",
                kind="audit",
                relative="diagnostics/pc-mechanism-c111-full-state-live-v3.json",
                features=["pc_mechanism_coverage"],
                evidence_root=root,
            ),
            _entry(
                evidence_id="l-audit-actor-hidden-state-current",
                kind="audit",
                relative="diagnostics/actor-audit-jungle2-current-v5.json",
                features=[
                    "actor_no_hidden_state",
                    "fruit_actor_observation",
                ],
                evidence_root=root,
            ),
            _entry(
                evidence_id="m-audit-original-fruit-assets",
                kind="audit",
                relative="diagnostics/original-asset-audit-jungle2-v1.json",
                features=["fruit_asset_geometry"],
                evidence_root=root,
            ),
            _entry(
                evidence_id="o-pc-source-c111-full-state",
                kind="pc_source",
                relative="sources/c111-u3615-u4240-full-state-source-v1.json",
                features=[
                    "back_insertion",
                    "front_insertion",
                    "fruit_visual_oscillator",
                    "long_horizon_drift",
                    "match3",
                    "match4",
                    "projectile_collision",
                    "rng_pending",
                    "shot_release",
                    "source_authenticity",
                    "swap",
                ],
                evidence_root=root,
            ),
            _entry(
                evidence_id="q-sim-c111-full-gameplay",
                kind="simulator_diff",
                relative="diagnostics/jungle2-c111-u3620-u4240-full-gameplay-diff-v12.json",
                features=[
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
                evidence_root=root,
            ),
        ]
    )

    bundle = Path(distribution_bundle).resolve(strict=True)
    bundle_relative = PurePosixPath(bundle.relative_to(root).as_posix())
    entries.append(
        _entry(
            evidence_id="r-distribution-v7",
            kind="distribution_audit",
            relative=(bundle_relative / "audit.json").as_posix(),
            features=["startup_actor_distribution"],
            evidence_root=root,
        )
    )
    manifests = sorted((bundle / "manifests").glob("pc-*.json"))
    if len(manifests) != 32:
        raise ValueError("distribution bundle must contain exactly 32 manifests")
    for index, manifest in enumerate(manifests, start=1):
        relative = PurePosixPath(manifest.relative_to(root).as_posix()).as_posix()
        entries.append(
            _entry(
                evidence_id=f"s{index:02d}-pc-source-startup-distribution",
                kind="pc_source",
                relative=relative,
                features=["source_authenticity"],
                evidence_root=root,
            )
        )

    candidate = {
        "schema": base["schema"],
        "version": base["version"],
        "policy": "original-transfer-jungle2-v3",
        "evidence": sorted(entries, key=lambda item: item["id"]),
    }
    FidelitySuite.from_dict(candidate)
    return candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-suite", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--distribution-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidate = build_candidate(
        base_suite=args.base_suite,
        evidence_root=args.evidence_root,
        distribution_bundle=args.distribution_bundle,
    )
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("candidate output must be absent with an existing parent")
    data = _canonical_bytes(candidate)
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "evidence_count": len(candidate["evidence"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
