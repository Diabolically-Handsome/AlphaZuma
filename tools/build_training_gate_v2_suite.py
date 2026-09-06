"""Create a content-addressed v2 transfer-bootstrap Training Gate suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zuma_rl.training_gate import (
    LEGACY_TRANSFER_POLICY_ID,
    LEGACY_TRANSFER_STAGE_ID,
    SUITE_SCHEMA,
    SUITE_VERSION,
    TrainingSuite,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _entry(
    *,
    root: Path,
    evidence_id: str,
    kind: str,
    path: Path,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = PurePosixPath(resolved.relative_to(root).as_posix())
    except ValueError as error:
        raise ValueError("training evidence is outside the evidence root") from error
    return {
        "id": evidence_id,
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": _sha256_path(resolved),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--outcome", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--environment-contract", required=True, type=Path)
    parser.add_argument("--model-migration", required=True, type=Path)
    parser.add_argument("--fidelity-suite", required=True, type=Path)
    parser.add_argument("--fidelity-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.evidence_root.resolve(strict=True)
    entries = [
        _entry(
            root=root,
            evidence_id="a-scale-stability-outcome",
            kind="scale_stability_outcome",
            path=args.outcome,
        ),
        _entry(
            root=root,
            evidence_id="b-scale-stability-preregistration",
            kind="preregistration",
            path=args.preregistration,
        ),
        _entry(
            root=root,
            evidence_id="c-training-environment-contract",
            kind="environment_contract",
            path=args.environment_contract,
        ),
        _entry(
            root=root,
            evidence_id="d-fruit-actor-model-migration",
            kind="model_migration",
            path=args.model_migration,
        ),
        _entry(
            root=root,
            evidence_id="e-fidelity-suite",
            kind="fidelity_suite",
            path=args.fidelity_suite,
        ),
        _entry(
            root=root,
            evidence_id="f-fidelity-report",
            kind="fidelity_report",
            path=args.fidelity_report,
        ),
    ]
    payload = {
        "schema": SUITE_SCHEMA,
        "version": SUITE_VERSION,
        "policy": LEGACY_TRANSFER_POLICY_ID,
        "stage": LEGACY_TRANSFER_STAGE_ID,
        "evidence": entries,
    }
    TrainingSuite.from_dict(payload)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent with an existing parent")
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "evidence_count": len(entries),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
