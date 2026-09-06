"""Bind a verified native MTRand proof to a gameplay differential."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Iterable

if __package__ in {None, ""}:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_mtrand_reconciliation import (
    MTRAND_RECONCILIATION_PROOF_CLASSIFICATION,
    MTRAND_RECONCILIATION_PROOF_SCHEMA,
    MTRAND_RECONCILIATION_PROOF_VERSION,
    reconciliation_transcript_sha256,
    sha256_path,
    verify_mtrand_reconciliation_proof_binding,
)


def _relative_artifact(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root = root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("proof artifact is outside the evidence root") from error
    value = PurePosixPath(*relative.parts).as_posix()
    if any(part in {"", ".", ".."} for part in PurePosixPath(value).parts):
        raise ValueError("proof artifact path is not normalized")
    return value


def bind_proof(
    report: dict[str, object],
    *,
    proof_path: Path,
    evidence_root: Path,
) -> dict[str, object]:
    if "global_mtrand_reconciliation_proof" in report:
        raise ValueError("gameplay diff already contains a proof binding")
    proof = verify_mtrand_reconciliation_proof_binding(
        proof_path,
        report=report,
        evidence_root=evidence_root,
    )
    transcript_sha256 = reconciliation_transcript_sha256(report)
    binding = proof.get("gameplay_diff_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("reconciliation_transcript_sha256")
        != transcript_sha256
    ):
        raise ValueError("proof transcript binding mismatch")
    result = dict(report)
    result["global_mtrand_reconciliation_proof"] = {
        "artifact": _relative_artifact(proof_path, evidence_root),
        "artifact_sha256": sha256_path(proof_path),
        "schema": MTRAND_RECONCILIATION_PROOF_SCHEMA,
        "version": MTRAND_RECONCILIATION_PROOF_VERSION,
        "status": "PASS",
        "classification": MTRAND_RECONCILIATION_PROOF_CLASSIFICATION,
        "reconciliation_transcript_sha256": transcript_sha256,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameplay-diff", required=True, type=Path)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    try:
        report = json.loads(args.gameplay_diff.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("gameplay diff JSON is invalid") from error
    if not isinstance(report, dict):
        raise ValueError("gameplay diff JSON root must be an object")
    bound = bind_proof(
        report,
        proof_path=args.proof,
        evidence_root=args.evidence_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bound, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)
    print(
        json.dumps(
            {
                "sha256": sha256_path(output),
                "proof_sha256": bound[
                    "global_mtrand_reconciliation_proof"
                ]["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
