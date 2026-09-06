"""Rebind a frozen distribution campaign beneath a larger evidence root.

The scientific inputs, samples, hashes, and thresholds remain unchanged.  The
tool only prefixes campaign-relative artifact paths and writes copied source
manifests whose own bindings resolve from the suite evidence root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from zuma_rl.distribution_fidelity import verify_distribution_audit


RECEIPT_SCHEMA = "zuma-rl.distribution-audit-suite-binding-receipt"
RECEIPT_VERSION = 1


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


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_canonical(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict) or data != _canonical_bytes(value):
        raise ValueError(f"artifact is not canonical JSON: {path}")
    return value


def _relative_posix(value: Any, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is not a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} is not a safe relative POSIX path")
    return path


def _prefixed(value: Any, prefix: PurePosixPath, *, name: str) -> str:
    return (prefix / _relative_posix(value, name=name)).as_posix()


def _rewrite_bindings(value: Any, prefix: PurePosixPath) -> Any:
    if isinstance(value, list):
        return [_rewrite_bindings(item, prefix) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten = {
        key: _rewrite_bindings(item, prefix) for key, item in value.items()
    }
    if set(value) == {"path", "sha256"}:
        rewritten["path"] = _prefixed(
            value["path"], prefix, name="artifact binding"
        )
    return rewritten


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    data = _canonical_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return _sha256_bytes(data)


def package_distribution_audit_binding(
    *,
    source_campaign: str | Path,
    evidence_root: str | Path,
    output_dir: str | Path,
    original_root: str | Path,
    policy_id: str,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    source = Path(source_campaign).resolve(strict=True)
    target = Path(output_dir).resolve(strict=False)
    source_relative = PurePosixPath(source.relative_to(root).as_posix())
    target_relative = PurePosixPath(target.relative_to(root).as_posix())
    if not source.is_dir() or target.exists() or not target.parent.is_dir():
        raise ValueError("source must exist and output directory must be absent")

    source_audit_path = source / "audit.json"
    source_report = verify_distribution_audit(
        source_audit_path,
        evidence_root=source,
        original_root=original_root,
        policy_id=policy_id,
    )
    if source_report.get("status") != "PASS":
        raise ValueError("source distribution audit did not pass recomputation")

    audit = _read_canonical(source_audit_path)
    prereg_binding = audit.get("preregistration")
    pc_binding = audit.get("pc_dataset")
    simulator_binding = audit.get("simulator_dataset")
    if not all(
        isinstance(binding, Mapping)
        and set(binding) == {"path", "sha256"}
        for binding in (prereg_binding, pc_binding, simulator_binding)
    ):
        raise ValueError("source audit bindings are invalid")

    def source_bound(binding: Mapping[str, Any], name: str) -> Path:
        path = source.joinpath(
            *_relative_posix(binding["path"], name=name).parts
        ).resolve(strict=True)
        path.relative_to(source)
        if _sha256_path(path) != binding["sha256"]:
            raise ValueError(f"{name} hash differs")
        return path

    source_preregistration = _read_canonical(
        source_bound(prereg_binding, "preregistration")
    )
    pc_dataset = _read_canonical(source_bound(pc_binding, "pc dataset"))
    simulator_dataset = _read_canonical(
        source_bound(simulator_binding, "simulator dataset")
    )
    preregistration = _rewrite_bindings(
        source_preregistration, source_relative
    )

    slots = preregistration.get("selection", {}).get("pc_slots")
    source_slots = source_preregistration.get("selection", {}).get(
        "pc_slots"
    )
    samples = pc_dataset.get("samples")
    if (
        not isinstance(slots, list)
        or not isinstance(source_slots, list)
        or not isinstance(samples, list)
        or len(source_slots) != len(slots)
        or len(slots) != len(samples)
        or not slots
    ):
        raise ValueError("PC slots and samples are not aligned")

    copied_manifest_bindings: list[dict[str, str]] = []
    for index, (source_slot, slot, sample) in enumerate(
        zip(source_slots, slots, samples, strict=True), start=1
    ):
        if (
            not isinstance(source_slot, dict)
            or not isinstance(slot, dict)
            or not isinstance(sample, dict)
        ):
            raise ValueError("PC slot or sample is invalid")
        source_manifest_relative = _relative_posix(
            source_slot.get("source_manifest_path"),
            name=f"PC slot {index} source manifest",
        )
        source_manifest_path = source.joinpath(
            *source_manifest_relative.parts
        ).resolve(strict=True)
        source_manifest_path.relative_to(source)
        source_manifest = _rewrite_bindings(
            _read_canonical(source_manifest_path), source_relative
        )
        copied_relative = target_relative / "manifests" / f"pc-{index:03d}.json"
        copied_path = root.joinpath(*copied_relative.parts)
        copied_hash = _write_exclusive(copied_path, source_manifest)
        copied_binding = {
            "path": copied_relative.as_posix(),
            "sha256": copied_hash,
        }
        copied_manifest_bindings.append(copied_binding)
        slot["source_manifest_path"] = copied_binding["path"]
        slot["output_root_path"] = _prefixed(
            source_slot.get("output_root_path"),
            source_relative,
            name=f"PC slot {index} output root",
        )
        sample["source_manifest"] = deepcopy(copied_binding)

        provenance_binding = source_slot.get("seed_provenance")
        if (
            not isinstance(provenance_binding, Mapping)
            or set(provenance_binding) != {"path", "sha256"}
        ):
            raise ValueError("PC slot seed provenance binding is invalid")
        provenance_path = source_bound(
            provenance_binding,
            f"PC slot {index} seed provenance",
        )
        provenance = _rewrite_bindings(
            _read_canonical(provenance_path), source_relative
        )
        copied_provenance_relative = (
            target_relative / "provenance" / f"pc-{index:03d}.seed.json"
        )
        copied_provenance_hash = _write_exclusive(
            root.joinpath(*copied_provenance_relative.parts), provenance
        )
        slot["seed_provenance"] = {
            "path": copied_provenance_relative.as_posix(),
            "sha256": copied_provenance_hash,
        }

    prereg_path = target / "preregistration.json"
    pc_path = target / "pc-dataset.json"
    simulator_path = target / "simulator-dataset.json"
    prereg_hash = _write_exclusive(prereg_path, preregistration)
    pc_hash = _write_exclusive(pc_path, pc_dataset)
    simulator_hash = _write_exclusive(simulator_path, simulator_dataset)

    packaged_audit = deepcopy(audit)
    packaged_audit["preregistration"] = {
        "path": (target_relative / "preregistration.json").as_posix(),
        "sha256": prereg_hash,
    }
    packaged_audit["pc_dataset"] = {
        "path": (target_relative / "pc-dataset.json").as_posix(),
        "sha256": pc_hash,
    }
    packaged_audit["simulator_dataset"] = {
        "path": (target_relative / "simulator-dataset.json").as_posix(),
        "sha256": simulator_hash,
    }
    packaged_audit_path = target / "audit.json"
    packaged_audit_hash = _write_exclusive(packaged_audit_path, packaged_audit)
    packaged_report = verify_distribution_audit(
        packaged_audit_path,
        evidence_root=root,
        original_root=original_root,
        policy_id=policy_id,
    )
    if packaged_report != source_report:
        raise ValueError("suite-bound audit recomputation differs from source")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "status": "PASS",
        "classification": "posthoc_path_rebinding_only",
        "source_campaign": source_relative.as_posix(),
        "source_audit_sha256": _sha256_path(source_audit_path),
        "packaged_audit": {
            "path": (target_relative / "audit.json").as_posix(),
            "sha256": packaged_audit_hash,
        },
        "copied_manifest_count": len(copied_manifest_bindings),
        "scientific_payload_unchanged": packaged_report == source_report,
        "recomputed_result": packaged_report,
    }
    _write_exclusive(target / "binding-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument(
        "--policy-id", default="original-transfer-jungle2-v2"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = package_distribution_audit_binding(
        source_campaign=args.source_campaign,
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        original_root=args.original_root,
        policy_id=args.policy_id,
    )
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
