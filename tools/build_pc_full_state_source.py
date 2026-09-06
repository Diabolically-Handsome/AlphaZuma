"""Package and verify one formal full-state retail trajectory source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
from typing import Any, Mapping

from zuma_rl.pc_source import (
    FULL_STATE_SOURCE_KIND,
    FULL_STATE_SOURCE_VERSION,
    SOURCE_BOUND_FULL_STATE_SOURCE_KIND,
    SOURCE_BOUND_FULL_STATE_SOURCE_VERSION,
    SOURCE_BOUND_REPLAY_BINDING_SCHEMA,
    SOURCE_SCHEMA,
    STRICT_REPLAY_BINDING_SCHEMA,
    verify_pc_source_manifest,
    write_pc_source_manifest,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not a mapping")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is invalid")
    return value


def _file_within(root: Path, path: Path, name: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes evidence root") from error
    if not resolved.is_file():
        raise ValueError(f"{name} is not a file")
    return resolved


def _binding(root: Path, path: Path) -> dict[str, str]:
    relative = PurePosixPath(path.relative_to(root).as_posix()).as_posix()
    return {"path": relative, "sha256": _sha256_path(path)}


def build_manifest(
    *,
    evidence_root: str | Path,
    original_root: str | Path,
    source_id: str,
    run_id: str,
    run_root: str | Path,
    runtime_payload: str | Path,
    dmo: str | Path,
    pre_snapshot: str | Path,
    post_snapshot: str | Path,
    start_update: int,
    end_update: int,
    maximum_startup_attempts: int,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    original = Path(original_root).resolve(strict=True)
    if not root.is_dir() or not original.is_dir():
        raise ValueError("evidence and original roots must be directories")
    original_executable = original / "ZumasRevenge.exe"
    if not original_executable.is_file():
        raise ValueError("original executable is missing")

    run_dir = Path(run_root).resolve(strict=True)
    try:
        run_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("run root escapes evidence root") from error
    attempts_path = _file_within(root, run_dir / "attempts.json", "attempts")
    attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError("formal source must contain exactly one attempt")
    attempt = _mapping(attempts[0], "attempt")
    if attempt.get("attempt") != 1 or attempt.get("status") != "PASS":
        raise ValueError("selected attempt did not pass")
    process_id = _integer(attempt.get("process_id"), "process_id", minimum=1)
    creation = _integer(
        attempt.get("process_creation_filetime_100ns"),
        "process_creation_filetime_100ns",
        minimum=1,
    )

    probe_path = _file_within(
        root, run_dir / "attempt-01" / "memory-probe.json", "memory probe"
    )
    probe = _mapping(
        json.loads(probe_path.read_text(encoding="utf-8")),
        "memory probe",
    )
    replay_binding = _mapping(
        probe.get("strict_command_replay"),
        "strict command replay",
    )
    replay_schema = replay_binding.get("schema")
    if replay_schema == STRICT_REPLAY_BINDING_SCHEMA:
        source_version = FULL_STATE_SOURCE_VERSION
        source_kind = FULL_STATE_SOURCE_KIND
    elif replay_schema == SOURCE_BOUND_REPLAY_BINDING_SCHEMA:
        source_version = SOURCE_BOUND_FULL_STATE_SOURCE_VERSION
        source_kind = SOURCE_BOUND_FULL_STATE_SOURCE_KIND
    else:
        raise ValueError("full-state source replay binding is unsupported")
    index_path = _file_within(
        root,
        run_dir / "attempt-01" / "trajectory" / "index.json",
        "trajectory index",
    )
    index = _mapping(
        json.loads(index_path.read_text(encoding="utf-8")), "trajectory index"
    )
    expected_ticks = end_update - start_update + 1
    if (
        index.get("start_update") != start_update
        or index.get("end_update") != end_update
        or index.get("freeze_update") != start_update
        or index.get("tick_count") != expected_ticks
    ):
        raise ValueError("trajectory window differs from requested window")
    identity = _mapping(index.get("process_identity"), "process identity")
    if (
        identity.get("process_id") != process_id
        or identity.get("process_creation_filetime_100ns") != creation
    ):
        raise ValueError("attempt and trajectory process identities differ")

    runtime_path = _file_within(root, Path(runtime_payload), "runtime payload")
    dmo_path = _file_within(root, Path(dmo), "DMO")
    pre_path = _file_within(root, Path(pre_snapshot), "pre snapshot")
    post_path = _file_within(root, Path(post_snapshot), "post snapshot")
    if end_update < start_update:
        raise ValueError("source window is invalid")
    if maximum_startup_attempts < 1:
        raise ValueError("maximum startup attempts is invalid")

    return {
        "schema": SOURCE_SCHEMA,
        "version": source_version,
        "source_id": source_id,
        "scope": {
            "level_id": "Jungle2",
            "hard": False,
            "profile_mode": "tutorials_completed",
            "mode": "adventure",
        },
        "original_executable_sha256": _sha256_path(original_executable),
        "runtime_payload": _binding(root, runtime_path),
        "dmo": _binding(root, dmo_path),
        "run": {
            "run_id": run_id,
            "selected_attempt": 1,
            "attempts": _binding(root, attempts_path),
            "memory_probe": _binding(root, probe_path),
            "trajectory_index": _binding(root, index_path),
            "process_id": process_id,
            "process_creation_filetime_100ns": creation,
            "maximum_startup_attempts": maximum_startup_attempts,
            "source_kind": source_kind,
        },
        "window": {
            "freeze_update": start_update,
            "start_update": start_update,
            "end_update": end_update,
            "warmup_tick_count": 0,
        },
        "state_transaction": {
            "pre_snapshot": _binding(root, pre_path),
            "post_snapshot": _binding(root, post_path),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--runtime-payload", required=True, type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--pre-snapshot", required=True, type=Path)
    parser.add_argument("--post-snapshot", required=True, type=Path)
    parser.add_argument("--start-update", required=True, type=int)
    parser.add_argument("--end-update", required=True, type=int)
    parser.add_argument("--maximum-startup-attempts", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent and its parent must exist")
    manifest = build_manifest(
        evidence_root=args.evidence_root,
        original_root=args.original_root,
        source_id=args.source_id,
        run_id=args.run_id,
        run_root=args.run_root,
        runtime_payload=args.runtime_payload,
        dmo=args.dmo,
        pre_snapshot=args.pre_snapshot,
        post_snapshot=args.post_snapshot,
        start_update=args.start_update,
        end_update=args.end_update,
        maximum_startup_attempts=args.maximum_startup_attempts,
    )
    staged = output.with_name(
        f".{output.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        write_pc_source_manifest(staged, manifest)
        report = verify_pc_source_manifest(
            staged,
            evidence_root=args.evidence_root,
            original_root=args.original_root,
        )
        if output.exists():
            raise ValueError("output appeared while source was being verified")
        staged.rename(output)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "path": str(output),
                "sha256": _sha256_path(output),
                "source_id": report["source_id"],
                "tick_count": report["tick_count"],
                "dmo_sha256": report["dmo_sha256"],
                "runtime_sha256": report["runtime_sha256"],
                "source_fingerprint": report["source_fingerprint"],
                "authorized_features": report["authorized_features"],
                "state_root": report["state_root"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
