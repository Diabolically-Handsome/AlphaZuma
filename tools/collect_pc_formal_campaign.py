"""Run a finite retained-attempt formal PC source campaign.

Each campaign attempt delegates to the unchanged one-shot formal-window
collector.  Only the exact pre-window source-bound global-RNG natural-state
mismatch is retryable.  Every started attempt remains on disk, host-state and
external-input receipts are checked before a retry, and the campaign stops at
the first formal PASS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_formal_window import (
    FormalWindowCollectionError,
    collect_formal_window,
)


CAMPAIGN_SCHEMA = "zuma-rl.pc-formal-source-campaign"
ATTEMPT_SCHEMA = "zuma-rl.pc-formal-source-campaign-attempt"
RETRYABLE_ERROR = "source_bound_global_precall_natural_state_mismatch"


class FormalCampaignError(RuntimeError):
    """The fixed formal-source campaign failed closed."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    with path.open("xb") as output:
        output.write(data)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCampaignError(
            f"campaign_evidence_invalid:{path}"
        ) from error


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalCampaignError(f"{label}_invalid")
    return value


def _descriptor(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise FormalCampaignError(
            f"campaign_artifact_missing:{path}"
        ) from error
    if not path.is_file():
        raise FormalCampaignError(f"campaign_artifact_missing:{path}")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": _sha256_path(path),
    }


def _optional_descriptor(path: Path) -> dict[str, Any] | None:
    return _descriptor(path) if path.is_file() else None


def _attempt_row(source_root: Path) -> Mapping[str, Any]:
    value = _read_json(source_root / "attempts.json")
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
        or value[0].get("attempt") != 1
    ):
        raise FormalCampaignError("inner_attempt_receipt_invalid")
    return value[0]


def _host_restore_binding(
    host_pre_path: Path,
    host_post_path: Path,
) -> dict[str, Any]:
    pre = _mapping(_read_json(host_pre_path), "host_pre")
    post = _mapping(_read_json(host_post_path), "host_post")
    pre_root = pre.get("state_root")
    post_root = post.get("state_root")
    if (
        not isinstance(pre_root, str)
        or not pre_root.startswith("sha256:")
        or len(pre_root) != 71
        or post_root != pre_root
    ):
        raise FormalCampaignError("host_state_restore_mismatch")
    return {
        "status": "PASS",
        "state_root": pre_root,
        "host_pre": _descriptor(host_pre_path),
        "host_post": _descriptor(host_post_path),
    }


def _external_input_binding(source_root: Path) -> dict[str, Any]:
    path = source_root / "attempt-01" / "external-input-guard.json"
    payload = _mapping(_read_json(path), "external_input_guard")
    counts = _mapping(payload.get("event_counts"), "external_event_counts")
    hooks = _mapping(payload.get("hooks"), "external_input_hooks")
    if (
        payload.get("status") != "PASS"
        or counts.get("external") != 0
        or payload.get("protocol_errors") != []
        or hooks.get("keyboard_unhooked") is not True
        or hooks.get("mouse_unhooked") is not True
    ):
        raise FormalCampaignError("external_input_guard_failed")
    return {
        "status": "PASS",
        "external_event_count": 0,
        "artifact": _descriptor(path),
    }


def _attempt_artifacts(source_root: Path) -> dict[str, Any]:
    attempt_root = source_root / "attempt-01"
    return {
        "attempts": _descriptor(source_root / "attempts.json"),
        "strict_replay": _optional_descriptor(
            attempt_root / "strict-replay.json"
        ),
        "strict_replay_log": _optional_descriptor(
            attempt_root / "strict-replay.log"
        ),
        "memory_probe": _optional_descriptor(
            attempt_root / "memory-probe.json"
        ),
        "trajectory_index": _optional_descriptor(
            attempt_root / "trajectory" / "index.json"
        ),
    }


def _campaign_receipt(
    *,
    status: str,
    plan_path: Path,
    prestate_path: Path,
    campaign_root: Path,
    maximum_attempts: int,
    nonce_stem: str,
    freeze_update: int,
    slowdown_update: int,
    trajectory_end_update: int,
    attempts: list[dict[str, Any]],
    selected_attempt: int | None,
    selected_source: Path | None,
    failure: str | None,
) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "version": 1,
        "status": status,
        "finished_utc": _now_utc(),
        "fixed_inputs": {
            "plan": _descriptor(plan_path),
            "prestate": _descriptor(prestate_path),
        },
        "campaign_root": str(campaign_root.resolve()),
        "maximum_attempts": maximum_attempts,
        "nonce_stem": nonce_stem,
        "window": {
            "slowdown_update": slowdown_update,
            "freeze_update": freeze_update,
            "trajectory_end_update": trajectory_end_update,
            "inclusive_tick_count": (
                trajectory_end_update - freeze_update + 1
            ),
        },
        "retryable_error": RETRYABLE_ERROR,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected_attempt": selected_attempt,
        "selected_source": (
            str(selected_source.resolve())
            if selected_source is not None
            else None
        ),
        "failure": failure,
        "selection": {
            "first_transport_pass_selected": selected_attempt is not None,
            "feature_presence_used": False,
            "post_window_outcome_used": False,
        },
    }


def collect_formal_campaign(
    *,
    plan_path: Path,
    prestate_path: Path,
    campaign_root: Path,
    maximum_attempts: int,
    nonce_stem: str,
    freeze_update: int,
    slowdown_update: int,
    trajectory_end_update: int,
) -> Path:
    """Run the fixed campaign and return its first formal source."""

    plan_path = plan_path.resolve()
    prestate_path = prestate_path.resolve()
    campaign_root = campaign_root.resolve()
    if (
        not plan_path.is_file()
        or not prestate_path.is_file()
        or campaign_root.exists()
        or not campaign_root.parent.is_dir()
        or isinstance(maximum_attempts, bool)
        or not 1 <= maximum_attempts <= 15
        or len(nonce_stem) != 31
        or any(char not in "0123456789abcdef" for char in nonce_stem)
        or slowdown_update < 0
        or freeze_update <= slowdown_update
        or trajectory_end_update < freeze_update
    ):
        raise FormalCampaignError("campaign_contract_invalid")

    campaign_root.mkdir()
    attempt_summaries: list[dict[str, Any]] = []
    for attempt_number in range(1, maximum_attempts + 1):
        attempt_root = campaign_root / f"attempt-{attempt_number:02d}"
        attempt_root.mkdir()
        source_root = attempt_root / "source"
        host_pre = attempt_root / "host-pre.json"
        host_post = attempt_root / "host-post.json"
        nonce = nonce_stem + format(attempt_number, "x")
        started_utc = _now_utc()
        try:
            source = collect_formal_window(
                plan_path=plan_path,
                prestate_path=prestate_path,
                output_root=source_root,
                host_pre_output=host_pre,
                host_post_output=host_post,
                session_nonce=nonce,
                freeze_update=freeze_update,
                slowdown_update=slowdown_update,
                trajectory_end_update=trajectory_end_update,
            )
        except FormalWindowCollectionError as collection_error:
            try:
                inner = _attempt_row(source_root)
                restore = _host_restore_binding(host_pre, host_post)
                external = _external_input_binding(source_root)
                inner_error = inner.get("error")
                retryable = (
                    inner.get("status") == "RETRY"
                    and inner_error == RETRYABLE_ERROR
                )
                attempt_payload = {
                    "schema": ATTEMPT_SCHEMA,
                    "version": 1,
                    "attempt": attempt_number,
                    "session_nonce": nonce,
                    "status": (
                        "RETRYABLE_TRANSPORT_MISMATCH"
                        if retryable
                        else "FAILED_CLOSED"
                    ),
                    "started_utc": started_utc,
                    "finished_utc": _now_utc(),
                    "collector_error": str(collection_error),
                    "inner_error": inner_error,
                    "host_restore": restore,
                    "external_input": external,
                    "artifacts": _attempt_artifacts(source_root),
                }
            except FormalCampaignError as evidence_error:
                failure = (
                    "attempt_evidence_validation_failed:"
                    f"{str(evidence_error)}"
                )
                receipt = _campaign_receipt(
                    status="FAILED_CLOSED",
                    plan_path=plan_path,
                    prestate_path=prestate_path,
                    campaign_root=campaign_root,
                    maximum_attempts=maximum_attempts,
                    nonce_stem=nonce_stem,
                    freeze_update=freeze_update,
                    slowdown_update=slowdown_update,
                    trajectory_end_update=trajectory_end_update,
                    attempts=attempt_summaries,
                    selected_attempt=None,
                    selected_source=None,
                    failure=failure,
                )
                _write_json_exclusive(campaign_root / "campaign.json", receipt)
                raise FormalCampaignError(failure) from evidence_error

            result_path = attempt_root / "attempt-result.json"
            _write_json_exclusive(result_path, attempt_payload)
            attempt_summaries.append(
                {
                    "attempt": attempt_number,
                    "status": attempt_payload["status"],
                    "inner_error": inner_error,
                    "result": _descriptor(result_path),
                }
            )
            if retryable:
                continue
            failure = f"non_retryable_attempt_failure:{inner_error}"
            receipt = _campaign_receipt(
                status="FAILED_CLOSED",
                plan_path=plan_path,
                prestate_path=prestate_path,
                campaign_root=campaign_root,
                maximum_attempts=maximum_attempts,
                nonce_stem=nonce_stem,
                freeze_update=freeze_update,
                slowdown_update=slowdown_update,
                trajectory_end_update=trajectory_end_update,
                attempts=attempt_summaries,
                selected_attempt=None,
                selected_source=None,
                failure=failure,
            )
            _write_json_exclusive(campaign_root / "campaign.json", receipt)
            raise FormalCampaignError(failure) from collection_error

        inner = _attempt_row(source_root)
        if inner.get("status") != "PASS":
            raise FormalCampaignError("successful_inner_attempt_invalid")
        source = source.resolve()
        try:
            source.relative_to(source_root.resolve())
        except ValueError as error:
            raise FormalCampaignError("selected_source_outside_attempt") from error
        if not source.is_file():
            raise FormalCampaignError("selected_source_missing")
        attempt_payload = {
            "schema": ATTEMPT_SCHEMA,
            "version": 1,
            "attempt": attempt_number,
            "session_nonce": nonce,
            "status": "PASS",
            "started_utc": started_utc,
            "finished_utc": _now_utc(),
            "collector_error": None,
            "inner_error": None,
            "host_restore": _host_restore_binding(host_pre, host_post),
            "external_input": _external_input_binding(source_root),
            "artifacts": _attempt_artifacts(source_root),
            "selected_source": _descriptor(source),
        }
        result_path = attempt_root / "attempt-result.json"
        _write_json_exclusive(result_path, attempt_payload)
        attempt_summaries.append(
            {
                "attempt": attempt_number,
                "status": "PASS",
                "inner_error": None,
                "result": _descriptor(result_path),
            }
        )
        receipt = _campaign_receipt(
            status="PASS",
            plan_path=plan_path,
            prestate_path=prestate_path,
            campaign_root=campaign_root,
            maximum_attempts=maximum_attempts,
            nonce_stem=nonce_stem,
            freeze_update=freeze_update,
            slowdown_update=slowdown_update,
            trajectory_end_update=trajectory_end_update,
            attempts=attempt_summaries,
            selected_attempt=attempt_number,
            selected_source=source,
            failure=None,
        )
        _write_json_exclusive(campaign_root / "campaign.json", receipt)
        return source

    receipt = _campaign_receipt(
        status="FAILED_CLOSED_EXHAUSTED",
        plan_path=plan_path,
        prestate_path=prestate_path,
        campaign_root=campaign_root,
        maximum_attempts=maximum_attempts,
        nonce_stem=nonce_stem,
        freeze_update=freeze_update,
        slowdown_update=slowdown_update,
        trajectory_end_update=trajectory_end_update,
        attempts=attempt_summaries,
        selected_attempt=None,
        selected_source=None,
        failure="campaign_exhausted",
    )
    _write_json_exclusive(campaign_root / "campaign.json", receipt)
    raise FormalCampaignError("campaign_exhausted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--maximum-attempts", required=True, type=int)
    parser.add_argument("--nonce-stem", required=True)
    parser.add_argument("--freeze-update", required=True, type=int)
    parser.add_argument("--slowdown-update", required=True, type=int)
    parser.add_argument(
        "--trajectory-end-update",
        required=True,
        type=int,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = collect_formal_campaign(
            plan_path=args.plan,
            prestate_path=args.prestate,
            campaign_root=args.campaign_root,
            maximum_attempts=args.maximum_attempts,
            nonce_stem=args.nonce_stem,
            freeze_update=args.freeze_update,
            slowdown_update=args.slowdown_update,
            trajectory_end_update=args.trajectory_end_update,
        )
    except FormalCampaignError as error:
        print(f"formal campaign error: {error}", file=sys.stderr)
        return 1
    print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
