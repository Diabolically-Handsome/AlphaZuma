"""Prepare and manage fail-closed startup-distribution evidence.

The first subcommand derives a direct-runtime plan from an already prepared
state transaction.  It can use either a debugger-free launch or the audited
startup command broker, and never edits the source plan.  Distribution-v3 PC
slots use the immutable slot seed schedule: the derived DMO differs from the
base DMO only in its four-byte framework gameplay-seed field, while the
unrelated natural CRT startup seed remains observed rather than controlled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_golden_v4 import (
    CollectionError,
    DIRECT_NATURAL_SEED_LAUNCH_MODE,
    DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE,
    EXPECTED_RUNTIME_SHA256,
    PLAN_SCHEMA,
    PLAN_VERSION,
    _font_cache_manifest_receipt,
)
from tools.pc_state_transaction import capture_state, write_snapshot_exclusive
from zuma_rl.pc_exact_step_evidence import (
    canonical_json_bytes,
    read_canonical_json,
)
from zuma_rl.distribution_fidelity import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    DATASET_SCHEMA,
    DATASET_VERSION,
    DMO_RANDOM_SEED_BYTES,
    DMO_RANDOM_SEED_OFFSET,
    GAMEPLAY_SEED_TRANSPORT,
    MAXIMUM_EXCESS_TOTAL_VARIATION,
    MAXIMUM_TOTAL_VARIATION,
    METRIC_NAME,
    MECHANISM_PROBE_SEED_ROLE,
    MINIMUM_COMPATIBILITY_P_VALUE,
    MINIMUM_PC_SAMPLES,
    MINIMUM_SIMULATOR_SAMPLES,
    NUM_COLORS,
    PC_EXPECTED_SCORE,
    PC_FREEZE_UPDATE,
    PC_SAMPLE_UPDATE,
    PC_WARMUP_TICK_COUNT,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    PREREGISTRATION_SCHEMA,
    PREREGISTRATION_VERSION,
    SEEDED_DMO_PROVENANCE_SCHEMA,
    SEEDED_DMO_PROVENANCE_VERSION,
    SEED_SCHEDULE_ALGORITHM,
    SEED_SCHEDULE_ENCODING,
    SEED_SCHEDULE_ID,
    SIMULATOR_SAMPLE_TICK,
    frozen_seed_schedules,
    mechanism_probe_seed,
    seed_sequence_sha256,
    verify_distribution_preregistration,
    verify_distribution_audit,
)
from zuma_rl.pc_protocol_evidence import PcStateSnapshot
from zuma_rl.pc_source import (
    STRICT_REPLAY_BINDING_SCHEMA,
    STRICT_REPLAY_BINDING_VERSION,
    verify_pc_source_manifest,
)
from zuma_rl.popcap_dmo import PopCapDemo


class DistributionCampaignError(RuntimeError):
    """A stable failure while preparing distribution evidence."""


CAMPAIGN_SCHEMA = "zuma-rl.startup-distribution-campaign"
CAMPAIGN_VERSION = 1
COLLECTION_JOBS_SCHEMA = "zuma-rl.startup-distribution-collection-jobs"
COLLECTION_JOBS_VERSION = 1
SEED_SCHEDULE_RECEIPT_SCHEMA = "zuma-rl.startup-distribution-seed-schedule"
SEED_SCHEDULE_RECEIPT_VERSION = 1
SLOT_EXECUTION_RECEIPT_SCHEMA = (
    "zuma-rl.startup-distribution-slot-execution"
)
SLOT_EXECUTION_RECEIPT_VERSION = 1
CAMPAIGN_COMPLETION_SCHEMA = "zuma-rl.startup-distribution-completion"
CAMPAIGN_COMPLETION_VERSION = 1
FINALIZATION_RECOVERY_SCHEMA = (
    "zuma-rl.startup-distribution-finalization-recovery"
)
FINALIZATION_RECOVERY_VERSION = 1
CAMPAIGN_INVALIDATION_SCHEMA = (
    "zuma-rl.startup-distribution-campaign-invalidation"
)
CAMPAIGN_INVALIDATION_VERSION = 1
PRE_SAMPLE_INFRASTRUCTURE_FAILURE = "PRE_SAMPLE_INFRASTRUCTURE_FAILURE"
PARTIAL_CAMPAIGN_INFRASTRUCTURE_FAILURE = (
    "PARTIAL_CAMPAIGN_INFRASTRUCTURE_FAILURE"
)
BROKER_RACE_FAILURE_CODE = (
    "BROKER_ADJACENT_CURRENT_UPDATE_FILE_WRITE_SCHEDULER_RACE"
)
REBASED_LATE_PAYLOAD_FAILURE_CODE = (
    "PRELOADING_FAILED_FILE_WRITE_REBASED_LATE_PAYLOAD_UNHANDLED"
)
TAIL_SHORT_HEADER_FAILURE_CODE = (
    "PRELOADING_FAILED_FILE_WRITE_TAIL_SHORT_HEADER_UNHANDLED"
)
POST_SAMPLE_FINALIZATION_OSERROR_CODE = (
    "POST_SAMPLE_ARTIFACT_FINALIZATION_OSERROR_22"
)
V7_FINALIZATION_RECOVERY_CAMPAIGN_ID = (
    "startup-actor-distribution-v7-controller-timeout-isolated-20260809"
)
V7_STALE_DISTRIBUTION_VALIDATOR_SHA256 = (
    "sha256:2fd3d290e4d59123abf7ecbe48a7b9d84c2acb01d691c4d5f7184eeaf69be439"
)
V7_CORRECTED_DISTRIBUTION_VALIDATOR_SHA256 = (
    "sha256:43620dd1c182b6a2d866b9e2b26a8d99ff1867d04069a3c6312bcd3c2936a422"
)
V7_PARTIAL_PC_DATASET_SHA256 = (
    "sha256:5b75b32a4851b05104ee2e93c3d328c1eb6112495265345a45ad2a289874777c"
)
V7_PARTIAL_AUDIT_SHA256 = (
    "sha256:77b04547957a06ef71bb1483d96253658cad00cc58ef67d1414ca3e88300533a"
)
DISTRIBUTION_POLICY_ID = "original-transfer-jungle2-v2"
DEFAULT_CAMPAIGN_ID = "startup-actor-distribution-v3-fixed-20260808"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DistributionCampaignError("artifact_unavailable") from error
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DistributionCampaignError(f"{name}_invalid")
    return value


def _strict_font_cache_manifest_receipt(
    changedir: Path,
    dmo: Path,
) -> dict[str, Any]:
    """Freeze the existing audited retail font-cache callback boundary."""

    try:
        return _font_cache_manifest_receipt(
            changedir,
            PopCapDemo.read(dmo),
            attach_at_update=0,
        )
    except (CollectionError, OSError, ValueError) as error:
        raise DistributionCampaignError(
            "strict_font_cache_manifest_invalid"
        ) from error


def _absolute_file(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DistributionCampaignError(f"{name}_unavailable") from error
    if not resolved.is_file():
        raise DistributionCampaignError(f"{name}_unavailable")
    return resolved


def _absolute_directory(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DistributionCampaignError(f"{name}_unavailable") from error
    if not resolved.is_dir():
        raise DistributionCampaignError(f"{name}_unavailable")
    return resolved


def _bound_member(root: Path, relative: Any, name: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
    ):
        raise DistributionCampaignError(f"{name}_invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise DistributionCampaignError(f"{name}_invalid")
    try:
        resolved = root.joinpath(*candidate.parts).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise DistributionCampaignError(f"{name}_invalid") from error
    if not resolved.is_file():
        raise DistributionCampaignError(f"{name}_invalid")
    return resolved


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value))
    try:
        with path.open("xb") as output:
            output.write(payload)
    except OSError as error:
        raise DistributionCampaignError("exclusive_write_failed") from error


def _read_strict_json_object(path: Path, name: str) -> Mapping[str, Any]:
    """Read a tracer JSON artifact without requiring canonical key order."""

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DistributionCampaignError(f"{name}_invalid") from error
    return _mapping(value, name)


def _evidence_relative(path: Path, root: Path, name: str) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as error:
        raise DistributionCampaignError(f"{name}_outside_evidence_root") from error
    if not relative.parts:
        raise DistributionCampaignError(f"{name}_invalid")
    return relative.as_posix()


def _seeded_dmo_payload(
    base_path: Path,
    *,
    gameplay_seed: int,
) -> tuple[bytes, list[int]]:
    try:
        base_bytes = base_path.read_bytes()
        base_demo = PopCapDemo.from_bytes(base_bytes)
    except (OSError, ValueError) as error:
        raise DistributionCampaignError("base_dmo_invalid") from error
    if len(base_bytes) < DMO_RANDOM_SEED_OFFSET + DMO_RANDOM_SEED_BYTES:
        raise DistributionCampaignError("base_dmo_invalid")
    derived = bytearray(base_bytes)
    struct.pack_into("<I", derived, DMO_RANDOM_SEED_OFFSET, gameplay_seed)
    try:
        derived_demo = PopCapDemo.from_bytes(bytes(derived))
    except ValueError as error:
        raise DistributionCampaignError("seeded_dmo_invalid") from error
    if (
        derived_demo.random_seed != gameplay_seed
        or derived_demo.version != base_demo.version
        or derived_demo.product_version != base_demo.product_version
        or derived_demo.length_updates != base_demo.length_updates
        or derived_demo.markers != base_demo.markers
        or derived_demo.commands != base_demo.commands
    ):
        raise DistributionCampaignError("seeded_dmo_semantics_changed")
    changed_offsets = [
        index
        for index, (before, after) in enumerate(
            zip(base_bytes, derived, strict=True)
        )
        if before != after
    ]
    if any(
        not DMO_RANDOM_SEED_OFFSET
        <= offset
        < DMO_RANDOM_SEED_OFFSET + DMO_RANDOM_SEED_BYTES
        for offset in changed_offsets
    ):
        raise DistributionCampaignError("seeded_dmo_semantics_changed")
    return bytes(derived), changed_offsets


def derive_natural_direct_plan(
    *,
    base_root: Path,
    output_root: Path,
    direct_runtime: Path,
    changedir: Path,
    strict_command_broker_stop_update: int | None = None,
    pc_slot_ordinal: int | None = None,
    mechanism_probe: bool = False,
    evidence_root: Path | None = None,
    frozen_font_cache_manifest_receipt: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically derive a direct retail plan with no runtime RNG writes."""

    base = _absolute_directory(base_root, "base_root")
    runtime_path = _absolute_file(direct_runtime, "direct_runtime")
    asset_root = _absolute_directory(changedir, "changedir")
    output = output_root.resolve()
    staging = output.with_name(output.name + ".part")
    if (
        not output.is_absolute()
        or output.exists()
        or staging.exists()
        or not output.parent.is_dir()
    ):
        raise DistributionCampaignError("output_root_invalid")
    if strict_command_broker_stop_update is not None and (
        isinstance(strict_command_broker_stop_update, bool)
        or strict_command_broker_stop_update <= 0
    ):
        raise DistributionCampaignError("strict_broker_stop_invalid")
    if pc_slot_ordinal is not None and (
        isinstance(pc_slot_ordinal, bool)
        or not isinstance(pc_slot_ordinal, int)
        or not 1 <= pc_slot_ordinal <= MINIMUM_PC_SAMPLES
    ):
        raise DistributionCampaignError("pc_slot_ordinal_invalid")
    if not isinstance(mechanism_probe, bool) or (
        mechanism_probe and pc_slot_ordinal is not None
    ):
        raise DistributionCampaignError("seed_role_invalid")
    seeded_dmo_requested = pc_slot_ordinal is not None or mechanism_probe
    if seeded_dmo_requested != (evidence_root is not None):
        raise DistributionCampaignError("seeded_dmo_evidence_root_required")
    if (
        frozen_font_cache_manifest_receipt is not None
        and strict_command_broker_stop_update is None
    ):
        raise DistributionCampaignError("font_cache_receipt_without_broker")
    evidence = (
        _absolute_directory(evidence_root, "evidence_root")
        if evidence_root is not None
        else None
    )

    plan_path = base / "plan.json"
    try:
        plan = dict(_mapping(read_canonical_json(plan_path), "base_plan"))
    except (OSError, ValueError) as error:
        raise DistributionCampaignError("base_plan_invalid") from error
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("version") != PLAN_VERSION
    ):
        raise DistributionCampaignError("base_plan_invalid")
    runtime = dict(_mapping(plan.get("runtime"), "base_runtime"))
    trace = dict(_mapping(plan.get("trace"), "base_trace"))
    dmo = dict(_mapping(plan.get("dmo"), "base_dmo"))
    prestate = _mapping(plan.get("prestate"), "base_prestate")
    safety = _mapping(plan.get("safety"), "base_safety")

    dmo_path = _bound_member(base, dmo.get("artifact"), "base_dmo")
    template_path = _bound_member(
        base,
        prestate.get("template_artifact"),
        "base_prestate",
    )
    host_path = _bound_member(
        base,
        safety.get("host_pre_artifact"),
        "base_host_snapshot",
    )
    source_path = _absolute_file(
        Path(str(runtime.get("runtime_source_executable"))),
        "runtime_source",
    )
    if (
        _sha256_path(dmo_path) != dmo.get("sha256")
        or _sha256_path(source_path)
        != runtime.get("runtime_source_sha256")
        or _sha256_path(runtime_path) != EXPECTED_RUNTIME_SHA256
        or runtime.get("expected_runtime_sha256")
        != EXPECTED_RUNTIME_SHA256
    ):
        raise DistributionCampaignError("base_identity_mismatch")

    gameplay_seed: int | None = None
    seeded_dmo_bytes: bytes | None = None
    seeded_changed_offsets: list[int] | None = None
    seed_provenance_relative = Path("inputs") / "input.dmo.seed.json"
    seed_role: str | None = None
    if seeded_dmo_requested:
        assert evidence is not None
        _evidence_relative(dmo_path, evidence, "base_dmo")
        _evidence_relative(output / dmo["artifact"], evidence, "seeded_dmo")
        if mechanism_probe:
            gameplay_seed = mechanism_probe_seed()
            seed_role = MECHANISM_PROBE_SEED_ROLE
        else:
            assert pc_slot_ordinal is not None
            gameplay_seed = frozen_seed_schedules()[0][pc_slot_ordinal - 1]
            seed_role = f"formal-pc-slot-{pc_slot_ordinal:03d}"
        seeded_dmo_bytes, seeded_changed_offsets = _seeded_dmo_payload(
            dmo_path,
            gameplay_seed=gameplay_seed,
        )
        dmo.update(
            {
                "bytes": len(seeded_dmo_bytes),
                "random_seed": gameplay_seed,
                "sha256": _sha256_bytes(seeded_dmo_bytes),
            }
        )
    plan["dmo"] = dmo

    launch_mode = (
        DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
        if strict_command_broker_stop_update is not None
        else DIRECT_NATURAL_SEED_LAUNCH_MODE
    )
    strict_broker = strict_command_broker_stop_update is not None
    font_cache_manifest_receipt = (
        dict(frozen_font_cache_manifest_receipt)
        if frozen_font_cache_manifest_receipt is not None
        else None
    )
    if font_cache_manifest_receipt is not None and (
        isinstance(font_cache_manifest_receipt.get("entry_count"), bool)
        or not isinstance(font_cache_manifest_receipt.get("entry_count"), int)
        or font_cache_manifest_receipt["entry_count"] <= 0
        or not isinstance(
            font_cache_manifest_receipt.get("manifest_sha256"),
            str,
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            font_cache_manifest_receipt["manifest_sha256"],
        )
        or not isinstance(
            font_cache_manifest_receipt.get("main_pak_sha256"),
            str,
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            font_cache_manifest_receipt["main_pak_sha256"],
        )
    ):
        raise DistributionCampaignError("font_cache_receipt_invalid")
    runtime.update(
        {
            "launch_mode": launch_mode,
            "runtime_executable": str(runtime_path),
            "runtime_executable_sha256": EXPECTED_RUNTIME_SHA256,
            "expected_runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "changedir": str(asset_root),
            "crt_rand_seed": None,
            "startup_seed_transport": None,
            "board_seed_call_address": None,
            "board_seed": None,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
        }
    )
    trace.update(
        {
            "attach_at_update": 0,
            "detach_at_update": None,
            "reattach_at_update": None,
            "allow_pre_stream_commands": strict_broker,
            "startup_trace_handoff": strict_broker,
            "natural_command_broker_stop_after_update": (
                strict_command_broker_stop_update
            ),
            "allow_attach_stabilization": False,
            "allow_pre_attach_file_write_debt": False,
            "allow_font_cache_manifest_completion_debt": strict_broker,
            "font_cache_manifest_receipt": font_cache_manifest_receipt,
            "seed_board_before_attach": False,
            "startup_priority_bias_until_update": None,
            "startup_process_affinity_mask": None,
            "source_bound_board_anchor": None,
            "source_bound_board_precall_global_restore": None,
            "gameplay_mtrand_sync": None,
            "allow_source_bound_board_global_correction": False,
        }
    )
    plan["runtime"] = runtime
    plan["trace"] = trace
    plan["state_comparison"] = {
        "projection": "gameplay_state_projection_v1",
        "volatile_registry_roles": list(
            plan.get("state_comparison", {}).get(
                "volatile_registry_roles",
                [],
            )
        ),
        "launcher_registry_contract": "preserved_from_prestate",
    }

    try:
        staging.mkdir()
        shutil.copytree(base / "inputs", staging / "inputs")
        staged_dmo_path = staging / dmo["artifact"]
        seed_provenance: dict[str, Any] | None = None
        if seeded_dmo_bytes is not None:
            assert evidence is not None
            assert gameplay_seed is not None
            assert seeded_changed_offsets is not None
            staged_dmo_path.write_bytes(seeded_dmo_bytes)
            final_dmo_path = output / dmo["artifact"]
            base_binding = {
                "path": _evidence_relative(dmo_path, evidence, "base_dmo"),
                "sha256": _sha256_path(dmo_path),
            }
            derived_binding = {
                "path": _evidence_relative(
                    final_dmo_path,
                    evidence,
                    "seeded_dmo",
                ),
                "sha256": dmo["sha256"],
            }
            seed_provenance = {
                "schema": SEEDED_DMO_PROVENANCE_SCHEMA,
                "version": SEEDED_DMO_PROVENANCE_VERSION,
                "status": "PASS",
                "base_dmo": base_binding,
                "derived_dmo": derived_binding,
                "random_seed": gameplay_seed,
                "seed_offset": DMO_RANDOM_SEED_OFFSET,
                "seed_size_bytes": DMO_RANDOM_SEED_BYTES,
                "changed_byte_offsets": seeded_changed_offsets,
                "unchanged_byte_count": (
                    len(seeded_dmo_bytes) - len(seeded_changed_offsets)
                ),
            }
            _write_exclusive(
                staging / seed_provenance_relative,
                seed_provenance,
            )
        if strict_broker and font_cache_manifest_receipt is None:
            font_cache_manifest_receipt = (
                _strict_font_cache_manifest_receipt(
                    asset_root,
                    staged_dmo_path,
                )
            )
        if strict_broker:
            trace["font_cache_manifest_receipt"] = (
                font_cache_manifest_receipt
            )
            plan["trace"] = trace
        (staging / "protocol").mkdir()
        shutil.copy2(template_path, staging / "protocol" / template_path.name)
        (staging / "safety").mkdir()
        shutil.copy2(host_path, staging / "safety" / host_path.name)
        _write_exclusive(staging / "plan.json", plan)
        receipt = {
            "schema": "zuma-rl.startup-distribution-plan-derivation",
            "version": 3,
            "status": "PASS",
            "base_plan_sha256": _sha256_path(plan_path),
            "derived_plan_sha256": _sha256_path(staging / "plan.json"),
            "launch_mode": launch_mode,
            "runtime_payload_sha256": EXPECTED_RUNTIME_SHA256,
            "base_dmo_sha256": _sha256_path(dmo_path),
            "dmo_sha256": dmo["sha256"],
            "pc_slot_ordinal": pc_slot_ordinal,
            "seed_role": seed_role,
            "gameplay_seed": gameplay_seed,
            "gameplay_seed_transport": (
                GAMEPLAY_SEED_TRANSPORT
                if gameplay_seed is not None
                else None
            ),
            "seed_provenance": (
                {
                    "artifact": seed_provenance_relative.as_posix(),
                    "sha256": _sha256_path(
                        staging / seed_provenance_relative
                    ),
                }
                if seed_provenance is not None
                else None
            ),
            "seed_controls": {
                "crt_rand_seed": None,
                "board_seed": None,
                "global_rng_seed": None,
                "thread_crt_rng_seed": None,
            },
            "command_debugger_attached": (
                strict_command_broker_stop_update is not None
            ),
            "command_broker": {
                "enabled": strict_broker,
                "attach_at_update": (
                    0
                    if strict_broker
                    else None
                ),
                "stop_after_update": strict_command_broker_stop_update,
                "rng_seed_override_count": 0,
                "font_cache_manifest_completion": (
                    {
                        "enabled": True,
                        "scope": (
                            "deferred_font_cache_file_write_false_return_only"
                        ),
                        "entry_count": font_cache_manifest_receipt[
                            "entry_count"
                        ],
                        "manifest_sha256": font_cache_manifest_receipt[
                            "manifest_sha256"
                        ],
                        "main_pak_sha256": font_cache_manifest_receipt[
                            "main_pak_sha256"
                        ],
                        "rng_process_memory_writes": 0,
                    }
                    if font_cache_manifest_receipt is not None
                    else None
                ),
            },
            "source_plan_modified": False,
            "same_dmo_replay_counted_as_distribution": False,
        }
        _write_exclusive(staging / "derivation.json", receipt)
        os.replace(staging, output)
    except DistributionCampaignError:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    except OSError as error:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise DistributionCampaignError("plan_derivation_failed") from error
    return output / "plan.json"


def _artifact_binding(path: Path, root: Path) -> dict[str, str]:
    resolved = _absolute_file(path, "bound_artifact")
    return {
        "path": _evidence_relative(resolved, root, "bound_artifact"),
        "sha256": _sha256_path(resolved),
    }


def _frozen_utc(value: str | None) -> str:
    result = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if value is None
        else value
    )
    if re.fullmatch(
        r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:"
        r"[0-5][0-9]:[0-5][0-9]Z",
        result,
    ) is None:
        raise DistributionCampaignError("frozen_utc_invalid")
    return result


def prepare_distribution_campaign(
    *,
    base_root: Path,
    output_root: Path,
    direct_runtime: Path,
    changedir: Path,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    frozen_utc: str | None = None,
) -> Path:
    """Freeze all 32 PC slots and 256 simulator seeds before collection."""

    if _IDENTIFIER_PATTERN.fullmatch(campaign_id) is None:
        raise DistributionCampaignError("campaign_id_invalid")
    base = _absolute_directory(base_root, "base_root")
    runtime_source = _absolute_file(direct_runtime, "direct_runtime")
    asset_root = _absolute_directory(changedir, "changedir")
    output = output_root.resolve()
    if (
        not output.is_absolute()
        or output.exists()
        or not output.parent.is_dir()
    ):
        raise DistributionCampaignError("output_root_invalid")
    timestamp = _frozen_utc(frozen_utc)

    project_root = Path(__file__).resolve().parents[1]
    source_tools = {
        "collector": _absolute_file(
            project_root / "tools" / "collect_pc_memory_probe.py",
            "collector_tool",
        ),
        "trace_tool": _absolute_file(
            project_root / "tools" / "trace_popcap_demo_commands.py",
            "trace_tool",
        ),
        "replay_boundary_tool": _absolute_file(
            project_root / "tools" / "popcap_replay_boundary.py",
            "replay_boundary_tool",
        ),
        "seed_deriver": _absolute_file(Path(__file__), "seed_deriver_tool"),
    }

    try:
        output.mkdir()
        frozen_root = output / "frozen"
        frozen_root.mkdir()
        frozen_base = frozen_root / "base-plan"
        shutil.copytree(base, frozen_base)
        frozen_runtime = frozen_root / "runtime" / runtime_source.name
        frozen_runtime.parent.mkdir()
        shutil.copy2(runtime_source, frozen_runtime)
        frozen_tools_root = frozen_root / "tools"
        frozen_tools_root.mkdir()
        frozen_tools: dict[str, Path] = {}
        for role, source in source_tools.items():
            target = frozen_tools_root / source.name
            shutil.copy2(source, target)
            frozen_tools[role] = target

        base_plan_path = frozen_base / "plan.json"
        base_plan = _mapping(
            read_canonical_json(base_plan_path),
            "frozen_base_plan",
        )
        base_dmo_spec = _mapping(base_plan.get("dmo"), "frozen_base_dmo")
        base_dmo_path = _bound_member(
            frozen_base,
            base_dmo_spec.get("artifact"),
            "frozen_base_dmo",
        )
        base_prestate = _mapping(
            base_plan.get("prestate"),
            "frozen_base_prestate",
        )
        replay_prestate_path = _bound_member(
            frozen_base,
            base_prestate.get("template_artifact"),
            "frozen_replay_prestate",
        )
        manifest_receipt = _strict_font_cache_manifest_receipt(
            asset_root,
            base_dmo_path,
        )

        pc_seeds, simulator_seeds = frozen_seed_schedules()
        seed_schedule_path = output / "seed-schedule.json"
        _write_exclusive(
            seed_schedule_path,
            {
                "schema": SEED_SCHEDULE_RECEIPT_SCHEMA,
                "version": SEED_SCHEDULE_RECEIPT_VERSION,
                "status": "FROZEN_BEFORE_DATA",
                "id": SEED_SCHEDULE_ID,
                "algorithm": SEED_SCHEDULE_ALGORITHM,
                "encoding": SEED_SCHEDULE_ENCODING,
                "pc_seeds": list(pc_seeds),
                "pc_seed_sha256": seed_sequence_sha256(pc_seeds),
                "simulator_seeds": list(simulator_seeds),
                "simulator_seed_sha256": seed_sequence_sha256(
                    simulator_seeds
                ),
                "populations_disjoint": not bool(
                    set(pc_seeds) & set(simulator_seeds)
                ),
            },
        )

        plans_root = output / "plans"
        captures_root = output / "captures"
        plans_root.mkdir()
        captures_root.mkdir()
        slots: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        for ordinal, gameplay_seed in enumerate(pc_seeds, start=1):
            slot_name = f"pc-{ordinal:03d}"
            plan_root = plans_root / slot_name
            plan_path = derive_natural_direct_plan(
                base_root=frozen_base,
                output_root=plan_root,
                direct_runtime=frozen_runtime,
                changedir=asset_root,
                strict_command_broker_stop_update=3151,
                pc_slot_ordinal=ordinal,
                evidence_root=output,
                frozen_font_cache_manifest_receipt=manifest_receipt,
            )
            derived_plan = _mapping(
                read_canonical_json(plan_path),
                "seeded_plan",
            )
            derived_dmo = _bound_member(
                plan_root,
                _mapping(derived_plan.get("dmo"), "seeded_plan_dmo").get(
                    "artifact"
                ),
                "seeded_plan_dmo",
            )
            derived_prestate = _bound_member(
                plan_root,
                _mapping(
                    derived_plan.get("prestate"),
                    "seeded_plan_prestate",
                ).get("template_artifact"),
                "seeded_plan_prestate",
            )
            derived_host_restore = _bound_member(
                plan_root,
                _mapping(
                    derived_plan.get("safety"),
                    "seeded_plan_safety",
                ).get("host_pre_artifact"),
                "seeded_plan_host_restore",
            )
            provenance = plan_root / "inputs" / "input.dmo.seed.json"
            capture_root = captures_root / slot_name
            source_id = f"startup-dist-v3-source-{ordinal:03d}"
            run_id = f"startup-dist-v3-run-{ordinal:03d}"
            slots.append(
                {
                    "ordinal": ordinal,
                    "source_id": source_id,
                    "run_id": run_id,
                    "source_manifest_path": _evidence_relative(
                        capture_root / "manifest.json",
                        output,
                        "source_manifest",
                    ),
                    "output_root_path": _evidence_relative(
                        capture_root,
                        output,
                        "capture_root",
                    ),
                    "gameplay_seed": gameplay_seed,
                    "dmo": _artifact_binding(derived_dmo, output),
                    "seed_provenance": _artifact_binding(
                        provenance,
                        output,
                    ),
                    "collector_plan": _artifact_binding(plan_path, output),
                }
            )
            jobs.append(
                {
                    "ordinal": ordinal,
                    "status": "NOT_STARTED",
                    "gameplay_seed": gameplay_seed,
                    "plan": _evidence_relative(
                        plan_path,
                        output,
                        "job_plan",
                    ),
                    "prestate": _evidence_relative(
                        derived_prestate,
                        output,
                        "job_prestate",
                    ),
                    "host_restore": _evidence_relative(
                        derived_host_restore,
                        output,
                        "job_host_restore",
                    ),
                    "output_root": _evidence_relative(
                        capture_root,
                        output,
                        "job_output_root",
                    ),
                    "arguments": [
                        "--probe-update",
                        str(PC_FREEZE_UPDATE),
                        "--slowdown-update",
                        str(PC_FREEZE_UPDATE - 5),
                        "--discover-score",
                        "--skip-int32-scan",
                        "--maximum-attempts",
                        "1",
                        "--trajectory-start-update",
                        str(PC_SAMPLE_UPDATE),
                        "--trajectory-end-update",
                        str(PC_SAMPLE_UPDATE),
                        "--trajectory-mode",
                        "rng",
                        "--skip-repaint-guard",
                        "--skip-frozen-snapshot",
                        "--snapshot-trajectory-frames",
                        "--snapshot-square-dwm-corners",
                        "--snapshot-trajectory-warmup-frame",
                        "--formal-exact-step-evidence",
                    ],
                }
            )

        jobs_path = output / "collection-jobs.json"
        _write_exclusive(
            jobs_path,
            {
                "schema": COLLECTION_JOBS_SCHEMA,
                "version": COLLECTION_JOBS_VERSION,
                "status": "FROZEN_BEFORE_DATA",
                "collector": _artifact_binding(
                    frozen_tools["collector"],
                    output,
                ),
                "replay_boundary_tool": _artifact_binding(
                    frozen_tools["replay_boundary_tool"],
                    output,
                ),
                "maximum_started_processes_per_slot": 1,
                "replacement_after_observation_forbidden": True,
                "jobs": jobs,
            },
        )
        simulator_dataset_path = output / "simulator-dataset.json"
        _write_exclusive(
            simulator_dataset_path,
            {
                "schema": DATASET_SCHEMA,
                "version": DATASET_VERSION,
                "population": "simulator",
                "metric": METRIC_NAME,
                "samples": [{"seed": seed} for seed in simulator_seeds],
            },
        )

        preregistration_path = output / "preregistration.json"
        scope = {
            "policy": DISTRIBUTION_POLICY_ID,
            "environment_id": "ZumaRevenge-v0",
            "level_id": "Jungle2",
            "hard": False,
            "profile_mode": "tutorials_completed",
        }
        _write_exclusive(
            preregistration_path,
            {
                "schema": PREREGISTRATION_SCHEMA,
                "version": PREREGISTRATION_VERSION,
                "status": "FROZEN_BEFORE_DATA",
                "frozen_utc": timestamp,
                "scope": scope,
                "metric": {
                    "name": METRIC_NAME,
                    "kind": "categorical_pair",
                    "minimum_pc_samples": MINIMUM_PC_SAMPLES,
                    "minimum_simulator_samples": MINIMUM_SIMULATOR_SAMPLES,
                    "num_colors": NUM_COLORS,
                    "maximum_total_variation": MAXIMUM_TOTAL_VARIATION,
                    "maximum_excess_total_variation": (
                        MAXIMUM_EXCESS_TOTAL_VARIATION
                    ),
                    "minimum_compatibility_p_value": (
                        MINIMUM_COMPATIBILITY_P_VALUE
                    ),
                    "permutation_count": PERMUTATION_COUNT,
                    "permutation_seed": PERMUTATION_SEED,
                },
                "pc_capture": {
                    "freeze_update": PC_FREEZE_UPDATE,
                    "sample_update": PC_SAMPLE_UPDATE,
                    "warmup_tick_count": PC_WARMUP_TICK_COUNT,
                    "expected_score": PC_EXPECTED_SCORE,
                    "base_dmo": _artifact_binding(base_dmo_path, output),
                    "runtime_payload": _artifact_binding(
                        frozen_runtime,
                        output,
                    ),
                    "base_collector_plan": _artifact_binding(
                        base_plan_path,
                        output,
                    ),
                    "replay_prestate": _artifact_binding(
                        replay_prestate_path,
                        output,
                    ),
                    "collector": _artifact_binding(
                        frozen_tools["collector"],
                        output,
                    ),
                    "trace_tool": _artifact_binding(
                        frozen_tools["trace_tool"],
                        output,
                    ),
                    "replay_boundary_tool": _artifact_binding(
                        frozen_tools["replay_boundary_tool"],
                        output,
                    ),
                    "seed_deriver": _artifact_binding(
                        frozen_tools["seed_deriver"],
                        output,
                    ),
                    "gameplay_seed_transport": GAMEPLAY_SEED_TRANSPORT,
                    "strict_command_replay_required": True,
                },
                "selection": {
                    "campaign_id": campaign_id,
                    "pc_process_count": MINIMUM_PC_SAMPLES,
                    "pc_slots": slots,
                    "seed_schedule": {
                        "id": SEED_SCHEDULE_ID,
                        "algorithm": SEED_SCHEDULE_ALGORITHM,
                        "encoding": SEED_SCHEDULE_ENCODING,
                        "pc_seed_count": MINIMUM_PC_SAMPLES,
                        "pc_seed_sha256": seed_sequence_sha256(pc_seeds),
                        "simulator_seed_count": MINIMUM_SIMULATOR_SAMPLES,
                        "simulator_seed_sha256": seed_sequence_sha256(
                            simulator_seeds
                        ),
                        "populations_disjoint": True,
                    },
                    "simulator_seed_count": MINIMUM_SIMULATOR_SAMPLES,
                    "simulator_sample_tick": SIMULATOR_SAMPLE_TICK,
                    "retain_every_started_pc_process": True,
                    "replacement_after_observation_forbidden": True,
                },
            },
        )
        verification = verify_distribution_preregistration(
            preregistration_path,
            evidence_root=output,
            policy_id=DISTRIBUTION_POLICY_ID,
        )
        verification_path = output / "preregistration-verification.json"
        _write_exclusive(verification_path, verification)
        campaign_path = output / "campaign.json"
        _write_exclusive(
            campaign_path,
            {
                "schema": CAMPAIGN_SCHEMA,
                "version": CAMPAIGN_VERSION,
                "status": "FROZEN_BEFORE_DATA",
                "campaign_id": campaign_id,
                "frozen_utc": timestamp,
                "policy_id": DISTRIBUTION_POLICY_ID,
                "preregistration": _artifact_binding(
                    preregistration_path,
                    output,
                ),
                "preregistration_verification": _artifact_binding(
                    verification_path,
                    output,
                ),
                "seed_schedule": _artifact_binding(
                    seed_schedule_path,
                    output,
                ),
                "collection_jobs": _artifact_binding(jobs_path, output),
                "simulator_dataset": _artifact_binding(
                    simulator_dataset_path,
                    output,
                ),
                "pc_dataset": None,
                "audit": None,
                "pc_samples_observed": 0,
                "same_dmo_replay_counted_as_distribution": False,
            },
        )
        return campaign_path
    except DistributionCampaignError:
        if output.is_dir():
            shutil.rmtree(output)
        raise
    except (OSError, ValueError) as error:
        if output.is_dir():
            shutil.rmtree(output)
        raise DistributionCampaignError("campaign_preparation_failed") from error


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DistributionCampaignError(f"{name}_invalid")
    return value


def _bound_artifact(
    root: Path,
    value: Any,
    name: str,
) -> Path:
    binding = _mapping(value, name)
    if set(binding) != {"path", "sha256"}:
        raise DistributionCampaignError(f"{name}_invalid")
    path = _bound_member(root, binding.get("path"), name)
    if _sha256_path(path) != binding.get("sha256"):
        raise DistributionCampaignError(f"{name}_identity_mismatch")
    return path


def _future_member(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DistributionCampaignError(f"{name}_invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise DistributionCampaignError(f"{name}_invalid")
    resolved = root.joinpath(*candidate.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DistributionCampaignError(f"{name}_invalid") from error
    return resolved


def _load_campaign_context(
    campaign_root: Path,
    *,
    require_current_collection_tools: bool = True,
) -> dict[str, Any]:
    root = _absolute_directory(campaign_root, "campaign_root")
    campaign_path = _absolute_file(root / "campaign.json", "campaign")
    try:
        campaign = dict(_mapping(read_canonical_json(campaign_path), "campaign"))
    except (OSError, ValueError) as error:
        raise DistributionCampaignError("campaign_invalid") from error
    if (
        set(campaign)
        != {
            "schema",
            "version",
            "status",
            "campaign_id",
            "frozen_utc",
            "policy_id",
            "preregistration",
            "preregistration_verification",
            "seed_schedule",
            "collection_jobs",
            "simulator_dataset",
            "pc_dataset",
            "audit",
            "pc_samples_observed",
            "same_dmo_replay_counted_as_distribution",
        }
        or campaign.get("schema") != CAMPAIGN_SCHEMA
        or campaign.get("version") != CAMPAIGN_VERSION
        or campaign.get("status") != "FROZEN_BEFORE_DATA"
        or campaign.get("policy_id") != DISTRIBUTION_POLICY_ID
        or campaign.get("pc_dataset") is not None
        or campaign.get("audit") is not None
        or campaign.get("pc_samples_observed") != 0
        or campaign.get("same_dmo_replay_counted_as_distribution") is not False
    ):
        raise DistributionCampaignError("campaign_invalid")
    preregistration_path = _bound_artifact(
        root,
        campaign["preregistration"],
        "campaign_preregistration",
    )
    preregistration_verification_path = _bound_artifact(
        root,
        campaign["preregistration_verification"],
        "campaign_preregistration_verification",
    )
    jobs_path = _bound_artifact(
        root,
        campaign["collection_jobs"],
        "campaign_collection_jobs",
    )
    simulator_dataset_path = _bound_artifact(
        root,
        campaign["simulator_dataset"],
        "campaign_simulator_dataset",
    )
    _bound_artifact(root, campaign["seed_schedule"], "campaign_seed_schedule")
    verification = _mapping(
        read_canonical_json(preregistration_verification_path),
        "preregistration_verification",
    )
    if verification.get("status") != "PASS":
        raise DistributionCampaignError("preregistration_verification_failed")
    recomputed = verify_distribution_preregistration(
        preregistration_path,
        evidence_root=root,
        policy_id=DISTRIBUTION_POLICY_ID,
    )
    if recomputed != verification:
        raise DistributionCampaignError("preregistration_verification_drift")
    preregistration = _mapping(
        read_canonical_json(preregistration_path),
        "preregistration",
    )
    selection = _mapping(preregistration.get("selection"), "selection")
    slots = _sequence(selection.get("pc_slots"), "pc_slots")
    jobs_document = _mapping(read_canonical_json(jobs_path), "collection_jobs")
    jobs = _sequence(jobs_document.get("jobs"), "collection_jobs_rows")
    if (
        campaign.get("campaign_id") != selection.get("campaign_id")
        or jobs_document.get("schema") != COLLECTION_JOBS_SCHEMA
        or jobs_document.get("version") != COLLECTION_JOBS_VERSION
        or jobs_document.get("status") != "FROZEN_BEFORE_DATA"
        or jobs_document.get("maximum_started_processes_per_slot") != 1
        or jobs_document.get("replacement_after_observation_forbidden") is not True
        or len(slots) != MINIMUM_PC_SAMPLES
        or len(jobs) != MINIMUM_PC_SAMPLES
    ):
        raise DistributionCampaignError("campaign_selection_invalid")
    capture = _mapping(preregistration.get("pc_capture"), "pc_capture")
    frozen_collector = _bound_artifact(
        root,
        capture.get("collector"),
        "frozen_collector",
    )
    if jobs_document.get("collector") != capture.get("collector"):
        raise DistributionCampaignError("collector_binding_mismatch")
    frozen_trace = _bound_artifact(
        root,
        capture.get("trace_tool"),
        "frozen_trace_tool",
    )
    replay_boundary_binding = capture.get("replay_boundary_tool")
    frozen_replay_boundary = (
        _bound_artifact(
            root,
            replay_boundary_binding,
            "frozen_replay_boundary_tool",
        )
        if replay_boundary_binding is not None
        else None
    )
    if jobs_document.get("replay_boundary_tool") != replay_boundary_binding:
        raise DistributionCampaignError("replay_boundary_binding_mismatch")
    current_collector = Path(__file__).resolve().parent / "collect_pc_memory_probe.py"
    current_trace = Path(__file__).resolve().parent / "trace_popcap_demo_commands.py"
    current_replay_boundary = (
        Path(__file__).resolve().parent / "popcap_replay_boundary.py"
    )
    if require_current_collection_tools and (
        frozen_replay_boundary is None
        or _sha256_path(current_collector) != _sha256_path(frozen_collector)
        or _sha256_path(current_trace) != _sha256_path(frozen_trace)
        or _sha256_path(current_replay_boundary)
        != _sha256_path(frozen_replay_boundary)
    ):
        raise DistributionCampaignError("frozen_collection_tool_drift")
    return {
        "root": root,
        "campaign": campaign,
        "preregistration": preregistration,
        "preregistration_path": preregistration_path,
        "simulator_dataset_path": simulator_dataset_path,
        "slots": slots,
        "jobs": jobs,
        "current_collector": current_collector,
        "collector_sha256": _sha256_path(current_collector),
        "trace_sha256": _sha256_path(current_trace),
        "replay_boundary_sha256": _sha256_path(current_replay_boundary),
    }


def invalidate_failed_distribution_campaign(
    *,
    campaign_root: Path,
    superseded_by_campaign_id: str,
    invalidated_utc: str | None = None,
) -> Path:
    """Append one fail-closed receipt for the exact pre-sample PC-001 race."""

    if (
        _IDENTIFIER_PATTERN.fullmatch(superseded_by_campaign_id) is None
    ):
        raise DistributionCampaignError("superseding_campaign_id_invalid")
    context = _load_campaign_context(
        campaign_root,
        require_current_collection_tools=False,
    )
    root = Path(context["root"])
    campaign = _mapping(context["campaign"], "campaign")
    campaign_id = campaign.get("campaign_id")
    if superseded_by_campaign_id == campaign_id:
        raise DistributionCampaignError("superseding_campaign_id_invalid")
    receipt_path = root / "invalidation.json"
    if receipt_path.exists():
        raise DistributionCampaignError("campaign_invalidation_already_exists")
    if any(
        (root / name).exists()
        for name in (
            "pc-dataset.json",
            "audit.json",
            "audit-verification.json",
            "completion.json",
        )
    ):
        raise DistributionCampaignError("failed_campaign_has_final_evidence")

    slots = _sequence(context["slots"], "pc_slots")
    jobs = _sequence(context["jobs"], "collection_jobs_rows")
    if len(slots) != MINIMUM_PC_SAMPLES or len(jobs) != MINIMUM_PC_SAMPLES:
        raise DistributionCampaignError("campaign_selection_invalid")
    absent_artifacts: list[str] = []
    for ordinal, slot_value in enumerate(slots, start=1):
        slot = _mapping(slot_value, f"slot_{ordinal}")
        manifest = _future_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        output_root = _future_member(
            root,
            slot.get("output_root_path"),
            f"slot_{ordinal}_output",
        )
        transaction_paths = tuple(
            root / "transactions" / f"pc-{ordinal:03d}-{suffix}.json"
            for suffix in ("pre", "post", "failure")
        )
        if manifest.exists():
            raise DistributionCampaignError("failed_campaign_has_pc_sample")
        absent_artifacts.append(
            _evidence_relative(manifest, root, "absent_manifest")
        )
        if ordinal == 1:
            if not output_root.is_dir() or not all(
                path.is_file() for path in transaction_paths
            ):
                raise DistributionCampaignError("failed_slot_evidence_missing")
        elif output_root.exists() or any(
            path.exists() for path in transaction_paths
        ):
            raise DistributionCampaignError("failed_campaign_started_after_gap")

    slot, job, plan_path, _, host_restore_path, output_root = (
        _slot_paths_and_job(context, 1)
    )
    if output_root.name != "pc-001":
        raise DistributionCampaignError("failed_slot_output_invalid")
    plan = _mapping(read_canonical_json(plan_path), "failed_slot_plan")
    nonce = plan.get("session_nonce")
    if not isinstance(nonce, str):
        raise DistributionCampaignError("failed_slot_plan_nonce_invalid")

    attempts_path = _absolute_file(
        output_root / "attempts.json",
        "failed_slot_attempts",
    )
    attempts = _sequence(
        read_canonical_json(attempts_path),
        "failed_slot_attempts",
    )
    if len(attempts) != 1:
        raise DistributionCampaignError("failed_slot_process_count_invalid")
    attempt = _mapping(attempts[0], "failed_slot_attempt")
    if (
        set(attempt)
        != {
            "attempt",
            "error",
            "error_type",
            "finished_perf_counter_ns",
            "process_id",
            "started_perf_counter_ns",
            "status",
        }
        or attempt.get("attempt") != 1
        or attempt.get("status") != "RETRY"
        or attempt.get("error_type") != "ProbeError"
        or attempt.get("error") != "natural_strict_trace_process_failed"
        or isinstance(attempt.get("process_id"), bool)
        or not isinstance(attempt.get("process_id"), int)
        or attempt["process_id"] <= 0
        or isinstance(attempt.get("started_perf_counter_ns"), bool)
        or not isinstance(attempt.get("started_perf_counter_ns"), int)
        or isinstance(attempt.get("finished_perf_counter_ns"), bool)
        or not isinstance(attempt.get("finished_perf_counter_ns"), int)
        or attempt["finished_perf_counter_ns"]
        <= attempt["started_perf_counter_ns"]
    ):
        raise DistributionCampaignError("failed_slot_attempt_invalid")

    attempt_root = output_root / "attempt-01"
    expected_attempt_files = {
        "external-input-guard.json",
        "strict-replay.json",
        "strict-replay.log",
    }
    observed_attempt_files = {
        path.relative_to(attempt_root).as_posix()
        for path in attempt_root.rglob("*")
        if path.is_file()
    }
    if observed_attempt_files != expected_attempt_files:
        raise DistributionCampaignError("failed_slot_artifact_inventory_invalid")
    guard_path = _absolute_file(
        attempt_root / "external-input-guard.json",
        "failed_slot_external_input_guard",
    )
    strict_path = _absolute_file(
        attempt_root / "strict-replay.json",
        "failed_slot_strict_replay",
    )
    strict_log_path = _absolute_file(
        attempt_root / "strict-replay.log",
        "failed_slot_strict_replay_log",
    )
    for relative in (
        "memory-probe.json",
        "trajectory/index.json",
        "trajectory/frames/u00003429.bmp",
        "manifest.json",
        "source-verification.json",
        "execution-receipt.json",
    ):
        path = output_root / relative
        if path.exists():
            raise DistributionCampaignError("failed_campaign_has_sample_artifact")
        absent_artifacts.append(
            _evidence_relative(path, root, "absent_sample_artifact")
        )

    guard = _mapping(
        read_canonical_json(guard_path),
        "failed_slot_external_input_guard",
    )
    counts = _mapping(guard.get("event_counts"), "external_input_counts")
    coverage = _mapping(guard.get("coverage"), "external_input_coverage")
    if (
        guard.get("schema") != "zuma-rl.pc-external-input-guard"
        or guard.get("version") != 1
        or guard.get("status") != "PASS"
        or counts.get("external") != 0
        or guard.get("external_events") != []
        or guard.get("protocol_errors") != []
        or coverage.get("start_perf_counter_ns")
        < attempt["started_perf_counter_ns"]
        or coverage.get("end_perf_counter_ns")
        < attempt["finished_perf_counter_ns"]
    ):
        raise DistributionCampaignError("failed_slot_external_input_invalid")

    strict = _read_strict_json_object(
        strict_path,
        "failed_slot_strict_replay",
    )
    options = _mapping(strict.get("options"), "failed_trace_options")
    startup_rng = _mapping(strict.get("startup_rng"), "failed_trace_rng")
    result = _mapping(strict.get("result"), "failed_trace_result")
    source_dmo = _mapping(strict.get("source_dmo"), "failed_trace_dmo")
    dmo_binding = _mapping(slot.get("dmo"), "failed_slot_dmo")
    dmo_path = _bound_artifact(root, dmo_binding, "failed_slot_dmo")
    if (
        strict.get("schema") != "zuma.popcap_strict_replay.v3"
        or strict.get("runtime_process_id") != attempt["process_id"]
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve() != dmo_path.resolve()
        or "sha256:" + str(source_dmo.get("sha256"))
        != dmo_binding.get("sha256")
        or options.get("direct_natural_seed") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != 3151
        or options.get("startup_priority_bias_until_update") is not None
        or options.get("startup_process_affinity_mask") is not None
        or options.get("rng_seed_override_count") != 0
        or startup_rng.get("mode") != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != attempt["process_id"]
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 1
        or result.get("boundary_failures")
        != ["service payload forced read changed its prepared command"]
        or result.get("broker_failures") != []
        or result.get("last_update") != 390
        or result.get("stopped_at_update") is not False
        or result.get("stopped_at_command_order") is not False
    ):
        raise DistributionCampaignError("failed_slot_trace_not_exact_race")
    try:
        strict_log = strict_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DistributionCampaignError("failed_slot_trace_log_invalid") from error
    if (
        "offline_boundary_failure" not in strict_log
        or "update=390" not in strict_log
        or "detail=service payload forced read changed its prepared command"
        not in strict_log
    ):
        raise DistributionCampaignError("failed_slot_trace_log_invalid")

    transactions_root = root / "transactions"
    pre_path = _absolute_file(
        transactions_root / "pc-001-pre.json",
        "failed_slot_pre_snapshot",
    )
    post_path = _absolute_file(
        transactions_root / "pc-001-post.json",
        "failed_slot_post_snapshot",
    )
    failure_path = _absolute_file(
        transactions_root / "pc-001-failure.json",
        "failed_slot_failure_receipt",
    )
    pre = PcStateSnapshot.read(pre_path)
    post = PcStateSnapshot.read(post_path)
    host = PcStateSnapshot.read(host_restore_path)
    failure = _mapping(
        read_canonical_json(failure_path),
        "failed_slot_failure_receipt",
    )
    if (
        pre.session_nonce != nonce
        or post.session_nonce != nonce
        or host.session_nonce != nonce
        or pre.state_root != post.state_root
        or pre.state_root != host.state_root
        or failure.get("schema") != SLOT_EXECUTION_RECEIPT_SCHEMA
        or failure.get("version") != SLOT_EXECUTION_RECEIPT_VERSION
        or failure.get("status") != "FAIL"
        or failure.get("ordinal") != 1
        or failure.get("collector_exit_code") != 1
        or failure.get("pre_state_root") != pre.state_root
        or failure.get("post_state_root") != post.state_root
    ):
        raise DistributionCampaignError("failed_slot_state_not_restored")

    campaign_path = _absolute_file(root / "campaign.json", "campaign")
    preregistration_verification_path = _bound_artifact(
        root,
        campaign["preregistration_verification"],
        "campaign_preregistration_verification",
    )
    receipt = {
        "schema": CAMPAIGN_INVALIDATION_SCHEMA,
        "version": CAMPAIGN_INVALIDATION_VERSION,
        "status": "INVALIDATED",
        "classification": PRE_SAMPLE_INFRASTRUCTURE_FAILURE,
        "campaign_id": campaign_id,
        "superseded_by_campaign_id": superseded_by_campaign_id,
        "invalidated_utc": _frozen_utc(invalidated_utc),
        "formal_samples_observed": 0,
        "same_dmo_replay_counted_as_distribution": False,
        "failed_slot": {
            "ordinal": 1,
            "gameplay_seed": slot.get("gameplay_seed"),
            "attempt_count": 1,
            "started_process_count": 1,
            "process_id": attempt["process_id"],
            "failure_stage": "STRICT_COMMAND_REPLAY_BEFORE_FREEZE",
            "last_framework_update": 390,
            "required_freeze_update": PC_FREEZE_UPDATE,
            "required_sample_update": PC_SAMPLE_UPDATE,
            "memory_probe_created": False,
            "trajectory_created": False,
            "source_manifest_created": False,
        },
        "root_cause": {
            "code": BROKER_RACE_FAILURE_CODE,
            "observed_boundary_failure": (
                "service payload forced read changed its prepared command"
            ),
            "corrected_mechanism": (
                "broker_adjacent_current_update_file_write"
            ),
            "external_input_event_count": 0,
            "rng_process_memory_write_count": 0,
            "persistent_file_modified": False,
        },
        "state_transaction": {
            "restored": True,
            "pre_state_root": pre.state_root,
            "post_state_root": post.state_root,
            "frozen_host_state_root": host.state_root,
        },
        "preregistration_reverification": {
            "status": "PASS",
            "verification": _artifact_binding(
                preregistration_verification_path,
                root,
            ),
        },
        "evidence": {
            "campaign": _artifact_binding(campaign_path, root),
            "collection_jobs": campaign["collection_jobs"],
            "attempts": _artifact_binding(attempts_path, root),
            "external_input_guard": _artifact_binding(guard_path, root),
            "strict_replay": _artifact_binding(strict_path, root),
            "strict_replay_log": _artifact_binding(strict_log_path, root),
            "pre_snapshot": _artifact_binding(pre_path, root),
            "post_snapshot": _artifact_binding(post_path, root),
            "failure_receipt": _artifact_binding(failure_path, root),
        },
        "absent_sample_artifacts": sorted(set(absent_artifacts)),
    }
    _write_exclusive(receipt_path, receipt)
    return receipt_path


def _validate_rebased_late_payload_failure_trace(
    *,
    strict: Mapping[str, Any],
    strict_log: str,
    attempt: Mapping[str, Any],
    dmo_path: Path,
    dmo_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact immutable formal PC-008 late-payload failure."""

    options = _mapping(strict.get("options"), "failed_trace_options")
    startup_rng = _mapping(strict.get("startup_rng"), "failed_trace_rng")
    result = _mapping(strict.get("result"), "failed_trace_result")
    source_dmo = _mapping(strict.get("source_dmo"), "failed_trace_dmo")
    claims = _sequence(
        result.get("service_file_write_header_claims"),
        "failed_trace_file_write_claims",
    )
    if len(claims) != 1:
        raise DistributionCampaignError("failed_slot_claim_count_invalid")
    claim = _mapping(claims[0], "failed_trace_file_write_claim")
    process_id = attempt.get("process_id")
    if (
        strict.get("schema") != "zuma.popcap_strict_replay.v3"
        or strict.get("runtime_process_id") != process_id
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve() != dmo_path.resolve()
        or "sha256:" + str(source_dmo.get("sha256"))
        != dmo_binding.get("sha256")
        or options.get("direct_natural_seed") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != 3151
        or options.get("startup_priority_bias_until_update") is not None
        or options.get("startup_process_affinity_mask") is not None
        or options.get("rng_seed_override_count") != 0
        or startup_rng.get("mode") != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != process_id
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 1
        or result.get("boundary_failures")
        != [
            "preloading failed file-write tail prefetch command_order "
            "mismatch: 67 != 66"
        ]
        or result.get("broker_failures") != []
        or result.get("last_update") != 390
        or result.get("stopped_at_update") is not False
        or result.get("stopped_at_command_order") is not False
        or result.get("command_order_offset") != 4
        or result.get("command_order_rebase_rows") != [64, 65, 68, 69]
    ):
        raise DistributionCampaignError(
            "failed_slot_trace_not_exact_rebased_tail"
        )
    if (
        claim.get("process_id") != process_id
        or claim.get("mechanism")
        != "broker_adjacent_current_update_file_write"
        or claim.get("framework_update") != 387
        or claim.get("row_index") != 61
        or claim.get("row_start") != 1739
        or claim.get("row_end") != 1750
        or claim.get("row_update") != 387
        or claim.get("command_order") != 61
        or claim.get("command_order_offset") != 0
        or claim.get("prior_brokered_row_index") != 60
        or claim.get("prior_brokered_command_order") != 60
        or claim.get("prior_brokered_read_bit_position") != 1739
        or claim.get("file_write_font_cache_member")
        != "cached\\fonts\\600\\shagexotica32_normal.txt.cfw2"
        or claim.get("file_write_argument_size") != 15623
        or claim.get("manifest_expected_size") != 15623
        or claim.get("manifest_entry_count") != 35
        or claim.get("recorded_success") is not False
        or claim.get("payload_completion_verified") is not True
        or claim.get("natural_payload_consumer") is not True
        or claim.get("payload_completion_framework_update") != 387
        or claim.get("payload_completion_command_order") != 61
        or claim.get("payload_completion_command_bit_position") != 1739
        or claim.get("payload_completion_buffer_read_bit_position") != 1750
        or isinstance(claim.get("thread_id"), bool)
        or not isinstance(claim.get("thread_id"), int)
        or claim["thread_id"] <= 0
        or isinstance(claim.get("payload_completion_thread_id"), bool)
        or not isinstance(claim.get("payload_completion_thread_id"), int)
        or claim["payload_completion_thread_id"] <= 0
        or claim["payload_completion_thread_id"] == claim["thread_id"]
    ):
        raise DistributionCampaignError("failed_slot_claim_invalid")
    required_log_fragments = (
        "service_file_write_header_claim",
        "row=61 update=387",
        "service_file_write_payload_completion",
        "service_command_order_rebase tid=",
        "offset=2 accounted_rows=64,65",
        "offset=4 accounted_rows=64,65,68,69",
        "service_payload_reentry tid=",
        "update=390 row=70 cmd_bitpos=1848 read_bitpos=1858",
        "offline_boundary_failure",
        "detail=preloading failed file-write tail prefetch "
        "command_order mismatch: 67 != 66",
    )
    if any(fragment not in strict_log for fragment in required_log_fragments):
        raise DistributionCampaignError("failed_slot_trace_log_invalid")
    return claim


def _validate_tail_short_header_failure_trace(
    *,
    strict: Mapping[str, Any],
    strict_log: str,
    attempt: Mapping[str, Any],
    dmo_path: Path,
    dmo_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact immutable formal PC-007 false short header."""

    options = _mapping(strict.get("options"), "failed_trace_options")
    startup_rng = _mapping(strict.get("startup_rng"), "failed_trace_rng")
    result = _mapping(strict.get("result"), "failed_trace_result")
    source_dmo = _mapping(strict.get("source_dmo"), "failed_trace_dmo")
    claims = _sequence(
        result.get("service_file_write_header_claims"),
        "failed_trace_file_write_claims",
    )
    if len(claims) != 1:
        raise DistributionCampaignError("failed_slot_claim_count_invalid")
    claim = _mapping(claims[0], "failed_trace_file_write_claim")
    process_id = attempt.get("process_id")
    observed_failure = (
        "service exit continuation is unaudited: kind=11, "
        "cmd_bitpos=1882, read_bitpos=1888, needs=0, short=1, "
        "num=1, order=68"
    )
    if (
        strict.get("schema") != "zuma.popcap_strict_replay.v3"
        or strict.get("runtime_process_id") != process_id
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve() != dmo_path.resolve()
        or "sha256:" + str(source_dmo.get("sha256"))
        != dmo_binding.get("sha256")
        or options.get("direct_natural_seed") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != 3151
        or options.get("startup_priority_bias_until_update") is not None
        or options.get("startup_process_affinity_mask") is not None
        or options.get("rng_seed_override_count") != 0
        or startup_rng.get("mode") != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != process_id
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 1
        or result.get("boundary_failures") != [observed_failure]
        or result.get("broker_failures") != []
        or result.get("last_update") != 401
        or result.get("stopped_at_update") is not False
        or result.get("stopped_at_command_order") is not False
        or result.get("command_order_offset") != 4
        or result.get("command_order_rebase_rows") != [64, 65, 68, 69]
        or result.get(
            "preloading_failed_file_write_tail_short_header_recoveries"
        )
        is not None
    ):
        raise DistributionCampaignError(
            "failed_slot_trace_not_exact_tail_short_header"
        )
    if (
        claim.get("process_id") != process_id
        or claim.get("mechanism")
        != "broker_adjacent_current_update_file_write"
        or claim.get("framework_update") != 387
        or claim.get("row_index") != 61
        or claim.get("row_start") != 1739
        or claim.get("row_end") != 1750
        or claim.get("row_update") != 387
        or claim.get("command_order") != 61
        or claim.get("command_order_offset") != 0
        or claim.get("prior_brokered_row_index") != 60
        or claim.get("prior_brokered_command_order") != 60
        or claim.get("prior_brokered_read_bit_position") != 1739
        or claim.get("file_write_font_cache_member")
        != "cached\\fonts\\600\\shagexotica32_normal.txt.cfw2"
        or claim.get("file_write_argument_size") != 15623
        or claim.get("manifest_expected_size") != 15623
        or claim.get("manifest_entry_count") != 35
        or claim.get("recorded_success") is not False
        or claim.get("payload_completion_verified") is not True
        or claim.get("natural_payload_consumer") is not True
        or claim.get("payload_completion_framework_update") != 387
        or claim.get("payload_completion_command_order") != 61
        or claim.get("payload_completion_command_bit_position") != 1739
        or claim.get("payload_completion_buffer_read_bit_position") != 1750
        or isinstance(claim.get("thread_id"), bool)
        or not isinstance(claim.get("thread_id"), int)
        or claim["thread_id"] <= 0
        or isinstance(claim.get("payload_completion_thread_id"), bool)
        or not isinstance(claim.get("payload_completion_thread_id"), int)
        or claim["payload_completion_thread_id"] <= 0
        or claim["payload_completion_thread_id"] == claim["thread_id"]
    ):
        raise DistributionCampaignError("failed_slot_claim_invalid")
    required_log_fragments = (
        "service_file_write_header_claim",
        "row=61 update=387",
        "service_file_write_payload_completion",
        "offset=2 accounted_rows=64,65",
        "offset=4 accounted_rows=64,65,68,69",
        "update=390 row=70 cmd_bitpos=1848 read_bitpos=1858",
        "preloading_failed_file_write_tail_prefetch",
        "update=390 row=73 cmd_bitpos=1848 read_bitpos=1882",
        "offline_boundary_failure",
        "update=401 cmd_bitpos=1882",
        f"detail={observed_failure}",
    )
    if any(fragment not in strict_log for fragment in required_log_fragments):
        raise DistributionCampaignError("failed_slot_trace_log_invalid")
    return claim


def _validate_post_sample_finalization_oserror_trace(
    *,
    strict: Mapping[str, Any],
    strict_log: str,
    attempt: Mapping[str, Any],
    dmo_path: Path,
    dmo_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact immutable PC-018 post-sample finalization incident."""

    options = _mapping(strict.get("options"), "failed_trace_options")
    startup_rng = _mapping(strict.get("startup_rng"), "failed_trace_rng")
    result = _mapping(strict.get("result"), "failed_trace_result")
    source_dmo = _mapping(strict.get("source_dmo"), "failed_trace_dmo")
    continuations = _sequence(
        result.get("service_continuation_verifications"),
        "failed_trace_continuations",
    )
    if len(continuations) != 1:
        raise DistributionCampaignError(
            "failed_slot_trace_not_exact_post_sample_oserror"
        )
    continuation = _mapping(
        continuations[0],
        "failed_trace_continuation",
    )
    process_id = attempt.get("process_id")
    if (
        attempt.get("status") != "RETRY"
        or attempt.get("error_type") != "OSError"
        or attempt.get("error") != "[Errno 22] Invalid argument"
        or strict.get("schema") != "zuma.popcap_strict_replay.v3"
        or strict.get("runtime_process_id") != process_id
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve() != dmo_path.resolve()
        or "sha256:" + str(source_dmo.get("sha256"))
        != dmo_binding.get("sha256")
        or options.get("direct_natural_seed") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != 3151
        or options.get("startup_priority_bias_until_update") is not None
        or options.get("startup_process_affinity_mask") is not None
        or options.get("rng_seed_override_count") != 0
        or startup_rng.get("mode") != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != process_id
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("last_update") != 3151
        or result.get("stopped_at_update") is not True
        or result.get("stopped_at_command_order") is not False
        or result.get("command_order_offset") != 0
        or result.get("command_order_rebase_rows") != []
        or result.get("service_file_write_header_claims") != []
        or result.get(
            "preloading_failed_file_write_tail_short_header_recoveries"
        )
        != []
        or result.get("service_exit_timeline_commits") != []
        or result.get("debug_exception_observations") != []
        or continuation.get("process_id") != process_id
        or continuation.get("framework_update") != 406
        or continuation.get("row_index") != 72
        or continuation.get("corridor_start_index") != 45
        or continuation.get("corridor_end_index") != 71
        or continuation.get("reentries") != 3
        or continuation.get("command_order_offset") != 0
        or continuation.get("service_exit_timeline_commit_count") != 0
    ):
        raise DistributionCampaignError(
            "failed_slot_trace_not_exact_post_sample_oserror"
        )
    required_log_fragments = (
        "service_continuation_verified",
        "update=406 row=72 reentries=3",
        "result hits=",
        "broker_failures=0 boundary_failures=0",
        "last_update=3151 stopped_at_update=True",
        "command_order_offset=0",
        "preloading_failed_file_write_tail_short_header_recoveries=0",
        "service_file_write_header_claims=0",
    )
    if (
        any(fragment not in strict_log for fragment in required_log_fragments)
        or "offline_boundary_failure" in strict_log
    ):
        raise DistributionCampaignError("failed_slot_trace_log_invalid")
    return continuation


def invalidate_partial_distribution_campaign(
    *,
    campaign_root: Path,
    original_root: Path,
    superseded_by_campaign_id: str,
    invalidated_utc: str | None = None,
) -> Path:
    """Invalidate one exact, known partial campaign append-only."""

    if _IDENTIFIER_PATTERN.fullmatch(superseded_by_campaign_id) is None:
        raise DistributionCampaignError("superseding_campaign_id_invalid")
    context = _load_campaign_context(
        campaign_root,
        require_current_collection_tools=False,
    )
    root = Path(context["root"])
    original = _absolute_directory(original_root, "original_root")
    campaign = _mapping(context["campaign"], "campaign")
    campaign_id = campaign.get("campaign_id")
    if superseded_by_campaign_id == campaign_id:
        raise DistributionCampaignError("superseding_campaign_id_invalid")
    receipt_path = root / "invalidation.json"
    if receipt_path.exists():
        raise DistributionCampaignError("campaign_invalidation_already_exists")
    if any(
        (root / name).exists()
        for name in (
            "pc-dataset.json",
            "audit.json",
            "audit-verification.json",
            "completion.json",
        )
    ):
        raise DistributionCampaignError("failed_campaign_has_final_evidence")

    slots = _sequence(context["slots"], "pc_slots")
    jobs = _sequence(context["jobs"], "collection_jobs_rows")
    if len(slots) != MINIMUM_PC_SAMPLES or len(jobs) != MINIMUM_PC_SAMPLES:
        raise DistributionCampaignError("campaign_selection_invalid")
    completed_ordinals: list[int] = []
    partial_ordinal: int | None = None
    pending_seen = False
    for ordinal, slot_value in enumerate(slots, start=1):
        slot = _mapping(slot_value, f"slot_{ordinal}")
        manifest_path = _future_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        output_root = _future_member(
            root,
            slot.get("output_root_path"),
            f"slot_{ordinal}_output",
        )
        transaction_paths = tuple(
            root / "transactions" / f"pc-{ordinal:03d}-{suffix}.json"
            for suffix in ("pre", "post", "failure")
        )
        if manifest_path.is_file():
            if pending_seen or partial_ordinal is not None:
                raise DistributionCampaignError("completed_slot_after_gap")
            completed_ordinals.append(ordinal)
            continue
        started = output_root.exists() or any(
            path.exists() for path in transaction_paths
        )
        if started:
            if pending_seen or partial_ordinal is not None:
                raise DistributionCampaignError(
                    "failed_campaign_started_after_gap"
                )
            partial_ordinal = ordinal
        pending_seen = True
    if completed_ordinals == list(range(1, 8)) and partial_ordinal == 8:
        failure_profile = "rebased_late_payload"
    elif completed_ordinals == list(range(1, 7)) and partial_ordinal == 7:
        failure_profile = "tail_short_header"
    elif completed_ordinals == list(range(1, 18)) and partial_ordinal == 18:
        failure_profile = "post_sample_finalization_oserror"
    else:
        raise DistributionCampaignError("partial_campaign_shape_invalid")

    completed_samples: list[dict[str, Any]] = []
    common_state_root: str | None = None
    for ordinal in completed_ordinals:
        slot, _, plan_path, _, host_restore_path, output_root = (
            _slot_paths_and_job(context, ordinal)
        )
        manifest_path = _bound_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        report = verify_pc_source_manifest(
            manifest_path,
            evidence_root=root,
            original_root=original,
        )
        if report.get("status") != "PASS":
            raise DistributionCampaignError("completed_slot_invalid")
        verification_path = _absolute_file(
            output_root / "source-verification.json",
            f"slot_{ordinal}_source_verification",
        )
        if read_canonical_json(verification_path) != report:
            raise DistributionCampaignError(
                "completed_slot_verification_mismatch"
            )
        execution_path = _absolute_file(
            output_root / "execution-receipt.json",
            f"slot_{ordinal}_execution_receipt",
        )
        execution = _mapping(
            read_canonical_json(execution_path),
            f"slot_{ordinal}_execution_receipt",
        )
        pre_path = _absolute_file(
            root / "transactions" / f"pc-{ordinal:03d}-pre.json",
            f"slot_{ordinal}_pre_snapshot",
        )
        post_path = _absolute_file(
            root / "transactions" / f"pc-{ordinal:03d}-post.json",
            f"slot_{ordinal}_post_snapshot",
        )
        if (root / "transactions" / f"pc-{ordinal:03d}-failure.json").exists():
            raise DistributionCampaignError("completed_slot_has_failure_receipt")
        plan = _mapping(read_canonical_json(plan_path), f"slot_{ordinal}_plan")
        nonce = plan.get("session_nonce")
        pre = PcStateSnapshot.read(pre_path)
        post = PcStateSnapshot.read(post_path)
        host = PcStateSnapshot.read(host_restore_path)
        if (
            not isinstance(nonce, str)
            or pre.session_nonce != nonce
            or post.session_nonce != nonce
            or host.session_nonce != nonce
            or pre.state_root != post.state_root
            or pre.state_root != host.state_root
            or execution.get("schema") != SLOT_EXECUTION_RECEIPT_SCHEMA
            or execution.get("version") != SLOT_EXECUTION_RECEIPT_VERSION
            or execution.get("status") != "PASS"
            or execution.get("campaign_id") != campaign_id
            or execution.get("ordinal") != ordinal
            or execution.get("gameplay_seed") != slot.get("gameplay_seed")
            or execution.get("collector_exit_code") != 0
            or execution.get("pre_state_root") != pre.state_root
            or execution.get("post_state_root") != post.state_root
            or execution.get("source_manifest")
            != _artifact_binding(manifest_path, root)
            or execution.get("source_verification")
            != _artifact_binding(verification_path, root)
        ):
            raise DistributionCampaignError("completed_slot_receipt_invalid")
        if common_state_root is None:
            common_state_root = pre.state_root
        elif pre.state_root != common_state_root:
            raise DistributionCampaignError("completed_slot_state_root_drift")
        completed_samples.append(
            {
                "ordinal": ordinal,
                "source_manifest": _artifact_binding(manifest_path, root),
                "source_verification": _artifact_binding(
                    verification_path,
                    root,
                ),
                "execution_receipt": _artifact_binding(execution_path, root),
                "pre_snapshot": _artifact_binding(pre_path, root),
                "post_snapshot": _artifact_binding(post_path, root),
            }
        )

    assert partial_ordinal is not None
    failed_ordinal = partial_ordinal
    failed_slot, _, failed_plan_path, _, failed_host_path, failed_output = (
        _slot_paths_and_job(context, failed_ordinal)
    )
    failed_manifest_path = _future_member(
        root,
        failed_slot.get("source_manifest_path"),
        "failed_slot_manifest",
    )
    if (
        failed_manifest_path.exists()
        or failed_output.name != f"pc-{failed_ordinal:03d}"
    ):
        raise DistributionCampaignError("failed_campaign_has_pc_sample")
    if failure_profile == "post_sample_finalization_oserror":
        expected_output_files = {
            "attempts.json",
            "attempt-01/active-board-field-068c.bin",
            "attempt-01/active-board.bin",
            "attempt-01/active-fired-bullet-gap-nodes.bin",
            "attempt-01/active-fired-bullets.bin",
            "attempt-01/curve-00-list-050-payloads.bin",
            "attempt-01/curve-00-list-05c-payloads.bin",
            "attempt-01/curve-00-list-068-payloads.bin",
            "attempt-01/curve-00-plan.bin",
            "attempt-01/curve-00.bin",
            "attempt-01/curve-manager.bin",
            "attempt-01/curve-plan-exhausted.bin",
            "attempt-01/external-input-guard.json",
            "attempt-01/global-mtrand.bin",
            "attempt-01/qrand-last_hit.bin",
            "attempt-01/qrand-previous_hit.bin",
            "attempt-01/qrand-sways.bin",
            "attempt-01/qrand-weights.bin",
            "attempt-01/qrand.bin",
            "attempt-01/replay-freeze-state.bin",
            "attempt-01/strict-replay.json",
            "attempt-01/strict-replay.log",
            "attempt-01/thread-crt-rand-state.bin",
            "attempt-01/trajectory/frames/u00003429.bmp",
            "attempt-01/trajectory/frames/warmup-u00003425.bmp",
            "attempt-01/trajectory/global-mtrand-frames.bin",
        }
    else:
        expected_output_files = {
            "attempts.json",
            "attempt-01/external-input-guard.json",
            "attempt-01/strict-replay.json",
            "attempt-01/strict-replay.log",
        }
    observed_output_files = {
        path.relative_to(failed_output).as_posix()
        for path in failed_output.rglob("*")
        if path.is_file()
    }
    if observed_output_files != expected_output_files:
        raise DistributionCampaignError("failed_slot_artifact_inventory_invalid")
    attempts_path = _absolute_file(
        failed_output / "attempts.json",
        "failed_slot_attempts",
    )
    attempts = _sequence(
        read_canonical_json(attempts_path),
        "failed_slot_attempts",
    )
    if len(attempts) != 1:
        raise DistributionCampaignError("failed_slot_process_count_invalid")
    attempt = _mapping(attempts[0], "failed_slot_attempt")
    if (
        set(attempt)
        != {
            "attempt",
            "error",
            "error_type",
            "finished_perf_counter_ns",
            "process_id",
            "started_perf_counter_ns",
            "status",
        }
        or attempt.get("attempt") != 1
        or attempt.get("status") != "RETRY"
        or isinstance(attempt.get("process_id"), bool)
        or not isinstance(attempt.get("process_id"), int)
        or attempt["process_id"] <= 0
        or isinstance(attempt.get("started_perf_counter_ns"), bool)
        or not isinstance(attempt.get("started_perf_counter_ns"), int)
        or isinstance(attempt.get("finished_perf_counter_ns"), bool)
        or not isinstance(attempt.get("finished_perf_counter_ns"), int)
        or attempt["finished_perf_counter_ns"]
        <= attempt["started_perf_counter_ns"]
    ):
        raise DistributionCampaignError("failed_slot_attempt_invalid")
    if failure_profile == "post_sample_finalization_oserror":
        if (
            attempt.get("error_type") != "OSError"
            or attempt.get("error") != "[Errno 22] Invalid argument"
        ):
            raise DistributionCampaignError("failed_slot_attempt_invalid")
    elif (
        attempt.get("error_type") != "ProbeError"
        or attempt.get("error") != "natural_strict_trace_process_failed"
    ):
        raise DistributionCampaignError("failed_slot_attempt_invalid")

    attempt_root = failed_output / "attempt-01"
    guard_path = _absolute_file(
        attempt_root / "external-input-guard.json",
        "failed_slot_external_input_guard",
    )
    strict_path = _absolute_file(
        attempt_root / "strict-replay.json",
        "failed_slot_strict_replay",
    )
    strict_log_path = _absolute_file(
        attempt_root / "strict-replay.log",
        "failed_slot_strict_replay_log",
    )
    guard = _mapping(
        read_canonical_json(guard_path),
        "failed_slot_external_input_guard",
    )
    counts = _mapping(guard.get("event_counts"), "external_input_counts")
    coverage = _mapping(guard.get("coverage"), "external_input_coverage")
    if (
        guard.get("schema") != "zuma-rl.pc-external-input-guard"
        or guard.get("version") != 1
        or guard.get("status") != "PASS"
        or counts.get("external") != 0
        or guard.get("external_events") != []
        or guard.get("protocol_errors") != []
        or not isinstance(coverage.get("start_perf_counter_ns"), int)
        or not isinstance(coverage.get("end_perf_counter_ns"), int)
        or coverage["start_perf_counter_ns"]
        < attempt["started_perf_counter_ns"]
        or coverage["end_perf_counter_ns"]
        < attempt["finished_perf_counter_ns"]
    ):
        raise DistributionCampaignError("failed_slot_external_input_invalid")
    strict = _read_strict_json_object(strict_path, "failed_slot_strict_replay")
    try:
        strict_log = strict_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DistributionCampaignError("failed_slot_trace_log_invalid") from error
    dmo_binding = _mapping(failed_slot.get("dmo"), "failed_slot_dmo")
    dmo_path = _bound_artifact(root, dmo_binding, "failed_slot_dmo")
    continuation: Mapping[str, Any] | None = None
    if failure_profile == "rebased_late_payload":
        claim = _validate_rebased_late_payload_failure_trace(
            strict=strict,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo_path,
            dmo_binding=dmo_binding,
        )
        last_framework_update = 390
        failure_code = REBASED_LATE_PAYLOAD_FAILURE_CODE
        observed_boundary_failure = (
            "preloading failed file-write tail prefetch command_order "
            "mismatch: 67 != 66"
        )
        corrected_tail_mechanism = "rebased_late_payload_refill"
    elif failure_profile == "tail_short_header":
        claim = _validate_tail_short_header_failure_trace(
            strict=strict,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo_path,
            dmo_binding=dmo_binding,
        )
        last_framework_update = 401
        failure_code = TAIL_SHORT_HEADER_FAILURE_CODE
        observed_boundary_failure = (
            "service exit continuation is unaudited: kind=11, "
            "cmd_bitpos=1882, read_bitpos=1888, needs=0, short=1, "
            "num=1, order=68"
        )
        corrected_tail_mechanism = (
            "audited_false_short_header_pre_payload_recovery"
        )
    else:
        claim = None
        continuation = _validate_post_sample_finalization_oserror_trace(
            strict=strict,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo_path,
            dmo_binding=dmo_binding,
        )
        last_framework_update = 3151
        failure_code = POST_SAMPLE_FINALIZATION_OSERROR_CODE
        observed_boundary_failure = None
        corrected_tail_mechanism = None

    failed_plan = _mapping(
        read_canonical_json(failed_plan_path),
        "failed_slot_plan",
    )
    failed_nonce = failed_plan.get("session_nonce")
    failed_pre_path = _absolute_file(
        root
        / "transactions"
        / f"pc-{failed_ordinal:03d}-pre.json",
        "failed_slot_pre_snapshot",
    )
    failed_post_path = _absolute_file(
        root
        / "transactions"
        / f"pc-{failed_ordinal:03d}-post.json",
        "failed_slot_post_snapshot",
    )
    failed_receipt_path = _absolute_file(
        root
        / "transactions"
        / f"pc-{failed_ordinal:03d}-failure.json",
        "failed_slot_failure_receipt",
    )
    failed_pre = PcStateSnapshot.read(failed_pre_path)
    failed_post = PcStateSnapshot.read(failed_post_path)
    failed_host = PcStateSnapshot.read(failed_host_path)
    failed_receipt = _mapping(
        read_canonical_json(failed_receipt_path),
        "failed_slot_failure_receipt",
    )
    if (
        not isinstance(failed_nonce, str)
        or failed_pre.session_nonce != failed_nonce
        or failed_post.session_nonce != failed_nonce
        or failed_host.session_nonce != failed_nonce
        or failed_pre.state_root != failed_post.state_root
        or failed_pre.state_root != failed_host.state_root
        or failed_pre.state_root != common_state_root
        or failed_receipt.get("schema") != SLOT_EXECUTION_RECEIPT_SCHEMA
        or failed_receipt.get("version") != SLOT_EXECUTION_RECEIPT_VERSION
        or failed_receipt.get("status") != "FAIL"
        or failed_receipt.get("ordinal") != failed_ordinal
        or failed_receipt.get("collector_exit_code")
        != (120 if failure_profile == "post_sample_finalization_oserror" else 1)
        or failed_receipt.get("pre_state_root") != failed_pre.state_root
        or failed_receipt.get("post_state_root") != failed_post.state_root
    ):
        raise DistributionCampaignError("failed_slot_state_not_restored")

    absent_artifacts = [
        _evidence_relative(failed_manifest_path, root, "absent_manifest")
    ]
    absent_failed_relatives = [
        "memory-probe.json",
        "trajectory/index.json",
        "source-verification.json",
        "execution-receipt.json",
    ]
    if failure_profile != "post_sample_finalization_oserror":
        absent_failed_relatives.append("trajectory/frames/u00003429.bmp")
    for relative in absent_failed_relatives:
        path = failed_output / relative
        if path.exists():
            raise DistributionCampaignError("failed_campaign_has_sample_artifact")
        absent_artifacts.append(
            _evidence_relative(path, root, "absent_sample_artifact")
        )
    for ordinal in range(failed_ordinal + 1, MINIMUM_PC_SAMPLES + 1):
        slot = _mapping(slots[ordinal - 1], f"slot_{ordinal}")
        manifest_path = _future_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        output_root = _future_member(
            root,
            slot.get("output_root_path"),
            f"slot_{ordinal}_output",
        )
        transaction_paths = tuple(
            root / "transactions" / f"pc-{ordinal:03d}-{suffix}.json"
            for suffix in ("pre", "post", "failure")
        )
        if manifest_path.exists() or output_root.exists() or any(
            path.exists() for path in transaction_paths
        ):
            raise DistributionCampaignError("failed_campaign_started_after_gap")
        absent_artifacts.append(
            _evidence_relative(manifest_path, root, "absent_manifest")
        )
        absent_artifacts.extend(
            _evidence_relative(path, root, "absent_transaction")
            for path in transaction_paths
        )

    campaign_path = _absolute_file(root / "campaign.json", "campaign")
    preregistration_verification_path = _bound_artifact(
        root,
        campaign["preregistration_verification"],
        "campaign_preregistration_verification",
    )
    if failure_profile == "post_sample_finalization_oserror":
        assert continuation is not None
        failed_slot_summary = {
            "ordinal": failed_ordinal,
            "gameplay_seed": failed_slot.get("gameplay_seed"),
            "attempt_count": 1,
            "started_process_count": 1,
            "process_id": attempt["process_id"],
            "failure_stage": "POST_SAMPLE_CAPTURE_BEFORE_PROBE_COMMIT",
            "last_strict_replay_update": last_framework_update,
            "required_freeze_update": PC_FREEZE_UPDATE,
            "required_sample_update": PC_SAMPLE_UPDATE,
            "sample_frame_update": PC_SAMPLE_UPDATE,
            "memory_probe_created": False,
            "trajectory_frame_created": True,
            "trajectory_index_created": False,
            "source_manifest_created": False,
        }
        root_cause = {
            "code": failure_code,
            "observed_error_type": attempt["error_type"],
            "observed_error": attempt["error"],
            "collector_exit_code": failed_receipt["collector_exit_code"],
            "strict_replay_failure_count": 0,
            "strict_replay_boundary_failure_count": 0,
            "strict_replay_broker_failure_count": 0,
            "service_continuation_verification_count": 1,
            "service_continuation_framework_update": continuation[
                "framework_update"
            ],
            "external_input_event_count": 0,
            "rng_process_memory_write_count": 0,
            "persistent_file_modified": False,
            "gameplay_boundary_implicated": False,
        }
    else:
        assert claim is not None
        failed_slot_summary = {
            "ordinal": failed_ordinal,
            "gameplay_seed": failed_slot.get("gameplay_seed"),
            "attempt_count": 1,
            "started_process_count": 1,
            "process_id": attempt["process_id"],
            "failure_stage": "STRICT_COMMAND_REPLAY_BEFORE_FREEZE",
            "last_framework_update": last_framework_update,
            "required_freeze_update": PC_FREEZE_UPDATE,
            "required_sample_update": PC_SAMPLE_UPDATE,
            "memory_probe_created": False,
            "trajectory_created": False,
            "source_manifest_created": False,
        }
        root_cause = {
            "code": failure_code,
            "observed_boundary_failure": observed_boundary_failure,
            "live_file_write_claim_mechanism": claim["mechanism"],
            "corrected_tail_mechanism": corrected_tail_mechanism,
            "command_order_offset": 4,
            "command_order_rebase_rows": [64, 65, 68, 69],
            "external_input_event_count": 0,
            "rng_process_memory_write_count": 0,
            "persistent_file_modified": False,
        }
    evidence = {
        "campaign": _artifact_binding(campaign_path, root),
        "collection_jobs": campaign["collection_jobs"],
        "attempts": _artifact_binding(attempts_path, root),
        "external_input_guard": _artifact_binding(guard_path, root),
        "strict_replay": _artifact_binding(strict_path, root),
        "strict_replay_log": _artifact_binding(strict_log_path, root),
        "pre_snapshot": _artifact_binding(failed_pre_path, root),
        "post_snapshot": _artifact_binding(failed_post_path, root),
        "failure_receipt": _artifact_binding(failed_receipt_path, root),
    }
    if failure_profile == "post_sample_finalization_oserror":
        evidence["partial_output_files"] = [
            _artifact_binding(path, root)
            for path in sorted(failed_output.rglob("*"))
            if path.is_file()
        ]
    receipt = {
        "schema": CAMPAIGN_INVALIDATION_SCHEMA,
        "version": CAMPAIGN_INVALIDATION_VERSION,
        "status": "INVALIDATED",
        "classification": PARTIAL_CAMPAIGN_INFRASTRUCTURE_FAILURE,
        "campaign_id": campaign_id,
        "superseded_by_campaign_id": superseded_by_campaign_id,
        "invalidated_utc": _frozen_utc(invalidated_utc),
        "formal_samples_observed": len(completed_ordinals),
        "same_dmo_replay_counted_as_distribution": False,
        "completed_prefix": {
            "count": len(completed_ordinals),
            "ordinals": completed_ordinals,
            "verification_status": "PASS",
            "disposition": "VALIDATED_BUT_NON_PROMOTABLE",
            "reason": "ONCE_ONLY_CAMPAIGN_GAP",
            "samples": completed_samples,
        },
        "failed_slot": failed_slot_summary,
        "root_cause": root_cause,
        "state_transaction": {
            "restored": True,
            "pre_state_root": failed_pre.state_root,
            "post_state_root": failed_post.state_root,
            "frozen_host_state_root": failed_host.state_root,
        },
        "continuation_policy": {
            "retry_failed_slot_forbidden": True,
            "reuse_completed_prefix_in_successor_forbidden": True,
            "successor_required_fresh_pc_slots": MINIMUM_PC_SAMPLES,
            "later_slots_started": False,
        },
        "preregistration_reverification": {
            "status": "PASS",
            "verification": _artifact_binding(
                preregistration_verification_path,
                root,
            ),
        },
        "evidence": evidence,
        "absent_sample_artifacts": sorted(set(absent_artifacts)),
    }
    _write_exclusive(receipt_path, receipt)
    return receipt_path


def campaign_collection_status(
    *,
    campaign_root: Path,
    original_root: Path,
) -> dict[str, Any]:
    context = _load_campaign_context(campaign_root)
    root = context["root"]
    original = _absolute_directory(original_root, "original_root")
    completed: list[int] = []
    partial: int | None = None
    first_pending: int | None = None
    for ordinal, slot_value in enumerate(context["slots"], start=1):
        slot = _mapping(slot_value, f"slot_{ordinal}")
        manifest_path = _future_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        output_root = _future_member(
            root,
            slot.get("output_root_path"),
            f"slot_{ordinal}_output",
        )
        pre_path = root / "transactions" / f"pc-{ordinal:03d}-pre.json"
        post_path = root / "transactions" / f"pc-{ordinal:03d}-post.json"
        if manifest_path.is_file():
            if first_pending is not None:
                raise DistributionCampaignError("completed_slot_after_gap")
            report = verify_pc_source_manifest(
                manifest_path,
                evidence_root=root,
                original_root=original,
            )
            if report.get("status") != "PASS":
                raise DistributionCampaignError("completed_slot_invalid")
            completed.append(ordinal)
        elif output_root.exists() or pre_path.exists() or post_path.exists():
            partial = ordinal
            break
        elif first_pending is None:
            first_pending = ordinal
    return {
        "schema": "zuma-rl.startup-distribution-campaign-status",
        "version": 1,
        "status": "PARTIAL_FAILURE" if partial is not None else "READY",
        "campaign_id": context["campaign"]["campaign_id"],
        "completed_slot_count": len(completed),
        "completed_ordinals": completed,
        "next_slot_ordinal": (
            None
            if partial is not None or len(completed) == MINIMUM_PC_SAMPLES
            else len(completed) + 1
        ),
        "partial_slot_ordinal": partial,
        "all_slots_complete": len(completed) == MINIMUM_PC_SAMPLES,
    }


def _slot_paths_and_job(
    context: Mapping[str, Any],
    ordinal: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, Path, Path, Path]:
    if isinstance(ordinal, bool) or not 1 <= ordinal <= MINIMUM_PC_SAMPLES:
        raise DistributionCampaignError("slot_ordinal_invalid")
    root = Path(context["root"])
    slot = _mapping(context["slots"][ordinal - 1], "slot")
    job = _mapping(context["jobs"][ordinal - 1], "job")
    if (
        slot.get("ordinal") != ordinal
        or job.get("ordinal") != ordinal
        or job.get("status") != "NOT_STARTED"
        or job.get("gameplay_seed") != slot.get("gameplay_seed")
        or job.get("plan") != _mapping(slot.get("collector_plan"), "collector_plan").get("path")
        or job.get("output_root") != slot.get("output_root_path")
    ):
        raise DistributionCampaignError("slot_job_mismatch")
    arguments = _sequence(job.get("arguments"), "job_arguments")
    if (
        any(not isinstance(value, str) for value in arguments)
        or any(
            value in arguments
            for value in ("--plan", "--prestate", "--host-restore", "--output-root")
        )
        or arguments.count("--maximum-attempts") != 1
        or arguments[arguments.index("--maximum-attempts") + 1] != "1"
    ):
        raise DistributionCampaignError("slot_job_arguments_invalid")
    plan_path = _bound_member(root, job.get("plan"), "slot_plan")
    prestate_path = _bound_member(root, job.get("prestate"), "slot_prestate")
    host_restore_path = _bound_member(
        root,
        job.get("host_restore"),
        "slot_host_restore",
    )
    output_root = _future_member(root, job.get("output_root"), "slot_output")
    return slot, job, plan_path, prestate_path, host_restore_path, output_root


def run_distribution_slot(
    *,
    campaign_root: Path,
    original_root: Path,
    ordinal: int,
) -> Path:
    context = _load_campaign_context(campaign_root)
    status = campaign_collection_status(
        campaign_root=campaign_root,
        original_root=original_root,
    )
    if status["status"] != "READY" or status["next_slot_ordinal"] != ordinal:
        raise DistributionCampaignError("slot_is_not_next_frozen_job")
    root = Path(context["root"])
    original = _absolute_directory(original_root, "original_root")
    slot, job, plan_path, prestate_path, host_restore_path, output_root = (
        _slot_paths_and_job(context, ordinal)
    )
    if output_root.exists():
        raise DistributionCampaignError("slot_output_already_exists")
    plan = _mapping(read_canonical_json(plan_path), "slot_plan")
    nonce = plan.get("session_nonce")
    if not isinstance(nonce, str):
        raise DistributionCampaignError("slot_plan_nonce_invalid")
    frozen_host = PcStateSnapshot.read(host_restore_path)
    live_pre = capture_state(
        session_nonce=nonce,
        phase=f"pc-{ordinal:03d}-pre",
    )
    if (
        live_pre.session_nonce != frozen_host.session_nonce
        or live_pre.state_root != frozen_host.state_root
    ):
        raise DistributionCampaignError("live_host_state_differs_from_frozen_restore")
    transactions_root = root / "transactions"
    transactions_root.mkdir(exist_ok=True)
    pre_path = transactions_root / f"pc-{ordinal:03d}-pre.json"
    post_path = transactions_root / f"pc-{ordinal:03d}-post.json"
    if pre_path.exists() or post_path.exists():
        raise DistributionCampaignError("slot_transaction_already_exists")
    write_snapshot_exclusive(live_pre, pre_path)
    command = [
        sys.executable,
        str(context["current_collector"]),
        "--plan",
        str(plan_path),
        "--prestate",
        str(prestate_path),
        "--host-restore",
        str(host_restore_path),
        "--output-root",
        str(output_root),
        *_sequence(job.get("arguments"), "job_arguments"),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
    except OSError as error:
        raise DistributionCampaignError("slot_collector_launch_failed") from error
    live_post = capture_state(
        session_nonce=nonce,
        phase=f"pc-{ordinal:03d}-post",
    )
    write_snapshot_exclusive(live_post, post_path)
    if live_post.state_root != live_pre.state_root:
        raise DistributionCampaignError("slot_host_state_not_restored")
    if completed.returncode != 0:
        failure_path = transactions_root / f"pc-{ordinal:03d}-failure.json"
        _write_exclusive(
            failure_path,
            {
                "schema": SLOT_EXECUTION_RECEIPT_SCHEMA,
                "version": SLOT_EXECUTION_RECEIPT_VERSION,
                "status": "FAIL",
                "ordinal": ordinal,
                "collector_exit_code": completed.returncode,
                "pre_state_root": live_pre.state_root,
                "post_state_root": live_post.state_root,
            },
        )
        raise DistributionCampaignError("formal_slot_collector_failed")
    attempts_path = output_root / "attempts.json"
    attempts = _sequence(read_canonical_json(attempts_path), "slot_attempts")
    if len(attempts) != 1:
        raise DistributionCampaignError("slot_attempt_count_invalid")
    attempt = _mapping(attempts[0], "slot_attempt")
    if (
        attempt.get("attempt") != 1
        or attempt.get("status") != "PASS"
        or isinstance(attempt.get("process_id"), bool)
        or not isinstance(attempt.get("process_id"), int)
        or attempt["process_id"] <= 0
        or isinstance(attempt.get("process_creation_filetime_100ns"), bool)
        or not isinstance(attempt.get("process_creation_filetime_100ns"), int)
        or attempt["process_creation_filetime_100ns"] <= 0
    ):
        raise DistributionCampaignError("slot_attempt_invalid")
    attempt_root = output_root / "attempt-01"
    memory_probe_path = _absolute_file(
        attempt_root / "memory-probe.json",
        "slot_memory_probe",
    )
    trajectory_index_path = _absolute_file(
        attempt_root / "trajectory" / "index.json",
        "slot_trajectory_index",
    )
    original_executable = _absolute_file(
        original / "ZumasRevenge.exe",
        "original_executable",
    )
    manifest_path = _future_member(
        root,
        slot.get("source_manifest_path"),
        "slot_source_manifest",
    )
    manifest = {
        "schema": "zuma-rl.pc-source-manifest",
        "version": 1,
        "source_id": slot["source_id"],
        "scope": {
            "level_id": "Jungle2",
            "hard": False,
            "profile_mode": "tutorials_completed",
            "mode": "adventure",
        },
        "original_executable_sha256": _sha256_path(original_executable),
        "runtime_payload": context["preregistration"]["pc_capture"][
            "runtime_payload"
        ],
        "dmo": slot["dmo"],
        "run": {
            "run_id": slot["run_id"],
            "selected_attempt": 1,
            "attempts": _artifact_binding(attempts_path, root),
            "memory_probe": _artifact_binding(memory_probe_path, root),
            "trajectory_index": _artifact_binding(
                trajectory_index_path,
                root,
            ),
            "process_id": attempt["process_id"],
            "process_creation_filetime_100ns": attempt[
                "process_creation_filetime_100ns"
            ],
            "maximum_startup_attempts": 1,
        },
        "window": {
            "freeze_update": PC_FREEZE_UPDATE,
            "start_update": PC_SAMPLE_UPDATE,
            "end_update": PC_SAMPLE_UPDATE,
            "warmup_tick_count": PC_WARMUP_TICK_COUNT,
        },
        "state_transaction": {
            "pre_snapshot": _artifact_binding(pre_path, root),
            "post_snapshot": _artifact_binding(post_path, root),
        },
    }
    _write_exclusive(manifest_path, manifest)
    report = verify_pc_source_manifest(
        manifest_path,
        evidence_root=root,
        original_root=original,
    )
    if report.get("status") != "PASS":
        raise DistributionCampaignError("slot_source_verification_failed")
    verification_path = output_root / "source-verification.json"
    _write_exclusive(verification_path, report)
    receipt_path = output_root / "execution-receipt.json"
    _write_exclusive(
        receipt_path,
        {
            "schema": SLOT_EXECUTION_RECEIPT_SCHEMA,
            "version": SLOT_EXECUTION_RECEIPT_VERSION,
            "status": "PASS",
            "campaign_id": context["campaign"]["campaign_id"],
            "ordinal": ordinal,
            "gameplay_seed": slot["gameplay_seed"],
            "collector_sha256": context["collector_sha256"],
            "trace_tool_sha256": context["trace_sha256"],
            "replay_boundary_tool_sha256": context[
                "replay_boundary_sha256"
            ],
            "collector_exit_code": completed.returncode,
            "process_id": attempt["process_id"],
            "process_creation_filetime_100ns": attempt[
                "process_creation_filetime_100ns"
            ],
            "pre_state_root": live_pre.state_root,
            "post_state_root": live_post.state_root,
            "source_manifest": _artifact_binding(manifest_path, root),
            "source_verification": _artifact_binding(
                verification_path,
                root,
            ),
        },
    )
    return receipt_path


def finalize_distribution_campaign(
    *,
    campaign_root: Path,
    original_root: Path,
) -> Path:
    context = _load_campaign_context(campaign_root)
    root = Path(context["root"])
    original = _absolute_directory(original_root, "original_root")
    status = campaign_collection_status(
        campaign_root=root,
        original_root=original,
    )
    if status["all_slots_complete"] is not True:
        raise DistributionCampaignError("campaign_slots_incomplete")
    pc_dataset_path = root / "pc-dataset.json"
    audit_path = root / "audit.json"
    verification_path = root / "audit-verification.json"
    completion_path = root / "completion.json"
    if any(
        path.exists()
        for path in (
            pc_dataset_path,
            audit_path,
            verification_path,
            completion_path,
        )
    ):
        raise DistributionCampaignError("campaign_finalization_already_exists")
    samples: list[dict[str, Any]] = []
    for ordinal, slot_value in enumerate(context["slots"], start=1):
        slot = _mapping(slot_value, f"slot_{ordinal}")
        manifest_path = _bound_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        samples.append(
            {
                "ordinal": ordinal,
                "source_manifest": _artifact_binding(manifest_path, root),
            }
        )
    _write_exclusive(
        pc_dataset_path,
        {
            "schema": DATASET_SCHEMA,
            "version": DATASET_VERSION,
            "population": "pc",
            "metric": METRIC_NAME,
            "samples": samples,
        },
    )
    _write_exclusive(
        audit_path,
        {
            "schema": AUDIT_SCHEMA,
            "version": AUDIT_VERSION,
            "status": "READY_FOR_RECOMPUTE",
            "scope": context["preregistration"]["scope"],
            "preregistration": _artifact_binding(
                context["preregistration_path"],
                root,
            ),
            "pc_dataset": _artifact_binding(pc_dataset_path, root),
            "simulator_dataset": _artifact_binding(
                context["simulator_dataset_path"],
                root,
            ),
        },
    )
    report = verify_distribution_audit(
        audit_path,
        evidence_root=root,
        original_root=original,
        policy_id=DISTRIBUTION_POLICY_ID,
    )
    _write_exclusive(verification_path, report)
    _write_exclusive(
        completion_path,
        {
            "schema": CAMPAIGN_COMPLETION_SCHEMA,
            "version": CAMPAIGN_COMPLETION_VERSION,
            "status": report["status"],
            "campaign_id": context["campaign"]["campaign_id"],
            "completed_pc_slots": MINIMUM_PC_SAMPLES,
            "pc_dataset": _artifact_binding(pc_dataset_path, root),
            "audit": _artifact_binding(audit_path, root),
            "audit_verification": _artifact_binding(
                verification_path,
                root,
            ),
        },
    )
    return completion_path


def resume_v7_distribution_finalization(
    *,
    campaign_root: Path,
    original_root: Path,
    recovered_utc: str | None = None,
) -> Path:
    """Resume the exact v7 finalization after the stale v1 binding rejection."""

    context = _load_campaign_context(campaign_root)
    root = Path(context["root"])
    original = _absolute_directory(original_root, "original_root")
    campaign = _mapping(context["campaign"], "campaign")
    if campaign.get("campaign_id") != V7_FINALIZATION_RECOVERY_CAMPAIGN_ID:
        raise DistributionCampaignError("finalization_recovery_campaign_invalid")
    status = campaign_collection_status(
        campaign_root=root,
        original_root=original,
    )
    if status["all_slots_complete"] is not True:
        raise DistributionCampaignError("campaign_slots_incomplete")

    pc_dataset_path = _absolute_file(
        root / "pc-dataset.json",
        "partial_pc_dataset",
    )
    audit_path = _absolute_file(root / "audit.json", "partial_audit")
    recovery_path = root / "finalization-recovery.json"
    verification_path = root / "audit-verification.json"
    completion_path = root / "completion.json"
    if any(
        path.exists()
        for path in (recovery_path, verification_path, completion_path)
    ):
        raise DistributionCampaignError("finalization_recovery_already_exists")
    if (
        _sha256_path(pc_dataset_path) != V7_PARTIAL_PC_DATASET_SHA256
        or _sha256_path(audit_path) != V7_PARTIAL_AUDIT_SHA256
    ):
        raise DistributionCampaignError("partial_finalization_artifact_drift")

    expected_samples: list[dict[str, Any]] = []
    source_verifications: list[dict[str, str]] = []
    observed_binding_versions: set[int] = set()
    for ordinal, slot_value in enumerate(context["slots"], start=1):
        slot = _mapping(slot_value, f"slot_{ordinal}")
        manifest_path = _bound_member(
            root,
            slot.get("source_manifest_path"),
            f"slot_{ordinal}_manifest",
        )
        expected_samples.append(
            {
                "ordinal": ordinal,
                "source_manifest": _artifact_binding(manifest_path, root),
            }
        )
        output_root = _future_member(
            root,
            slot.get("output_root_path"),
            f"slot_{ordinal}_output",
        )
        source_verification_path = _absolute_file(
            output_root / "source-verification.json",
            f"slot_{ordinal}_source_verification",
        )
        recomputed = verify_pc_source_manifest(
            manifest_path,
            evidence_root=root,
            original_root=original,
        )
        if read_canonical_json(source_verification_path) != recomputed:
            raise DistributionCampaignError("source_verification_drift")
        strict = _mapping(
            recomputed.get("strict_command_replay"),
            f"slot_{ordinal}_strict_command_replay",
        )
        binding_version = strict.get("version")
        if (
            strict.get("schema") != STRICT_REPLAY_BINDING_SCHEMA
            or isinstance(binding_version, bool)
            or not isinstance(binding_version, int)
            or binding_version != STRICT_REPLAY_BINDING_VERSION
        ):
            raise DistributionCampaignError(
                "finalization_recovery_binding_version_invalid"
            )
        observed_binding_versions.add(binding_version)
        source_verifications.append(
            _artifact_binding(source_verification_path, root)
        )

    expected_pc_dataset = {
        "schema": DATASET_SCHEMA,
        "version": DATASET_VERSION,
        "population": "pc",
        "metric": METRIC_NAME,
        "samples": expected_samples,
    }
    if read_canonical_json(pc_dataset_path) != expected_pc_dataset:
        raise DistributionCampaignError("partial_pc_dataset_not_exact")
    expected_audit = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "status": "READY_FOR_RECOMPUTE",
        "scope": context["preregistration"]["scope"],
        "preregistration": _artifact_binding(
            context["preregistration_path"],
            root,
        ),
        "pc_dataset": _artifact_binding(pc_dataset_path, root),
        "simulator_dataset": _artifact_binding(
            context["simulator_dataset_path"],
            root,
        ),
    }
    if read_canonical_json(audit_path) != expected_audit:
        raise DistributionCampaignError("partial_audit_not_exact")

    validator_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "zuma_rl"
        / "distribution_fidelity.py"
    )
    if (
        _sha256_path(validator_path)
        != V7_CORRECTED_DISTRIBUTION_VALIDATOR_SHA256
    ):
        raise DistributionCampaignError("corrected_distribution_validator_drift")
    report = verify_distribution_audit(
        audit_path,
        evidence_root=root,
        original_root=original,
        policy_id=DISTRIBUTION_POLICY_ID,
    )
    if report.get("status") not in {"PASS", "FAIL"}:
        raise DistributionCampaignError("distribution_report_status_invalid")

    recovery = {
        "schema": FINALIZATION_RECOVERY_SCHEMA,
        "version": FINALIZATION_RECOVERY_VERSION,
        "status": "APPLIED",
        "classification": "POST_HOC_VALIDATOR_SCHEMA_VERSION_SYNC",
        "campaign_id": campaign["campaign_id"],
        "recovered_utc": _frozen_utc(recovered_utc),
        "original_failure": {
            "stage": "VERIFY_DISTRIBUTION_AUDIT",
            "exception_type": "DistributionFidelityError",
            "message": "PC source gameplay seed or strict replay differs",
            "stale_validator_sha256": (
                V7_STALE_DISTRIBUTION_VALIDATOR_SHA256
            ),
            "stale_expected_strict_binding_version": 1,
        },
        "correction": {
            "mechanism": "AUTHORITATIVE_PC_SOURCE_BINDING_VERSION_IMPORT",
            "corrected_validator_sha256": (
                V7_CORRECTED_DISTRIBUTION_VALIDATOR_SHA256
            ),
            "strict_binding_schema": STRICT_REPLAY_BINDING_SCHEMA,
            "strict_binding_version": STRICT_REPLAY_BINDING_VERSION,
            "observed_binding_versions": sorted(observed_binding_versions),
            "sample_count": len(source_verifications),
            "data_dependent_threshold_changed": False,
            "sample_selection_changed": False,
            "sample_artifact_changed": False,
            "statistical_recipe_changed": False,
        },
        "frozen_data": {
            "pc_dataset": _artifact_binding(pc_dataset_path, root),
            "audit": _artifact_binding(audit_path, root),
            "source_verifications": source_verifications,
        },
        "recomputation": {
            "status": report["status"],
            "pc_sample_count": report["pc_sample_count"],
            "simulator_sample_count": report["simulator_sample_count"],
            "total_variation": report["total_variation"],
            "compatibility_p_value": report["compatibility_p_value"],
        },
        "implementation": {
            "recovery_manager_sha256": _sha256_path(Path(__file__).resolve()),
            "frozen_seed_deriver": context["preregistration"]["pc_capture"][
                "seed_deriver"
            ],
        },
    }
    verification_binding = {
        "path": _evidence_relative(
            verification_path,
            root,
            "audit_verification",
        ),
        "sha256": _sha256_bytes(canonical_json_bytes(report)),
    }
    completion = {
        "schema": CAMPAIGN_COMPLETION_SCHEMA,
        "version": CAMPAIGN_COMPLETION_VERSION,
        "status": report["status"],
        "campaign_id": campaign["campaign_id"],
        "completed_pc_slots": MINIMUM_PC_SAMPLES,
        "pc_dataset": _artifact_binding(pc_dataset_path, root),
        "audit": _artifact_binding(audit_path, root),
        "audit_verification": verification_binding,
    }
    _write_exclusive(recovery_path, recovery)
    _write_exclusive(verification_path, report)
    _write_exclusive(completion_path, completion)
    return completion_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive-plan")
    derive.add_argument("--base-root", required=True, type=Path)
    derive.add_argument("--output-root", required=True, type=Path)
    derive.add_argument("--direct-runtime", required=True, type=Path)
    derive.add_argument("--changedir", required=True, type=Path)
    derive.add_argument(
        "--strict-command-broker-stop-update",
        type=int,
        help=(
            "Use the audited gapless command broker through this update "
            "while observing, never replacing, the retail RNG seed."
        ),
    )
    seed_role = derive.add_mutually_exclusive_group()
    seed_role.add_argument(
        "--pc-slot-ordinal",
        type=int,
        help=(
            "Derive this frozen distribution-v3 PC slot (1 through 32) "
            "by changing only the DMO gameplay-seed field."
        ),
    )
    seed_role.add_argument(
        "--mechanism-probe",
        action="store_true",
        help=(
            "Use the deterministic probe seed that is excluded from both "
            "formal PC and simulator populations."
        ),
    )
    derive.add_argument(
        "--evidence-root",
        type=Path,
        help=(
            "Evidence root used to bind base and derived DMO identities; "
            "required with --pc-slot-ordinal."
        ),
    )
    prepare = commands.add_parser("prepare-campaign")
    prepare.add_argument("--base-root", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--direct-runtime", required=True, type=Path)
    prepare.add_argument("--changedir", required=True, type=Path)
    prepare.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    prepare.add_argument(
        "--frozen-utc",
        help="Testing/reproduction override for the preregistration timestamp.",
    )
    status = commands.add_parser("campaign-status")
    status.add_argument("--campaign-root", required=True, type=Path)
    status.add_argument("--original-root", required=True, type=Path)
    invalidate = commands.add_parser("invalidate-failed-campaign")
    invalidate.add_argument("--campaign-root", required=True, type=Path)
    invalidate.add_argument(
        "--superseded-by-campaign-id",
        required=True,
    )
    invalidate.add_argument(
        "--invalidated-utc",
        help="Testing/reproduction override for the invalidation timestamp.",
    )
    invalidate_partial = commands.add_parser("invalidate-partial-campaign")
    invalidate_partial.add_argument("--campaign-root", required=True, type=Path)
    invalidate_partial.add_argument("--original-root", required=True, type=Path)
    invalidate_partial.add_argument(
        "--superseded-by-campaign-id",
        required=True,
    )
    invalidate_partial.add_argument(
        "--invalidated-utc",
        help="Testing/reproduction override for the invalidation timestamp.",
    )
    run_slot = commands.add_parser("run-slot")
    run_slot.add_argument("--campaign-root", required=True, type=Path)
    run_slot.add_argument("--original-root", required=True, type=Path)
    run_slot.add_argument("--ordinal", required=True, type=int)
    finalize = commands.add_parser("finalize-campaign")
    finalize.add_argument("--campaign-root", required=True, type=Path)
    finalize.add_argument("--original-root", required=True, type=Path)
    resume = commands.add_parser("resume-v7-finalization")
    resume.add_argument("--campaign-root", required=True, type=Path)
    resume.add_argument("--original-root", required=True, type=Path)
    resume.add_argument(
        "--recovered-utc",
        help="Testing/reproduction override for the recovery timestamp.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "derive-plan":
        path = derive_natural_direct_plan(
            base_root=args.base_root,
            output_root=args.output_root,
            direct_runtime=args.direct_runtime,
            changedir=args.changedir,
            strict_command_broker_stop_update=(
                args.strict_command_broker_stop_update
            ),
            pc_slot_ordinal=args.pc_slot_ordinal,
            mechanism_probe=args.mechanism_probe,
            evidence_root=args.evidence_root,
        )
        print(path)
        return 0
    if args.command == "prepare-campaign":
        path = prepare_distribution_campaign(
            base_root=args.base_root,
            output_root=args.output_root,
            direct_runtime=args.direct_runtime,
            changedir=args.changedir,
            campaign_id=args.campaign_id,
            frozen_utc=args.frozen_utc,
        )
        print(path)
        return 0
    if args.command == "campaign-status":
        report = campaign_collection_status(
            campaign_root=args.campaign_root,
            original_root=args.original_root,
        )
        print(canonical_json_bytes(report).decode("ascii"), end="")
        return 0
    if args.command == "invalidate-failed-campaign":
        path = invalidate_failed_distribution_campaign(
            campaign_root=args.campaign_root,
            superseded_by_campaign_id=args.superseded_by_campaign_id,
            invalidated_utc=args.invalidated_utc,
        )
        print(path)
        return 0
    if args.command == "invalidate-partial-campaign":
        path = invalidate_partial_distribution_campaign(
            campaign_root=args.campaign_root,
            original_root=args.original_root,
            superseded_by_campaign_id=args.superseded_by_campaign_id,
            invalidated_utc=args.invalidated_utc,
        )
        print(path)
        return 0
    if args.command == "run-slot":
        path = run_distribution_slot(
            campaign_root=args.campaign_root,
            original_root=args.original_root,
            ordinal=args.ordinal,
        )
        print(path)
        return 0
    if args.command == "finalize-campaign":
        path = finalize_distribution_campaign(
            campaign_root=args.campaign_root,
            original_root=args.original_root,
        )
        print(path)
        return 0
    if args.command == "resume-v7-finalization":
        path = resume_v7_distribution_finalization(
            campaign_root=args.campaign_root,
            original_root=args.original_root,
            recovered_utc=args.recovered_utc,
        )
        print(path)
        return 0
    raise DistributionCampaignError("command_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
