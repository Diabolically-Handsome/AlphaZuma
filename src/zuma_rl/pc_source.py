"""Independent validation for one authentic retail exact-step trajectory.

Unlike a PC Golden, a PC source manifest does not claim that two retail
processes reproduce identical RNG state or pixels.  It proves the narrower
facts needed by Fidelity Gate v2: one unmodified, input-isolated retail run
was captured with a complete per-tick trajectory and the user's state was
restored afterwards.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from zuma_rl.pc_exact_step_evidence import (
    PcExactStepEvidenceError,
    canonical_json_bytes,
    load_formal_exact_step_run,
    read_canonical_json,
)
from zuma_rl.pc_golden import PcGoldenValidationError
from zuma_rl.pc_full_state_evidence import (
    PcFullStateEvidenceError,
    load_formal_full_state_run,
)
from zuma_rl.pc_protocol_evidence import PcStateSnapshot
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError
from zuma_rl.retail_dmo_provenance import (
    RetailDmoProvenanceError,
    certifying_recording_outcome,
    read_certifying_provenance,
)

SOURCE_SCHEMA = "zuma-rl.pc-source-manifest"
SOURCE_VERSION = 1
FULL_STATE_SOURCE_VERSION = 2
FULL_STATE_SOURCE_KIND = "formal_full_state_exact_step"
SOURCE_BOUND_FULL_STATE_SOURCE_VERSION = 3
SOURCE_BOUND_FULL_STATE_SOURCE_KIND = (
    "formal_full_state_source_bound_original_replay"
)
SOURCE_REPORT_SCHEMA = "zuma-rl.pc-source-verification"
SOURCE_REPORT_VERSION = 1
STRICT_REPLAY_BINDING_SCHEMA = (
    "zuma-rl.pc-natural-strict-command-replay-binding"
)
LEGACY_STRICT_REPLAY_BINDING_VERSION = 1
STRICT_REPLAY_BINDING_VERSION = 2
SOURCE_BOUND_REPLAY_BINDING_SCHEMA = (
    "zuma-rl.pc-source-bound-original-command-replay-binding"
)
SOURCE_BOUND_REPLAY_BINDING_VERSION = 1
OUTCOME_SOURCE_BOUND_REPLAY_BINDING_VERSION = 2

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class PcSourceValidationError(ValueError):
    """A source manifest or one of its bound artifacts is invalid."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PcSourceValidationError(f"{name} must be an object")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise PcSourceValidationError(f"{name} fields are invalid: {'; '.join(details)}")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PcSourceValidationError(
            f"{name} must be an integer at least {minimum}"
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise PcSourceValidationError(f"{name} is not a valid identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PcSourceValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PcSourceValidationError("a bound source artifact cannot be read") from error
    return "sha256:" + digest.hexdigest()


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PcSourceValidationError(f"{name} must be a relative POSIX path")
    if "\\" in value:
        raise PcSourceValidationError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PcSourceValidationError(f"{name} must stay inside evidence_root")
    return path


def _resolve(root: Path, value: Any, name: str) -> Path:
    relative = _relative_path(value, name)
    resolved_root = root.resolve(strict=True)
    try:
        resolved = resolved_root.joinpath(*relative.parts).resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise PcSourceValidationError(f"{name} does not resolve inside evidence_root") from error
    if not resolved.is_file():
        raise PcSourceValidationError(f"{name} must identify a file")
    return resolved


def _bound_path(
    root: Path,
    value: Mapping[str, Any],
    name: str,
) -> Path:
    _check_keys(value, required={"path", "sha256"}, name=name)
    path = _resolve(root, value["path"], f"{name}.path")
    expected = _digest(value["sha256"], f"{name}.sha256")
    if _sha256_path(path) != expected:
        raise PcSourceValidationError(f"{name} SHA-256 differs")
    return path


def _source_fingerprint(
    *,
    run_id: str,
    process_creation_filetime_100ns: int,
    dmo_sha256: str,
    runtime_sha256: str,
    probe_sha256: str,
    index_sha256: str,
) -> str:
    payload = {
        "dmo_sha256": dmo_sha256,
        "index_sha256": index_sha256,
        "probe_sha256": probe_sha256,
        "process_creation_filetime_100ns": process_creation_filetime_100ns,
        "run_id": run_id,
        "runtime_sha256": runtime_sha256,
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_strict_command_replay(
    probe_path: Path,
    *,
    expected_process_id: int,
    expected_runtime_path: Path,
    expected_runtime_sha256: str,
    expected_dmo_path: Path,
    expected_dmo_sha256: str,
) -> Mapping[str, Any] | None:
    try:
        probe = _mapping(read_canonical_json(probe_path), "memory probe")
    except PcExactStepEvidenceError as error:
        raise PcSourceValidationError(str(error)) from error

    value = probe.get("strict_command_replay")
    if value is None:
        return None
    binding = _mapping(value, "strict_command_replay")
    if binding.get("schema") == SOURCE_BOUND_REPLAY_BINDING_SCHEMA:
        return None
    if binding.get("schema") != STRICT_REPLAY_BINDING_SCHEMA:
        raise PcSourceValidationError(
            "strict replay binding schema is unsupported"
        )
    _check_keys(
        binding,
        required={
            "schema",
            "version",
            "status",
            "artifact",
            "artifact_bytes",
            "artifact_sha256",
            "runtime_process_id",
            "runtime_executable",
            "runtime_executable_sha256",
            "source_dmo",
            "source_dmo_sha256",
            "stop_after_update",
            "failure_count",
            "startup_seed",
            "seed_source",
            "register_override",
            "rng_seed_override_count",
            "rng_process_memory_writes",
            "font_cache_manifest",
            "transport",
        },
        name="strict_command_replay",
    )
    artifact_name = binding.get("artifact")
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
    ):
        raise PcSourceValidationError("strict replay artifact path is invalid")
    artifact_path = (probe_path.parent / artifact_name).resolve()
    try:
        artifact_path.relative_to(probe_path.parent.resolve())
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise PcSourceValidationError("strict replay artifact is invalid") from error
    payload = _mapping(payload, "strict replay artifact")
    _check_keys(
        payload,
        required={
            "finished_utc",
            "options",
            "result",
            "runtime_executable",
            "runtime_process_id",
            "schema",
            "source_dmo",
            "started_utc",
            "startup_rng",
            "trace_finished_perf_counter_ns",
            "trace_started_perf_counter_ns",
        },
        name="strict replay artifact",
    )
    options = _mapping(payload.get("options"), "strict replay options")
    startup_rng = _mapping(
        payload.get("startup_rng"),
        "strict replay startup_rng",
    )
    result = _mapping(payload.get("result"), "strict replay result")
    source_dmo = _mapping(
        payload.get("source_dmo"),
        "strict replay source_dmo",
    )
    _check_keys(
        source_dmo,
        required={"path", "sha256", "size_bytes"},
        name="strict replay source_dmo",
    )
    manifest = _mapping(
        binding.get("font_cache_manifest"),
        "strict replay font_cache_manifest",
    )
    _check_keys(
        manifest,
        required={"entry_count", "manifest_sha256", "main_pak_sha256"},
        name="strict replay font_cache_manifest",
    )
    transport = _mapping(
        binding.get("transport"),
        "strict replay transport",
    )
    binding_version = binding.get("version")
    if binding_version not in (
        LEGACY_STRICT_REPLAY_BINDING_VERSION,
        STRICT_REPLAY_BINDING_VERSION,
    ):
        raise PcSourceValidationError(
            "strict replay binding version is unsupported"
        )
    transport_fields = {
        "service_broker",
        "startup_worker_yield_count",
        "temporary_code_breakpoint_memory_write_count",
        "callback_context_emulation_count",
        "gameplay_or_rng_process_memory_write_count",
    }
    if binding_version == STRICT_REPLAY_BINDING_VERSION:
        transport_fields.update(
            {
                "replay_state_recovery_count",
                "replay_state_recovery_memory_write_count",
                "replay_state_recovery_memory_write_bytes",
            }
        )
    _check_keys(
        transport,
        required=transport_fields,
        name="strict replay transport",
    )
    worker_yields = result.get("startup_post_bypass_worker_yields")
    if not isinstance(worker_yields, list):
        raise PcSourceValidationError("strict replay worker receipt is invalid")
    temporary_writes = 0
    for receipt_value in worker_yields:
        receipt = _mapping(receipt_value, "strict replay worker receipt")
        writes = receipt.get("temporary_breakpoint_memory_writes")
        if (
            receipt.get("process_id") != expected_process_id
            or receipt.get("memory_writes") != 0
            or receipt.get("failure") is not None
            or receipt.get("command_breakpoint_rearmed") is not True
            or isinstance(writes, bool)
            or not isinstance(writes, int)
            or writes <= 0
        ):
            raise PcSourceValidationError(
                "strict replay worker receipt is invalid"
            )
        temporary_writes += writes
    callback_groups = (
        "deferred_file_write_discharges",
        "font_cache_manifest_completion_discharges",
        "terminal_file_write_payload_handoffs",
        "service_file_write_header_claims",
    )
    callback_count = 0
    for name in callback_groups:
        rows = result.get(name)
        if not isinstance(rows, list):
            raise PcSourceValidationError(
                "strict replay callback receipt is invalid"
            )
        callback_count += len(rows)
    recovery_count = 0
    recovery_write_count = 0
    recovery_write_bytes = 0
    recovery_rows = result.get(
        "preloading_failed_file_write_tail_short_header_recoveries"
    )
    if binding_version == LEGACY_STRICT_REPLAY_BINDING_VERSION:
        if recovery_rows is not None:
            raise PcSourceValidationError(
                "legacy strict replay contains a recovery receipt"
            )
    else:
        if not isinstance(recovery_rows, list) or len(recovery_rows) > 1:
            raise PcSourceValidationError(
                "strict replay recovery receipt is invalid"
            )
        for receipt_value in recovery_rows:
            receipt = _mapping(
                receipt_value,
                "strict replay recovery receipt",
            )
            expected_before = {
                "buffer_read_bit_position": 1888,
                "last_demo_update": 401,
                "needs_command": 0,
                "is_short": 1,
                "command_number": 1,
                "command_order": 68,
                "command_bit_position": 1882,
                "demo_loading_complete": 0,
            }
            expected_after = {
                "buffer_read_bit_position": 1860,
                "last_demo_update": 391,
                "needs_command": 1,
                "is_short": 0,
                "command_number": 16,
                "command_order": 67,
                "command_bit_position": 1849,
                "demo_loading_complete": 0,
            }
            if (
                set(receipt)
                != {
                    "mechanism",
                    "process_id",
                    "thread_id",
                    "framework_update",
                    "false_header_row_index",
                    "corridor_start_index",
                    "corridor_end_index",
                    "command_order_offset",
                    "command_order_rebase_rows",
                    "before",
                    "after",
                    "false_payload_executed",
                    "write_count",
                    "bytes_written",
                }
                or receipt.get("mechanism")
                != "audited_false_short_header_pre_payload_recovery"
                or receipt.get("process_id") != expected_process_id
                or receipt.get("thread_id") != startup_rng.get("thread_id")
                or receipt.get("framework_update") != 401
                or receipt.get("false_header_row_index") != 74
                or receipt.get("corridor_start_index") != 45
                or receipt.get("corridor_end_index") != 71
                or receipt.get("command_order_offset") != 4
                or receipt.get("command_order_rebase_rows")
                != [64, 65, 68, 69]
                or receipt.get("before") != expected_before
                or receipt.get("after") != expected_after
                or receipt.get("false_payload_executed") is not False
                or receipt.get("write_count") != 8
                or receipt.get("bytes_written") != 23
            ):
                raise PcSourceValidationError(
                    "strict replay recovery receipt is invalid"
                )
            recovery_count += 1
            recovery_write_count += 8
            recovery_write_bytes += 23
    natural_control_nulls = (
        "crt_rand_seed",
        "startup_seed_transport",
        "board_seed",
        "global_rng_seed",
        "thread_crt_rng_seed",
        "gameplay_mtrand_oracle",
        "gameplay_mtrand_oracle_seed",
        "initial_global_mtrand_oracle",
        "initial_global_mtrand_seed",
        "startup_global_mtrand_observation_oracle",
        "startup_global_mtrand_observation_seed",
        "global_mtrand_restore_log",
        "global_mtrand_restore_seed",
        "global_mtrand_restore_command_order",
        "qrand_restore_log",
        "qrand_restore_command_order",
        "thread_crt_restore_log",
        "thread_crt_restore_command_order",
        "source_bound_board_seed",
    )
    try:
        bound_runtime_path = Path(str(binding.get("runtime_executable"))).resolve(
            strict=True
        )
        payload_runtime_path = Path(str(payload.get("runtime_executable"))).resolve(
            strict=True
        )
        bound_dmo_path = Path(str(binding.get("source_dmo"))).resolve(
            strict=True
        )
        payload_dmo_path = Path(str(source_dmo.get("path"))).resolve(
            strict=True
        )
    except OSError as error:
        raise PcSourceValidationError(
            "strict replay source identity is unavailable"
        ) from error
    digest = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    startup_seed = binding.get("startup_seed")
    stop_after_update = binding.get("stop_after_update")
    probe_update = probe.get("framework_update")
    if (
        binding.get("schema") != STRICT_REPLAY_BINDING_SCHEMA
        or binding_version
        not in (
            LEGACY_STRICT_REPLAY_BINDING_VERSION,
            STRICT_REPLAY_BINDING_VERSION,
        )
        or binding.get("status") != "PASS"
        or binding.get("artifact_bytes") != len(artifact_bytes)
        or binding.get("artifact_sha256") != digest
        or binding.get("runtime_process_id") != expected_process_id
        or bound_runtime_path != expected_runtime_path.resolve()
        or payload_runtime_path != expected_runtime_path.resolve()
        or binding.get("runtime_executable_sha256")
        != expected_runtime_sha256
        or bound_dmo_path != expected_dmo_path.resolve()
        or payload_dmo_path != expected_dmo_path.resolve()
        or binding.get("source_dmo_sha256") != expected_dmo_sha256
        or isinstance(stop_after_update, bool)
        or not isinstance(stop_after_update, int)
        or stop_after_update <= 0
        or (
            probe_update is not None
            and (
                isinstance(probe_update, bool)
                or not isinstance(probe_update, int)
                or stop_after_update >= probe_update
            )
        )
        or binding.get("failure_count") != 0
        or binding.get("seed_source")
        != "retail_eax_before_push_to_srand"
        or isinstance(startup_seed, bool)
        or not isinstance(startup_seed, int)
        or not 0 <= startup_seed <= 0xFFFFFFFF
        or binding.get("register_override") is not None
        or binding.get("rng_seed_override_count") != 0
        or binding.get("rng_process_memory_writes") != 0
        or transport.get("service_broker") is not True
        or transport.get("startup_worker_yield_count") != len(worker_yields)
        or transport.get("temporary_code_breakpoint_memory_write_count")
        != temporary_writes
        or transport.get("callback_context_emulation_count")
        != callback_count
        or (
            binding_version == STRICT_REPLAY_BINDING_VERSION
            and (
                transport.get("replay_state_recovery_count")
                != recovery_count
                or transport.get(
                    "replay_state_recovery_memory_write_count"
                )
                != recovery_write_count
                or transport.get(
                    "replay_state_recovery_memory_write_bytes"
                )
                != recovery_write_bytes
            )
        )
        or transport.get("gameplay_or_rng_process_memory_write_count") != 0
        or payload.get("schema") != "zuma.popcap_strict_replay.v3"
        or payload.get("runtime_process_id") != expected_process_id
        or source_dmo.get("sha256")
        != expected_dmo_sha256.removeprefix("sha256:")
        or source_dmo.get("size_bytes") != expected_dmo_path.stat().st_size
        or options.get("direct_runtime") is not True
        or options.get("direct_natural_seed") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("rng_seed_override_count") != 0
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != stop_after_update
        or options.get("startup_trace_handoff") is not True
        or options.get("allow_font_cache_manifest_completion_debt") is not True
        or options.get("seed_board_before_attach") is not False
        or options.get("allow_source_bound_board_global_correction") is not False
        or options.get("source_bound_board_anchor") is not False
        or options.get("source_bound_board_precall_global_restore") is not False
        or any(options.get(name) is not None for name in natural_control_nulls)
        or startup_rng.get("mode") != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != expected_process_id
        or startup_rng.get("observed_seed") != startup_seed
        or startup_rng.get("effective_seed") != startup_seed
        or startup_rng.get("seed_source")
        != "retail_eax_before_push_to_srand"
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("stopped_at_update") is not True
        or result.get("last_update") != stop_after_update
        or isinstance(manifest.get("entry_count"), bool)
        or not isinstance(manifest.get("entry_count"), int)
        or manifest.get("entry_count") <= 0
        or _digest(
            manifest.get("manifest_sha256"),
            "strict replay manifest SHA-256",
        )
        != manifest.get("manifest_sha256")
        or _digest(
            manifest.get("main_pak_sha256"),
            "strict replay main.pak SHA-256",
        )
        != manifest.get("main_pak_sha256")
        or result.get("font_cache_manifest_entry_count")
        != manifest.get("entry_count")
        or "sha256:" + str(result.get("font_cache_manifest_sha256"))
        != manifest.get("manifest_sha256")
        or "sha256:"
        + str(result.get("font_cache_manifest_main_pak_sha256"))
        != manifest.get("main_pak_sha256")
    ):
        raise PcSourceValidationError("strict command replay differs")
    return dict(binding)


def _source_bound_descriptor(
    root: Path,
    value: Any,
    name: str,
) -> Path:
    descriptor = _mapping(value, name)
    _check_keys(
        descriptor,
        required={"path", "bytes", "sha256"},
        name=name,
    )
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PcSourceValidationError(f"{name}.path is invalid")
    try:
        path = Path(path_value).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcSourceValidationError(
            f"{name}.path escapes evidence_root"
        ) from error
    size = descriptor.get("bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
        or _sha256_path(path)
        != _digest(descriptor.get("sha256"), f"{name}.sha256")
    ):
        raise PcSourceValidationError(f"{name} differs")
    return path


def _validate_source_bound_command_replay(
    probe_path: Path,
    *,
    evidence_root: Path,
    expected_process_id: int,
    expected_runtime_path: Path,
    expected_runtime_sha256: str,
    expected_dmo_path: Path,
    expected_dmo_sha256: str,
) -> Mapping[str, Any] | None:
    """Independently revalidate one source-recording-normalized replay."""

    try:
        probe = _mapping(read_canonical_json(probe_path), "memory probe")
    except PcExactStepEvidenceError as error:
        raise PcSourceValidationError(str(error)) from error
    value = probe.get("strict_command_replay")
    if value is None:
        return None
    binding = _mapping(value, "strict_command_replay")
    if binding.get("schema") == STRICT_REPLAY_BINDING_SCHEMA:
        return None
    if binding.get("schema") != SOURCE_BOUND_REPLAY_BINDING_SCHEMA:
        raise PcSourceValidationError(
            "strict replay binding schema is unsupported"
        )
    _check_keys(
        binding,
        required={
            "schema",
            "version",
            "status",
            "classification",
            "artifact",
            "artifact_bytes",
            "artifact_sha256",
            "runtime_process_id",
            "runtime_executable",
            "runtime_executable_sha256",
            "source_dmo",
            "source_dmo_sha256",
            "stop_after_update",
            "failure_count",
            "collector_plan",
            "retail_dmo_provenance",
            "source_recording",
            "normalization",
            "font_cache_manifest",
            "transport",
        },
        name="source-bound strict command replay",
    )
    artifact_name = binding.get("artifact")
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
    ):
        raise PcSourceValidationError(
            "source-bound strict replay artifact path is invalid"
        )
    artifact_path = (probe_path.parent / artifact_name).resolve()
    try:
        artifact_path.relative_to(probe_path.parent.resolve())
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise PcSourceValidationError(
            "source-bound strict replay artifact is invalid"
        ) from error
    payload = _mapping(payload, "source-bound strict replay artifact")
    options = _mapping(
        payload.get("options"),
        "source-bound strict replay options",
    )
    result = _mapping(
        payload.get("result"),
        "source-bound strict replay result",
    )
    startup_rng = _mapping(
        payload.get("startup_rng"),
        "source-bound strict replay startup_rng",
    )
    payload_dmo = _mapping(
        payload.get("source_dmo"),
        "source-bound strict replay source_dmo",
    )
    source_receipt = _mapping(
        payload.get("source_bound_board_anchor"),
        "source-bound board receipt",
    )
    board_rng = _mapping(
        payload.get("board_rng"),
        "source-bound board RNG receipt",
    )

    root = evidence_root.resolve(strict=True)
    plan_path = _source_bound_descriptor(
        root,
        binding.get("collector_plan"),
        "source-bound collector_plan",
    )
    provenance_path = _source_bound_descriptor(
        root,
        binding.get("retail_dmo_provenance"),
        "source-bound retail_dmo_provenance",
    )
    source_recording = _mapping(
        binding.get("source_recording"),
        "source-bound source_recording",
    )
    _check_keys(
        source_recording,
        required={
            "report",
            "raw_dmo",
            "rng_monitor",
            "global_mtrand_trace",
            "process_id",
            "outcome",
            "process_memory_writes",
            "normal_exit",
            "host_restored_exactly",
        },
        name="source-bound source_recording",
    )
    recording_report_path = _source_bound_descriptor(
        root,
        source_recording.get("report"),
        "source-bound recording report",
    )
    raw_dmo_path = _source_bound_descriptor(
        root,
        source_recording.get("raw_dmo"),
        "source-bound raw DMO",
    )
    monitor_path = _source_bound_descriptor(
        root,
        source_recording.get("rng_monitor"),
        "source-bound RNG monitor",
    )
    global_trace_path = _source_bound_descriptor(
        root,
        source_recording.get("global_mtrand_trace"),
        "source-bound global MTRand trace",
    )
    try:
        plan = _mapping(
            json.loads(plan_path.read_text(encoding="utf-8")),
            "source-bound collector plan",
        )
        recording_report = _mapping(
            json.loads(recording_report_path.read_text(encoding="utf-8")),
            "source-bound recording report",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PcSourceValidationError(
            "source-bound provenance JSON is invalid"
        ) from error
    runtime = _mapping(plan.get("runtime"), "source-bound plan runtime")
    trace = _mapping(plan.get("trace"), "source-bound plan trace")
    plan_dmo = _mapping(plan.get("dmo"), "source-bound plan DMO")
    anchor = _mapping(
        trace.get("source_bound_board_anchor"),
        "source-bound plan anchor",
    )
    precall = _mapping(
        trace.get("source_bound_board_precall_global_restore"),
        "source-bound plan precall",
    )
    normalization = _mapping(
        binding.get("normalization"),
        "source-bound normalization",
    )
    _check_keys(
        normalization,
        required={
            "startup_seed",
            "startup_seed_transport",
            "board_seed",
            "board_seed_call_address",
            "source_bound_framework_update",
            "global_precall_state_sha256",
            "global_precall_memory_write_bytes",
            "global_postcall_memory_write_bytes",
            "thread_crt_state",
            "thread_crt_memory_write_bytes",
            "gameplay_or_rng_process_memory_write_count",
            "gameplay_or_rng_process_memory_write_bytes",
            "last_mutation_update",
            "persistent_file_modified",
        },
        name="source-bound normalization",
    )
    font_manifest = _mapping(
        binding.get("font_cache_manifest"),
        "source-bound font_cache_manifest",
    )
    _check_keys(
        font_manifest,
        required={"entry_count", "manifest_sha256", "main_pak_sha256"},
        name="source-bound font_cache_manifest",
    )
    transport = _mapping(
        binding.get("transport"),
        "source-bound transport",
    )
    _check_keys(
        transport,
        required={
            "service_broker",
            "startup_worker_yield_count",
            "temporary_code_breakpoint_memory_write_count",
            "callback_context_emulation_count",
            "post_board_construction_process_memory_write_count",
        },
        name="source-bound transport",
    )

    stop_after_update = binding.get("stop_after_update")
    probe_update = probe.get("framework_update")
    try:
        bound_runtime = Path(str(binding.get("runtime_executable"))).resolve(
            strict=True
        )
        payload_runtime = Path(str(payload.get("runtime_executable"))).resolve(
            strict=True
        )
        bound_dmo = Path(str(binding.get("source_dmo"))).resolve(strict=True)
        trace_dmo = Path(str(payload_dmo.get("path"))).resolve(strict=True)
        plan_runtime = Path(str(runtime.get("runtime_executable"))).resolve(
            strict=True
        )
        provenance_declared = Path(
            str(trace.get("source_bound_dmo_provenance_path"))
        ).resolve(strict=True)
    except OSError as error:
        raise PcSourceValidationError(
            "source-bound replay identity is unavailable"
        ) from error
    binding_version = binding.get("version")
    expected_binding_classification = {
        SOURCE_BOUND_REPLAY_BINDING_VERSION: (
            "source_bound_original_natural_retail_recording"
        ),
        OUTCOME_SOURCE_BOUND_REPLAY_BINDING_VERSION: (
            "source_bound_original_natural_retail_outcome"
        ),
    }.get(binding_version)
    digest = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    if (
        expected_binding_classification is None
        or binding.get("status") != "PASS"
        or binding.get("classification")
        != expected_binding_classification
        or binding.get("artifact_bytes") != len(artifact_bytes)
        or binding.get("artifact_sha256") != digest
        or binding.get("runtime_process_id") != expected_process_id
        or payload.get("runtime_process_id") != expected_process_id
        or bound_runtime != expected_runtime_path.resolve()
        or payload_runtime != expected_runtime_path.resolve()
        or plan_runtime != expected_runtime_path.resolve()
        or binding.get("runtime_executable_sha256")
        != expected_runtime_sha256
        or runtime.get("runtime_executable_sha256")
        != expected_runtime_sha256
        or runtime.get("expected_runtime_sha256")
        != expected_runtime_sha256
        or bound_dmo != expected_dmo_path.resolve()
        or trace_dmo != expected_dmo_path.resolve()
        or binding.get("source_dmo_sha256") != expected_dmo_sha256
        or payload_dmo.get("sha256")
        != expected_dmo_sha256.removeprefix("sha256:")
        or payload_dmo.get("size_bytes") != expected_dmo_path.stat().st_size
        or plan_dmo.get("sha256") != expected_dmo_sha256
        or plan_dmo.get("bytes") != expected_dmo_path.stat().st_size
        or provenance_declared != provenance_path
        or plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan"
        or plan.get("version") != 5
        or runtime.get("launch_mode") != "direct_byte_identical_fixed_seed"
        or isinstance(stop_after_update, bool)
        or not isinstance(stop_after_update, int)
        or stop_after_update <= 0
        or trace.get("source_bound_command_broker_stop_after_update")
        != stop_after_update
        or trace.get("attach_at_update") != 0
        or trace.get("detach_at_update") is not None
        or trace.get("reattach_at_update") is not None
        or (
            probe_update is not None
            and (
                isinstance(probe_update, bool)
                or not isinstance(probe_update, int)
                or stop_after_update >= probe_update
            )
        )
        or binding.get("failure_count") != 0
    ):
        raise PcSourceValidationError("source-bound command replay differs")

    try:
        provenance_summary = read_certifying_provenance(
            source_path=raw_dmo_path,
            output_path=expected_dmo_path,
            recording_report_path=recording_report_path,
            collector_plan_path=plan_path,
            provenance_path=provenance_path,
            expected_runtime_sha256=expected_runtime_sha256,
        )
    except RetailDmoProvenanceError as error:
        raise PcSourceValidationError(
            "source-bound retail DMO provenance differs"
        ) from error
    try:
        recording_outcome = certifying_recording_outcome(
            recording_report
        )
    except RetailDmoProvenanceError as error:
        raise PcSourceValidationError(
            "source-bound retail DMO provenance differs"
        ) from error
    raw_dmo_sha256 = _sha256_path(raw_dmo_path)
    if (
        provenance_summary.get("source_dmo_sha256") != raw_dmo_sha256
        or provenance_summary.get("playback_dmo_sha256")
        != expected_dmo_sha256
        or provenance_summary.get("gameplay_input_commands_preserved")
        is not True
        or provenance_summary.get("recording_process_memory_writes") != 0
        or provenance_summary.get("recording_natural_outcome") is not True
        or provenance_summary.get("recording_outcome")
        != recording_outcome
        or provenance_summary.get("recording_normal_retail_exit") is not True
        or provenance_summary.get("recording_host_restored_exactly") is not True
    ):
        raise PcSourceValidationError(
            "source-bound retail DMO provenance differs"
        )

    gameplay = _mapping(
        recording_report.get("gameplay"),
        "source-bound recording gameplay",
    )
    monitor = _mapping(
        recording_report.get("rng_monitor"),
        "source-bound recording RNG monitor",
    )
    global_trace = _mapping(
        recording_report.get("gameplay_mtrand_trace"),
        "source-bound recording MTRand trace",
    )
    main_thread = _mapping(
        recording_report.get("main_thread"),
        "source-bound recording main_thread",
    )
    report_descriptor = _mapping(
        source_recording.get("report"),
        "source-bound recording report descriptor",
    )
    raw_descriptor = _mapping(
        source_recording.get("raw_dmo"),
        "source-bound raw DMO descriptor",
    )
    monitor_descriptor = _mapping(
        source_recording.get("rng_monitor"),
        "source-bound RNG monitor descriptor",
    )
    trace_descriptor = _mapping(
        source_recording.get("global_mtrand_trace"),
        "source-bound MTRand trace descriptor",
    )
    if (
        source_recording.get("process_id") != precall.get("source_process_id")
        or source_recording.get("process_id") != recording_report.get("process_id")
        or source_recording.get("outcome") != recording_outcome
        or source_recording.get("process_memory_writes") != 0
        or source_recording.get("normal_exit") is not True
        or source_recording.get("host_restored_exactly") is not True
        or recording_report.get("process_memory_writes") != 0
        or recording_report.get("normal_exit") is not True
        or recording_report.get("host_restored") is not True
        or recording_report.get("host_pre_state_root")
        != recording_report.get("host_restored_state_root")
        or gameplay.get("status") != "PASS"
        or gameplay.get("outcome") != recording_outcome
        or main_thread.get("process_memory_writes") != 0
        or monitor.get("process_memory_writes") != 0
        or monitor.get("output") != str(monitor_path)
        or monitor.get("output_sha256") != monitor_descriptor.get("sha256")
        or global_trace.get("process_memory_writes") != 0
        or global_trace.get("output") != str(global_trace_path)
        or global_trace.get("output_sha256") != trace_descriptor.get("sha256")
        or recording_report.get("dmo") != str(raw_dmo_path)
        or recording_report.get("dmo_sha256") != raw_descriptor.get("sha256")
        or anchor.get("source_dmo_path") != str(raw_dmo_path)
        or anchor.get("source_dmo_sha256") != raw_descriptor.get("sha256")
        or anchor.get("recording_report_path") != str(recording_report_path)
        or anchor.get("recording_report_sha256")
        != report_descriptor.get("sha256")
        or anchor.get("monitor_path") != str(monitor_path)
        or anchor.get("monitor_sha256") != monitor_descriptor.get("sha256")
        or anchor.get("trace_path") != str(global_trace_path)
        or anchor.get("trace_sha256") != trace_descriptor.get("sha256")
    ):
        raise PcSourceValidationError(
            "source-bound recording provenance differs"
        )

    observations = source_receipt.get("observations")
    if (
        not isinstance(observations, list)
        or len(observations) != 1
        or not isinstance(observations[0], Mapping)
    ):
        raise PcSourceValidationError(
            "source-bound board receipt differs"
        )
    source_row = observations[0]
    global_row = _mapping(
        source_row.get("global"),
        "source-bound global post-call receipt",
    )
    crt_row = _mapping(
        source_row.get("thread_crt"),
        "source-bound thread CRT receipt",
    )
    precall_row = _mapping(
        source_row.get("global_precall_restore"),
        "source-bound global pre-call receipt",
    )
    board_observations = board_rng.get("observations")
    applied_rows = (
        [
            row
            for row in board_observations
            if isinstance(row, Mapping)
            and row.get("source_bound_board_anchor_applied") is True
        ]
        if isinstance(board_observations, list)
        else []
    )
    if (
        source_receipt.get("persistent_file_modified") is not False
        or source_receipt.get("process_memory_mutation") is not True
        or source_row.get("classification")
        != "source-bound-natural-retail-post-call-board-anchor"
        or source_row.get("process_id") != expected_process_id
        or source_row.get("framework_update") != anchor.get("framework_update")
        or source_row.get("effective_seed") != runtime.get("board_seed")
        or source_row.get("observed_seed") != runtime.get("board_seed")
        or source_row.get("observed_seed_matches_source") is not True
        or source_row.get("bytes_written") != 4
        or source_row.get("process_memory_mutation") is not True
        or global_row.get("bounded_global_correction_applied") is not False
        or global_row.get("global_correction_authorized") is not False
        or global_row.get("natural_post_state_match") is not True
        or global_row.get("bytes_written") != 0
        or global_row.get("changed") is not False
        or global_row.get("source_call_output") != runtime.get("board_seed")
        or global_row.get("source_call_post_index")
        != anchor.get("expected_post_index")
        or global_row.get("source_call_post_state_sha256")
        != anchor.get("expected_post_state_sha256")
        or precall_row.get("classification")
        != "source-bound-natural-retail-global-precall-restore"
        or precall_row.get("framework_update") != precall.get("framework_update")
        or precall_row.get("changed") is not False
        or precall_row.get("bytes_written") != 0
        or precall_row.get("process_memory_mutation") is not False
        or precall_row.get("restored") != precall.get("global_state") | {
            "source_path": anchor.get("monitor_path"),
            "source_sha256": anchor.get("monitor_sha256"),
            "source_process_id": precall.get("source_process_id"),
            "source_framework_update": anchor.get("monitor_framework_update"),
            "source_native_game_time": precall.get("source_native_game_time"),
            "source_kind": "natural_retail_hardware_trace_and_monitor",
            "source_trace_path": anchor.get("trace_path"),
            "source_trace_sha256": anchor.get("trace_sha256"),
            "source_recording_report_path": anchor.get("recording_report_path"),
            "source_recording_report_sha256": anchor.get("recording_report_sha256"),
            "source_dmo_path": anchor.get("source_dmo_path"),
            "source_dmo_sha256": anchor.get("source_dmo_sha256"),
            "source_call_order": anchor.get("source_order"),
            "source_caller": anchor.get("caller"),
            "source_caller_hex": f"0x{int(anchor.get('caller')):08x}",
        }
        or crt_row.get("restored_state") != anchor.get("thread_crt_state")
        or crt_row.get("bytes_written") != 4
        or crt_row.get("changed") is not True
        or len(applied_rows) != 1
        or applied_rows[0].get("effective_seed") != runtime.get("board_seed")
        or applied_rows[0].get("source_bound_global_post_state_match") is not True
    ):
        raise PcSourceValidationError(
            "source-bound board receipt differs"
        )

    expected_normalization = {
        "startup_seed": runtime.get("crt_rand_seed"),
        "startup_seed_transport": "debugger_register",
        "board_seed": runtime.get("board_seed"),
        "board_seed_call_address": runtime.get("board_seed_call_address"),
        "source_bound_framework_update": anchor.get("framework_update"),
        "global_precall_state_sha256": precall.get("global_state", {}).get(
            "state_sha256"
        ),
        "global_precall_memory_write_bytes": 0,
        "global_postcall_memory_write_bytes": 0,
        "thread_crt_state": anchor.get("thread_crt_state"),
        "thread_crt_memory_write_bytes": 4,
        "gameplay_or_rng_process_memory_write_count": 1,
        "gameplay_or_rng_process_memory_write_bytes": 4,
        "last_mutation_update": anchor.get("framework_update"),
        "persistent_file_modified": False,
    }
    trace_manifest = _mapping(
        trace.get("font_cache_manifest_receipt"),
        "source-bound plan font manifest",
    )
    if (
        dict(normalization) != expected_normalization
        or normalization.get("last_mutation_update") >= stop_after_update
        or not _font_manifest_binding_matches(
            font_manifest,
            trace_manifest,
        )
        or binding.get("failure_count") != 0
        or options.get("direct_runtime") is not True
        or options.get("direct_natural_seed", False) is not False
        or options.get("broker_service_blocks") is not True
        or options.get("allow_pre_stream_commands") is not True
        or options.get("startup_trace_handoff") is not True
        or options.get("allow_font_cache_manifest_completion_debt") is not True
        or options.get("attach_at_update") != 0
        or options.get("detach_at_update") is not None
        or options.get("reattach_at_update") is not None
        or options.get("stop_after_update") != stop_after_update
        or options.get("stop_after_command_order") is not None
        or options.get("crt_rand_seed") != runtime.get("crt_rand_seed")
        or options.get("startup_seed_transport") != "debugger_register"
        or options.get("board_seed") != runtime.get("board_seed")
        or options.get("board_seed_address")
        != runtime.get("board_seed_call_address")
        or options.get("global_rng_seed") is not None
        or options.get("thread_crt_rng_seed") is not None
        or options.get("source_bound_board_anchor") is not True
        or options.get("source_bound_board_precall_global_restore") is not True
        or options.get("allow_source_bound_board_global_correction") is not False
        or startup_rng.get("process_id") != expected_process_id
        or startup_rng.get("seed") != runtime.get("crt_rand_seed")
        or startup_rng.get("register_override")
        != "eax_before_push_to_srand"
        or startup_rng.get("persistent_file_modified") is not False
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("stopped_at_update") is not True
        or result.get("stopped_at_command_order") is not False
        or result.get("last_update") != stop_after_update
        or result.get("font_cache_manifest_entry_count")
        != font_manifest.get("entry_count")
        or "sha256:" + str(result.get("font_cache_manifest_sha256"))
        != font_manifest.get("manifest_sha256")
        or "sha256:"
        + str(result.get("font_cache_manifest_main_pak_sha256"))
        != font_manifest.get("main_pak_sha256")
    ):
        raise PcSourceValidationError("source-bound command replay differs")

    worker_yields = result.get("startup_post_bypass_worker_yields")
    if not isinstance(worker_yields, list):
        raise PcSourceValidationError(
            "source-bound worker receipts are invalid"
        )
    temporary_writes = 0
    for receipt in worker_yields:
        row = _mapping(receipt, "source-bound worker receipt")
        write_count = row.get("temporary_breakpoint_memory_writes")
        if (
            row.get("process_id") != expected_process_id
            or row.get("memory_writes") != 0
            or row.get("failure") is not None
            or row.get("command_breakpoint_rearmed") is not True
            or isinstance(write_count, bool)
            or not isinstance(write_count, int)
            or write_count <= 0
        ):
            raise PcSourceValidationError(
                "source-bound worker receipts are invalid"
            )
        temporary_writes += write_count
    callback_count = 0
    for name in (
        "deferred_file_write_discharges",
        "font_cache_manifest_completion_discharges",
        "terminal_file_write_payload_handoffs",
        "service_file_write_header_claims",
    ):
        rows = result.get(name)
        if not isinstance(rows, list):
            raise PcSourceValidationError(
                "source-bound callback receipts are invalid"
            )
        callback_count += len(rows)
    if dict(transport) != {
        "service_broker": True,
        "startup_worker_yield_count": len(worker_yields),
        "temporary_code_breakpoint_memory_write_count": temporary_writes,
        "callback_context_emulation_count": callback_count,
        "post_board_construction_process_memory_write_count": 0,
    }:
        raise PcSourceValidationError(
            "source-bound transport receipts differ"
        )
    return dict(binding)


def _font_manifest_binding_matches(
    binding: Mapping[str, Any],
    plan_manifest: Mapping[str, Any],
) -> bool:
    """Compare the compact replay binding with its richer plan receipt."""

    return all(
        binding.get(name) == plan_manifest.get(name)
        for name in (
            "entry_count",
            "manifest_sha256",
            "main_pak_sha256",
        )
    )


def _actor_visible_first_frame(
    frames: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Project the first exact source frame to actor-visible shooter state."""

    if not frames:
        return None
    frame = frames[0]
    current = getattr(frame, "current_ball", None)
    following = getattr(frame, "next_ball", None)
    required = (
        "update",
        "score",
        "displayed_score",
        "chain_ball_count",
        "inserting_ball_count",
        "fired_bullet_count",
        "board_color_counts",
    )
    if (
        current is None
        or following is None
        or any(not hasattr(frame, name) for name in required)
        or not hasattr(current, "ball_id")
        or not hasattr(current, "color_id")
        or not hasattr(following, "ball_id")
        or not hasattr(following, "color_id")
    ):
        return None
    values = {
        "framework_update": getattr(frame, "update"),
        "score": getattr(frame, "score"),
        "displayed_score": getattr(frame, "displayed_score"),
        "chain_ball_count": getattr(frame, "chain_ball_count"),
        "inserting_ball_count": getattr(frame, "inserting_ball_count"),
        "fired_bullet_count": getattr(frame, "fired_bullet_count"),
        "current_ball_id": getattr(current, "ball_id"),
        "current_color_id": getattr(current, "color_id"),
        "next_ball_id": getattr(following, "ball_id"),
        "next_color_id": getattr(following, "color_id"),
        "board_color_counts": list(getattr(frame, "board_color_counts")),
    }
    integer_fields = tuple(values)[:-1]
    if any(
        isinstance(values[name], bool)
        or not isinstance(values[name], int)
        or values[name] < 0
        for name in integer_fields
    ):
        return None
    counts = values["board_color_counts"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in counts
    ):
        return None
    return values


def verify_pc_source_manifest(
    manifest_path: str | Path,
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
) -> dict[str, Any]:
    """Recompute one source manifest without comparing it to another run."""

    if original_root is None:
        raise PcSourceValidationError("original_root is required")
    root = Path(evidence_root)
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise PcSourceValidationError("evidence_root does not exist") from error
    if not root.is_dir():
        raise PcSourceValidationError("evidence_root must be a directory")

    source = Path(manifest_path).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise PcSourceValidationError("source manifest escapes evidence_root") from error
    try:
        manifest = _mapping(read_canonical_json(source), "source manifest")
    except PcExactStepEvidenceError as error:
        raise PcSourceValidationError(str(error)) from error
    _check_keys(
        manifest,
        required={
            "schema",
            "version",
            "source_id",
            "scope",
            "original_executable_sha256",
            "runtime_payload",
            "dmo",
            "run",
            "window",
            "state_transaction",
        },
        name="source manifest",
    )
    source_version = manifest["version"]
    if (
        manifest["schema"] != SOURCE_SCHEMA
        or source_version
        not in {
            SOURCE_VERSION,
            FULL_STATE_SOURCE_VERSION,
            SOURCE_BOUND_FULL_STATE_SOURCE_VERSION,
        }
    ):
        raise PcSourceValidationError("unsupported source manifest schema/version")
    full_state_source = source_version in {
        FULL_STATE_SOURCE_VERSION,
        SOURCE_BOUND_FULL_STATE_SOURCE_VERSION,
    }

    source_id = _identifier(manifest["source_id"], "source_id")
    scope = _mapping(manifest["scope"], "scope")
    _check_keys(
        scope,
        required={"level_id", "hard", "profile_mode", "mode"},
        name="scope",
    )
    if (
        not isinstance(scope["level_id"], str)
        or not scope["level_id"]
        or not isinstance(scope["hard"], bool)
        or not isinstance(scope["profile_mode"], str)
        or scope["mode"] != "adventure"
    ):
        raise PcSourceValidationError("source scope is invalid")

    original_executable_sha256 = _digest(
        manifest["original_executable_sha256"],
        "original_executable_sha256",
    )
    original_executable = Path(original_root).resolve(strict=True) / "ZumasRevenge.exe"
    if not original_executable.is_file() or _sha256_path(original_executable) != original_executable_sha256:
        raise PcSourceValidationError("original executable identity differs")

    runtime_payload = _bound_path(
        root,
        _mapping(manifest["runtime_payload"], "runtime_payload"),
        "runtime_payload",
    )
    dmo_path = _bound_path(root, _mapping(manifest["dmo"], "dmo"), "dmo")
    runtime_sha256 = _sha256_path(runtime_payload)
    dmo_sha256 = _sha256_path(dmo_path)
    try:
        demo = PopCapDemo.read(dmo_path)
    except (OSError, PopCapDemoError) as error:
        raise PcSourceValidationError("bound DMO is invalid") from error

    run = _mapping(manifest["run"], "run")
    run_fields = {
            "run_id",
            "selected_attempt",
            "attempts",
            "memory_probe",
            "trajectory_index",
            "process_id",
            "process_creation_filetime_100ns",
            "maximum_startup_attempts",
    }
    if full_state_source:
        run_fields.add("source_kind")
    _check_keys(
        run,
        required=run_fields,
        name="run",
    )
    expected_source_kind = (
        SOURCE_BOUND_FULL_STATE_SOURCE_KIND
        if source_version == SOURCE_BOUND_FULL_STATE_SOURCE_VERSION
        else FULL_STATE_SOURCE_KIND
    )
    if full_state_source and run.get("source_kind") != expected_source_kind:
        raise PcSourceValidationError("full-state source kind is invalid")
    run_id = _identifier(run["run_id"], "run.run_id")
    attempts_path = _bound_path(root, _mapping(run["attempts"], "run.attempts"), "run.attempts")
    probe_path = _bound_path(root, _mapping(run["memory_probe"], "run.memory_probe"), "run.memory_probe")
    index_path = _bound_path(root, _mapping(run["trajectory_index"], "run.trajectory_index"), "run.trajectory_index")

    window = _mapping(manifest["window"], "window")
    _check_keys(
        window,
        required={"freeze_update", "start_update", "end_update", "warmup_tick_count"},
        name="window",
    )
    freeze_update = _integer(window["freeze_update"], "window.freeze_update")
    start_update = _integer(window["start_update"], "window.start_update")
    end_update = _integer(window["end_update"], "window.end_update")
    warmup_tick_count = _integer(window["warmup_tick_count"], "window.warmup_tick_count")
    exact_window_valid = freeze_update < start_update <= end_update
    full_state_window_valid = (
        freeze_update == start_update <= end_update
        and warmup_tick_count == 0
    )
    if (
        source_version == SOURCE_VERSION
        and not exact_window_valid
    ) or (
        full_state_source
        and not full_state_window_valid
    ):
        raise PcSourceValidationError("source window ordering is invalid")

    try:
        loader = (
            load_formal_exact_step_run
            if source_version == SOURCE_VERSION
            else load_formal_full_state_run
        )
        loaded = loader(
                run_id=run_id,
                selected_attempt=_integer(
                    run["selected_attempt"],
                    "run.selected_attempt",
                    minimum=1,
                ),
                attempts_path=attempts_path,
                probe_path=probe_path,
                index_path=index_path,
                expected_process_id=_integer(
                    run["process_id"],
                    "run.process_id",
                    minimum=1,
                ),
                expected_process_creation_filetime_100ns=_integer(
                    run["process_creation_filetime_100ns"],
                    "run.process_creation_filetime_100ns",
                    minimum=1,
                ),
                expected_runtime_sha256=runtime_sha256,
                expected_dmo_sha256=dmo_sha256,
                expected_freeze_update=freeze_update,
                expected_start_update=start_update,
                expected_end_update=end_update,
                expected_warmup_tick_count=warmup_tick_count,
                maximum_startup_attempts=_integer(
                    run["maximum_startup_attempts"],
                    "run.maximum_startup_attempts",
                    minimum=1,
                ),
            )
    except (PcExactStepEvidenceError, PcFullStateEvidenceError) as error:
        raise PcSourceValidationError(str(error)) from error

    strict_command_replay = _validate_strict_command_replay(
        probe_path,
        expected_process_id=loaded.process_id,
        expected_runtime_path=runtime_payload,
        expected_runtime_sha256=runtime_sha256,
        expected_dmo_path=dmo_path,
        expected_dmo_sha256=dmo_sha256,
    )
    source_bound_command_replay = _validate_source_bound_command_replay(
        probe_path,
        evidence_root=root,
        expected_process_id=loaded.process_id,
        expected_runtime_path=runtime_payload,
        expected_runtime_sha256=runtime_sha256,
        expected_dmo_path=dmo_path,
        expected_dmo_sha256=dmo_sha256,
    )
    if source_version == FULL_STATE_SOURCE_VERSION and (
        strict_command_replay is None
        or source_bound_command_replay is not None
    ):
        raise PcSourceValidationError(
            "full-state source requires strict command replay"
        )
    if source_version == SOURCE_BOUND_FULL_STATE_SOURCE_VERSION and (
        source_bound_command_replay is None
        or strict_command_replay is not None
    ):
        raise PcSourceValidationError(
            "source-bound full-state source requires source-bound replay"
        )
    replay_binding = (
        source_bound_command_replay
        if source_bound_command_replay is not None
        else strict_command_replay
    )

    # Rebind the derived runtime payload to the user's installed retail EXE.
    try:
        index = _mapping(read_canonical_json(index_path), "trajectory index")
    except PcExactStepEvidenceError as error:
        raise PcSourceValidationError(str(error)) from error
    process_identity = _mapping(index.get("process_identity"), "process_identity")
    provenance = _mapping(
        process_identity.get("executable_hash_provenance"),
        "executable_hash_provenance",
    )
    if (
        process_identity.get("executable_sha256") != runtime_sha256
        or provenance.get("payload_sha256") != runtime_sha256
        or provenance.get("source_sha256") != original_executable_sha256
        or provenance.get("runtime_file_direct_hash_verified") is not True
    ):
        raise PcSourceValidationError("runtime payload provenance differs")

    transaction = _mapping(manifest["state_transaction"], "state_transaction")
    _check_keys(
        transaction,
        required={"pre_snapshot", "post_snapshot"},
        name="state_transaction",
    )
    pre_path = _bound_path(
        root,
        _mapping(transaction["pre_snapshot"], "state_transaction.pre_snapshot"),
        "state_transaction.pre_snapshot",
    )
    post_path = _bound_path(
        root,
        _mapping(transaction["post_snapshot"], "state_transaction.post_snapshot"),
        "state_transaction.post_snapshot",
    )
    try:
        pre = PcStateSnapshot.read(pre_path)
        post = PcStateSnapshot.read(post_path)
    except (OSError, UnicodeError, PcGoldenValidationError) as error:
        raise PcSourceValidationError("state transaction snapshot is invalid") from error
    if (
        pre.session_nonce != post.session_nonce
        or pre.state_root != post.state_root
        or pre.captured_perf_counter_ns >= loaded.campaign_started_perf_counter_ns
        or post.captured_perf_counter_ns <= loaded.attempt_finished_perf_counter_ns
    ):
        raise PcSourceValidationError("state transaction was not restored exactly")

    tick_count = len(loaded.frames)
    features = ["source_authenticity"]
    if tick_count >= 500:
        features.append("long_horizon_drift")
    fingerprint = _source_fingerprint(
        run_id=run_id,
        process_creation_filetime_100ns=loaded.process_creation_filetime_100ns,
        dmo_sha256=dmo_sha256,
        runtime_sha256=runtime_sha256,
        probe_sha256=loaded.probe_sha256,
        index_sha256=loaded.index_sha256,
    )
    return {
        "schema": SOURCE_REPORT_SCHEMA,
        "version": SOURCE_REPORT_VERSION,
        "status": "PASS",
        "source_id": source_id,
        "scope": dict(scope),
        "random_seed": int(demo.random_seed),
        "tick_count": tick_count,
        # Expose the independently recomputed transport identities to the
        # suite verifier.  A successful strict retail replay proves that the
        # process consumed these bytes; it does not, by itself, prove that the
        # DMO originated from an unedited retail recording.  The Fidelity Gate
        # performs that final same-suite provenance join against a certifying
        # PC Golden case.
        "dmo_sha256": dmo_sha256,
        "runtime_sha256": runtime_sha256,
        "source_fingerprint": fingerprint,
        "authorized_features": features,
        "independent_process_identity": {
            "process_id": loaded.process_id,
            "process_creation_filetime_100ns": loaded.process_creation_filetime_100ns,
        },
        "state_root": pre.state_root,
        "transport": (
            "single_retail_process_source_bound_full_state_exact_step_source"
            if source_version == SOURCE_BOUND_FULL_STATE_SOURCE_VERSION
            else (
                "single_retail_process_full_state_exact_step_source"
                if source_version == FULL_STATE_SOURCE_VERSION
                else "single_retail_process_exact_step_source"
            )
        ),
        "cross_process_exact_replay_required": False,
        "strict_command_replay": replay_binding,
        "actor_visible_first_frame": _actor_visible_first_frame(
            loaded.frames
        ),
    }


def write_pc_source_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a canonical manifest without overwriting an existing file."""

    target = Path(path)
    data = canonical_json_bytes(dict(payload))
    try:
        with target.open("xb") as output:
            output.write(data)
    except OSError as error:
        raise PcSourceValidationError("source manifest could not be written exclusively") from error


__all__ = [
    "PcSourceValidationError",
    "SOURCE_REPORT_SCHEMA",
    "SOURCE_REPORT_VERSION",
    "SOURCE_SCHEMA",
    "SOURCE_VERSION",
    "FULL_STATE_SOURCE_KIND",
    "FULL_STATE_SOURCE_VERSION",
    "SOURCE_BOUND_FULL_STATE_SOURCE_KIND",
    "SOURCE_BOUND_FULL_STATE_SOURCE_VERSION",
    "STRICT_REPLAY_BINDING_SCHEMA",
    "STRICT_REPLAY_BINDING_VERSION",
    "SOURCE_BOUND_REPLAY_BINDING_SCHEMA",
    "SOURCE_BOUND_REPLAY_BINDING_VERSION",
    "OUTCOME_SOURCE_BOUND_REPLAY_BINDING_VERSION",
    "verify_pc_source_manifest",
    "write_pc_source_manifest",
]
