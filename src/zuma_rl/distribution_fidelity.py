"""Recomputable startup-distribution evidence for retail transfer.

The datasets in this lane contain identities, not claimed outcomes.  Retail
colour pairs are re-extracted from authenticated exact-step source manifests,
and simulator colour pairs are rerun from a code-frozen seed schedule.  Each
retail slot uses a separately derived DMO whose only change is the four-byte
framework gameplay seed in the DMO header.  Replaying one DMO in many natural
processes is deliberately rejected: PopCap restores gameplay MTRand from the
DMO, so varying only the unrelated startup CRT seed does not sample the actor
distribution.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from zuma_rl.pc_exact_step_evidence import (
    PcExactStepEvidenceError,
    read_canonical_json,
)
from zuma_rl.pc_source import (
    SOURCE_SCHEMA,
    SOURCE_VERSION,
    STRICT_REPLAY_BINDING_SCHEMA,
    STRICT_REPLAY_BINDING_VERSION,
    PcSourceValidationError,
    verify_pc_source_manifest,
)
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError
from zuma_rl.revenge_core import RevengeSimulator

PREREGISTRATION_SCHEMA = "zuma-rl.distribution-fidelity-preregistration"
LEGACY_PREREGISTRATION_VERSION = 3
PREREGISTRATION_VERSION = 4
DATASET_SCHEMA = "zuma-rl.distribution-fidelity-samples"
DATASET_VERSION = 3
AUDIT_SCHEMA = "zuma-rl.pc-simulator-distribution-audit"
AUDIT_VERSION = 3
PREREGISTRATION_REPORT_SCHEMA = (
    "zuma-rl.distribution-fidelity-preregistration-verification"
)
PREREGISTRATION_REPORT_VERSION = 1

METRIC_NAME = "startup_shooter_color_pair"
MINIMUM_PC_SAMPLES = 32
MINIMUM_SIMULATOR_SAMPLES = 256
NUM_COLORS = 4
MAXIMUM_TOTAL_VARIATION = 0.35
MAXIMUM_EXCESS_TOTAL_VARIATION = 0.10
MINIMUM_COMPATIBILITY_P_VALUE = 0.05
PERMUTATION_COUNT = 1_999
PERMUTATION_SEED = 1_515_015_442

# The retail point is the first retained, actor-visible exact-step frame in
# the frozen Jungle2 replay.  A one-tick simulator reset reaches the matching
# first loaded-shooter observation without an agent action.
PC_FREEZE_UPDATE = 3_425
PC_SAMPLE_UPDATE = 3_429
PC_WARMUP_TICK_COUNT = 4
PC_EXPECTED_SCORE = 7_950
SIMULATOR_SAMPLE_TICK = 1

SEED_SCHEDULE_ID = "startup-actor-distribution-v3-fixed-20260808"
SEED_SCHEDULE_ALGORITHM = "sha256-domain-ordinal-le32-rejection-v1"
SEED_SCHEDULE_ENCODING = "little_endian_uint32_sequence"
MECHANISM_PROBE_SEED_ROLE = "mechanism-probe-excluded-from-formal-samples"
GAMEPLAY_SEED_TRANSPORT = "dmo_header_random_seed_only"
DMO_RANDOM_SEED_OFFSET = 8
DMO_RANDOM_SEED_BYTES = 4
SEEDED_DMO_PROVENANCE_SCHEMA = "zuma-rl.seeded-dmo-provenance"
SEEDED_DMO_PROVENANCE_VERSION = 1
STRICT_NATURAL_REPLAY_BINDING_SCHEMA = STRICT_REPLAY_BINDING_SCHEMA
STRICT_NATURAL_REPLAY_BINDING_VERSION = STRICT_REPLAY_BINDING_VERSION

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_UTC_PATTERN = re.compile(
    r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:"
    r"[0-5][0-9]Z$"
)


class DistributionFidelityError(ValueError):
    """The audit, preregistration, or raw identity dataset is invalid."""


def _derive_seed_population(
    population: str,
    count: int,
    *,
    seen: set[int],
) -> tuple[int, ...]:
    seeds: list[int] = []
    for ordinal in range(count):
        rejection = 0
        while True:
            material = (
                f"{SEED_SCHEDULE_ID}\0{population}\0{ordinal}\0{rejection}"
            ).encode("ascii")
            seed = int.from_bytes(
                hashlib.sha256(material).digest()[:4],
                "little",
            )
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
                break
            rejection += 1
    return tuple(seeds)


def frozen_seed_schedules() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the immutable, disjoint PC and simulator gameplay seeds."""

    seen: set[int] = set()
    pc = _derive_seed_population("pc", MINIMUM_PC_SAMPLES, seen=seen)
    simulator = _derive_seed_population(
        "simulator",
        MINIMUM_SIMULATOR_SAMPLES,
        seen=seen,
    )
    return pc, simulator


def mechanism_probe_seed() -> int:
    """Return one deterministic seed excluded from both formal populations."""

    pc, simulator = frozen_seed_schedules()
    seen = set(pc) | set(simulator)
    return _derive_seed_population("mechanism-probe", 1, seen=seen)[0]


def seed_sequence_sha256(seeds: Sequence[int]) -> str:
    """Hash one ordered uint32 seed schedule without JSON ambiguity."""

    payload = bytearray()
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise DistributionFidelityError("seed schedule contains a non-uint32")
        payload.extend(struct.pack("<I", seed))
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DistributionFidelityError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DistributionFidelityError(f"{name} must be an array")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    if set(value) != required:
        raise DistributionFidelityError(f"{name} fields are invalid")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DistributionFidelityError(
            f"{name} must be an integer at least {minimum}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistributionFidelityError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise DistributionFidelityError(f"{name} must be finite")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise DistributionFidelityError(f"{name} is not a valid identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DistributionFidelityError(f"{name} must be a lowercase SHA-256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DistributionFidelityError(
            "a distribution artifact cannot be read"
        ) from error
    return "sha256:" + digest.hexdigest()


def _relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DistributionFidelityError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DistributionFidelityError(f"{name} escapes evidence_root")
    return path


def _bound_path(root: Path, value: Any, name: str) -> Path:
    binding = _mapping(value, name)
    _check_keys(binding, required={"path", "sha256"}, name=name)
    relative = _relative_path(binding["path"], f"{name}.path")
    try:
        path = root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise DistributionFidelityError(f"{name} escapes evidence_root") from error
    if not path.is_file() or _sha256_path(path) != _digest(
        binding["sha256"], f"{name}.sha256"
    ):
        raise DistributionFidelityError(f"{name} identity differs")
    return path


def _validate_seeded_dmo_provenance(
    *,
    root: Path,
    base_binding: Mapping[str, Any],
    derived_binding: Mapping[str, Any],
    provenance_binding: Mapping[str, Any],
    expected_seed: int,
) -> None:
    """Prove one retail replay differs only at the DMO gameplay seed."""

    base_path = _bound_path(root, base_binding, "base DMO")
    derived_path = _bound_path(root, derived_binding, "seeded DMO")
    provenance_path = _bound_path(
        root,
        provenance_binding,
        "seeded DMO provenance",
    )
    provenance = _read_canonical(provenance_path, "seeded DMO provenance")
    _check_keys(
        provenance,
        required={
            "schema",
            "version",
            "status",
            "base_dmo",
            "derived_dmo",
            "random_seed",
            "seed_offset",
            "seed_size_bytes",
            "changed_byte_offsets",
            "unchanged_byte_count",
        },
        name="seeded DMO provenance",
    )
    try:
        base_bytes = base_path.read_bytes()
        derived_bytes = derived_path.read_bytes()
        demo = PopCapDemo.from_bytes(derived_bytes)
    except (OSError, PopCapDemoError) as error:
        raise DistributionFidelityError("seeded DMO is invalid") from error
    if len(base_bytes) < DMO_RANDOM_SEED_OFFSET + DMO_RANDOM_SEED_BYTES:
        raise DistributionFidelityError("base DMO is too short for its seed")
    expected = bytearray(base_bytes)
    struct.pack_into("<I", expected, DMO_RANDOM_SEED_OFFSET, expected_seed)
    changed_offsets = [
        index
        for index, (before, after) in enumerate(
            zip(base_bytes, derived_bytes, strict=True)
        )
        if before != after
    ] if len(base_bytes) == len(derived_bytes) else []
    if (
        provenance["schema"] != SEEDED_DMO_PROVENANCE_SCHEMA
        or provenance["version"] != SEEDED_DMO_PROVENANCE_VERSION
        or provenance["status"] != "PASS"
        or provenance["base_dmo"] != dict(base_binding)
        or provenance["derived_dmo"] != dict(derived_binding)
        or provenance["random_seed"] != expected_seed
        or provenance["seed_offset"] != DMO_RANDOM_SEED_OFFSET
        or provenance["seed_size_bytes"] != DMO_RANDOM_SEED_BYTES
        or provenance["changed_byte_offsets"] != changed_offsets
        or provenance["unchanged_byte_count"]
        != len(base_bytes) - len(changed_offsets)
        or bytes(expected) != derived_bytes
        or demo.random_seed != expected_seed
    ):
        raise DistributionFidelityError(
            "seeded DMO changes more than its frozen gameplay seed"
        )


def _validate_seeded_collector_plan(
    path: Path,
    *,
    dmo_path: Path,
    dmo_sha256: str,
    runtime_sha256: str,
    expected_seed: int,
) -> None:
    plan = _read_canonical(path, "seeded collector plan")
    dmo = _mapping(plan.get("dmo"), "seeded collector plan.dmo")
    runtime = _mapping(
        plan.get("runtime"),
        "seeded collector plan.runtime",
    )
    trace = _mapping(plan.get("trace"), "seeded collector plan.trace")
    artifact = _relative_path(
        dmo.get("artifact"),
        "seeded collector plan.dmo.artifact",
    )
    try:
        planned_dmo = path.parent.joinpath(*artifact.parts).resolve(strict=True)
    except OSError as error:
        raise DistributionFidelityError(
            "seeded collector plan DMO is unavailable"
        ) from error
    seed_controls = (
        runtime.get("crt_rand_seed"),
        runtime.get("startup_seed_transport"),
        runtime.get("board_seed_call_address"),
        runtime.get("board_seed"),
        runtime.get("global_rng_seed"),
        runtime.get("thread_crt_rng_seed"),
    )
    if (
        plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan"
        or plan.get("version") != 5
        or planned_dmo != dmo_path.resolve()
        or dmo.get("sha256") != dmo_sha256
        or dmo.get("random_seed") != expected_seed
        or runtime.get("launch_mode")
        != "direct_byte_identical_natural_seed_strict_command_broker"
        or runtime.get("runtime_executable_sha256") != runtime_sha256
        or runtime.get("expected_runtime_sha256") != runtime_sha256
        or any(value is not None for value in seed_controls)
        or trace.get("attach_at_update") != 0
        or trace.get("startup_trace_handoff") is not True
        or trace.get("allow_pre_stream_commands") is not True
        or trace.get("allow_font_cache_manifest_completion_debt") is not True
        or trace.get("allow_pre_attach_file_write_debt") is not False
        or trace.get("seed_board_before_attach") is not False
        or trace.get("natural_command_broker_stop_after_update") != 3151
        or not isinstance(trace.get("font_cache_manifest_receipt"), dict)
    ):
        raise DistributionFidelityError(
            "seeded collector plan violates the natural strict protocol"
        )


def _read_canonical_value(path: Path, name: str) -> Any:
    try:
        return read_canonical_json(path)
    except PcExactStepEvidenceError as error:
        raise DistributionFidelityError(str(error)) from error


def _read_canonical(path: Path, name: str) -> Mapping[str, Any]:
    return _mapping(_read_canonical_value(path, name), name)


def _scope(value: Any, *, policy_id: str) -> Mapping[str, Any]:
    scope = _mapping(value, "scope")
    _check_keys(
        scope,
        required={
            "policy",
            "environment_id",
            "level_id",
            "hard",
            "profile_mode",
        },
        name="scope",
    )
    if (
        scope["policy"] != policy_id
        or scope["environment_id"] != "ZumaRevenge-v0"
        or scope["level_id"] != "Jungle2"
        or scope["hard"] is not False
        or scope["profile_mode"] != "tutorials_completed"
    ):
        raise DistributionFidelityError("distribution scope differs from policy")
    return scope


def _binding_copy(value: Any, name: str) -> dict[str, str]:
    binding = _mapping(value, name)
    _check_keys(binding, required={"path", "sha256"}, name=name)
    relative = _relative_path(binding["path"], f"{name}.path")
    return {
        "path": relative.as_posix(),
        "sha256": _digest(binding["sha256"], f"{name}.sha256"),
    }


def _validate_preregistration(
    value: Mapping[str, Any],
    *,
    root: Path,
    policy_id: str,
) -> dict[str, Any]:
    _check_keys(
        value,
        required={
            "schema",
            "version",
            "status",
            "frozen_utc",
            "scope",
            "metric",
            "selection",
            "pc_capture",
        },
        name="preregistration",
    )
    preregistration_version = value["version"]
    if (
        value["schema"] != PREREGISTRATION_SCHEMA
        or preregistration_version
        not in {LEGACY_PREREGISTRATION_VERSION, PREREGISTRATION_VERSION}
        or value["status"] != "FROZEN_BEFORE_DATA"
        or not isinstance(value["frozen_utc"], str)
        or _UTC_PATTERN.fullmatch(value["frozen_utc"]) is None
    ):
        raise DistributionFidelityError("preregistration is not frozen")
    _scope(value["scope"], policy_id=policy_id)

    metric = _mapping(value["metric"], "preregistration.metric")
    _check_keys(
        metric,
        required={
            "name",
            "kind",
            "minimum_pc_samples",
            "minimum_simulator_samples",
            "num_colors",
            "maximum_total_variation",
            "maximum_excess_total_variation",
            "minimum_compatibility_p_value",
            "permutation_count",
            "permutation_seed",
        },
        name="preregistration.metric",
    )
    if (
        metric["name"] != METRIC_NAME
        or metric["kind"] != "categorical_pair"
        or metric["minimum_pc_samples"] != MINIMUM_PC_SAMPLES
        or metric["minimum_simulator_samples"] != MINIMUM_SIMULATOR_SAMPLES
        or metric["num_colors"] != NUM_COLORS
        or _finite(
            metric["maximum_total_variation"],
            "maximum_total_variation",
        )
        != MAXIMUM_TOTAL_VARIATION
        or _finite(
            metric["maximum_excess_total_variation"],
            "maximum_excess_total_variation",
        )
        != MAXIMUM_EXCESS_TOTAL_VARIATION
        or _finite(
            metric["minimum_compatibility_p_value"],
            "minimum_compatibility_p_value",
        )
        != MINIMUM_COMPATIBILITY_P_VALUE
        or metric["permutation_count"] != PERMUTATION_COUNT
        or metric["permutation_seed"] != PERMUTATION_SEED
    ):
        raise DistributionFidelityError("preregistered metric policy differs")

    capture = _mapping(value["pc_capture"], "preregistration.pc_capture")
    capture_fields = {
        "freeze_update",
        "sample_update",
        "warmup_tick_count",
        "expected_score",
        "base_dmo",
        "runtime_payload",
        "base_collector_plan",
        "replay_prestate",
        "collector",
        "trace_tool",
        "seed_deriver",
        "gameplay_seed_transport",
        "strict_command_replay_required",
    }
    if preregistration_version == PREREGISTRATION_VERSION:
        capture_fields.add("replay_boundary_tool")
    _check_keys(
        capture,
        required=capture_fields,
        name="preregistration.pc_capture",
    )
    if (
        capture["freeze_update"] != PC_FREEZE_UPDATE
        or capture["sample_update"] != PC_SAMPLE_UPDATE
        or capture["warmup_tick_count"] != PC_WARMUP_TICK_COUNT
        or capture["expected_score"] != PC_EXPECTED_SCORE
        or capture["gameplay_seed_transport"] != GAMEPLAY_SEED_TRANSPORT
        or capture["strict_command_replay_required"] is not True
    ):
        raise DistributionFidelityError("preregistered PC sample point differs")
    base_dmo = _binding_copy(
        capture["base_dmo"],
        "preregistration.pc_capture.base_dmo",
    )
    runtime = _binding_copy(
        capture["runtime_payload"],
        "preregistration.pc_capture.runtime_payload",
    )
    base_collector_plan = _binding_copy(
        capture["base_collector_plan"],
        "preregistration.pc_capture.base_collector_plan",
    )
    replay_prestate = _binding_copy(
        capture["replay_prestate"],
        "preregistration.pc_capture.replay_prestate",
    )
    collector = _binding_copy(
        capture["collector"],
        "preregistration.pc_capture.collector",
    )
    trace_tool = _binding_copy(
        capture["trace_tool"],
        "preregistration.pc_capture.trace_tool",
    )
    replay_boundary_tool = (
        _binding_copy(
            capture["replay_boundary_tool"],
            "preregistration.pc_capture.replay_boundary_tool",
        )
        if preregistration_version == PREREGISTRATION_VERSION
        else None
    )
    seed_deriver = _binding_copy(
        capture["seed_deriver"],
        "preregistration.pc_capture.seed_deriver",
    )
    _bound_path(root, base_dmo, "preregistration.pc_capture.base_dmo")
    _bound_path(
        root,
        runtime,
        "preregistration.pc_capture.runtime_payload",
    )
    _bound_path(
        root,
        base_collector_plan,
        "preregistration.pc_capture.base_collector_plan",
    )
    _bound_path(
        root,
        replay_prestate,
        "preregistration.pc_capture.replay_prestate",
    )
    _bound_path(root, collector, "preregistration.pc_capture.collector")
    _bound_path(root, trace_tool, "preregistration.pc_capture.trace_tool")
    if replay_boundary_tool is not None:
        _bound_path(
            root,
            replay_boundary_tool,
            "preregistration.pc_capture.replay_boundary_tool",
        )
    _bound_path(root, seed_deriver, "preregistration.pc_capture.seed_deriver")

    selection = _mapping(value["selection"], "preregistration.selection")
    _check_keys(
        selection,
        required={
            "campaign_id",
            "pc_process_count",
            "pc_slots",
            "seed_schedule",
            "simulator_seed_count",
            "simulator_sample_tick",
            "retain_every_started_pc_process",
            "replacement_after_observation_forbidden",
        },
        name="preregistration.selection",
    )
    campaign_id = _identifier(selection["campaign_id"], "campaign_id")
    seed_schedule = _mapping(
        selection["seed_schedule"],
        "preregistration.seed_schedule",
    )
    _check_keys(
        seed_schedule,
        required={
            "id",
            "algorithm",
            "encoding",
            "pc_seed_count",
            "pc_seed_sha256",
            "simulator_seed_count",
            "simulator_seed_sha256",
            "populations_disjoint",
        },
        name="preregistration.seed_schedule",
    )
    pc_seeds, simulator_seeds = frozen_seed_schedules()
    if (
        selection["pc_process_count"] != MINIMUM_PC_SAMPLES
        or selection["simulator_seed_count"] != MINIMUM_SIMULATOR_SAMPLES
        or selection["simulator_sample_tick"] != SIMULATOR_SAMPLE_TICK
        or selection["retain_every_started_pc_process"] is not True
        or selection["replacement_after_observation_forbidden"] is not True
        or seed_schedule["id"] != SEED_SCHEDULE_ID
        or seed_schedule["algorithm"] != SEED_SCHEDULE_ALGORITHM
        or seed_schedule["encoding"] != SEED_SCHEDULE_ENCODING
        or seed_schedule["pc_seed_count"] != MINIMUM_PC_SAMPLES
        or seed_schedule["pc_seed_sha256"]
        != seed_sequence_sha256(pc_seeds)
        or seed_schedule["simulator_seed_count"]
        != MINIMUM_SIMULATOR_SAMPLES
        or seed_schedule["simulator_seed_sha256"]
        != seed_sequence_sha256(simulator_seeds)
        or seed_schedule["populations_disjoint"] is not True
        or set(pc_seeds) & set(simulator_seeds)
    ):
        raise DistributionFidelityError("preregistered selection policy differs")

    raw_slots = _sequence(selection["pc_slots"], "preregistration.pc_slots")
    if len(raw_slots) != MINIMUM_PC_SAMPLES:
        raise DistributionFidelityError("preregistered PC slot count differs")
    slots: list[dict[str, Any]] = []
    for index, raw_slot in enumerate(raw_slots, start=1):
        slot = _mapping(raw_slot, f"pc_slots[{index - 1}]")
        _check_keys(
            slot,
            required={
                "ordinal",
                "source_id",
                "run_id",
                "source_manifest_path",
                "output_root_path",
                "gameplay_seed",
                "dmo",
                "seed_provenance",
                "collector_plan",
            },
            name="pc slot",
        )
        if slot["ordinal"] != index:
            raise DistributionFidelityError("PC slot ordinals are not contiguous")
        if slot["gameplay_seed"] != pc_seeds[index - 1]:
            raise DistributionFidelityError(
                "PC slot gameplay seeds differ from the frozen schedule"
            )
        dmo = _binding_copy(slot["dmo"], "pc slot dmo")
        seed_provenance = _binding_copy(
            slot["seed_provenance"],
            "pc slot seed_provenance",
        )
        collector_plan = _binding_copy(
            slot["collector_plan"],
            "pc slot collector_plan",
        )
        dmo_path = _bound_path(root, dmo, "pc slot dmo")
        _bound_path(
            root,
            seed_provenance,
            "pc slot seed_provenance",
        )
        plan_path = _bound_path(
            root,
            collector_plan,
            "pc slot collector_plan",
        )
        _validate_seeded_dmo_provenance(
            root=root,
            base_binding=base_dmo,
            derived_binding=dmo,
            provenance_binding=seed_provenance,
            expected_seed=pc_seeds[index - 1],
        )
        _validate_seeded_collector_plan(
            plan_path,
            dmo_path=dmo_path,
            dmo_sha256=dmo["sha256"],
            runtime_sha256=runtime["sha256"],
            expected_seed=pc_seeds[index - 1],
        )
        slots.append(
            {
                "ordinal": index,
                "source_id": _identifier(slot["source_id"], "source_id"),
                "run_id": _identifier(slot["run_id"], "run_id"),
                "source_manifest_path": _relative_path(
                    slot["source_manifest_path"],
                    "source_manifest_path",
                ).as_posix(),
                "output_root_path": _relative_path(
                    slot["output_root_path"],
                    "output_root_path",
                ).as_posix(),
                "gameplay_seed": pc_seeds[index - 1],
                "dmo": dmo,
                "seed_provenance": seed_provenance,
                "collector_plan": collector_plan,
            }
        )
    for field in (
        "source_id",
        "run_id",
        "source_manifest_path",
        "output_root_path",
        "gameplay_seed",
    ):
        if len({slot[field] for slot in slots}) != len(slots):
            raise DistributionFidelityError(f"PC slot {field} values repeat")
    for field in ("dmo", "seed_provenance", "collector_plan"):
        if len({slot[field]["path"] for slot in slots}) != len(slots):
            raise DistributionFidelityError(f"PC slot {field} paths repeat")

    return {
        "campaign_id": campaign_id,
        "pc_slots": tuple(slots),
        "pc_seeds": pc_seeds,
        "simulator_seeds": simulator_seeds,
        "base_dmo": base_dmo,
        "runtime_payload": runtime,
        "base_collector_plan": base_collector_plan,
        "replay_prestate": replay_prestate,
        "collector": collector,
        "trace_tool": trace_tool,
        "replay_boundary_tool": replay_boundary_tool,
        "seed_deriver": seed_deriver,
    }


def _validate_dataset_identity(
    value: Mapping[str, Any],
    *,
    population: str,
) -> Sequence[Any]:
    _check_keys(
        value,
        required={"schema", "version", "population", "metric", "samples"},
        name=f"{population} dataset",
    )
    if (
        value["schema"] != DATASET_SCHEMA
        or value["version"] != DATASET_VERSION
        or value["population"] != population
        or value["metric"] != METRIC_NAME
    ):
        raise DistributionFidelityError(f"{population} dataset identity differs")
    rows = _sequence(value["samples"], f"{population}.samples")
    expected = (
        MINIMUM_PC_SAMPLES
        if population == "pc"
        else MINIMUM_SIMULATOR_SAMPLES
    )
    if len(rows) != expected:
        raise DistributionFidelityError(f"{population} sample count differs")
    return rows


def _validate_attempt_ledger(
    attempts_path: Path,
    *,
    process_id: int,
) -> None:
    rows = _sequence(
        _read_canonical_value(attempts_path, "attempt ledger"),
        "attempt ledger",
    )
    if len(rows) != 1:
        raise DistributionFidelityError(
            "PC slot started more or fewer than one retail process"
        )
    row = _mapping(rows[0], "attempt ledger row")
    started = _integer(
        row.get("started_perf_counter_ns"),
        "attempt started_perf_counter_ns",
        minimum=1,
    )
    finished = _integer(
        row.get("finished_perf_counter_ns"),
        "attempt finished_perf_counter_ns",
        minimum=1,
    )
    if (
        row.get("attempt") != 1
        or row.get("status") != "PASS"
        or row.get("process_id") != process_id
        or finished <= started
    ):
        raise DistributionFidelityError("PC slot attempt ledger is not one PASS")


def _validate_actor_sample(value: Any) -> tuple[int, int]:
    sample = _mapping(value, "PC actor-visible first frame")
    _check_keys(
        sample,
        required={
            "framework_update",
            "score",
            "displayed_score",
            "chain_ball_count",
            "inserting_ball_count",
            "fired_bullet_count",
            "current_ball_id",
            "current_color_id",
            "next_ball_id",
            "next_color_id",
            "board_color_counts",
        },
        name="PC actor-visible first frame",
    )
    integers = {
        name: _integer(sample[name], name)
        for name in (
            "framework_update",
            "score",
            "displayed_score",
            "chain_ball_count",
            "inserting_ball_count",
            "fired_bullet_count",
            "current_ball_id",
            "current_color_id",
            "next_ball_id",
            "next_color_id",
        )
    }
    counts = _sequence(sample["board_color_counts"], "board_color_counts")
    if (
        integers["framework_update"] != PC_SAMPLE_UPDATE
        or integers["score"] != PC_EXPECTED_SCORE
        or integers["displayed_score"] != PC_EXPECTED_SCORE
        or integers["chain_ball_count"] != 0
        or integers["inserting_ball_count"] != 0
        or integers["fired_bullet_count"] != 0
        or integers["current_ball_id"] == integers["next_ball_id"]
        or integers["current_color_id"] >= NUM_COLORS
        or integers["next_color_id"] >= NUM_COLORS
        or list(counts) != [0, 0, 0, 0, 0, 0]
    ):
        raise DistributionFidelityError("PC actor-visible sample point differs")
    return (
        integers["current_color_id"],
        integers["next_color_id"],
    )


def _validate_pc_dataset(
    value: Mapping[str, Any],
    *,
    root: Path,
    original_root: Path,
    selection: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    rows = _validate_dataset_identity(value, population="pc")
    values: list[int] = []
    fingerprints: list[str] = []
    process_identities: list[tuple[int, int]] = []
    gameplay_seeds: list[int] = []
    for index, (raw, slot) in enumerate(
        zip(rows, selection["pc_slots"], strict=True),
        start=1,
    ):
        row = _mapping(raw, f"pc.samples[{index - 1}]")
        _check_keys(
            row,
            required={"ordinal", "source_manifest"},
            name="PC sample",
        )
        if row["ordinal"] != index:
            raise DistributionFidelityError("PC sample ordinals are not contiguous")
        binding = _binding_copy(row["source_manifest"], "source_manifest")
        if binding["path"] != slot["source_manifest_path"]:
            raise DistributionFidelityError("PC sample replaced a frozen source slot")
        manifest_path = _bound_path(root, binding, "source_manifest")
        try:
            report = verify_pc_source_manifest(
                manifest_path,
                evidence_root=root,
                original_root=original_root,
            )
        except (OSError, PcSourceValidationError) as error:
            raise DistributionFidelityError(
                "PC distribution source is not authentic"
            ) from error
        manifest = _read_canonical(manifest_path, "PC source manifest")
        if (
            manifest.get("schema") != SOURCE_SCHEMA
            or manifest.get("version") != SOURCE_VERSION
            or manifest.get("source_id") != slot["source_id"]
            or manifest.get("dmo") != slot["dmo"]
            or manifest.get("runtime_payload") != selection["runtime_payload"]
        ):
            raise DistributionFidelityError("PC source differs from frozen slot")
        window = _mapping(manifest.get("window"), "PC source window")
        if window != {
            "freeze_update": PC_FREEZE_UPDATE,
            "start_update": PC_SAMPLE_UPDATE,
            "end_update": PC_SAMPLE_UPDATE,
            "warmup_tick_count": PC_WARMUP_TICK_COUNT,
        }:
            raise DistributionFidelityError("PC source sample window differs")
        run = _mapping(manifest.get("run"), "PC source run")
        if (
            run.get("run_id") != slot["run_id"]
            or run.get("selected_attempt") != 1
            or run.get("maximum_startup_attempts") != 1
        ):
            raise DistributionFidelityError("PC source retry policy differs")
        output_prefix = slot["output_root_path"] + "/"
        for artifact_name in (
            "attempts",
            "memory_probe",
            "trajectory_index",
        ):
            artifact = _mapping(
                run.get(artifact_name),
                f"PC source {artifact_name}",
            )
            artifact_path = artifact.get("path")
            if (
                not isinstance(artifact_path, str)
                or not artifact_path.startswith(output_prefix)
            ):
                raise DistributionFidelityError(
                    "PC source artifacts differ from frozen output slot"
                )
        identity = _mapping(
            report.get("independent_process_identity"),
            "independent_process_identity",
        )
        process_id = _integer(identity.get("process_id"), "process_id", minimum=1)
        process_filetime = _integer(
            identity.get("process_creation_filetime_100ns"),
            "process_creation_filetime_100ns",
            minimum=1,
        )
        if (
            run.get("process_id") != process_id
            or run.get("process_creation_filetime_100ns") != process_filetime
        ):
            raise DistributionFidelityError("PC source process identity differs")
        attempts_path = _bound_path(
            root,
            run.get("attempts"),
            "PC source attempts",
        )
        _validate_attempt_ledger(attempts_path, process_id=process_id)
        pair = _validate_actor_sample(report.get("actor_visible_first_frame"))
        strict_replay = _mapping(
            report.get("strict_command_replay"),
            "strict command replay",
        )
        if (
            report.get("random_seed") != slot["gameplay_seed"]
            or strict_replay.get("schema")
            != STRICT_NATURAL_REPLAY_BINDING_SCHEMA
            or strict_replay.get("version")
            != STRICT_NATURAL_REPLAY_BINDING_VERSION
            or strict_replay.get("status") != "PASS"
            or strict_replay.get("source_dmo_sha256")
            != slot["dmo"]["sha256"]
            or strict_replay.get("runtime_executable_sha256")
            != selection["runtime_payload"]["sha256"]
            or strict_replay.get("runtime_process_id") != process_id
            or strict_replay.get("register_override") is not None
            or strict_replay.get("rng_seed_override_count") != 0
            or strict_replay.get("rng_process_memory_writes") != 0
            or strict_replay.get("stop_after_update") != 3151
            or strict_replay.get("failure_count") != 0
        ):
            raise DistributionFidelityError(
                "PC source gameplay seed or strict replay differs"
            )
        fingerprint = _digest(
            report.get("source_fingerprint"),
            "source_fingerprint",
        )
        values.append(pair[0] * NUM_COLORS + pair[1])
        fingerprints.append(fingerprint)
        process_identities.append((process_id, process_filetime))
        gameplay_seeds.append(slot["gameplay_seed"])
    if len(set(fingerprints)) != len(fingerprints):
        raise DistributionFidelityError("PC source fingerprints repeat")
    if len(set(process_identities)) != len(process_identities):
        raise DistributionFidelityError("PC process identities repeat")
    if tuple(gameplay_seeds) != tuple(selection["pc_seeds"]):
        raise DistributionFidelityError("PC gameplay seed schedule differs")
    return tuple(values), tuple(fingerprints), tuple(gameplay_seeds)


def _validate_simulator_dataset(
    value: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> tuple[int, ...]:
    rows = _validate_dataset_identity(value, population="simulator")
    expected = tuple(selection["simulator_seeds"])
    seeds: list[int] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"simulator.samples[{index}]")
        _check_keys(row, required={"seed"}, name="simulator sample")
        seed = _integer(row["seed"], "simulator seed")
        if seed != expected[index]:
            raise DistributionFidelityError(
                "simulator seeds are not the frozen schedule"
            )
        seeds.append(seed)
    return tuple(seeds)


def _recompute_simulator_values(
    seeds: Sequence[int],
    *,
    original_root: Path,
) -> tuple[int, ...]:
    if not seeds:
        raise DistributionFidelityError("simulator seed range is empty")
    try:
        simulator = RevengeSimulator.from_installed(
            "Jungle2",
            root=original_root,
            hard=False,
            seed=int(seeds[0]),
            profile_mode="tutorials_completed",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise DistributionFidelityError(
            "simulator startup samples could not be initialized"
        ) from error
    values: list[int] = []
    for seed in seeds:
        try:
            simulator.reset(seed=int(seed))
            if (
                simulator.tick_count != 0
                or simulator.current_color is not None
                or simulator.next_color is not None
                or simulator.score != 0
                or simulator.outcome is not None
                or simulator.curve_count != 1
                or simulator.num_colors != NUM_COLORS
            ):
                raise DistributionFidelityError(
                    "simulator reset state differs from frozen sample point"
                )
            simulator.tick()
        except DistributionFidelityError:
            raise
        except (RuntimeError, ValueError) as error:
            raise DistributionFidelityError(
                "simulator startup sample could not be replayed"
            ) from error
        current = simulator.current_color
        following = simulator.next_color
        if (
            simulator.tick_count != SIMULATOR_SAMPLE_TICK
            or isinstance(current, bool)
            or not isinstance(current, int)
            or isinstance(following, bool)
            or not isinstance(following, int)
            or not 0 <= current < NUM_COLORS
            or not 0 <= following < NUM_COLORS
        ):
            raise DistributionFidelityError(
                "simulator actor-visible sample point differs"
            )
        values.append(current * NUM_COLORS + following)
    return tuple(values)


def categorical_total_variation(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    """Return empirical total-variation distance for two samples."""

    if not left or not right:
        raise DistributionFidelityError("distribution samples must not be empty")
    left_counts = Counter(left)
    right_counts = Counter(right)
    support = set(left_counts) | set(right_counts)
    return 0.5 * math.fsum(
        abs(left_counts[item] / len(left) - right_counts[item] / len(right))
        for item in support
    )


def verify_distribution_preregistration(
    preregistration_path: str | Path,
    *,
    evidence_root: str | Path,
    policy_id: str,
) -> dict[str, Any]:
    """Recompute a frozen 32/256 protocol before any PC sample exists."""

    try:
        root = Path(evidence_root).resolve(strict=True)
        source = Path(preregistration_path).resolve(strict=True)
        source.relative_to(root)
    except (OSError, ValueError) as error:
        raise DistributionFidelityError(
            "distribution preregistration is unavailable"
        ) from error
    selection = _validate_preregistration(
        _read_canonical(source, "preregistration"),
        root=root,
        policy_id=policy_id,
    )
    pc_seeds = tuple(selection["pc_seeds"])
    simulator_seeds = tuple(selection["simulator_seeds"])
    return {
        "schema": PREREGISTRATION_REPORT_SCHEMA,
        "version": PREREGISTRATION_REPORT_VERSION,
        "status": "PASS",
        "campaign_id": selection["campaign_id"],
        "policy_id": policy_id,
        "pc_slot_count": len(selection["pc_slots"]),
        "simulator_seed_count": len(simulator_seeds),
        "seed_schedule_id": SEED_SCHEDULE_ID,
        "pc_gameplay_seed_sha256": seed_sequence_sha256(pc_seeds),
        "simulator_seed_sha256": seed_sequence_sha256(simulator_seeds),
        "seed_populations_disjoint": not bool(
            set(pc_seeds) & set(simulator_seeds)
        ),
        "gameplay_seed_transport": GAMEPLAY_SEED_TRANSPORT,
        "strict_command_replay_required": True,
        "same_dmo_replay_rejected": True,
        "samples_observed": False,
    }


def _permutation_calibration(
    left: Sequence[int],
    right: Sequence[int],
    *,
    observed: float,
) -> tuple[float, float]:
    """Return a deterministic null median and upper-tail p-value."""

    pooled = list(left) + list(right)
    left_count = len(left)
    state = PERMUTATION_SEED
    null_distances: list[float] = []
    exceedances = 0
    for _ in range(PERMUTATION_COUNT):
        shuffled = pooled.copy()
        for index in range(len(shuffled) - 1, 0, -1):
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
            swap_index = state % (index + 1)
            shuffled[index], shuffled[swap_index] = (
                shuffled[swap_index],
                shuffled[index],
            )
        distance = categorical_total_variation(
            shuffled[:left_count],
            shuffled[left_count:],
        )
        null_distances.append(distance)
        if distance >= observed:
            exceedances += 1
    null_distances.sort()
    null_median = null_distances[len(null_distances) // 2]
    p_value = (exceedances + 1) / (PERMUTATION_COUNT + 1)
    return null_median, p_value


def verify_distribution_audit(
    audit_path: str | Path,
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    policy_id: str,
) -> dict[str, Any]:
    """Recompute the frozen PC-versus-simulator distribution decision."""

    if original_root is None:
        raise DistributionFidelityError("original_root is required")
    try:
        root = Path(evidence_root).resolve(strict=True)
        original = Path(original_root).resolve(strict=True)
        source = Path(audit_path).resolve(strict=True)
        source.relative_to(root)
    except (OSError, ValueError) as error:
        raise DistributionFidelityError(
            "distribution audit or original_root is unavailable"
        ) from error
    audit = _read_canonical(source, "distribution audit")
    _check_keys(
        audit,
        required={
            "schema",
            "version",
            "status",
            "scope",
            "preregistration",
            "pc_dataset",
            "simulator_dataset",
        },
        name="distribution audit",
    )
    if (
        audit["schema"] != AUDIT_SCHEMA
        or audit["version"] != AUDIT_VERSION
        or audit["status"] != "READY_FOR_RECOMPUTE"
    ):
        raise DistributionFidelityError("distribution audit identity differs")
    scope = _scope(audit["scope"], policy_id=policy_id)
    prereg_path = _bound_path(root, audit["preregistration"], "preregistration")
    pc_path = _bound_path(root, audit["pc_dataset"], "pc_dataset")
    simulator_path = _bound_path(root, audit["simulator_dataset"], "simulator_dataset")
    selection = _validate_preregistration(
        _read_canonical(prereg_path, "preregistration"),
        root=root,
        policy_id=policy_id,
    )
    pc_values, pc_sources, pc_seeds = _validate_pc_dataset(
        _read_canonical(pc_path, "pc dataset"),
        root=root,
        original_root=original,
        selection=selection,
    )
    simulator_seeds = _validate_simulator_dataset(
        _read_canonical(simulator_path, "simulator dataset"),
        selection=selection,
    )
    simulator_values = _recompute_simulator_values(
        simulator_seeds,
        original_root=original,
    )
    distance = categorical_total_variation(pc_values, simulator_values)
    null_median, p_value = _permutation_calibration(
        pc_values,
        simulator_values,
        observed=distance,
    )
    excess_distance = max(0.0, distance - null_median)
    passed = (
        distance <= MAXIMUM_TOTAL_VARIATION
        and excess_distance <= MAXIMUM_EXCESS_TOTAL_VARIATION
        and p_value >= MINIMUM_COMPATIBILITY_P_VALUE
    )
    return {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "status": "PASS" if passed else "FAIL",
        "scope": dict(scope),
        "feature": "startup_actor_distribution",
        "metric": METRIC_NAME,
        "pc_sample_count": len(pc_values),
        "simulator_sample_count": len(simulator_values),
        "total_variation": distance,
        "maximum_total_variation": MAXIMUM_TOTAL_VARIATION,
        "permutation_null_median_total_variation": null_median,
        "excess_total_variation": excess_distance,
        "maximum_excess_total_variation": MAXIMUM_EXCESS_TOTAL_VARIATION,
        "compatibility_p_value": p_value,
        "minimum_compatibility_p_value": MINIMUM_COMPATIBILITY_P_VALUE,
        "permutation_count": PERMUTATION_COUNT,
        "permutation_seed": PERMUTATION_SEED,
        "source_fingerprints": sorted(pc_sources),
        "seed_schedule_id": SEED_SCHEDULE_ID,
        "pc_gameplay_seed_sha256": seed_sequence_sha256(pc_seeds),
        "simulator_seed_sha256": seed_sequence_sha256(simulator_seeds),
        "seed_populations_disjoint": not bool(
            set(pc_seeds) & set(simulator_seeds)
        ),
        "gameplay_seed_transport": GAMEPLAY_SEED_TRANSPORT,
        "same_dmo_replay_rejected": True,
        "preregistered_before_data": True,
        "raw_samples_recomputed": True,
        "every_started_pc_process_retained": True,
        "campaign_id": selection["campaign_id"],
    }


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "DATASET_SCHEMA",
    "DATASET_VERSION",
    "DMO_RANDOM_SEED_BYTES",
    "DMO_RANDOM_SEED_OFFSET",
    "DistributionFidelityError",
    "GAMEPLAY_SEED_TRANSPORT",
    "MAXIMUM_TOTAL_VARIATION",
    "MAXIMUM_EXCESS_TOTAL_VARIATION",
    "MECHANISM_PROBE_SEED_ROLE",
    "METRIC_NAME",
    "MINIMUM_COMPATIBILITY_P_VALUE",
    "MINIMUM_PC_SAMPLES",
    "MINIMUM_SIMULATOR_SAMPLES",
    "NUM_COLORS",
    "PC_EXPECTED_SCORE",
    "PC_FREEZE_UPDATE",
    "PC_SAMPLE_UPDATE",
    "PC_WARMUP_TICK_COUNT",
    "PERMUTATION_COUNT",
    "PERMUTATION_SEED",
    "PREREGISTRATION_SCHEMA",
    "PREREGISTRATION_VERSION",
    "PREREGISTRATION_REPORT_SCHEMA",
    "PREREGISTRATION_REPORT_VERSION",
    "SEEDED_DMO_PROVENANCE_SCHEMA",
    "SEEDED_DMO_PROVENANCE_VERSION",
    "SEED_SCHEDULE_ALGORITHM",
    "SEED_SCHEDULE_ENCODING",
    "SEED_SCHEDULE_ID",
    "SIMULATOR_SAMPLE_TICK",
    "STRICT_NATURAL_REPLAY_BINDING_SCHEMA",
    "STRICT_NATURAL_REPLAY_BINDING_VERSION",
    "categorical_total_variation",
    "frozen_seed_schedules",
    "mechanism_probe_seed",
    "seed_sequence_sha256",
    "verify_distribution_audit",
    "verify_distribution_preregistration",
]
