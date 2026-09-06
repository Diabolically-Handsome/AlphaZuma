"""Insert failed file-write results after one exact DMO command.

This is a provenance-bound diagnostic transformation for recordings whose
asynchronous file-service results were emitted before the corresponding
playback calls.  It does not modify the source.  The caller must bind the
source SHA-256 and the complete anchor identity; output and provenance are
created exclusively and are explicitly classified as diagnostic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.align_popcap_dmo_startup import (
    _bit_tuple,
    _command_stream_offset,
    _load_popcap_dmo_module,
    _pack_lsb_bits,
    _scan_commands,
)
from tools.remove_popcap_dmo_command import _normalise_sha256


FILE_WRITE_COMMAND_NUMBER = 16
DIAGNOSTIC_INSERTION_SCHEMA = (
    "zuma-rl.diagnostic-dmo-file-write-insertion"
)
DIAGNOSTIC_CLASSIFICATION = "diagnostic-not-unmodified-pc-evidence"
SUCCESSFUL_PADDING_REASON = (
    "pad one audited successful file-write corridor so the retail "
    "fixed-width replay prefetch ends at its exact service boundary"
)


def _file_write_bits(
    timing_delta: int,
    *,
    success: bool,
) -> tuple[int, ...]:
    if not 0 <= timing_delta <= 15:
        raise ValueError("file-write timing delta must fit four bits")
    return (
        tuple((timing_delta >> bit) & 1 for bit in range(4))
        + (0,)  # long command
        + tuple(
            (FILE_WRITE_COMMAND_NUMBER >> bit) & 1
            for bit in range(5)
        )
        + (int(success),)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_dict(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} fields are invalid")
    return value


def _metadata_int(
    mapping: Mapping[str, object],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"provenance {key} is invalid")
    return value


def validate_successful_file_write_padding_provenance(
    data: bytes,
    provenance: Mapping[str, object],
) -> tuple[int, ...]:
    """Validate and locate provenance-bound successful padding rows.

    This accepts only the one-command, eleven-bit ``file_write(true)``
    diagnostic transform used to align the retail fixed-width service read.
    The inserted bits are removed again and the original DMO is reconstructed,
    so both source and output hashes and all original command semantics are
    independently checked rather than trusted from the sidecar.
    """

    if not isinstance(data, bytes):
        raise TypeError("DMO data must be bytes")
    top = _exact_dict(
        dict(provenance) if isinstance(provenance, Mapping) else provenance,
        name="provenance",
        keys={
            "schema",
            "version",
            "classification",
            "source",
            "transformation",
            "output",
        },
    )
    if top["schema"] != DIAGNOSTIC_INSERTION_SCHEMA:
        raise ValueError("provenance schema is invalid")
    if type(top["version"]) is not int or top["version"] != 1:
        raise ValueError("provenance version is invalid")
    if top["classification"] != DIAGNOSTIC_CLASSIFICATION:
        raise ValueError("provenance classification is invalid")

    metadata_keys = {
        "bytes",
        "sha256",
        "commands",
        "length_updates",
        "meaningful_command_bits",
    }
    source = _exact_dict(
        top["source"], name="provenance source", keys=metadata_keys
    )
    output = _exact_dict(
        top["output"], name="provenance output", keys=metadata_keys
    )
    transformation_keys = {
        "anchor_command_index",
        "anchor_kind",
        "anchor_update",
        "anchor_bit_start",
        "anchor_bit_end",
        "inserted_after_anchor",
        "inserted_command_count",
        "inserted_command_kind",
        "inserted_command_number",
        "inserted_payload",
        "inserted_update",
        "first_inserted_timing_delta",
        "following_inserted_timing_delta",
        "inserted_bits_per_command",
        "following_command_update",
        "following_original_timing_delta",
        "following_replacement_timing_delta",
        "source_command_semantics_preserved",
        "reason",
    }
    transformation = _exact_dict(
        top["transformation"],
        name="provenance transformation",
        keys=transformation_keys,
    )

    for section_name, metadata in (("source", source), ("output", output)):
        for key in (
            "bytes",
            "commands",
            "length_updates",
            "meaningful_command_bits",
        ):
            _metadata_int(metadata, key)
        sha256 = metadata.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"provenance {section_name} sha256 is invalid")

    if output["bytes"] != len(data) or output["sha256"] != _sha256(data):
        raise ValueError("provenance output identity does not match the DMO")

    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    output_rows, output_meaningful_bits = _scan_commands(
        module,
        data[offset:],
        length_updates,
    )
    parsed_output = module.PopCapDemo.from_bytes(data)
    expected_output_metadata = {
        "bytes": len(data),
        "sha256": _sha256(data),
        "commands": len(parsed_output.commands),
        "length_updates": parsed_output.length_updates,
        "meaningful_command_bits": output_meaningful_bits,
    }
    if output != expected_output_metadata:
        raise ValueError("provenance output metadata is inconsistent")

    anchor_index = _metadata_int(
        transformation, "anchor_command_index"
    )
    anchor_update = _metadata_int(transformation, "anchor_update")
    anchor_start = _metadata_int(transformation, "anchor_bit_start")
    anchor_end = _metadata_int(transformation, "anchor_bit_end")
    inserted_update = _metadata_int(transformation, "inserted_update")
    first_delta = _metadata_int(
        transformation, "first_inserted_timing_delta"
    )
    following_delta = _metadata_int(
        transformation, "following_inserted_timing_delta"
    )
    inserted_bits = _metadata_int(
        transformation, "inserted_bits_per_command", minimum=1
    )
    if (
        transformation["inserted_after_anchor"] is not True
        or transformation["inserted_command_count"] != 1
        or transformation["inserted_command_kind"] != "file_write"
        or transformation["inserted_command_number"] != FILE_WRITE_COMMAND_NUMBER
        or transformation["inserted_payload"] != {"success": True}
        or transformation["source_command_semantics_preserved"] is not True
        or transformation["reason"] != SUCCESSFUL_PADDING_REASON
        or inserted_bits != 11
        or following_delta != 0
        or first_delta != inserted_update - anchor_update
        or not 0 <= first_delta <= 15
    ):
        raise ValueError("provenance is not an exact successful padding transform")
    if anchor_index + 1 >= len(output_rows):
        raise ValueError("provenance inserted row is absent from the output")

    anchor = output_rows[anchor_index]
    inserted_index = anchor_index + 1
    inserted = output_rows[inserted_index]
    if (
        transformation["anchor_kind"] != anchor.get("kind")
        or anchor_update != int(anchor["update"])
        or anchor_start != int(anchor["start"])
        or anchor_end != int(anchor["end"])
        or int(inserted["start"]) != anchor_end
        or int(inserted["end"]) != anchor_end + inserted_bits
        or int(inserted["update"]) != inserted_update
        or inserted.get("kind") != "file_write"
        or bool(inserted["short_form"])
        or int(inserted["command_number"]) != FILE_WRITE_COMMAND_NUMBER
        or inserted.get("payload") != {"success": True}
    ):
        raise ValueError("provenance anchor or inserted row is inconsistent")

    output_bits = _bit_tuple(data[offset:], 0, output_meaningful_bits)
    following_index = inserted_index + 1
    following_update = transformation["following_command_update"]
    original_delta = transformation["following_original_timing_delta"]
    replacement_delta = transformation["following_replacement_timing_delta"]
    if following_index < len(output_rows):
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (following_update, original_delta, replacement_delta)
        ):
            raise ValueError("provenance following-command metadata is invalid")
        assert isinstance(following_update, int)
        assert isinstance(original_delta, int)
        assert isinstance(replacement_delta, int)
        following = output_rows[following_index]
        expected_replacement_delta = following_update - inserted_update
        expected_original_delta = following_update - anchor_update
        if (
            int(following["start"]) != int(inserted["end"])
            or int(following["update"]) != following_update
            or original_delta != expected_original_delta
            or replacement_delta != expected_replacement_delta
            or not 0 <= original_delta <= 15
            or not 0 <= replacement_delta <= 15
        ):
            raise ValueError("provenance following-command timing is inconsistent")
        original_timing_bits = tuple(
            (original_delta >> bit) & 1 for bit in range(4)
        )
        source_bits = (
            output_bits[:anchor_end]
            + original_timing_bits
            + output_bits[anchor_end + inserted_bits + 4 :]
        )
    else:
        if any(
            value is not None
            for value in (following_update, original_delta, replacement_delta)
        ):
            raise ValueError("terminal insertion has following-command metadata")
        source_bits = (
            output_bits[:anchor_end]
            + output_bits[anchor_end + inserted_bits :]
        )

    source_data = data[:offset] + _pack_lsb_bits(source_bits)
    source_rows, source_meaningful_bits = _scan_commands(
        module,
        source_data[offset:],
        length_updates,
    )
    parsed_source = module.PopCapDemo.from_bytes(source_data)
    expected_source_metadata = {
        "bytes": len(source_data),
        "sha256": _sha256(source_data),
        "commands": len(parsed_source.commands),
        "length_updates": parsed_source.length_updates,
        "meaningful_command_bits": source_meaningful_bits,
    }
    if source != expected_source_metadata:
        raise ValueError("provenance source identity cannot be reconstructed")
    expected_output_rows = (
        source_rows[:inserted_index]
        + [inserted]
        + source_rows[inserted_index:]
    )
    semantic_keys = (
        "update",
        "short_form",
        "command_number",
        "kind",
        "payload",
    )
    if len(expected_output_rows) != len(output_rows):
        raise ValueError("provenance reconstructed command count is inconsistent")
    for index, (observed, expected) in enumerate(
        zip(output_rows, expected_output_rows, strict=True)
    ):
        if any(observed[key] != expected[key] for key in semantic_keys):
            raise ValueError(
                f"provenance source command semantics changed at row {index}"
            )
    return (inserted_index,)


def insert_failed_file_writes(
    data: bytes,
    *,
    anchor_index: int,
    expected_anchor_kind: str,
    expected_anchor_update: int,
    expected_anchor_start_bit: int,
    expected_anchor_end_bit: int,
    count: int,
    inserted_update: int | None = None,
    inserted_success: bool = False,
) -> tuple[bytes, dict[str, object]]:
    """Insert provenance-bound file-write results after an anchor.

    ``inserted_success=False`` preserves the historical startup-debt use case.
    A successful result is allowed only when the caller explicitly requests it
    and remains fully represented in the output provenance.
    """

    if anchor_index < 0:
        raise ValueError("anchor index must be non-negative")
    if count <= 0:
        raise ValueError("inserted result count must be positive")

    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if anchor_index >= len(rows):
        raise ValueError(f"anchor index {anchor_index} does not exist")

    anchor = rows[anchor_index]
    expected = {
        "kind": expected_anchor_kind,
        "update": expected_anchor_update,
        "start": expected_anchor_start_bit,
        "end": expected_anchor_end_bit,
    }
    for key, value in expected.items():
        if anchor[key] != value:
            raise ValueError(
                f"anchor {anchor_index} {key} mismatch: "
                f"{anchor[key]!r} != {value!r}"
            )
    if expected_anchor_update > length_updates:
        raise ValueError("anchor update exceeds the declared DMO length")
    effective_inserted_update = (
        expected_anchor_update
        if inserted_update is None
        else inserted_update
    )
    if not expected_anchor_update <= effective_inserted_update <= length_updates:
        raise ValueError(
            "inserted update must be between the anchor and DMO length"
        )
    following_update: int | None = None
    following_original_timing_delta: int | None = None
    following_replacement_timing_delta: int | None = None
    if anchor_index + 1 < len(rows):
        following_update = int(rows[anchor_index + 1]["update"])
        if effective_inserted_update > following_update:
            raise ValueError(
                "inserted update exceeds the following command update"
            )
        following_original_timing_delta = (
            following_update - expected_anchor_update
        )
        following_replacement_timing_delta = (
            following_update - effective_inserted_update
        )
        if not 0 <= following_replacement_timing_delta <= 15:
            raise ValueError(
                "replacement following-command timing does not fit four bits"
            )
    first_timing_delta = effective_inserted_update - expected_anchor_update
    first_bits = _file_write_bits(
        first_timing_delta,
        success=inserted_success,
    )
    following_bits = _file_write_bits(0, success=inserted_success)

    inserted_bits = first_bits + following_bits * (count - 1)
    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    if following_replacement_timing_delta is None:
        source_tail = source_bits[expected_anchor_end_bit:]
    else:
        replacement_timing_bits = tuple(
            (following_replacement_timing_delta >> bit) & 1
            for bit in range(4)
        )
        source_tail = (
            replacement_timing_bits
            + source_bits[expected_anchor_end_bit + 4 :]
        )
    output_bits = (
        source_bits[:expected_anchor_end_bit]
        + inserted_bits
        + source_tail
    )
    output = data[:offset] + _pack_lsb_bits(output_bits)

    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_output = module.PopCapDemo.from_bytes(output)
    if parsed_output.length_updates != parsed_source.length_updates:
        raise RuntimeError("output DMO length_updates changed")
    if len(parsed_output.commands) != len(parsed_source.commands) + count:
        raise RuntimeError("output command count is inconsistent")

    output_rows, output_meaningful_bits = _scan_commands(
        module,
        output[offset:],
        length_updates,
    )
    inserted_rows = [
        {
            "update": effective_inserted_update,
            "short_form": False,
            "command_number": FILE_WRITE_COMMAND_NUMBER,
            "kind": "file_write",
            "payload": {"success": inserted_success},
        }
        for _ in range(count)
    ]
    expected_rows = (
        rows[: anchor_index + 1]
        + inserted_rows
        + rows[anchor_index + 1 :]
    )
    if len(output_rows) != len(expected_rows):
        raise RuntimeError("output scan command count is inconsistent")
    for index, (observed, expected_row) in enumerate(
        zip(output_rows, expected_rows, strict=True)
    ):
        for key in (
            "update",
            "short_form",
            "command_number",
            "kind",
            "payload",
        ):
            if observed[key] != expected_row[key]:
                raise RuntimeError(
                    f"output command {index} {key} changed: "
                    f"{observed[key]!r} != {expected_row[key]!r}"
                )
    if output_meaningful_bits != len(output_bits):
        raise RuntimeError("output meaningful bit count is inconsistent")

    provenance = {
        "schema": DIAGNOSTIC_INSERTION_SCHEMA,
        "version": 1,
        "classification": DIAGNOSTIC_CLASSIFICATION,
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(parsed_source.commands),
            "length_updates": parsed_source.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
        "transformation": {
            "anchor_command_index": anchor_index,
            "anchor_kind": expected_anchor_kind,
            "anchor_update": expected_anchor_update,
            "anchor_bit_start": expected_anchor_start_bit,
            "anchor_bit_end": expected_anchor_end_bit,
            "inserted_after_anchor": True,
            "inserted_command_count": count,
            "inserted_command_kind": "file_write",
            "inserted_command_number": FILE_WRITE_COMMAND_NUMBER,
            "inserted_payload": {"success": inserted_success},
            "inserted_update": effective_inserted_update,
            "first_inserted_timing_delta": first_timing_delta,
            "following_inserted_timing_delta": 0,
            "inserted_bits_per_command": len(following_bits),
            "following_command_update": following_update,
            "following_original_timing_delta": (
                following_original_timing_delta
            ),
            "following_replacement_timing_delta": (
                following_replacement_timing_delta
            ),
            "source_command_semantics_preserved": True,
            "reason": (
                SUCCESSFUL_PADDING_REASON
                if inserted_success
                else (
                    "replay file-service consumers execute immediately after "
                    "loading_complete, later than the recorded asynchronous "
                    "result block"
                )
            ),
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed_output.commands),
            "length_updates": parsed_output.length_updates,
            "meaningful_command_bits": len(output_bits),
        },
    }
    return output, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--anchor-index", required=True, type=int)
    parser.add_argument("--expected-anchor-kind", required=True)
    parser.add_argument("--expected-anchor-update", required=True, type=int)
    parser.add_argument("--expected-anchor-start-bit", required=True, type=int)
    parser.add_argument("--expected-anchor-end-bit", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--inserted-update", type=int)
    parser.add_argument(
        "--inserted-success",
        action="store_true",
        help="Insert file_write(true) instead of the historical false result.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"input DMO does not exist: {args.input}")
    provenance_path = args.output.with_name(
        args.output.name + ".provenance.json"
    )
    for path in (args.output, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")

    source = args.input.read_bytes()
    source_sha256 = _sha256(source)
    expected_source = _normalise_sha256(args.expected_source_sha256)
    if source_sha256 != expected_source:
        raise ValueError(
            f"source SHA-256 mismatch: {source_sha256} != {expected_source}"
        )

    output, provenance = insert_failed_file_writes(
        source,
        anchor_index=args.anchor_index,
        expected_anchor_kind=args.expected_anchor_kind,
        expected_anchor_update=args.expected_anchor_update,
        expected_anchor_start_bit=args.expected_anchor_start_bit,
        expected_anchor_end_bit=args.expected_anchor_end_bit,
        count=args.count,
        inserted_update=args.inserted_update,
        inserted_success=args.inserted_success,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(output)
    with provenance_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            provenance,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    print(f"output={args.output}")
    print(f"provenance={provenance_path}")
    print(f"sha256={provenance['output']['sha256']}")
    print(f"bytes={provenance['output']['bytes']}")
    print(f"commands={provenance['output']['commands']}")
    print(f"length_updates={provenance['output']['length_updates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
