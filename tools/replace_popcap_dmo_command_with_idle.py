"""Replace one fully specified DMO command with a same-update idle command.

This diagnostic transformation is for recording-only service results whose
timing cannot be transferred to the following command because the combined
four-bit timing delta would exceed 15.  The source hash, command identity,
bit span, and payload hash are mandatory.  The source is never modified and
the output plus provenance record are created exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.align_popcap_dmo_startup import (
    _bit_tuple,
    _command_stream_offset,
    _load_popcap_dmo_module,
    _pack_lsb_bits,
    _scan_commands,
)
from tools.remove_popcap_dmo_command import (
    _normalise_sha256,
    _payload_sha256,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_command_with_idle(
    data: bytes,
    *,
    command_index: int,
    expected_kind: str,
    expected_update: int,
    expected_start_bit: int,
    expected_end_bit: int,
    expected_payload_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    if command_index < 0:
        raise ValueError("command index must be non-negative")
    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if command_index >= len(rows):
        raise ValueError(f"command index {command_index} does not exist")

    row = rows[command_index]
    expected = {
        "kind": expected_kind,
        "update": expected_update,
        "start": expected_start_bit,
        "end": expected_end_bit,
    }
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(
                f"command {command_index} {key} mismatch: "
                f"{row[key]!r} != {value!r}"
            )
    if row["short_form"]:
        raise ValueError("refusing to replace a short-form command")
    observed_payload_sha256 = _payload_sha256(row)
    normalised_expected_payload = _normalise_sha256(
        expected_payload_sha256
    )
    if observed_payload_sha256 != normalised_expected_payload:
        raise ValueError(
            "command payload SHA-256 mismatch: "
            f"{observed_payload_sha256} != {normalised_expected_payload}"
        )

    previous_update = (
        int(rows[command_index - 1]["update"])
        if command_index
        else 0
    )
    timing_delta = expected_update - previous_update
    if not 0 <= timing_delta <= 15:
        raise ValueError("idle timing does not fit the four-bit DMO field")

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    idle_bits = (
        tuple((timing_delta >> bit) & 1 for bit in range(4))
        + (0,)
        + tuple((31 >> bit) & 1 for bit in range(5))
    )
    output_bits = (
        source_bits[:expected_start_bit]
        + idle_bits
        + source_bits[expected_end_bit:]
    )
    output = data[:offset] + _pack_lsb_bits(output_bits)

    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_output = module.PopCapDemo.from_bytes(output)
    if parsed_output.length_updates != parsed_source.length_updates:
        raise RuntimeError("output DMO length_updates changed")
    if len(parsed_output.commands) != len(parsed_source.commands):
        raise RuntimeError("output command count changed")

    output_rows, output_meaningful_bits = _scan_commands(
        module,
        output[offset:],
        length_updates,
    )
    expected_rows = [dict(value) for value in rows]
    expected_rows[command_index] = {
        **expected_rows[command_index],
        "short_form": False,
        "command_number": 31,
        "kind": "idle",
        "payload": {},
    }
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
        "schema": "zuma-rl.diagnostic-dmo-command-idle-replacement",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(parsed_source.commands),
            "length_updates": parsed_source.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
        "transformation": {
            "replaced_command_index": command_index,
            "replaced_kind": expected_kind,
            "replaced_update": expected_update,
            "replaced_bit_start": expected_start_bit,
            "replaced_bit_end": expected_end_bit,
            "replaced_bits": expected_end_bit - expected_start_bit,
            "replaced_payload_sha256": normalised_expected_payload,
            "replacement_kind": "idle",
            "replacement_bits": len(idle_bits),
            "replacement_timing_delta": timing_delta,
            "timeline_preserved": True,
            "reason": (
                "recording-only service result had no matching playback "
                "service consumer"
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
    parser.add_argument("--command-index", required=True, type=int)
    parser.add_argument("--expected-kind", required=True)
    parser.add_argument("--expected-update", required=True, type=int)
    parser.add_argument("--expected-start-bit", required=True, type=int)
    parser.add_argument("--expected-end-bit", required=True, type=int)
    parser.add_argument("--expected-payload-sha256", required=True)
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

    output, provenance = replace_command_with_idle(
        source,
        command_index=args.command_index,
        expected_kind=args.expected_kind,
        expected_update=args.expected_update,
        expected_start_bit=args.expected_start_bit,
        expected_end_bit=args.expected_end_bit,
        expected_payload_sha256=args.expected_payload_sha256,
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
