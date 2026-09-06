"""Replace one exact DMO service-result sequence with same-update idles.

This diagnostic transformation is for a contiguous asynchronous service
result sequence that was emitted by the recorder but has no matching playback
consumer.  The source hash, command indices, update pattern, bit span, kind,
and payload hash are mandatory.  Every replacement keeps the command's
absolute framework update, while every non-target command is preserved
bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Iterable, Sequence

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


def _idle_bits(timing_delta: int) -> tuple[int, ...]:
    if not 0 <= timing_delta <= 15:
        raise ValueError("idle timing does not fit the four-bit DMO field")
    return (
        tuple((timing_delta >> bit) & 1 for bit in range(4))
        + (0,)
        + tuple((31 >> bit) & 1 for bit in range(5))
    )


def replace_service_sequence_with_idle(
    data: bytes,
    *,
    start_index: int,
    end_index: int,
    expected_updates: Sequence[int],
    expected_start_bit: int,
    expected_end_bit: int,
    expected_kind: str,
    expected_payload_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    """Replace a fully bound contiguous service sequence with idle commands."""

    if start_index < 0 or end_index < start_index:
        raise ValueError("service sequence indices are invalid")
    count = end_index - start_index + 1
    if len(expected_updates) != count:
        raise ValueError("service sequence update count is invalid")

    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if end_index >= len(rows):
        raise ValueError("service sequence exceeds the command stream")

    sequence = rows[start_index : end_index + 1]
    if (
        int(sequence[0]["start"]) != expected_start_bit
        or int(sequence[-1]["end"]) != expected_end_bit
    ):
        raise ValueError("service sequence bit span mismatch")
    for previous, current in zip(sequence, sequence[1:]):
        if int(previous["end"]) != int(current["start"]):
            raise ValueError("service sequence command spans are not contiguous")
    observed_updates = tuple(int(row["update"]) for row in sequence)
    if observed_updates != tuple(expected_updates):
        raise ValueError("service sequence update pattern mismatch")
    if any(str(row["kind"]) != expected_kind for row in sequence):
        raise ValueError("service sequence kind mismatch")
    if any(bool(row["short_form"]) for row in sequence):
        raise ValueError("refusing to replace a short-form command")

    expected_payload = _normalise_sha256(expected_payload_sha256)
    if any(_payload_sha256(row) != expected_payload for row in sequence):
        raise ValueError("service sequence payload hash mismatch")

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    output_parts: list[tuple[int, ...]] = [
        source_bits[:expected_start_bit],
    ]
    replacements: list[dict[str, object]] = []
    previous_update = (
        int(rows[start_index - 1]["update"])
        if start_index
        else 0
    )
    for command_index, row in enumerate(sequence, start=start_index):
        update = int(row["update"])
        timing_delta = update - previous_update
        replacement = _idle_bits(timing_delta)
        output_parts.append(replacement)
        replacements.append(
            {
                "command_index": command_index,
                "update": update,
                "source_bit_start": int(row["start"]),
                "source_bit_end": int(row["end"]),
                "source_bits": int(row["end"]) - int(row["start"]),
                "replacement_bits": len(replacement),
                "replacement_timing_delta": timing_delta,
            }
        )
        previous_update = update
    output_parts.append(source_bits[expected_end_bit:])
    output_bits = tuple(
        bit
        for part in output_parts
        for bit in part
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
    if output_meaningful_bits != len(output_bits):
        raise RuntimeError("output meaningful bit count is inconsistent")
    if len(output_rows) != len(rows):
        raise RuntimeError("output scan command count is inconsistent")

    output_command_bits = _bit_tuple(
        output[offset:],
        0,
        output_meaningful_bits,
    )
    for index, (observed, source) in enumerate(
        zip(output_rows, rows, strict=True)
    ):
        target = start_index <= index <= end_index
        expected_command_number = 31 if target else source["command_number"]
        expected_kind_value = "idle" if target else source["kind"]
        expected_payload_value = {} if target else source["payload"]
        expected_short_form = False if target else source["short_form"]
        expected_fields = {
            "update": source["update"],
            "short_form": expected_short_form,
            "command_number": expected_command_number,
            "kind": expected_kind_value,
            "payload": expected_payload_value,
        }
        for key, value in expected_fields.items():
            if observed[key] != value:
                raise RuntimeError(
                    f"output command {index} {key} changed: "
                    f"{observed[key]!r} != {value!r}"
                )
        if not target:
            source_span = source_bits[
                int(source["start"]) : int(source["end"])
            ]
            output_span = output_command_bits[
                int(observed["start"]) : int(observed["end"])
            ]
            if output_span != source_span:
                raise RuntimeError(
                    f"output command {index} raw bits changed"
                )

    provenance = {
        "schema": (
            "zuma-rl.diagnostic-dmo-service-sequence-idle-replacement"
        ),
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
            "start_command_index": start_index,
            "end_command_index": end_index,
            "command_count": count,
            "command_kind": expected_kind,
            "payload_sha256": expected_payload,
            "recorded_updates": list(expected_updates),
            "source_bit_start": expected_start_bit,
            "source_bit_end": expected_end_bit,
            "source_bits": expected_end_bit - expected_start_bit,
            "replacement_kind": "idle",
            "replacement_bits": sum(
                int(item["replacement_bits"]) for item in replacements
            ),
            "replacements": replacements,
            "timeline_preserved": True,
            "command_count_preserved": True,
            "non_target_command_bits_preserved": True,
            "reason": (
                "recording-only service results had no matching playback "
                "service consumers"
            ),
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed_output.commands),
            "length_updates": parsed_output.length_updates,
            "meaningful_command_bits": output_meaningful_bits,
        },
    }
    return output, provenance


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip(), 0) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "updates must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("updates must not be empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--expected-updates", required=True, type=_int_csv)
    parser.add_argument("--expected-start-bit", required=True, type=int)
    parser.add_argument("--expected-end-bit", required=True, type=int)
    parser.add_argument("--expected-kind", required=True)
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
    observed_sha256 = _sha256(source)
    expected_sha256 = _normalise_sha256(args.expected_source_sha256)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "source SHA-256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )

    output, provenance = replace_service_sequence_with_idle(
        source,
        start_index=args.start_index,
        end_index=args.end_index,
        expected_updates=args.expected_updates,
        expected_start_bit=args.expected_start_bit,
        expected_end_bit=args.expected_end_bit,
        expected_kind=args.expected_kind,
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
