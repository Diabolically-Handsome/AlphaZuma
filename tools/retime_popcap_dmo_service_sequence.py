"""Retime one exact homogeneous PopCap DMO service-result sequence.

Unlike a same-update block, an asynchronous result burst may span several
framework updates.  This diagnostic transformation rewrites only the
four-bit timing field of each bound command and of the immediately following
command.  Command order, payloads, all later absolute updates, and the DMO
length remain unchanged.
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


def _timing_bits(delta: int) -> tuple[int, ...]:
    if not 0 <= delta <= 15:
        raise ValueError("DMO timing delta must fit four bits")
    return tuple((delta >> bit) & 1 for bit in range(4))


def retime_service_sequence(
    data: bytes,
    *,
    start_index: int,
    end_index: int,
    expected_updates: Sequence[int],
    new_updates: Sequence[int],
    expected_start_bit: int,
    expected_end_bit: int,
    expected_kind: str,
    expected_payload_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    """Retime a contiguous sequence while preserving its command payloads."""

    if start_index < 0 or end_index < start_index:
        raise ValueError("service sequence indices are invalid")
    count = end_index - start_index + 1
    if len(expected_updates) != count or len(new_updates) != count:
        raise ValueError("service sequence update count is invalid")
    if any(
        current < previous
        for previous, current in zip(new_updates, new_updates[1:])
    ):
        raise ValueError("new service updates must be nondecreasing")

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
    if tuple(int(row["update"]) for row in sequence) != tuple(
        expected_updates
    ):
        raise ValueError("service sequence update pattern mismatch")
    if any(str(row["kind"]) != expected_kind for row in sequence):
        raise ValueError("service sequence kind mismatch")
    expected_payload = _normalise_sha256(expected_payload_sha256)
    if any(
        _payload_sha256(row) != expected_payload
        for row in sequence
    ):
        raise ValueError("service sequence payload hash mismatch")

    previous_update = (
        int(rows[start_index - 1]["update"])
        if start_index
        else 0
    )
    desired_updates = [int(value) for value in new_updates]
    deltas: list[int] = []
    cursor = previous_update
    for update in desired_updates:
        deltas.append(update - cursor)
        cursor = update
    following_index = end_index + 1
    following_update: int | None = None
    if following_index < len(rows):
        following_update = int(rows[following_index]["update"])
        deltas.append(following_update - cursor)
    for delta in deltas:
        _timing_bits(delta)

    output_bits = list(_bit_tuple(command_data, 0, meaningful_bits))
    for row, delta in zip(sequence, deltas[:count], strict=True):
        start = int(row["start"])
        output_bits[start : start + 4] = _timing_bits(delta)
    if following_update is not None:
        following_start = int(rows[following_index]["start"])
        output_bits[following_start : following_start + 4] = _timing_bits(
            deltas[-1]
        )
    output = data[:offset] + _pack_lsb_bits(output_bits)

    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_output = module.PopCapDemo.from_bytes(output)
    if parsed_output.length_updates != parsed_source.length_updates:
        raise RuntimeError("retiming changed the DMO length")
    if len(parsed_output.commands) != len(parsed_source.commands):
        raise RuntimeError("retiming changed the command count")
    output_rows, output_meaningful_bits = _scan_commands(
        module,
        output[offset:],
        length_updates,
    )
    if output_meaningful_bits != meaningful_bits:
        raise RuntimeError("retiming changed the meaningful bit count")
    for index, (observed, source) in enumerate(
        zip(output_rows, rows, strict=True)
    ):
        expected_update = (
            desired_updates[index - start_index]
            if start_index <= index <= end_index
            else int(source["update"])
        )
        if int(observed["update"]) != expected_update:
            raise RuntimeError(
                f"output command {index} update changed unexpectedly"
            )
        for key in ("short_form", "command_number", "kind", "payload"):
            if observed[key] != source[key]:
                raise RuntimeError(
                    f"output command {index} {key} changed"
                )

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-service-sequence-retiming",
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
            "new_updates": desired_updates,
            "bit_start": expected_start_bit,
            "bit_end": expected_end_bit,
            "following_command_update": following_update,
            "command_payloads_preserved": True,
            "later_timeline_preserved": True,
            "reason": (
                "match the deterministic playback consumer schedule for "
                "an asynchronous service-result sequence"
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
    parser.add_argument("--new-updates", required=True, type=_int_csv)
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

    output, provenance = retime_service_sequence(
        source,
        start_index=args.start_index,
        end_index=args.end_index,
        expected_updates=args.expected_updates,
        new_updates=args.new_updates,
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
