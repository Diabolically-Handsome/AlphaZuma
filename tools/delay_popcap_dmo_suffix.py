"""Delay one provenance-bound PopCap DMO command suffix.

Asynchronous framework work can finish a few updates earlier while recording
than its deterministic replay consumer does.  This diagnostic transformation
inserts one idle command at an exact bound command boundary, delays the
following command and every later command by a declared number of updates,
and increases the declared demo length by the same amount.

The source hash and complete adjacent-command identities are mandatory.  No
source payload bits are changed: only the following command's four timing
bits, the inserted idle bits, and the declared length differ.
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
from tools.remove_popcap_dmo_command import _normalise_sha256


IDLE_COMMAND_NUMBER = 31


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timing_bits(delta: int) -> tuple[int, ...]:
    if not 0 <= delta <= 15:
        raise ValueError("timing delta must fit four bits")
    return tuple((delta >> bit) & 1 for bit in range(4))


def _idle_bits(delta: int) -> tuple[int, ...]:
    return (
        _timing_bits(delta)
        + (0,)
        + tuple(
            (IDLE_COMMAND_NUMBER >> bit) & 1
            for bit in range(5)
        )
    )


def delay_suffix(
    data: bytes,
    module,
    *,
    anchor_index: int,
    expected_anchor_kind: str,
    expected_anchor_update: int,
    expected_anchor_start_bit: int,
    expected_anchor_end_bit: int,
    target_index: int,
    expected_target_kind: str,
    expected_target_update: int,
    expected_target_start_bit: int,
    expected_target_end_bit: int,
    delay_updates: int,
) -> tuple[bytes, dict[str, object]]:
    """Insert an idle at ``target`` and delay that command's suffix."""

    if anchor_index < 0 or target_index != anchor_index + 1:
        raise ValueError("target must immediately follow a valid anchor")
    if not 1 <= delay_updates <= 15:
        raise ValueError("delay_updates must be between 1 and 15")

    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if target_index >= len(rows):
        raise ValueError("target command does not exist")

    anchor = rows[anchor_index]
    target = rows[target_index]
    expected_anchor = {
        "kind": expected_anchor_kind,
        "update": expected_anchor_update,
        "start": expected_anchor_start_bit,
        "end": expected_anchor_end_bit,
    }
    expected_target = {
        "kind": expected_target_kind,
        "update": expected_target_update,
        "start": expected_target_start_bit,
        "end": expected_target_end_bit,
    }
    for label, observed, expected in (
        ("anchor", anchor, expected_anchor),
        ("target", target, expected_target),
    ):
        for key, value in expected.items():
            if observed[key] != value:
                raise ValueError(
                    f"{label} {key} mismatch: "
                    f"{observed[key]!r} != {value!r}"
                )

    idle_delta = expected_target_update - expected_anchor_update
    if not 0 <= idle_delta <= 15:
        raise ValueError("anchor-to-target timing does not fit four bits")
    if expected_target_start_bit != expected_anchor_end_bit:
        raise ValueError("anchor and target bit spans are not adjacent")
    new_length = length_updates + delay_updates
    if new_length > 0xFFFFFFFF:
        raise ValueError("delayed DMO length exceeds uint32")

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    output_bits = (
        source_bits[:expected_target_start_bit]
        + _idle_bits(idle_delta)
        + _timing_bits(delay_updates)
        + source_bits[expected_target_start_bit + 4 :]
    )
    header = bytearray(data[:offset])
    struct.pack_into("<I", header, offset - 4, new_length)
    output = bytes(header) + _pack_lsb_bits(output_bits)

    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_output = module.PopCapDemo.from_bytes(output)
    if parsed_output.length_updates != new_length:
        raise RuntimeError("delayed DMO length is inconsistent")
    if len(parsed_output.commands) != len(parsed_source.commands) + 1:
        raise RuntimeError("delayed DMO command count is inconsistent")
    output_rows, output_meaningful_bits = _scan_commands(
        module,
        output[offset:],
        new_length,
    )
    inserted_row = {
        "update": expected_target_update,
        "short_form": False,
        "command_number": IDLE_COMMAND_NUMBER,
        "kind": "idle",
        "payload": {},
    }
    delayed_rows: list[dict[str, object]] = []
    for row in rows[target_index:]:
        delayed = dict(row)
        delayed["update"] = int(delayed["update"]) + delay_updates
        delayed_rows.append(delayed)
    expected_rows = (
        rows[:target_index]
        + [inserted_row]
        + delayed_rows
    )
    for index, (observed, expected) in enumerate(
        zip(output_rows, expected_rows, strict=True)
    ):
        for key in (
            "update",
            "short_form",
            "command_number",
            "kind",
            "payload",
        ):
            if observed[key] != expected[key]:
                raise RuntimeError(
                    f"output command {index} {key} changed: "
                    f"{observed[key]!r} != {expected[key]!r}"
                )
    if output_meaningful_bits != meaningful_bits + 10:
        raise RuntimeError("delayed DMO bit count is inconsistent")

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-suffix-delay",
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
            "anchor_command_index": anchor_index,
            "anchor_kind": expected_anchor_kind,
            "anchor_update": expected_anchor_update,
            "anchor_bit_start": expected_anchor_start_bit,
            "anchor_bit_end": expected_anchor_end_bit,
            "target_command_index": target_index,
            "target_kind": expected_target_kind,
            "target_original_update": expected_target_update,
            "target_bit_start": expected_target_start_bit,
            "target_bit_end": expected_target_end_bit,
            "inserted_command_index": target_index,
            "inserted_kind": "idle",
            "inserted_update": expected_target_update,
            "delay_updates": delay_updates,
            "source_payload_bits_preserved": True,
            "reason": (
                "delay an asynchronous service-result suffix until the "
                "deterministic replay consumer is ready"
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
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--expected-target-kind", required=True)
    parser.add_argument("--expected-target-update", required=True, type=int)
    parser.add_argument("--expected-target-start-bit", required=True, type=int)
    parser.add_argument("--expected-target-end-bit", required=True, type=int)
    parser.add_argument("--delay-updates", required=True, type=int)
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

    output, provenance = delay_suffix(
        source,
        _load_popcap_dmo_module(),
        anchor_index=args.anchor_index,
        expected_anchor_kind=args.expected_anchor_kind,
        expected_anchor_update=args.expected_anchor_update,
        expected_anchor_start_bit=args.expected_anchor_start_bit,
        expected_anchor_end_bit=args.expected_anchor_end_bit,
        target_index=args.target_index,
        expected_target_kind=args.expected_target_kind,
        expected_target_update=args.expected_target_update,
        expected_target_start_bit=args.expected_target_start_bit,
        expected_target_end_bit=args.expected_target_end_bit,
        delay_updates=args.delay_updates,
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
