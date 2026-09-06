"""Relocate one exact DMO service block behind a later anchor.

The original block is replaced by one same-update idle per command so its
recorded timing remains represented.  The exact original command bits,
including payload bytes, are then copied after a bound later anchor with a
new timing delta.  The following command's timing nibble is compensated so
all later updates remain unchanged.  This output is diagnostic evidence and
the source is never modified.
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


def _idle_bits(delta: int) -> tuple[int, ...]:
    return (
        _timing_bits(delta)
        + (0,)
        + tuple((31 >> bit) & 1 for bit in range(5))
    )


def relocate_service_block(
    data: bytes,
    *,
    start_index: int,
    end_index: int,
    expected_update: int,
    expected_start_bit: int,
    expected_end_bit: int,
    expected_kinds: Sequence[str],
    expected_payload_sha256: Sequence[str],
    anchor_index: int,
    expected_anchor_kind: str,
    expected_anchor_update: int,
    expected_anchor_start_bit: int,
    expected_anchor_end_bit: int,
    new_update: int,
) -> tuple[bytes, dict[str, object]]:
    """Replace, copy, and retime a service block after a later anchor."""

    if (
        start_index < 0
        or end_index < start_index
        or anchor_index <= end_index
    ):
        raise ValueError("block and anchor indices are invalid")
    count = end_index - start_index + 1
    if len(expected_kinds) != count:
        raise ValueError("expected kind count does not match the block")
    if len(expected_payload_sha256) != count:
        raise ValueError("expected payload hash count does not match the block")
    if new_update <= expected_anchor_update:
        raise ValueError("new update must be later than the anchor update")

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
        raise ValueError("anchor extends beyond the command stream")

    block = rows[start_index : end_index + 1]
    if (
        int(block[0]["start"]) != expected_start_bit
        or int(block[-1]["end"]) != expected_end_bit
    ):
        raise ValueError("service block bit span mismatch")
    if any(int(row["update"]) != expected_update for row in block):
        raise ValueError("service block is not at the expected update")
    if tuple(str(row["kind"]) for row in block) != tuple(expected_kinds):
        raise ValueError("service block kind sequence mismatch")
    observed_payload_hashes = tuple(
        _payload_sha256(row) for row in block
    )
    normalised_payload_hashes = tuple(
        _normalise_sha256(value)
        for value in expected_payload_sha256
    )
    if observed_payload_hashes != normalised_payload_hashes:
        raise ValueError("service block payload hash sequence mismatch")

    anchor = rows[anchor_index]
    expected_anchor = {
        "kind": expected_anchor_kind,
        "update": expected_anchor_update,
        "start": expected_anchor_start_bit,
        "end": expected_anchor_end_bit,
    }
    for key, value in expected_anchor.items():
        if anchor[key] != value:
            raise ValueError(
                f"anchor {anchor_index} {key} mismatch: "
                f"{anchor[key]!r} != {value!r}"
            )
    if expected_anchor_start_bit < expected_end_bit:
        raise ValueError("anchor overlaps the service block")
    if new_update > length_updates:
        raise ValueError("new update exceeds the declared DMO length")

    previous_update = (
        int(rows[start_index - 1]["update"])
        if start_index
        else 0
    )
    placeholder_first_delta = expected_update - previous_update
    placeholder_bits = _idle_bits(placeholder_first_delta)
    placeholder_bits += _idle_bits(0) * (count - 1)

    relocated_first_delta = new_update - expected_anchor_update
    relocated_first_timing = _timing_bits(relocated_first_delta)

    following_update: int | None = None
    following_original_delta: int | None = None
    following_replacement_delta: int | None = None
    if anchor_index + 1 < len(rows):
        following = rows[anchor_index + 1]
        if int(following["start"]) != expected_anchor_end_bit:
            raise RuntimeError("following command is not bit-contiguous")
        following_update = int(following["update"])
        if new_update > following_update:
            raise ValueError(
                "new update exceeds the following command update"
            )
        following_original_delta = (
            following_update - expected_anchor_update
        )
        following_replacement_delta = following_update - new_update
        following_replacement_timing = _timing_bits(
            following_replacement_delta
        )
    else:
        following_replacement_timing = ()

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    relocated_bits = (
        relocated_first_timing
        + source_bits[expected_start_bit + 4 : expected_end_bit]
    )
    if following_update is None:
        source_tail = source_bits[expected_anchor_end_bit:]
    else:
        source_tail = (
            following_replacement_timing
            + source_bits[expected_anchor_end_bit + 4 :]
        )
    output_bits = (
        source_bits[:expected_start_bit]
        + placeholder_bits
        + source_bits[expected_end_bit:expected_anchor_end_bit]
        + relocated_bits
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

    placeholder_rows = [
        {
            "update": expected_update,
            "short_form": False,
            "command_number": 31,
            "kind": "idle",
            "payload": {},
        }
        for _ in range(count)
    ]
    relocated_rows = [
        {
            **dict(row),
            "update": new_update,
        }
        for row in block
    ]
    expected_rows = (
        rows[:start_index]
        + placeholder_rows
        + rows[end_index + 1 : anchor_index + 1]
        + relocated_rows
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
    expected_meaningful_bits = (
        meaningful_bits
        - (expected_end_bit - expected_start_bit)
        + len(placeholder_bits)
        + len(relocated_bits)
    )
    if output_meaningful_bits != expected_meaningful_bits:
        raise RuntimeError("output meaningful bit count is inconsistent")

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-service-block-relocation",
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
            "command_kinds": list(expected_kinds),
            "payload_sha256": list(normalised_payload_hashes),
            "recorded_update": expected_update,
            "recorded_bit_start": expected_start_bit,
            "recorded_bit_end": expected_end_bit,
            "placeholder_kind": "idle",
            "placeholder_count": count,
            "placeholder_first_timing_delta": (
                placeholder_first_delta
            ),
            "anchor_command_index": anchor_index,
            "anchor_kind": expected_anchor_kind,
            "anchor_update": expected_anchor_update,
            "anchor_bit_start": expected_anchor_start_bit,
            "anchor_bit_end": expected_anchor_end_bit,
            "new_update": new_update,
            "relocated_first_timing_delta": relocated_first_delta,
            "following_command_update": following_update,
            "following_original_timing_delta": following_original_delta,
            "following_replacement_timing_delta": (
                following_replacement_delta
            ),
            "payload_bits_copied_exactly": True,
            "later_timeline_preserved": True,
            "reason": (
                "playback service consumer is triggered after the recorded "
                "result block and after the selected anchor"
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


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(","))
    if not result or any(not item for item in result):
        raise argparse.ArgumentTypeError("CSV values must be non-empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--expected-update", required=True, type=int)
    parser.add_argument("--expected-start-bit", required=True, type=int)
    parser.add_argument("--expected-end-bit", required=True, type=int)
    parser.add_argument("--expected-kinds", required=True, type=_csv)
    parser.add_argument(
        "--expected-payload-sha256",
        required=True,
        type=_csv,
    )
    parser.add_argument("--anchor-index", required=True, type=int)
    parser.add_argument("--expected-anchor-kind", required=True)
    parser.add_argument("--expected-anchor-update", required=True, type=int)
    parser.add_argument(
        "--expected-anchor-start-bit",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--expected-anchor-end-bit",
        required=True,
        type=int,
    )
    parser.add_argument("--new-update", required=True, type=int)
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
    output, provenance = relocate_service_block(
        source,
        start_index=args.start_index,
        end_index=args.end_index,
        expected_update=args.expected_update,
        expected_start_bit=args.expected_start_bit,
        expected_end_bit=args.expected_end_bit,
        expected_kinds=args.expected_kinds,
        expected_payload_sha256=args.expected_payload_sha256,
        anchor_index=args.anchor_index,
        expected_anchor_kind=args.expected_anchor_kind,
        expected_anchor_update=args.expected_anchor_update,
        expected_anchor_start_bit=args.expected_anchor_start_bit,
        expected_anchor_end_bit=args.expected_anchor_end_bit,
        new_update=args.new_update,
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
