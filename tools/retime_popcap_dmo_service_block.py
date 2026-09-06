"""Retime one exact same-update DMO service-result block.

The transformation changes only the first timing nibble of the selected
block and the following command's timing nibble, preserving every command,
payload, and all later updates.  It is intended for diagnostic replay when
the real consumer is triggered one or more framework updates after the
recorded asynchronous result.  Source and output are parsed and the source
is never modified.
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


def retime_service_block(
    data: bytes,
    *,
    start_index: int,
    end_index: int,
    expected_update: int,
    new_update: int,
    expected_start_bit: int,
    expected_end_bit: int,
    expected_kinds: Sequence[str],
    expected_payload_sha256: Sequence[str],
) -> tuple[bytes, dict[str, object]]:
    """Move a contiguous same-update block while preserving later timing."""

    if start_index < 0 or end_index < start_index:
        raise ValueError("service block indices are invalid")
    count = end_index - start_index + 1
    if len(expected_kinds) != count:
        raise ValueError("expected kind count does not match the block")
    if len(expected_payload_sha256) != count:
        raise ValueError("expected payload hash count does not match the block")
    if new_update <= expected_update:
        raise ValueError("new update must be later than the recorded update")

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
        raise ValueError("service block extends beyond the command stream")
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

    previous_update = (
        int(rows[start_index - 1]["update"])
        if start_index
        else 0
    )
    if new_update > length_updates:
        raise ValueError("new update exceeds the declared DMO length")
    first_original_delta = expected_update - previous_update
    first_replacement_delta = new_update - previous_update
    first_replacement_bits = _timing_bits(first_replacement_delta)

    following_update: int | None = None
    following_original_delta: int | None = None
    following_replacement_delta: int | None = None
    if end_index + 1 < len(rows):
        following = rows[end_index + 1]
        if int(following["start"]) != expected_end_bit:
            raise RuntimeError("following command is not bit-contiguous")
        following_update = int(following["update"])
        if new_update > following_update:
            raise ValueError(
                "new update exceeds the following command update"
            )
        following_original_delta = following_update - expected_update
        following_replacement_delta = following_update - new_update
        following_replacement_bits = _timing_bits(
            following_replacement_delta
        )
    else:
        following_replacement_bits = ()

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    if following_update is None:
        output_bits = (
            source_bits[:expected_start_bit]
            + first_replacement_bits
            + source_bits[expected_start_bit + 4 :]
        )
    else:
        output_bits = (
            source_bits[:expected_start_bit]
            + first_replacement_bits
            + source_bits[expected_start_bit + 4 : expected_end_bit]
            + following_replacement_bits
            + source_bits[expected_end_bit + 4 :]
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
    if output_meaningful_bits != meaningful_bits:
        raise RuntimeError("meaningful command bit count changed")
    if len(output_rows) != len(rows):
        raise RuntimeError("output scan command count changed")
    for index, (observed, source_row) in enumerate(
        zip(output_rows, rows, strict=True)
    ):
        expected_row_update = (
            new_update
            if start_index <= index <= end_index
            else int(source_row["update"])
        )
        if int(observed["update"]) != expected_row_update:
            raise RuntimeError(
                f"output command {index} update changed unexpectedly"
            )
        for key in (
            "short_form",
            "command_number",
            "kind",
            "payload",
        ):
            if observed[key] != source_row[key]:
                raise RuntimeError(
                    f"output command {index} {key} changed"
                )

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-service-block-retiming",
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
            "new_update": new_update,
            "bit_start": expected_start_bit,
            "bit_end": expected_end_bit,
            "first_original_timing_delta": first_original_delta,
            "first_replacement_timing_delta": first_replacement_delta,
            "following_command_update": following_update,
            "following_original_timing_delta": following_original_delta,
            "following_replacement_timing_delta": (
                following_replacement_delta
            ),
            "command_payloads_preserved": True,
            "later_timeline_preserved": True,
            "reason": (
                "playback consumer is triggered after the recorded "
                "asynchronous service-result update"
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
    parser.add_argument("--new-update", required=True, type=int)
    parser.add_argument("--expected-start-bit", required=True, type=int)
    parser.add_argument("--expected-end-bit", required=True, type=int)
    parser.add_argument("--expected-kinds", required=True, type=_csv)
    parser.add_argument(
        "--expected-payload-sha256",
        required=True,
        type=_csv,
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
    output, provenance = retime_service_block(
        source,
        start_index=args.start_index,
        end_index=args.end_index,
        expected_update=args.expected_update,
        new_update=args.new_update,
        expected_start_bit=args.expected_start_bit,
        expected_end_bit=args.expected_end_bit,
        expected_kinds=args.expected_kinds,
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
