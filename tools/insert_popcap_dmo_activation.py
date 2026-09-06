"""Insert one provenance-bound ``activate_app(true)`` DMO command.

A full-screen recording can begin with the application already active and
therefore contain no activation transition.  The retail ``-play`` path starts
in a new window, where external foreground state does not replace the demo
framework's recorded activation state.  This diagnostic tool inserts one
same-update activation command after an exact startup anchor.

The source hash and complete anchor identity are mandatory.  Existing command
bits are copied unchanged, the output is reparsed command-by-command, and the
source is never overwritten.
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert_activation(
    data: bytes,
    module,
    *,
    anchor_index: int,
    expected_anchor_kind: str,
    expected_anchor_update: int,
    expected_anchor_start_bit: int,
    expected_anchor_end_bit: int,
) -> tuple[bytes, dict[str, object]]:
    if anchor_index < 0:
        raise ValueError("anchor index must be non-negative")
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if anchor_index >= len(rows):
        raise ValueError("anchor index does not exist")
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
    following = (
        rows[anchor_index + 1]
        if anchor_index + 1 < len(rows)
        else None
    )
    if (
        following is not None
        and int(following["update"]) < expected_anchor_update
    ):
        raise ValueError("following command update regressed")

    timing_delta = 0
    activation_bits = (
        tuple((timing_delta >> bit) & 1 for bit in range(4))
        + (0,)
        + tuple((1 >> bit) & 1 for bit in range(5))
        + (1,)
    )
    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    output_bits = (
        source_bits[:expected_anchor_end_bit]
        + activation_bits
        + source_bits[expected_anchor_end_bit:]
    )
    output = data[:offset] + _pack_lsb_bits(output_bits)

    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_output = module.PopCapDemo.from_bytes(output)
    if parsed_output.length_updates != parsed_source.length_updates:
        raise RuntimeError("activation insertion changed DMO length")
    if len(parsed_output.commands) != len(parsed_source.commands) + 1:
        raise RuntimeError("activation insertion command count is invalid")
    output_rows, output_meaningful_bits = _scan_commands(
        module,
        output[offset:],
        length_updates,
    )
    inserted_row = {
        "update": expected_anchor_update,
        "short_form": False,
        "command_number": 1,
        "kind": "activate_app",
        "payload": {"active": True},
    }
    expected_rows = (
        rows[: anchor_index + 1]
        + [inserted_row]
        + rows[anchor_index + 1 :]
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
    if output_meaningful_bits != meaningful_bits + len(activation_bits):
        raise RuntimeError("activation insertion bit count is invalid")

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-activation-insertion",
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
            "inserted_command_index": anchor_index + 1,
            "inserted_update": expected_anchor_update,
            "inserted_kind": "activate_app",
            "inserted_payload": {"active": True},
            "inserted_bits": len(activation_bits),
            "existing_command_bits_preserved": True,
            "timeline_preserved": True,
            "reason": (
                "recreate the already-active internal application state "
                "implicit at the start of the full-screen recording"
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
            f"source SHA-256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )

    output, provenance = insert_activation(
        source,
        _load_popcap_dmo_module(),
        anchor_index=args.anchor_index,
        expected_anchor_kind=args.expected_anchor_kind,
        expected_anchor_update=args.expected_anchor_update,
        expected_anchor_start_bit=args.expected_anchor_start_bit,
        expected_anchor_end_bit=args.expected_anchor_end_bit,
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
