"""Insert provenance-bound left-click edges into a diagnostic PopCap DMO.

The source is never overwritten.  Existing command bodies, command updates,
payloads, random seed, markers, and declared demo length are preserved.  New
short-form mouse-button commands are inserted into an interval where the
source left button is known to be up.  Every command timing nibble is rebuilt
and the output is reparsed before exclusive publication.

This produces diagnostic calibration material, not unmodified PC evidence.
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bits(value: int, width: int) -> tuple[int, ...]:
    if width < 1 or not 0 <= value < (1 << width):
        raise ValueError("bit field is out of range")
    return tuple((value >> offset) & 1 for offset in range(width))


def _short_left_button_body(down: bool) -> tuple[int, ...]:
    # short-form=1, short command=1, state, signed 3-bit button number=1.
    return (1, 1, int(down), *_bits(1, 3))


def _command_identity(command: object) -> dict[str, object]:
    result = command.to_dict(include_sensitive_input=True)
    result.pop("sequence")
    return result


def _validate_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    length_updates: int,
) -> tuple[tuple[int, bool], ...]:
    edges: list[tuple[int, bool]] = []
    previous_up = -1
    for down, up in pairs:
        if (
            isinstance(down, bool)
            or isinstance(up, bool)
            or not isinstance(down, int)
            or not isinstance(up, int)
            or down <= previous_up
            or up <= down
            or up > length_updates
        ):
            raise ValueError(
                "click pairs must be ordered, disjoint update intervals"
            )
        edges.extend(((down, True), (up, False)))
        previous_up = up
    if not edges:
        raise ValueError("at least one click pair is required")
    return tuple(edges)


def inject_click_burst(
    data: bytes,
    module: object,
    *,
    pairs: Sequence[tuple[int, int]],
) -> tuple[bytes, dict[str, object]]:
    """Insert exact left-button edges while preserving all source commands."""

    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    source = module.PopCapDemo.from_bytes(data)
    if len(rows) != len(source.commands):
        raise RuntimeError("scanner/parser command counts differ")
    edges = _validate_pairs(pairs, length_updates=length_updates)
    first_update = edges[0][0]
    last_update = edges[-1][0]

    source_left_edges = [
        command
        for command in source.commands
        if (
            command.kind == "mouse_button"
            and int(command.payload["button"]) == 1
        )
    ]
    state_before = False
    for command in source_left_edges:
        if command.update >= first_update:
            break
        state_before = bool(command.payload["down"])
    if state_before:
        raise ValueError("source left button is down before injection")
    if any(
        first_update <= command.update <= last_update
        for command in source_left_edges
    ):
        raise ValueError("source left-button edge overlaps injection")
    if any(
        command.update == update
        for command in source.commands
        for update, _ in edges
    ):
        raise ValueError("injected edge shares an existing command update")

    edge_index = 0
    previous_update = 0
    output_bits: list[int] = []
    combined_kinds: list[tuple[str, int, bool | None]] = []

    def append_command(
        update: int,
        body: Sequence[int],
        *,
        kind: str,
        down: bool | None,
    ) -> None:
        nonlocal previous_update
        delta = update - previous_update
        if not 0 <= delta <= 15:
            raise ValueError(
                "injection leaves a command timing gap outside 4 bits"
            )
        output_bits.extend(_bits(delta, 4))
        output_bits.extend(body)
        previous_update = update
        combined_kinds.append((kind, update, down))

    for row in rows:
        row_update = int(row["update"])
        while edge_index < len(edges) and edges[edge_index][0] < row_update:
            update, down = edges[edge_index]
            append_command(
                update,
                _short_left_button_body(down),
                kind="inserted",
                down=down,
            )
            edge_index += 1
        start = int(row["start"])
        end = int(row["end"])
        append_command(
            row_update,
            _bit_tuple(command_data, start + 4, end),
            kind="source",
            down=None,
        )
    while edge_index < len(edges):
        update, down = edges[edge_index]
        append_command(
            update,
            _short_left_button_body(down),
            kind="inserted",
            down=down,
        )
        edge_index += 1

    output_command_data = _pack_lsb_bits(output_bits)
    output = data[:offset] + output_command_data
    parsed = module.PopCapDemo.from_bytes(output)
    if (
        parsed.version != source.version
        or parsed.random_seed != source.random_seed
        or parsed.product_version != source.product_version
        or parsed.length_updates != source.length_updates
        or parsed.markers != source.markers
        or len(parsed.commands) != len(source.commands) + len(edges)
    ):
        raise RuntimeError("injected DMO top-level contract changed")

    source_index = 0
    inserted_rows: list[dict[str, object]] = []
    for output_index, (command, expected) in enumerate(
        zip(parsed.commands, combined_kinds, strict=True)
    ):
        kind, update, down = expected
        if command.update != update:
            raise RuntimeError("injected DMO update mismatch")
        if kind == "source":
            source_command = source.commands[source_index]
            if _command_identity(command) != _command_identity(source_command):
                raise RuntimeError(
                    f"source command changed at index {source_index}"
                )
            source_index += 1
            continue
        payload = dict(command.payload)
        if (
            command.kind != "mouse_button"
            or command.short_form is not True
            or command.command_number != 1
            or payload.get("button") != 1
            or payload.get("down") is not down
        ):
            raise RuntimeError("inserted mouse-button semantics mismatch")
        inserted_rows.append(
            {
                "output_command_index": output_index,
                "update": update,
                "down": down,
                "x": payload["x"],
                "y": payload["y"],
            }
        )
    if source_index != len(source.commands):
        raise RuntimeError("not all source commands were preserved")

    provenance: dict[str, object] = {
        "schema": "zuma-rl.diagnostic-dmo-click-burst",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(source.commands),
            "length_updates": source.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
        "transformation": {
            "button": "left",
            "pairs": [
                {"down_update": down, "up_update": up}
                for down, up in pairs
            ],
            "inserted_edges": inserted_rows,
            "preserved": [
                "all source command updates and payloads",
                "source command order",
                "framework random seed",
                "markers",
                "declared demo length",
            ],
            "reason": "measure retail continuous-click acceptance phase",
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed.commands),
            "length_updates": parsed.length_updates,
            "meaningful_command_bits": len(output_bits),
        },
    }
    return output, provenance


def _pairs(value: str) -> tuple[tuple[int, int], ...]:
    try:
        result = tuple(
            tuple(int(item) for item in pair.split("-", 1))
            for pair in value.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pairs must use down-up,down-up syntax"
        ) from exc
    if any(len(pair) != 2 for pair in result):
        raise argparse.ArgumentTypeError(
            "pairs must use down-up,down-up syntax"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--pairs", required=True, type=_pairs)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"input DMO does not exist: {args.input}")
    output = args.output
    provenance_path = output.with_name(output.name + ".provenance.json")
    for path in (output, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    source = args.input.read_bytes()
    source_sha256 = _sha256(source)
    if source_sha256 != args.expected_source_sha256.lower():
        raise ValueError("source SHA-256 mismatch")
    module = _load_popcap_dmo_module()
    transformed, provenance = inject_click_burst(
        source,
        module,
        pairs=args.pairs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(transformed)
    with provenance_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            provenance,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(f"output={output}")
    print(f"provenance={provenance_path}")
    print(f"sha256={provenance['output']['sha256']}")
    print(f"bytes={provenance['output']['bytes']}")
    print(f"commands={provenance['output']['commands']}")
    print(f"inserted_edges={len(provenance['transformation']['inserted_edges'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
