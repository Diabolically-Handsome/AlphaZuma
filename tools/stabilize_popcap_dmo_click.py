"""Stabilize one replayed click without changing its timing or later path.

Strict debugger tracing stretches wall-clock time between recorded framework
updates.  A small recorded pointer drift while a button is held can therefore
be interpreted as a drag instead of a click.  This diagnostic transformation
zeros only the short mouse deltas between one bound down/up pair, then changes
the first following short mouse move so the original trajectory is restored
at that command.

The source hash, command indices, updates, and coordinates are mandatory.
Command identities, updates, button states, non-mouse payloads, total bit
length, and every payload after the restoration command must remain exact.
The source is never overwritten.
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
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)
from tools.remove_popcap_dmo_command import _normalise_sha256
from tools.transform_popcap_dmo_mouse_space import _write_lsb_bits


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain_payload(command) -> dict[str, object]:
    return dict(command.payload)


def stabilize_click(
    data: bytes,
    module,
    *,
    down_index: int,
    up_index: int,
    restore_index: int,
    expected_down_update: int,
    expected_up_update: int,
    expected_restore_update: int,
    expected_down_x: int,
    expected_down_y: int,
    expected_up_x: int,
    expected_up_y: int,
    expected_restore_x: int,
    expected_restore_y: int,
    expected_button: int = 1,
) -> tuple[bytes, dict[str, object]]:
    if not 0 <= down_index < up_index < restore_index:
        raise ValueError("click and restoration indices are invalid")

    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    parsed_source = module.PopCapDemo.from_bytes(data)
    commands = parsed_source.commands
    if restore_index >= len(commands) or len(rows) != len(commands):
        raise ValueError("click or restoration index does not exist")

    down = commands[down_index]
    up = commands[up_index]
    restore = commands[restore_index]
    expected_commands = (
        (
            down,
            down_index,
            "mouse_button",
            expected_down_update,
            expected_down_x,
            expected_down_y,
        ),
        (
            up,
            up_index,
            "mouse_button",
            expected_up_update,
            expected_up_x,
            expected_up_y,
        ),
        (
            restore,
            restore_index,
            "mouse_move",
            expected_restore_update,
            expected_restore_x,
            expected_restore_y,
        ),
    )
    for command, index, kind, update, x, y in expected_commands:
        payload = _plain_payload(command)
        if (
            command.kind != kind
            or command.update != update
            or int(payload["x"]) != x
            or int(payload["y"]) != y
        ):
            raise ValueError(f"command {index} identity mismatch")
    if (
        not down.short_form
        or not up.short_form
        or int(down.payload["button"]) != expected_button
        or int(up.payload["button"]) != expected_button
        or not bool(down.payload["down"])
        or bool(up.payload["down"])
    ):
        raise ValueError("bound commands are not one short down/up pair")
    if not restore.short_form:
        raise ValueError("restoration command must be a short mouse move")

    held_move_indices: list[int] = []
    for index in range(down_index + 1, up_index):
        command = commands[index]
        if command.kind == "mouse_move":
            if not command.short_form:
                raise ValueError(
                    "held interval contains a long mouse move"
                )
            held_move_indices.append(index)
        elif command.kind != "idle":
            raise ValueError(
                "held interval contains a non-mouse effectful command"
            )
    if not held_move_indices:
        raise ValueError("held interval contains no mouse drift")
    for index in range(up_index + 1, restore_index):
        if commands[index].kind != "idle":
            raise ValueError(
                "pre-restoration interval is not idle-only"
            )

    restore_delta_x = expected_restore_x - expected_down_x
    restore_delta_y = expected_restore_y - expected_down_y
    for value in (restore_delta_x, restore_delta_y):
        if not -32 <= value <= 31:
            raise ValueError(
                "restoration delta does not fit a short mouse command"
            )

    transformed_command_data = bytearray(command_data)
    for index in held_move_indices:
        start = int(rows[index]["start"])
        _write_lsb_bits(
            transformed_command_data,
            start + 6,
            6,
            0,
            signed=True,
        )
        _write_lsb_bits(
            transformed_command_data,
            start + 12,
            6,
            0,
            signed=True,
        )
    restore_start = int(rows[restore_index]["start"])
    _write_lsb_bits(
        transformed_command_data,
        restore_start + 6,
        6,
        restore_delta_x,
        signed=True,
    )
    _write_lsb_bits(
        transformed_command_data,
        restore_start + 12,
        6,
        restore_delta_y,
        signed=True,
    )
    output = data[:offset] + bytes(transformed_command_data)
    if len(output) != len(data):
        raise RuntimeError("click stabilization changed DMO byte length")

    parsed_output = module.PopCapDemo.from_bytes(output)
    if (
        parsed_output.length_updates != parsed_source.length_updates
        or len(parsed_output.commands) != len(commands)
    ):
        raise RuntimeError("click stabilization changed DMO structure")

    for index, (source_command, output_command) in enumerate(
        zip(commands, parsed_output.commands, strict=True)
    ):
        source_identity = (
            source_command.update,
            source_command.kind,
            source_command.short_form,
            source_command.command_number,
        )
        output_identity = (
            output_command.update,
            output_command.kind,
            output_command.short_form,
            output_command.command_number,
        )
        if output_identity != source_identity:
            raise RuntimeError(
                f"command identity changed at command {index}"
            )
        source_payload = _plain_payload(source_command)
        output_payload = _plain_payload(output_command)
        if index in held_move_indices:
            if output_payload != {
                "x": expected_down_x,
                "y": expected_down_y,
                "delta_x": 0,
                "delta_y": 0,
            }:
                raise RuntimeError(
                    f"held mouse command {index} was not stabilized"
                )
        elif index == up_index:
            expected_payload = {
                **source_payload,
                "x": expected_down_x,
                "y": expected_down_y,
            }
            if output_payload != expected_payload:
                raise RuntimeError("button-up coordinate was not stabilized")
        elif index == restore_index:
            expected_payload = {
                **source_payload,
                "delta_x": restore_delta_x,
                "delta_y": restore_delta_y,
            }
            if output_payload != expected_payload:
                raise RuntimeError("original mouse path was not restored")
        elif output_payload != source_payload:
            raise RuntimeError(
                f"unexpected payload change at command {index}"
            )

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-click-stabilization",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(commands),
            "length_updates": parsed_source.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
        "transformation": {
            "down_command_index": down_index,
            "down_update": expected_down_update,
            "up_command_index": up_index,
            "up_update": expected_up_update,
            "button": expected_button,
            "stable_coordinate": [expected_down_x, expected_down_y],
            "recorded_up_coordinate": [expected_up_x, expected_up_y],
            "held_mouse_command_indices": held_move_indices,
            "restore_command_index": restore_index,
            "restore_update": expected_restore_update,
            "restore_coordinate": [
                expected_restore_x,
                expected_restore_y,
            ],
            "replacement_restore_delta": [
                restore_delta_x,
                restore_delta_y,
            ],
            "button_timing_preserved": True,
            "post_restoration_trajectory_preserved": True,
            "command_bit_length_preserved": True,
            "reason": (
                "prevent debugger-stretched pointer drift from turning "
                "the recorded click into a drag"
            ),
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed_output.commands),
            "length_updates": parsed_output.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
    }
    return output, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--down-index", required=True, type=int)
    parser.add_argument("--up-index", required=True, type=int)
    parser.add_argument("--restore-index", required=True, type=int)
    parser.add_argument("--expected-down-update", required=True, type=int)
    parser.add_argument("--expected-up-update", required=True, type=int)
    parser.add_argument("--expected-restore-update", required=True, type=int)
    parser.add_argument("--expected-down-x", required=True, type=int)
    parser.add_argument("--expected-down-y", required=True, type=int)
    parser.add_argument("--expected-up-x", required=True, type=int)
    parser.add_argument("--expected-up-y", required=True, type=int)
    parser.add_argument("--expected-restore-x", required=True, type=int)
    parser.add_argument("--expected-restore-y", required=True, type=int)
    parser.add_argument("--expected-button", type=int, default=1)
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
    output, provenance = stabilize_click(
        source,
        _load_popcap_dmo_module(),
        down_index=args.down_index,
        up_index=args.up_index,
        restore_index=args.restore_index,
        expected_down_update=args.expected_down_update,
        expected_up_update=args.expected_up_update,
        expected_restore_update=args.expected_restore_update,
        expected_down_x=args.expected_down_x,
        expected_down_y=args.expected_down_y,
        expected_up_x=args.expected_up_x,
        expected_up_y=args.expected_up_y,
        expected_restore_x=args.expected_restore_x,
        expected_restore_y=args.expected_restore_y,
        expected_button=args.expected_button,
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
