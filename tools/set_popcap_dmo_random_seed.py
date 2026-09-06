"""Set one provenance-bound PopCap DMO framework random seed.

The DMO header seed initializes framework RNG state before a replay reaches
the separately controlled board RNG boundary.  This diagnostic tool changes
only that four-byte header field, reparses both artifacts, and proves that
markers, command semantics, command timing, and the declared length are
unchanged.
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

from tools.align_popcap_dmo_startup import _load_popcap_dmo_module
from tools.remove_popcap_dmo_command import _normalise_sha256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def set_random_seed(
    data: bytes,
    module,
    *,
    expected_seed: int,
    new_seed: int,
) -> tuple[bytes, dict[str, object]]:
    if not 0 <= expected_seed <= 0xFFFFFFFF:
        raise ValueError("expected_seed must fit uint32")
    if not 0 <= new_seed <= 0xFFFFFFFF:
        raise ValueError("new_seed must fit uint32")
    if len(data) < 12:
        raise ValueError("DMO is too short")
    observed_seed = struct.unpack_from("<I", data, 8)[0]
    if observed_seed != expected_seed:
        raise ValueError(
            f"DMO random seed mismatch: {observed_seed} != {expected_seed}"
        )

    output_bytes = bytearray(data)
    struct.pack_into("<I", output_bytes, 8, new_seed)
    output = bytes(output_bytes)
    source_demo = module.PopCapDemo.from_bytes(data)
    output_demo = module.PopCapDemo.from_bytes(output)
    if source_demo.random_seed != expected_seed:
        raise RuntimeError("parsed source random seed is inconsistent")
    if output_demo.random_seed != new_seed:
        raise RuntimeError("parsed output random seed is inconsistent")
    for field in (
        "version",
        "product_version",
        "length_updates",
        "markers",
        "commands",
    ):
        if getattr(output_demo, field) != getattr(source_demo, field):
            raise RuntimeError(f"DMO {field} changed with the random seed")
    if output[:8] != data[:8] or output[12:] != data[12:]:
        raise RuntimeError("bytes outside the DMO random seed changed")

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-random-seed",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "random_seed": expected_seed,
            "commands": len(source_demo.commands),
            "length_updates": source_demo.length_updates,
        },
        "transformation": {
            "header_byte_start": 8,
            "header_byte_end": 12,
            "original_random_seed": expected_seed,
            "new_random_seed": new_seed,
            "command_stream_preserved": True,
            "timeline_preserved": True,
            "reason": (
                "reuse a validated framework startup RNG path before the "
                "separately fixed board RNG boundary"
            ),
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "random_seed": new_seed,
            "commands": len(output_demo.commands),
            "length_updates": output_demo.length_updates,
        },
    }
    return output, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--new-seed", required=True, type=int)
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

    output, provenance = set_random_seed(
        source,
        _load_popcap_dmo_module(),
        expected_seed=args.expected_seed,
        new_seed=args.new_seed,
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
