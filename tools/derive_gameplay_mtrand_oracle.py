"""Derive one gameplay-return MTRand oracle from a full retail call trace."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.popcap_global_mtrand_call_oracle import (
    GlobalMTRandCallOracle,
    load_global_mtrand_call_oracle,
)


SCHEMA = "zuma-rl.pc-gameplay-mtrand-call-trace"
VERSION = 1
DEFAULT_GAMEPLAY_RETURN = 0x004B5ADF


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def derive_gameplay_oracle(
    source_path: Path,
    *,
    seed: int,
    start_after_update: int,
    end_at_update: int,
    caller: int = DEFAULT_GAMEPLAY_RETURN,
    bootstrap_source_orders: Iterable[int] = (),
    maximum_draws: int = 100_000,
) -> dict[str, Any]:
    """Validate a full oracle and select one exact call per update."""

    if caller <= 0:
        raise ValueError("gameplay return caller must be positive")
    source_path = source_path.resolve()
    oracle: GlobalMTRandCallOracle = load_global_mtrand_call_oracle(
        source_path,
        seed=seed,
        start_after_update=start_after_update,
        end_at_update=end_at_update,
        maximum_draws=maximum_draws,
    )
    bootstrap_orders = tuple(bootstrap_source_orders)
    if (
        any(
            isinstance(order, bool) or not isinstance(order, int) or order < 0
            for order in bootstrap_orders
        )
        or tuple(sorted(set(bootstrap_orders))) != bootstrap_orders
    ):
        raise ValueError(
            "bootstrap source orders must be canonical distinct integers"
        )
    gameplay = [
        entry for entry in oracle.entries if entry.caller == caller
    ]
    if not gameplay:
        raise ValueError("full trace has no gameplay return calls")
    counts = Counter(entry.framework_update for entry in gameplay)
    duplicate_updates = sorted(
        update for update, count in counts.items() if count != 1
    )
    updates = sorted(counts)
    expected_updates = list(range(updates[0], updates[-1] + 1))
    if duplicate_updates or updates != expected_updates:
        raise ValueError(
            "gameplay return calls are not exactly one contiguous call "
            "per update"
        )
    bootstrap_order_set = set(bootstrap_orders)
    bootstrap = [
        entry
        for entry in oracle.entries
        if entry.source_order in bootstrap_order_set
    ]
    if (
        len(bootstrap) != len(bootstrap_orders)
        or [entry.source_order for entry in bootstrap]
        != list(bootstrap_orders)
        or (
            bootstrap
            and bootstrap[-1].source_order >= gameplay[0].source_order
        )
    ):
        raise ValueError(
            "bootstrap source orders are missing, reordered, or overlap "
            "the gameplay suffix"
        )
    selected = bootstrap + gameplay
    calls = [
        {
            "order": order,
            "source_order": entry.source_order,
            "framework_update": entry.framework_update,
            "native_game_time": entry.native_game_time,
            "score": entry.score,
            "score_target": entry.score_target,
            "caller": entry.caller,
            "caller_hex": f"0x{entry.caller:08x}",
            "output": entry.output,
            "pre_index": entry.pre_index,
            "pre_state_sha256": entry.pre_state_sha256,
            "post_index": entry.post_index,
            "post_state_sha256": entry.post_state_sha256,
            "pre_draw_count": entry.pre_draw_count,
            "post_draw_count": entry.post_draw_count,
        }
        for order, entry in enumerate(selected)
    ]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "PASS",
        "failure": None,
        "classification": "derived-read-only-gameplay-mtrand-oracle",
        "breakpoint_kind": "selected_gameplay_return_addresses",
        "process_id": oracle.source_process_id,
        "main_thread_id": oracle.source_main_thread_id,
        "runtime_executable_sha256": oracle.runtime_executable_sha256,
        "process_memory_writes": 0,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "call_count": len(calls),
        "calls": calls,
        "provenance": {
            "source_path": str(source_path),
            "source_sha256": _sha256_path(source_path),
            "source_schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
            "source_breakpoint_kind": "global_wrapper_entry",
            "source_process_id": oracle.source_process_id,
            "source_main_thread_id": oracle.source_main_thread_id,
            "source_oracle_semantic_sha256": oracle.semantic_sha256,
            "seed": seed,
            "start_after_update": start_after_update,
            "end_at_update": end_at_update,
            "selected_caller": caller,
            "selected_caller_hex": f"0x{caller:08x}",
            "bootstrap_source_orders": list(bootstrap_orders),
            "bootstrap_callers": [entry.caller for entry in bootstrap],
            "bootstrap_caller_hex": [
                f"0x{entry.caller:08x}" for entry in bootstrap
            ],
            "first_selected_update": updates[0],
            "last_selected_update": updates[-1],
            "contiguous_update_count": len(updates),
            "maximum_draws": maximum_draws,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--seed",
        required=True,
        type=lambda value: int(value, 0),
    )
    parser.add_argument("--start-after-update", required=True, type=int)
    parser.add_argument("--end-at-update", required=True, type=int)
    parser.add_argument(
        "--caller",
        type=lambda value: int(value, 0),
        default=DEFAULT_GAMEPLAY_RETURN,
    )
    parser.add_argument(
        "--bootstrap-source-order",
        action="append",
        type=int,
        default=[],
        dest="bootstrap_source_orders",
        help=(
            "Include one earlier gameplay-relevant source call by exact "
            "full-trace order; repeat in ascending order."
        ),
    )
    parser.add_argument("--maximum-draws", type=int, default=100_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    report = derive_gameplay_oracle(
        args.source,
        seed=args.seed,
        start_after_update=args.start_after_update,
        end_at_update=args.end_at_update,
        caller=args.caller,
        bootstrap_source_orders=args.bootstrap_source_orders,
        maximum_draws=args.maximum_draws,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)
    print(
        json.dumps(
            {
                "call_count": report["call_count"],
                "provenance": report["provenance"],
                "sha256": _sha256_path(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
