"""Observe independent retail startup RNG seeds with exact state rollback."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_golden_v4 import (
    EXPECTED_RUNTIME_SHA256,
    _require_game_stopped,
    _terminate_exact_runtime,
    _wait_game_stopped,
    _write_exclusive,
)
from tools.launch_fixed_seed_replay import launch_fixed_seed_replay
from tools.launch_popcap_replay import process_ids_by_name
from tools.pc_state_transaction import (
    capture_state,
    overlay_snapshot_files,
    restore_state,
    write_snapshot_exclusive,
)
from tools.trace_popcap_demo_commands import (
    DEFAULT_BOARD_RESEED_CALL,
    DEFAULT_PREPARE_DEMO_COMMAND,
    _offline_rows,
    trace_demo_commands,
    wait_for_update_before_attach,
)
from zuma_rl.popcap_dmo import PopCapDemo


DEFAULT_DMO = Path(
    r"D:\ZumaGolden\session-20260728-203153"
    r"\native-windowed-record-001-aligned-diagnostic\input.dmo"
)
DEFAULT_PRESTATE = Path(
    r"D:\ZumaGolden\session-20260728-203153"
    r"\native-windowed-record-001-prestate-reconstructed"
)
DEFAULT_RUNTIME = Path(
    r"D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe"
)
DEFAULT_CHANGEDIR = Path(
    r"D:\SteamLibrary\steamapps\common\Zuma's Revenge"
)
OVERLAY_NAMES = ("adv_in_game2.sav", "user2.dat", "users.dat")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("Windows is required")
    if (
        args.launch_timeout <= 0
        or args.attach_timeout <= 0
        or args.trace_timeout <= 0
        or args.service_wait_timeout <= 0
        or args.progress_every_updates <= 0
    ):
        raise ValueError(
            "diagnostic timeouts and progress interval must be positive"
        )
    output_root = args.output_root.resolve()
    if output_root.exists() or not output_root.parent.is_dir():
        raise ValueError("output_root must be a new directory")
    dmo_path = args.dmo.resolve()
    prestate_root = args.prestate_dir.resolve()
    runtime = args.runtime_executable.resolve()
    changedir = args.changedir.resolve()
    if (
        not dmo_path.is_file()
        or not prestate_root.is_dir()
        or not runtime.is_file()
        or not changedir.is_dir()
        or _sha256(runtime) != EXPECTED_RUNTIME_SHA256
    ):
        raise ValueError("diagnostic input identity is invalid")
    demo = PopCapDemo.read(dmo_path)
    seed = demo.random_seed if args.seed is None else args.seed
    offline_rows = _offline_rows(dmo_path)

    _require_game_stopped()
    output_root.mkdir()
    nonce = args.session_nonce
    host = capture_state(session_nonce=nonce, phase="host-pre")
    write_snapshot_exclusive(host, output_root / "host-pre.json")
    pre = overlay_snapshot_files(
        host,
        tuple(
            (name, prestate_root / name) for name in OVERLAY_NAMES
        ),
        phase="diagnostic-pre",
        captured_perf_counter_ns=time.perf_counter_ns(),
    )
    write_snapshot_exclusive(pre, output_root / "diagnostic-pre.json")

    runs: list[dict[str, Any]] = []
    primary_error: BaseException | None = None
    try:
        for run_id in ("r1", "r2"):
            restored = restore_state(pre)
            if restored.state_root != pre.state_root:
                raise RuntimeError("diagnostic prestate restore mismatch")
            evidence = launch_fixed_seed_replay(
                runtime_executable=runtime,
                changedir=changedir,
                dmo=dmo_path,
                app_id=args.app_id,
                seed=seed,
                timeout=args.launch_timeout,
            )
            attach_gate: dict[str, int] | None = None
            if args.attach_at_update:
                base, observed_update = wait_for_update_before_attach(
                    pid=evidence.process_id,
                    target_update=args.attach_at_update,
                    timeout=args.attach_timeout,
                )
                attach_gate = {
                    "requested_update": args.attach_at_update,
                    "observed_update": observed_update,
                    "framework_base": base,
                }
            board_observations: list[dict[str, Any]] = []
            trace_result = trace_demo_commands(
                pid=evidence.process_id,
                executable=runtime,
                address=DEFAULT_PREPARE_DEMO_COMMAND,
                maximum_hits=100_000,
                timeout=args.trace_timeout,
                offline_rows=offline_rows,
                broker_service_blocks=True,
                service_wait_timeout=args.service_wait_timeout,
                quiet_nonservice=True,
                progress_every_updates=args.progress_every_updates,
                board_seed_address=args.board_seed_address,
                board_seed_override=args.board_seed,
                global_rng_seed_override=args.global_rng_seed,
                thread_crt_rng_seed_override=args.thread_crt_rng_seed,
                board_seed_observations=board_observations,
                stop_after_board_seed=True,
                allow_pre_stream_commands=True,
            )
            if trace_result.failure_count:
                raise RuntimeError(
                    "strict DMO trace failed before the board seed"
                )
            if trace_result.exited:
                raise RuntimeError(
                    "runtime exited before the board seed diagnostic stopped"
                )
            if len(board_observations) != 1:
                raise RuntimeError(
                    "expected exactly one board seed observation, got "
                    f"{len(board_observations)}"
                )
            runs.append(
                {
                    "run_id": run_id,
                    "startup_rng": evidence.to_dict(),
                    "attach_gate": attach_gate,
                    "board_rng": board_observations[0],
                    "trace": {
                        **asdict(trace_result),
                        "failure_count": trace_result.failure_count,
                    },
                }
            )
            _terminate_exact_runtime(evidence.process_id, runtime)
            _wait_game_stopped()
    except BaseException as error:
        primary_error = error
    finally:
        for process_id in process_ids_by_name(runtime.name):
            _terminate_exact_runtime(process_id, runtime)
        _wait_game_stopped()
        restored_host = restore_state(host)
        if restored_host.state_root != host.state_root:
            raise RuntimeError("host state restore mismatch")
        host_after = capture_state(
            session_nonce=nonce,
            phase="host-restored",
        )
        write_snapshot_exclusive(
            host_after,
            output_root / "host-restored.json",
        )
        if host_after.state_root != host.state_root:
            raise RuntimeError("host state verification mismatch")
    if primary_error is not None:
        raise primary_error

    observed_board_seeds = tuple(
        int(run["board_rng"]["observed_seed"]) for run in runs
    )
    effective_board_seeds = tuple(
        int(run["board_rng"]["effective_seed"]) for run in runs
    )
    thread_crt_states_before = tuple(
        int(run["board_rng"]["thread_crt_rng_state_before"])
        for run in runs
        if "thread_crt_rng_state_before" in run["board_rng"]
    )
    thread_crt_states_after = tuple(
        int(run["board_rng"]["thread_crt_rng_state_after"])
        for run in runs
        if "thread_crt_rng_state_after" in run["board_rng"]
    )
    result = {
        "schema": "zuma-rl.popcap-startup-rng-diagnostic",
        "version": 3,
        "session_nonce": nonce,
        "dmo_sha256": _sha256(dmo_path),
        "runtime_sha256": _sha256(runtime),
        "framework_seed": seed,
        "attach_at_update": args.attach_at_update,
        "board_seed_address": args.board_seed_address,
        "board_seed_override": args.board_seed,
        "global_rng_seed_override": args.global_rng_seed,
        "thread_crt_rng_seed_override": args.thread_crt_rng_seed,
        "observed_board_seed_equal": (
            observed_board_seeds[0] == observed_board_seeds[1]
        ),
        "effective_board_seed_equal": (
            effective_board_seeds[0] == effective_board_seeds[1]
        ),
        "observed_thread_crt_rng_state_equal": (
            None
            if len(thread_crt_states_before) != 2
            else (
                thread_crt_states_before[0]
                == thread_crt_states_before[1]
            )
        ),
        "effective_thread_crt_rng_state_equal": (
            None
            if len(thread_crt_states_after) != 2
            else (
                thread_crt_states_after[0]
                == thread_crt_states_after[1]
            )
        ),
        "pre_state_root": pre.state_root,
        "host_pre_state_root": host.state_root,
        "host_restored_state_root": host.state_root,
        "runs": runs,
    }
    _write_exclusive(
        output_root / "diagnostic.json",
        (
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    return result


def _u32(text: str) -> int:
    value = int(text, 0)
    if not 0 <= value <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("value must fit uint32")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--session-nonce", required=True)
    parser.add_argument("--dmo", type=Path, default=DEFAULT_DMO)
    parser.add_argument("--prestate-dir", type=Path, default=DEFAULT_PRESTATE)
    parser.add_argument(
        "--runtime-executable",
        type=Path,
        default=DEFAULT_RUNTIME,
    )
    parser.add_argument("--changedir", type=Path, default=DEFAULT_CHANGEDIR)
    parser.add_argument("--app-id", type=int, default=3620)
    parser.add_argument("--seed", type=_u32)
    parser.add_argument(
        "--board-seed-address",
        type=_u32,
        default=DEFAULT_BOARD_RESEED_CALL,
    )
    parser.add_argument("--board-seed", type=_u32)
    parser.add_argument("--global-rng-seed", type=_u32)
    parser.add_argument("--thread-crt-rng-seed", type=_u32)
    parser.add_argument(
        "--attach-at-update",
        type=_u32,
        default=0,
    )
    parser.add_argument("--attach-timeout", type=float, default=30.0)
    parser.add_argument(
        "--launch-timeout",
        "--timeout",
        dest="launch_timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument("--trace-timeout", type=float, default=180.0)
    parser.add_argument("--service-wait-timeout", type=float, default=5.0)
    parser.add_argument("--progress-every-updates", type=int, default=2000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    result = diagnose(build_parser().parse_args(argv))
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
