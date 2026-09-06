"""Compare two read-only retail Zuma live RNG change monitors.

The monitor is asynchronous: it can miss framework updates and can observe
more than one in-update state.  This comparator therefore groups rows by
framework update and treats an update as shared when the two runs captured at
least one identical semantic state.  Process-local addresses and sampling
timestamps are deliberately excluded.

The report distinguishes transient disjoint samples from the terminal shared
state.  A terminal divergence bound is observational: it brackets the gap
between the last common sampled state and the first later common update whose
sample sets are disjoint.  It does not claim that an unsampled update is the
exact causal instruction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "zuma-rl.live-rng-monitor-comparison"
VERSION = 1
MONITOR_SCHEMA = "zuma-rl.live-rng-change-monitor"
COMPONENTS = (
    "global_mtrand",
    "thread_crt_rand",
    "qrand",
    "board",
    "complete_semantic_state",
)


class MonitorComparisonError(RuntimeError):
    """Raised when live-monitor evidence cannot be compared safely."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise MonitorComparisonError(reason)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MonitorComparisonError(f"{context}_is_not_an_integer")
    return value


def _ball_semantic(value: Any, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    _require(isinstance(value, Mapping), f"{context}_is_not_an_object")
    fields = (
        "ball_id",
        "color_id",
        "kind",
        "powerup_previous_type",
        "powerup_primary_type",
        "powerup_secondary_type",
    )
    result = {field: value.get(field) for field in fields}
    _strict_int(result["ball_id"], f"{context}_ball_id")
    _strict_int(result["color_id"], f"{context}_color_id")
    return result


def _chain_semantic(value: Any, context: str) -> list[dict[str, int]]:
    _require(isinstance(value, list), f"{context}_is_not_a_list")
    result: list[dict[str, int]] = []
    for index, row in enumerate(value):
        _require(
            isinstance(row, Mapping),
            f"{context}_{index}_is_not_an_object",
        )
        result.append(
            {
                "ball_id": _strict_int(
                    row.get("ball_id"),
                    f"{context}_{index}_ball_id",
                ),
                "color_id": _strict_int(
                    row.get("color_id"),
                    f"{context}_{index}_color_id",
                ),
            }
        )
    return result


def _projection(row: Mapping[str, Any], component: str) -> Any:
    global_mtrand = {
        "index": _strict_int(
            row.get("global_mtrand_index"),
            "global_mtrand_index",
        ),
        "sha256": row.get("global_mtrand_sha256"),
    }
    _require(
        isinstance(global_mtrand["sha256"], str)
        and len(global_mtrand["sha256"]) == 64,
        "global_mtrand_sha256_invalid",
    )
    thread_crt = row.get("thread_crt_rand_state")
    if thread_crt is not None:
        _strict_int(thread_crt, "thread_crt_rand_state")
    qrand = row.get("qrand")
    _require(isinstance(qrand, Mapping), "qrand_is_not_an_object")
    vectors = qrand.get("vectors")
    _require(isinstance(vectors, Mapping), "qrand_vectors_invalid")
    qrand_semantic = {
        "update_count": _strict_int(
            qrand.get("update_count"),
            "qrand_update_count",
        ),
        "selected_index": _strict_int(
            qrand.get("selected_index"),
            "qrand_selected_index",
        ),
        "vectors": dict(vectors),
    }
    board_fields = (
        "native_game_time",
        "score",
        "displayed_score",
        "score_target",
        "chain_ball_count",
        "pending_ball_count",
        "inserting_ball_count",
        "fired_bullet_count",
    )
    board = {
        field: _strict_int(row.get(field), field) for field in board_fields
    }
    colors = row.get("board_color_counts")
    _require(isinstance(colors, list), "board_color_counts_invalid")
    board["board_color_counts"] = [
        _strict_int(value, "board_color_count") for value in colors
    ]
    board["chain"] = _chain_semantic(row.get("chain"), "chain")
    board["current_ball"] = _ball_semantic(
        row.get("current_ball"),
        "current_ball",
    )
    board["next_ball"] = _ball_semantic(
        row.get("next_ball"),
        "next_ball",
    )

    values = {
        "global_mtrand": global_mtrand,
        "thread_crt_rand": thread_crt,
        "qrand": qrand_semantic,
        "board": board,
    }
    if component == "complete_semantic_state":
        return values
    _require(component in values, f"unknown_component:{component}")
    return values[component]


def _load_monitor(path: Path) -> dict[str, Any]:
    path = path.resolve()
    data = path.read_bytes()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise MonitorComparisonError(
            f"monitor_is_not_utf8:{path.name}"
        ) from error
    _require(bool(lines), f"monitor_is_empty:{path.name}")
    parsed: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise MonitorComparisonError(
                f"monitor_json_invalid:{path.name}:{line_number}"
            ) from error
    header = parsed[0]
    _require(isinstance(header, Mapping), "monitor_header_invalid")
    _require(
        header.get("schema") == MONITOR_SCHEMA
        and header.get("version") == 1
        and header.get("classification")
        == "read-only-localization-diagnostic"
        and header.get("process_memory_writes") == 0,
        f"monitor_header_contract_invalid:{path.name}",
    )
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    unavailable_count = 0
    for line_number, row in enumerate(parsed[1:], start=2):
        _require(
            isinstance(row, Mapping),
            f"monitor_row_invalid:{path.name}:{line_number}",
        )
        row_type = row.get("type")
        if row_type == "unavailable":
            unavailable_count += 1
            continue
        _require(
            row_type == "rng",
            f"monitor_row_type_invalid:{path.name}:{line_number}",
        )
        update = _strict_int(
            row.get("framework_update"),
            f"framework_update:{path.name}:{line_number}",
        )
        _require(update >= 0, "framework_update_is_negative")
        for component in COMPONENTS:
            _projection(row, component)
        groups[update].append(row)
    _require(bool(groups), f"monitor_has_no_rng_rows:{path.name}")
    updates = sorted(groups)
    row_counts = Counter(
        update for update, rows in groups.items() for _ in rows
    )
    missing_update_count = sum(
        max(0, following - current - 1)
        for current, following in zip(updates, updates[1:])
    )
    return {
        "path": path,
        "sha256": _sha256_bytes(data),
        "header": dict(header),
        "groups": groups,
        "summary": {
            "rng_row_count": sum(row_counts.values()),
            "unique_update_count": len(updates),
            "first_update": updates[0],
            "last_update": updates[-1],
            "duplicate_update_row_count": sum(
                count - 1 for count in row_counts.values()
            ),
            "maximum_rows_per_update": max(row_counts.values()),
            "missing_update_count_inside_observed_range": (
                missing_update_count
            ),
            "unavailable_change_row_count": unavailable_count,
        },
    }


def _state_map(
    rows: Iterable[Mapping[str, Any]],
    component: str,
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for row in rows:
        value = _projection(row, component)
        states.setdefault(_canonical(value), value)
    return states


def _different_fields(left: Any, right: Any) -> list[str]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return [] if left == right else ["value"]
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )


def _closest_disjoint_examples(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: list[tuple[int, str, str, list[str]]] = []
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            fields = _different_fields(left_value, right_value)
            candidates.append(
                (len(fields), left_key, right_key, fields)
            )
    candidates.sort(key=lambda value: value[:3])
    _, left_key, right_key, fields = candidates[0]
    return {
        "differing_fields": fields,
        "left_example": left[left_key],
        "right_example": right[right_key],
        "left_state_count": len(left),
        "right_state_count": len(right),
    }


def _compare_component(
    left_groups: Mapping[int, list[Mapping[str, Any]]],
    right_groups: Mapping[int, list[Mapping[str, Any]]],
    common_updates: list[int],
    component: str,
) -> dict[str, Any]:
    state_pairs: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    shared_updates: list[int] = []
    disjoint_updates: list[int] = []
    for update in common_updates:
        left_states = _state_map(left_groups[update], component)
        right_states = _state_map(right_groups[update], component)
        state_pairs[update] = (left_states, right_states)
        if set(left_states) & set(right_states):
            shared_updates.append(update)
        else:
            disjoint_updates.append(update)

    first_disjoint = disjoint_updates[0] if disjoint_updates else None
    last_shared = shared_updates[-1] if shared_updates else None
    first_after_last_shared = next(
        (
            update
            for update in common_updates
            if last_shared is None or update > last_shared
        ),
        None,
    )
    terminal_bound = None
    if first_after_last_shared is not None:
        left_states, right_states = state_pairs[first_after_last_shared]
        _require(
            not (set(left_states) & set(right_states)),
            "terminal_divergence_bound_is_not_disjoint",
        )
        terminal_bound = {
            "last_shared_update": last_shared,
            "first_later_common_disjoint_update": (
                first_after_last_shared
            ),
            "unsampled_or_unpaired_update_count_between": (
                first_after_last_shared
                - (last_shared if last_shared is not None else -1)
                - 1
            ),
            "all_later_common_updates_disjoint": True,
            "later_common_disjoint_update_count": sum(
                update >= first_after_last_shared
                for update in disjoint_updates
            ),
            **_closest_disjoint_examples(left_states, right_states),
        }
    return {
        "shared_common_update_count": len(shared_updates),
        "disjoint_common_update_count": len(disjoint_updates),
        "first_observed_disjoint_update": first_disjoint,
        "last_observed_shared_update": last_shared,
        "terminal_divergence_observed": terminal_bound is not None,
        "terminal_divergence_bound": terminal_bound,
    }


def compare_monitors(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Return a hash-bound, address-independent comparison report."""

    left = _load_monitor(left_path)
    right = _load_monitor(right_path)
    _require(
        left["sha256"] != right["sha256"],
        "monitor_inputs_are_identical",
    )
    common_updates = sorted(
        set(left["groups"]) & set(right["groups"])
    )
    _require(bool(common_updates), "monitors_have_no_common_updates")
    components = {
        component: _compare_component(
            left["groups"],
            right["groups"],
            common_updates,
            component,
        )
        for component in COMPONENTS
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "PASS",
        "classification": "read-only-localization-diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": {
            "group_by": "framework_update",
            "same_update_rule": "any-shared-semantic-state",
            "excluded_fields": [
                "perf_counter_ns",
                "process-local-addresses",
            ],
            "terminal_divergence_bound_is_observational": True,
        },
        "inputs": {
            "left": {
                "path": str(left["path"]),
                "sha256": left["sha256"],
                "process_id": left["header"].get("process_id"),
                "summary": left["summary"],
            },
            "right": {
                "path": str(right["path"]),
                "sha256": right["sha256"],
                "process_id": right["header"].get("process_id"),
                "summary": right["summary"],
            },
        },
        "common_updates": {
            "count": len(common_updates),
            "first": common_updates[0],
            "last": common_updates[-1],
        },
        "components": components,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    report = compare_monitors(args.left, args.right)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(output)
    print(json.dumps(report["components"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
