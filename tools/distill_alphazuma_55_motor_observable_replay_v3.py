"""Retry-audited motor-observable replay after the V2 anchor incident."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as v2
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_ANCHOR_PASSES = 3
EXPECTED_ANCHOR_ATTEMPTS_PER_LEVEL = 3
EXPECTED_SCHEDULE = (0.5, 0.1, 0.0)
EXPECTED_EPOCHS = 4

_prototype = v2._prototype
_build_motor_model = v2._build_motor_model
_verify_initial_equivalence = v2._verify_initial_equivalence


def _validate_reference(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} bytes changed")
    return path


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if not (
        prereg.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-replay-preregistration"
        and prereg.get("version") == 2
        and prereg.get("status") == "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected retry-audited preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("retry-audited trainer binding differs")
    _validate_reference(prereg.get("builder", {}), "builder")
    master_path = _validate_reference(
        prereg.get("master_preregistration", {}), "master preregistration"
    )
    master = legacy._read_json(master_path)
    for name, artifact in prereg.get("implementation", {}).items():
        _validate_reference(artifact, f"implementation {name}")

    incident = prereg.get("predecessor_runtime_incident", {})
    if incident.get("classification") != (
        "V2_ANCHOR_54_OF_55_ABORT_BEFORE_ANY_OPTIMIZATION"
    ):
        raise ValueError("unexpected predecessor runtime incident")
    incident_paths = {
        name: _validate_reference(reference, f"incident {name}")
        for name, reference in incident.get("artifacts", {}).items()
    }
    failure = legacy._read_json(incident_paths["failure"])
    status = legacy._read_json(incident_paths["training_status"])
    if not (
        failure.get("status") == "ERROR"
        and failure.get("error_type") == "RuntimeError"
        and failure.get("error")
        == "motor-observable teacher anchor failed a level: pass=0 wins=54"
        and failure.get("formal_seed_consumption") is False
        and status.get("stage") == "COLLECTING_TEACHER_ANCHOR"
        and int(status.get("completed_anchor_passes", -1)) == 0
        and int(status.get("completed_student_rounds", -1)) == 0
    ):
        raise ValueError("predecessor incident facts changed")

    source = prereg.get("source", {})
    source_path = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_path):
        raise ValueError("source model bytes changed")
    source_audit = Path(
        str(source.get("selection_audit_path", ""))
    ).resolve(strict=True)
    if source.get("selection_audit_sha256") != legacy._sha256(source_audit):
        raise ValueError("source audit bytes changed")
    if legacy._read_json(source_audit).get("status") != "PASS":
        raise ValueError("source audit is not PASS")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("level order changed")

    run = prereg.get("run", {})
    retry = run.get("anchor_retry_policy", {})
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_motor_observable_replay"
        and run.get("motor_observation_profile") == "motor-observable-v1"
        and int(run.get("anchor_teacher_passes", -1))
        == EXPECTED_ANCHOR_PASSES
        and int(run.get("student_rounds", -1)) == len(EXPECTED_SCHEDULE)
        and tuple(float(value) for value in run.get(
            "teacher_execution_probabilities", ()
        ))
        == EXPECTED_SCHEDULE
        and int(run.get("epochs_per_round", -1)) == EXPECTED_EPOCHS
        and int(run.get("checkpoint_interval_epochs", -1)) == 1
        and int(retry.get("attempts_per_level", -1))
        == EXPECTED_ANCHOR_ATTEMPTS_PER_LEVEL
        and retry.get("first_attempt_mode") == "paired_full55_batch"
        and retry.get("retry_mode") == "failed_levels_only_fixed_seed"
        and retry.get("sample_retention") == "all_attempts"
        and retry.get("pass_requirement")
        == "at_least_one_win_for_each_of_55_levels"
        and retry.get("seed_formula")
        == (
            "anchor_seed_base + pass_index*(55*attempts_per_level) + "
            "attempt_index*55 + level_index"
        )
    ):
        raise ValueError("retry-audited training semantics changed")

    anchor_first = int(run["anchor_seed_base"])
    anchor_last = int(run["anchor_seed_last_reserved"])
    student_first = int(run["student_seed_base"])
    student_last = int(run["student_seed_last_consumed"])
    expected_anchor_last = (
        anchor_first
        + EXPECTED_ANCHOR_PASSES
        * EXPECTED_ANCHOR_ATTEMPTS_PER_LEVEL
        * len(INCLUDED_LEVELS)
        - 1
    )
    if not (
        anchor_last == expected_anchor_last
        and student_first == anchor_last + 1
        and student_last
        == student_first + len(EXPECTED_SCHEDULE) * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("retry-audited seed intervals changed")
    registry = master["seed_registry"]["training"]
    if not (
        int(registry["first"])
        <= anchor_first
        <= student_last
        <= int(registry["last"])
    ):
        raise ValueError("retry-audited seeds escaped training registry")
    disclosure = prereg.get("post_hoc_disclosure", {})
    if not (
        disclosure.get("trigger") == "observed_v2_anchor_54_of_55"
        and disclosure.get("new_training_seeds_unseen_before_freeze") is True
        and disclosure.get("formal_gate_unchanged") is True
    ):
        raise ValueError("post-hoc disclosure changed")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"authority boundary changed: {name}")
    return prereg


def _anchor_seed(
    config: dict[str, Any],
    *,
    pass_index: int,
    level_index: int,
    attempt_index: int,
) -> int:
    return (
        int(config["anchor_seed_base"])
        + pass_index
        * len(INCLUDED_LEVELS)
        * EXPECTED_ANCHOR_ATTEMPTS_PER_LEVEL
        + attempt_index * len(INCLUDED_LEVELS)
        + level_index
    )


def _dataset_sha256(
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _counter_sum(
    collections: Sequence[dict[str, Any]], key: str
) -> dict[str, int]:
    result: Counter[str] = Counter()
    for collection in collections:
        result.update(
            {
                str(name): int(value)
                for name, value in collection.get(key, {}).items()
            }
        )
    return dict(result)


def _write_attempt(
    *,
    run_dir: Path,
    pass_index: int,
    attempt_index: int,
    level_index: int | None,
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    collection: dict[str, Any],
) -> dict[str, Any]:
    suffix = "full55" if level_index is None else f"level-{level_index:02d}"
    path = (
        run_dir
        / "anchor_attempts"
        / f"pass-{pass_index:02d}-attempt-{attempt_index:02d}-{suffix}.json"
    )
    receipt = {
        "schema": "zuma-rl.alphazuma-55-anchor-attempt",
        "version": 1,
        "status": "COMPLETE",
        "pass_index": pass_index,
        "attempt_index": attempt_index,
        "level_index": level_index,
        "dataset_rows": len(rows),
        "dataset_sha256": _dataset_sha256(rows),
        "collection": collection,
        "formal_seed_consumption": False,
    }
    dagger._write_atomic(path, receipt)
    return {
        "path": str(path),
        "sha256": legacy._sha256(path),
        "dataset_rows": len(rows),
        "dataset_sha256": receipt["dataset_sha256"],
    }


def _collect_teacher_anchor_with_retries(
    *,
    base_collector: Callable[..., tuple[list[Any], dict[str, Any]]],
    config: dict[str, Any],
    run_dir: Path,
    original_root: Path,
    level_ids: Sequence[str],
    round_index: int,
    seed_base: int,
    parallel_envs: int,
    max_ticks: int,
    capacities: dict[str, int],
    strides: dict[str, int],
    model_seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    levels = tuple(str(value) for value in level_ids)
    if levels != tuple(INCLUDED_LEVELS):
        raise ValueError("anchor retry collector requires the frozen full55 order")
    if int(seed_base) != int(config["anchor_seed_base"]):
        raise ValueError("anchor seed base differs from preregistration")
    if not 0 <= int(round_index) < EXPECTED_ANCHOR_PASSES:
        raise ValueError("anchor pass index is outside the frozen schedule")

    pass_index = int(round_index)
    all_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    collections: list[dict[str, Any]] = []
    attempt_receipts: list[dict[str, Any]] = []
    won = {level_id: False for level_id in levels}
    consumed_seeds: list[int] = []

    def record(
        *,
        rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        collection: dict[str, Any],
        attempt_index: int,
        expected_level_indices: Sequence[int],
        receipt_level_index: int | None,
        sampling_seed: int,
    ) -> None:
        normalized = copy.deepcopy(collection)
        normalized["anchor_pass"] = pass_index
        normalized["attempt_index"] = attempt_index
        normalized["sampling_seed"] = sampling_seed
        episodes = normalized.get("episodes", [])
        if len(episodes) != len(expected_level_indices):
            raise RuntimeError("anchor attempt episode count differs")
        for episode, level_index in zip(
            episodes, expected_level_indices, strict=True
        ):
            expected_level = levels[level_index]
            expected_seed = _anchor_seed(
                config,
                pass_index=pass_index,
                level_index=level_index,
                attempt_index=attempt_index,
            )
            if not (
                str(episode.get("level_id")) == expected_level
                and int(episode.get("seed", -1)) == expected_seed
            ):
                raise RuntimeError("anchor attempt level or seed differs")
            episode["anchor_pass"] = pass_index
            episode["attempt_index"] = attempt_index
            episode["level_index"] = level_index
            episode["sampling_seed"] = sampling_seed
            consumed_seeds.append(expected_seed)
            if episode.get("outcome") == "win":
                won[expected_level] = True
        all_rows.extend(rows)
        collections.append(normalized)
        attempt_receipts.append(
            _write_attempt(
                run_dir=run_dir,
                pass_index=pass_index,
                attempt_index=attempt_index,
                level_index=receipt_level_index,
                rows=rows,
                collection=normalized,
            )
        )

    first_seed = _anchor_seed(
        config, pass_index=pass_index, level_index=0, attempt_index=0
    )
    first_sampling_seed = int(model_seed) + pass_index * 1_000_003
    first_rows, first_collection = base_collector(
        original_root=original_root,
        level_ids=levels,
        round_index=0,
        seed_base=first_seed,
        parallel_envs=parallel_envs,
        max_ticks=max_ticks,
        capacities=capacities,
        strides=strides,
        model_seed=first_sampling_seed,
    )
    record(
        rows=first_rows,
        collection=first_collection,
        attempt_index=0,
        expected_level_indices=tuple(range(len(levels))),
        receipt_level_index=None,
        sampling_seed=first_sampling_seed,
    )

    for attempt_index in range(1, EXPECTED_ANCHOR_ATTEMPTS_PER_LEVEL):
        missing_indices = [
            index for index, level_id in enumerate(levels) if not won[level_id]
        ]
        if not missing_indices:
            break
        for level_index in missing_indices:
            retry_seed = _anchor_seed(
                config,
                pass_index=pass_index,
                level_index=level_index,
                attempt_index=attempt_index,
            )
            sampling_seed = (
                int(model_seed)
                + pass_index * 1_000_003
                + attempt_index * 100_003
                + level_index
            )
            retry_rows, retry_collection = base_collector(
                original_root=original_root,
                level_ids=(levels[level_index],),
                round_index=0,
                seed_base=retry_seed,
                parallel_envs=1,
                max_ticks=max_ticks,
                capacities=capacities,
                strides=strides,
                model_seed=sampling_seed,
            )
            record(
                rows=retry_rows,
                collection=retry_collection,
                attempt_index=attempt_index,
                expected_level_indices=(level_index,),
                receipt_level_index=level_index,
                sampling_seed=sampling_seed,
            )
            dagger._write_atomic(
                run_dir / "anchor_retry_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-anchor-retry-status",
                    "version": 1,
                    "status": "RUNNING",
                    "pass_index": pass_index,
                    "attempt_index": attempt_index,
                    "completed_level_coverage": sum(won.values()),
                    "expected_level_coverage": len(levels),
                    "missing_level_ids": [
                        level_id for level_id in levels if not won[level_id]
                    ],
                    "attempts_consumed": len(consumed_seeds),
                    "formal_seed_consumption": False,
                },
            )

    if len(consumed_seeds) != len(set(consumed_seeds)):
        raise RuntimeError("anchor retry seed ledger contains duplicates")
    missing = [level_id for level_id in levels if not won[level_id]]
    episodes = [
        episode
        for collection in collections
        for episode in collection.get("episodes", [])
    ]
    merged = {
        "round_index": pass_index,
        "anchor_pass": pass_index,
        "wall_seconds": sum(
            float(value.get("wall_seconds", 0.0)) for value in collections
        ),
        "episodes": episodes,
        "wins": len(levels) - len(missing),
        "level_coverage_wins": len(levels) - len(missing),
        "attempt_wins": sum(
            episode.get("outcome") == "win" for episode in episodes
        ),
        "losses": sum(
            episode.get("outcome") == "loss" for episode in episodes
        ),
        "truncations": sum(
            bool(episode.get("time_limit_truncated", False))
            for episode in episodes
        ),
        "capacity_overflows": sum(
            bool(episode.get("observation_capacity_overflow", False))
            for episode in episodes
        ),
        "retained_samples": len(all_rows),
        "retained_samples_by_verb": _counter_sum(
            collections, "retained_samples_by_verb"
        ),
        "raw_intent_action_counts": _counter_sum(
            collections, "raw_intent_action_counts"
        ),
        "effective_execution_action_counts": _counter_sum(
            collections, "effective_execution_action_counts"
        ),
        "intent_mask_relaxations": sum(
            int(value.get("intent_mask_relaxations", 0))
            for value in collections
        ),
        "training_label_semantics": "raw_teacher_intent",
        "environment_execution_semantics": "exact_mask_effective_action",
        "teacher_policy_id": collections[0].get("teacher_policy_id"),
        "anchor_retry_policy": config["anchor_retry_policy"],
        "attempts_consumed": len(consumed_seeds),
        "consumed_seeds": consumed_seeds,
        "missing_level_ids": missing,
        "sample_retention": "all_attempts",
        "dataset_sha256": _dataset_sha256(all_rows),
        "attempt_receipts": attempt_receipts,
    }
    summary_path = (
        run_dir / "anchor_attempts" / f"pass-{pass_index:02d}-summary.json"
    )
    dagger._write_atomic(
        summary_path,
        {
            "schema": "zuma-rl.alphazuma-55-anchor-pass-summary",
            "version": 1,
            "status": "PASS" if not missing else "FAIL",
            "collection": merged,
            "formal_seed_consumption": False,
        },
    )
    return all_rows, merged


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_dir = Path(str(prereg["run"]["run_dir"])).resolve()
    original_validate = v2._validate_preregistration
    original_collector = v2.intent._collect_intent_round

    def retry_collector(**kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        return _collect_teacher_anchor_with_retries(
            base_collector=original_collector,
            config=prereg["run"],
            run_dir=run_dir,
            **kwargs,
        )

    try:
        v2._validate_preregistration = lambda _: prereg
        v2.intent._collect_intent_round = retry_collector
        completion = v2.run(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    except BaseException:
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = legacy._read_json(failure_path)
            failure.update(
                {
                    "version": 2,
                    "route_version": "v3_retry_audited",
                    "post_hoc_disclosure": prereg["post_hoc_disclosure"],
                }
            )
            dagger._write_atomic(failure_path, failure)
        raise
    finally:
        v2._validate_preregistration = original_validate
        v2.intent._collect_intent_round = original_collector

    completion.update(
        {
            "version": 2,
            "route_version": "v3_retry_audited",
            "anchor_retry_policy": prereg["run"]["anchor_retry_policy"],
            "post_hoc_disclosure": prereg["post_hoc_disclosure"],
        }
    )
    dagger._write_atomic(run_dir / "completion.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        prototype = _prototype(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = _build_motor_model(prototype, prereg["run"])
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": legacy._sha256(prereg_path),
                        },
                        "observation_shape": list(model.observation_space.shape),
                        "motor_feature_count": len(
                            prototype.motor_feature_names
                        ),
                        "policy_parameter_count": sum(
                            parameter.numel()
                            for parameter in model.policy.parameters()
                        ),
                        "anchor_retry_policy": prereg["run"][
                            "anchor_retry_policy"
                        ],
                        "formal_seed_consumption": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
        finally:
            prototype.close()
        return 0
    completion = run(
        preregistration_path=prereg_path,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "final_model": completion["final_model"],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
