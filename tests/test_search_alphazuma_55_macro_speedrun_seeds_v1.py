from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from tools import distill_alphazuma_55_park_settle_v1 as distill_v1
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from tools import probe_alphazuma_55_macro_native_eval_v1 as native_eval
from tools import search_alphazuma_55_macro_speedrun_seeds_v1 as search
import zuma_rl.park_settle_action_wrapper as wrapper_module


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")
STACK_LAGS = (0, 4, 8)
# Engineering region: outside the frozen embargoes, outside every
# reserved block, and ABOVE the planned production tail-search block
# 1_496_000_000..1_496_009_999 so tests never touch its (level, seed)
# keys.
TEST_SEED_START = 1_496_500_000


def _row(
    seed: int,
    outcome: str | None,
    ticks: int,
    *,
    shots: int = 3,
    score: int = 1_000,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "outcome": outcome,
        "ticks": ticks,
        "shots": shots,
        "score": score,
        "time_limit_truncated": truncated,
    }


# ---------------------------------------------------------------------------
# (a) Pure top-k / percentile math on synthetic rows.
# ---------------------------------------------------------------------------


def test_summarize_rows_percentiles_and_top_k() -> None:
    rows = []
    # 20 wins, ticks 100..2000, appended in DESCENDING order so the
    # summary has to sort rather than trust input order.
    win_ticks = list(range(2_000, 0, -100))
    for offset, ticks in enumerate(win_ticks):
        rows.append(
            _row(
                seed=TEST_SEED_START + offset,
                outcome="win",
                ticks=ticks,
                shots=10 + offset,
                score=50_000 - ticks,
            )
        )
    rows.append(_row(TEST_SEED_START + 90, "loss", 500))
    rows.append(
        _row(TEST_SEED_START + 91, None, 30_000, truncated=True)
    )
    summary = search.summarize_rows(rows, top_k=5)
    assert summary["total_seeds"] == 22
    assert summary["wins"] == 20
    assert summary["losses"] == 1
    assert summary["truncations"] == 1
    assert summary["win_rate"] == pytest.approx(20 / 22)
    # method="lower" over the 20 sorted win ticks [100, 200, ..., 2000]:
    # index floor(q/100 * 19) -> p1/p5 -> 100, p10 -> 200, p50 -> 1000.
    # Every reported percentile is an actually-achieved win tick.
    assert summary["win_tick_percentiles"] == {
        "p1": 100,
        "p5": 100,
        "p10": 200,
        "p50": 1_000,
    }
    top = summary["top_k_fastest_wins"]
    assert [entry["ticks"] for entry in top] == [100, 200, 300, 400, 500]
    assert all(
        set(entry) == {"seed", "ticks", "shots", "score"} for entry in top
    )
    # The fastest win row's own seed/shots travelled with it.
    fastest_source = rows[len(win_ticks) - 1]
    assert top[0]["seed"] == fastest_source["seed"]
    assert top[0]["shots"] == fastest_source["shots"]
    json.dumps(search.summarize_rows(rows, top_k=5), allow_nan=False)


def test_summarize_rows_tie_breaks_empty_wins_and_edge_cases() -> None:
    tie_rows = [
        _row(7, "win", 400),
        _row(5, "win", 400),
        _row(6, "win", 300),
    ]
    top = search.summarize_rows(tie_rows, top_k=2)["top_k_fastest_wins"]
    assert [(entry["ticks"], entry["seed"]) for entry in top] == [
        (300, 6),
        (400, 5),
    ]
    # top_k larger than the win count returns every win, no padding.
    assert (
        len(
            search.summarize_rows(tie_rows, top_k=25)[
                "top_k_fastest_wins"
            ]
        )
        == 3
    )
    # Zero wins: null percentiles, empty table, zero win rate.
    no_wins = search.summarize_rows([_row(1, "loss", 100)], top_k=3)
    assert no_wins["wins"] == 0
    assert no_wins["win_rate"] == 0.0
    assert no_wins["win_tick_percentiles"] == {
        "p1": None,
        "p5": None,
        "p10": None,
        "p50": None,
    }
    assert no_wins["top_k_fastest_wins"] == []
    # Zero rows: the honest fraction is NaN (serialized null), never 0.
    empty = search.summarize_rows([], top_k=1)
    assert empty["total_seeds"] == 0
    assert math.isnan(empty["win_rate"])
    with pytest.raises(ValueError, match="top_k"):
        search.summarize_rows([], top_k=0)


# ---------------------------------------------------------------------------
# (b) Seed-range discipline: embargoes and reserved engineering blocks.
# ---------------------------------------------------------------------------


def test_validate_seed_range_blocks_embargoes_and_reserved() -> None:
    # The planned production block validates cleanly.
    assert search.validate_seed_range(1_496_000_000, 10_000) == (
        1_496_000_000,
        1_496_009_999,
    )
    assert search.validate_seed_range(TEST_SEED_START, 4) == (
        TEST_SEED_START,
        TEST_SEED_START + 3,
    )
    with pytest.raises(ValueError, match="embargo"):
        search.validate_seed_range(1_600_000_000, 1)
    # A straddling range is refused just like an inside one.
    with pytest.raises(ValueError, match="embargo"):
        search.validate_seed_range(1_599_999_990, 100)
    with pytest.raises(ValueError, match="reserved"):
        search.validate_seed_range(1_493_999_999, 2)
    with pytest.raises(ValueError, match="reserved"):
        search.validate_seed_range(1_494_000_000, 1)
    with pytest.raises(ValueError, match="reserved"):
        search.validate_seed_range(1_400_920_500, 10)
    # 1_495_999_999 is the reserved macro-decisions headroom ceiling;
    # one above it is the first free engineering seed.
    with pytest.raises(ValueError, match="reserved"):
        search.validate_seed_range(1_495_999_999, 2)
    assert search.validate_seed_range(1_496_000_000, 1) == (
        1_496_000_000,
        1_496_000_000,
    )
    with pytest.raises(ValueError, match="positive"):
        search.validate_seed_range(1, 0)
    with pytest.raises(ValueError, match="negative"):
        search.validate_seed_range(-1, 5)


# ---------------------------------------------------------------------------
# (c) Real-engine CPU smoke: Jungle1, four streamed seeds over two
# persistent envs, a tiny in-test macro-native stacked model, then
# resume (full and partial with a torn tail).
# ---------------------------------------------------------------------------


def _tiny_macro_native_stacked_model(vector: Any) -> Any:
    """Small macro-native stacked MaskablePPO on the REAL interface.

    Mirrors tests/test_probe_alphazuma_55_macro_native_eval_v1.py: the
    v3 stacked extractor over the real env's entity_polar layout, an
    interface-only env carrying the stacked Box and the macro
    MultiDiscrete([3, 180]).
    """

    from sb3_contrib import MaskablePPO

    prototype = wrapper_module.resolve_human_speedrun_wrapper(
        vector.envs[0]
    )
    policy_kwargs = trainer_v3.stacked_entity_polar_policy_kwargs(
        prototype, learned_features_dim=16, stack_lags=STACK_LAGS
    )
    return MaskablePPO(
        "MlpPolicy",
        distill_v1._SpaceOnlyMaskableEnv(
            vector.observation_space,
            spaces.MultiDiscrete(np.array((3, 180), dtype=np.int64)),
        ),
        policy_kwargs=policy_kwargs,
        n_steps=8,
        batch_size=8,
        device="cpu",
        verbose=0,
        seed=99_081_699,
    )


def test_real_engine_smoke_stream_and_resume(tmp_path: Path) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 300
    seed_count = 4
    factory = native_eval._make_stacked_macro_env_factory(
        original_root=RETAIL_ROOT,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
    )
    prototype_vector = DummyVecEnv([factory])
    try:
        model = _tiny_macro_native_stacked_model(prototype_vector)
    finally:
        prototype_vector.close()
    model_path = tmp_path / "tiny_macro_native.zip"
    model.save(str(model_path))

    run_dir = tmp_path / "tail-search"
    run_kwargs = dict(
        model_path=model_path,
        original_root=RETAIL_ROOT,
        run_dir=run_dir,
        level_id="Jungle1",
        seed_start=TEST_SEED_START,
        seed_count=seed_count,
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        parallel_envs=2,
        top_k=3,
        device="cpu",
        vec_env="dummy",
    )
    completion = search.run(**run_kwargs)

    # results.jsonl: one valid row per planned seed.
    results_path = run_dir / "results.jsonl"
    lines = results_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == seed_count
    assert {row["seed"] for row in rows} == {
        TEST_SEED_START + offset for offset in range(seed_count)
    }
    for row in rows:
        assert row["schema"] == search.ROW_SCHEMA
        assert row["level_id"] == "Jungle1"
        assert row["outcome"] in {"win", "loss", None}
        assert 0 < row["ticks"] <= max_ticks
        assert row["macro_decisions"] >= 1
        assert 0 < row["native_ticks_consumed"] <= max_ticks
        assert row["shots"] >= 0
        if row["outcome"] == "win":
            assert row["win_trigger_tick"] == row["ticks"]
            assert row["win_pending_at_termination"] in {True, False}
        else:
            assert row["win_trigger_tick"] is None
            assert row["win_pending_at_termination"] is None
            assert row["time_limit_truncated"] or row["outcome"] == "loss"
        assert row["board_cleared_tick"] is None

    # completion.json: schema, spec'd fields, and honest tail math.
    assert completion["schema"] == search.COMPLETION_SCHEMA
    assert completion["status"] == "COMPLETE"
    assert completion["level"] == "Jungle1"
    assert completion["total_seeds"] == seed_count
    wins = sum(row["outcome"] == "win" for row in rows)
    assert completion["wins"] == wins
    if wins:
        assert completion["win_rate"] == pytest.approx(wins / seed_count)
    assert set(completion["win_tick_percentiles"]) == {
        "p1",
        "p5",
        "p10",
        "p50",
    }
    assert len(completion["top_k_fastest_wins"]) == min(3, wins)
    assert completion["model_sha256"].startswith("sha256:")
    assert (
        completion["model_sha256"]
        == completion["source"]["model_sha256"]
    )
    assert completion["wall_seconds"] > 0
    # Record-mode flags absent: prior receipts stay byte-compatible --
    # the exact convention label, no stochastic/settle blocks anywhere,
    # and the wrapper contract echoes its own default tolerance.
    assert completion["convention"] == search.BEST_OF_N_CONVENTION
    assert "stochastic_sampling" not in completion
    assert "settle_tolerance_bins" not in completion
    assert completion["wrapper_contract"]["config"][
        "settle_tolerance_bins"
    ] == search.DEFAULT_SETTLE_TOLERANCE_BINS
    baseline_config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert "stochastic_sampling" not in baseline_config
    assert "settle_tolerance_bins" not in baseline_config
    assert "stochastic_temperature" not in (
        baseline_config["resume_identity"]
    )
    assert "settle_tolerance_bins" not in (
        baseline_config["resume_identity"]
    )
    assert completion["convention_note"] == (
        search.SPEEDRUN_CONVENTION_NOTE
    )
    assert "RTA" in completion["convention_note"]
    assert completion["timing_convention"]["rta_audit_status"] == "PENDING"
    assert completion["formal_seed_consumption"] is False
    assert completion["formal_candidate_authority"] is False
    assert completion["training_authority"] is False
    assert completion["resumed_rows"] == 0
    assert completion["new_rows"] == seed_count
    assert completion["stream_runtime"]["parallel_envs"] == 2
    assert (
        completion["seed_discipline"]["formal_seed_consumption"] is False
    )
    json.dumps(completion, allow_nan=False)
    disk = json.loads(
        (run_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert disk["total_seeds"] == seed_count
    index = json.loads(
        (run_dir / "results_index.json").read_text(encoding="utf-8")
    )
    assert index["schema"] == search.INDEX_SCHEMA
    assert index["rows"] == seed_count
    assert index["wins"] == wins

    # Full resume: every seed already present is skipped; no new rows,
    # no duplicate lines, receipts rebuilt in place.
    resumed = search.run(**run_kwargs)
    assert resumed["resumed_rows"] == seed_count
    assert resumed["new_rows"] == 0
    assert resumed["total_seeds"] == seed_count
    assert (
        len(results_path.read_text(encoding="utf-8").splitlines())
        == seed_count
    )

    # Partial resume: keep two rows plus a TORN final line (a crash mid
    # append); the torn line is truncated away, the two kept rows
    # survive byte-identical, and only the missing seeds re-run.
    kept = lines[:2]
    results_path.write_text(
        "\n".join(kept) + "\n" + '{"seed": 14965',
        encoding="utf-8",
    )
    (run_dir / "completion.json").unlink()
    partial = search.run(**run_kwargs)
    assert partial["resumed_rows"] == 2
    assert partial["new_rows"] == 2
    final_lines = results_path.read_text(encoding="utf-8").splitlines()
    final_rows = [json.loads(line) for line in final_lines]
    assert len(final_rows) == seed_count
    assert {row["seed"] for row in final_rows} == {
        TEST_SEED_START + offset for offset in range(seed_count)
    }
    assert final_lines[:2] == kept

    # Resume identity is enforced: a different seed plan on the same
    # run dir refuses before touching any environment.
    mismatched = dict(run_kwargs)
    mismatched["seed_count"] = seed_count + 1
    with pytest.raises(ValueError, match="resume identity"):
        search.run(**mismatched)


# ---------------------------------------------------------------------------
# (d) Seed-file mode (seed_list_tas): parsing, per-seed discipline,
# CLI mutual exclusion -- pure CPU, no engine.
# ---------------------------------------------------------------------------


def _write_topk_jsonl(path: Path, seeds: list[int]) -> None:
    """Rows in the golden-seed enumerator's top-K output format."""

    lines = [
        json.dumps(
            {
                "schema": (
                    "zuma-rl.alphazuma-55-golden-seed-enumeration-v1"
                    "-topk-row"
                ),
                "level": "Jungle1",
                "rank": rank + 1,
                "seed": seed,
                "score": 22.0 - rank,
                "opening_colors": [1] * 10,
                "components": {
                    "longest_run": 6,
                    "runs_ge3": 2,
                    "adjacency": 8,
                    "distinct_colors": 2,
                },
            }
        )
        for rank, seed in enumerate(seeds)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_seed_file_formats_and_refusals(tmp_path: Path) -> None:
    seeds = [TEST_SEED_START + 100, TEST_SEED_START + 5, 42]
    # Enumerator JSONL rows: file order (rank order) is preserved.
    jsonl = tmp_path / "top_k.jsonl"
    _write_topk_jsonl(jsonl, seeds)
    assert search.load_seed_file(jsonl) == seeds
    # Bare-integer JSONL.
    (tmp_path / "bare.jsonl").write_text(
        "\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8"
    )
    assert search.load_seed_file(tmp_path / "bare.jsonl") == seeds
    # JSON array of integers, and of objects.
    (tmp_path / "array.json").write_text(
        json.dumps(seeds), encoding="utf-8"
    )
    assert search.load_seed_file(tmp_path / "array.json") == seeds
    (tmp_path / "objects.json").write_text(
        json.dumps([{"seed": seed} for seed in seeds]),
        encoding="utf-8",
    )
    assert search.load_seed_file(tmp_path / "objects.json") == seeds
    # JSON object with a "seeds" array.
    (tmp_path / "object.json").write_text(
        json.dumps({"seeds": seeds}), encoding="utf-8"
    )
    assert search.load_seed_file(tmp_path / "object.json") == seeds
    # Refusals: empty, duplicate, non-integer, out-of-range.
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        search.load_seed_file(tmp_path / "empty.jsonl")
    (tmp_path / "dupe.json").write_text(
        json.dumps([7, 7]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="more than once"):
        search.load_seed_file(tmp_path / "dupe.json")
    (tmp_path / "text.json").write_text(
        json.dumps(["seven"]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not an integer"):
        search.load_seed_file(tmp_path / "text.json")
    (tmp_path / "big.json").write_text(
        json.dumps([2**32]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside"):
        search.load_seed_file(tmp_path / "big.json")
    (tmp_path / "neg.json").write_text(
        json.dumps([-1]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside"):
        search.load_seed_file(tmp_path / "neg.json")


def test_validate_seed_list_embargo_and_reserved_discipline() -> None:
    clean = [TEST_SEED_START, TEST_SEED_START + 1]
    kept, dropped, reserved = search.validate_seed_list(clean)
    assert kept == clean
    assert dropped == []
    assert reserved == []
    # A frozen-embargo seed refuses loudly by default...
    embargoed = 1_600_000_000
    with pytest.raises(ValueError, match="embargo"):
        search.validate_seed_list([*clean, embargoed])
    # ...and is skipped + recorded with drop_embargoed.
    kept, dropped, reserved = search.validate_seed_list(
        [*clean, embargoed], drop_embargoed=True
    )
    assert kept == clean
    assert dropped == [
        {"seed": embargoed, "embargo": "formal_selection"}
    ]
    # Reserved engineering blocks refuse unless allow_reserved, and
    # the overlap is recorded either way.
    reserved_seed = 1_494_000_000
    with pytest.raises(ValueError, match="reserved"):
        search.validate_seed_list([*clean, reserved_seed])
    kept, dropped, reserved = search.validate_seed_list(
        [*clean, reserved_seed], allow_reserved=True
    )
    assert kept == [*clean, reserved_seed]
    assert reserved == [
        {
            "seed": reserved_seed,
            "block": "macro_decisions_collection_v1",
        }
    ]
    # A list that filters down to nothing is refused.
    with pytest.raises(ValueError, match="empty after"):
        search.validate_seed_list([embargoed], drop_embargoed=True)


def test_seed_file_cli_mutual_exclusion(tmp_path: Path) -> None:
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(json.dumps([TEST_SEED_START]), "utf-8")
    model_path = tmp_path / "model.zip"
    model_path.write_text("stub", "utf-8")
    common = [
        "--model-path",
        str(model_path),
        "--original-root",
        str(tmp_path),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        search.main(
            [
                *common,
                "--seed-file",
                str(seed_path),
                "--seed-start",
                "5",
            ]
        )
    with pytest.raises(SystemExit):
        search.main(
            [
                *common,
                "--seed-file",
                str(seed_path),
                "--seed-count",
                "5",
            ]
        )
    with pytest.raises(SystemExit):
        search.main([*common, "--drop-embargoed"])
    # run() enforces the same exclusion for programmatic callers.
    with pytest.raises(ValueError, match="mutually exclusive"):
        search.run(
            model_path=model_path,
            original_root=tmp_path,
            run_dir=tmp_path / "run2",
            seed_file=seed_path,
            seed_start=search.DEFAULT_SEED_START + 1,
        )


# ---------------------------------------------------------------------------
# (e) Real-engine seed-file smoke: the enumerator's top-K format
# streamed through the same tiny macro-native model, then resumed.
# ---------------------------------------------------------------------------


def test_real_engine_seed_file_smoke_and_resume(tmp_path: Path) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 300
    seeds = [
        TEST_SEED_START + 210,
        TEST_SEED_START + 205,
        TEST_SEED_START + 201,
    ]
    seed_dir = tmp_path / "golden"
    seed_dir.mkdir()
    seed_path = seed_dir / "top_k.jsonl"
    _write_topk_jsonl(seed_path, seeds)
    # A sibling enumerator completion travels into the provenance.
    (seed_dir / "completion.json").write_text(
        json.dumps(
            {
                "schema": (
                    "zuma-rl.alphazuma-55-golden-seed-enumeration-v1"
                ),
                "status": "COMPLETE",
            }
        ),
        encoding="utf-8",
    )

    factory = native_eval._make_stacked_macro_env_factory(
        original_root=RETAIL_ROOT,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
    )
    prototype_vector = DummyVecEnv([factory])
    try:
        model = _tiny_macro_native_stacked_model(prototype_vector)
    finally:
        prototype_vector.close()
    model_path = tmp_path / "tiny_macro_native.zip"
    model.save(str(model_path))

    run_dir = tmp_path / "seed-list-run"
    run_kwargs = dict(
        model_path=model_path,
        original_root=RETAIL_ROOT,
        run_dir=run_dir,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        parallel_envs=2,
        top_k=3,
        device="cpu",
        vec_env="dummy",
        seed_file=seed_path,
        seed_file_note="unit-test golden list",
    )
    completion = search.run(**run_kwargs)

    assert completion["convention"] == search.SEED_LIST_CONVENTION
    assert completion["convention_note"] == (
        search.SEED_LIST_CONVENTION_NOTE
    )
    assert (
        completion["timing_convention"]["convention_note"]
        == search.SEED_LIST_CONVENTION_NOTE
    )
    assert completion["seed_start"] is None
    assert completion["seed_last"] is None
    assert completion["seed_count"] == len(seeds)
    assert completion["total_seeds"] == len(seeds)
    source = completion["seed_source"]
    assert source["mode"] == "seed_list_tas"
    assert source["seed_file"]["path"] == str(seed_path)
    assert source["seed_file"]["sha256"].startswith("sha256:")
    assert source["seed_file"]["first_row_schema"] == (
        "zuma-rl.alphazuma-55-golden-seed-enumeration-v1-topk-row"
    )
    assert source["seed_file"]["sibling_completion"]["schema"] == (
        "zuma-rl.alphazuma-55-golden-seed-enumeration-v1"
    )
    assert source["requested_seeds"] == len(seeds)
    assert source["planned_seeds"] == len(seeds)
    assert source["dropped_embargoed"] == []
    assert source["reserved_overlaps"] == []
    assert source["note"] == "unit-test golden list"
    assert completion["formal_seed_consumption"] is False

    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["seed"] for row in rows} == set(seeds)
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    assert config["convention"] == search.SEED_LIST_CONVENTION
    identity = config["resume_identity"]
    assert identity["convention"] == search.SEED_LIST_CONVENTION
    assert identity["seed_file_sha256"] == (
        source["seed_file"]["sha256"]
    )
    assert identity["seed_count"] == len(seeds)
    assert "seed_start" not in identity

    # Resume: every listed seed already has a row; nothing re-runs.
    resumed = search.run(**run_kwargs)
    assert resumed["resumed_rows"] == len(seeds)
    assert resumed["new_rows"] == 0
    assert resumed["convention"] == search.SEED_LIST_CONVENTION

    # A different seed list on the same run dir refuses via the
    # recorded seed-file sha256.
    _write_topk_jsonl(seed_path, [*seeds, TEST_SEED_START + 233])
    with pytest.raises(ValueError, match="resume identity"):
        search.run(**run_kwargs)


# ---------------------------------------------------------------------------
# (f) Record-mode stochastic sampling: mask discipline, per-episode
# RNG reproducibility, temperature semantics -- pure CPU, no engine.
# ---------------------------------------------------------------------------


def test_sample_masked_macro_action_respects_masks_and_reproduces() -> None:
    logits = np.random.default_rng(20_260_817).normal(
        size=search.MACRO_MASK_WIDTH
    )
    mask = np.ones(search.MACRO_MASK_WIDTH, dtype=np.bool_)
    mask[0] = False  # wait_hold masked off
    mask[2] = False  # swap masked off: fire is the only legal verb
    mask[3 + 90 :] = False  # only aim bins 0..89 legal
    rng = search.stochastic_episode_rng(TEST_SEED_START)
    draws = [
        search.sample_masked_macro_action(logits, mask, 1.0, rng)
        for _ in range(300)
    ]
    for verb, aim in draws:
        assert int(verb) == 1
        assert 0 <= int(aim) < 90
    # It really samples: many distinct aims at temperature 1.
    assert len({int(aim) for _, aim in draws}) > 1

    def sequence(seed: int, temperature: float) -> np.ndarray:
        episode = search.stochastic_episode_rng(seed)
        return np.stack(
            [
                search.sample_masked_macro_action(
                    logits, mask, temperature, episode
                )
                for _ in range(50)
            ]
        )

    # Bit-for-bit reproducible from the same episode seed; a different
    # episode seed yields a different stream.
    assert np.array_equal(
        sequence(TEST_SEED_START, 1.3), sequence(TEST_SEED_START, 1.3)
    )
    assert not np.array_equal(
        sequence(TEST_SEED_START, 1.3),
        sequence(TEST_SEED_START + 1, 1.3),
    )
    # Refusals: bad temperatures, wrong widths, no legal choice.
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            search.sample_masked_macro_action(logits, mask, bad, rng)
    with pytest.raises(ValueError, match="width"):
        search.sample_masked_macro_action(logits[:-1], mask, 1.0, rng)
    with pytest.raises(ValueError, match="width"):
        search.sample_masked_macro_action(logits, mask[:-1], 1.0, rng)
    verbless = mask.copy()
    verbless[:3] = False
    with pytest.raises(RuntimeError, match="no legal choice"):
        search.sample_masked_macro_action(logits, verbless, 1.0, rng)


def test_sample_masked_macro_action_low_temperature_is_argmax() -> None:
    logits = np.random.default_rng(99).normal(
        size=search.MACRO_MASK_WIDTH
    )
    mask = np.ones(search.MACRO_MASK_WIDTH, dtype=np.bool_)
    # Mask off each segment's RAW argmax so agreement below proves the
    # sampler renormalizes over the masked logits, not the raw ones.
    mask[int(np.argmax(logits[:3]))] = False
    mask[3 + int(np.argmax(logits[3:]))] = False
    masked = np.where(mask, logits, -np.inf)
    expected_verb = int(np.argmax(masked[:3]))
    expected_aim = int(np.argmax(masked[3:]))
    rng = search.stochastic_episode_rng(TEST_SEED_START + 7)
    for _ in range(200):
        verb, aim = search.sample_masked_macro_action(
            logits, mask, 1e-6, rng
        )
        assert (int(verb), int(aim)) == (expected_verb, expected_aim)


def test_stochastic_convention_suffix_and_validation() -> None:
    assert search._stochastic_convention(
        search.BEST_OF_N_CONVENTION, 1.2
    ) == "best_of_n_tail_search_stochasticT1.2"
    assert search._stochastic_convention(
        search.SEED_LIST_CONVENTION, 2.0
    ) == "seed_list_tas_stochasticT2"
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            search._validate_stochastic_temperature(bad)
    assert search._validate_stochastic_temperature(0.7) == 0.7


class _ScriptedMacroVector:
    """Minimal vector honoring run_seed_stream's contract (CPU stub).

    Observations are deterministic per (seed, decision index) so a
    seed's episode replays identically regardless of which env slot
    runs it; episodes last ``2 + seed % 3`` macro decisions; every
    action received is recorded per seed so tests can compare whole
    action streams across runs.
    """

    def __init__(self, count: int, width: int = 6) -> None:
        self.num_envs = int(count)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(width,), dtype=np.float32
        )
        self._width = int(width)
        self._seed: list[int | None] = [None] * self.num_envs
        self._steps = [0] * self.num_envs
        self.actions_by_seed: dict[int, list[tuple[int, int]]] = {}

    def _observation(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(
            (int(self._seed[index]), int(self._steps[index]))
        )
        return rng.normal(size=self._width).astype(np.float32)

    def env_method(
        self, name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        assert name == "reset"
        (index,) = indices
        seed = int(kwargs["seed"])
        self._seed[index] = seed
        self._steps[index] = 0
        self.actions_by_seed.setdefault(seed, [])
        return [(self._observation(index), {})]

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        observations = np.zeros(
            (self.num_envs, self._width), dtype=np.float32
        )
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            seed = self._seed[index]
            info: dict[str, Any] = {
                "park_settle": {"ticks_consumed": 3, "released": True}
            }
            if seed is None:  # drained env: inert, telemetry ignored
                infos.append(info)
                continue
            self.actions_by_seed[seed].append(
                (int(actions[index][0]), int(actions[index][1]))
            )
            self._steps[index] += 1
            length = 2 + seed % 3
            if self._steps[index] >= length:
                dones[index] = True
                info.update(
                    {
                        "outcome": "loss",
                        "native_outcome": "loss",
                        "ticks": 3 * length,
                        "score": 100 + seed % 7,
                    }
                )
                self._seed[index] = None
            else:
                observations[index] = self._observation(index)
            infos.append(info)
        return observations, np.zeros(self.num_envs), dones, infos


def _stub_masks(vector: Any) -> np.ndarray:
    masks = np.ones(
        (vector.num_envs, search.MACRO_MASK_WIDTH), dtype=np.bool_
    )
    masks[:, 0] = False  # wait_hold masked off
    masks[:, 3 + 120 :] = False  # aim bins 120+ masked off
    return masks


def test_run_seed_stream_stochastic_masked_reproducible_distinct() -> None:
    seeds = [TEST_SEED_START + offset for offset in range(5)]
    # Near-flat logits so temperature sampling visibly spreads.
    logits = np.random.default_rng(7).normal(
        scale=0.1, size=search.MACRO_MASK_WIDTH
    )

    def provider(model: Any, observations: Any) -> np.ndarray:
        rows = np.asarray(observations).shape[0]
        return np.tile(logits, (rows, 1))

    def run_stochastic(count: int) -> dict[int, list[tuple[int, int]]]:
        vector = _ScriptedMacroVector(count)
        rows: list[dict[str, Any]] = []
        result = search.run_seed_stream(
            model=object(),
            vector=vector,
            level_id="Jungle1",
            pending_seeds=seeds,
            max_ticks=100,
            append_row=rows.append,
            masks_provider=_stub_masks,
            stochastic_temperature=1.2,
            logits_provider=provider,
        )
        assert {row["seed"] for row in result["rows"]} == set(seeds)
        return vector.actions_by_seed

    single = run_stochastic(1)
    doubled = run_stochastic(2)
    # Reproducible bit-for-bit AND independent of parallel-env
    # scheduling: per-seed action streams are identical either way.
    assert single == doubled
    # Masks respected exactly: the masked-off verb and aim bins never
    # appear in any sampled action.
    for actions in single.values():
        assert actions
        for verb, aim in actions:
            assert verb in (1, 2)
            assert 0 <= aim < 120
    # Truly stochastic: near-flat logits yield more than one distinct
    # action across the run...
    sampled = {
        action for actions in single.values() for action in actions
    }
    assert len(sampled) > 1

    # ...whereas the deterministic path on the SAME logits repeats the
    # single masked argmax forever -- so the two paths differ.
    class _ArgmaxModel:
        def predict(
            self,
            observations: Any,
            deterministic: bool = True,
            action_masks: Any = None,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            masks = np.asarray(action_masks, dtype=np.bool_)
            masked = np.where(
                masks, np.tile(logits, (masks.shape[0], 1)), -np.inf
            )
            verbs = np.argmax(masked[:, :3], axis=1)
            aims = np.argmax(masked[:, 3:], axis=1)
            return (
                np.stack([verbs, aims], axis=1).astype(np.int64),
                None,
            )

    vector = _ScriptedMacroVector(2)
    rows: list[dict[str, Any]] = []
    search.run_seed_stream(
        model=_ArgmaxModel(),
        vector=vector,
        level_id="Jungle1",
        pending_seeds=seeds,
        max_ticks=100,
        append_row=rows.append,
        masks_provider=_stub_masks,
    )
    deterministic = vector.actions_by_seed
    assert (
        len(
            {
                action
                for actions in deterministic.values()
                for action in actions
            }
        )
        == 1
    )
    assert deterministic != single


# ---------------------------------------------------------------------------
# (g) Real-engine record mode: settle tolerance reaches the frozen
# wrapper (contract echo), convention self-describes the sampler, and
# identical CLIs reproduce byte-identical rows.
# ---------------------------------------------------------------------------


def test_real_engine_record_mode_flags(tmp_path: Path) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_ticks = 300
    seed_count = 2
    factory = native_eval._make_stacked_macro_env_factory(
        original_root=RETAIL_ROOT,
        level_id="Jungle1",
        max_ticks=max_ticks,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
    )
    prototype_vector = DummyVecEnv([factory])
    try:
        model = _tiny_macro_native_stacked_model(prototype_vector)
    finally:
        prototype_vector.close()
    model_path = tmp_path / "tiny_macro_native.zip"
    model.save(str(model_path))

    def run_kwargs(run_dir: Path) -> dict[str, Any]:
        return dict(
            model_path=model_path,
            original_root=RETAIL_ROOT,
            run_dir=run_dir,
            level_id="Jungle1",
            seed_start=TEST_SEED_START + 300,
            seed_count=seed_count,
            max_ticks=max_ticks,
            hold_ticks=8,
            stack_lags=STACK_LAGS,
            parallel_envs=2,
            top_k=3,
            device="cpu",
            vec_env="dummy",
            stochastic_temperature=1.2,
            settle_tolerance_bins=3,
        )

    completion = search.run(**run_kwargs(tmp_path / "record-a"))

    # The convention self-describes the sampler and the receipts carry
    # the flag plus the RNG scheme.
    assert completion["convention"] == (
        "best_of_n_tail_search_stochasticT1.2"
    )
    stochastic = completion["stochastic_sampling"]
    assert stochastic["temperature"] == pytest.approx(1.2)
    assert stochastic["rng_scheme"] == search.STOCHASTIC_RNG_SCHEME
    assert stochastic["rng_stream_tag"] == (
        search.STOCHASTIC_RNG_STREAM_TAG
    )
    # (c) The tolerance reached the frozen wrapper: its own contract()
    # echo carries the effective value (and hold_ticks survived).
    assert completion["wrapper_contract"]["config"][
        "settle_tolerance_bins"
    ] == 3
    assert completion["wrapper_contract"]["config"]["hold_ticks"] == 8
    settle = completion["settle_tolerance_bins"]
    assert settle["requested"] == 3
    assert settle["effective"] == 3
    assert settle["wrapper_default"] == (
        search.DEFAULT_SETTLE_TOLERANCE_BINS
    )
    config = json.loads(
        (tmp_path / "record-a" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["convention"] == (
        "best_of_n_tail_search_stochasticT1.2"
    )
    identity = config["resume_identity"]
    assert identity["stochastic_temperature"] == pytest.approx(1.2)
    assert identity["settle_tolerance_bins"] == 3
    assert completion["total_seeds"] == seed_count
    assert completion["formal_seed_consumption"] is False

    # Bit-for-bit reproducibility: the same CLI in a fresh run dir
    # yields byte-identical rows.
    search.run(**run_kwargs(tmp_path / "record-b"))
    rows_a = (tmp_path / "record-a" / "results.jsonl").read_bytes()
    rows_b = (tmp_path / "record-b" / "results.jsonl").read_bytes()
    assert rows_a == rows_b

    # Record-mode flags change WHAT is measured, so they are part of
    # the resume identity.
    retempered = run_kwargs(tmp_path / "record-a")
    retempered["stochastic_temperature"] = 1.3
    with pytest.raises(ValueError, match="resume identity"):
        search.run(**retempered)
    retoleranced = run_kwargs(tmp_path / "record-a")
    retoleranced["settle_tolerance_bins"] = 2
    with pytest.raises(ValueError, match="resume identity"):
        search.run(**retoleranced)
