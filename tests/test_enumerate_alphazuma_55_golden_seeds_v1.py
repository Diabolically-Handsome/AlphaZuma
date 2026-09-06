from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tools import enumerate_alphazuma_55_golden_seeds_v1 as enum_mod
from zuma_rl.revenge_core import PopCapMTRandom


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


def _fixture_params(**overrides) -> enum_mod.LevelDrawParams:
    base = dict(
        colors=4,
        num_balls=0,
        opening_balls=10,
        curve_count=1,
        provenance="test fixture: probed live from the retail data "
        "on 2026-08-19",
    )
    base.update(overrides)
    return enum_mod.LevelDrawParams(**base)


# Frozen fixtures (pure test data; derivation is cross-checked against
# these AND against the real env below).  JUNGLE1 keeps the pure-math
# tests hermetic on machines without the retail extraction.
JUNGLE1 = _fixture_params(
    ball_repeat_chance=45, max_clump_size=6, max_single=2
)
LEVEL_FIXTURES = {
    "Jungle1": JUNGLE1,
    "Jungle4": _fixture_params(
        ball_repeat_chance=45, max_clump_size=5, max_single=4
    ),
    "jungle6": _fixture_params(
        ball_repeat_chance=39, max_clump_size=5, max_single=5
    ),
    "Jungle8": _fixture_params(
        ball_repeat_chance=39, max_clump_size=5, max_single=5
    ),
}
# Engineering seed region shared with the search tool's tests: outside
# every frozen embargo and reserved block.  (This enumerator plays no
# episodes, so it consumes no (level, seed) eval keys either way.)
TEST_SEED_START = 1_496_500_000
EDGE_SEEDS = (
    0,
    1,
    2,
    4357,
    0x7FFF_FFFF,
    0x8000_0000,
    2**32 - 2,
    2**32 - 1,
)


def _random_seeds(count: int, generator_seed: int) -> np.ndarray:
    sampler = np.random.default_rng(generator_seed)
    return sampler.integers(0, 2**32, size=count, dtype=np.int64)


# ---------------------------------------------------------------------------
# (a) Batch MTRand bitwise parity -- the load-bearing test.
# ---------------------------------------------------------------------------


def test_batch_mtrand_bitwise_parity_1000_seeds_x_64_draws() -> None:
    seeds = np.concatenate(
        [
            np.asarray(EDGE_SEEDS, dtype=np.int64),
            _random_seeds(1_000, 20_260_819),
        ]
    )
    outputs = enum_mod.batch_mtrand_u31(seeds, 64)
    assert outputs.shape == (len(seeds), 64)
    assert outputs.dtype == np.uint32
    # Signed-positive quirk: bit 31 is always clear.
    assert int(outputs.max()) <= 0x7FFF_FFFF
    for row, seed in enumerate(seeds):
        reference = PopCapMTRandom(int(seed))
        expected = [reference.next_u31() for _ in range(64)]
        assert outputs[row].tolist() == expected, f"seed {int(seed)}"


def test_batch_mtrand_parity_at_maximum_supported_outputs() -> None:
    seeds = np.concatenate(
        [np.asarray(EDGE_SEEDS[:4], dtype=np.int64), _random_seeds(8, 3)]
    )
    outputs = enum_mod.batch_mtrand_u31(
        seeds, enum_mod.MAX_MTRAND_OUTPUTS
    )
    for row, seed in enumerate(seeds):
        reference = PopCapMTRandom(int(seed))
        expected = [
            reference.next_u31()
            for _ in range(enum_mod.MAX_MTRAND_OUTPUTS)
        ]
        assert outputs[row].tolist() == expected, f"seed {int(seed)}"


def test_batch_mtrand_seed_zero_aliases_to_4357() -> None:
    outputs = enum_mod.batch_mtrand_u31(
        np.asarray([0, 4357], dtype=np.int64), 16
    )
    assert outputs[0].tolist() == outputs[1].tolist()


def test_batch_mtrand_refuses_unsupported_output_counts() -> None:
    seeds = np.asarray([1], dtype=np.int64)
    with pytest.raises(ValueError, match="n_outputs"):
        enum_mod.batch_mtrand_u31(seeds, 0)
    with pytest.raises(ValueError, match="n_outputs"):
        enum_mod.batch_mtrand_u31(
            seeds, enum_mod.MAX_MTRAND_OUTPUTS + 1
        )


# ---------------------------------------------------------------------------
# (b) Opening-color prediction: vectorized batch == scalar reference.
# ---------------------------------------------------------------------------


def test_batch_opening_colors_match_scalar_reference() -> None:
    seeds = np.concatenate(
        [
            np.asarray(EDGE_SEEDS, dtype=np.int64),
            _random_seeds(500, 99_081_701),
        ]
    )
    colors, fallbacks = enum_mod.batch_opening_colors(seeds, JUNGLE1)
    assert colors.shape == (len(seeds), JUNGLE1.opening_balls)
    assert int(colors.min()) >= 0
    assert int(colors.max()) < JUNGLE1.colors
    # At Jungle1's 4 colors the 64 precomputed outputs are ample; the
    # scalar fallback is expected to stay unused.
    assert fallbacks == 0
    for row, seed in enumerate(seeds):
        expected = enum_mod.predict_opening_colors_scalar(
            int(seed), JUNGLE1
        )
        assert colors[row].tolist() == expected, f"seed {int(seed)}"


def test_scalar_fallback_still_exact_when_forced() -> None:
    # Starve the batch walker of precomputed outputs so lanes overflow
    # and take the scalar-fallback path; the result must be unchanged.
    seeds = np.concatenate(
        [np.asarray(EDGE_SEEDS, dtype=np.int64), _random_seeds(64, 5)]
    )
    reference, _ = enum_mod.batch_opening_colors(
        seeds, JUNGLE1, n_outputs=64
    )
    starved, fallbacks = enum_mod.batch_opening_colors(
        seeds, JUNGLE1, n_outputs=8
    )
    assert fallbacks > 0
    assert np.array_equal(starved, reference)


# ---------------------------------------------------------------------------
# (c) MANDATORY draw-layout regression: predictions match the REAL
# Jungle1 environment's opening chain colors, closing the loop
# end-to-end (env seed -> MTRand -> opening layout).
# ---------------------------------------------------------------------------


def test_draw_layout_matches_real_jungle1_env() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from zuma_rl.revenge_env import RevengeEnv

    check_seeds = [TEST_SEED_START + offset for offset in range(24)]
    check_seeds += [0, 4357, 2**32 - 1]
    env = RevengeEnv(
        level_id="Jungle1", root=RETAIL_ROOT, seed=check_seeds[0]
    )
    try:
        assert env.sim.curve_count == JUNGLE1.curve_count
        assert env.sim.active_num_colors == JUNGLE1.colors
        parameters = env.sim.parameters
        assert int(parameters.ball_repeat_chance) == (
            JUNGLE1.ball_repeat_chance
        )
        assert int(parameters.max_clump_size) == JUNGLE1.max_clump_size
        assert int(parameters.max_single) == JUNGLE1.max_single
        assert int(getattr(parameters, "num_balls", 0)) == (
            JUNGLE1.num_balls
        )
        assert env.sim.config.initial_pending_balls == (
            JUNGLE1.opening_balls
        )
        predicted, _ = enum_mod.batch_opening_colors(
            np.asarray(check_seeds, dtype=np.int64), JUNGLE1
        )
        for row, seed in enumerate(check_seeds):
            env.reset(seed=seed)
            actual = [int(color) for color in env.sim.pending_colors]
            assert len(actual) == JUNGLE1.opening_balls
            assert actual == predicted[row].tolist(), f"seed {seed}"
            assert actual == enum_mod.predict_opening_colors_scalar(
                seed, JUNGLE1
            ), f"seed {seed}"
        # The documented seed alias: env seeds 0 and 4357 share one
        # MTRand stream and therefore one opening.
        env.reset(seed=0)
        opening_zero = list(env.sim.pending_colors)
        env.reset(seed=4357)
        assert list(env.sim.pending_colors) == opening_zero
    finally:
        env.close()


def test_derived_params_match_probed_fixtures() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    for level_id, expected in LEVEL_FIXTURES.items():
        derived = enum_mod.derive_level_draw_params(
            level_id, original_root=RETAIL_ROOT
        )
        for field in (
            "colors",
            "ball_repeat_chance",
            "max_clump_size",
            "max_single",
            "num_balls",
            "opening_balls",
            "curve_count",
        ):
            assert getattr(derived, field) == getattr(
                expected, field
            ), f"{level_id}.{field}"
        assert "derived at run time" in derived.provenance


def test_derivation_refuses_levels_outside_the_campaign_set() -> None:
    # Membership fails closed BEFORE any catalog IO, so no retail gate.
    with pytest.raises(ValueError, match="INCLUDED_LEVELS"):
        enum_mod.derive_level_draw_params("Boss1")


# Deterministic, level-distinct engineering seeds (2 per level), kept
# clear of the Jungle1 test's TEST_SEED_START + range(24) block.
EXTENDED_LEVEL_SEED_OFFSETS = {
    "Jungle4": 100,
    "jungle6": 102,
    "Jungle8": 104,
}


@pytest.mark.parametrize(
    "level_id", sorted(EXTENDED_LEVEL_SEED_OFFSETS)
)
def test_draw_layout_matches_real_env_for_derived_jungle_levels(
    level_id: str,
) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from zuma_rl.revenge_env import RevengeEnv

    params = enum_mod.derive_level_draw_params(
        level_id, original_root=RETAIL_ROOT
    )
    offset = EXTENDED_LEVEL_SEED_OFFSETS[level_id]
    check_seeds = [TEST_SEED_START + offset, TEST_SEED_START + offset + 1]
    env = RevengeEnv(
        level_id=level_id, root=RETAIL_ROOT, seed=check_seeds[0]
    )
    try:
        assert env.sim.curve_count == params.curve_count
        assert env.sim.active_num_colors == params.colors
        predicted, _ = enum_mod.batch_opening_colors(
            np.asarray(check_seeds, dtype=np.int64), params
        )
        for row, seed in enumerate(check_seeds):
            env.reset(seed=seed)
            actual = [int(color) for color in env.sim.pending_colors]
            assert len(actual) == params.opening_balls
            assert actual == predicted[row].tolist(), (
                f"{level_id} seed {seed}"
            )
            assert actual == enum_mod.predict_opening_colors_scalar(
                seed, params
            ), f"{level_id} seed {seed}"
    finally:
        env.close()


def test_live_check_passes_and_fails_closed() -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    receipt = enum_mod.run_live_check(
        level_id="Jungle1",
        original_root=RETAIL_ROOT,
        params=JUNGLE1,
        start_seed=TEST_SEED_START,
        end_seed=TEST_SEED_START + 1_000,
        sample_count=6,
    )
    assert receipt["performed"] is True
    assert receipt["mismatches"] == 0
    assert receipt["checked_seeds"] >= 6
    assert receipt["params_verified"]["colors"] == JUNGLE1.colors
    # Drifted frozen params must refuse loudly.
    drifted = enum_mod.LevelDrawParams(
        **{
            **{
                key: getattr(JUNGLE1, key)
                for key in (
                    "colors",
                    "ball_repeat_chance",
                    "max_clump_size",
                    "max_single",
                    "num_balls",
                    "opening_balls",
                    "curve_count",
                    "provenance",
                )
            },
            "ball_repeat_chance": 40,
        }
    )
    with pytest.raises(ValueError, match="drifted"):
        enum_mod.run_live_check(
            level_id="Jungle1",
            original_root=RETAIL_ROOT,
            params=drifted,
            start_seed=TEST_SEED_START,
            end_seed=TEST_SEED_START + 10,
            sample_count=2,
        )


# ---------------------------------------------------------------------------
# (d) Scorer units.
# ---------------------------------------------------------------------------


def test_opening_components_hand_computed_cases() -> None:
    colors = np.asarray(
        [
            [1] * 10,  # one run of 10
            [0, 1] * 5,  # fully alternating
            [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],  # runs 3, 2, 4, 1
            [2, 2, 2, 0, 2, 0, 0, 3, 3, 3],  # runs 3, 1, 1, 2, 3
        ],
        dtype=np.int8,
    )
    components = enum_mod.opening_components(colors, 4)
    assert components["longest_run"].tolist() == [10, 1, 4, 3]
    assert components["runs_ge3"].tolist() == [1, 0, 2, 2]
    assert components["adjacency"].tolist() == [9, 0, 6, 5]
    assert components["distinct_colors"].tolist() == [1, 2, 4, 3]


def test_score_openings_formula_and_weights() -> None:
    colors = np.asarray(
        [[1] * 10, [0, 1] * 5, [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]],
        dtype=np.int8,
    )
    scores, _ = enum_mod.score_openings(
        colors, enum_mod.OpeningScoreWeights(), 4
    )
    # Defaults: 2*longest + 3*runs3 + 1*adjacency - 2*distinct.
    assert scores.tolist() == [
        2 * 10 + 3 * 1 + 9 - 2 * 1,
        2 * 1 + 0 + 0 - 2 * 2,
        2 * 4 + 3 * 2 + 6 - 2 * 4,
    ]
    only_longest, _ = enum_mod.score_openings(
        colors,
        enum_mod.OpeningScoreWeights(
            longest_run=1.0, runs3=0.0, adjacency=0.0, distinct=0.0
        ),
        4,
    )
    assert only_longest.tolist() == [10.0, 1.0, 4.0]
    with pytest.raises(ValueError, match="2-D"):
        enum_mod.opening_components(
            np.empty((2, 0), dtype=np.int8), 4
        )


def test_select_top_k_orders_by_score_then_seed() -> None:
    seeds = np.asarray([9, 3, 7, 5], dtype=np.int64)
    scores = np.asarray([4.0, 8.0, 4.0, 1.0], dtype=np.float64)
    kept_seeds, kept_scores = enum_mod._select_top_k(seeds, scores, 3)
    assert kept_seeds.tolist() == [3, 7, 9]
    assert kept_scores.tolist() == [8.0, 4.0, 4.0]


# ---------------------------------------------------------------------------
# (e) Chunked enumeration: partial run, resume, determinism, receipts.
# ---------------------------------------------------------------------------


def test_chunk_enumeration_partial_resume_and_receipts(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "golden"
    kwargs = dict(
        level_id="Jungle1",
        out_dir=out_dir,
        start_seed=0,
        end_seed=16_384,
        chunk_size=4_096,
        top_k=64,
        threads=1,
        batch_size=2_048,
        # Hermetic engineering override (no retail data needed); real
        # runs derive params and gate them with the live check.
        params=JUNGLE1,
    )
    partial = enum_mod.run_enumeration(**kwargs, max_chunks=2)
    assert partial["status"] == "PARTIAL"
    assert partial["chunks"] == {
        "total": 4,
        "done": 2,
        "resumed": 0,
        "computed_this_run": 2,
    }
    chunk_dir = out_dir / enum_mod.CHUNK_DIRNAME
    assert len(list(chunk_dir.glob("chunk_*.npz"))) == 2
    progress = json.loads(
        (out_dir / enum_mod.PROGRESS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert progress["chunks_done"] == 2
    assert progress["chunks_total"] == 4
    assert not (out_dir / enum_mod.COMPLETION_FILENAME).exists()

    lines: list[str] = []
    completion = enum_mod.run_enumeration(
        **kwargs, progress=lines.append
    )
    assert completion["status"] == "COMPLETE"
    assert completion["chunks"] == {
        "total": 4,
        "resumed": 2,
        "computed_this_run": 2,
    }
    assert len(lines) == 2
    assert re.fullmatch(
        r"CHUNK 3/4 done rate=\d+ seeds/s topscore=-?\d+\.\d{3}",
        lines[0],
    )
    assert re.fullmatch(
        r"CHUNK 4/4 done rate=\d+ seeds/s topscore=-?\d+\.\d{3}",
        lines[1],
    )
    assert completion["top_k_kept"] == 64
    assert completion["scorer"]["weights"] == {
        "longest_run": 2.0,
        "runs3": 3.0,
        "adjacency": 1.0,
        "distinct": -2.0,
    }
    assert any(
        "revenge_core.py:1946-1991" in citation
        for citation in completion["draw_layout"]["citations"]
    )
    assert completion["formal_seed_consumption"] is False
    assert completion["live_check"] == {"performed": False}

    # top_k.jsonl equals a direct brute-force over the same range.
    rows = [
        json.loads(line)
        for line in (out_dir / enum_mod.TOPK_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 64
    assert [row["rank"] for row in rows] == list(range(1, 65))
    all_seeds = np.arange(0, 16_384, dtype=np.int64)
    all_colors, _ = enum_mod.batch_opening_colors(all_seeds, JUNGLE1)
    all_scores, _ = enum_mod.score_openings(
        all_colors, enum_mod.OpeningScoreWeights(), JUNGLE1.colors
    )
    best_seeds, best_scores = enum_mod._select_top_k(
        all_seeds, all_scores, 64
    )
    assert [row["seed"] for row in rows] == best_seeds.tolist()
    assert [row["score"] for row in rows] == best_scores.tolist()
    for row in rows[:5]:
        assert row["schema"] == enum_mod.TOPK_ROW_SCHEMA
        assert row["level"] == "Jungle1"
        assert len(row["opening_colors"]) == JUNGLE1.opening_balls
        assert set(row["components"]) == {
            "longest_run",
            "runs_ge3",
            "adjacency",
            "distinct_colors",
        }

    # Fully-resumed re-run recomputes nothing and reproduces receipts.
    resumed = enum_mod.run_enumeration(**kwargs)
    assert resumed["status"] == "COMPLETE"
    assert resumed["chunks"]["resumed"] == 4
    assert resumed["chunks"]["computed_this_run"] == 0

    # Identity mismatch refuses before any work.
    with pytest.raises(ValueError, match="resume identity"):
        enum_mod.run_enumeration(**{**kwargs, "top_k": 128})


def test_enumeration_is_deterministic_across_chunking_and_threads(
    tmp_path: Path,
) -> None:
    results: list[list[dict]] = []
    for label, chunk_size, batch_size, threads in (
        ("a", 4_096, 1_024, 2),
        ("b", 8_192, 4_096, 1),
    ):
        out_dir = tmp_path / label
        completion = enum_mod.run_enumeration(
            level_id="Jungle1",
            out_dir=out_dir,
            start_seed=0,
            end_seed=8_192,
            chunk_size=chunk_size,
            top_k=32,
            threads=threads,
            batch_size=batch_size,
            params=JUNGLE1,
        )
        assert completion["status"] == "COMPLETE"
        results.append(
            [
                json.loads(line)
                for line in (out_dir / enum_mod.TOPK_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        )
    assert results[0] == results[1]


def test_enumeration_refuses_unknown_level_and_bad_ranges(
    tmp_path: Path,
) -> None:
    # Level membership fails closed before any IO; seed-range checks
    # run even earlier, so neither branch needs the retail data.
    with pytest.raises(ValueError, match="INCLUDED_LEVELS"):
        enum_mod.run_enumeration(
            level_id="Boss1", out_dir=tmp_path / "x"
        )
    with pytest.raises(ValueError, match="seed range"):
        enum_mod.run_enumeration(
            level_id="Jungle1",
            out_dir=tmp_path / "y",
            start_seed=10,
            end_seed=10,
        )
    with pytest.raises(ValueError, match="seed range"):
        enum_mod.run_enumeration(
            level_id="Jungle1",
            out_dir=tmp_path / "z",
            start_seed=0,
            end_seed=2**32 + 1,
        )
