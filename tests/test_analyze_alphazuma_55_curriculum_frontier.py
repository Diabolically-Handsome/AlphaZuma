from tools import analyze_alphazuma_55_curriculum_frontier as analysis


def test_rescue_weight_replays_solved_levels() -> None:
    assert analysis._rescue_weight(
        level_id="Jungle1", episodes=20, wins=10, truncations=0
    ) == 0.5
    assert analysis._rescue_weight(
        level_id="Jungle2", episodes=20, wins=10, truncations=0
    ) == 1.0


def test_rescue_weight_prioritizes_repeated_zero_win_level() -> None:
    assert analysis._rescue_weight(
        level_id="Coast3", episodes=89, wins=0, truncations=3
    ) == 3.5


def test_rescue_weight_adds_dual_and_truncation_bonuses() -> None:
    assert analysis._rescue_weight(
        level_id="grotto10", episodes=68, wins=0, truncations=61
    ) == 4.75
