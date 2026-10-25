"""The standings-weighted reward and the match-win model (SPEC §6 M4)."""

from typing import Any

import pytest

from estimation.rl.standings import (
    RewardConfig,
    first_place_shares,
    opponent_weights,
    points_reward,
    round_reward,
    rounds_remaining,
    standings_weights,
    win_probabilities,
)


def test_opponents_ahead_count_fully_and_those_behind_count_half():
    assert standings_weights((10, 20, 5, 10), 0) == (0.0, 1.0, 0.5, 1.0)
    assert standings_weights((10, 20, 5, 10), 1) == (0.5, 0.0, 0.5, 0.5)
    assert standings_weights((10, 20, 5, 10), 2) == (1.0, 1.0, 0.0, 1.0)


def test_the_weights_are_config_values():
    config = RewardConfig(ahead_weight=0.8, behind_weight=0.2, tie_weight=0.6)
    assert standings_weights((10, 20, 5, 10), 0, config) == (0.0, 0.8, 0.2, 0.6)
    assert opponent_weights((10, 20, 5, 10), 0, config) == pytest.approx((0.0, 0.5, 0.125, 0.375))


def test_points_reward_is_own_change_minus_weighted_opponent_changes():
    literal = RewardConfig(normalize_weights=False)
    # Seat 0 trails seat 1, leads seat 2 and is level with seat 3 before the round.
    reward = points_reward((10, 20, 5, 10), (15, -11, 13, 12), 0, literal)
    assert reward == 15 - (1.0 * -11 + 0.5 * 13 + 1.0 * 12)
    # In round 1 everyone is level, so every opponent counts fully.
    assert points_reward((0, 0, 0, 0), (25, 13, -11, 12), 2, literal) == -11 - (25 + 13 + 12)


def test_by_default_the_opponents_term_is_a_weighted_mean():
    reward = points_reward((10, 20, 5, 10), (15, -11, 13, 12), 0)
    assert reward == pytest.approx(15 - (1.0 * -11 + 0.5 * 13 + 1.0 * 12) / 2.5)
    assert opponent_weights((10, 20, 5, 10), 0) == pytest.approx((0.0, 0.4, 0.2, 0.4))
    # An opponent ahead still counts twice as much as one behind.
    assert points_reward((0, 0, 0, 0), (25, 13, -11, 12), 2) == pytest.approx(-11 - 50 / 3)


def test_a_round_where_everyone_scores_the_same_is_worth_as_much_as_a_skipped_one():
    for before in [(0, 0, 0, 0), (50, 20, 5, -10), (5, 90, 90, 0)]:
        for seat in range(4):
            assert points_reward(before, (15, 15, 15, 15), seat) == pytest.approx(0.0)
            assert points_reward(before, (0, 0, 0, 0), seat) == 0.0
    # Without normalizing, the same round costs every seat: passing beats playing.
    assert (
        points_reward((0, 0, 0, 0), (15, 15, 15, 15), 0, RewardConfig(normalize_weights=False))
        == -30
    )


def test_first_place_is_shared_on_a_tie():
    assert first_place_shares((5, 9, 9, 1)) == (0.0, 0.5, 0.5, 0.0)
    assert first_place_shares((0, 0, 0, 0)) == (0.25, 0.25, 0.25, 0.25)


def test_with_no_rounds_left_the_win_probability_is_the_result():
    assert win_probabilities((100, 90, 80, 120), 0) == (0.0, 0.0, 0.0, 1.0)


def test_win_probabilities_are_a_distribution_that_favours_the_leader():
    level = win_probabilities((50, 50, 50, 50), 5)
    assert level == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=1e-9)
    probs = win_probabilities((120, 90, 60, 100), 5)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[3] > probs[1] > probs[2] > 0
    # A lead counts for more as the game nears its end.
    assert win_probabilities((120, 90, 60, 100), 1)[0] > probs[0]
    assert win_probabilities((400, 0, 0, 0), 2)[0] == pytest.approx(1.0, abs=1e-6)


def test_rounds_remaining_counts_open_rounds_and_fixed_trumps():
    assert rounds_remaining(1, 5, fixed_round=False) == 18
    assert rounds_remaining(13, 5, fixed_round=False) == 6
    assert rounds_remaining(14, 5, fixed_round=True) == 5
    assert rounds_remaining(19, 1, fixed_round=True) == 1


def test_open_rounds_are_rewarded_on_points_alone():
    before, change = (10, 20, 5, 10), (15, -11, 13, 12)
    assert round_reward(before, change, 0, False, 10, 9) == points_reward(before, change, 0)


def test_fixed_trump_rounds_blend_in_the_change_in_match_win_probability():
    config = RewardConfig(win_blend=0.5, win_points=100.0)
    before, change = (100, 90, 80, 95), (-11, 25, 12, 13)
    after = (89, 115, 92, 108)
    reward = round_reward(before, change, 0, True, 3, 2, config)
    gain = win_probabilities(after, 2)[0] - win_probabilities(before, 3)[0]
    assert reward == pytest.approx(0.5 * points_reward(before, change, 0) + 50.0 * gain)
    assert gain < 0  # seat 0 lost its lead


def test_the_last_round_scores_the_actual_match_result():
    config = RewardConfig(win_blend=1.0, win_points=100.0)
    # Seat 1 overtakes seat 0 in the final round: seat 1 gains the win, seat 0 loses it.
    before, change = (100, 95, 50, 40), (-1, 13, 0, 0)
    p_before = win_probabilities(before, 1)
    assert round_reward(before, change, 1, True, 1, 0, config) == pytest.approx(
        100.0 * (1.0 - p_before[1])
    )
    assert round_reward(before, change, 0, True, 1, 0, config) == pytest.approx(
        100.0 * (0.0 - p_before[0])
    )


def test_without_the_blend_fixed_trump_rounds_score_like_open_ones():
    config = RewardConfig(win_blend=0.0)
    before, change = (100, 90, 80, 95), (-11, 25, 12, 13)
    assert round_reward(before, change, 3, True, 3, 2, config) == points_reward(
        before, change, 3, config
    )


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"win_blend": 1.5}, "win_blend"),
        ({"win_blend": -0.1}, "win_blend"),
        ({"round_sd": 0.0}, "round_sd"),
    ],
)
def test_bad_reward_configs_are_rejected(bad: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        RewardConfig(**bad)
