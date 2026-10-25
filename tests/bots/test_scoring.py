"""Scoring rollouts from one seat's side, with optional standings weights (design 5.9)."""

import pytest

from estimation.bots.scoring import RoundScorer, standings_weights
from estimation.engine import SPADES

from .helpers import play_observation

G1_ESTIMATES, G1_TRICKS = (5, 3, 4, 2), (5, 3, 3, 2)  # scores 25, 13, -11, 12 (SPEC §3.3)


def g1_observation():
    return play_observation(
        hand=0, estimates=G1_ESTIMATES, tricks_won=(0, 0, 0, 0), caller=0, trump=SPADES
    )


def test_standings_weights_with_equal_scores_are_the_plain_mean():
    assert standings_weights((0, 0, 0, 0), 0) == pytest.approx((0, 1 / 3, 1 / 3, 1 / 3))


def test_standings_weights_count_a_player_ahead_double():
    assert standings_weights((10, 50, 0, 0), 0) == pytest.approx((0, 0.5, 0.25, 0.25))


def test_a_scorer_with_even_weights_equals_the_mean():
    obs = g1_observation()
    even = RoundScorer(obs, weights=(0, 1 / 3, 1 / 3, 1 / 3)).value(G1_TRICKS)
    assert even == pytest.approx(RoundScorer(obs).value(G1_TRICKS))


def test_a_weighted_scorer_counts_the_leader_more():
    obs = g1_observation()
    value = RoundScorer(obs, weights=(0, 0.5, 0.25, 0.25)).value(G1_TRICKS)
    assert value == pytest.approx(25 - (0.5 * 13 + 0.25 * -11 + 0.25 * 12))
