"""The seeded random-play helper used by the property tests and the benchmark."""

import pytest

from estimation.engine.cards import deal
from estimation.engine.random_play import RandomPolicy, play_random_bola, play_random_round
from estimation.engine.round import Phase, Round


def test_random_policy_picks_from_the_legal_actions_reproducibly():
    picks = [RandomPolicy(5)([3, 7, 9]) for _ in range(3)]
    assert picks == [RandomPolicy(5)([3, 7, 9])] * 3
    policy = RandomPolicy(1)
    assert {policy([3, 7, 9]) for _ in range(200)} == {3, 7, 9}


def test_play_random_round_finishes_the_round_reproducibly():
    first = Round.new(1, deal(3, 1))
    second = Round.new(1, deal(3, 1))
    assert play_random_round(first, seed=4) == play_random_round(second, seed=4)
    assert first.phase is Phase.DONE
    assert first.log == second.log


def test_play_random_bola_plays_a_whole_game_reproducibly():
    first = play_random_bola(seed=2, policy_seed=9)
    second = play_random_bola(seed=2, policy_seed=9)
    assert first.is_over
    assert first.history == second.history
    assert first.history != play_random_bola(seed=2, policy_seed=10).history


def test_play_random_bola_stops_at_its_step_limit():
    with pytest.raises(RuntimeError, match="steps"):
        play_random_bola(seed=2, policy_seed=9, max_steps=10)
