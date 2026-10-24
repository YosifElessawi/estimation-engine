"""Noisy bots: a bot that sometimes plays a random legal move instead, for an Elo ladder of bots
between RandomBot and HeuristicBot."""

import pytest

from estimation.bots import HeuristicBot, make_bot
from estimation.bots.noisy import NoisyBot
from estimation.engine import Bola, Observation, Phase, Round, deal


class First:
    """Always plays the first legal action."""

    def act(self, obs: Observation, legal: list[int]) -> int:
        return legal[0]


def a_game_of_decisions(seed: int) -> list[tuple[Observation, list[int]]]:
    """Every decision of a HeuristicBot game, as (observation, legal actions)."""
    bot = HeuristicBot()
    bola = Bola(seed)
    decisions: list[tuple[Observation, list[int]]] = []
    while not bola.is_over:
        seat = bola.current_player
        assert seat is not None
        obs, legal = bola.observation(seat), bola.legal_actions()
        decisions.append((obs, legal))
        bola.step(bot.act(obs, legal))
    return decisions


def test_without_noise_it_plays_exactly_like_the_bot_it_wraps():
    noisy, plain = NoisyBot(HeuristicBot(), seed=1), HeuristicBot()
    for obs, legal in a_game_of_decisions(0):
        assert noisy.act(obs, legal) == plain.act(obs, legal)


def test_noise_replaces_that_share_of_decisions_with_random_legal_moves():
    obs = Round.new(1, deal(0, 1)).observation(0)
    legal = list(range(10))
    noisy = NoisyBot(First(), noise=0.2, seed=2)
    changed = sum(noisy.act(obs, legal) != 0 for _ in range(5000))
    assert 0.15 < changed / 5000 < 0.21  # 20% random, of which 9 in 10 differ from legal[0]


def test_play_noise_touches_only_card_play():
    noisy, plain = NoisyBot(HeuristicBot(), play_noise=1.0, seed=3), HeuristicBot()
    differs = 0
    for obs, legal in a_game_of_decisions(1):
        action = noisy.act(obs, legal)
        assert action in legal
        if obs.phase is Phase.PLAY:
            differs += action != plain.act(obs, legal)
        else:
            assert action == plain.act(obs, legal)
    assert differs > 50


def test_the_same_seed_gives_the_same_choices():
    decisions = a_game_of_decisions(2)
    first = NoisyBot(HeuristicBot(), noise=0.5, seed=4)
    second = NoisyBot(HeuristicBot(), noise=0.5, seed=4)
    picks = [first.act(obs, legal) for obs, legal in decisions]
    assert picks == [second.act(obs, legal) for obs, legal in decisions]


def test_noise_rates_must_be_between_0_and_1():
    with pytest.raises(ValueError, match="noise"):
        NoisyBot(HeuristicBot(), noise=1.5, seed=0)
    with pytest.raises(ValueError, match="noise"):
        NoisyBot(HeuristicBot(), play_noise=-0.1, seed=0)


def test_any_bot_takes_noise_options_by_name():
    bot = make_bot("heuristic:noise=0.2:risk_aversion=0.5", 7)
    assert isinstance(bot, NoisyBot)
    assert (bot.noise, bot.play_noise) == (0.2, None)
    assert isinstance(bot.inner, HeuristicBot)
    assert bot.inner.config.risk_aversion == 0.5
    played = make_bot("pimc:samples=4:play_noise=0.1", 7)
    assert isinstance(played, NoisyBot)
    assert played.play_noise == 0.1
    assert isinstance(make_bot("heuristic", 7), HeuristicBot)


@pytest.mark.parametrize("spec", ["heuristic:noise=2", "heuristic:noise=x", "pimc:play_noise=-1"])
def test_bad_noise_options_are_rejected(spec: str):
    with pytest.raises(ValueError, match="bot"):
        make_bot(spec, 0)
