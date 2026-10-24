"""RandomBot (SPEC §6 M2): picks uniformly from the legal actions, from its own seed."""

from estimation.bots.random_bot import RandomBot
from estimation.engine import Round, deal


def test_picks_one_of_the_legal_actions():
    rnd = Round.new(1, deal(0, 1))
    obs = rnd.observation(0)
    bot = RandomBot(1)
    assert {bot.act(obs, [3, 7, 9]) for _ in range(200)} == {3, 7, 9}


def test_the_same_seed_gives_the_same_choices():
    obs = Round.new(1, deal(0, 1)).observation(0)
    first, second = RandomBot(5), RandomBot(5)
    picks = [first.act(obs, list(range(10))) for _ in range(20)]
    assert picks == [second.act(obs, list(range(10))) for _ in range(20)]
    assert picks != [RandomBot(6).act(obs, list(range(10))) for _ in range(20)]
