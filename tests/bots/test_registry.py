"""Building bots by name, as the arena CLI does (SPEC §6 M2)."""

import pytest

from estimation.bots import BOT_NAMES, Bot, make_bot
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig
from estimation.bots.random_bot import RandomBot


def test_random_and_heuristic_are_registered():
    assert {"random", "heuristic"} <= set(BOT_NAMES)
    assert isinstance(make_bot("random", 1), RandomBot)
    assert isinstance(make_bot("heuristic", 1), HeuristicBot)


def test_options_after_a_colon_set_config_fields():
    bot = make_bot("heuristic:risk_aversion=0.5:dash_max_suit=3", 0)
    assert isinstance(bot, HeuristicBot)
    assert bot.config == HeuristicConfig(risk_aversion=0.5, dash_max_suit=3)


def test_every_bot_satisfies_the_bot_interface():
    for name in BOT_NAMES:
        bot: Bot = make_bot(name, 0)
        assert callable(bot.act)


@pytest.mark.parametrize(
    "spec",
    [
        "nobody",
        "heuristic:nope=1",
        "heuristic:risk_aversion",
        "random:x=1",
        "heuristic:dash_max_suit=x",
    ],
)
def test_unknown_names_and_bad_options_are_rejected(spec: str):
    with pytest.raises(ValueError, match="bot"):
        make_bot(spec, 0)
