"""Building bots by name, as the arena CLI does (SPEC §6 M2)."""

from pathlib import Path

import pytest

from estimation.bots import BOT_NAMES, Bot, make_bot
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig
from estimation.bots.pimc import PIMCBot, PIMCConfig
from estimation.bots.random_bot import RandomBot
from estimation.engine import RulesConfig
from tests.rl.helpers import make_checkpoint


def test_random_heuristic_pimc_and_rl_are_registered():
    assert {"random", "heuristic", "pimc", "rl"} <= set(BOT_NAMES)
    assert isinstance(make_bot("random", 1), RandomBot)
    assert isinstance(make_bot("heuristic", 1), HeuristicBot)


def test_options_after_a_colon_set_config_fields():
    bot = make_bot("heuristic:risk_aversion=0.5:dash_max_suit=3", 0)
    assert isinstance(bot, HeuristicBot)
    assert bot.config == HeuristicConfig(risk_aversion=0.5, dash_max_suit=3)


def test_pimc_takes_its_search_settings_as_options():
    bot = make_bot("pimc:samples=10:oversample=1:enforce_budget=true", 3)
    assert isinstance(bot, PIMCBot)
    assert bot.seed == 3
    assert bot.config == PIMCConfig(samples=10, oversample=1, enforce_budget=True)


def test_bots_are_built_for_the_rules_being_played():
    rules = RulesConfig(super_call_min=11)
    heuristic = make_bot("heuristic", 0, rules)
    assert isinstance(heuristic, HeuristicBot)
    assert heuristic.config.super_call_min == 11
    pimc = make_bot("pimc", 0, rules)
    assert isinstance(pimc, PIMCBot)
    assert pimc.config.rules == rules
    assert pimc.config.heuristic.super_call_min == 11
    override = make_bot("heuristic:super_call_min=9", 0, rules)
    assert isinstance(override, HeuristicBot)
    assert override.config.super_call_min == 9


def test_every_bot_satisfies_the_bot_interface(tmp_path: Path):
    required = {"rl": f":checkpoint={make_checkpoint(tmp_path / 'net.pt')}"}
    for name in BOT_NAMES:
        bot: Bot = make_bot(name + required.get(name, ""), 0)
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


def test_yosif_is_registered_with_its_options():
    from estimation.bots.yosif import YosifBot

    bot = make_bot("yosif:dash_break=0.5", 1)
    assert isinstance(bot, YosifBot)
    assert bot.config.dash_break == 0.5
    assert bot.config.pimc.leader_weight
