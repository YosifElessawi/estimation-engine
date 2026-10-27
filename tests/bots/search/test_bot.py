"""The Search bot (SPEC §6 M3): PIMC card search with its own play-out rules, and bidding by
Yosif's rules or HeuristicBot's."""

import pytest

from estimation.bots import HeuristicBot, PIMCConfig, SearchBot, SearchConfig, card_play, make_bot
from estimation.bots.search import HEURISTIC, play_rules
from estimation.engine import Bola, Phase

CARDS = PIMCConfig(samples=2, oversample=1, leader_weight=True)


def test_the_search_bot_searches_with_its_own_play_out_rules():
    bot = SearchBot(0, SearchConfig(pimc=CARDS))
    # The card rule is internal to the bot; this pins which copy of the rules it searches with.
    assert bot._pimc._play is play_rules.choose_card  # pyright: ignore[reportPrivateUsage]
    assert play_rules.choose_card is not card_play.choose_card


def test_bidding_defaults_to_yosif_and_counts_players_ahead_double():
    config = SearchConfig()
    assert config.bidding == "yosif"
    assert config.pimc.samples == 30
    assert config.pimc.leader_weight


def test_heuristic_bidding_bids_like_heuristic_bot():
    bot = SearchBot(3, SearchConfig(bidding=HEURISTIC, pimc=CARDS))
    heuristic = HeuristicBot()
    checked = 0
    bola = Bola(5)
    while not bola.is_over and len(bola.history) < 6:
        seat = bola.current_player
        assert seat is not None
        obs, legal = bola.observation(seat), bola.legal_actions()
        action = bot.act(obs, legal)
        if obs.phase is not Phase.PLAY:
            assert action == heuristic.act(obs, legal)
            checked += 1
        bola.step(action)
    assert checked > 20


def test_heuristic_bidding_predicts_no_make_chances():
    bot = SearchBot(0, SearchConfig(bidding=HEURISTIC, pimc=CARDS))
    bola = Bola(0)
    assert bot.predicted_make(bola.observation(0), bola.legal_actions()[0]) is None


def test_an_unknown_bidding_is_refused():
    with pytest.raises(ValueError, match="bidding"):
        SearchBot(0, SearchConfig(bidding="wizard", pimc=CARDS))


def test_search_and_its_old_name_yosif_are_the_same_bot():
    old, new = make_bot("yosif:samples=3", 1), make_bot("search:samples=3", 1)
    assert isinstance(old, SearchBot)
    assert isinstance(new, SearchBot)
    assert old.config == new.config
