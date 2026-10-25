"""Search bidding: Dash Call, auction and estimation chosen by simulating the round on sampled
deals and keeping the best average score advantage."""

import random

import pytest

from estimation.bots import HeuristicBot, PIMCBot, PIMCConfig, make_bot
from estimation.bots.bid_search import BidSearch
from estimation.engine import PASS, WITH, Bola, Phase, Round, RulesConfig
from estimation.engine.actions import is_estimate

FAST = PIMCConfig(samples=4, bid_search=True, bid_samples=4)


def bidding_positions(seed: int) -> list[Round]:
    """Every Dash Call, auction and estimation position of a HeuristicBot game's rounds."""
    bot = HeuristicBot()
    bola = Bola(seed)
    positions: list[Round] = []
    while not bola.is_over:
        seat = bola.current_player
        assert seat is not None
        if bola.phase is not Phase.PLAY:
            positions.append(bola.round.clone())
        bola.step(bot.act(bola.observation(seat), bola.legal_actions()))
    return positions


def searcher(seed: int = 0, samples: int = 4) -> BidSearch:
    config = PIMCConfig()
    return BidSearch(
        random.Random(seed), samples, 3, config.estimate_sigma, config.heuristic, RulesConfig()
    )


def test_bid_search_is_an_opt_in_setting():
    assert not PIMCConfig().bid_search
    assert PIMCConfig().bid_samples == 30
    bot = make_bot("pimc:bid_search=true:bid_samples=8", 1)
    assert isinstance(bot, PIMCBot)
    assert bot.config.bid_search
    assert bot.config.bid_samples == 8


def test_a_choice_is_scored_exactly_as_the_engine_plays_the_round_out():
    """With the true deal as its only sample, the value of each candidate equals the score
    advantage the engine gives when that action is taken and four HeuristicBots finish."""
    bot, search = HeuristicBot(), searcher()
    checked = 0
    for rnd in bidding_positions(3)[:60:3]:
        me = rnd.current_player
        assert me is not None
        obs = rnd.observation(me)
        truth = [rnd.observation(seat).hand for seat in range(4)]
        for action in search.candidates(
            obs, rnd.legal_actions(), bot.act(obs, rnd.legal_actions())
        ):
            value = search.evaluate(obs, [action], [truth])[0]
            world = rnd.clone()
            world.step(action)
            while world.phase is not Phase.DONE:
                seat = world.current_player
                assert seat is not None
                world.step(bot.act(world.observation(seat), world.legal_actions()))
            scores = world.result().scores
            assert value == pytest.approx(scores[me] - (sum(scores) - scores[me]) / 3)
            checked += 1
    assert checked > 50


def test_candidates_put_the_heuristic_choice_first_and_cover_the_key_actions():
    bot, search = HeuristicBot(), searcher()
    seen_auction = seen_estimate = False
    for rnd in bidding_positions(4):
        me = rnd.current_player
        assert me is not None
        obs, legal = rnd.observation(me), rnd.legal_actions()
        hint = bot.act(obs, legal)
        picks = search.candidates(obs, legal, hint)
        assert picks[0] == hint
        assert set(picks) <= set(legal)
        assert len(picks) == len(set(picks))
        if obs.phase is Phase.AUCTION:
            seen_auction = True
            assert PASS in picks
            assert WITH in picks or WITH not in legal
        if obs.phase is Phase.ESTIMATION:
            seen_estimate = True
            assert {a for a in legal if is_estimate(a)} <= set(picks)
    assert seen_auction
    assert seen_estimate


def test_always_chooses_a_legal_action_in_whole_games():
    for seed in range(2):
        bots = [PIMCBot(seed, FAST), HeuristicBot(), PIMCBot(seed + 9, FAST), HeuristicBot()]
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            legal = bola.legal_actions()
            action = bots[seat].act(bola.observation(seat), legal)
            assert action in legal
            bola.step(action)


def test_the_same_seed_gives_the_same_choices():
    positions = bidding_positions(5)[:40]
    first = [
        PIMCBot(2, FAST).act(r.observation(r.current_player or 0), r.legal_actions())
        for r in positions
    ]
    again = [
        PIMCBot(2, FAST).act(r.observation(r.current_player or 0), r.legal_actions())
        for r in positions
    ]
    assert first == again
