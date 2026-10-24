"""PIMCBot (SPEC §6 M3): sampled deals, HeuristicBot rollouts, best mean score advantage."""

import random
import time

import pytest

from estimation.bots.heuristic import HeuristicBot
from estimation.bots.pimc import PIMCBot, PIMCConfig, representatives
from estimation.engine import Bola, Observation, Phase, Round, deal, parse_card, parse_hand
from estimation.engine.cards import SUIT_MASKS

FAST = PIMCConfig(samples=6, oversample=2)


def play_position(seed: int, plays: int) -> Round:
    """A round played by HeuristicBots up to card play, then ``plays`` cards into it."""
    bot = HeuristicBot()
    round_no = 1 + seed % 18
    fixed = None if round_no <= 13 else (4, 3, 2, 1, 0)[round_no - 14]
    rnd = Round.new(round_no, deal(seed, round_no), fixed)
    while rnd.phase is not Phase.PLAY:
        seat = rnd.current_player
        assert seat is not None
        rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
    for _ in range(plays):
        seat = rnd.current_player
        assert seat is not None
        rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
    return rnd


def decide(bot: PIMCBot, rnd: Round) -> int:
    seat = rnd.current_player
    assert seat is not None
    return bot.act(rnd.observation(seat), rnd.legal_actions())


def test_dash_bids_and_estimates_are_heuristic_bots():
    pimc, heuristic = PIMCBot(1, FAST), HeuristicBot()
    for seed in range(30):
        bola = Bola(seed)
        while bola.phase is not Phase.PLAY:
            seat = bola.current_player
            assert seat is not None
            obs, legal = bola.observation(seat), bola.legal_actions()
            action = heuristic.act(obs, legal)
            assert pimc.act(obs, legal) == action
            bola.step(action)


def test_plays_its_only_legal_card_at_once():
    rnd = play_position(0, plays=48)  # the last trick: one card each
    assert len(rnd.legal_actions()) == 1
    assert decide(PIMCBot(0, FAST), rnd) == rnd.legal_actions()[0]


def test_searches_every_legal_card_by_default():
    assert not PIMCConfig().merge_equivalent


@pytest.mark.parametrize("merge", [False, True])
def test_always_plays_a_legal_card(merge: bool):
    bot = PIMCBot(3, PIMCConfig(samples=6, oversample=2, merge_equivalent=merge))
    for seed in range(40):
        rnd = play_position(seed, plays=seed % 48)
        assert decide(bot, rnd) in rnd.legal_actions()


def test_the_same_seed_gives_the_same_choices():
    positions = [play_position(seed, plays=seed % 40) for seed in range(15)]
    first = [decide(PIMCBot(7, FAST), rnd) for rnd in positions]
    assert first == [decide(PIMCBot(7, FAST), rnd) for rnd in positions]


def test_its_choice_depends_only_on_its_own_observation():
    """Swap unplayed cards of one suit between two other hands: the acting seat sees the same
    Observation, so it must choose the same card."""
    swapped = 0
    rng = random.Random(4)
    for seed in range(40):
        rnd = play_position(seed, plays=seed % 36)
        me = rnd.current_player
        assert me is not None
        a, b = [seat for seat in range(4) if seat != me][:2]
        views = [rnd.observation(seat) for seat in range(4)]
        for suit in rng.sample(range(4), 4):
            mine, theirs = views[a].hand & SUIT_MASKS[suit], views[b].hand & SUIT_MASKS[suit]
            if mine and theirs:
                swap = (mine & -mine) | (theirs & -theirs)
                hands = list(rnd.hands)
                hands[a] ^= swap
                hands[b] ^= swap
                other = Round.replay(
                    rnd.round_no,
                    (hands[0], hands[1], hands[2], hands[3]),
                    rnd.fixed_trump,
                    rnd.carry,
                    rnd.scores,
                    rnd.config,
                    [action for _, action in rnd.log],
                )
                assert other.observation(me) == rnd.observation(me)
                assert decide(PIMCBot(seed, FAST), other) == decide(PIMCBot(seed, FAST), rnd)
                swapped += 1
                break
    assert swapped > 25


def test_touching_cards_are_searched_once():
    out = parse_hand("AH 9H 2S")  # cards in other hands
    cards = [parse_card(c) for c in ("KH", "QH", "JH", "TH", "8H")]
    # K Q J T touch (nothing between them is out); 9H sits between TH and 8H.
    assert representatives(cards, out, hint=parse_card("8H")) == [
        parse_card("8H"),
        parse_card("TH"),
    ]
    assert representatives(cards, out, hint=parse_card("QH")) == [
        parse_card("QH"),
        parse_card("8H"),
    ]


def test_cards_of_different_suits_never_merge():
    cards = [parse_card("AC"), parse_card("2D")]
    assert sorted(representatives(cards, 0, hint=cards[0])) == sorted(cards)


def test_settings_are_validated():
    with pytest.raises(ValueError, match="samples"):
        PIMCBot(0, PIMCConfig(samples=0))


def test_a_one_sample_search_still_plays_legally():
    bot = PIMCBot(5, PIMCConfig(samples=1, oversample=1, merge_equivalent=False))
    for seed in range(10):
        rnd = play_position(seed, plays=seed * 3 % 40)
        assert decide(bot, rnd) in rnd.legal_actions()


def test_an_enforced_budget_cuts_the_search_short():
    bot = PIMCBot(6, PIMCConfig(samples=2000, oversample=1, budget_ms=1, enforce_budget=True))
    rnd = play_position(2, plays=0)
    seat = rnd.current_player
    assert seat is not None
    obs: Observation = rnd.observation(seat)
    start = time.perf_counter()
    action = bot.act(obs, rnd.legal_actions())
    assert time.perf_counter() - start < 2.0  # 2,000 full samples would take far longer
    assert action in rnd.legal_actions()
