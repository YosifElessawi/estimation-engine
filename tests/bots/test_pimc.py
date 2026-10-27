"""PIMCBot (SPEC §6 M3): sampled deals, HeuristicBot rollouts, best mean score advantage."""

import dataclasses
import random
import time
from collections.abc import Sequence

import pytest

from estimation.bots import make_bot
from estimation.bots.card_play import choose_card
from estimation.bots.hand_eval import expected_tricks
from estimation.bots.heuristic import HeuristicBot
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import (
    PIMCBot,
    PIMCConfig,
    RoundScorer,
    fits_announcements,
    representatives,
)
from estimation.bots.sampling import deal_unseen
from estimation.engine import (
    CLUBS,
    NO_DASH_CALL,
    PASS,
    SANS,
    SPADES,
    Bola,
    Observation,
    Phase,
    Round,
    RulesConfig,
    TrickRecord,
    bid_action,
    deal,
    parse_card,
    parse_hand,
    play_action,
)
from estimation.engine.actions import PLAY_BASE
from estimation.engine.cards import SUIT_MASKS
from tests.engine.test_void_announcement import HANDS as VOID_HANDS

from .helpers import play_observation

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


def highest_legal(hand: int, trick: Sequence[int], trump: int, want: bool, out: int) -> int:
    """A card rule unlike HeuristicBot's: the highest legal card, whatever the estimate."""
    follow = hand & SUIT_MASKS[trick[0] // 13] if trick else 0
    return (follow or hand).bit_length() - 1


def test_its_own_card_rule_plays_every_seat_of_every_rollout():
    calls: list[int] = []

    def spy(hand: int, trick: Sequence[int], trump: int, want: bool, out: int) -> int:
        calls.append(hand)
        return choose_card(hand, trick, trump, want, out)

    positions = [play_position(seed, plays=4 + seed % 30) for seed in range(8)]
    counted = 0
    for rnd in positions:
        calls.clear()
        assert decide(PIMCBot(2, FAST, spy), rnd) == decide(PIMCBot(2, FAST), rnd)
        cards_left = sum(rnd.observation(seat).hand.bit_count() for seat in range(4))
        searched = len(rnd.legal_actions())
        if searched > 1:  # the first-card hint, then every other card left, in every rollout
            assert len(calls) == 1 + FAST.samples * searched * (cards_left - 1)
            counted += 1
    assert counted >= 4
    changed = 0
    for rnd in positions:
        seat = rnd.current_player
        assert seat is not None
        obs = rnd.observation(seat)
        cards = [action - PLAY_BASE for action in rnd.legal_actions()]
        if len(cards) > 1:
            plain = PIMCBot(2, FAST).search(obs, cards)
            other = PIMCBot(2, FAST, highest_legal).search(obs, cards)
            changed += plain != other
    assert changed > 0


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


def test_an_enforced_budget_stops_the_search_after_the_deal_in_progress():
    # A zero budget is spent before the first deal is rolled out, so only that deal counts:
    # the choice equals a one-deal search's. Without the cap, 12 deals change some choices.
    capped = PIMCConfig(samples=12, oversample=1, budget_ms=0, enforce_budget=True)
    one = PIMCConfig(samples=1, oversample=1)
    full = PIMCConfig(samples=12, oversample=1)
    differ = 0
    for seed in range(30):
        rnd = play_position(seed, plays=seed % 40)
        seat = rnd.current_player
        assert seat is not None
        obs, legal = rnd.observation(seat), rnd.legal_actions()
        action = PIMCBot(seed, capped).act(obs, legal)
        assert action == PIMCBot(seed, one).act(obs, legal)
        differ += PIMCBot(seed, full).act(obs, legal) != action
    assert differ > 0


# What PIMC decides


def test_the_defaults_are_the_specs_30_deals_and_250_ms():
    config = PIMCConfig()
    assert config.samples == 30
    assert config.budget_ms == 250.0
    assert not config.enforce_budget


def test_a_card_is_worth_own_score_minus_the_mean_of_the_other_three():
    # G1: Caller seat 0 at 5 Spades, estimates 5, 3, 4, 2 and tricks 5, 3, 3, 2 score
    # 25, 13, -11, 12 (SPEC §3.3).
    estimates, tricks = (5, 3, 4, 2), (5, 3, 3, 2)
    for seat, expected in [(0, 25 - (13 - 11 + 12) / 3), (2, -11 - (25 + 13 + 12) / 3)]:
        obs = play_observation(
            hand=0, estimates=estimates, tricks_won=(0, 0, 0, 0), caller=0, trump=SPADES, seat=seat
        )
        assert RoundScorer(obs).value(tricks) == pytest.approx(expected)


def test_the_scorer_scores_by_the_rules_it_is_given():
    obs = play_observation(
        hand=0, estimates=(5, 3, 4, 2), tricks_won=(0, 0, 0, 0), caller=0, trump=SPADES
    )
    richer = RulesConfig(caller_with_bonus=20)  # G1's Caller then scores 35, not 25
    assert RoundScorer(obs, richer).value((5, 3, 3, 2)) == pytest.approx(35 - (13 - 11 + 12) / 3)


def endgame(estimates: tuple[int, int, int, int], won: tuple[int, int, int, int]) -> Observation:
    """Two tricks left in Sans. Seat 0 is on lead with AS and 2H; the other three hold 3H-8H,
    two each. The 44 other cards went in 11 tricks that show no Heart voids."""
    left = parse_hand("AS 2H 3H 4H 5H 6H 7H 8H")
    played = [card for card in range(52) if not left >> card & 1]
    winners = [0, 1, 2, 1, 2, 1, 2, 1, 3, 0, 0]
    records: list[TrickRecord] = []
    leader = 0
    for i, winner in enumerate(winners):
        cards = (played[4 * i], played[4 * i + 1], played[4 * i + 2], played[4 * i + 3])
        records.append(TrickRecord(leader, cards, winner))
        leader = winner
    return play_observation(
        hand=parse_hand("AS 2H"),
        estimates=estimates,
        tricks_won=won,
        caller=0,
        trump=SANS,
        tricks=tuple(records),
    )


def test_plays_the_card_that_makes_its_contract():
    # Seat 0 needs one more trick. AS takes trick 12 and makes the contract (seat 0 is then the
    # only winner); leading 2H loses both tricks and every player fails.
    obs = endgame(estimates=(4, 2, 1, 0), won=(3, 4, 3, 1))
    legal = [play_action(parse_card("2H")), play_action(parse_card("AS"))]
    for seed in range(5):
        assert PIMCBot(seed).act(obs, legal) == play_action(parse_card("AS"))


def test_search_scores_by_the_configured_rules():
    obs = endgame(estimates=(4, 2, 1, 0), won=(3, 4, 3, 1))
    deal = [parse_hand("AS 2H"), parse_hand("3H 4H"), parse_hand("5H 6H"), parse_hand("7H 8H")]
    ace = [parse_card("AS")]
    plain = PIMCBot(0).evaluate(obs, ace, [deal])[0]
    richer = PIMCConfig(rules=RulesConfig(caller_with_bonus=20))
    # Seat 0 makes its contract as Caller, so a 10-point larger Caller bonus is 10 more for it.
    assert PIMCBot(0, richer).evaluate(obs, ace, [deal])[0] == pytest.approx(plain + 10)


def test_scores_a_card_on_a_deal_exactly_as_the_engine_plays_it_out():
    """With the true deal as its only sample, PIMC's value for each legal card equals the score
    advantage the engine gives when that card is played and four HeuristicBots finish."""
    bot = PIMCBot(0)
    heuristic = HeuristicBot()
    checked = 0
    for seed in range(40):
        rnd = play_position(seed, plays=(seed * 7) % 44)
        me = rnd.current_player
        assert me is not None
        obs = rnd.observation(me)
        truth = [rnd.observation(seat).hand for seat in range(4)]
        for action in rnd.legal_actions():
            card = action - 68
            value = bot.evaluate(obs, [card], [truth])[0]
            after = rnd.clone()
            after.step(action)
            while after.phase is not Phase.DONE:
                seat = after.current_player
                assert seat is not None
                after.step(heuristic.act(after.observation(seat), after.legal_actions()))
            scores = after.result().scores
            assert value == pytest.approx(scores[me] - (sum(scores) - scores[me]) / 3)
            checked += 1
    assert checked > 100


def test_sampled_deals_fit_hand_sizes_voids_and_played_cards():
    bot = PIMCBot(1, PIMCConfig(samples=10, oversample=3))
    voids_seen = 0
    for seed in range(40):
        rnd = play_position(seed, plays=(seed * 11) % 48)
        me = rnd.current_player
        assert me is not None
        obs = rnd.observation(me)
        know = read_knowledge(obs)
        deals = bot.sample_deals(obs)
        assert len(deals) == 10
        for hands in deals:
            assert hands[me] == obs.hand
            others = [seat for seat in range(4) if seat != me]
            assert hands[others[0]] | hands[others[1]] | hands[others[2]] == know.unseen
            for seat in others:
                assert hands[seat].bit_count() == know.sizes[seat]
                assert not hands[seat] & know.played
                for suit in range(4):
                    if know.voids[seat] >> suit & 1:
                        voids_seen += 1
                        assert not hands[seat] & SUIT_MASKS[suit]
    assert voids_seen > 100


def misfit(hands: list[int], obs: Observation) -> float:
    """How far the sampled hands of the other seats are from their estimates, in tricks²."""
    know = read_knowledge(obs)
    assert obs.trump is not None
    total = 0.0
    for seat in range(4):
        if seat == obs.seat:
            continue
        estimate = obs.estimates[seat]
        assert estimate is not None
        whole = hands[seat] | know.played_by[seat]
        total += (expected_tricks(whole, obs.trump) - estimate) ** 2
    return total


def test_kept_deals_fit_the_estimates_better_than_plain_deals():
    better = 0
    positions = 30
    for seed in range(positions):
        rnd = play_position(seed, plays=seed % 4)  # early in play, where estimates say most
        me = rnd.current_player
        assert me is not None
        obs = rnd.observation(me)
        if obs.dash_callers:
            continue
        kept = PIMCBot(seed, PIMCConfig(samples=30, oversample=3)).sample_deals(obs)
        plain = PIMCBot(seed, PIMCConfig(samples=30, oversample=1)).sample_deals(obs)
        if sum(misfit(h, obs) for h in kept) < sum(misfit(h, obs) for h in plain):
            better += 1
    assert better >= 0.8 * positions - 3


def test_card_decisions_stay_within_the_250_ms_budget():
    bot = PIMCBot(2)
    heuristic = HeuristicBot()
    slowest = 0.0
    for seed in range(4):
        rnd = play_position(seed, plays=0)
        me = rnd.current_player
        while rnd.phase is not Phase.DONE:
            seat = rnd.current_player
            assert seat is not None
            obs, legal = rnd.observation(seat), rnd.legal_actions()
            if seat == me:
                start = time.perf_counter()
                action = bot.act(obs, legal)
                slowest = max(slowest, time.perf_counter() - start)
            else:
                action = heuristic.act(obs, legal)
            rnd.step(action)
    assert slowest < 0.25


def test_leader_weight_is_off_by_default():
    assert not PIMCConfig().leader_weight


def test_leader_weight_counts_the_player_ahead_more():
    obs = endgame(estimates=(4, 2, 1, 0), won=(3, 4, 3, 1))
    deal = [parse_hand("AS 2H"), parse_hand("3H 4H"), parse_hand("5H 6H"), parse_hand("7H 8H")]
    cards = [parse_card("AS"), parse_card("2H")]
    weighted = PIMCConfig(leader_weight=True)
    # Level scores weight everyone equally, so the search is unchanged.
    assert PIMCBot(0, weighted).evaluate(obs, cards, [deal]) == pytest.approx(
        PIMCBot(0).evaluate(obs, cards, [deal])
    )
    behind = dataclasses.replace(obs, scores=(0, 100, 0, 0))
    expected = PIMCBot(0).evaluate(behind, cards, [deal])
    assert PIMCBot(0, weighted).evaluate(behind, cards, [deal]) != pytest.approx(expected)


def test_pimc_leader_weight_can_be_set_by_name():
    bot = make_bot("pimc:leader_weight=true", 0)
    assert isinstance(bot, PIMCBot)
    assert bot.config.leader_weight


def test_sampled_deals_fit_the_void_announcements():
    """TBL-7: seat 2 is announced void, so every sampled deal gives it a missing suit, and seats
    1 and 3 all four suits; plain dealing of the unseen cards would usually break both."""
    rnd = Round.new(1, VOID_HANDS)
    for action in (NO_DASH_CALL,) * 4 + (bid_action(4, CLUBS), PASS, PASS, PASS, PASS):
        rnd.step(action)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[0])
    obs = rnd.observation(0)
    assert obs.void_announced == (2,)
    know = read_knowledge(obs)
    for hands in PIMCBot(0).sample_deals(obs, know):
        three = [hands[1], hands[2], hands[3]]
        assert fits_announcements(three, (1, 2, 3), know.played_by, obs.void_announced)
    rng = random.Random(0)
    sizes = [know.sizes[seat] for seat in (1, 2, 3)]
    plain = [deal_unseen(know.unseen, sizes, [0, 0, 0], rng) for _ in range(50)]
    assert sum(fits_announcements(t, (1, 2, 3), know.played_by, (2,)) for t in plain) < 25
