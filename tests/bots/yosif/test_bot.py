"""Yosif's Dash Calls, bids and estimates (design 5.2-5.8), played by the Search bot, and their
acceptance hand."""

import dataclasses

from estimation.bots.pimc import PIMCConfig
from estimation.bots.search import SearchBot, SearchConfig
from estimation.bots.yosif import YosifConfig
from estimation.bots.yosif.odds import Outlook, stake
from estimation.bots.yosif.policy import Planner, best_number, needs_jump
from estimation.engine import (
    DASH_CALL,
    HEARTS,
    NO_DASH_CALL,
    PASS,
    SPADES,
    WITH,
    Bola,
    RulesConfig,
    bid_action,
    bid_of,
    estimate_action,
)
from tests.bots.helpers import LATEST_CALL_ONLY, round_with

# The rules tests check the decision logic on hands with clear counting, so they price calls by
# the counting itself; the calibrated default has its own tests (test_calibration.py).
CARDS = PIMCConfig(samples=2, oversample=1, leader_weight=True)
FAST = SearchConfig(pimc=CARDS, yosif=YosifConfig(calibrated=False))
SHIPPED = SearchConfig(pimc=CARDS)
EXAMPLE = "AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D"
RULES = RulesConfig()


def act(bot: SearchBot, rnd) -> int:
    seat = rnd.current_player
    assert seat is not None
    return bot.act(rnd.observation(seat), rnd.legal_actions())


def example_auction() -> tuple[int, int]:
    """Round 1 with the example hand in seat 0, the other seats passing: the bot's opening bid
    and the bid the auction ends on."""
    bot = SearchBot(0, FAST)
    rnd = round_with(0, EXAMPLE)
    rnd.step(act(bot, rnd))  # seat 0's Dash decision
    for _ in range(3):
        rnd.step(NO_DASH_CALL)
    opening = act(bot, rnd)
    rnd.step(opening)
    for _ in range(3):
        rnd.step(PASS)
    rnd.step(act(bot, rnd))  # the final turn
    final = rnd.observation(0).top_bid
    assert final is not None
    return opening, final


def test_example_hand_ends_the_auction_at_8_spades():
    assert example_auction()[1] == bid_action(8, SPADES)


def test_opening_bid_starts_low():
    assert example_auction()[0] == bid_action(4, SPADES)


def test_lone_bidder_never_bids_13_L3():
    bot = SearchBot(0, FAST)
    rnd = round_with(3, "AS KS QS JS TS 9S 8S 7S 6S 5S 4S 3S 2S", DASH_CALL, DASH_CALL, DASH_CALL)
    while rnd.current_player == 3:
        rnd.step(act(bot, rnd))
    top = rnd.observation(3).top_bid
    assert top is None or bid_of(top)[0] <= 12


def test_a_with_player_who_can_make_the_raise_joins_it_L1():
    # L1: seat 0 bids 4 Hearts, seat 1 bids 5 Hearts, seat 2 says With (w = 5), seat 3 passes,
    # seat 0 passes, and seat 1 raises to 6 Hearts on its final turn. Seat 2 holds six top
    # Hearts and nothing else, so it takes 6 and joins the raise.
    bot = SearchBot(0, FAST)
    rnd = round_with(
        2,
        "AH KH QH JH TH 9H 2C 3C 4C 2D 3D 4D 2S",
        *[NO_DASH_CALL] * 4,
        bid_action(4, HEARTS),
        bid_action(5, HEARTS),
        WITH,
        PASS,
        PASS,
        bid_action(6, HEARTS),
    )
    assert rnd.current_player == 2
    assert act(bot, rnd) == WITH


def test_a_with_player_who_cannot_make_the_raise_stays_L1():
    # Same auction, but seat 2's five top Hearts take 5: it stays, held at 5.
    bot = SearchBot(0, FAST)
    rnd = round_with(
        2,
        "AH KH QH JH TH 2C 3C 4C 5C 2D 3D 4D 2S",
        *[NO_DASH_CALL] * 4,
        bid_action(4, HEARTS),
        bid_action(5, HEARTS),
        WITH,
        PASS,
        PASS,
        bid_action(6, HEARTS),
    )
    assert rnd.current_player == 2
    assert act(bot, rnd) == PASS


def test_outbid_in_its_own_trump_it_passes_to_hold_its_bid():
    # Seat 0 bid 4 Hearts and seat 1 outbid it with 5 Hearts. Seat 0's four top Hearts take 4,
    # so passing (held at 4: With at 4 if Hearts wins, AUC-4) beats With at 5 or a raise.
    bot = SearchBot(0, FAST)
    rnd = round_with(
        0,
        "AH KH QH JH 2C 3C 4C 5C 2D 3D 4D 5D 2S",
        *[NO_DASH_CALL] * 4,
        bid_action(4, HEARTS),
        bid_action(5, HEARTS),
        PASS,
        PASS,
    )
    assert rnd.current_player == 0
    assert act(bot, rnd) == PASS


def test_final_turn_raises_when_keeping_is_not_allowed_AUC_6():
    # Under the With rules before 2026-10-30: seat 3 Dash Calls, seat 0 bids 4 Hearts, seat 1
    # says With, seat 2 bids 5 Hearts and seat 0 passes holding 4 Hearts. Seats 0 and 1 would
    # keep 4 each, so keeping 5 Hearts would total 13 with nobody to estimate.
    bot = SearchBot(0, FAST)
    rnd = round_with(
        2,
        "5H 4H 3H 2H 2C 3C 4C 5C 2D 3D 4D 5D 2S",
        NO_DASH_CALL,
        NO_DASH_CALL,
        NO_DASH_CALL,
        DASH_CALL,
        bid_action(4, HEARTS),
        WITH,
        bid_action(5, HEARTS),
        PASS,
        config=LATEST_CALL_ONLY,
    )
    assert rnd.current_player == 2
    legal = rnd.legal_actions()
    assert PASS not in legal
    assert bot.act(rnd.observation(2), legal) in legal


def test_with_is_priced_at_exactly_the_top_number():
    planner = Planner(FAST.yosif)
    o = Outlook((0.0,) * 6 + (1.0,), (0.0,) * 6 + (1.0,))
    expected = stake(4, caller=True, dash=False, last_risks=0, over=None, rules=RULES)
    from estimation.bots.yosif.odds import expected_points

    assert planner.with_value(o, 4) == expected_points(o, 4, expected, 0.5, RULES)


def test_last_estimator_banned_number_picks_the_best_legal_value_L4():
    planner = Planner(FAST.yosif)
    rnd = round_with(
        3,
        "AH 2H 3H 4H 2C 3C 4C 5C 2D 3D 4D 5D 6D",
        *[NO_DASH_CALL] * 4,
        bid_action(5, SPADES),
        PASS,
        PASS,
        PASS,
        PASS,
        estimate_action(3),
        estimate_action(4),
    )
    obs, legal = rnd.observation(3), rnd.legal_actions()
    values = planner.estimate_values(obs, legal)
    assert 1 not in values
    assert SearchBot(0, FAST).act(obs, legal) == estimate_action(
        max(values, key=values.__getitem__)
    )


def test_needs_jump_when_far_behind():
    obs = round_with(0, EXAMPLE, round_no=12).observation(0)
    assert needs_jump(dataclasses.replace(obs, scores=(0, 400, 0, 0)), 25.0)
    assert not needs_jump(dataclasses.replace(obs, scores=(0, 10, 0, 0)), 25.0)


def test_risk_mode_prefers_the_bigger_call():
    # make(8) = 0.7 and make(9) = 0.6: the expected points favour 8, the payoff when made 9.
    most = (0.0,) * 8 + (0.4, 0.6)
    least = (0.0,) * 8 + (0.7, 0.3)
    o = Outlook(most, least)

    def stake_of(k: int):
        return stake(k, caller=True, dash=False, last_risks=0, over=None, rules=RULES)

    assert best_number([8, 9], lambda k: o, stake_of, 0.5, RULES) == 8
    assert best_number([8, 9], lambda k: o, stake_of, 0.5, RULES, risk=True) == 9


def test_full_games_are_legal():
    for seed in range(3):
        bots = [SearchBot(seed * 4 + seat, FAST) for seat in range(4)]
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
        assert len(bola.history) >= 18


def test_a_weak_hand_does_not_say_with():
    # Seat 0 opens 4 Hearts; seat 1 holds nothing that can make 4 Hearts.
    bot = SearchBot(0, FAST)
    rnd = round_with(
        1, "2C 3C 4C 5C 2D 3D 4D 5D 2H 3H 4H 5H 6S", *[NO_DASH_CALL] * 4, bid_action(4, HEARTS)
    )
    assert rnd.current_player == 1
    assert act(bot, rnd) != WITH


def test_a_hand_of_low_cards_dash_calls():
    bot = SearchBot(0, FAST)
    rnd = round_with(0, "2C 3C 4C 5C 2D 3D 4D 5D 2H 3H 4H 5H 2S")
    assert act(bot, rnd) == DASH_CALL


def test_a_hand_with_aces_does_not_dash_call():
    bot = SearchBot(0, FAST)
    rnd = round_with(0, "AC 3C 4C 5C AD 3D 4D 5D AH 3H 4H 5H 2S")
    assert act(bot, rnd) == NO_DASH_CALL


def test_yosif_predicts_its_estimate_and_returns_none_for_an_opening():
    bot = SearchBot(0, FAST)
    opening = round_with(0, EXAMPLE, *[NO_DASH_CALL] * 4)
    assert bot.predicted_make(opening.observation(0), bid_action(4, SPADES)) is None
    estimating = round_with(
        1, EXAMPLE, *[NO_DASH_CALL] * 4, bid_action(5, 0), PASS, PASS, PASS, PASS
    )
    chance = bot.predicted_make(estimating.observation(1), estimating.legal_actions()[3])
    assert chance is not None
    assert 0.0 <= chance <= 1.0


def test_the_shipped_calibrated_bot_ends_the_example_auction_at_8_spades():
    bot = SearchBot(0, SHIPPED)
    assert bot.config.yosif.calibrated
    rnd = round_with(0, EXAMPLE)
    rnd.step(act(bot, rnd))
    for _ in range(3):
        rnd.step(NO_DASH_CALL)
    rnd.step(act(bot, rnd))
    for _ in range(3):
        rnd.step(PASS)
    rnd.step(act(bot, rnd))
    assert rnd.observation(0).top_bid == bid_action(8, SPADES)
