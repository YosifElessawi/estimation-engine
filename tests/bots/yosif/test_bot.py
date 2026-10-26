"""YosifBot's Dash Calls, bids and estimates (design 5.2-5.8), and its acceptance hand."""

import dataclasses

from estimation.bots.pimc import PIMCConfig
from estimation.bots.yosif import YosifBot, YosifConfig
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
from tests.bots.helpers import round_with

FAST = YosifConfig(pimc=PIMCConfig(samples=2, oversample=1, leader_weight=True))
EXAMPLE = "AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D"
RULES = RulesConfig()


def act(bot: YosifBot, rnd) -> int:
    seat = rnd.current_player
    assert seat is not None
    return bot.act(rnd.observation(seat), rnd.legal_actions())


def example_auction() -> tuple[int, int]:
    """Round 1 with the example hand in seat 0, the other seats passing: the bot's opening bid
    and the bid the auction ends on."""
    bot = YosifBot(0, FAST)
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
    bot = YosifBot(0, FAST)
    rnd = round_with(3, "AS KS QS JS TS 9S 8S 7S 6S 5S 4S 3S 2S", DASH_CALL, DASH_CALL, DASH_CALL)
    while rnd.current_player == 3:
        rnd.step(act(bot, rnd))
    top = rnd.observation(3).top_bid
    assert top is None or bid_of(top)[0] <= 12


def test_last_estimator_banned_number_picks_the_best_legal_value_L4():
    planner = Planner(FAST)
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
    assert YosifBot(0, FAST).act(obs, legal) == estimate_action(max(values, key=values.__getitem__))


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
        bots = [YosifBot(seed * 4 + seat, FAST) for seat in range(4)]
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
        assert len(bola.history) >= 18


def test_a_weak_hand_does_not_say_with():
    # Seat 0 opens 4 Hearts; seat 1 holds nothing that can make 4 Hearts.
    bot = YosifBot(0, FAST)
    rnd = round_with(
        1, "2C 3C 4C 5C 2D 3D 4D 5D 2H 3H 4H 5H 6S", *[NO_DASH_CALL] * 4, bid_action(4, HEARTS)
    )
    assert rnd.current_player == 1
    assert act(bot, rnd) != WITH


def test_a_hand_of_low_cards_dash_calls():
    bot = YosifBot(0, FAST)
    rnd = round_with(0, "2C 3C 4C 5C 2D 3D 4D 5D 2H 3H 4H 5H 2S")
    assert act(bot, rnd) == DASH_CALL


def test_a_hand_with_aces_does_not_dash_call():
    bot = YosifBot(0, FAST)
    rnd = round_with(0, "AC 3C 4C 5C AD 3D 4D 5D AH 3H 4H 5H 2S")
    assert act(bot, rnd) == NO_DASH_CALL
