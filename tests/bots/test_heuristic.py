"""HeuristicBot (SPEC §6 M2): Dash Calls hopeless hands, bids and estimates from A/K/Q and trump
length, super-calls only with a very strong hand, and plays to its estimate, then ducks."""

import pytest

from estimation.bots.heuristic import HeuristicBot, HeuristicConfig, pick_estimate, target_tricks
from estimation.bots.random_bot import RandomBot
from estimation.engine import (
    CLUBS,
    DASH_CALL,
    HEARTS,
    NO_DASH_CALL,
    PASS,
    SANS,
    SPADES,
    WITH,
    Bola,
    Phase,
    Round,
    bid_action,
    estimate_action,
    parse_card,
    play_action,
)

from .helpers import round_with

NO = NO_DASH_CALL
NO_DASH = (NO, NO, NO, NO)
HOPELESS = "2C 3C 4C 5C 5D 6D 7D 2H 3H 4H 8S 9S TS"
SPADE_CALLER = "AS KS QS JS 9S 8S AH 2H 3D 4D 5D 2C 3C"  # 6.8 expected tricks in Spades
HEART_FIT = "AH KH QH 9H 8H 2S 3S 4S 4D 5D 6D 2C 3C"  # 4.5 expected tricks in Hearts


def act(rnd: Round, bot: HeuristicBot | None = None) -> int:
    seat = rnd.current_player
    assert seat is not None
    return (bot or HeuristicBot()).act(rnd.observation(seat), rnd.legal_actions())


# Dash Call


def test_dash_calls_a_hopeless_hand():
    assert act(round_with(0, HOPELESS)) == DASH_CALL


def test_does_not_dash_call_holding_an_ace():
    assert act(round_with(0, HOPELESS.replace("TS", "AS"))) == NO_DASH_CALL


def test_does_not_dash_call_with_a_five_card_suit():
    assert act(round_with(0, "2C 3C 4C 5C 6C 5D 6D 2H 3H 4H 8S 9S TS")) == NO_DASH_CALL


def test_dash_call_thresholds_are_configurable():
    strict = HeuristicBot(HeuristicConfig(dash_max_suit=3))
    assert act(round_with(0, HOPELESS), strict) == NO_DASH_CALL


# Auction


def test_opens_with_the_lowest_bid_in_its_best_trump():
    assert act(round_with(0, SPADE_CALLER, *NO_DASH)) == bid_action(4, SPADES)


def test_raises_to_its_target_on_its_final_turn():
    rnd = round_with(0, SPADE_CALLER, *NO_DASH, bid_action(4, SPADES), PASS, PASS, PASS)
    assert act(rnd) == bid_action(7, SPADES)


def test_keeps_its_bid_on_the_final_turn_when_already_at_target():
    rnd = round_with(0, SPADE_CALLER, *NO_DASH, bid_action(7, SPADES), PASS, PASS, PASS)
    assert act(rnd) == PASS


def test_says_with_when_its_hand_matches_the_top_bid():
    rnd = round_with(1, HEART_FIT, *NO_DASH, bid_action(5, HEARTS))
    assert act(rnd) == WITH


def test_outbids_when_its_hand_beats_the_top_bid():
    rnd = round_with(1, HEART_FIT, *NO_DASH, bid_action(4, CLUBS))
    assert act(rnd) == bid_action(4, HEARTS)


def test_passes_with_a_weak_hand():
    rnd = round_with(1, HOPELESS, *NO_DASH, bid_action(5, HEARTS))
    assert act(rnd) == PASS


def test_keeps_its_with_when_its_turn_comes_back():
    rnd = round_with(1, HEART_FIT, *NO_DASH, bid_action(5, HEARTS), WITH, PASS, PASS)
    assert rnd.current_player == 1
    assert act(rnd) == PASS


@pytest.mark.parametrize(
    ("expected", "target"),
    [(3.4, 3), (4.4, 4), (4.5, 5), (7.4, 7), (7.6, 7), (8.4, 7), (8.5, 9), (12.6, 13), (13.4, 13)],
)
def test_targets_round_expected_tricks_and_super_call_only_when_very_strong(
    expected: float, target: int
):
    assert target_tricks(expected, HeuristicConfig()) == target


def test_super_calls_in_a_fixed_round_with_a_much_better_trump():
    rnd = round_with(1, "AH KH QH JH TH 9H 8H 7H 6H 5H AS 2D 2C", round_no=14, fixed_trump=SANS)
    assert rnd.current_player == 1
    assert act(rnd) == bid_action(8, HEARTS)


def test_estimates_a_strong_fixed_trump_hand_instead_of_super_calling():
    rnd = round_with(1, "AS AH AD AC KS KH KD KC QS QH QD QC 2S", round_no=14, fixed_trump=SANS)
    assert act(rnd) == estimate_action(9)


# Estimation


def test_estimates_the_legal_value_nearest_its_expected_tricks():
    hand = "AH AD KS 2S 3H 4H 5D 6D 7S 8H 9D TH 2C"  # 2.75 expected tricks in Clubs
    rnd = round_with(1, hand, *NO_DASH, bid_action(4, CLUBS), PASS, PASS, PASS, PASS)
    assert rnd.phase is Phase.ESTIMATION
    assert act(rnd) == estimate_action(3)


def test_last_estimator_never_picks_the_forbidden_value():
    hand = "AH AD AS KH 2H 3S 4S 5D 6D 7D 8H 9S 2C"  # 3.75 expected tricks in Clubs
    auction = (bid_action(4, CLUBS), PASS, PASS, PASS, PASS)
    rnd = round_with(3, hand, *NO_DASH, *auction, estimate_action(3), estimate_action(2))
    assert estimate_action(4) not in rnd.legal_actions()
    assert act(rnd) == estimate_action(3)


def test_pick_estimate_takes_the_nearest_value_and_the_lower_on_a_tie():
    assert pick_estimate(2.75, [0, 1, 2, 3, 4], None, risk_aversion=0.25) == 3
    assert pick_estimate(2.5, [2, 3], None, risk_aversion=0.25) == 2


def test_pick_estimate_lets_the_last_estimator_avoid_a_risk():
    assert pick_estimate(2.4, [2, 3], 9, risk_aversion=0.25) == 3
    assert pick_estimate(2.4, [2, 3], 9, risk_aversion=0.0) == 2


# Card play


def test_leads_a_sure_winner_while_it_needs_tricks():
    auction = (bid_action(4, SPADES), PASS, PASS, PASS, bid_action(7, SPADES))
    estimates = (estimate_action(1), estimate_action(1), estimate_action(1))
    rnd = round_with(0, SPADE_CALLER, *NO_DASH, *auction, *estimates)
    assert rnd.phase is Phase.PLAY
    assert act(rnd) == play_action(parse_card("AH"))


def test_a_dash_caller_ducks_with_its_highest_losing_card():
    script = (NO, DASH_CALL, NO, NO, bid_action(4, SPADES), PASS, PASS, PASS)
    rnd = round_with(1, HOPELESS, *script, estimate_action(1), estimate_action(2))
    rnd.step(play_action(parse_card("QH")))
    assert rnd.current_player == 1
    assert act(rnd) == play_action(parse_card("4H"))


# Whole games


@pytest.mark.parametrize("lineup", ["HHHH", "HRHR", "RHHR"])
def test_always_chooses_a_legal_action_in_whole_games(lineup: str):
    for seed in range(4):
        bots = [
            HeuristicBot() if kind == "H" else RandomBot(seed * 4 + i)
            for i, kind in enumerate(lineup)
        ]
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            legal = bola.legal_actions()
            action = bots[seat].act(bola.observation(seat), legal)
            assert action in legal
            bola.step(action)


def test_is_deterministic():
    rnd = round_with(0, SPADE_CALLER, *NO_DASH)
    picks = {HeuristicBot().act(rnd.observation(0), rnd.legal_actions()) for _ in range(5)}
    assert len(picks) == 1
