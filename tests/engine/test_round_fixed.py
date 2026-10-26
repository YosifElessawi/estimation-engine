"""Fixed-trump rounds inside one Round: FIX-2 to FIX-4, L6 to L9, and G7, G9, G10 end to end."""

import dataclasses

import pytest

from estimation.engine.actions import PASS, WITH, bid_action, bid_of, is_bid
from estimation.engine.cards import CLUBS, DIAMONDS, HEARTS, SANS, SPADES, deal
from estimation.engine.config import RulesConfig
from estimation.engine.observation import WithInfo
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import bid, build_play, est, legal_bids, legal_estimates, new_round, play_cards, run


def super_calls(fixed_trump: int, minimum: int = 8) -> list[int]:
    trumps = (CLUBS, DIAMONDS, HEARTS, SPADES, SANS)
    return [bid_action(n, t) for n in range(minimum, 14) for t in trumps if t != fixed_trump]


def test_FIX_2_everyone_estimates_0_to_13_from_F_with_no_minimum():
    rnd = new_round(14, SANS)
    order = []
    for k in (0, 1, 2):
        order.append(rnd.current_player)
        assert legal_estimates(rnd) == list(range(14))
        rnd.step(est(k))
    order.append(rnd.current_player)
    assert order == [1, 2, 3, 0]


def test_FIX_2_sum_rule_applies_to_the_last_estimator():
    rnd = run(new_round(14, SANS), est(4), est(4), est(4))
    assert legal_estimates(rnd) == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    with pytest.raises(IllegalActionError):
        rnd.step(est(1))


def test_FIX_2_risk_applies_to_the_last_estimator():
    rnd = run(new_round(14, SANS), est(0), est(1), est(1), est(0))
    obs = rnd.observation(0)
    assert (obs.total, obs.over, obs.last_estimator, obs.risks) == (2, False, 0, 5)


def test_FIX_3_highest_estimate_is_caller_and_first_said_wins_ties():
    rnd = run(new_round(14, SANS), est(3), est(5), est(5), est(1))
    assert rnd.phase is Phase.PLAY
    obs = rnd.observation(1)
    assert obs.caller == 2
    assert obs.with_players == (WithInfo(3, 5, SANS),)
    assert obs.trump == SANS
    assert rnd.current_player == 2  # the Caller leads


def test_FIX_3_last_estimator_can_be_caller():
    rnd = run(new_round(14, SANS), est(1), est(2), est(2), est(7))
    obs = rnd.observation(0)
    assert obs.caller == 0
    assert obs.with_players == ()
    assert (obs.last_estimator, obs.risks) == (0, 0)


def test_FIX_4_super_call_clears_estimates_and_opens_an_auction():
    rnd = run(new_round(16, HEARTS), est(2))
    rnd.step(bid(8, SPADES))
    assert rnd.phase is Phase.AUCTION
    obs = rnd.observation(1)
    assert obs.super_call
    assert obs.estimates == (None, None, None, None)
    assert (obs.top_bid, obs.top_bidder) == (bid(8, SPADES), 0)
    assert obs.trump is None
    assert rnd.current_player == 1  # the player after the super caller


def test_FIX_4_super_call_is_8_or_more_in_another_trump():
    rnd = new_round(15, SPADES)
    assert legal_bids(rnd) == super_calls(SPADES)
    for illegal in (bid(7, HEARTS), bid(8, SPADES), bid(13, SPADES)):
        with pytest.raises(IllegalActionError):
            rnd.step(illegal)


def test_FIX_4_super_call_minimum_follows_config():
    config = dataclasses.replace(RulesConfig(), super_call_min=10)
    rnd = Round.new(15, deal(0, 15), SPADES, config=config)
    assert legal_bids(rnd) == super_calls(SPADES, minimum=10)


def test_FIX_4_answers_may_be_in_any_trump_including_the_fixed_one():
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES))
    assert legal_bids(rnd) == list(range(bid(8, SPADES) + 1, bid(13, SANS) + 1))
    assert bid(9, HEARTS) in rnd.legal_actions()
    assert WITH in rnd.legal_actions()


def test_FIX_4_with_answers_and_final_turn_work_in_the_super_call_auction():
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES), WITH, PASS, PASS)
    assert rnd.current_player == 0  # the super caller's final turn
    rnd.step(bid(9, SPADES))
    assert rnd.current_player == 1  # seat 1 answers the raise (AUC-3)
    rnd.step(PASS)
    obs = rnd.observation(0)
    assert (obs.caller, obs.trump, obs.estimates[0]) == (0, SPADES, 9)
    assert obs.with_players == (WithInfo(1, 8, SPADES),)
    assert obs.estimates[1] == 8  # EST-2: a With player's estimate is its w
    assert rnd.current_player == 2
    assert legal_estimates(rnd) == list(range(10))


@pytest.mark.parametrize("may_bid_fixed", [True, False])
def test_FIX_4_super_caller_answering_a_raise_follows_the_fixed_trump_switch(may_bid_fixed: bool):
    config = RulesConfig(super_caller_may_bid_fixed=may_bid_fixed)
    rnd = Round.new(16, deal(0, 16), HEARTS, config=config)
    run(rnd, est(2), bid(8, SPADES), bid(9, SPADES), PASS, PASS, WITH)  # seat 0 joins seat 1
    assert rnd.current_player == 1  # seat 1's final turn
    rnd.step(bid(10, SPADES))
    assert rnd.current_player == 0  # the super caller answers
    hearts = [a for a in legal_bids(rnd) if (a - bid(4, CLUBS)) % 5 == HEARTS]
    assert bool(hearts) == may_bid_fixed
    assert WITH in rnd.legal_actions()


def test_FIX_4_super_caller_may_raise_into_the_fixed_trump_on_its_final_turn():
    """AUC-3 allows any higher bid on the final turn, so a raise into the fixed trump keeps it
    and the fixed trump is not replayed."""
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES), PASS, PASS, PASS)
    assert rnd.current_player == 0
    assert bid(9, HEARTS) in rnd.legal_actions()
    rnd.step(bid(9, HEARTS))
    obs = rnd.observation(0)
    assert (obs.caller, obs.trump, obs.estimates[0]) == (0, HEARTS, 9)
    run(rnd, est(0), est(0), est(0))
    while rnd.phase is Phase.PLAY:
        rnd.step(rnd.legal_actions()[0])
    assert not rnd.result().trump_changed


def test_FIX_4_super_caller_may_bid_fixed_off_keeps_its_final_raise_out_of_the_fixed_trump():
    config = RulesConfig(super_caller_may_bid_fixed=False)
    rnd = Round.new(16, deal(0, 16), HEARTS, config=config)
    run(rnd, est(2), bid(8, SPADES), PASS, PASS, PASS)
    assert rnd.current_player == 0  # the super caller's final turn
    assert all(bid_of(a)[1] != HEARTS for a in legal_bids(rnd))
    assert bid(8, SANS) in rnd.legal_actions()
    with pytest.raises(IllegalActionError):
        rnd.step(bid(9, HEARTS))
    rnd.step(bid(9, SPADES))
    assert rnd.observation(0).trump == SPADES


def test_FIX_4_super_caller_may_bid_fixed_off_still_lets_others_answer_in_the_fixed_trump():
    config = RulesConfig(super_caller_may_bid_fixed=False)
    rnd = Round.new(16, deal(0, 16), HEARTS, config=config)
    run(rnd, est(2), bid(8, SPADES))
    assert bid(9, HEARTS) in rnd.legal_actions()  # seat 1 may answer in the fixed trump
    run(rnd, bid(9, HEARTS), PASS, PASS)
    assert rnd.current_player == 0  # the super caller, outbid, on a normal turn
    assert all(bid_of(a)[1] != HEARTS for a in legal_bids(rnd))
    with pytest.raises(IllegalActionError):
        rnd.step(bid(10, HEARTS))


def test_FIX_4_estimation_after_the_auction_allows_no_new_super_call():
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES), PASS, PASS, PASS, PASS)
    assert rnd.phase is Phase.ESTIMATION
    assert not any(is_bid(a) for a in rnd.legal_actions())


def test_FIX_4_last_estimator_may_super_call():
    rnd = run(new_round(15, SPADES), est(3), est(3), est(3))
    rnd.step(bid(9, CLUBS))
    assert rnd.phase is Phase.AUCTION
    assert rnd.current_player == 2


def test_L6_first_decision_in_a_fixed_trump_round():
    rnd = new_round(15, SPADES)
    assert rnd.current_player == 2
    assert rnd.legal_actions() == [*super_calls(SPADES), *(est(k) for k in range(14))]
    assert len(rnd.legal_actions()) == 38


def test_L7_super_call_resets_the_round():
    rnd = new_round(16, HEARTS)
    assert rnd.current_player == 3
    run(rnd, est(2), bid(8, SPADES), PASS, PASS, PASS)
    assert rnd.current_player == 0
    rnd.step(PASS)
    obs = rnd.observation(3)
    assert (obs.caller, obs.trump, obs.estimates) == (0, SPADES, (8, None, None, None))
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        assert set(legal_estimates(rnd)) <= set(range(9))
        rnd.step(est(1))
    assert order == [1, 2, 3]


def test_L8_yosifs_classic():
    rnd = run(new_round(15, SPADES), est(3), est(3), est(3))
    assert rnd.current_player == 1
    assert est(4) not in rnd.legal_actions()
    assert len(rnd.legal_actions()) == 37
    all_three = run(rnd.clone(), est(3)).observation(0)
    assert all_three.caller == 2
    assert all_three.with_players == tuple(WithInfo(s, 3, SPADES) for s in (0, 1, 3))
    assert (all_three.total, all_three.over, all_three.risks) == (12, False, 0)
    two = run(rnd.clone(), est(2)).observation(0)
    assert two.caller == 2
    assert two.with_players == (WithInfo(0, 3, SPADES), WithInfo(3, 3, SPADES))
    assert (two.total, two.over, two.risks) == (11, False, 1)


def test_L9_answering_a_super_call_in_the_fixed_trump():
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES), bid(9, HEARTS), PASS, PASS, PASS)
    assert rnd.current_player == 1
    rnd.step(PASS)
    obs = rnd.observation(0)
    assert (obs.caller, obs.trump, obs.estimates[1]) == (1, HEARTS, 9)


def test_FIX_5_round_reports_whether_a_super_call_changed_its_trump():
    changed = run(new_round(16, HEARTS), est(2), bid(8, SPADES), PASS, PASS, PASS, PASS)
    run(changed, est(2), est(1), est(3))
    while changed.phase is Phase.PLAY:
        changed.step(changed.legal_actions()[0])
    assert changed.result().trump_changed
    kept = run(new_round(16, HEARTS), est(2), bid(8, SPADES), bid(9, HEARTS), PASS, PASS, PASS)
    run(kept, PASS, est(0), est(0), est(0))
    while kept.phase is Phase.PLAY:
        kept.step(kept.legal_actions()[0])
    assert not kept.result().trump_changed


def test_G7_fixed_trump_tie_for_caller_full_round():
    hands, cards = build_play(SANS, leader=2, targets=(1, 3, 5, 4))
    rnd = run(Round.new(14, hands, SANS), est(3), est(5), est(5))
    assert est(0) not in rnd.legal_actions()
    rnd.step(est(1))
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.scores == (11, 13, 25, -21)
    assert (result.caller, result.total) == (2, 14)
    assert result.seats[3].is_with


def test_G9_super_call_in_a_fixed_trump_round_full_round():
    hands, cards = build_play(SPADES, leader=0, targets=(8, 2, 0, 3))
    rnd = run(Round.new(16, hands, HEARTS), est(2), bid(8, SPADES), PASS, PASS, PASS, PASS)
    run(rnd, est(2), est(1))
    assert est(2) not in rnd.legal_actions()
    rnd.step(est(3))
    assert rnd.current_player == 0
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.scores == (74, 12, -11, 13)
    assert result.trump_changed
    assert result.seats[0].is_super


@pytest.mark.parametrize(
    ("last", "tricks", "scores"),
    [(3, (3, 4, 3, 3), (46, -42, 46, 46)), (2, (3, 3, 3, 4), (46, -22, 46, -22))],
)
def test_G10_yosifs_classic_full_round(
    last: int, tricks: tuple[int, int, int, int], scores: tuple[int, int, int, int]
):
    hands, cards = build_play(SPADES, leader=2, targets=tricks)
    rnd = run(Round.new(15, hands, SPADES), est(3), est(3), est(3), est(last))
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.scores == scores
    assert result.multiplier == 2
