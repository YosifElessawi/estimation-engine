"""The auction in open rounds: AUC-1 to AUC-6, GAM-2 (opener), GAM-3, L1 to L3, L5 and L10 to
L12. The With rules before 2026-10-30 are tested with ``LATEST_CALL_ONLY``."""

import dataclasses

import pytest

from estimation.engine.actions import PASS, WITH
from estimation.engine.cards import CLUBS, DIAMONDS, HEARTS, SANS, SPADES
from estimation.engine.config import DEFAULT_CONFIG, RulesConfig
from estimation.engine.observation import WithInfo
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import (
    DASH,
    LATEST_CALL_ONLY,
    NO,
    NO_DASH_ROUND,
    bid,
    est,
    legal_bids,
    legal_estimates,
    new_round,
    run,
)


def auction(
    round_no: int = 1, dash: tuple[int, ...] = NO_DASH_ROUND, config: RulesConfig = DEFAULT_CONFIG
) -> Round:
    rnd = run(new_round(round_no, config=config), *dash)
    assert rnd.phase is Phase.AUCTION
    return rnd


def turns(rnd: Round, *actions: int) -> list[int | None]:
    """Play actions and return the seat that made each one."""
    seats = []
    for action in actions:
        seats.append(rnd.current_player)
        rnd.step(action)
    return seats


def held(rnd: Round, seat: int = 0) -> tuple[tuple[int, int, int], ...]:
    """The With players a seat sees, as (seat, w, trump)."""
    return tuple((w.seat, w.w, w.trump) for w in rnd.observation(seat).with_players)


def finish_estimates(rnd: Round) -> list[int | None]:
    """Seats that estimate, in order, each taking its first legal estimate."""
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    return order


def test_AUC_1_turns_skip_passed_players_dash_callers_and_the_top_bidder():
    rnd = auction(1, (NO, NO, DASH, NO))
    seats = turns(rnd, bid(4, CLUBS), bid(4, DIAMONDS), PASS, bid(4, HEARTS), WITH)
    assert seats == [0, 1, 3, 0, 1]
    assert rnd.current_player == 0  # seat 1 said With and seat 3 passed: the final turn


def test_AUC_1_saying_with_takes_you_out_of_the_turns():
    rnd = auction()
    seats = turns(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, PASS)
    assert seats == [0, 1, 2, 3, 0]  # seat 1 is never asked again after its With
    assert rnd.current_player == 2  # the top bidder's final turn


def test_GAM_2_opener_is_the_first_non_dash_seat_from_F():
    assert auction(1).current_player == 0
    assert auction(2, (DASH, DASH, NO, NO)).current_player == 3  # F = 1; seats 1 and 2 Dash
    assert auction(4, (NO, NO, NO, NO)).current_player == 3
    assert auction(4, (DASH, NO, NO, NO)).current_player == 0  # F = 3 Dash Called


def test_AUC_2_lowest_bid_is_4_clubs_and_highest_13_sans():
    rnd = auction()
    assert rnd.legal_actions() == [*range(bid(4, CLUBS), bid(13, SANS) + 1), PASS]


def test_AUC_2_bid_must_beat_the_top_bid():
    rnd = auction()
    rnd.step(bid(5, HEARTS))
    for weaker in (bid(5, HEARTS), bid(5, DIAMONDS), bid(4, SANS)):
        assert weaker not in rnd.legal_actions()
        with pytest.raises(IllegalActionError):
            rnd.step(weaker)
    rnd.step(bid(5, SPADES))
    assert rnd.observation(0).top_bid == bid(5, SPADES)
    assert rnd.observation(0).top_bidder == 1


def test_AUC_2_with_needs_another_players_top_bid():
    rnd = auction()
    assert WITH not in rnd.legal_actions()
    with pytest.raises(IllegalActionError):
        rnd.step(WITH)
    rnd.step(bid(4, SPADES))
    assert WITH in rnd.legal_actions()


def test_AUC_2_with_joins_at_the_top_bids_trick_count_and_trump():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH)
    assert rnd.observation(3).with_players == (WithInfo(1, 5, HEARTS),)


def test_AUC_2_each_bidder_holds_its_latest_call():
    rnd = auction(config=LATEST_CALL_ONLY)
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, bid(6, SPADES), bid(7, HEARTS), PASS)
    assert rnd.current_player == 1  # final turn
    rnd.step(PASS)
    # Seat 0 bid 4 Hearts first, but its latest call is 6 Spades: it is not With in Hearts.
    assert held(rnd) == ()
    assert rnd.observation(0).caller == 1


def test_AUC_4_any_earlier_call_in_the_winning_trump_makes_you_with():
    rnd = auction()
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, bid(6, SPADES), bid(7, HEARTS), PASS)
    rnd.step(PASS)
    # Seat 0's latest call is 6 Spades, but it called 4 Hearts: With, choosing from 4 to 7.
    assert held(rnd) == ((0, 4, HEARTS),)
    assert rnd.current_player == 0
    assert legal_estimates(rnd) == [4, 5, 6, 7]


def test_AUC_4_with_any_call_off_counts_only_the_latest_call():
    rnd = auction(config=dataclasses.replace(DEFAULT_CONFIG, with_any_call=False))
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, bid(6, SPADES), bid(7, HEARTS), PASS)
    rnd.step(PASS)
    assert held(rnd) == ()
    assert finish_estimates(rnd) == [2, 3, 0]


def test_AUC_3_top_bidder_gets_one_final_turn_to_keep_or_raise():
    rnd = auction()
    run(rnd, bid(5, HEARTS), PASS, PASS, PASS)
    assert rnd.phase is Phase.AUCTION
    assert rnd.current_player == 0
    assert legal_bids(rnd) == list(range(bid(5, SPADES), bid(13, SANS) + 1))
    assert rnd.legal_actions()[-1] == PASS
    rnd.step(PASS)
    assert rnd.phase is Phase.ESTIMATION
    obs = rnd.observation(0)
    assert obs.caller == 0
    assert obs.trump == HEARTS
    assert obs.estimates[0] == 5


def test_AUC_3_auction_ends_after_a_raise_on_the_final_turn():
    rnd = auction()
    run(rnd, bid(5, HEARTS), PASS, PASS, PASS, bid(7, SANS))
    assert rnd.phase is Phase.ESTIMATION
    obs = rnd.observation(1)
    assert (obs.caller, obs.trump, obs.estimates[0]) == (0, SANS, 7)


def test_AUC_3_with_players_answer_a_raise_on_the_final_turn():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, PASS, PASS)
    assert rnd.current_player == 0
    rnd.step(bid(6, SPADES))
    assert rnd.phase is Phase.AUCTION
    assert rnd.current_player == 1  # seat 1 answers before the auction ends
    assert rnd.legal_actions() == [*range(bid(6, SPADES) + 1, bid(13, SANS) + 1), WITH, PASS]
    rnd.step(WITH)
    assert rnd.phase is Phase.ESTIMATION
    assert held(rnd) == ((1, 6, SPADES),)


def test_AUC_3_staying_keeps_the_earlier_call():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, PASS, PASS, bid(6, SPADES), PASS)
    assert rnd.phase is Phase.ESTIMATION
    assert held(rnd) == ((1, 4, SPADES),)


def test_AUC_3_answers_go_counterclockwise_from_the_top_bidder():
    rnd = auction(2)  # F = seat 1
    seats = turns(rnd, bid(4, SPADES), WITH, PASS, WITH, bid(5, SPADES), PASS, WITH)
    assert seats == [1, 2, 3, 0, 1, 2, 0]
    assert rnd.phase is Phase.ESTIMATION
    assert held(rnd) == ((0, 5, SPADES), (2, 4, SPADES))


def test_AUC_3_an_answer_that_outbids_reopens_the_auction():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, PASS, PASS, bid(5, SPADES))
    assert rnd.current_player == 1
    rnd.step(bid(6, HEARTS))
    assert rnd.phase is Phase.AUCTION
    assert rnd.current_player == 0  # the old top bidder is back in
    assert WITH in rnd.legal_actions()
    rnd.step(PASS)
    assert rnd.current_player == 1  # now seat 1 has the final turn
    rnd.step(PASS)
    obs = rnd.observation(0)
    assert (obs.caller, obs.trump) == (1, HEARTS)
    assert held(rnd) == ()  # seat 0 holds 5 Spades


def test_AUC_3_a_raise_into_another_trump_is_answered_too():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, PASS, PASS, bid(5, HEARTS))
    assert rnd.current_player == 1
    rnd.step(WITH)  # joins the new trump
    assert held(rnd) == ((1, 5, HEARTS),)
    stayed = run(auction(), bid(4, SPADES), WITH, PASS, PASS, bid(5, HEARTS), PASS)
    assert held(stayed) == ()  # staying keeps 4 Spades, which is not the trump
    assert legal_estimates(stayed) == [0, 1, 2, 3, 4, 5]


def test_AUC_3_after_an_outbidding_answer_the_others_no_longer_answer():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, WITH, PASS, bid(5, SPADES))
    assert rnd.current_player == 1  # seat 2 would answer next
    rnd.step(bid(6, HEARTS))
    assert rnd.current_player == 0  # the old top bidder's turn, not seat 2's answer
    rnd.step(PASS)
    assert rnd.current_player == 1  # seat 1's final turn: seat 2 is out
    rnd.step(PASS)
    assert held(rnd) == ()  # seat 0 holds 5 Spades, seat 2 4 Spades


def test_AUC_3_nobody_bids_skips_the_round():
    rnd = auction(2)
    run(rnd, PASS, PASS, PASS, PASS)
    assert rnd.phase is Phase.DONE
    assert rnd.result().skipped
    assert rnd.result().next_carry == 2


def test_AUC_3_last_bidder_left_may_still_bid_after_three_passes():
    rnd = auction()
    run(rnd, PASS, PASS, PASS)
    assert rnd.current_player == 3
    rnd.step(bid(4, CLUBS))
    assert rnd.current_player == 3  # final turn
    rnd.step(PASS)
    assert rnd.phase is Phase.ESTIMATION


def test_AUC_4_outbid_in_your_own_trump_and_passing_holds_you_at_your_bid():
    rnd = auction()
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, PASS, PASS)
    assert rnd.observation(0).caller == 1
    assert held(rnd) == ((0, 4, HEARTS),)
    assert rnd.observation(0).estimates[0] is None  # EST-2: it chooses 4 or 5, first
    assert rnd.current_player == 0
    assert legal_estimates(rnd) == [4, 5]


def test_AUC_4_latest_call_only_holds_you_at_exactly_your_bid():
    rnd = auction(config=LATEST_CALL_ONLY)
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, PASS, PASS)
    assert held(rnd) == ((0, 4, HEARTS),)
    assert rnd.observation(0).estimates[0] == 4


def test_AUC_4_with_follows_the_trump_however_high_it_goes():
    rnd = auction()
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS, PASS, PASS, bid(9, HEARTS))
    assert held(rnd) == ((0, 4, HEARTS),)  # seat 0 does not answer: it passed, it did not join


def test_AUC_4_with_is_lost_when_another_player_wins_in_another_trump():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, bid(5, SPADES), PASS, PASS, PASS)
    assert rnd.phase is Phase.ESTIMATION
    assert held(rnd) == ()


def test_AUC_4_a_with_player_outbid_by_someone_else_does_not_answer():
    rnd = auction()
    run(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, bid(6, HEARTS), PASS)
    assert rnd.current_player == 0  # seat 0's final turn; seat 1 was never asked again
    rnd.step(bid(7, HEARTS))
    assert rnd.phase is Phase.ESTIMATION  # nobody answers: seat 1 joined seat 0's 4 Hearts,
    assert held(rnd) == ((1, 4, HEARTS), (2, 5, HEARTS))  # which seat 2 outbid


def test_AUC_4_several_players_may_be_with():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, WITH, PASS, PASS)
    assert held(rnd) == ((1, 4, SPADES), (2, 4, SPADES))


def test_AUC_5_lone_bidder_cannot_bid_13_on_either_turn():
    rnd = auction(1, (DASH, NO, DASH, DASH))
    assert rnd.current_player == 1
    assert max(legal_bids(rnd)) == bid(12, SANS)
    with pytest.raises(IllegalActionError):
        rnd.step(bid(13, CLUBS))
    rnd.step(bid(10, CLUBS))
    assert rnd.current_player == 1
    assert max(legal_bids(rnd)) == bid(12, SANS)
    with pytest.raises(IllegalActionError):
        rnd.step(bid(13, SANS))


def test_AUC_5_two_bidders_may_bid_13():
    rnd = auction(1, (DASH, NO, DASH, NO))
    assert max(legal_bids(rnd)) == bid(13, SANS)


def test_AUC_6_a_final_raise_that_would_leave_13_is_not_allowed():
    rnd = auction(1, (NO, NO, DASH, DASH), LATEST_CALL_ONLY)
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS)  # seat 0 holds 4 Hearts
    assert rnd.current_player == 1
    legal = legal_bids(rnd)
    assert bid(8, HEARTS) in legal  # 8 + 4 = 12
    assert bid(9, HEARTS) not in legal  # 9 + 4 = 13 with nobody left to estimate
    assert bid(9, SPADES) in legal  # seat 0 would hold Hearts, so it estimates
    with pytest.raises(IllegalActionError):
        rnd.step(bid(9, HEARTS))


def test_AUC_6_an_answer_that_would_leave_13_is_not_allowed():
    rnd = auction(1, (NO, NO, NO, DASH))
    run(rnd, bid(4, HEARTS), WITH, WITH, bid(5, HEARTS), PASS)
    assert rnd.current_player == 2  # the last answer
    assert PASS not in rnd.legal_actions()  # staying at 4: 5 + 4 + 4 + 0 = 13
    with pytest.raises(IllegalActionError):
        rnd.step(PASS)
    assert WITH in rnd.legal_actions()  # 5 + 4 + 5 = 14
    rnd.step(WITH)
    assert rnd.phase is Phase.PLAY  # nobody estimates
    assert rnd.observation(0).total == 14


def test_AUC_6_a_with_player_who_chooses_is_someone_left_to_estimate():
    rnd = auction(1, (NO, NO, DASH, DASH))
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), PASS)  # seat 0 called 4 Hearts
    assert bid(9, HEARTS) in legal_bids(rnd)  # seat 0 will choose from 4 to 9
    rnd.step(bid(9, HEARTS))
    assert rnd.current_player == 0
    assert legal_estimates(rnd) == [5, 6, 7, 8, 9]  # EST-5: 4 would make 13, and it is last


def test_AUC_6_latest_call_only_blocks_keeping_a_bid_that_leaves_13():
    rnd = auction(1, (NO, NO, NO, DASH), LATEST_CALL_ONLY)
    run(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS)
    assert rnd.current_player == 2  # the final turn
    assert PASS not in rnd.legal_actions()  # keeping: 5 + 4 + 4 + 0 = 13
    with pytest.raises(IllegalActionError):
        rnd.step(PASS)
    rnd.step(bid(6, HEARTS))
    assert rnd.phase is Phase.PLAY  # seats 0 and 1 keep 4 each: nobody left to estimate
    obs = rnd.observation(0)
    assert obs.with_players == (WithInfo(0, 4, HEARTS), WithInfo(1, 4, HEARTS))
    assert obs.total == 14


def test_AUC_6_with_choose_off_blocks_keeping_a_bid_that_leaves_13():
    rnd = auction(1, (NO, NO, NO, DASH), dataclasses.replace(DEFAULT_CONFIG, with_choose=False))
    run(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS)
    assert rnd.current_player == 2
    assert PASS not in rnd.legal_actions()  # seats 0 and 1 keep 4: 5 + 4 + 4 + 0 = 13


def test_AUC_6_ending_with_estimators_left_is_never_blocked():
    rnd = auction(1, (NO, NO, NO, DASH))
    run(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS))
    # Seat 2 is still in, so keeping can never leave 13 with nobody to estimate.
    assert PASS in rnd.legal_actions()


def test_L1_answering_a_raise_and_held_when_outbid_in_your_own_trump():
    rnd = auction()
    seats = turns(rnd, bid(4, HEARTS), bid(5, HEARTS), WITH, PASS, PASS)
    assert seats == [0, 1, 2, 3, 0]
    assert rnd.current_player == 1  # everyone but seat 1 is out: its final turn
    rnd.step(bid(6, HEARTS))
    assert rnd.current_player == 2  # seat 2 answers
    rnd.step(WITH)
    obs = rnd.observation(3)
    assert (obs.caller, obs.trump) == (1, HEARTS)
    assert obs.with_players == (WithInfo(0, 4, HEARTS), WithInfo(2, 6, HEARTS))
    assert obs.estimates == (None, 6, 6, None)  # seat 2 keeps 6; seat 0 chooses
    assert rnd.current_player == 0  # EST-2: seat 0 chooses first
    assert legal_estimates(rnd) == [4, 5, 6]
    rnd.step(est(5))
    assert rnd.current_player == 3  # then the only other player, last
    assert legal_estimates(rnd) == [0, 1, 2, 3, 4, 5, 6]  # 6 + 5 + 6 = 17 already
    rnd.step(rnd.legal_actions()[0])
    while rnd.phase is Phase.PLAY:
        rnd.step(rnd.legal_actions()[0])
    assert rnd.result().calls == 3  # Caller and two With: the round counts double


def test_L1_staying_leaves_the_answering_player_with_at_5():
    rnd = auction()
    run(rnd, bid(4, HEARTS), bid(5, HEARTS), WITH, PASS, PASS, bid(6, HEARTS), PASS)
    assert held(rnd) == ((0, 4, HEARTS), (2, 5, HEARTS))
    assert rnd.observation(0).estimates == (None, 6, 5, None)  # seat 2 stayed: it keeps 5
    assert finish_estimates(rnd) == [0, 3]


def test_L2_minimum_raise_over_5_hearts():
    rnd = auction()
    rnd.step(bid(5, HEARTS))
    raises = [bid(5, SPADES), bid(5, SANS), *range(bid(6, CLUBS), bid(13, SANS) + 1)]
    assert rnd.legal_actions() == [*raises, WITH, PASS]
    assert len(rnd.legal_actions()) == 44
    run(rnd, PASS, PASS, PASS)
    assert rnd.current_player == 0
    assert rnd.legal_actions() == [*raises, PASS]
    assert len(rnd.legal_actions()) == 43


def test_L3_lone_bidder_legal_actions_and_final_turn():
    rnd = auction(1, (DASH, DASH, DASH, NO))
    assert rnd.current_player == 3
    assert rnd.legal_actions() == [*range(bid(4, CLUBS), bid(12, SANS) + 1), PASS]
    assert len(rnd.legal_actions()) == 46
    rnd.step(bid(6, HEARTS))
    assert rnd.phase is Phase.AUCTION
    assert rnd.current_player == 3
    assert rnd.legal_actions() == [*range(bid(6, HEARTS) + 1, bid(12, SANS) + 1), PASS]


def test_L5_staying_would_leave_13_with_nobody_left_to_estimate():
    rnd = auction(1, (NO, NO, NO, DASH))
    seats = turns(rnd, bid(4, HEARTS), WITH, WITH, bid(5, HEARTS), PASS)
    assert seats == [0, 1, 2, 0, 1]  # seat 0's final turn raise, then seat 1 stays at 4
    assert rnd.current_player == 2  # the last answer
    assert PASS not in rnd.legal_actions()  # staying too: 5 + 4 + 4 + 0 = 13
    with pytest.raises(IllegalActionError):
        rnd.step(PASS)
    rnd.step(WITH)  # 5 + 4 + 5 = 14
    assert rnd.phase is Phase.PLAY  # nobody estimates
    obs = rnd.observation(0)
    assert obs.with_players == (WithInfo(1, 4, HEARTS), WithInfo(2, 5, HEARTS))
    assert obs.total == 14


def test_L5_with_players_who_choose_end_on_a_total_off_13():
    rnd = auction(1, (NO, NO, NO, DASH))
    run(rnd, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS)
    assert PASS in rnd.legal_actions()  # seats 0 and 1 choose, so keeping is allowed
    rnd.step(PASS)
    assert held(rnd) == ((0, 4, HEARTS), (1, 4, HEARTS))
    assert rnd.current_player == 0  # EST-2: from the Caller's left, past seat 3's Dash
    assert legal_estimates(rnd) == [4, 5]
    rnd.step(est(4))
    assert rnd.current_player == 1  # the last to estimate
    assert legal_estimates(rnd) == [5]  # EST-5: 5 + 4 + 4 + 0 would make 13


def test_L10_with_lost_to_another_trump():
    rnd = auction()
    seats = turns(rnd, bid(4, HEARTS), WITH, bid(5, SPADES), PASS, PASS)
    assert seats == [0, 1, 2, 3, 0]
    assert rnd.current_player == 2  # seat 1 never answers seat 2's bid
    rnd.step(PASS)
    assert held(rnd) == ()
    assert legal_estimates(rnd) == [0, 1, 2, 3, 4, 5]
    assert finish_estimates(rnd) == [3, 0, 1]


def test_L11_outbid_in_another_trump_but_own_trump_wins():
    rnd = auction()
    run(rnd, bid(4, HEARTS), bid(5, SPADES), bid(6, HEARTS), PASS, PASS, PASS)
    assert rnd.current_player == 2
    rnd.step(PASS)
    obs = rnd.observation(0)
    assert (obs.caller, obs.trump) == (2, HEARTS)
    assert obs.with_players == (WithInfo(0, 4, HEARTS),)
    assert obs.estimates[0] is None
    assert rnd.current_player == 0  # EST-2: seat 0 chooses first
    assert legal_estimates(rnd) == [4, 5, 6]
    rnd.step(est(4))
    assert legal_estimates(rnd) == [0, 1, 2, 3, 4, 5, 6]  # seat 1 never called Hearts
    assert finish_estimates(rnd) == [3, 1]


def test_L12_an_earlier_call_in_the_winning_trump_makes_you_with():
    rnd = auction(10)  # F = seat 1
    seats = turns(rnd, PASS, PASS, bid(5, SANS), WITH, bid(6, HEARTS), bid(6, SANS), PASS)
    assert seats == [1, 2, 3, 0, 3, 0, 3]  # seat 3's final turn raise, seat 0 outbids it
    assert held(rnd) == ((3, 5, SANS),)  # seat 3 passed holding 6 Hearts, but called 5 Sans
    assert rnd.current_player == 0  # seat 0's final turn
    rnd.step(PASS)
    obs = rnd.observation(1)
    assert (obs.caller, obs.trump) == (0, SANS)
    assert obs.with_players == (WithInfo(3, 5, SANS),)
    assert rnd.current_player == 3  # EST-2: seat 3 chooses first
    assert legal_estimates(rnd) == [5, 6]
    rnd.step(est(5))
    assert finish_estimates(rnd) == [1, 2]
    assert rnd.observation(1).last_estimator == 2


def test_L12_latest_call_only_leaves_seat_3_out():
    rnd = auction(10, config=LATEST_CALL_ONLY)
    run(rnd, PASS, PASS, bid(5, SANS), WITH, bid(6, HEARTS), bid(6, SANS), PASS, PASS)
    assert held(rnd) == ()
    assert finish_estimates(rnd) == [1, 2, 3]  # what the engine did before 2026-10-30
