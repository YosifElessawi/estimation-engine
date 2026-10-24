"""The auction in open rounds: AUC-1 to AUC-5, GAM-2 (opener), GAM-3 and L1 to L3."""

import pytest

from estimation.engine.actions import PASS, WITH
from estimation.engine.cards import CLUBS, DIAMONDS, HEARTS, SANS, SPADES
from estimation.engine.observation import WithInfo
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import DASH, NO, NO_DASH_ROUND, bid, legal_bids, legal_estimates, new_round, run


def auction(round_no: int = 1, dash: tuple[int, ...] = NO_DASH_ROUND) -> Round:
    rnd = run(new_round(round_no), *dash)
    assert rnd.phase is Phase.AUCTION
    return rnd


def turns(rnd: Round, *actions: int) -> list[int | None]:
    """Play actions and return the seat that made each one."""
    seats = []
    for action in actions:
        seats.append(rnd.current_player)
        rnd.step(action)
    return seats


def test_GAM_2_opener_is_the_first_non_dash_seat_from_F():
    assert auction(1).current_player == 0
    assert auction(2, (DASH, DASH, NO, NO)).current_player == 3  # F = 1; seats 1 and 2 Dash
    assert auction(4, (NO, NO, NO, NO)).current_player == 3
    assert auction(4, (DASH, NO, NO, NO)).current_player == 0  # F = 3 Dash Called


def test_AUC_1_turns_skip_passed_players_dash_callers_and_the_top_bidder():
    rnd = auction(1, (NO, NO, DASH, NO))
    seats = turns(rnd, bid(4, CLUBS), bid(4, DIAMONDS), PASS, bid(4, HEARTS), WITH, PASS)
    assert seats == [0, 1, 3, 0, 1, 1]
    assert rnd.current_player == 0  # only the top bidder is left: its final turn


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


def test_AUC_2_with_cannot_be_said_twice_on_the_same_top_bid():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, PASS, PASS)
    assert rnd.current_player == 1
    assert WITH not in rnd.legal_actions()
    with pytest.raises(IllegalActionError):
        rnd.step(WITH)


def test_AUC_2_with_again_on_a_newer_top_bid():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, bid(6, HEARTS), PASS, PASS)
    assert rnd.current_player == 1
    assert WITH in rnd.legal_actions()
    rnd.step(WITH)
    assert rnd.observation(0).with_players == (WithInfo(1, 6, HEARTS),)


def test_AUC_2_bidding_drops_your_with():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, PASS, PASS)  # seat 0 holds the top bid, so seat 1 is next
    rnd.step(bid(6, CLUBS))  # seat 1 bids over the bid it joined
    assert rnd.observation(0).with_players == ()
    assert rnd.observation(0).top_bidder == 1


def test_AUC_2_pass_keeps_your_with_and_takes_you_out_of_the_auction():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, PASS, PASS, PASS)
    assert rnd.observation(0).with_players == (WithInfo(1, 5, HEARTS),)
    assert rnd.current_player == 0  # seat 1 is out; seat 0's final turn
    rnd.step(PASS)
    assert rnd.observation(0).with_players == (WithInfo(1, 5, HEARTS),)


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


def test_AUC_4_with_survives_when_another_player_wins_in_the_joined_trump():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, bid(6, HEARTS), PASS, PASS, PASS, PASS)
    assert rnd.phase is Phase.ESTIMATION
    obs = rnd.observation(0)
    assert obs.caller == 2
    assert obs.with_players == (WithInfo(1, 5, HEARTS),)
    run(rnd, rnd.legal_actions()[0], rnd.legal_actions()[0])  # seats 3 and 0
    assert rnd.current_player == 1
    assert legal_estimates(rnd)[0] == 5


def test_AUC_4_with_is_lost_when_another_player_wins_in_another_trump():
    rnd = auction()
    run(rnd, bid(5, HEARTS), WITH, bid(5, SPADES), PASS, PASS, PASS, PASS)
    assert rnd.observation(0).with_players == ()


def test_AUC_4_several_players_may_be_with():
    rnd = auction()
    run(rnd, bid(4, SPADES), WITH, WITH, PASS, PASS, PASS, PASS)
    assert rnd.observation(0).with_players == (WithInfo(1, 4, SPADES), WithInfo(2, 4, SPADES))


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


def test_L1_with_survives_a_raise_in_the_same_trump():
    rnd = auction()
    run(rnd, bid(4, HEARTS), WITH, PASS, PASS, PASS)
    assert rnd.current_player == 0
    rnd.step(bid(6, HEARTS))
    obs = rnd.observation(2)
    assert (obs.caller, obs.trump, obs.estimates[0]) == (0, HEARTS, 6)
    assert obs.with_players == (WithInfo(1, 4, HEARTS),)
    assert rnd.current_player == 1
    assert legal_estimates(rnd) == [4, 5, 6]


def test_L1_with_is_lost_when_the_raise_changes_trump():
    rnd = auction()
    run(rnd, bid(4, HEARTS), WITH, PASS, PASS, PASS, bid(5, SPADES))
    assert rnd.observation(1).with_players == ()
    assert rnd.current_player == 1
    assert legal_estimates(rnd) == [0, 1, 2, 3, 4, 5]


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
