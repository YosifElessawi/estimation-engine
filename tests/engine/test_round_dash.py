"""The Dash Call phase (DSH-1 to DSH-5) and how each round starts (GAM-2, GAM-3)."""

import itertools

import pytest

from estimation.engine.actions import PASS, is_bid, is_estimate
from estimation.engine.cards import CLUBS, DIAMONDS, SANS, SPADES, deal
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import DASH, NO, bid, new_round, run


def test_DSH_1_open_round_starts_with_a_dash_call_decision():
    rnd = new_round(1)
    assert rnd.phase is Phase.DASH
    assert rnd.current_player == 0
    assert rnd.legal_actions() == [NO, DASH]


def test_DSH_1_fixed_trump_round_has_no_dash_call():
    rnd = new_round(14, SANS)
    assert rnd.phase is Phase.ESTIMATION
    assert NO not in rnd.legal_actions()
    assert DASH not in rnd.legal_actions()
    with pytest.raises(IllegalActionError):
        rnd.step(DASH)


@pytest.mark.parametrize(("round_no", "first"), [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (13, 0)])
def test_GAM_2_first_seat_to_act_is_round_minus_one_mod_4(round_no: int, first: int):
    rnd = new_round(round_no)
    assert rnd.first == first
    assert rnd.current_player == first


@pytest.mark.parametrize(
    ("round_no", "fixed_trump", "first"),
    [(14, SANS, 1), (15, SPADES, 2), (16, SPADES, 3), (17, SPADES, 0), (22, SPADES, 1)],
)
def test_GAM_2_F_estimates_first_in_a_fixed_trump_round(
    round_no: int, fixed_trump: int, first: int
):
    rnd = new_round(round_no, fixed_trump)
    assert rnd.current_player == first
    assert all(is_estimate(a) or is_bid(a) for a in rnd.legal_actions())


def test_DSH_2_engine_asks_each_seat_in_turn_from_F():
    rnd = new_round(3)
    asked = []
    while rnd.phase is Phase.DASH:
        asked.append(rnd.current_player)
        rnd.step(NO)
    assert asked == [2, 3, 0, 1]


def test_DSH_2_every_dash_call_is_public_as_soon_as_it_is_made():
    rnd = run(new_round(1), DASH)
    for seat in range(4):
        obs = rnd.observation(seat)
        assert obs.dash_callers == (0,)
        assert obs.log == ((0, DASH),)
        assert obs.estimates == (0, None, None, None)
    run(rnd, NO, DASH)
    assert rnd.current_player == 3
    obs = rnd.observation(3)
    assert obs.dash_callers == (0, 2)
    assert obs.log == ((0, DASH), (1, NO), (2, DASH))
    assert obs.estimates == (0, None, 0, None)


@pytest.mark.parametrize("choices", list(itertools.product((NO, DASH), repeat=4)))
def test_DSH_2_every_seat_sees_the_same_dash_calls_while_the_others_decide(
    choices: tuple[int, ...],
):
    rnd = new_round(2)
    for made, choice in enumerate(choices, start=1):
        rnd.step(choice)
        seats = [(rnd.first + k) % 4 for k in range(made)]
        callers = tuple(sorted(s for s, c in zip(seats, choices, strict=False) if c == DASH))
        views = [rnd.observation(seat) for seat in range(4)]
        for obs in views:
            assert obs.dash_callers == callers
            assert obs.log == tuple(zip(seats, choices[:made], strict=True))
            assert all(obs.estimates[s] == 0 for s in callers)


def test_DSH_3_any_number_of_players_may_dash_call():
    rnd = run(new_round(1), DASH, DASH, NO, DASH)
    assert rnd.phase is Phase.AUCTION
    assert rnd.observation(0).dash_callers == (0, 1, 3)


def test_DSH_4_dash_caller_sits_out_the_auction_and_estimates_zero():
    rnd = run(new_round(1), NO, DASH, NO, NO)
    assert rnd.observation(2).estimates == (None, 0, None, None)
    rnd.step(bid(4, CLUBS))
    assert rnd.current_player == 2  # seat 1 is skipped
    run(rnd, bid(5, DIAMONDS), PASS, PASS)  # seat 2 bids, seats 3 and 0 pass
    assert rnd.current_player == 2  # the top bidder's final turn; seat 1 is never asked
    rnd.step(PASS)
    assert rnd.phase is Phase.ESTIMATION
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    assert order == [3, 0]
    obs = rnd.observation(0)
    assert obs.estimates[1] == 0
    assert obs.caller == 2
    assert all(w.seat != 1 for w in obs.with_players)


def test_DSH_5_all_four_dash_calls_skip_the_round():
    rnd = run(new_round(1), DASH, DASH, DASH, DASH)
    assert rnd.phase is Phase.DONE
    assert rnd.current_player is None
    assert rnd.legal_actions() == []
    result = rnd.result()
    assert result.skipped
    assert result.scores == (0, 0, 0, 0)
    assert result.next_carry == 2
    assert all(seat.is_dash for seat in result.seats)


def test_GAM_3_all_dash_skipped_round_scores_zero_even_with_a_carry():
    rnd = run(Round.new(1, deal(0, 1), carry=4), DASH, DASH, DASH, DASH)
    assert rnd.result().scores == (0, 0, 0, 0)
    assert rnd.result().next_carry == 8
