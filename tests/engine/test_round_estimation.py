"""Estimation once the Caller is known: EST-1 to EST-7 and L4."""

import dataclasses

import pytest

from estimation.engine.actions import PASS, WITH
from estimation.engine.cards import CLUBS, HEARTS, SPADES
from estimation.engine.config import DEFAULT_CONFIG
from estimation.engine.observation import WithInfo
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import (
    DASH,
    NO,
    NO_DASH_ROUND,
    bid,
    build_play,
    est,
    legal_estimates,
    new_round,
    play_cards,
    run,
)

CHOOSE_OFF = dataclasses.replace(DEFAULT_CONFIG, with_choose=False)


def caller_round(tricks: int, trump: int = SPADES, dash: tuple[int, ...] = NO_DASH_ROUND) -> Round:
    """Round 1 (F = seat 0, assumed not to Dash Call): seat 0 wins the auction at its bid."""
    rnd = run(new_round(1), *dash, bid(tricks, trump))
    while rnd.phase is Phase.AUCTION:
        rnd.step(PASS)
    return rnd


def test_EST_1_caller_estimate_is_the_winning_trick_count():
    rnd = caller_round(5)
    assert rnd.observation(2).estimates == (5, None, None, None)


def test_EST_2_others_estimate_in_turn_from_the_seat_after_the_caller():
    rnd = run(new_round(1), *NO_DASH_ROUND, PASS, PASS, bid(5, HEARTS), PASS, PASS)
    assert rnd.observation(0).caller == 2
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    assert order == [3, 0, 1]


def test_EST_2_dash_callers_do_not_estimate():
    rnd = caller_round(5, dash=(NO, NO, DASH, NO))
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    assert order == [1, 3]


def test_EST_3_other_players_estimate_0_to_c():
    rnd = caller_round(6)
    assert legal_estimates(rnd) == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(IllegalActionError):
        rnd.step(est(7))


def test_EST_2_with_players_have_their_w_as_estimate_and_do_not_estimate():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(5, HEARTS), WITH, PASS, PASS, bid(7, HEARTS))
    rnd.step(PASS)  # seat 1 stays at 5 when seat 0 raises to 7
    obs = rnd.observation(0)
    assert obs.with_players == (WithInfo(1, 5, HEARTS),)
    assert obs.estimates == (7, 5, None, None)
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    assert order == [2, 3]


def estimating_order(rnd: Round) -> list[int | None]:
    order = []
    while rnd.phase is Phase.ESTIMATION:
        order.append(rnd.current_player)
        rnd.step(rnd.legal_actions()[0])
    return order


def two_choosers() -> Round:
    """Seat 0 bids 4 Hearts, seat 1 joins it, seat 2 outbids both with 5 Hearts and keeps it:
    seats 0 and 1 are With at 4 and choose 4 or 5; seat 3 never called Hearts."""
    return run(new_round(1), *NO_DASH_ROUND, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, PASS, PASS)


def test_EST_2_with_players_who_choose_go_first_from_the_callers_left():
    rnd = two_choosers()
    obs = rnd.observation(0)
    assert obs.caller == 2
    assert obs.with_players == (WithInfo(0, 4, HEARTS), WithInfo(1, 4, HEARTS))
    assert obs.estimates == (None, None, 5, None)
    assert estimating_order(rnd) == [0, 1, 3]  # choosers from the Caller's left, then seat 3


def test_EST_2_a_with_player_at_the_callers_number_keeps_it():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(5, HEARTS), WITH, PASS, PASS, PASS)
    assert rnd.observation(0).estimates == (5, 5, None, None)
    assert estimating_order(rnd) == [2, 3]


def test_EST_2_with_choose_off_every_with_player_keeps_its_w():
    rnd = run(
        new_round(1, config=CHOOSE_OFF), *NO_DASH_ROUND, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS
    )
    run(rnd, PASS, PASS)
    assert rnd.observation(0).estimates == (4, 4, 5, None)
    assert estimating_order(rnd) == [3]


def test_EST_5_with_choose_off_kept_with_players_count_their_w():
    rnd = run(
        new_round(1, config=CHOOSE_OFF), *NO_DASH_ROUND, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS
    )
    run(rnd, PASS, PASS)
    assert rnd.current_player == 3  # the only estimator: 5 + 4 + 4 = 13 so far
    assert legal_estimates(rnd) == [1, 2, 3, 4, 5]


def test_EST_7_with_choose_off_nobody_estimates_so_nobody_takes_risk():
    rnd = run(new_round(1, config=CHOOSE_OFF), NO, NO, NO, DASH, bid(4, HEARTS), WITH)
    run(rnd, bid(5, HEARTS), PASS, bid(6, HEARTS))  # seats 0 and 1 keep 4 each
    assert rnd.phase is Phase.PLAY
    obs = rnd.observation(0)
    assert (obs.last_estimator, obs.risks, obs.total) == (None, 0, 14)


def test_EST_3_a_choosing_with_player_estimates_from_its_w_to_c():
    rnd = two_choosers()
    assert legal_estimates(rnd) == [4, 5]
    with pytest.raises(IllegalActionError):
        rnd.step(est(3))
    rnd.step(est(5))
    assert rnd.observation(3).with_players == (WithInfo(0, 5, HEARTS), WithInfo(1, 4, HEARTS))


def test_EST_5_the_last_chooser_is_last_when_nobody_else_estimates():
    rnd = run(new_round(1), NO, NO, NO, DASH, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, PASS)
    rnd.step(est(4))
    assert rnd.current_player == 1
    assert rnd.observation(1).last_estimator == 1
    assert legal_estimates(rnd) == [5]  # 5 + 4 + 4 + 0 would make 13
    with pytest.raises(IllegalActionError):
        rnd.step(est(4))


def test_EST_7_the_last_chooser_takes_risk():
    rnd = run(new_round(1), NO, NO, NO, DASH, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, PASS)
    run(rnd, est(5), est(5))  # 5 + 5 + 5 + 0 = 15
    obs = rnd.observation(0)
    assert obs.last_estimator == 1
    assert obs.risks == 1
    while rnd.phase is Phase.PLAY:
        rnd.step(rnd.legal_actions()[0])
    seat = rnd.result().seats[1]
    assert (seat.is_with, seat.is_last, seat.risks) == (True, True, 1)


def test_EST_2_with_players_who_choose_are_scored_as_with_in_a_full_round():
    """SCR-3, SCR-5 and SCR-8 for With players who chose below c: Caller seat 2 at 5 Hearts,
    seats 0 and 1 pick 4 (seat 0 makes it, seat 1 takes 3), seat 3 estimates 1 and makes it."""
    hands, cards = build_play(HEARTS, leader=2, targets=(4, 3, 5, 1))
    rnd = Round.new(1, hands)
    run(rnd, *NO_DASH_ROUND, bid(4, HEARTS), WITH, bid(5, HEARTS), PASS, PASS, PASS)
    run(rnd, est(4), est(4))
    assert est(0) not in rnd.legal_actions()  # seat 3, last: 5 + 4 + 4 + 0 would make 13
    rnd.step(est(1))
    play_cards(rnd, cards)
    result = rnd.result()
    assert [s.is_with for s in result.seats] == [True, True, False, False]
    assert result.calls == 3  # Caller and two With: x2
    # x2 of: 10 + 4 + 10 With; -1 - 10 With - 10 only loser; 10 + 5 + 10 Caller; 10 + 1
    assert result.scores == (48, -42, 50, 22)


def test_EST_4_estimating_exactly_c_makes_a_player_with():
    rnd = caller_round(5)
    rnd.step(est(5))
    assert rnd.observation(0).with_players == (WithInfo(1, 5, SPADES),)
    rnd.step(est(4))
    assert rnd.observation(0).with_players == (WithInfo(1, 5, SPADES),)


def test_EST_5_last_estimator_cannot_make_13():
    rnd = caller_round(5)
    run(rnd, est(3), est(4))
    assert 1 not in legal_estimates(rnd)
    with pytest.raises(IllegalActionError):
        rnd.step(est(1))


def test_EST_5_with_players_count_their_w_toward_the_total():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(5, HEARTS), WITH, PASS, PASS, PASS)
    rnd.step(est(2))  # 5 + 5 + 2 so far
    assert rnd.current_player == 3
    assert 1 not in legal_estimates(rnd)  # 13
    assert legal_estimates(rnd) == [0, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("estimates", "total", "over"),
    [((3, 4, 2), 14, True), ((3, 4, 0), 12, False), ((5, 5, 5), 20, True), ((0, 0, 0), 5, False)],
)
def test_EST_6_total_of_14_or_more_is_over_and_12_or_less_is_under(
    estimates: tuple[int, int, int], total: int, over: bool
):
    rnd = run(caller_round(5), *(est(k) for k in estimates))
    assert rnd.phase is Phase.PLAY
    obs = rnd.observation(0)
    assert (obs.total, obs.over) == (total, over)


@pytest.mark.parametrize(
    ("estimates", "risks"),
    [
        ((3, 3, 0), 0),  # total 12
        ((3, 4, 1), 0),  # 14
        ((2, 2, 1), 1),  # 11
        ((2, 2, 0), 1),  # 10
        ((3, 4, 2), 1),  # 15
        ((4, 4, 2), 1),  # 16
        ((1, 1, 1), 2),  # 9
        ((0, 0, 2), 2),  # 8
        ((4, 4, 3), 2),  # 17
        ((0, 0, 0), 3),  # 6
        ((6, 6, 6), 5),  # 24
    ],
)
def test_EST_7_last_estimator_takes_a_risk_for_every_2_off_13(
    estimates: tuple[int, int, int], risks: int
):
    rnd = run(caller_round(6), *(est(k) for k in estimates))
    obs = rnd.observation(0)
    assert obs.last_estimator == 3
    assert obs.total is not None
    assert obs.risks == risks == abs(obs.total - 13) // 2


def test_EST_7_no_risk_when_everyone_but_the_caller_dash_called():
    rnd = run(new_round(1), DASH, DASH, NO, DASH, bid(4, CLUBS), PASS)
    assert rnd.phase is Phase.PLAY
    obs = rnd.observation(2)
    assert obs.caller == 2
    assert obs.last_estimator is None
    assert obs.risks == 0
    assert (obs.total, obs.over) == (4, False)


def test_EST_7_no_risk_when_everyone_but_the_caller_is_with_or_dash():
    rnd = run(new_round(1), NO, NO, DASH, NO, bid(4, HEARTS), WITH, WITH)
    rnd.step(PASS)  # seat 0 keeps 4 Hearts: 4 + 4 + 0 + 4 = 12
    assert rnd.phase is Phase.PLAY
    obs = rnd.observation(0)
    assert obs.last_estimator is None
    assert obs.risks == 0
    assert (obs.total, obs.over) == (12, False)


def test_L4_sum_rule_example():
    rnd = caller_round(5)
    run(rnd, est(3), est(4))
    assert rnd.current_player == 3
    assert legal_estimates(rnd) == [0, 2, 3, 4, 5]
