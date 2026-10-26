"""Estimation once the Caller is known: EST-1 to EST-7 and L4."""

import pytest

from estimation.engine.actions import PASS, WITH
from estimation.engine.cards import CLUBS, HEARTS, SPADES
from estimation.engine.observation import WithInfo
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import DASH, NO, NO_DASH_ROUND, bid, est, legal_estimates, new_round, run


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
