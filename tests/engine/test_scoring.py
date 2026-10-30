"""Scoring (SPEC §3): score_round, next_carry, the golden rounds G1-G10 and SCR-1 to SCR-10."""

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from estimation.engine.cards import HEARTS, SANS, SPADES
from estimation.engine.config import RulesConfig
from estimation.engine.scoring import RoundOutcome, RoundResult, next_carry, score_round

DEFAULT = RulesConfig()


def cfg(**changes: object) -> RulesConfig:
    return dataclasses.replace(DEFAULT, **changes)  # type: ignore[arg-type]  # test-only overrides


def components(result: RoundResult, seat: int) -> dict[str, int]:
    s = result.seats[seat]
    points = {
        "base": s.base_points,
        "dash": s.dash_points,
        "call": s.call_points,
        "risk": s.risk_points,
        "only": s.only_points,
        "super": s.super_points,
    }
    return {name: value for name, value in points.items() if value}


# Golden rounds (SPEC §3.3)

G1 = RoundOutcome(estimates=(5, 3, 4, 2), tricks=(5, 3, 3, 2), caller=0, last_estimator=3)


def test_G1_plain_over_round():
    result = score_round(G1)
    assert result.scores == (25, 13, -11, 12)
    assert result.total == 14
    assert result.over is True
    assert result.calls == 1
    assert result.multiplier == 1
    assert components(result, 0) == {"base": 15, "call": 10}
    assert components(result, 1) == {"base": 13}
    assert components(result, 2) == {"base": -1, "only": -10}
    assert components(result, 3) == {"base": 12}
    assert [s.won for s in result.seats] == [True, True, False, True]


G2 = RoundOutcome(
    estimates=(0, 4, 4, 1),
    tricks=(3, 4, 4, 2),
    caller=1,
    with_players=(2,),
    last_estimator=0,
    risks=2,
)


def test_G2_under_round_with_with_and_risk():
    result = score_round(G2)
    assert result.scores == (-23, 24, 24, -1)
    assert result.total == 9
    assert result.over is False
    assert result.calls == 2
    assert components(result, 0) == {"base": -3, "risk": -20}
    assert components(result, 2) == {"base": 14, "call": 10}
    assert result.seats[0].risks == 2
    assert result.seats[0].is_last


def test_G2_calls_count_risk_and_dash_on():
    result = score_round(G2, cfg(calls_count_risk_and_dash=True))
    assert result.calls == 3
    assert result.scores == (-46, 48, 48, -2)


G3 = RoundOutcome(
    estimates=(4, 4, 0, 6),
    tricks=(4, 3, 0, 6),
    caller=3,
    dash_callers=(2,),
    last_estimator=1,
)


def test_G3_dash_call_wins_in_an_over_round():
    result = score_round(G3)
    assert result.scores == (14, -11, 25, 26)
    assert components(result, 2) == {"dash": 25}
    assert result.seats[2].is_dash


def test_G3_dash_replaces_base_off():
    result = score_round(G3, cfg(dash_replaces_base=False))
    assert result.scores == (14, -11, 35, 26)
    assert components(result, 2) == {"base": 10, "dash": 25}


G4 = RoundOutcome(
    estimates=(0, 5, 3, 4),
    tricks=(1, 5, 3, 4),
    caller=1,
    dash_callers=(0,),
    last_estimator=3,
)


def test_G4_dash_call_fails_in_an_under_round():
    result = score_round(G4)
    assert result.scores == (-43, 25, 13, 14)
    assert components(result, 0) == {"dash": -33, "only": -10}


def test_G4_dash_replaces_base_off():
    assert score_round(G4, cfg(dash_replaces_base=False)).scores[0] == -44


def test_G4_only_bonus_for_dash_calls_off():
    assert score_round(G4, cfg(only_bonus_for_dash_calls=False)).scores[0] == -33


G5A = RoundOutcome(estimates=(8, 2, 1, 3), tricks=(8, 2, 1, 2), caller=0, last_estimator=3)
G5B = dataclasses.replace(G5A, tricks=(7, 2, 1, 3))


def test_G5_super_call_at_8_made():
    result = score_round(G5A)
    assert result.scores == (64, 12, 11, -11)
    assert components(result, 0) == {"super": 64}
    assert result.seats[0].is_super


def test_G5_super_call_at_8_failed():
    result = score_round(G5B)
    assert result.scores == (-42, 12, 11, 13)
    assert components(result, 0) == {"super": -32, "only": -10}


def test_G5_super_call_scoring_off():
    off = cfg(super_call_scoring=False)
    assert score_round(G5A, off).scores[0] == 28
    assert score_round(G5B, off).scores[0] == -21


def test_G5_only_loser_for_super_calls_off():
    assert score_round(G5B, cfg(only_loser_for_super_calls=False)).scores[0] == -32


G6 = RoundOutcome(
    estimates=(4, 4, 4, 0),
    tricks=(5, 3, 2, 3),
    caller=0,
    with_players=(1, 2),
    last_estimator=3,
)


def test_G6_everyone_loses():
    result = score_round(G6)
    assert result.all_lose
    assert result.multiplier == 0
    assert result.scores == (0, 0, 0, 0)
    assert result.next_carry == 2


def test_G6_all_lose_doubles_next_off():
    result = score_round(G6, cfg(all_lose_doubles_next=False))
    assert result.all_lose
    assert result.calls == 3
    assert result.multiplier == 2
    assert result.scores == (-22, -22, -24, -6)
    assert result.next_carry == 1


G7 = RoundOutcome(
    estimates=(1, 3, 5, 5),
    tricks=(1, 3, 5, 4),
    caller=2,
    with_players=(3,),
    last_estimator=0,
    round_no=14,
    trump=SANS,
    fixed_trump=SANS,
)


def test_G7_fixed_trump_round_tie_for_caller():
    result = score_round(G7)
    assert result.scores == (11, 13, 25, -21)
    assert components(result, 3) == {"base": -1, "call": -10, "only": -10}
    assert result.trump_changed is False


def test_G8_skipped_round_and_carry():
    skipped = score_round(RoundOutcome(skipped=True, round_no=3))
    assert skipped.skipped
    assert skipped.scores == (0, 0, 0, 0)
    assert skipped.next_carry == 2
    assert next_carry(1, skipped=True, all_lost=False, config=DEFAULT) == 2
    round_4 = score_round(dataclasses.replace(G1, carry=2))
    assert round_4.scores == (50, 26, -22, 24)
    assert round_4.next_carry == 1
    round_7 = next_carry(1, skipped=True, all_lost=False, config=DEFAULT)
    round_8 = next_carry(round_7, skipped=True, all_lost=False, config=DEFAULT)
    assert (round_7, round_8) == (2, 4)


def test_G8_carry_stacks_off():
    off = cfg(carry_stacks=False)
    round_7 = next_carry(1, skipped=True, all_lost=False, config=off)
    assert next_carry(round_7, skipped=True, all_lost=False, config=off) == 2


G9 = RoundOutcome(
    estimates=(8, 2, 1, 3),
    tricks=(8, 2, 0, 3),
    caller=0,
    last_estimator=3,
    round_no=16,
    trump=SPADES,
    fixed_trump=HEARTS,
)


def test_G9_super_call_in_a_fixed_trump_round():
    result = score_round(G9)
    assert result.scores == (64, 12, -11, 13)
    assert components(result, 0) == {"super": 64}
    assert result.trump_changed is True


G10A = RoundOutcome(
    estimates=(3, 3, 3, 3),
    tricks=(3, 4, 3, 3),
    caller=2,
    with_players=(0, 1, 3),
    last_estimator=1,
)
G10B = RoundOutcome(
    estimates=(3, 2, 3, 3),
    tricks=(3, 3, 3, 4),
    caller=2,
    with_players=(0, 3),
    last_estimator=1,
    risks=1,
)


def test_G10_three_three_three_three_all_with():
    result = score_round(G10A)
    assert result.total == 12
    assert result.calls == 4
    assert result.multiplier == 2
    assert result.scores == (46, -42, 46, 46)


def test_G10_three_three_three_two_with_risk():
    result = score_round(G10B)
    assert result.total == 11
    assert result.calls == 3
    assert result.multiplier == 2
    assert result.scores == (46, -22, 46, -22)


# Scoring rules


def test_SCR_1_base_win_and_loss():
    result = score_round(G1)
    assert result.seats[1].base_points == 10 + 3
    assert result.seats[2].base_points == -abs(4 - 3)
    big_miss = RoundOutcome(estimates=(5, 0, 4, 2), tricks=(5, 3, 3, 2), caller=0, last_estimator=3)
    assert score_round(big_miss).seats[1].base_points == -3


def test_SCR_1_win_base_value_is_configurable():
    assert score_round(G1, cfg(win_base=5)).scores == (20, 8, -11, 7)


@pytest.mark.parametrize(
    ("tricks", "estimates", "expected"),
    [
        ((4, 3, 0, 6), (4, 4, 0, 6), 25),  # Over, Dash caller takes 0: win
        ((3, 3, 1, 6), (4, 4, 0, 6), -25),  # Over, Dash caller takes a trick: loss
        ((4, 3, 0, 6), (4, 2, 0, 6), 33),  # Under, win
        ((4, 2, 1, 6), (4, 2, 0, 6), -33),  # Under, loss
    ],
)
def test_SCR_2_dash_call_values(
    tricks: tuple[int, int, int, int], estimates: tuple[int, int, int, int], expected: int
):
    outcome = RoundOutcome(
        estimates=estimates, tricks=tricks, caller=3, dash_callers=(2,), last_estimator=1
    )
    assert score_round(outcome).seats[2].dash_points == expected


def test_SCR_2_dash_point_values_are_configurable():
    result = score_round(G3, cfg(dash_over=20))
    assert result.seats[2].dash_points == 20
    assert score_round(G4, cfg(dash_under=30)).seats[0].dash_points == -30


def test_SCR_3_caller_and_with_bonus():
    result = score_round(G2)
    assert result.seats[1].call_points == 10
    assert result.seats[2].call_points == 10
    assert result.seats[3].call_points == 0
    failed = score_round(G7)
    assert failed.seats[3].call_points == -10
    assert score_round(G2, cfg(caller_with_bonus=7)).seats[1].call_points == 7


def test_SCR_4_risk_points_per_risk_on_win_and_loss():
    assert score_round(G2).seats[0].risk_points == -20
    made = RoundOutcome(
        estimates=(0, 4, 4, 1), tricks=(0, 4, 4, 5), caller=1, with_players=(2,), last_estimator=0,
        risks=2,
    )  # fmt: skip
    assert score_round(made).seats[0].risk_points == 20
    assert score_round(G2, cfg(risk_points=5)).seats[0].risk_points == -10


def test_SCR_5_only_winner_and_only_loser():
    only_loser = score_round(G1)
    assert [s.only_points for s in only_loser.seats] == [0, 0, -10, 0]
    only_winner = RoundOutcome(
        estimates=(5, 3, 4, 2), tricks=(5, 2, 2, 4), caller=0, last_estimator=3
    )
    assert [s.only_points for s in score_round(only_winner).seats] == [10, 0, 0, 0]
    two_two = score_round(G2)
    assert [s.only_points for s in two_two.seats] == [0, 0, 0, 0]
    assert score_round(G1, cfg(only_bonus=4)).seats[2].only_points == -4


def test_SCR_5_applies_to_dash_and_super_callers_by_default():
    assert score_round(G4).seats[0].only_points == -10
    assert score_round(G5B).seats[0].only_points == -10
    solo_super_win = dataclasses.replace(G5A, tricks=(8, 1, 2, 2))
    assert score_round(solo_super_win).seats[0].only_points == 10
    assert (
        score_round(solo_super_win, cfg(only_winner_for_super_calls=False)).seats[0].only_points
        == 0
    )
    assert (
        score_round(
            solo_super_win,
            cfg(only_winner_for_super_calls=True, only_loser_for_super_calls=False),
        )
        .seats[0]
        .only_points
        == 10
    )
    assert (
        score_round(
            G5B,
            cfg(only_winner_for_super_calls=False, only_loser_for_super_calls=True),
        )
        .seats[0]
        .only_points
        == -10
    )


@pytest.mark.parametrize(("bid", "made", "missed"), [(8, 64, -32), (9, 81, -40), (10, 100, -50)])
def test_SCR_6_super_call_values(bid: int, made: int, missed: int):
    others = 13 - bid
    win = RoundOutcome(
        estimates=(bid, others, 0, 1), tricks=(bid, others, 0, 0), caller=0, last_estimator=3
    )
    loss = dataclasses.replace(win, tricks=(bid - 1, others + 1, 0, 0))
    assert score_round(win).seats[0].super_points == made
    assert score_round(win).seats[0].call_points == 0
    assert score_round(loss).seats[0].super_points == missed
    assert score_round(loss).seats[0].call_points == 0


def test_SCR_6_applies_to_with_players_at_8_or_more():
    outcome = RoundOutcome(
        estimates=(9, 9, 0, 0), tricks=(9, 4, 0, 0), caller=0, with_players=(1,), last_estimator=3
    )
    result = score_round(outcome)
    assert result.seats[1].is_super
    assert result.seats[1].super_points == -40
    assert result.seats[1].base_points == 0
    assert result.seats[1].call_points == 0


def test_SCR_6_threshold_follows_super_call_min():
    result = score_round(G5A, cfg(super_call_min=9))
    assert not result.seats[0].is_super
    assert result.seats[0].base_points == 18


def test_SCR_7_carry_multiplies_every_score():
    result = score_round(dataclasses.replace(G1, carry=4))
    assert result.multiplier == 4
    assert result.scores == (100, 52, -44, 48)


def test_SCR_8_three_calls_double_the_round():
    assert score_round(G6, cfg(all_lose_doubles_next=False)).multiplier == 2
    assert score_round(G10A).multiplier == 2
    assert score_round(G2).multiplier == 1


def test_SCR_8_calls_count_risk_and_dash_counts_each_dash_and_one_for_risk():
    on = cfg(calls_count_risk_and_dash=True)
    assert score_round(G3, on).calls == 2  # Caller + one Dash caller
    assert score_round(G2, on).calls == 3  # Caller + With + 1 for Risk (2 Risks count once)
    assert score_round(G2).calls == 2


def test_SCR_8_calls_to_double_threshold_is_configurable():
    assert score_round(G10B, cfg(calls_to_double=4)).multiplier == 1
    assert score_round(G2, cfg(calls_to_double=2)).multiplier == 2


def test_SCR_9_all_lose_scores_zero_and_carries():
    result = score_round(dataclasses.replace(G6, carry=2))
    assert result.multiplier == 0
    assert result.scores == (0, 0, 0, 0)
    assert result.next_carry == 4


def test_SCR_10_multipliers_multiply():
    result = score_round(dataclasses.replace(G10A, carry=2))
    assert result.multiplier == 4
    assert result.scores == (92, -84, 92, 92)


# next_carry (GAM-4)


def test_GAM_4_carry_doubles_after_skipped_or_all_lose_and_resets_otherwise():
    assert next_carry(1, skipped=True, all_lost=False, config=DEFAULT) == 2
    assert next_carry(2, skipped=False, all_lost=True, config=DEFAULT) == 4
    assert next_carry(4, skipped=True, all_lost=False, config=DEFAULT) == 8
    assert next_carry(8, skipped=False, all_lost=False, config=DEFAULT) == 1


def test_GAM_4_carry_stacks_off_always_gives_two():
    off = cfg(carry_stacks=False)
    assert next_carry(1, skipped=True, all_lost=False, config=off) == 2
    assert next_carry(2, skipped=True, all_lost=False, config=off) == 2
    assert next_carry(2, skipped=False, all_lost=False, config=off) == 1


def test_SCR_9_all_lose_doubles_next_off_gives_no_carry():
    off = cfg(all_lose_doubles_next=False)
    assert next_carry(2, skipped=False, all_lost=True, config=off) == 1
    assert next_carry(1, skipped=True, all_lost=False, config=off) == 2


# Skipped rounds and validation


def test_GAM_3_skipped_round_scores_zero_for_everyone():
    result = score_round(RoundOutcome(skipped=True, dash_callers=(0, 1, 2, 3), carry=2))
    assert result.scores == (0, 0, 0, 0)
    assert all(s.points == 0 for s in result.seats)
    assert result.multiplier == 2  # the carry it was played at; every component is 0
    assert all(s.multiplier == 2 and s.is_dash for s in result.seats)
    assert result.over is None
    assert result.total is None
    assert result.caller is None
    assert result.next_carry == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimates": (5, 3, 4, 1), "tricks": (5, 3, 3, 2), "caller": 0},  # estimates total 13
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 3), "caller": 0},  # tricks total 14
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 2), "caller": None},
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 2), "caller": 0, "dash_callers": (1,)},
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 2), "caller": 0, "with_players": (0,)},
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 2), "caller": 0, "risks": -1},
        {"estimates": (5, 3, 4, 2), "tricks": (5, 3, 3, 2), "caller": 0, "carry": 0},
        {"estimates": (5, 3, 4), "tricks": (5, 3, 3, 2), "caller": 0},
    ],
)
def test_round_outcome_rejects_impossible_inputs(kwargs: dict[str, object]):
    with pytest.raises(ValueError, match=r"must|never|needs"):
        RoundOutcome(**kwargs)  # type: ignore[arg-type]  # deliberately bad inputs


# Properties


@st.composite
def outcomes(draw: st.DrawFn) -> RoundOutcome:
    tricks = draw(
        st.lists(st.integers(0, 13), min_size=3, max_size=3).filter(lambda t: sum(t) <= 13)
    )
    tricks = [*tricks, 13 - sum(tricks)]
    caller = draw(st.integers(0, 3))
    others = [s for s in range(4) if s != caller]
    dash = draw(st.lists(st.sampled_from(others), unique=True, max_size=3))
    estimators = [s for s in others if s not in dash]
    withs = draw(st.lists(st.sampled_from(estimators), unique=True)) if estimators else []
    estimates = [0 if s in dash else draw(st.integers(0, 13)) for s in range(4)]
    if sum(estimates) == 13:
        estimates[caller] += 1 if estimates[caller] < 13 else -1
    last = draw(st.sampled_from(estimators)) if estimators else None
    risks = abs(sum(estimates) - 13) // 2 if last is not None else 0
    return RoundOutcome(
        estimates=(estimates[0], estimates[1], estimates[2], estimates[3]),
        tricks=(tricks[0], tricks[1], tricks[2], tricks[3]),
        caller=caller,
        dash_callers=tuple(sorted(dash)),
        with_players=tuple(sorted(withs)),
        last_estimator=last,
        risks=risks,
        carry=draw(st.sampled_from([1, 2, 4])),
    )


@given(outcome=outcomes(), flags=st.fixed_dictionaries({
    "only_bonus_for_dash_calls": st.booleans(),
    "only_winner_for_super_calls": st.booleans(),
    "only_loser_for_super_calls": st.booleans(),
    "dash_replaces_base": st.booleans(),
    "super_call_scoring": st.booleans(),
    "calls_count_risk_and_dash": st.booleans(),
    "all_lose_doubles_next": st.booleans(),
    "carry_stacks": st.booleans(),
}))  # fmt: skip
def test_components_times_multiplier_is_the_final_score(
    outcome: RoundOutcome, flags: dict[str, bool]
):
    result = score_round(outcome, cfg(**flags))
    for seat in result.seats:
        assert seat.points * seat.multiplier == seat.score
        assert seat.multiplier == result.multiplier
    assert any(not seat.won for seat in result.seats)
    assert result.scores == tuple(seat.score for seat in result.seats)
