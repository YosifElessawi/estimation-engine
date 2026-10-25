"""Trick-count odds and expected points (design 5.2), checked against the engine's scoring."""

import dataclasses

import pytest

from estimation.bots.yosif.counting import count_hand
from estimation.bots.yosif.odds import (
    Outlook,
    expected_points,
    make_chance,
    outlook,
    spread,
    stake,
)
from estimation.engine import SPADES, RoundOutcome, RulesConfig, score_round
from estimation.engine.cards import parse_hand

RULES = RulesConfig()
EXAMPLE = parse_hand("AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D")


def engine_points(outcome: RoundOutcome, seat: int, rules: RulesConfig = RULES) -> int:
    """A seat's SCR-1 to SCR-4 and SCR-6 points: everything ``stake`` covers."""
    s = score_round(outcome, rules).seats[seat]
    return s.base_points + s.dash_points + s.call_points + s.risk_points + s.super_points


G1 = RoundOutcome(
    estimates=(5, 3, 4, 2), tricks=(5, 3, 3, 2), caller=0, last_estimator=3, trump=SPADES
)
G2 = RoundOutcome(
    estimates=(0, 4, 4, 1),
    tricks=(3, 4, 4, 2),
    caller=1,
    with_players=(2,),
    last_estimator=0,
    risks=2,
    trump=2,
)
G4 = RoundOutcome(
    estimates=(0, 5, 3, 4),
    tricks=(1, 5, 3, 4),
    caller=1,
    dash_callers=(0,),
    last_estimator=3,
    trump=SPADES,
)
G5A = RoundOutcome(
    estimates=(8, 2, 1, 3), tricks=(8, 2, 1, 2), caller=0, last_estimator=3, trump=SPADES
)
G5B = dataclasses.replace(G5A, tricks=(7, 2, 1, 3))


def test_spread_of_two_coin_flips():
    assert spread([0.5, 0.5]) == [0.25, 0.5, 0.25]


def test_make_chance_when_any_count_is_reachable():
    o = Outlook(most=(0.0, 0.0, 1.0), least=(1.0, 0.0, 0.0))
    assert [make_chance(o, k) for k in range(3)] == [1.0, 1.0, 1.0]


def test_make_chance_is_zero_when_tricks_cannot_be_avoided():
    o = Outlook(most=(0.0, 0.0, 1.0), least=(0.0, 0.0, 1.0))
    assert make_chance(o, 1) == 0.0
    assert make_chance(o, 2) == 1.0


def test_stake_matches_G1_caller():
    s = stake(5, caller=True, dash=False, last_risks=0, over=True, rules=RULES)
    assert s.win == engine_points(G1, 0) == 25


def test_stake_matches_G1_plain_loss():
    s = stake(4, caller=False, dash=False, last_risks=0, over=True, rules=RULES)
    assert -(s.lose + s.per_trick * 1) == engine_points(G1, 2) == -1


def test_stake_matches_G2_last_estimator_two_risks():
    s = stake(0, caller=False, dash=False, last_risks=2, over=False, rules=RULES)
    assert -(s.lose + s.per_trick * 3) == engine_points(G2, 0) == -23


def test_stake_matches_G4_dash_under():
    s = stake(0, caller=False, dash=True, last_risks=0, over=False, rules=RULES)
    assert -(s.lose + s.per_trick * 1) == engine_points(G4, 0) == -33


def test_stake_matches_G5_super_call_made_and_missed():
    s = stake(8, caller=True, dash=False, last_risks=0, over=True, rules=RULES)
    assert s.win == engine_points(G5A, 0) == 74
    assert -(s.lose + s.per_trick * 1) == engine_points(G5B, 0) == -42


def test_stake_matches_super_call_scoring_off():
    rules = RulesConfig(super_call_scoring=False)
    s = stake(8, caller=True, dash=False, last_risks=0, over=True, rules=rules)
    assert s.win == engine_points(G5A, 0, rules) == 28
    assert -(s.lose + s.per_trick * 1) == engine_points(G5B, 0, rules) == -11


def test_dash_stake_with_over_unknown_uses_the_mean():
    s = stake(0, caller=False, dash=True, last_risks=0, over=None, rules=RULES)
    assert s.win == s.lose == pytest.approx((25 + 33) / 2)


def test_example_hand_scores_best_at_8_spades():
    o = outlook(count_hand(EXAMPLE, SPADES))
    points = {
        k: expected_points(
            o,
            k,
            stake(k, caller=True, dash=False, last_risks=0, over=None, rules=RULES),
            0.5,
            RULES,
        )
        for k in range(4, 14)
    }
    assert max(points, key=points.__getitem__) == 8
