"""Trick play inside a Round (PLY-1 to PLY-5) and the open golden rounds G1-G6 end to end."""

import random

import pytest

from estimation.engine.actions import PASS, WITH, card_of, is_play, play_action
from estimation.engine.cards import (
    CLUBS,
    DIAMONDS,
    HEARTS,
    SPADES,
    SUIT_MASKS,
    deal,
    suit_of,
    trick_winner,
)
from estimation.engine.config import RulesConfig
from estimation.engine.round import IllegalActionError, Phase, Round

from .helpers import DASH, NO, NO_DASH_ROUND, bid, build_play, est, new_round, play_cards, run


def play_round(round_no: int = 1, seed: int = 0) -> Round:
    """Round with seat 0 (or F) Caller at 4 Clubs and everyone else estimating their minimum."""
    rnd = run(new_round(round_no, seed=seed), *NO_DASH_ROUND, bid(4, CLUBS), PASS, PASS, PASS, PASS)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[0])
    assert rnd.phase is Phase.PLAY
    return rnd


def test_TBL_2_every_turn_in_a_round_passes_counterclockwise():
    for seed in range(20):
        rnd = Round.new(1 + seed % 13, deal(seed, 1 + seed % 13))
        rng = random.Random(seed)
        while rnd.phase is not Phase.DONE:
            rnd.step(rng.choice(rnd.legal_actions()))
        assert [seat for seat, _ in rnd.log[:4]] == [(rnd.first + k) % 4 for k in range(4)]
        plays = [seat for seat, action in rnd.log if is_play(action)]
        for start in range(0, len(plays), 4):
            leader = plays[start]
            assert plays[start : start + 4] == [(leader + k) % 4 for k in range(4)]


def test_PLY_1_caller_leads_the_first_trick():
    rnd = run(new_round(1), *NO_DASH_ROUND, PASS, bid(4, HEARTS), PASS, PASS, PASS)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[0])
    assert rnd.current_player == 1
    assert rnd.observation(0).leader == 1


def test_PLY_2_followers_must_follow_suit_in_a_round():
    rnd = play_round()
    lead = rnd.legal_actions()[0]
    rnd.step(lead)
    led = suit_of(card_of(lead))
    seat = rnd.current_player
    assert seat is not None
    hand = rnd.observation(seat).hand
    follow = hand & SUIT_MASKS[led]
    legal_cards = [card_of(a) for a in rnd.legal_actions()]
    if follow:
        assert all(suit_of(card) == led for card in legal_cards)
        off_suit = next(card for card in range(52) if hand >> card & 1 and suit_of(card) != led)
        with pytest.raises(IllegalActionError):
            rnd.step(play_action(off_suit))
    else:
        assert len(legal_cards) == hand.bit_count()


def test_PLY_2_cannot_play_a_card_not_in_hand():
    rnd = play_round()
    hand = rnd.observation(0).hand
    missing = next(card for card in range(52) if not hand >> card & 1)
    with pytest.raises(IllegalActionError):
        rnd.step(play_action(missing))


def test_PLY_3_round_awards_each_trick_by_the_trick_rules():
    for seed in range(20):
        rnd = play_round(seed=seed)
        while rnd.phase is Phase.PLAY:
            rnd.step(rnd.legal_actions()[-1])
        obs = rnd.observation(0)
        for trick in obs.tricks:
            pairs = [((trick.leader + i) % 4, card) for i, card in enumerate(trick.cards)]
            assert trick_winner(pairs, CLUBS) == trick.winner


def test_PLY_4_trick_winner_leads_the_next_trick():
    rnd = play_round()
    for _ in range(4):
        rnd.step(rnd.legal_actions()[0])
    obs = rnd.observation(0)
    assert len(obs.tricks) == 1
    assert rnd.current_player == obs.tricks[0].winner == obs.leader
    assert obs.trick == ()
    assert sum(obs.tricks_won) == 1


def test_PLY_4_winner_may_lead_trump():
    hands, cards = build_play(SPADES, leader=0, targets=(5, 3, 3, 2))
    rnd = run(Round.new(1, hands), *NO_DASH_ROUND, bid(5, SPADES), PASS, PASS, PASS, PASS)
    run(rnd, est(3), est(4), est(2))
    play_cards(rnd, cards[:48])
    leader = rnd.current_player
    assert leader is not None
    trump_cards = [card_of(a) for a in rnd.legal_actions() if suit_of(card_of(a)) == SPADES]
    assert trump_cards  # the closing seat leads its trump


def test_PLY_5_round_is_scored_after_13_tricks():
    rnd = play_round()
    plays = 0
    while rnd.phase is Phase.PLAY:
        rnd.step(rnd.legal_actions()[0])
        plays += 1
    assert plays == 52
    assert rnd.phase is Phase.DONE
    assert rnd.current_player is None
    result = rnd.result()
    assert sum(seat.tricks for seat in result.seats) == 13
    assert result.scores == tuple(seat.score for seat in result.seats)


def test_G1_plain_over_round_full_round():
    hands, cards = build_play(SPADES, leader=0, targets=(5, 3, 3, 2))
    rnd = run(Round.new(1, hands), *NO_DASH_ROUND, bid(5, SPADES), PASS, PASS, PASS, PASS)
    run(rnd, est(3), est(4))
    assert est(1) not in rnd.legal_actions()
    rnd.step(est(2))
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.scores == (25, 13, -11, 12)
    assert (result.total, result.over) == (14, True)


@pytest.mark.parametrize(
    ("config", "scores"),
    [
        (RulesConfig(), (-23, 24, 24, -1)),
        (RulesConfig(calls_count_risk_and_dash=True), (-46, 48, 48, -2)),
    ],
)
def test_G2_under_round_with_with_and_risk_full_round(
    config: RulesConfig, scores: tuple[int, int, int, int]
):
    hands, cards = build_play(HEARTS, leader=1, targets=(3, 4, 4, 2))
    rnd = run(Round.new(2, hands, config=config), *NO_DASH_ROUND)
    run(rnd, bid(4, HEARTS), WITH, PASS, PASS, PASS, PASS)
    assert rnd.legal_actions() == [est(4)]
    run(rnd, est(4), est(1))
    assert est(4) not in rnd.legal_actions()
    rnd.step(est(0))
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.scores == scores
    assert result.seats[0].risks == 2


@pytest.mark.parametrize(
    ("config", "dash_score"),
    [(RulesConfig(), 25), (RulesConfig(dash_replaces_base=False), 35)],
)
def test_G3_dash_call_wins_in_an_over_round_full_round(config: RulesConfig, dash_score: int):
    hands, cards = build_play(CLUBS, leader=3, targets=(4, 3, 0, 6))
    rnd = run(Round.new(1, hands, config=config), NO, NO, DASH, NO)
    run(rnd, bid(5, DIAMONDS), PASS, bid(6, CLUBS), PASS, PASS)
    run(rnd, est(4))
    assert est(3) not in rnd.legal_actions()
    rnd.step(est(4))
    play_cards(rnd, cards)
    assert rnd.result().scores == (14, -11, dash_score, 26)


@pytest.mark.parametrize(
    ("config", "dash_score"),
    [
        (RulesConfig(), -43),
        (RulesConfig(dash_replaces_base=False), -44),
        (RulesConfig(only_bonus_for_dash_calls=False), -33),
    ],
)
def test_G4_dash_call_fails_in_an_under_round_full_round(config: RulesConfig, dash_score: int):
    hands, cards = build_play(SPADES, leader=1, targets=(1, 5, 3, 4))
    rnd = run(Round.new(2, hands, config=config), NO, NO, NO, DASH)
    run(rnd, bid(5, SPADES), PASS, PASS, PASS, est(3))
    assert est(5) not in rnd.legal_actions()
    rnd.step(est(4))
    play_cards(rnd, cards)
    assert rnd.result().scores == (dash_score, 25, 13, 14)


@pytest.mark.parametrize(
    ("config", "made", "failed"),
    [
        (RulesConfig(), (74, 12, 11, -11), (-52, 12, 11, 13)),
        (RulesConfig(super_call_scoring=False), (28, 12, 11, -11), (-21, 12, 11, 13)),
        (RulesConfig(only_bonus_for_super_calls=False), (74, 12, 11, -11), (-42, 12, 11, 13)),
    ],
)
def test_G5_super_call_at_8_full_round(
    config: RulesConfig, made: tuple[int, int, int, int], failed: tuple[int, int, int, int]
):
    for tricks, scores in (((8, 2, 1, 2), made), ((7, 2, 1, 3), failed)):
        hands, cards = build_play(SPADES, leader=0, targets=tricks)
        rnd = run(Round.new(1, hands, config=config), *NO_DASH_ROUND)
        run(rnd, bid(8, SPADES), PASS, PASS, PASS, PASS, est(2), est(1), est(3))
        play_cards(rnd, cards)
        assert rnd.result().scores == scores


@pytest.mark.parametrize(
    ("config", "scores", "next_carry"),
    [
        (RulesConfig(), (0, 0, 0, 0), 2),
        (RulesConfig(all_lose_doubles_next=False), (-22, -22, -24, -6), 1),
    ],
)
def test_G6_everyone_loses_full_round(
    config: RulesConfig, scores: tuple[int, int, int, int], next_carry: int
):
    hands, cards = build_play(CLUBS, leader=0, targets=(5, 3, 2, 3))
    rnd = run(Round.new(1, hands, config=config), *NO_DASH_ROUND)
    run(rnd, bid(4, CLUBS), WITH, PASS, PASS, PASS, PASS)
    assert rnd.legal_actions() == [est(4)]
    run(rnd, est(4), est(4))
    assert est(1) not in rnd.legal_actions()
    rnd.step(est(0))
    play_cards(rnd, cards)
    result = rnd.result()
    assert result.all_lose
    assert result.scores == scores
    assert result.next_carry == next_carry
