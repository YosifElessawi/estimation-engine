"""The full game (SPEC §5.4 Bola): GAM-1 to GAM-5, FIX-1, FIX-5, G8 and G9 across rounds."""

import dataclasses
from collections.abc import Callable

import pytest

from estimation.engine.actions import PASS, is_estimate
from estimation.engine.bola import Bola, top_seats
from estimation.engine.cards import CLUBS, DIAMONDS, HEARTS, SANS, SPADES, deal
from estimation.engine.config import RulesConfig
from estimation.engine.round import FIXED_ORDER, IllegalActionError, Phase

from .helpers import NO, bid, est

Policy = Callable[[Bola], int]


def calm(bola: Bola) -> int:
    """No Dash Calls; the opener bids 4 Clubs and everyone else passes; lowest estimates and
    no super calls; first legal card. Rounds always play out and trumps never change."""
    legal = bola.legal_actions()
    phase = bola.phase
    if phase is Phase.DASH:
        return NO
    if phase is Phase.AUCTION:
        first_bid = bid(4, CLUBS)
        return first_bid if first_bid in legal else PASS
    if phase is Phase.ESTIMATION:
        return next(a for a in legal if is_estimate(a))
    return legal[0]


def passive(bola: Bola) -> int:
    """Like ``calm`` but nobody bids, so every open round is skipped."""
    return PASS if bola.phase is Phase.AUCTION else calm(bola)


def play_round(bola: Bola, policy: Policy) -> None:
    """Play the current round to its end with ``policy``."""
    round_no = bola.round_no
    while bola.round_no == round_no and not bola.is_over:
        bola.step(policy(bola))


def play_out(bola: Bola, policy: Policy) -> Bola:
    for _ in range(100_000):
        if bola.is_over:
            return bola
        bola.step(policy(bola))
    raise AssertionError("the game did not end")


def test_GAM_1_a_game_is_13_open_rounds_then_the_5_fixed_trump_rounds():
    bola = play_out(Bola(1), calm)
    assert bola.round_no == 18
    assert len(bola.history) == 18
    assert [r.round_no for r in bola.history] == list(range(1, 19))
    assert all(r.fixed_trump is None for r in bola.history[:13])
    assert [r.fixed_trump for r in bola.history[13:]] == list(FIXED_ORDER)


def test_FIX_1_fixed_trumps_follow_sans_spades_hearts_diamonds_clubs():
    bola = Bola(2)
    while bola.round_no < 14:
        play_round(bola, calm)
    for fixed in (SANS, SPADES, HEARTS, DIAMONDS, CLUBS):
        assert bola.round.fixed_trump == fixed
        assert bola.observation(0).fixed_left == FIXED_ORDER[FIXED_ORDER.index(fixed) :]
        play_round(bola, calm)
    assert bola.is_over


def test_GAM_2_first_seat_follows_the_round_number():
    bola = Bola(3)
    for round_no in range(1, 19):
        assert bola.round_no == round_no
        assert bola.current_player == (round_no - 1) % 4
        play_round(bola, calm)


def test_GAM_3_skipped_rounds_still_count_as_open_rounds():
    bola = Bola(4)
    for _ in range(13):
        play_round(bola, passive)
    assert bola.round_no == 14
    assert bola.round.fixed_trump == SANS
    assert all(r.skipped for r in bola.history)
    assert bola.scores == (0, 0, 0, 0)


def test_GAM_3_fixed_trump_rounds_are_never_skipped():
    bola = play_out(Bola(4), passive)
    assert [r.skipped for r in bola.history[13:]] == [False] * 5


def test_G8_skipped_round_carries_into_the_next_round():
    bola = Bola(37)  # a seed whose played rounds here are not all-lose (that would also carry)
    play_round(bola, calm)
    play_round(bola, calm)
    assert not bola.history[1].all_lose
    assert bola.round.carry == 1
    play_round(bola, passive)  # round 3: everyone passes
    assert bola.history[2].skipped
    assert bola.round_no == 4
    assert bola.round.carry == 2
    assert bola.observation(1).carry == 2
    play_round(bola, calm)
    assert bola.history[3].carry == 2
    assert not bola.history[3].all_lose
    assert bola.round.carry == 1  # round 5 is back to x1
    play_round(bola, calm)
    assert not bola.history[4].all_lose
    play_round(bola, passive)  # round 6
    play_round(bola, passive)  # round 7
    assert bola.round_no == 8
    assert bola.round.carry == 4


def test_G8_carry_stacks_off_caps_the_carry_at_2():
    bola = Bola(37, RulesConfig(carry_stacks=False))
    for policy in (calm, calm, passive, calm, calm, passive, passive):
        play_round(bola, policy)
    assert bola.round_no == 8
    assert bola.round.carry == 2


def test_GAM_4_all_lose_round_carries_into_the_next_round():
    for seed in range(200):
        bola = Bola(seed)
        while not bola.is_over:
            play_round(bola, calm)
            last = bola.history[-1]
            if last.all_lose and not bola.is_over:
                assert bola.round.carry == last.carry * 2
                return
    pytest.fail("no all-lose round found")


def test_GAM_5_game_score_is_the_sum_of_round_scores():
    bola = play_out(Bola(6), calm)
    for seat in range(4):
        assert bola.scores[seat] == sum(r.scores[seat] for r in bola.history)


def test_GAM_5_highest_score_wins_and_ties_share_the_win():
    assert top_seats((10, 30, 30, -5)) == (1, 2)
    assert top_seats((0, 0, 0, 0)) == (0, 1, 2, 3)
    assert top_seats((-3, -7, -1, -2)) == (2,)
    bola = play_out(Bola(7), calm)
    assert bola.winners() == top_seats(bola.scores)


def test_G9_changed_trump_is_replayed_and_the_game_ends_after_round_19():
    bola = Bola(8)
    while bola.round_no < 16:
        play_round(bola, calm)
    assert bola.round.fixed_trump == HEARTS
    assert bola.current_player == 3
    for action in (est(2), bid(8, SPADES), PASS, PASS, PASS, PASS, est(2), est(1), est(3)):
        bola.step(action)
    play_round(bola, calm)
    assert bola.history[-1].trump_changed
    assert (bola.round_no, bola.round.fixed_trump, bola.replays) == (17, HEARTS, 1)
    assert bola.observation(0).replays == 1
    assert bola.observation(0).fixed_left == (HEARTS, DIAMONDS, CLUBS)
    play_round(bola, calm)
    assert (bola.round_no, bola.round.fixed_trump) == (18, DIAMONDS)
    play_round(bola, calm)
    assert (bola.round_no, bola.round.fixed_trump) == (19, CLUBS)
    play_round(bola, calm)
    assert bola.is_over
    assert len(bola.history) == 19


def test_FIX_5_answer_in_the_fixed_trump_is_not_replayed():
    bola = Bola(8)
    while bola.round_no < 16:
        play_round(bola, calm)
    for action in (est(2), bid(8, SPADES), bid(9, HEARTS), PASS, PASS, PASS, PASS):
        bola.step(action)
    play_round(bola, calm)
    assert not bola.history[-1].trump_changed
    assert (bola.round_no, bola.round.fixed_trump, bola.replays) == (17, DIAMONDS, 0)


def super_call_always(bola: Bola) -> int:
    """In fixed-trump estimation the first seat super-calls; everything else is calm."""
    if bola.phase is Phase.ESTIMATION and bola.round.fixed_trump is not None:
        obs = bola.observation(0)
        if not obs.super_call and obs.caller is None:
            return bid(8, SANS if bola.round.fixed_trump != SANS else SPADES)
    return calm(bola)


@pytest.mark.parametrize("cap", [0, 1, 3])
def test_FIX_5_replay_cap_stops_replays(cap: int):
    bola = play_out(Bola(9, RulesConfig(max_fixed_replays=cap)), super_call_always)
    fixed_rounds = bola.history[13:]
    assert all(r.trump_changed for r in fixed_rounds)
    assert bola.replays == cap
    assert len(fixed_rounds) == 5 + cap
    assert [r.fixed_trump for r in fixed_rounds] == [SANS] * (cap + 1) + list(FIXED_ORDER[1:])


def test_bola_over_behaves_like_a_round_in_done():
    bola = play_out(Bola(10), calm)
    assert bola.phase is Phase.DONE
    assert bola.current_player is None
    assert bola.legal_actions() == []
    assert not bola.legal_mask().any()
    assert bola.observation(2).phase is Phase.DONE
    with pytest.raises(IllegalActionError):
        bola.step(0)


def test_bola_deals_each_round_from_seed_and_round_number():
    bola = Bola(11)
    for _ in range(3):
        assert bola.round.hands == deal(11, bola.round_no)
        play_round(bola, calm)


def test_bola_passes_scores_and_config_into_each_round():
    config = RulesConfig(win_base=7)
    bola = Bola(12, config)
    play_round(bola, calm)
    assert bola.round.config is config
    assert bola.observation(0).scores == bola.scores == bola.history[0].scores


def test_two_bolas_with_the_same_seed_and_actions_are_identical():
    first = play_out(Bola(13), calm)
    second = play_out(Bola(13), calm)
    assert first.scores == second.scores
    assert first.history == second.history
    third = play_out(Bola(14), calm)
    assert third.history != first.history


def test_bola_rejects_a_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        Bola(-1)


def test_bola_config_flags_reach_scoring():
    base = play_out(Bola(15), calm)
    changed = play_out(Bola(15, dataclasses.replace(RulesConfig(), win_base=11)), calm)
    assert base.scores != changed.scores
