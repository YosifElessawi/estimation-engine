"""The Round API contract (SPEC §5.4): step errors, legal_mask, clone, replay, observation."""

import random

import numpy as np
import pytest

from estimation.engine.actions import NUM_ACTIONS, PASS, play_action
from estimation.engine.cards import CLUBS, DIAMONDS, FULL_DECK, HEARTS, SANS, SPADES, deal
from estimation.engine.round import FIXED_ORDER, IllegalActionError, Phase, Round

from .helpers import DASH, NO, NO_DASH_ROUND, bid, new_round, run


def random_round(seed: int, round_no: int = 1, fixed_trump: int | None = None) -> Round:
    rng = random.Random(seed)
    rnd = Round.new(round_no, deal(seed, round_no), fixed_trump)
    while rnd.phase is not Phase.DONE:
        rnd.step(rng.choice(rnd.legal_actions()))
    return rnd


def same_state(a: Round, b: Round) -> bool:
    """SPEC §5.4: phase, current player, every observation and (once DONE) the result."""
    if a.phase is not b.phase or a.current_player != b.current_player:
        return False
    if any(a.observation(seat) != b.observation(seat) for seat in range(4)):
        return False
    return a.phase is not Phase.DONE or a.result() == b.result()


def test_step_rejects_illegal_actions_without_changing_state():
    rnd = new_round(1)
    before = rnd.clone()
    for action in (2, 52, 53, 54, 68, 119, -1, 120):
        with pytest.raises(IllegalActionError):
            rnd.step(action)
    assert same_state(rnd, before)


def test_step_rejects_any_action_once_the_round_is_done():
    rnd = run(new_round(1), DASH, DASH, DASH, DASH)
    with pytest.raises(IllegalActionError):
        rnd.step(NO)


@pytest.mark.parametrize("action", [1.0, "1", None])
def test_step_rejects_non_integer_actions(action: object):
    with pytest.raises(IllegalActionError):
        new_round(1).step(action)  # type: ignore[arg-type]  # deliberately wrong type


def test_step_accepts_numpy_integers():
    rnd = new_round(1)
    rnd.step(np.int64(1))  # type: ignore[arg-type]  # RL policies return numpy integers
    assert rnd.observation(0).log == ((0, 1),)
    assert type(rnd.log[0][1]) is int


def test_legal_mask_matches_legal_actions_in_every_phase():
    rng = random.Random(3)
    rnd = new_round(5)
    while True:
        mask = rnd.legal_mask()
        assert mask.dtype == np.bool_
        assert mask.shape == (NUM_ACTIONS,)
        assert np.flatnonzero(mask).tolist() == rnd.legal_actions()
        if rnd.phase is Phase.DONE:
            break
        rnd.step(rng.choice(rnd.legal_actions()))
    assert not rnd.legal_mask().any()


def test_legal_actions_are_sorted_and_unique():
    for seed in range(30):
        rng = random.Random(seed)
        rnd = (
            Round.new(14 + seed % 5, deal(seed, 1), FIXED_ORDER[seed % 5])
            if seed % 2
            else new_round(1, seed=seed)
        )
        while rnd.phase is not Phase.DONE:
            legal = rnd.legal_actions()
            assert legal == sorted(set(legal))
            rnd.step(rng.choice(legal))


def test_result_is_unavailable_before_the_round_ends():
    with pytest.raises(RuntimeError, match="not over"):
        new_round(1).result()


def test_new_validates_its_inputs():
    hands = deal(0, 1)
    with pytest.raises(ValueError, match="hands"):
        Round.new(1, (hands[0], hands[1], hands[2], hands[2]))
    with pytest.raises(ValueError, match="hands"):
        Round.new(1, (hands[0], hands[1], hands[2]))  # type: ignore[arg-type]  # deliberately short
    with pytest.raises(ValueError, match="round"):
        Round.new(0, hands)
    with pytest.raises(ValueError, match="trump"):
        Round.new(14, hands, 5)
    with pytest.raises(ValueError, match="carry"):
        Round.new(1, hands, carry=0)
    with pytest.raises(ValueError, match="scores"):
        Round.new(1, hands, scores=(0, 0, 0))  # type: ignore[arg-type]  # deliberately short


def test_new_keeps_the_deal_and_context():
    hands = deal(4, 7)
    rnd = Round.new(7, hands, carry=2, scores=(10, -5, 0, 3))
    obs = rnd.observation(2)
    assert rnd.hands == hands
    assert obs.hand == hands[2]
    assert (obs.round_no, obs.carry, obs.scores) == (7, 2, (10, -5, 0, 3))
    assert obs.fixed_trump is None
    assert obs.fixed_left == FIXED_ORDER
    assert obs.replays == 0


@pytest.mark.parametrize(
    ("round_no", "fixed_trump", "fixed_left", "replays"),
    [
        (14, SANS, FIXED_ORDER, 0),
        (16, HEARTS, (HEARTS, DIAMONDS, CLUBS), 0),
        (17, HEARTS, (HEARTS, DIAMONDS, CLUBS), 1),  # G9: Hearts replayed once
        (20, SPADES, FIXED_ORDER[1:], 5),
        (38, CLUBS, (CLUBS,), 20),
    ],
)
def test_fixed_round_context_follows_from_round_number_and_fixed_trump(
    round_no: int, fixed_trump: int, fixed_left: tuple[int, ...], replays: int
):
    """GAM-2 counts every round, replays included, so each fixed-trump round either completes a
    fixed trump or replays one: replays so far = round_no - 14 - position of the fixed trump."""
    obs = Round.new(round_no, deal(0, round_no), fixed_trump).observation(0)
    assert (obs.fixed_trump, obs.trump) == (fixed_trump, fixed_trump)
    assert obs.fixed_left == fixed_left
    assert obs.replays == replays


@pytest.mark.parametrize(
    ("round_no", "fixed_trump"),
    [
        (14, None),  # rounds after 13 are fixed-trump rounds (FIX-1)
        (13, SANS),  # rounds 1-13 are open
        (14, SPADES),  # Spades comes after Sans, so not before round 15
        (17, CLUBS),
        (39, CLUBS),  # 21 replays: more than max_fixed_replays allows
    ],
)
def test_GAM_1_round_must_fit_the_game_schedule(round_no: int, fixed_trump: int | None):
    with pytest.raises(ValueError, match="round"):
        Round.new(round_no, deal(0, round_no), fixed_trump)


def test_observation_shows_own_hand_shrinking_as_cards_are_played():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(4, SANS), PASS, PASS, PASS, PASS)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[0])
    lead = rnd.legal_actions()[0]
    rnd.step(lead)
    obs = rnd.observation(0)
    assert obs.hand == rnd.hands[0] & ~(1 << (lead - 68))
    assert obs.hand.bit_count() == 12
    assert obs.trick == ((0, lead - 68),)
    assert rnd.observation(1).hand == rnd.hands[1]


def test_observation_rejects_a_bad_seat():
    with pytest.raises(ValueError, match="seat"):
        new_round(1).observation(4)


def test_clone_is_independent():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(5, SPADES))
    copy = rnd.clone()
    assert same_state(rnd, copy)
    copy.step(PASS)
    assert not same_state(rnd, copy)
    assert rnd.observation(0).passed == ()
    assert copy.observation(0).passed == (1,)


def test_clone_mid_play_then_finish_both_identically():
    rng = random.Random(9)
    rnd = new_round(3, seed=9)
    for _ in range(40):
        rnd.step(rng.choice(rnd.legal_actions()))
    copy = rnd.clone()
    while rnd.phase is not Phase.DONE:
        action = rnd.legal_actions()[0]
        rnd.step(action)
        copy.step(action)
    assert same_state(rnd, copy)


@pytest.mark.parametrize("seed", range(25))
def test_replay_of_the_action_log_reproduces_the_same_state(seed: int):
    fixed = None if seed % 3 else FIXED_ORDER[seed % 5]
    round_no = 14 + seed if fixed is not None else 1 + seed % 13
    original = random_round(seed, round_no, fixed)
    actions = [action for _, action in original.log]
    replayed = Round.replay(
        round_no, deal(seed, round_no), fixed, 1, (0, 0, 0, 0), original.config, actions
    )
    assert same_state(original, replayed)
    partial = Round.replay(
        round_no, deal(seed, round_no), fixed, 1, (0, 0, 0, 0), original.config, actions[:20]
    )
    midway = Round.new(round_no, deal(seed, round_no), fixed)
    run(midway, *actions[:20])
    assert same_state(partial, midway)


def test_log_records_every_action_with_its_seat():
    rnd = run(new_round(2), NO, DASH, NO, NO, bid(4, CLUBS))
    assert rnd.log == ((1, NO), (2, DASH), (3, NO), (0, NO), (1, bid(4, CLUBS)))


def test_round_uses_every_card_once_across_13_tricks():
    rnd = random_round(11)
    if rnd.result().skipped:
        rnd = random_round(12)
    obs = rnd.observation(0)
    seen = 0
    for trick in obs.tricks:
        for card in trick.cards:
            assert not seen >> card & 1
            seen |= 1 << card
    assert len(obs.tricks) == 13
    assert seen == FULL_DECK


def test_fixed_order_is_sans_spades_hearts_diamonds_clubs():
    assert FIXED_ORDER == (SANS, SPADES, HEARTS, DIAMONDS, CLUBS)


def test_play_action_ids_cover_the_hand():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(4, SANS), PASS, PASS, PASS, PASS)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[0])
    assert rnd.legal_actions() == [
        play_action(card) for card in range(52) if rnd.hands[0] >> card & 1
    ]
