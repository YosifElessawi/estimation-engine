"""What a seat knows during card play (SPEC §6 M3): played cards and voids, from its Observation."""

import random

from estimation.bots.knowledge import read_knowledge
from estimation.engine import (
    NO_DASH_CALL,
    PASS,
    SPADES,
    Phase,
    Round,
    bid_action,
    deal,
    estimate_action,
    parse_card,
    play_action,
)
from estimation.engine.cards import SUIT_MASKS

from .helpers import round_with

NO_DASH = (NO_DASH_CALL,) * 4


def random_play_position(seed: int, plays: int) -> Round:
    """A random round advanced to card play, then ``plays`` cards into it."""
    rng = random.Random(seed)
    rnd = Round.new(1 + seed % 13, deal(seed, 1 + seed % 13))
    while rnd.phase is not Phase.PLAY:
        rnd.step(rng.choice(rnd.legal_actions()))
        if rnd.phase is Phase.DONE:
            rnd = Round.new(1 + seed % 13, deal(seed + 1000, 1 + seed % 13))
            seed += 1000
    for _ in range(plays):
        rnd.step(rng.choice(rnd.legal_actions()))
    return rnd


def test_a_seat_that_did_not_follow_suit_is_void_in_the_led_suit():
    # Seat 1 holds only Clubs and Diamonds, and seat 0 (dealt every Spade) leads a Spade.
    hand = "2C 3C 4C 5C 6C 7C 8C 2D 3D 4D 5D 6D 7D"
    script = (*NO_DASH, bid_action(4, SPADES), PASS, PASS, PASS, PASS)
    rnd = round_with(1, hand, *script, estimate_action(1), estimate_action(1), estimate_action(1))
    assert rnd.phase is Phase.PLAY
    rnd.step(play_action(parse_card("AS")))
    rnd.step(play_action(parse_card("2C")))
    know = read_knowledge(rnd.observation(2))
    assert know.voids == (0, 1 << SPADES, 0, 0)


def test_tracks_played_cards_hand_sizes_and_unseen_cards_exactly():
    for seed in range(60):
        rnd = random_play_position(seed, plays=seed % 50)
        views = [rnd.observation(seat) for seat in range(4)]
        for seat in range(4):
            know = read_knowledge(views[seat])
            assert know.hand == views[seat].hand
            assert know.sizes == tuple(view.hand.bit_count() for view in views)
            others = [view.hand for other, view in enumerate(views) if other != seat]
            assert know.unseen == others[0] | others[1] | others[2]
            remaining = views[0].hand | views[1].hand | views[2].hand | views[3].hand
            assert know.played == (2**52 - 1) & ~remaining
            for other in range(4):
                assert know.played_by[other] & views[other].hand == 0


def test_every_void_it_infers_is_real():
    found = 0
    for seed in range(80):
        rnd = random_play_position(seed, plays=20 + seed % 30)
        views = [rnd.observation(seat) for seat in range(4)]
        know = read_knowledge(views[seed % 4])
        for other in range(4):
            for suit in range(4):
                if know.voids[other] >> suit & 1:
                    found += 1
                    assert views[other].hand & SUIT_MASKS[suit] == 0
    assert found > 20
