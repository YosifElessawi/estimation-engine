"""Dealing the unseen cards (SPEC §6 M3): deals that fit hand sizes, voids and played cards."""

import random
from collections import Counter

import pytest

from estimation.bots.sampling import deal_unseen
from estimation.engine import HEARTS, SPADES, mask_of, parse_hand
from estimation.engine.cards import SUIT_MASKS


def test_deals_hold_exactly_the_unseen_cards_in_the_right_sizes():
    rng = random.Random(1)
    for _ in range(500):
        cards = rng.sample(range(52), rng.randint(3, 39))
        sizes = [len(cards) // 3, len(cards) // 3, len(cards) - 2 * (len(cards) // 3)]
        hands = deal_unseen(mask_of(cards), sizes, [0, 0, 0], rng)
        assert [hand.bit_count() for hand in hands] == sizes
        assert hands[0] | hands[1] | hands[2] == mask_of(cards)
        assert not hands[0] & hands[1]
        assert not hands[0] & hands[2]
        assert not hands[1] & hands[2]


def test_deals_never_give_a_hand_a_suit_it_is_void_in():
    rng = random.Random(2)
    checked = 0
    for _ in range(3000):
        # Deal real hands first, so the voids always allow some deal.
        cards = rng.sample(range(52), 3 * rng.randint(1, 13))
        size = len(cards) // 3
        truth = [mask_of(cards[i * size : (i + 1) * size]) for i in range(3)]
        voids = [
            sum(
                1 << suit for suit in range(4) if not hand & SUIT_MASKS[suit] and rng.random() < 0.8
            )
            for hand in truth
        ]
        hands = deal_unseen(mask_of(cards), [size] * 3, voids, rng)
        assert hands[0] | hands[1] | hands[2] == mask_of(cards)
        for hand, void in zip(hands, voids, strict=True):
            assert hand.bit_count() == size
            for suit in range(4):
                if void >> suit & 1:
                    checked += 1
                    assert not hand & SUIT_MASKS[suit]
    assert checked > 1000


def test_a_forced_deal_is_found_without_dead_ends():
    # Hand 0 is void in three suits, so it must take all three Clubs.
    unseen = parse_hand("2C 3C 4C AH KH QH JH TH AS KS")
    voids = [0b1110, 0, 0]  # Diamonds, Hearts, Spades
    rng = random.Random(3)
    for _ in range(200):
        hands = deal_unseen(unseen, [3, 4, 3], voids, rng)
        assert hands[0] == parse_hand("2C 3C 4C")


def test_unconstrained_deals_are_uniform():
    unseen = parse_hand("2C 3C 4C 5C 6C 7C")
    rng = random.Random(4)
    counts = Counter(deal_unseen(unseen, [1, 2, 3], [0, 0, 0], rng)[0] for _ in range(6000))
    assert len(counts) == 6
    assert all(900 < n < 1100 for n in counts.values())


def test_a_void_hand_leaves_the_others_uniform_over_the_fitting_deals():
    # Hand 0 cannot hold Hearts: it takes one of the two Spades, and hand 1 then takes any one
    # of the three cards left. All 6 fitting deals are equally likely.
    unseen = parse_hand("2H 3H 2S 3S")
    rng = random.Random(5)
    deals = Counter(
        tuple(deal_unseen(unseen, [1, 1, 2], [1 << HEARTS, 0, 0], rng)) for _ in range(6000)
    )
    assert len(deals) == 6
    assert all(900 < n < 1100 for n in deals.values())
    assert all(hands[0] & SUIT_MASKS[SPADES] for hands in deals)


def test_the_same_generator_seed_gives_the_same_deal():
    unseen = parse_hand("2C 3C 4C 5C 6C 7C 8D 9D TH")
    first = deal_unseen(unseen, [3, 3, 3], [0, 1, 0], random.Random(9))
    assert first == deal_unseen(unseen, [3, 3, 3], [0, 1, 0], random.Random(9))


@pytest.mark.parametrize(
    ("sizes", "voids"),
    [([1, 1, 1], [1, 1, 1]), ([2, 2, 0], [0, 0, 0]), ([3, 0, 0], [1, 0, 0])],
)
def test_impossible_constraints_are_rejected(sizes: list[int], voids: list[int]):
    with pytest.raises(ValueError, match=r"deal|hand|void"):
        deal_unseen(parse_hand("2C 3C 4C"), sizes, voids, random.Random(0))
