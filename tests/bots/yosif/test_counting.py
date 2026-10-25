"""Counting a hand's trick chances by Yosif's rules (design 5.1)."""

import pytest

from estimation.bots.yosif.counting import Chance, Reads, count_hand, read_bids
from estimation.engine import CLUBS, DIAMONDS, NO_DASH_CALL, PASS, SANS, SPADES, bid_action
from estimation.engine.cards import parse_hand
from tests.bots.helpers import round_with

EXAMPLE = parse_hand("AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D")


def test_example_hand_in_spades():
    count = count_hand(EXAMPLE, SPADES)
    assert count.trumps[6] >= 0.85
    assert sorted(c.win for c in count.side) == [0.45, 0.45, 1.0]
    assert Chance(1.0, 0.5) in count.side  # the club void lets the diamond ace be thrown away


def test_queen_behind_the_bidder_is_worth_more():
    hand = parse_hand("AD QD 7D 4D 2C 3C 4C 5C 6C 7C 8C 9C TC")
    plain = count_hand(hand, CLUBS)
    behind = count_hand(hand, CLUBS, Reads(before=1 << DIAMONDS))
    in_front = count_hand(hand, CLUBS, Reads(after=1 << DIAMONDS))

    def queen(count) -> float:
        return max(c.win for c in count.side if c.win < 1)

    assert queen(behind) == pytest.approx(queen(plain) + 0.25)
    assert queen(in_front) == pytest.approx(queen(plain) - 0.25)


def test_low_cards_raise_the_dodge():
    covered = count_hand(parse_hand("KH 2H 3H 4H 5C 6C 7C 8C 9C 5D 6D 7D 8D"), CLUBS)
    bare = count_hand(parse_hand("KH 9H TH JH 5C 6C 7C 8C 9C 5D 6D 7D 8D"), CLUBS)
    assert covered.side[0].dodge > bare.side[0].dodge


def test_over_and_under_change_the_honours():
    hand = parse_hand("KH 5C 6C 7C 8C 9C TC 5D 6D 7D 8D 9D TD")
    assert count_hand(hand, CLUBS, over=False).side[0].win == 0.75
    assert count_hand(hand, CLUBS, over=True).side[0].win == pytest.approx(0.30 * 0.75)


def test_sans_counts_long_cards_and_has_no_trumps():
    count = count_hand(parse_hand("AS KS 9S 8S 7S 6S 2H 3H 4H 2D 3D 4D 5D"), SANS)
    assert count.trumps == (1.0,)
    assert len(count.side) == 4  # A, K and two long spades


def test_no_trumps_held_counts_zero_trump_tricks():
    count = count_hand(parse_hand("AH KH 2H 3H 4H AD 2D 3D 4D AC 2C 3C 4C"), SPADES)
    assert count.trumps == (1.0,)


def test_read_bids_finds_neighbours_and_bidders():
    # Round 2, so seat 1 is F and opens; seats 1 and 2 pass, seat 3 bids 4 Diamonds before seat 0.
    rnd = round_with(
        0,
        "AD QD 7D 4D 2C 3C 4C 5C 6C 7C 8C 9C TC",
        *[NO_DASH_CALL] * 4,
        PASS,
        PASS,
        bid_action(4, DIAMONDS),
        round_no=2,
    )
    assert rnd.current_player == 0
    reads = read_bids(rnd.observation(0))
    assert reads.before == 1 << DIAMONDS
    assert reads.after == 0
    assert reads.bidders[DIAMONDS] == 1
