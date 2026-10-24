"""HeuristicBot's hand model: expected tricks from A, K, Q and trump length (SPEC §6 M2)."""

import pytest

from estimation.bots.hand_eval import HandValues, expected_tricks, trump_values
from estimation.engine import CLUBS, DIAMONDS, HEARTS, SANS, SPADES, parse_hand


def test_each_ace_is_worth_one_trick_in_sans():
    hand = parse_hand("AS AH AD AC 2S 3S 4H 5H 6D 7D 2C 3C 4C")
    assert expected_tricks(hand, SANS) == pytest.approx(4.0)


def test_a_guarded_king_is_worth_more_than_a_bare_king():
    guarded = parse_hand("KS 2S 3H 4H 5H 6H 7D 8D 9D 2C 3C 4C 5C")
    bare = parse_hand("KS 2H 3H 4H 5H 6H 7D 8D 9D 2C 3C 4C 5C")
    assert expected_tricks(guarded, SANS) == pytest.approx(0.75)
    assert expected_tricks(bare, SANS) == pytest.approx(0.25)


def test_a_queen_needs_two_guards_to_count_fully():
    guarded = parse_hand("QS 2S 3S 4H 5H 6H 7D 8D 9D 2C 3C 4C 5C")
    short = parse_hand("QS 2S 3H 4H 5H 6H 7D 8D 9D 2C 3C 4C 5C")
    assert expected_tricks(guarded, SANS) == pytest.approx(0.4)
    assert expected_tricks(short, SANS) == pytest.approx(0.1)


def test_long_trumps_add_a_trick_each_beyond_the_third():
    hand = parse_hand("2S 3S 4S 5S 6S 7S 2H 3H 4H 2D 3D 2C 3C")
    assert expected_tricks(hand, SPADES) == pytest.approx(3.0)
    assert expected_tricks(hand, SANS) == pytest.approx(0.0)


def test_trump_honours_count_more_than_side_honours():
    hand = parse_hand("AS KS QS 2S 3H 4H 5H 6D 7D 8D 9C TC JC")
    assert expected_tricks(hand, SPADES) == pytest.approx(1.0 + 0.9 + 0.6 + 1.0)
    assert expected_tricks(hand, SANS) == pytest.approx(1.0 + 0.75 + 0.4)


def test_a_void_side_suit_adds_a_ruff_when_holding_trumps():
    hand = parse_hand("2S 3S 4S 5H 6H 7H 8H 2D 3D 4D 5D 6D 7D")
    assert expected_tricks(hand, SPADES) == pytest.approx(0.8)


def test_a_singleton_side_suit_adds_half_a_ruff():
    hand = parse_hand("2S 3S 4S 5H 6H 7H 8H 2D 3D 4D 5D 6D 2C")
    assert expected_tricks(hand, SPADES) == pytest.approx(0.4)


def test_trump_tricks_never_exceed_the_number_of_trumps():
    hand = parse_hand("AS 2H 3H 4H 5H 6H 7H 2D 3D 4D 5D 6D 7D")
    assert expected_tricks(hand, SPADES) == pytest.approx(1.0)


def test_sans_rewards_long_suits_headed_by_an_ace_or_king():
    hand = parse_hand("AH KH 2H 3H 4H 5H 6H 2S 3S 4D 5D 2C 3C")
    assert expected_tricks(hand, SANS) == pytest.approx(1.0 + 0.75 + 3 * 0.5)
    headless = parse_hand("QH JH 2H 3H 4H 5H 6H 2S 3S 4D 5D 2C 3C")
    assert expected_tricks(headless, SANS) == pytest.approx(0.4)


def test_trump_values_lists_every_trump_in_index_order():
    hand = parse_hand("AS KS QS 2S 3H 4H 5H 6D 7D 8D 9C TC JC")
    trumps = (CLUBS, DIAMONDS, HEARTS, SPADES, SANS)
    assert trump_values(hand) == tuple(expected_tricks(hand, t) for t in trumps)


def test_hand_values_are_configurable():
    hand = parse_hand("AS AH AD AC 2S 3S 4H 5H 6D 7D 2C 3C 4C")
    assert expected_tricks(hand, SANS, HandValues(ace=0.5)) == pytest.approx(2.0)
    assert trump_values(hand, HandValues(ace=0.5))[SANS] == pytest.approx(2.0)
