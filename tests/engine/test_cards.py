"""Cards, the deal and the trick rules: TBL-1 to TBL-4, PLY-2, PLY-3 and T1 to T4."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from estimation.engine.cards import (
    CLUBS,
    DIAMONDS,
    FULL_DECK,
    HEARTS,
    SANS,
    SPADES,
    card_id,
    card_str,
    cards_of,
    deal,
    legal_plays,
    mask_of,
    next_seat,
    parse_card,
    parse_hand,
    rank_of,
    suit_of,
    trick_winner,
)


def c(text: str) -> int:
    return parse_card(text)


# Encodings (§5.1)


def test_card_ids_follow_suit_times_13_plus_rank():
    assert c("2C") == 0
    assert c("AC") == 12
    assert c("2D") == 13
    assert c("TS") == 3 * 13 + 8
    assert c("JH") == 2 * 13 + 9
    assert c("AS") == 51
    assert card_id(12, SPADES) == 51
    assert suit_of(c("JH")) == HEARTS
    assert rank_of(c("JH")) == 9


def test_card_text_round_trips_for_every_card():
    for card in range(52):
        assert parse_card(card_str(card)) == card
    assert card_str(c("TS")) == "TS"


@pytest.mark.parametrize("text", ["", "1C", "10S", "AX", "ts", "2CC", "Z"])
def test_parse_card_rejects_bad_text(text: str):
    with pytest.raises(ValueError, match="card"):
        parse_card(text)


def test_parse_hand_and_mask_helpers_agree():
    hand = parse_hand("5D 8D 2C TS AS 6H KH")
    assert hand == mask_of([c("5D"), c("8D"), c("2C"), c("TS"), c("AS"), c("6H"), c("KH")])
    assert cards_of(hand) == sorted(cards_of(hand))
    assert len(cards_of(hand)) == 7


def test_parse_hand_rejects_duplicates():
    with pytest.raises(ValueError, match="twice"):
        parse_hand("AS AS")


# Table and cards


def test_TBL_1_four_seats_four_hands():
    assert len(deal(0, 1)) == 4


def test_TBL_2_next_seat_is_counterclockwise():
    assert [next_seat(s) for s in range(4)] == [1, 2, 3, 0]


def test_TBL_3_deal_gives_13_cards_to_each_seat_from_one_deck():
    hands = deal(0, 1)
    assert all(h.bit_count() == 13 for h in hands)
    assert hands[0] | hands[1] | hands[2] | hands[3] == FULL_DECK
    assert sum(h.bit_count() for h in hands) == 52


@given(seed=st.integers(0, 2**63), round_no=st.integers(1, 80))
def test_TBL_3_every_deal_is_a_partition_of_the_deck(seed: int, round_no: int):
    hands = deal(seed, round_no)
    assert all(h.bit_count() == 13 for h in hands)
    assert hands[0] | hands[1] | hands[2] | hands[3] == FULL_DECK


def test_deal_uses_the_specified_numpy_permutation():
    perm = np.random.default_rng([7, 3]).permutation(52).tolist()
    hands = deal(7, 3)
    for seat in range(4):
        assert hands[seat] == mask_of(perm[13 * seat : 13 * seat + 13])


def test_deal_depends_only_on_seed_and_round_number():
    assert deal(5, 2) == deal(5, 2)
    assert deal(5, 2) != deal(5, 3)
    assert deal(5, 2) != deal(6, 2)


def test_TBL_4_rank_order_is_ace_high_two_low():
    order = [c(f"{r}H") for r in "23456789TJQKA"]
    assert order == sorted(order)
    trick = [(0, c("KH")), (1, c("AH")), (2, c("2H")), (3, c("QH"))]
    assert trick_winner(trick, CLUBS) == 1


# Trick play


def test_T1_must_follow_suit_page_example():
    trick = [(0, c("JH"))]
    hand = parse_hand("5D 8D 2C TS AS 6H KH")
    assert legal_plays(hand, trick) == [c("6H"), c("KH")]
    void = parse_hand("5D 8D 2C TS AS")
    assert legal_plays(void, trick) == cards_of(void)


def test_PLY_2_leader_may_play_any_card():
    hand = parse_hand("5D 8D 2C TS AS 6H KH")
    assert legal_plays(hand, []) == cards_of(hand)


def test_PLY_2_follower_must_follow_even_with_trump_in_hand():
    trick = [(2, c("3D"))]
    hand = parse_hand("AS KS 9D")
    assert legal_plays(hand, trick) == [c("9D")]


def test_T2_trump_wins():
    trick = [(0, c("JH")), (1, c("KH")), (2, c("2S")), (3, c("AH"))]
    assert trick_winner(trick, SPADES) == 2


def test_T3_off_suit_card_has_no_value():
    trick = [(0, c("JH")), (1, c("KH")), (2, c("AD")), (3, c("3H"))]
    assert trick_winner(trick, SPADES) == 1


def test_T4_sans_has_no_trump():
    trick = [(0, c("TC")), (1, c("AS")), (2, c("JC")), (3, c("2C"))]
    assert trick_winner(trick, SANS) == 2


def test_PLY_3_highest_trump_wins_when_several_are_played():
    trick = [(1, c("AD")), (2, c("3C")), (3, c("9C")), (0, c("KD"))]
    assert trick_winner(trick, CLUBS) == 3


def test_PLY_3_leading_trump_wins_against_higher_off_suit():
    trick = [(3, c("2H")), (0, c("AS")), (1, c("AD")), (2, c("AC"))]
    assert trick_winner(trick, HEARTS) == 3


def test_PLY_3_highest_card_of_led_suit_wins_without_trump():
    trick = [(0, c("4D")), (1, c("9D")), (2, c("AS")), (3, c("8D"))]
    assert trick_winner(trick, CLUBS) == 1
    assert trick_winner(trick, DIAMONDS) == 1


@given(
    cards=st.lists(st.integers(0, 51), min_size=4, max_size=4, unique=True),
    leader=st.integers(0, 3),
    trump=st.integers(0, 4),
)
def test_PLY_3_winner_holds_highest_trump_or_highest_of_led_suit(
    cards: list[int], leader: int, trump: int
):
    trick = [((leader + i) % 4, card) for i, card in enumerate(cards)]
    winner = trick_winner(trick, trump)
    win_card = dict(trick)[winner]
    trumps = [card for card in cards if suit_of(card) == trump]
    led = suit_of(cards[0])
    if trumps:
        assert win_card == max(trumps)
    else:
        assert win_card == max(card for card in cards if suit_of(card) == led)


def test_trick_winner_rejects_an_empty_trick():
    with pytest.raises(ValueError, match="empty"):
        trick_winner([], SANS)
