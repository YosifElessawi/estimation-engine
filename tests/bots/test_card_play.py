"""HeuristicBot's card play (SPEC §6 M2): win tricks until the estimate is reached, then duck."""

import random

from estimation.bots.card_play import choose_card
from estimation.engine import SANS, SPADES, legal_plays, mask_of, parse_card, parse_hand
from estimation.engine.cards import FULL_DECK


def cards(text: str) -> list[int]:
    return [parse_card(word) for word in text.split()]


def pick(hand_text: str, trick_text: str, trump: int, want: bool, played_text: str = "") -> str:
    """The card chosen by a player holding ``hand_text`` with ``trick_text`` on the table."""
    hand = parse_hand(hand_text)
    trick = cards(trick_text)
    gone = mask_of(trick) | parse_hand(played_text)
    card = choose_card(hand, trick, trump, want, FULL_DECK & ~hand & ~gone)
    return f"{'23456789TJQKA'[card % 13]}{'CDHS'[card // 13]}"


# Following suit while it still needs tricks


def test_plays_the_cheapest_sure_winner_when_it_needs_tricks():
    assert pick("AH KH 2H 5C", "9H", SPADES, want=True) == "KH"


def test_plays_its_highest_winner_when_no_winner_is_sure():
    assert pick("KH QH 2H 5C", "9H", SPADES, want=True) == "KH"


def test_last_to_play_wins_as_cheaply_as_it_can():
    assert pick("AH KH 2H", "9H TH JH", SPADES, want=True) == "KH"


def test_plays_its_lowest_card_when_it_cannot_win_the_trick():
    assert pick("2H 3H 5C", "AH", SPADES, want=True) == "2H"


def test_cannot_win_a_ruffed_trick_by_following_suit():
    assert pick("AH 2H", "KH 3S", SPADES, want=True) == "2H"


# Following suit once its estimate is reached


def test_ducks_with_its_highest_losing_card():
    assert pick("QH 9H 4H", "JH", SPADES, want=False) == "9H"


def test_plays_low_when_every_card_wins_and_others_play_after_it():
    assert pick("QH KH", "JH", SPADES, want=False) == "QH"


def test_last_to_play_unloads_its_highest_card_when_every_card_wins():
    assert pick("QH KH", "2H 3H 4H", SPADES, want=False) == "KH"


# Void in the led suit


def test_ruffs_with_its_lowest_trump_when_it_needs_tricks():
    assert pick("2S 7S AC", "KH", SPADES, want=True) == "2S"


def test_overruffs_with_the_lowest_trump_that_beats_the_trick():
    assert pick("2S 7S AC", "KH 5S", SPADES, want=True) == "7S"


def test_ruffs_with_a_sure_trump_when_it_has_one():
    assert pick("2S AS 4C", "KH", SPADES, want=True, played_text="KS QS JS TS 9S 8S 7S 6S") == "AS"


def test_discards_its_lowest_side_card_when_it_cannot_win():
    assert pick("AC 3D 5S", "KH", SANS, want=True) == "3D"


def test_keeps_its_trumps_when_it_needs_tricks_but_cannot_overruff():
    assert pick("2S 9C 4D", "KH AS", SPADES, want=True) == "4D"


def test_dumps_its_highest_side_card_when_ducking():
    assert pick("AC 3D 5S", "KH", SPADES, want=False) == "AC"


def test_never_ruffs_when_ducking_if_it_can_discard():
    assert pick("2S 3D", "KH", SPADES, want=False) == "3D"


# Leading


def test_leads_a_sure_winner_when_it_needs_tricks():
    assert pick("AC 4D 7H", "", SPADES, want=True) == "AC"


def test_leads_a_sure_trump_when_it_has_no_side_winner():
    assert pick("AS 4D 7H", "", SPADES, want=True) == "AS"


def test_leads_low_from_its_longest_side_suit_without_a_sure_winner():
    assert pick("KD 4D 3D 7H 2S", "", SPADES, want=True) == "3D"


def test_leads_its_lowest_beatable_card_when_ducking():
    assert pick("AC 4D 7H", "", SPADES, want=False) == "4D"


def test_does_not_lead_a_card_nobody_can_beat_when_ducking():
    played = "3C 4C 5C 6C 7C 8C 9C TC JC QC KC AC"
    assert pick("2C 9D", "", SPADES, want=False, played_text=played) == "9D"


def test_the_chosen_card_is_always_legal():
    rng = random.Random(7)
    for _ in range(3000):
        deck = list(range(52))
        rng.shuffle(deck)
        size = rng.randint(1, 13)
        hand = mask_of(deck[:size])
        trick = deck[size : size + rng.randint(0, 3)]
        out = mask_of(deck[size + len(trick) : size + len(trick) + rng.randint(0, 39)])
        trump = rng.randint(0, SANS)
        card = choose_card(hand, trick, trump, rng.random() < 0.5, out)
        seats = [(i, c) for i, c in enumerate(trick)]
        assert card in legal_plays(hand, seats)
