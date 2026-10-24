"""Cards, hands, the deal and the trick rules (SPEC §5.1, TBL-1 to TBL-5, PLY-2, PLY-3).

A card id is ``suit * 13 + rank``. Within one suit a higher id is a higher card, which the
trick rules rely on. Hands are 52-bit integer bitmasks: bit ``i`` set means card id ``i``.
"""

from collections.abc import Iterable, Sequence

import numpy as np

NUM_SEATS = 4
NUM_CARDS = 52
HAND_SIZE = 13

CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3
SANS = 4
"""Trump index for no trump. A higher trump index is a stronger trump (TBL-5)."""

SUITS = (CLUBS, DIAMONDS, HEARTS, SPADES)
TRUMPS = (CLUBS, DIAMONDS, HEARTS, SPADES, SANS)
TRUMP_NAMES = ("Clubs", "Diamonds", "Hearts", "Spades", "Sans")
SUIT_LETTERS = "CDHS"
RANK_LETTERS = "23456789TJQKA"

FULL_DECK = (1 << NUM_CARDS) - 1
SUIT_MASKS = tuple(0x1FFF << (13 * suit) for suit in SUITS)

_BITS = tuple(1 << card for card in range(NUM_CARDS))


def next_seat(seat: int) -> int:
    """The seat after ``seat``: play goes counterclockwise (TBL-2)."""
    return (seat + 1) % NUM_SEATS


def card_id(rank: int, suit: int) -> int:
    """Card id from a rank (2 is 0, A is 12) and a suit (Clubs 0 to Spades 3)."""
    if not (0 <= rank < 13 and 0 <= suit < 4):
        raise ValueError(f"no card with rank {rank} and suit {suit}")
    return suit * 13 + rank


def suit_of(card: int) -> int:
    return card // 13


def rank_of(card: int) -> int:
    return card % 13


def card_str(card: int) -> str:
    """Text form: rank then suit letter, with T for ten, for example ``TS`` or ``2C``."""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"no card with id {card}")
    return RANK_LETTERS[card % 13] + SUIT_LETTERS[card // 13]


def parse_card(text: str) -> int:
    """Card id from its text form, for example ``"JH"``."""
    if len(text) != 2 or text[0] not in RANK_LETTERS or text[1] not in SUIT_LETTERS:
        raise ValueError(f"not a card: {text!r} (expected rank then suit, like 'TS')")
    return SUIT_LETTERS.index(text[1]) * 13 + RANK_LETTERS.index(text[0])


def mask_of(cards: Iterable[int]) -> int:
    """Bitmask holding the given card ids."""
    mask = 0
    for card in cards:
        mask |= _BITS[card]
    return mask


def cards_of(mask: int) -> list[int]:
    """Card ids in a bitmask, lowest first."""
    cards: list[int] = []
    while mask:
        low = mask & -mask
        cards.append(low.bit_length() - 1)
        mask ^= low
    return cards


def parse_hand(text: str) -> int:
    """Bitmask from space-separated card texts, for example ``"5D 8D 2C TS"``."""
    mask = 0
    for word in text.split():
        bit = _BITS[parse_card(word)]
        if mask & bit:
            raise ValueError(f"card {word} appears twice")
        mask |= bit
    return mask


def deal(seed: int, round_no: int) -> tuple[int, int, int, int]:
    """The four hands for a round (SPEC §5.1).

    Uses ``numpy.random.default_rng([seed, round_no]).permutation(52)``; seat ``i`` gets
    positions ``13*i`` to ``13*i + 12``. The deal depends only on the seed and round number.
    """
    perm: list[int] = np.random.default_rng([seed, round_no]).permutation(NUM_CARDS).tolist()
    bit = _BITS.__getitem__
    return (
        sum(map(bit, perm[0:13])),
        sum(map(bit, perm[13:26])),
        sum(map(bit, perm[26:39])),
        sum(map(bit, perm[39:52])),
    )


def legal_plays(hand: int, trick: Sequence[tuple[int, int]]) -> list[int]:
    """Cards ``hand`` may play to ``trick``, the ``(seat, card)`` pairs played so far (PLY-2).

    A player holding a card of the led suit must play one; otherwise any card is legal.
    """
    if not trick:
        return cards_of(hand)
    follow = hand & SUIT_MASKS[trick[0][1] // 13]
    return cards_of(follow if follow else hand)


def trick_winner(trick: Sequence[tuple[int, int]], trump: int) -> int:
    """Seat winning ``trick`` so far, given as ``(seat, card)`` pairs in play order (PLY-3).

    The highest trump wins if any was played, otherwise the highest card of the led suit.
    An off-suit card that is not trump never wins. ``trump`` is a suit, or ``SANS``.
    """
    if not trick:
        raise ValueError("cannot pick the winner of an empty trick")
    if not 0 <= trump <= SANS:
        raise ValueError(f"no trump with index {trump}")
    best_seat, best = trick[0]
    best_suit = best // 13
    for seat, card in trick[1:]:
        suit = card // 13
        if suit == best_suit:
            if card > best:
                best_seat, best = seat, card
        elif suit == trump:
            best_seat, best, best_suit = seat, card, suit
    return best_seat
