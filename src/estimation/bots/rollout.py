"""Playing out a round on a known deal with one card rule for every seat (SPEC §6 M3).

PIMC calls this hundreds of times per decision, so it works on bitmasks and plain lists rather
than the engine's Round. With HeuristicBot's card play (the default rule) it must play exactly
like four HeuristicBots in the engine; the tests check that against the engine itself.
"""

from collections.abc import Callable, Sequence

from .card_play import choose_card

type CardRule = Callable[[int, Sequence[int], int, bool, int], int]
"""A card choice like ``card_play.choose_card``: ``(hand, trick, trump, want, out) -> card``."""


def trick_winner_offset(c0: int, c1: int, c2: int, c3: int, trump: int) -> int:
    """Position (0-3, in play order) of the card that wins a full trick (PLY-3)."""
    best = c0
    best_suit = c0 // 13
    offset = 0
    suit = c1 // 13
    if suit == best_suit:
        if c1 > best:
            best, offset = c1, 1
    elif suit == trump:
        best, best_suit, offset = c1, suit, 1
    suit = c2 // 13
    if suit == best_suit:
        if c2 > best:
            best, offset = c2, 2
    elif suit == trump:
        best, best_suit, offset = c2, suit, 2
    suit = c3 // 13
    if (suit == best_suit and c3 > best) or (suit != best_suit and suit == trump):
        offset = 3
    return offset


def finish_round(
    hands: list[int],
    trick: list[int],
    leader: int,
    trump: int,
    estimates: Sequence[int],
    won: list[int],
    choose: CardRule = choose_card,
) -> list[int]:
    """Play the round to its end, every seat choosing its cards by ``choose``, and return the
    tricks each seat won.

    ``hands`` are every seat's remaining cards, ``trick`` the cards of the trick in progress
    (leader first, possibly complete) and ``won`` the tricks won so far. ``hands``, ``trick`` and
    ``won`` are used up: pass copies.
    """
    unplayed = hands[0] | hands[1] | hands[2] | hands[3]
    while True:
        if len(trick) == 4:
            leader = (
                leader + trick_winner_offset(trick[0], trick[1], trick[2], trick[3], trump)
            ) & 3
            won[leader] += 1
            if not unplayed:
                return won
            trick = []
        seat = (leader + len(trick)) & 3
        hand = hands[seat]
        card = choose(hand, trick, trump, estimates[seat] > won[seat], unplayed & ~hand)
        bit = 1 << card
        hands[seat] = hand ^ bit
        unplayed ^= bit
        trick.append(card)
