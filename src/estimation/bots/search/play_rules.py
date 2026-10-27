"""The Search bot's play-out rules: the card each seat plays in the games it imagines.

Every seat of every imagined game plays by these rules: while a player still needs tricks it
tries to win them; once its estimate is reached it ducks. They started (2026-10-27) as an exact
copy of HeuristicBot's card play (``bots/card_play``) and are the Search bot's to change:
HeuristicBot's own copy stays frozen, since it anchors the Elo ladder and ``pimc`` rolls out with
it. Card ids within a suit rise with rank, so the lowest set bit of a one-suit mask is its lowest
card.
"""

from collections.abc import Sequence

from estimation.engine.cards import SANS, SUIT_MASKS

_ABOVE = tuple(SUIT_MASKS[card // 13] & ~((2 << card) - 1) for card in range(52))
"""For each card, the cards of its suit that outrank it."""


def choose_card(hand: int, trick: Sequence[int], trump: int, want: bool, out: int) -> int:
    """The card to play.

    ``hand`` is the player's remaining cards and ``trick`` the cards already played to this trick,
    leader first. ``trump`` is a suit or SANS. ``want`` is true while the player still needs
    tricks to reach its estimate. ``out`` is every unplayed card outside the hand: the cards that
    could still beat this player's cards.
    """
    if not trick:
        return _lead(hand, trump, want, out)
    best = trick[0]
    led = best // 13
    best_suit = led
    for i in range(1, len(trick)):
        card = trick[i]
        suit = card // 13
        if suit == best_suit:
            if card > best:
                best = card
        elif suit == trump:
            best = card
            best_suit = suit
    follow = hand & SUIT_MASKS[led]
    if follow:
        legal = follow
        beat = follow & _ABOVE[best] if best_suit == led else 0
    else:
        legal = hand
        if trump == SANS:
            beat = 0
        elif best_suit == trump:
            beat = hand & _ABOVE[best]
        else:
            beat = hand & SUIT_MASKS[trump]
    last = len(trick) == 3
    if want:
        if beat:
            low = (beat & -beat).bit_length() - 1
            if last:
                return low
            rivals = out & SUIT_MASKS[low // 13]
            safe = beat & _ABOVE[rivals.bit_length() - 1] if rivals else beat
            if safe:
                return (safe & -safe).bit_length() - 1  # the cheapest card nobody can top
            return beat.bit_length() - 1 if follow else low
        if follow:
            return (follow & -follow).bit_length() - 1
        return _discard_low(hand, trump)
    lose = legal & ~beat
    if lose:
        if follow:
            return lose.bit_length() - 1  # the highest card that still loses
        return _discard_high(lose, trump)
    if last:
        return beat.bit_length() - 1
    return (beat & -beat).bit_length() - 1


def _discard_low(cards: int, trump: int) -> int:
    """The lowest-ranked side card, from the shorter suit on a tie; trumps only if nothing else."""
    best = -1
    best_rank = 13
    best_length = 14
    for suit in range(4):
        if suit == trump:
            continue
        held = cards & SUIT_MASKS[suit]
        if held:
            low = (held & -held).bit_length() - 1
            rank = low % 13
            length = held.bit_count()
            if rank < best_rank or (rank == best_rank and length < best_length):
                best, best_rank, best_length = low, rank, length
    if best < 0:
        return (cards & -cards).bit_length() - 1
    return best


def _discard_high(cards: int, trump: int) -> int:
    """The highest-ranked side card, from the shorter suit on a tie; trumps only if nothing else."""
    best = -1
    best_rank = -1
    best_length = 14
    for suit in range(4):
        if suit == trump:
            continue
        held = cards & SUIT_MASKS[suit]
        if held:
            high = held.bit_length() - 1
            rank = high % 13
            length = held.bit_count()
            if rank > best_rank or (rank == best_rank and length < best_length):
                best, best_rank, best_length = high, rank, length
    if best < 0:
        return cards.bit_length() - 1
    return best


def _lead(hand: int, trump: int, want: bool, out: int) -> int:
    if want:
        return _lead_to_win(hand, trump, out)
    return _lead_to_lose(hand, trump, out)


def _lead_to_win(hand: int, trump: int, out: int) -> int:
    """Cash a side-suit card nobody can top, else a sure trump, else lead low from length."""
    best = -1
    best_rank = -1
    best_length = -1
    longest = 0
    longest_length = 0
    for suit in range(4):
        if suit == trump:
            continue
        held = hand & SUIT_MASKS[suit]
        if not held:
            continue
        top = held.bit_length() - 1
        length = held.bit_count()
        if not out & _ABOVE[top]:
            rank = top % 13
            if rank > best_rank or (rank == best_rank and length > best_length):
                best, best_rank, best_length = top, rank, length
        if length > longest_length:
            longest, longest_length = held, length
    if best >= 0:
        return best
    if trump != SANS:
        trumps = hand & SUIT_MASKS[trump]
        if trumps:
            top = trumps.bit_length() - 1
            if not out & _ABOVE[top]:
                return top
    if longest:
        return (longest & -longest).bit_length() - 1
    return (hand & -hand).bit_length() - 1


def _lead_to_lose(hand: int, trump: int, out: int) -> int:
    """Lead the lowest-ranked card that someone can still beat: a side card before a trump, and
    the one with more cards out above it on a tie. If every such card is unbeatable, lead the
    lowest-ranked card."""
    best = -1
    best_key = (14, 2, 0)
    lowest = -1
    lowest_rank = 13
    for suit in range(4):
        held = hand & SUIT_MASKS[suit]
        if not held:
            continue
        low = (held & -held).bit_length() - 1
        rank = low % 13
        if rank < lowest_rank:
            lowest, lowest_rank = low, rank
        beaters = out & _ABOVE[low]
        if beaters:
            key = (rank, 1 if suit == trump else 0, -beaters.bit_count())
            if key < best_key:
                best, best_key = low, key
    return best if best >= 0 else lowest
