"""Exact play of a round's last tricks (docs/rl/hybrid-iteration-plan.md, step 7).

The search finishes an imagined deal roughly: every seat plays the player's top card to the end.
With a few cards left in each hand there are few enough ways to finish that all of them can be
played. ``solve`` does that for a batch of imagined deals at once, all four hands open:

- **Every line of play.** From the card to be played now, every legal card of every seat, to the
  round's last trick (PLY-2 and PLY-3, as ``estimation.engine.cards`` has them).
- **Each seat plays its best for itself.** What a finished round is worth to a seat is given by
  ``worths`` (``worths_of``: its own round score less the others', as the search counts it). At
  its turn a seat takes the card whose finish is worth the most to it.
- **Ties.** A seat that is as well off with several cards plays each of them equally often: the
  worth of its turn is the mean over those cards. This is a choice of the solver, not a rule of
  the game: a seat with nothing left to play for has to play something.

The answer is the worth of the round to every seat after each legal card of the seat to play,
deal by deal. It is exact for hands that are open. The table has them hidden, so a card is judged
by its mean over many imagined deals, never by the one real deal.

The cost is the number of lines, at most ``(k!)^4`` with ``k`` cards in each hand: 1,296 at 3
cards and 331,776 at 4, a deal. 5 cards (207 million) are out of reach this way.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from estimation.bots.scoring import outcome_of, standings_weights
from estimation.engine import Observation, RulesConfig, score_round
from estimation.engine.cards import NUM_SEATS, SANS, cards_of, trick_winner
from estimation.engine.config import DEFAULT_CONFIG

Floats = npt.NDArray[np.float64]
Tricks = tuple[int, int, int, int]
Worths = Callable[[Tricks], Sequence[float]]

MAX_CARDS = 4  # the most cards in a hand that every line is played from
TIE = 1e-9  # worths closer than this are the same to a seat
_PLAIN = 1.0 / 3.0


class Solved(NamedTuple):
    """``solve``'s answer: the legal ``cards`` of the seat to play, lowest first, and
    ``worths[deal, card, seat]``, what the round is worth to each seat after each of them."""

    cards: list[int]
    worths: Floats


def worths_of(
    obs: Observation, rules: RulesConfig = DEFAULT_CONFIG, leader_weight: bool = True
) -> Worths:
    """What the round in ``obs`` is worth to each seat once it ends with given trick counts: the
    seat's round score less the other three's, the players ahead of it in the game counting
    double (``standings_weights``) or, without ``leader_weight``, their plain mean. For
    ``obs.seat`` it is the value ``RoundScorer`` gives the search."""
    outcome = outcome_of(obs)
    weights = [
        standings_weights(obs.scores, seat)
        if leader_weight
        else tuple(0.0 if other == seat else _PLAIN for other in range(NUM_SEATS))
        for seat in range(NUM_SEATS)
    ]

    def worths(tricks: Tricks) -> tuple[float, ...]:
        scores = score_round(dataclasses.replace(outcome, tricks=tricks), rules).scores
        return tuple(
            scores[seat] - sum(w * s for w, s in zip(weights[seat], scores, strict=True))
            for seat in range(NUM_SEATS)
        )

    return worths


def _set_bits(size: int) -> npt.NDArray[np.intp]:
    """``[1 << size, size]``: the positions of a mask's set bits, lowest first (0 past them)."""
    table = np.zeros((1 << size, size), np.intp)
    for mask in range(1 << size):
        bits = [bit for bit in range(size) if mask >> bit & 1]
        table[mask, : len(bits)] = bits
    return table


def _plies(played: int, size: int) -> list[tuple[int, int]]:
    """Every card still to be played, as its place in its trick and the cards its seat holds:
    the rest of the trick under way, then the whole tricks after it."""
    plies = [(place, size) for place in range(played, NUM_SEATS)]
    for left in range(size - 1, 0, -1):
        plies.extend((place, left) for place in range(NUM_SEATS))
    return plies


def _check(
    deals: npt.NDArray[np.uint64], trick: Sequence[tuple[int, int]], leader: int, trump: int
) -> int:
    """The cards the seat to play holds, after checking that ``solve`` can play the deals."""
    if deals.ndim != 2 or deals.shape[1] != NUM_SEATS or not len(deals):
        raise ValueError("deals are rows of four hands, at least one row")
    if not 0 <= trump <= SANS or not 0 <= leader < NUM_SEATS or len(trick) >= NUM_SEATS:
        raise ValueError("solve needs a trump, a leader and a trick with a card still to play")
    if any(seat != (leader + place) % NUM_SEATS for place, (seat, _) in enumerate(trick)):
        raise ValueError("the trick's cards are not in play order from its leader")
    counts = np.array([[int(hand).bit_count() for hand in deal] for deal in deals.tolist()])
    size = int(counts[0, (leader + len(trick)) % NUM_SEATS])
    if not 1 <= size <= MAX_CARDS:
        raise ValueError(f"solve plays hands of 1 to {MAX_CARDS} cards, not {size}")
    played = {seat for seat, _ in trick}
    wanted = [size - 1 if seat in played else size for seat in range(NUM_SEATS)]
    if (counts != wanted).any():
        raise ValueError(f"every deal must hold {wanted} cards by seat")
    table = 0
    for _, card in trick:
        table |= 1 << card
    for deal in deals.tolist():
        held = deal[0] | deal[1] | deal[2] | deal[3]
        if held.bit_count() != sum(wanted) or held & table:
            raise ValueError("a card is in two hands, or in a hand and on the table")
    return size


def solve(
    deals: Sequence[Sequence[int]] | npt.NDArray[np.uint64],
    trick: Sequence[tuple[int, int]],
    leader: int,
    won: Sequence[int],
    trump: int,
    worths: Worths,
    chunk: int = 8,
) -> Solved:
    """Play ``deals`` to the end of the round every legal way (see the module). A deal is the
    four hands as they are now, by seat; ``trick`` the ``(seat, card)`` pairs on the table, led
    by ``leader``; ``won`` the tricks each seat has taken; ``trump`` a suit or ``SANS``. The seat
    to play holds the same cards in every deal. ``chunk`` deals are played at a time, which
    bounds the memory: 4 cards in each hand take about 15 MB a deal."""
    hands = np.asarray(deals, dtype=np.uint64)
    size = _check(hands, trick, leader, trump)
    mover = (leader + len(trick)) % NUM_SEATS
    if len(np.unique(hands[:, mover])) != 1:
        raise ValueError("the seat to play must hold the same cards in every deal")
    base = size + 1  # a seat takes 0 to ``size`` of the tricks left
    table = np.full((base**NUM_SEATS, NUM_SEATS), np.nan)
    for index in range(base**NUM_SEATS):
        more = [index // base**seat % base for seat in range(NUM_SEATS)]
        if sum(more) == size:
            final = (won[0] + more[0], won[1] + more[1], won[2] + more[2], won[3] + more[3])
            table[index] = worths(final)
    parts = [
        _solve(hands[at : at + chunk], trick, leader, trump, size, table)
        for at in range(0, len(hands), chunk)
    ]
    legal = parts[0][0]
    cards = cards_of(int(hands[0, mover]))
    return Solved(
        [card for card, ok in zip(cards, legal, strict=True) if ok],
        np.concatenate([values for _, values in parts])[:, legal],
    )


def _solve(
    hands: npt.NDArray[np.uint64],
    trick: Sequence[tuple[int, int]],
    leader: int,
    trump: int,
    size: int,
    table: Floats,
) -> tuple[npt.NDArray[np.bool_], Floats]:
    """One chunk of ``solve``: which of the cards of the seat to play are legal, and
    ``[deal, card, seat]`` worths for all of them."""
    count = len(hands)
    held = np.zeros((count, NUM_SEATS, size), np.int8)  # each seat's cards, lowest first
    in_suit = np.zeros((count, NUM_SEATS, 4), np.uint8)  # their positions by suit, as a mask
    masks = np.zeros((count, 1, NUM_SEATS), np.uint8)  # the positions each seat still holds
    for deal, four in enumerate(hands.tolist()):
        for seat, hand in enumerate(four):
            for position, card in enumerate(cards_of(hand)):
                held[deal, seat, position] = card
                in_suit[deal, seat, card // 13] |= 1 << position
                masks[deal, 0, seat] |= 1 << position
    set_bits = _set_bits(size)
    bit_of = (1 << np.arange(size)).astype(np.uint8)
    power = ((size + 1) ** np.arange(NUM_SEATS)).astype(np.int16)
    rows = np.arange(count)[:, None]

    lead = np.full((count, 1), leader, np.int8)
    more = np.zeros((count, 1), np.int16)  # the tricks taken from here on, as an index
    led = np.full((count, 1), trick[0][1] // 13 if trick else 0, np.int8)
    top_seat = np.full((count, 1), trick_winner(trick, trump) if trick else 0, np.int8)
    top = np.full((count, 1), dict(trick)[int(top_seat[0, 0])] if trick else 0, np.int8)

    choices: list[tuple[npt.NDArray[np.int8], npt.NDArray[np.bool_] | None, int]] = []
    for place, cards in _plies(len(trick), size):
        if cards == 1 and place == 0 and choices:
            # The last trick has no choice in it, and most of the lines end in it: only its
            # winner is worked out, from each seat's one card.
            last = held[rows[..., None], np.arange(NUM_SEATS), set_bits[masks][..., 0]]
            for turn in range(NUM_SEATS):
                seat = (lead + turn) & 3
                card = np.take_along_axis(last, seat[..., None].astype(np.intp), 2)[..., 0]
                if turn == 0:
                    top, top_seat = card, seat
                    continue
                suit, top_suit = card // 13, top // 13
                beats = ((suit == top_suit) & (card > top)) | (
                    (suit == trump) & (top_suit != trump)
                )
                top = np.where(beats, card, top)
                top_seat = np.where(beats, seat, top_seat)
            more = more + power[top_seat]
            break
        seat = (lead + place) & 3  # [deal, node]
        mine = np.take_along_axis(masks, seat[..., None].astype(np.intp), 2)[..., 0]
        position = set_bits[mine][..., :cards]  # [deal, node, card]
        card = held[rows[..., None], seat[..., None], position]
        suit = card // 13
        if place == 0:
            legal = None
            led_next, top_next = suit, card
            top_seat_next = np.broadcast_to(seat[..., None], card.shape)
        else:
            follow = mine & in_suit[rows, seat, led]
            legal = (follow == 0)[..., None] | (suit == led[..., None])
            top_suit = (top // 13)[..., None]
            beats = ((suit == top_suit) & (card > top[..., None])) | (
                (suit == trump) & (top_suit != trump)
            )
            led_next = np.broadcast_to(led[..., None], card.shape)
            top_next = np.where(beats, card, top[..., None])
            top_seat_next = np.where(beats, seat[..., None], top_seat[..., None])
        masks_next = np.repeat(masks[:, :, None, :], cards, axis=2)
        at = np.broadcast_to(seat[..., None, None], (*card.shape, 1)).astype(np.intp)
        np.put_along_axis(masks_next, at, (mine[..., None] ^ bit_of[position])[..., None], 3)
        if place == NUM_SEATS - 1:
            lead_next = top_seat_next
            more_next = more[..., None] + power[top_seat_next]
        else:
            lead_next = np.broadcast_to(lead[..., None], card.shape)
            more_next = np.broadcast_to(more[..., None], card.shape)
        if cards > 1 or not choices:
            choices.append((seat, legal, cards))
        masks = masks_next.reshape(count, -1, NUM_SEATS)
        lead = lead_next.reshape(count, -1).astype(np.int8)
        more = more_next.reshape(count, -1).astype(np.int16)
        led = led_next.reshape(count, -1).astype(np.int8)
        top = top_next.reshape(count, -1).astype(np.int8)
        top_seat = top_seat_next.reshape(count, -1).astype(np.int8)

    values = table[more]  # [deal, line, seat]
    for seat, legal, cards in reversed(choices[1:]):
        values = values.reshape(count, -1, cards, NUM_SEATS)
        own = np.take_along_axis(values, seat[..., None, None].astype(np.intp), 3)[..., 0]
        if legal is not None:
            own = np.where(legal, own, -np.inf)
        tied = own >= own.max(2, keepdims=True) - TIE
        values = (values * tied[..., None]).sum(2) / tied.sum(2)[..., None]
    _, legal, cards = choices[0]
    first = np.ones(cards, bool) if legal is None else legal[0, 0]
    return first, values.reshape(count, cards, NUM_SEATS)
