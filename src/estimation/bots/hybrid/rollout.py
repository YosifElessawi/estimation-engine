"""Playing out many deals at once with CardPlayNet for every seat (SPEC §6 M5).

The hybrid scores a candidate card on a sampled deal by playing the rest of the round with the RL
bot's greedy card play for all four seats, as four RL bots would in the engine. Every game of one
search starts from the same position (the same trick and the same number of cards left), so the
games move in lockstep: each step plays one card in every game, with one batched CardPlayNet pass
for the games that have more than one legal card.

Each seat's features are the state encoder's (``rl.encoder``), built incrementally in NumPy: a
per-seat template holds everything that stays fixed during card play (the auction, stakes,
scores, announcements), and each step writes only the parts that change (cards, voids, the trick,
tricks won and who is still alive) and the seat's hand shape. The tests check every step's
features and every card against the engine's observations and ``RLBot``.

``play_rounds`` plays whole rounds from their first card whose auctions differ, game by game (the
bid search imagines a different auction on every deal): each game has its own templates, leader,
trump and estimates.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from estimation.bots.hand_eval import suit_tables, trump_values
from estimation.engine import Observation, RulesConfig
from estimation.engine.cards import FULL_DECK, SUIT_MASKS
from estimation.rl.encoder import (
    CONTEXT,
    HAND,
    NUM_FEATURES,
    PLAYER_FEATURES,
    PLAYERS,
    SHAPE,
    TRICK,
    VOIDS,
    encode,
)

type Floats = npt.NDArray[np.float32]
type Masks = npt.NDArray[np.uint64]
type Ints = npt.NDArray[np.int64]
type CardNet = Callable[[Floats], Floats]
"""Advantage of each of the 52 cards for a batch of encoded positions (``Policy.card_values``)."""

CARDS = 52
SHAPE_SIZE = 9
_CARD_BLOCKS = 6  # hand, three opponents' played cards, table, unseen: contiguous from HAND
_SUIT_MASKS = np.array(SUIT_MASKS, dtype=np.uint64)
_ONE = np.uint64(1)
_FULL = np.uint64(FULL_DECK)
_SIDE, _SANS, _TRUMP, _RUFF = (np.array(t, dtype=np.float32) for t in suit_tables())
_COUNT = np.array([p.bit_count() for p in range(1 << 13)], dtype=np.float32)
_SHIFTS = np.array([0, 13, 26, 39], dtype=np.uint64)


def seat_templates(obs: Observation, rules: RulesConfig) -> Floats:
    """``(4, NUM_FEATURES)``: each seat's encoding of ``obs``'s public state with an empty hand.
    ``play_out`` overwrites every part that depends on the cards."""
    rows = [
        encode(replace(obs, seat=seat, hand=0, current_player=seat), rules) for seat in range(4)
    ]
    return np.stack(rows)


def shape_features(dealt: int) -> list[float]:
    """The encoder's ``shape`` block for a dealt hand: suit lengths and HeuristicBot's expected
    tricks under each trump, over 13."""
    lengths = [((dealt >> (13 * suit)) & 0x1FFF).bit_count() / 13.0 for suit in range(4)]
    return lengths + [value / 13.0 for value in trump_values(dealt)]


def shapes_of(dealt: Masks) -> Floats:
    """``(*dealt.shape, 9)`` encoder ``shape`` block for each 52-card hand bitmask in ``dealt``."""
    patterns = (dealt[..., None] >> _SHIFTS) & np.uint64(0x1FFF)
    counts = _COUNT[patterns]
    side = _SIDE[patterns]
    ruff = _RUFF[patterns]
    side_sum = side.sum(axis=-1, keepdims=True) - side
    ruff_sum = ruff.sum(axis=-1, keepdims=True) - ruff
    trump_tricks = side_sum + np.where(
        counts > 0, np.minimum(counts, _TRUMP[patterns] + ruff_sum), 0.0
    )
    sans_tricks = _SANS[patterns].sum(axis=-1, keepdims=True)
    return (np.concatenate([counts, trump_tricks, sans_tricks], axis=-1) / np.float32(13.0)).astype(
        np.float32
    )


def bits(masks: Masks) -> npt.NDArray[np.uint8]:
    """``(..., 52)`` 0/1 bytes of each 52-card bitmask in ``masks``."""
    flat = np.ascontiguousarray(masks, dtype="<u8")
    unpacked = np.unpackbits(flat.view(np.uint8), bitorder="little")
    return unpacked.reshape(*flat.shape, 64)[..., :CARDS]


def _only_card(masks: Masks) -> Ints:
    """The card id of each single-bit mask in ``masks`` (IEEE-754 exponent of ``1 << card``)."""
    return (masks.astype(np.float64).view(np.int64) >> 52) - 1023


@dataclass(slots=True)
class Start:
    """The position every game of a search starts from, apart from the hidden hands: the trick
    in progress (cards in play order), its leader, tricks won and finished so far, the round's
    estimates and trump."""

    trick: Sequence[int]
    leader: int
    won: Sequence[int]
    tricks_done: int
    estimates: Sequence[int]
    trump: int


def play_out(
    net: CardNet,
    templates: Floats,
    start: Start,
    hands: Masks,
    played: Masks,
    voids: npt.NDArray[np.uint8],
    shapes: Floats,
    first_cards: Ints,
) -> Ints:
    """Play every game to the end of the round and return each game's tricks won, ``(B, 4)``.

    Game ``b`` has the remaining hands ``hands[b]`` (``(B, 4)`` bitmasks), the cards each seat
    has played ``played[b]`` (the current trick included), the suits each has shown out of
    ``voids[b]`` (4-bit masks) and the shape block of each seat's dealt hand ``shapes[b]``
    (``(B, 4, 9)``). The seat to act first plays ``first_cards[b]``; from there every seat plays
    the legal card with the highest advantage under ``net``, the lowest card on a tie, as
    ``RLBot`` does. ``hands``, ``played`` and ``voids`` are used up: pass copies.
    """
    games = hands.shape[0]
    trick_cards = np.zeros((games, 4), np.int64)
    best_card, best_at = -1, 0
    for i, c in enumerate(start.trick):
        trick_cards[:, i] = c
        if i == 0 or _beats(c, best_card, start.trump):
            best_card, best_at = c, i
    return _play(
        net,
        np.broadcast_to(templates, (games, 4, NUM_FEATURES)),
        np.full(games, start.trump, np.int64),
        np.tile(np.asarray(start.estimates, np.int64), (games, 1)),
        np.full(games, start.leader, np.int64),
        np.tile(np.asarray(start.won, np.int64), (games, 1)),
        start.tricks_done,
        trick_cards,
        len(start.trick),
        np.full(games, best_card, np.int64),
        np.full(games, best_at, np.int64),
        hands,
        played,
        voids,
        shapes,
        first_cards.astype(np.int64),
    )


def play_rounds(
    net: CardNet,
    templates: Floats,
    leaders: Ints,
    trumps: Ints,
    estimates: Ints,
    hands: Masks,
    shapes: Floats,
) -> Ints:
    """Play every game's round from its first card to the end and return each game's tricks won,
    ``(B, 4)``. Game ``b`` has its own ``templates[b]`` (``(B, 4, NUM_FEATURES)``, as
    ``seat_templates``), leader, trump and ``estimates[b]``, the dealt hands ``hands[b]`` and
    their shape blocks ``shapes[b]``; every seat plays as ``RLBot`` does. ``hands`` is used up."""
    games = hands.shape[0]
    return _play(
        net,
        templates,
        trumps.astype(np.int64),
        estimates.astype(np.int64),
        leaders.astype(np.int64),
        np.zeros((games, 4), np.int64),
        0,
        np.zeros((games, 4), np.int64),
        0,
        np.full(games, -1, np.int64),
        np.zeros(games, np.int64),
        hands,
        np.zeros((games, 4), np.uint64),
        np.zeros((games, 4), np.uint8),
        shapes,
        None,
    )


def _play(
    net: CardNet,
    templates: Floats,
    trump: Ints,
    est: Ints,
    leaders: Ints,
    won: Ints,
    tricks_done: int,
    trick_cards: Ints,
    tlen: int,
    best: Ints,
    best_pos: Ints,
    hands: Masks,
    played: Masks,
    voids: npt.NDArray[np.uint8],
    shapes: Floats,
    card: Ints | None,
) -> Ints:
    """``play_out`` and ``play_rounds``: every game's own ``templates`` (``(B, 4, F)``), trump
    (Sans, 4, makes no card trump), estimates and position, played to the end of the round."""
    games = hands.shape[0]
    ar = np.arange(games, dtype=np.int64)
    buf = np.empty((games, NUM_FEATURES), np.float32)
    while True:
        acting = (leaders + tlen) & 3
        hand = hands[ar, acting]
        led = trick_cards[:, 0] // 13
        if card is None:
            if tlen:
                follow = hand & _SUIT_MASKS[led]
                legal = np.where(follow != 0, follow, hand)
            else:
                legal = hand
            choice = (legal & (legal - _ONE)) != 0
            if choice.all():
                _features(
                    buf, templates, played, voids, shapes, acting, hand, trick_cards, tlen,
                    leaders, best, best_pos, won, est, tricks_done, trump, ar,
                )  # fmt: skip
                values = np.where(bits(legal).astype(bool), net(buf), -np.inf)
                card = values.argmax(axis=1).astype(np.int64)
            else:
                card = _only_card(legal)
                rows = np.flatnonzero(choice)
                if rows.size:
                    sub = buf[: rows.size]
                    _features(
                        sub, templates, played[rows], voids[rows], shapes[rows], acting[rows],
                        hand[rows], trick_cards[rows], tlen, leaders[rows], best[rows],
                        best_pos[rows], won[rows], est[rows], tricks_done, trump[rows], rows,
                    )  # fmt: skip
                    values = np.where(bits(legal[rows]).astype(bool), net(sub), -np.inf)
                    card[rows] = values.argmax(axis=1)
        bit = np.left_shift(_ONE, card.astype(np.uint64))
        hands[ar, acting] = hand ^ bit
        played[ar, acting] |= bit
        if tlen:
            off = card // 13 != led
            voids[ar[off], acting[off]] |= np.left_shift(1, led[off]).astype(np.uint8)
            suit = card // 13
            best_suit = best // 13
            beats = ((suit == best_suit) & (card > best)) | ((suit != best_suit) & (suit == trump))
            best = np.where(beats, card, best)
            best_pos = np.where(beats, tlen, best_pos)
        else:
            best = card.copy()
            best_pos = np.zeros(games, np.int64)
        trick_cards[:, tlen] = card
        card = None
        tlen += 1
        if tlen == 4:
            winners = (leaders + best_pos) & 3
            won[ar, winners] += 1
            leaders = winners
            tlen = 0
            tricks_done += 1
            if tricks_done == 13:
                return won


def _beats(card: int, best: int, trump: int) -> bool:
    if card // 13 == best // 13:
        return card > best
    return card // 13 == trump


def _features(
    x: Floats,
    templates: Floats,
    played: Masks,
    voids: npt.NDArray[np.uint8],
    shapes: Floats,
    acting: Ints,
    hand: Masks,
    trick_cards: Ints,
    tlen: int,
    leaders: Ints,
    best: Ints,
    best_pos: Ints,
    won: Ints,
    est: Ints,
    tricks_done: int,
    trump: Ints,
    games_of: Ints,
) -> None:
    """Write each game's acting seat's features into ``x``, exactly as ``encode`` would. Row
    ``i`` is game ``games_of[i]``: its templates are ``templates[games_of[i]]``; every other
    array is already the rows'."""
    games = x.shape[0]
    ar = np.arange(games, dtype=np.int64)
    x[:] = templates[games_of, acting]
    everyone = played[:, 0] | played[:, 1] | played[:, 2] | played[:, 3]
    table = np.zeros(games, np.uint64)
    for i in range(tlen):
        table |= np.left_shift(_ONE, trick_cards[:, i].astype(np.uint64))
    masks = np.stack(
        [
            hand,
            played[ar, (acting + 1) & 3],
            played[ar, (acting + 2) & 3],
            played[ar, (acting + 3) & 3],
            table,
            _FULL & ~everyone & ~hand,
        ],
        axis=1,
    )
    x[:, HAND : HAND + CARDS * _CARD_BLOCKS] = bits(masks).reshape(games, -1)
    for r in range(1, 4):
        void = voids[ar, (acting + r) & 3]
        base = VOIDS + 4 * (r - 1)
        for suit in range(4):
            x[:, base + suit] = (void >> suit) & 1
    x[:, TRICK : TRICK + 14] = 0.0
    x[:, TRICK + 4 + tlen] = 1.0
    if tlen:
        x[ar, TRICK + trick_cards[:, 0] // 13] = 1.0
        best_seat = (leaders + best_pos) & 3
        x[ar, TRICK + 8 + ((best_seat - acting) & 3)] = 1.0
        x[:, TRICK + 12] = best // 13 == trump
        x[:, TRICK + 13] = (best % 13) / 12.0
    x[:, CONTEXT + 22] = tricks_done / 13.0
    left = 13 - tricks_done
    for r in range(4):
        seat = (acting + r) & 3
        base = PLAYERS + PLAYER_FEATURES * r
        w = won[ar, seat]
        e = est[ar, seat]
        alive = (w <= e) & (e <= w + left)
        x[:, base + 2] = w / 13.0
        x[:, base + 3] = (e - w) / 13.0
        x[:, base + 6] = alive
        x[:, base + 7] = ~alive
    x[:, SHAPE : SHAPE + SHAPE_SIZE] = shapes[ar, acting]
