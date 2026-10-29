"""How many tricks a hand should take under each trump (SPEC §6 M2).

HeuristicBot bids and estimates from this model: high cards (A, K, Q), trump length, and ruffs
from short side suits. PIMC (M3) uses the same model to weight sampled deals by how well each
player's hand fits its estimate. The values are bot design choices, not rules.
"""

from dataclasses import dataclass
from functools import cache

from estimation.engine.cards import SANS

ACE = 1 << 12
KING = 1 << 11
QUEEN = 1 << 10
JACK = 1 << 9
SUIT_PATTERNS = 1 << 13


@dataclass(frozen=True, slots=True)
class HandValues:
    """Tricks credited to each feature of a hand."""

    ace: float = 1.0
    king: float = 0.75  # with at least one other card of its suit
    bare_king: float = 0.25
    queen: float = 0.4  # with at least two other cards of its suit
    short_queen: float = 0.1
    trump_king: float = 0.9  # trump honours, with the same guards as side honours
    trump_queen: float = 0.6
    trump_jack: float = 0.3  # with at least three other trumps
    long_trump: float = 1.0  # each trump beyond the third
    void_ruff: float = 0.8  # each void side suit, when holding trumps
    singleton_ruff: float = 0.4  # each singleton side suit, when holding trumps
    sans_long_card: float = 0.5  # in Sans, each card beyond the fourth of a suit headed by A or K


DEFAULT_VALUES = HandValues()


def _side_value(pattern: int, values: HandValues) -> float:
    """High-card tricks of one non-trump suit, given as a 13-bit rank pattern."""
    length = pattern.bit_count()
    tricks = 0.0
    if pattern & ACE:
        tricks += values.ace
    if pattern & KING:
        tricks += values.king if length >= 2 else values.bare_king
    if pattern & QUEEN:
        tricks += values.queen if length >= 3 else values.short_queen
    return tricks


def _sans_value(pattern: int, values: HandValues) -> float:
    tricks = _side_value(pattern, values)
    length = pattern.bit_count()
    if length > 4 and pattern & (ACE | KING):
        tricks += (length - 4) * values.sans_long_card
    return tricks


def _trump_value(pattern: int, values: HandValues) -> float:
    """Tricks of the trump suit itself, before ruffs: honours plus length beyond three."""
    length = pattern.bit_count()
    tricks = 0.0
    if pattern & ACE:
        tricks += values.ace
    if pattern & KING and length >= 2:
        tricks += values.trump_king
    if pattern & QUEEN and length >= 3:
        tricks += values.trump_queen
    if pattern & JACK and length >= 4:
        tricks += values.trump_jack
    if length > 3:
        tricks += (length - 3) * values.long_trump
    return tricks


@cache
def suit_tables(
    values: HandValues = DEFAULT_VALUES,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Per 13-bit suit pattern: side value, Sans value, trump value and ruff value."""
    side = tuple(_side_value(p, values) for p in range(SUIT_PATTERNS))
    sans = tuple(_sans_value(p, values) for p in range(SUIT_PATTERNS))
    trump = tuple(_trump_value(p, values) for p in range(SUIT_PATTERNS))
    ruff = tuple(
        values.void_ruff if p == 0 else values.singleton_ruff if p.bit_count() == 1 else 0.0
        for p in range(SUIT_PATTERNS)
    )
    return side, sans, trump, ruff


def expected_tricks(hand: int, trump: int, values: HandValues = DEFAULT_VALUES) -> float:
    """Tricks ``hand`` (a bitmask) should take with ``trump`` (a suit, or SANS) as trump.

    Trump tricks (honours, length and ruffs together) never exceed the number of trumps held.
    """
    side, sans, trumps, ruffs = suit_tables(values)
    patterns = (hand & 0x1FFF, (hand >> 13) & 0x1FFF, (hand >> 26) & 0x1FFF, hand >> 39)
    if trump == SANS:
        return sans[patterns[0]] + sans[patterns[1]] + sans[patterns[2]] + sans[patterns[3]]
    total = 0.0
    ruff = 0.0
    for suit in range(4):
        if suit != trump:
            total += side[patterns[suit]]
            ruff += ruffs[patterns[suit]]
    held = patterns[trump]
    count = held.bit_count()
    if count:
        total += min(float(count), trumps[held] + ruff)
    return total


def trump_values(
    hand: int, values: HandValues = DEFAULT_VALUES
) -> tuple[float, float, float, float, float]:
    """Expected tricks under each trump, indexed Clubs, Diamonds, Hearts, Spades, Sans."""
    return (
        expected_tricks(hand, 0, values),
        expected_tricks(hand, 1, values),
        expected_tricks(hand, 2, values),
        expected_tricks(hand, 3, values),
        expected_tricks(hand, SANS, values),
    )
