"""A hand's trick chances by Yosif's rules (design 5.1).

Every card that could take a trick becomes a ``Chance``: how likely it takes one when the player
wants tricks (``win``), and how likely the player can avoid that trick when not (``dodge``). Trump
tricks come from the split model in ``trumps``; side suits from the honours, how many cards cover
them, where the bidders sit, and whether the round is Over or Under.

Rules are Yosif's; the numbers in ``CountValues`` are Claude's starting values, to be calibrated
by the quiz and the arena.
"""

from dataclasses import dataclass

from estimation.engine import SANS, Observation
from estimation.engine.actions import bid_of, is_bid

from .trumps import trump_odds

ACE, KING, QUEEN = 12, 11, 10
LOW_CARDS = 0b111  # ranks 2, 3 and 4: the cards that let a player get out of a trick [Yosif]
SUIT = 0x1FFF
MAX_LONG_OPPONENTS = 3


@dataclass(frozen=True, slots=True)
class Chance:
    """One possible trick: the chance it is taken when wanted, and dodged when not."""

    win: float
    dodge: float


@dataclass(frozen=True, slots=True)
class Reads:
    """What the bids so far say. ``before`` and ``after`` are 4-bit masks of the suits bid by
    the player right before and right after this seat; ``bidders[s]`` is how many other
    players bid suit ``s``."""

    before: int = 0
    after: int = 0
    bidders: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class HandCount:
    """The chance of each trump trick count (0 up), and every side card's trick chance."""

    trumps: tuple[float, ...]
    side: tuple[Chance, ...]


@dataclass(frozen=True, slots=True)
class CountValues:
    """Starting values [Claude] for Yosif's rules [Yosif]: honours are worth more with cover,
    behind a bidder, and in Under rounds; low cards let a player dodge."""

    ace_win: float = 1.0  # an ace is a sure trick [Yosif]
    ace_dodge: float = 0.15
    ace_dodge_short: float = 0.5  # another side suit is void or a singleton to throw it on
    king_alone: float = 0.30
    king_one: float = 0.50
    king_more: float = 0.75
    queen_short: float = 0.20  # alone or with one card
    queen_more: float = 0.45  # with two or more cards
    honour_dodge: float = 0.5
    low_card_dodge: float = 0.2  # per card ranked 2-4 in the suit
    max_dodge: float = 0.95
    position_shift: float = 0.25  # behind (+) or in front of (-) a player who bid the suit
    over_factor: float = 0.75  # Over: everyone fights, weak honours lose value [Yosif]
    under_bare_king: float = 0.75  # Under: a bare king is a very likely trick [Yosif]
    sans_long_win: float = 0.5  # Sans: each card beyond the fourth of a suit headed by A or K
    sans_long_dodge: float = 0.5


NO_READS = Reads()
DEFAULT_COUNT = CountValues()


def read_bids(obs: Observation) -> Reads:
    """The suits other players bid this round, and who sits next to this seat (design 5.4)."""
    me = obs.seat
    before = after = 0
    bidders: list[set[int]] = [set(), set(), set(), set()]
    for seat, action in obs.log:
        if seat == me or not is_bid(action):
            continue
        suit = bid_of(action)[1]
        if suit == SANS:
            continue
        bidders[suit].add(seat)
        if seat == (me + 3) % 4:
            before |= 1 << suit
        elif seat == (me + 1) % 4:
            after |= 1 << suit
    counts = tuple(len(seats) for seats in bidders)
    return Reads(before, after, (counts[0], counts[1], counts[2], counts[3]))


@dataclass(frozen=True, slots=True)
class _Suit:
    pattern: int
    suit: int
    short_elsewhere: bool  # another side suit is void or a singleton


def count_hand(
    hand: int,
    trump: int,
    reads: Reads = NO_READS,
    over: bool | None = None,
    values: CountValues = DEFAULT_COUNT,
) -> HandCount:
    """The trick chances of ``hand`` with ``trump`` (a suit 0-3, or SANS) as trump.

    ``over`` is True in an Over round, False in an Under round and None while unknown.
    """
    patterns = [(hand >> (13 * suit)) & SUIT for suit in range(4)]
    side_suits = [suit for suit in range(4) if suit != trump]
    if trump == SANS:
        trumps: tuple[float, ...] = (1.0,)
    else:
        trumps = trump_odds(patterns[trump], min(reads.bidders[trump], MAX_LONG_OPPONENTS))
    side: list[Chance] = []
    for suit in side_suits:
        short = any(patterns[other].bit_count() <= 1 for other in side_suits if other != suit)
        side += _suit_chances(_Suit(patterns[suit], suit, short), trump, reads, over, values)
    return HandCount(trumps, tuple(side))


def _suit_chances(
    held: _Suit, trump: int, reads: Reads, over: bool | None, values: CountValues
) -> list[Chance]:
    pattern = held.pattern
    length = pattern.bit_count()
    chances: list[Chance] = []
    if pattern >> ACE & 1:
        dodge = values.ace_dodge_short if held.short_elsewhere else values.ace_dodge
        chances.append(Chance(values.ace_win, dodge))
    dodge = min(
        values.max_dodge,
        values.honour_dodge + values.low_card_dodge * (pattern & LOW_CARDS).bit_count(),
    )
    if pattern >> KING & 1:
        if length == 1:
            win = values.under_bare_king if over is False else values.king_alone
        else:
            win = values.king_one if length == 2 else values.king_more
        chances.append(Chance(_adjust(win, held.suit, reads, over, values), dodge))
    if pattern >> QUEEN & 1:
        win = values.queen_short if length <= 2 else values.queen_more
        chances.append(Chance(_adjust(win, held.suit, reads, over, values), dodge))
    if trump == SANS and length > 4 and pattern & (1 << ACE | 1 << KING):
        chances += [Chance(values.sans_long_win, values.sans_long_dodge)] * (length - 4)
    return chances


def _adjust(win: float, suit: int, reads: Reads, over: bool | None, values: CountValues) -> float:
    """A king's or queen's chance after position (design 5.1) and an Over round."""
    if reads.before >> suit & 1:
        win += values.position_shift
    if reads.after >> suit & 1:
        win -= values.position_shift
    if over:
        win *= values.over_factor
    return min(1.0, max(0.0, win))
