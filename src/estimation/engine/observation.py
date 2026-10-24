"""What one seat may see (SPEC §5.4): everything public plus its own hand, and nothing else."""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


class Phase(IntEnum):
    """Round phases. Open rounds run DASH, AUCTION, ESTIMATION, PLAY, DONE; fixed-trump rounds
    run ESTIMATION, then AUCTION and ESTIMATION again after a super call, then PLAY, DONE."""

    DASH = 0
    AUCTION = 1
    ESTIMATION = 2
    PLAY = 3
    DONE = 4


class WithInfo(NamedTuple):
    """A With player: its seat, the trick count ``w`` it joined at, and the trump it joined."""

    seat: int
    w: int
    trump: int


class TrickRecord(NamedTuple):
    """A finished trick: its leader, its cards in play order (leader first) and its winner."""

    leader: int
    cards: tuple[int, int, int, int]
    winner: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    """One seat's view of a round.

    ``hand`` is the seat's own remaining hand as a bitmask; no field holds another seat's
    unplayed cards. Every Dash Call is public as soon as it is made (DSH-2).

    ``scores`` are the game scores before this round. ``fixed_left`` lists the fixed trumps not
    yet played, the current round's included. ``super_call`` is true once a super call (FIX-4)
    has opened bidding in this fixed-trump round; a Caller or With player at 8 or more is
    scored as a super call either way (SCR-6). ``top_bid`` and ``top_bidder`` describe the
    highest bid so far and, once the auction ends, the winning bid; both are ``None`` when
    nobody bid. ``with_players`` are the current With declarations during an auction and the
    players scored like the Caller after it. ``total``, ``over`` and ``risks`` are known once
    estimation is complete.
    """

    seat: int
    round_no: int
    phase: Phase
    current_player: int | None
    carry: int
    scores: tuple[int, int, int, int]
    fixed_trump: int | None
    fixed_left: tuple[int, ...]
    replays: int
    super_call: bool
    trump: int | None
    hand: int
    log: tuple[tuple[int, int], ...]
    dash_callers: tuple[int, ...]
    top_bid: int | None
    top_bidder: int | None
    passed: tuple[int, ...]
    caller: int | None
    with_players: tuple[WithInfo, ...]
    estimates: tuple[int | None, int | None, int | None, int | None]
    last_estimator: int | None
    total: int | None
    over: bool | None
    risks: int | None
    leader: int | None
    trick: tuple[tuple[int, int], ...]
    tricks: tuple[TrickRecord, ...]
    tricks_won: tuple[int, int, int, int]
