"""One round of Estimation as a state machine (SPEC §2.4-§2.9, §5.4).

Hands are bitmasks and legal card plays come from per-suit lookup tables, so a complete
random-play round fits the SPEC §5.5 speed budget. Internally ``-1`` means "none" for seats,
estimates and trumps; observations turn it into ``None``.
"""

import operator
from collections.abc import Iterable
from functools import cache
from typing import Self, SupportsIndex

import numpy as np
import numpy.typing as npt

from .actions import (
    DASH_CALL,
    ESTIMATE_BASE,
    FIRST_BID,
    LAST_BID,
    NO_DASH_CALL,
    NUM_ACTIONS,
    PASS,
    PLAY_BASE,
    WITH,
    action_str,
    bid_action,
)
from .cards import CLUBS, DIAMONDS, FULL_DECK, HEARTS, SANS, SPADES, SUIT_MASKS, card_str
from .config import DEFAULT_CONFIG, RulesConfig
from .observation import Observation, Phase, TrickRecord, WithInfo
from .scoring import RoundOutcome, RoundResult, score_round

__all__ = ["FIXED_ORDER", "IllegalActionError", "Phase", "Round"]

FIXED_ORDER = (SANS, SPADES, HEARTS, DIAMONDS, CLUBS)
"""The fixed trumps of rounds 14 onward, in the order they are played (FIX-1)."""

DASH = Phase.DASH
AUCTION = Phase.AUCTION
ESTIMATION = Phase.ESTIMATION
PLAY = Phase.PLAY
DONE = Phase.DONE

_NO_BID = FIRST_BID - 1  # top-bid value before anyone bids: every bid beats it
_LONE_MAX_BID = bid_action(12, SANS)  # AUC-5: a lone bidder may not bid 13 tricks

Trick = tuple[int, tuple[int, int, int, int], int]  # leader, cards in play order, winner


class IllegalActionError(ValueError):
    """``step`` was given an action that is not legal in the current state."""


def _build_play_tables() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """For each suit, the play-action ids for every 13-bit pattern of that suit's cards."""
    tables: list[tuple[tuple[int, ...], ...]] = []
    for suit in range(4):
        base = PLAY_BASE + 13 * suit
        table: list[tuple[int, ...]] = [()]
        for pattern in range(1, 1 << 13):
            top = pattern.bit_length() - 1
            table.append((*table[pattern ^ (1 << top)], base + top))
        tables.append(tuple(table))
    return tuple(tables)


_PLAYS_C, _PLAYS_D, _PLAYS_H, _PLAYS_S = _PLAYS = _build_play_tables()


@cache
def _super_calls(fixed_trump: int, minimum: int) -> tuple[int, ...]:
    """Bid ids that open a super call in a round with this fixed trump (FIX-4)."""
    return tuple(
        bid_action(n, t) for n in range(minimum, 14) for t in range(SANS + 1) if t != fixed_trump
    )


def _describe(action: int) -> str:
    return action_str(action) if 0 <= action < NUM_ACTIONS else f"action {action}"


def _as_action(action: object) -> int:
    """Accept any integer-like action id (numpy integers included), reject everything else."""
    if isinstance(action, SupportsIndex):
        return operator.index(action)
    raise IllegalActionError(f"an action must be an integer id, not {action!r}")


class Round:
    """One round. Create it with ``Round.new``; then loop ``legal_actions()`` and ``step()``.

    ``phase`` and ``current_player`` say whose decision it is. Every action goes through
    ``step``, which raises ``IllegalActionError`` on an illegal one and never corrects it.
    """

    __slots__ = (
        "_c",
        "_caller",
        "_chosen",
        "_cur",
        "_dash",
        "_est",
        "_final",
        "_hands",
        "_last",
        "_leader",
        "_led",
        "_log",
        "_max_bid",
        "_order",
        "_passed",
        "_past",
        "_pos",
        "_result",
        "_super",
        "_supers",
        "_top",
        "_top_seat",
        "_total",
        "_trick",
        "_trump",
        "_with_bid",
        "_with_w",
        "_won",
        "carry",
        "config",
        "first",
        "fixed_left",
        "fixed_trump",
        "hands",
        "phase",
        "replays",
        "round_no",
        "scores",
    )

    round_no: int
    first: int  # F, the first seat to act (GAM-2)
    fixed_trump: int | None
    carry: int
    scores: tuple[int, int, int, int]  # game scores before this round
    config: RulesConfig
    fixed_left: tuple[int, ...]
    replays: int
    hands: tuple[int, int, int, int]  # the deal
    phase: Phase

    _cur: int  # seat to act
    _hands: list[int]  # remaining hands
    _log: list[tuple[int, int]]  # every (seat, action), hidden Dash choices included
    _dash: list[bool]
    _chosen: int  # Dash choices made so far
    _passed: list[bool]  # out of the auction: passed, or a Dash caller
    _top: int  # top bid id, or _NO_BID
    _top_seat: int
    _with_bid: list[int]  # bid id each seat said With on in the auction
    _final: bool  # the top bidder's final turn (AUC-3)
    _max_bid: int
    _super: bool  # a super call was made (FIX-4)
    _supers: tuple[int, ...]  # super-call bid ids open to fixed-trump estimators
    _est: list[int]
    _total: int  # sum of the estimates made so far
    _order: tuple[int, ...]  # seats estimating, in order
    _pos: int  # index in _order of the next estimator
    _caller: int
    _c: int  # the Caller's trick count
    _with_w: list[int]  # w of each player scored like the Caller
    _trump: int
    _last: int  # last estimator
    _leader: int
    _trick: list[int]  # cards of the current trick, leader first
    _led: int  # led suit of the current trick
    _won: list[int]
    _past: list[Trick]
    _result: RoundResult | None

    @classmethod
    def new(
        cls,
        round_no: int,
        hands: tuple[int, int, int, int],
        fixed_trump: int | None = None,
        carry: int = 1,
        scores: tuple[int, int, int, int] = (0, 0, 0, 0),
        config: RulesConfig = DEFAULT_CONFIG,
        *,
        fixed_left: tuple[int, ...] | None = None,
        replays: int = 0,
    ) -> Self:
        """Start round ``round_no`` from the dealt ``hands``.

        ``fixed_trump`` is ``None`` in open rounds. ``fixed_left`` (the fixed trumps still to
        play, this round's included) and ``replays`` are game context shown in observations;
        ``fixed_left`` defaults to the fixed trumps from this round's onward.
        """
        if round_no < 1:
            raise ValueError("round_no must be 1 or more")
        if len(hands) != 4:
            raise ValueError("hands must have one hand per seat")
        deal = (int(hands[0]), int(hands[1]), int(hands[2]), int(hands[3]))
        if any(h.bit_count() != 13 for h in deal) or deal[0] | deal[1] | deal[2] | deal[3] != (
            FULL_DECK
        ):
            raise ValueError("hands must split the 52 cards into four hands of 13")
        if fixed_trump is not None and not 0 <= fixed_trump <= SANS:
            raise ValueError(f"no trump with index {fixed_trump}")
        if carry < 1:
            raise ValueError("carry must be at least 1")
        if len(scores) != 4:
            raise ValueError("scores must have one entry per seat")
        if replays < 0:
            raise ValueError("replays must be 0 or more")
        if fixed_left is None:
            start = 0 if fixed_trump is None else FIXED_ORDER.index(fixed_trump)
            fixed_left = FIXED_ORDER[start:]

        rnd = object.__new__(cls)
        rnd.round_no = round_no
        rnd.first = first = (round_no - 1) % 4
        rnd.fixed_trump = fixed_trump
        rnd.carry = carry
        rnd.scores = (scores[0], scores[1], scores[2], scores[3])
        rnd.config = config
        rnd.fixed_left = tuple(fixed_left)
        rnd.replays = replays
        rnd.hands = deal
        rnd._cur = first
        rnd._hands = list(deal)
        rnd._log = []
        rnd._dash = [False, False, False, False]
        rnd._chosen = 0
        rnd._passed = [False, False, False, False]
        rnd._top = _NO_BID
        rnd._top_seat = -1
        rnd._with_bid = [-1, -1, -1, -1]
        rnd._final = False
        rnd._max_bid = LAST_BID
        rnd._super = False
        rnd._est = [-1, -1, -1, -1]
        rnd._total = 0
        rnd._pos = 0
        rnd._caller = -1
        rnd._c = 0
        rnd._with_w = [-1, -1, -1, -1]
        rnd._leader = -1
        rnd._trick = []
        rnd._led = -1
        rnd._won = [0, 0, 0, 0]
        rnd._past = []
        rnd._result = None
        if fixed_trump is None:
            rnd.phase = DASH  # DSH-1
            rnd._supers = ()
            rnd._order = ()
            rnd._trump = -1
            rnd._last = -1
        else:
            rnd.phase = ESTIMATION  # FIX-2: no Dash Call, no auction; F estimates first
            rnd._supers = _super_calls(fixed_trump, config.super_call_min)
            rnd._order = (first, (first + 1) & 3, (first + 2) & 3, (first + 3) & 3)
            rnd._trump = fixed_trump
            rnd._last = rnd._order[3]
        return rnd

    @classmethod
    def replay(
        cls,
        round_no: int,
        hands: tuple[int, int, int, int],
        fixed_trump: int | None,
        carry: int,
        scores: tuple[int, int, int, int],
        config: RulesConfig,
        actions: Iterable[int],
        *,
        fixed_left: tuple[int, ...] | None = None,
        replays: int = 0,
    ) -> Self:
        """Rebuild a round from its action log."""
        rnd = cls.new(
            round_no,
            hands,
            fixed_trump,
            carry,
            scores,
            config,
            fixed_left=fixed_left,
            replays=replays,
        )
        for action in actions:
            rnd.step(action)
        return rnd

    @property
    def current_player(self) -> int | None:
        """The seat to act, or ``None`` once the round is DONE."""
        return None if self.phase is DONE else self._cur

    @property
    def log(self) -> tuple[tuple[int, int], ...]:
        """Every ``(seat, action)`` so far, hidden Dash choices included. Not for bots."""
        return tuple(self._log)

    def result(self) -> RoundResult:
        """The scored round, once ``phase`` is DONE."""
        if self._result is None:
            raise RuntimeError("the round is not over yet")
        return self._result

    # Legal actions

    def legal_actions(self) -> list[int]:
        """Legal action ids for the current player, in ascending order; empty once DONE."""
        phase = self.phase
        if phase is PLAY:
            hand = self._hands[self._cur]
            if self._trick:
                led = self._led
                follow = (hand >> (13 * led)) & 0x1FFF
                if follow:  # PLY-2: must follow suit
                    return list(_PLAYS[led][follow])
            return [
                *_PLAYS_C[hand & 0x1FFF],
                *_PLAYS_D[(hand >> 13) & 0x1FFF],
                *_PLAYS_H[(hand >> 26) & 0x1FFF],
                *_PLAYS_S[hand >> 39],
            ]
        if phase is AUCTION:
            seat = self._cur
            top = self._top
            actions = list(range(top + 1, self._max_bid + 1))
            top_seat = self._top_seat
            if (
                not self._final
                and top_seat >= 0
                and top_seat != seat
                and self._with_bid[seat] != top
            ):
                actions.append(WITH)
            actions.append(PASS)
            return actions
        if phase is ESTIMATION:
            return self._legal_estimates()
        if phase is DASH:
            return [NO_DASH_CALL, DASH_CALL]
        return []

    def _legal_estimates(self) -> list[int]:
        if self._caller < 0:  # fixed-trump estimation before any super call (FIX-2, FIX-4)
            actions = list(self._supers)
            if self._pos == 3:
                forbidden = ESTIMATE_BASE + 13 - self._total
                actions.extend(a for a in range(ESTIMATE_BASE, PLAY_BASE) if a != forbidden)
            else:
                actions.extend(range(ESTIMATE_BASE, PLAY_BASE))
            return actions
        c = self._c
        w = self._with_w[self._cur]
        low = w if w >= 0 else 0  # EST-3
        if self._pos == len(self._order) - 1:
            forbidden = 13 - self._total  # EST-5
            if low <= forbidden <= c:
                if low == c:
                    low = 0  # EST-5: nothing from w to c is allowed, so the floor drops
                return [ESTIMATE_BASE + k for k in range(low, c + 1) if k != forbidden]
        return list(range(ESTIMATE_BASE + low, ESTIMATE_BASE + c + 1))

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Legal actions as a bool array of length 120."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        actions = self.legal_actions()
        if actions:
            mask[actions] = True
        return mask

    # Stepping

    def step(self, action: int) -> None:
        """Apply a legal action for the current player."""
        if type(action) is not int:
            action = _as_action(action)
        phase = self.phase
        if phase is PLAY:
            seat = self._cur
            card = action - PLAY_BASE
            if not 0 <= card < 52:
                raise IllegalActionError(f"{_describe(action)} is not a card play")
            hand = self._hands[seat]
            bit = 1 << card
            trick = self._trick
            if trick:
                follow = hand & SUIT_MASKS[self._led]
                legal = follow if follow else hand  # PLY-2
            else:
                legal = hand
            if not legal & bit:
                raise IllegalActionError(f"seat {seat} may not play {card_str(card)}")
            self._hands[seat] = hand ^ bit
            self._log.append((seat, action))
            if not trick:
                self._led = card // 13
            trick.append(card)
            if len(trick) == 4:
                self._end_trick()
            else:
                self._cur = (seat + 1) & 3
        elif phase is AUCTION:
            self._step_auction(action)
        elif phase is ESTIMATION:
            self._step_estimation(action)
        elif phase is DASH:
            self._step_dash(action)
        else:
            raise IllegalActionError("the round is over: no action is legal")

    def _step_dash(self, action: int) -> None:
        if action != NO_DASH_CALL and action != DASH_CALL:
            raise IllegalActionError(f"{_describe(action)} is not a Dash Call decision")
        seat = self._cur
        self._dash[seat] = action == DASH_CALL
        self._log.append((seat, action))
        self._chosen += 1
        if self._chosen < 4:
            self._cur = (seat + 1) & 3  # DSH-2: each seat in turn from F
        else:
            self._start_auction()

    def _start_auction(self) -> None:
        dash = self._dash
        callers = dash.count(True)
        for seat in range(4):
            if dash[seat]:
                self._est[seat] = 0  # DSH-4: a fixed estimate of 0
        if callers == 4:
            self._skip()  # DSH-5
            return
        self._passed = dash.copy()  # DSH-4: Dash callers sit out the auction
        if callers == 3:
            self._max_bid = _LONE_MAX_BID  # AUC-5
        self.phase = AUCTION
        first = self.first
        for k in range(4):  # GAM-2: the opener is the first non-Dash seat from F
            seat = (first + k) & 3
            if not dash[seat]:
                self._cur = seat
                return

    def _step_auction(self, action: int) -> None:
        seat = self._cur
        top = self._top
        if FIRST_BID <= action <= self._max_bid:
            if action <= top:
                raise IllegalActionError(f"{_describe(action)} does not beat the top bid")
            self._top = action
            self._top_seat = seat
            self._with_bid[seat] = -1  # AUC-2: bidding drops your With
        elif action == PASS:
            if not self._final:
                self._passed[seat] = True
        elif action == WITH:
            if (
                self._final
                or self._top_seat < 0
                or self._top_seat == seat
                or self._with_bid[seat] == top
            ):
                raise IllegalActionError("With needs another player's top bid, joined only once")
            self._with_bid[seat] = top
        else:
            raise IllegalActionError(f"{_describe(action)} is not legal in the auction")
        self._log.append((seat, action))
        if self._final:
            self._end_auction()
        else:
            self._next_bidder(seat)

    def _next_bidder(self, seat: int) -> None:
        """AUC-1: the next seat counterclockwise that has not passed and is not the top bidder."""
        passed = self._passed
        top_seat = self._top_seat
        for k in (1, 2, 3, 4):
            seat_next = (seat + k) & 3
            if not passed[seat_next] and seat_next != top_seat:
                self._cur = seat_next
                return
        if top_seat < 0:
            self._skip()  # AUC-3, GAM-3: nobody bid
        else:
            self._final = True  # AUC-3: the top bidder's final turn
            self._cur = top_seat

    def _end_auction(self) -> None:
        caller = self._top_seat
        tricks, trump = divmod(self._top - FIRST_BID, 5)
        tricks += 4
        self._caller = caller
        self._c = tricks
        self._trump = trump
        with_w = self._with_w
        for seat, joined in enumerate(self._with_bid):
            if joined >= 0 and (joined - FIRST_BID) % 5 == trump:  # AUC-4: With follows the trump
                with_w[seat] = (joined - FIRST_BID) // 5 + 4
        self._est[caller] = tricks  # EST-1
        self._total += tricks
        dash = self._dash
        order = tuple(  # EST-2: from the seat after the Caller, skipping Dash callers
            seat
            for seat in ((caller + 1) & 3, (caller + 2) & 3, (caller + 3) & 3)
            if not dash[seat]
        )
        self._order = order
        self._pos = 0
        if order:
            self._last = order[-1]
            self.phase = ESTIMATION
            self._cur = order[0]
        else:
            self._last = -1  # EST-7: nobody estimates, so nobody takes Risk
            self._start_play()

    def _step_estimation(self, action: int) -> None:
        seat = self._cur
        k = action - ESTIMATE_BASE
        if self._caller < 0:  # fixed-trump estimation before any super call (FIX-2)
            if 0 <= k <= 13:
                if self._pos == 3 and k == 13 - self._total:
                    raise IllegalActionError(f"estimate {k} would make the total 13 (EST-5)")
                self._record_estimate(seat, k, action)
                if self._pos == 4:
                    self._end_fixed_estimation()
                else:
                    self._cur = self._order[self._pos]
            elif action in self._supers:
                self._open_super_call(seat, action)
            else:
                raise IllegalActionError(f"{_describe(action)} is not an estimate or super call")
            return
        c = self._c
        if not 0 <= k <= c:
            raise IllegalActionError(f"{_describe(action)} is not an estimate from 0 to {c}")
        w = self._with_w[seat]
        low = w if w >= 0 else 0  # EST-3
        if self._pos == len(self._order) - 1:
            forbidden = 13 - self._total
            if k == forbidden:
                raise IllegalActionError(f"estimate {k} would make the total 13 (EST-5)")
            if low == c == forbidden:
                low = 0  # EST-5: nothing from w to c is allowed, so the floor drops
        if k < low:
            raise IllegalActionError(f"a With player estimates {low} to {c} (EST-3)")
        if k == c and w < 0:
            self._with_w[seat] = c  # EST-4
        self._record_estimate(seat, k, action)
        if self._pos == len(self._order):
            self._start_play()
        else:
            self._cur = self._order[self._pos]

    def _record_estimate(self, seat: int, k: int, action: int) -> None:
        self._est[seat] = k
        self._total += k
        self._log.append((seat, action))
        self._pos += 1

    def _end_fixed_estimation(self) -> None:
        """FIX-3: the highest estimate is Caller (first said wins ties); equal ones are With."""
        est = self._est
        best = max(est)
        caller = next(seat for seat in self._order if est[seat] == best)
        self._caller = caller
        self._c = best
        for seat in self._order:
            if seat != caller and est[seat] == best:
                self._with_w[seat] = best
        self._start_play()

    def _open_super_call(self, seat: int, action: int) -> None:
        """FIX-4: clear every estimate and open an auction with the super call as top bid."""
        self._log.append((seat, action))
        self._super = True
        self._est = [-1, -1, -1, -1]
        self._total = 0
        self._pos = 0
        self._trump = -1
        self._top = action
        self._top_seat = seat
        self._passed = [False, False, False, False]
        self._with_bid = [-1, -1, -1, -1]
        self._final = False
        self.phase = AUCTION
        self._next_bidder(seat)  # starts with the player after the super caller

    def _start_play(self) -> None:
        self.phase = PLAY
        self._leader = self._caller  # PLY-1
        self._cur = self._caller

    def _end_trick(self) -> None:
        """PLY-3: highest trump wins, else highest of the led suit; PLY-4: winner leads."""
        c0, c1, c2, c3 = self._trick
        trump = self._trump
        best = c0
        best_suit = self._led
        offset = 0
        suit = c1 // 13
        if suit == best_suit:
            if c1 > best:
                best = c1
                offset = 1
        elif suit == trump:
            best = c1
            best_suit = suit
            offset = 1
        suit = c2 // 13
        if suit == best_suit:
            if c2 > best:
                best = c2
                offset = 2
        elif suit == trump:
            best = c2
            best_suit = suit
            offset = 2
        suit = c3 // 13
        if (suit == best_suit and c3 > best) or (suit != best_suit and suit == trump):
            offset = 3
        leader = self._leader
        winner = (leader + offset) & 3
        self._won[winner] += 1
        self._past.append((leader, (c0, c1, c2, c3), winner))
        self._trick = []
        self._leader = winner
        self._cur = winner
        if len(self._past) == 13:
            self._finish()  # PLY-5

    def _finish(self) -> None:
        est = self._est
        won = self._won
        dash = self._dash
        last = self._last
        outcome = RoundOutcome(
            estimates=(est[0], est[1], est[2], est[3]),
            tricks=(won[0], won[1], won[2], won[3]),
            caller=self._caller,
            dash_callers=tuple(seat for seat in range(4) if dash[seat]),
            with_players=tuple(seat for seat in range(4) if self._with_w[seat] >= 0),
            last_estimator=last if last >= 0 else None,
            risks=abs(self._total - 13) // 2 if last >= 0 else 0,  # EST-7
            carry=self.carry,
            round_no=self.round_no,
            trump=self._trump,
            fixed_trump=self.fixed_trump,
        )
        self._result = score_round(outcome, self.config)
        self.phase = DONE

    def _skip(self) -> None:
        """GAM-3: a round with no Caller scores 0 for everyone."""
        dash = self._dash
        outcome = RoundOutcome(
            skipped=True,
            dash_callers=tuple(seat for seat in range(4) if dash[seat]),
            carry=self.carry,
            round_no=self.round_no,
            fixed_trump=self.fixed_trump,
        )
        self._result = score_round(outcome, self.config)
        self.phase = DONE

    # Views and copies

    def observation(self, seat: int) -> Observation:
        """Everything ``seat`` may see, and nothing else."""
        if not 0 <= seat < 4:
            raise ValueError(f"no seat {seat}")
        phase = self.phase
        dash = self._dash
        estimates: tuple[int | None, int | None, int | None, int | None]
        if phase is DASH:  # DSH-2: only this seat's own Dash choice is visible
            log = tuple(entry for entry in self._log if entry[0] == seat)
            dash_callers: tuple[int, ...] = (seat,) if dash[seat] else ()
            estimates = (None, None, None, None)
        else:
            log = tuple(self._log)
            dash_callers = tuple(s for s in range(4) if dash[s])
            e0, e1, e2, e3 = self._est
            estimates = (
                e0 if e0 >= 0 else None,
                e1 if e1 >= 0 else None,
                e2 if e2 >= 0 else None,
                e3 if e3 >= 0 else None,
            )
        if phase is AUCTION:
            with_players = tuple(
                WithInfo(s, (b - FIRST_BID) // 5 + 4, (b - FIRST_BID) % 5)
                for s, b in enumerate(self._with_bid)
                if b >= 0
            )
        else:
            trump = self._trump
            with_players = tuple(
                WithInfo(s, w, trump) for s, w in enumerate(self._with_w) if w >= 0
            )
        caller = self._caller
        settled = caller >= 0 and (phase is PLAY or phase is DONE)
        last = self._last
        total = self._total if settled else None
        leader = self._leader
        won = self._won
        return Observation(
            seat=seat,
            round_no=self.round_no,
            phase=phase,
            current_player=self.current_player,
            carry=self.carry,
            scores=self.scores,
            fixed_trump=self.fixed_trump,
            fixed_left=self.fixed_left,
            replays=self.replays,
            super_call=self._super,
            trump=self._trump if self._trump >= 0 else None,
            hand=self._hands[seat],
            log=log,
            dash_callers=dash_callers,
            top_bid=self._top if self._top_seat >= 0 else None,
            top_bidder=self._top_seat if self._top_seat >= 0 else None,
            passed=tuple(s for s in range(4) if self._passed[s] and not dash[s]),
            caller=caller if caller >= 0 else None,
            with_players=with_players,
            estimates=estimates,
            last_estimator=last if last >= 0 and (phase is ESTIMATION or settled) else None,
            total=total,
            over=None if total is None else total > 13,
            risks=None if total is None else (abs(total - 13) // 2 if last >= 0 else 0),
            leader=leader if phase is PLAY else None,
            trick=tuple(((leader + i) & 3, card) for i, card in enumerate(self._trick)),
            tricks=tuple(TrickRecord(*trick) for trick in self._past),
            tricks_won=(won[0], won[1], won[2], won[3]),
        )

    def clone(self) -> Self:
        """An independent copy of this round in the same state."""
        rnd = object.__new__(type(self))
        rnd.round_no = self.round_no
        rnd.first = self.first
        rnd.fixed_trump = self.fixed_trump
        rnd.carry = self.carry
        rnd.scores = self.scores
        rnd.config = self.config
        rnd.fixed_left = self.fixed_left
        rnd.replays = self.replays
        rnd.hands = self.hands
        rnd.phase = self.phase
        rnd._cur = self._cur
        rnd._hands = self._hands.copy()
        rnd._log = self._log.copy()
        rnd._dash = self._dash.copy()
        rnd._chosen = self._chosen
        rnd._passed = self._passed.copy()
        rnd._top = self._top
        rnd._top_seat = self._top_seat
        rnd._with_bid = self._with_bid.copy()
        rnd._final = self._final
        rnd._max_bid = self._max_bid
        rnd._super = self._super
        rnd._supers = self._supers
        rnd._est = self._est.copy()
        rnd._total = self._total
        rnd._order = self._order
        rnd._pos = self._pos
        rnd._caller = self._caller
        rnd._c = self._c
        rnd._with_w = self._with_w.copy()
        rnd._trump = self._trump
        rnd._last = self._last
        rnd._leader = self._leader
        rnd._trick = self._trick.copy()
        rnd._led = self._led
        rnd._won = self._won.copy()
        rnd._past = self._past.copy()
        rnd._result = self._result
        return rnd
