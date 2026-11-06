"""Weighting sampled deals by the bidding (SPEC §6 M5): BiddingNet as a model of the others.

A sampled deal gives each other seat a dealt hand. For every Dash Call, bid, With, Pass, estimate
and super call that seat chose this round, BiddingNet values each legal option as if the seat held
that hand, and a softmax of those values (``temperature``, in the nets' units of 50 points) says
how likely the seat's actual choice was. Mixed with a uniform floor, so no deal is ruled out by a
bidder that does not bid like BiddingNet, the product over every choice of every other seat is
the deal's likelihood: deals where the bidding makes sense count more.

Everything but the hand in those decision positions is public, so each decision is encoded once
per round from ``Observation.log`` with an empty hand (``BiddingHistory``) and each deal only
writes the hand, the unseen cards and the hand's shape (nothing is played during bidding).

``PublicBidding`` is that public state, stepped by the rules without any seat's cards (bots never
touch a Round). It also finishes the bidding of an imagined round, for the bid search: once the
last estimate is made it moves to card play with the Caller leading (the highest estimate in a
fixed-trump round, FIX-3), with the total, Over or Under and Risks known, and a round nobody bid
in ends skipped (GAM-3).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from estimation.engine import Observation, Phase, RulesConfig, WithInfo, bid_action
from estimation.engine.actions import (
    DASH_CALL,
    ESTIMATE_BASE,
    FIRST_BID,
    LAST_BID,
    NO_DASH_CALL,
    PASS,
    PLAY_BASE,
    WITH,
)
from estimation.engine.cards import SANS
from estimation.rl.encoder import HAND, NUM_FEATURES, OFFSETS, SHAPE, encode

from .rollout import CARDS, SHAPE_SIZE, Floats, bits

UNSEEN = OFFSETS["unseen"]
BID_ACTIONS = PLAY_BASE
_LONE_MAX_BID = bid_action(12, SANS)  # AUC-5: a lone bidder may not bid 13 tricks

type BidNet = Callable[[Floats], Floats]
"""The value of each of the 68 non-card actions for a batch of positions (``Policy.bid_values``)."""


@dataclass(frozen=True, slots=True)
class BiddingHistory:
    """The other seats' choices this round with more than one legal option: who chose, the
    position encoded with an empty hand, the legal options and the choice."""

    seats: tuple[int, ...]
    templates: Floats  # (decisions, NUM_FEATURES)
    legal: npt.NDArray[np.bool_]  # (decisions, BID_ACTIONS)
    chosen: npt.NDArray[np.int64]  # (decisions,)

    def __len__(self) -> int:
        return len(self.seats)


def bidding_history(obs: Observation, rules: RulesConfig) -> BiddingHistory:
    """Rebuild this round's non-card decisions of every seat but ``obs.seat`` from ``obs.log``."""
    me = obs.seat
    state = PublicBidding(obs, rules)
    seats: list[int] = []
    rows: list[Floats] = []
    legal_rows: list[npt.NDArray[np.bool_]] = []
    chosen: list[int] = []
    for seat, action in obs.log:
        if action >= PLAY_BASE:
            break
        if seat != me:
            legal = state.legal_actions()
            if len(legal) > 1:
                seats.append(seat)
                rows.append(encode(state.observation(seat), rules))
                mask = np.zeros(BID_ACTIONS, np.bool_)
                mask[legal] = True
                legal_rows.append(mask)
                chosen.append(action)
        state.step(seat, action)
    if not rows:
        return BiddingHistory(
            (),
            np.zeros((0, NUM_FEATURES), np.float32),
            np.zeros((0, BID_ACTIONS), np.bool_),
            np.zeros(0, np.int64),
        )
    return BiddingHistory(
        tuple(seats), np.stack(rows), np.stack(legal_rows), np.asarray(chosen, np.int64)
    )


class PublicBidding:
    """The public state of a round before card play, stepped from ``Observation.log`` without
    any seat's cards. ``phase`` ends at PLAY, or DONE for a skipped round."""

    __slots__ = (
        "answers",
        "c",
        "caller",
        "calls",
        "chosen",
        "cur",
        "dash",
        "est",
        "final",
        "first",
        "hold",
        "joined",
        "last",
        "log",
        "max_bid",
        "obs",
        "order",
        "out",
        "phase",
        "pos",
        "rules",
        "skipped",
        "stayed",
        "super_call",
        "super_seat",
        "supers",
        "top",
        "top_seat",
        "total",
        "trump",
        "with_w",
    )

    def __init__(self, obs: Observation, rules: RulesConfig) -> None:
        self.obs = obs
        self.rules = rules
        first = (obs.round_no - 1) & 3
        self.first = first
        self.cur = first
        self.log: list[tuple[int, int]] = []
        self.dash = [False, False, False, False]
        self.chosen = 0
        self.out = [False, False, False, False]
        self.top = -1
        self.top_seat = -1
        self.hold = [-1, -1, -1, -1]
        self.calls = [-1] * 20  # each seat's last call in each trump (AUC-4)
        self.stayed = [-1, -1, -1, -1]  # the call each seat stayed at when answering (EST-2)
        self.joined = [-1, -1, -1, -1]
        self.final = False
        self.answers: list[int] = []
        self.max_bid = LAST_BID
        self.skipped = False
        self.super_call = False
        self.super_seat = -1
        self.est = [-1, -1, -1, -1]
        self.total = 0
        self.pos = 0
        self.caller = -1
        self.c = 0
        self.with_w = [-1, -1, -1, -1]
        if obs.fixed_trump is None:
            self.phase = Phase.DASH
            self.supers: tuple[int, ...] = ()
            self.order: tuple[int, ...] = ()
            self.trump = -1
            self.last = -1
        else:
            self.phase = Phase.ESTIMATION
            self.supers = tuple(
                FIRST_BID + (n - 4) * 5 + t
                for n in range(rules.super_call_min, 14)
                for t in range(5)
                if t != obs.fixed_trump
            )
            self.order = (first, (first + 1) & 3, (first + 2) & 3, (first + 3) & 3)
            self.trump = obs.fixed_trump
            self.last = self.order[3]

    def clone(self) -> "PublicBidding":
        """An independent copy in the same state."""
        twin = object.__new__(PublicBidding)
        for name in PublicBidding.__slots__:
            value = getattr(self, name)
            setattr(twin, name, value.copy() if isinstance(value, list) else value)
        return twin

    @property
    def bidding(self) -> bool:
        """Whether a Dash Call, bid or estimate is still to be made."""
        return self.phase is not Phase.PLAY and self.phase is not Phase.DONE

    def legal_actions(self) -> list[int]:
        if not self.bidding:
            return []
        if self.phase is Phase.DASH:
            return [NO_DASH_CALL, DASH_CALL]
        if self.phase is Phase.AUCTION:
            seat = self.cur
            bids = range(max(self.top + 1, FIRST_BID), self.max_bid + 1)
            if seat == self.super_seat and not self.rules.super_caller_may_bid_fixed:
                actions = [a for a in bids if (a - FIRST_BID) % 5 != self.obs.fixed_trump]
            else:
                actions = list(bids)
            if self.answers:
                if len(self.answers) > 1 or not self._ends_on_13(self.top, seat, WITH):
                    actions.append(WITH)
                if len(self.answers) > 1 or not self._ends_on_13(self.top, seat, PASS):
                    actions.append(PASS)
            elif self.final:
                if not self._answerers(seat):
                    actions = [a for a in actions if not self._ends_on_13(a)]
                if not self._ends_on_13(self.top):
                    actions.append(PASS)
            else:
                if self.top_seat >= 0:
                    actions.append(WITH)
                actions.append(PASS)
            return actions
        if self.caller < 0:
            actions = list(self.supers)
            forbidden = ESTIMATE_BASE + 13 - self.total if self.pos == 3 else -1
            actions.extend(a for a in range(ESTIMATE_BASE, PLAY_BASE) if a != forbidden)
            return actions
        forbidden = 13 - self.total if self.pos == len(self.order) - 1 else -1
        low = max(self.with_w[self.cur], 0)  # EST-2: a choosing With player's floor
        return [ESTIMATE_BASE + k for k in range(low, self.c + 1) if k != forbidden]

    def _answerers(self, bidder: int) -> list[int]:
        return [
            s
            for s in ((bidder + 1) & 3, (bidder + 2) & 3, (bidder + 3) & 3)
            if self.joined[s] == bidder
        ]

    def _call(self, seat: int, action: int) -> None:
        self.hold[seat] = action
        self.calls[seat * 5 + (action - FIRST_BID) % 5] = action

    def _with_call(self, seat: int, trump: int) -> int:
        if self.rules.with_any_call:
            return self.calls[seat * 5 + trump]
        held = self.hold[seat]
        return held if held >= 0 and (held - FIRST_BID) % 5 == trump else -1

    def _keeps(self, call: int, c: int, stayed: int) -> bool:
        if not self.rules.with_choose:
            return True
        return (call - FIRST_BID) // 5 + 4 == c or call == stayed

    def _ends_on_13(self, bid: int, seat: int = -1, answer: int = -1) -> bool:
        tricks, trump = divmod(bid - FIRST_BID, 5)
        c = tricks + 4
        total = c
        for s in range(4):
            if s == self.top_seat or self.dash[s]:
                continue
            if s == seat and answer == WITH:
                total += c
                continue
            call = self._with_call(s, trump)
            if call < 0:
                return False
            stayed = self.hold[s] if s == seat and answer == PASS else self.stayed[s]
            if not self._keeps(call, c, stayed):
                return False
            total += (call - FIRST_BID) // 5 + 4
        return total == 13

    def step(self, seat: int, action: int) -> None:
        self.log.append((seat, action))
        if self.phase is Phase.DASH:
            if action == DASH_CALL:
                self.dash[seat] = True
                self.est[seat] = 0
            self.chosen += 1
            if self.chosen < 4:
                self.cur = (seat + 1) & 3
            elif all(self.dash):
                self._skip()  # DSH-5: all four Dash Called
            else:
                self.out = self.dash.copy()
                if self.dash.count(True) == 3:
                    self.max_bid = _LONE_MAX_BID
                self.phase = Phase.AUCTION
                self.cur = next(
                    (self.first + k) & 3 for k in range(4) if not self.dash[(self.first + k) & 3]
                )
            return
        if self.phase is Phase.AUCTION:
            if self.answers:
                if FIRST_BID <= action <= self.max_bid:
                    self.answers = []
                    self._take_top(seat, action)
                    self._next_bidder(seat)
                    return
                if action == WITH:
                    self._call(seat, self.top)
                else:
                    self.stayed[seat] = self.hold[seat]
                    self.joined[seat] = -1
                self.answers.pop(0)
                if self.answers:
                    self.cur = self.answers[0]
                else:
                    self._end_auction()
                return
            if FIRST_BID <= action <= self.max_bid:
                if self.final:
                    answerers = self._answerers(seat)
                    self.top = action
                    self._call(seat, action)
                    if answerers:
                        self.final = False
                        self.answers = answerers
                        self.cur = answerers[0]
                    else:
                        self._end_auction()
                    return
                self._take_top(seat, action)
            elif action == PASS:
                if self.final:
                    self._end_auction()
                    return
                self.out[seat] = True
            else:  # WITH
                self._call(seat, self.top)
                self.joined[seat] = self.top_seat
                self.out[seat] = True
            self._next_bidder(seat)
            return
        k = action - ESTIMATE_BASE
        if self.caller < 0 and not 0 <= k <= 13:  # FIX-4: super call opens an auction
            self.super_call = True
            self.super_seat = seat
            self.est = [-1, -1, -1, -1]
            self.total = 0
            self.pos = 0
            self.trump = -1
            self.top = action
            self.top_seat = seat
            self.out = [False, False, False, False]
            self.hold = [-1, -1, -1, -1]
            self.calls = [-1] * 20
            self.stayed = [-1, -1, -1, -1]
            self._call(seat, action)
            self.joined = [-1, -1, -1, -1]
            self.final = False
            self.answers = []
            self.phase = Phase.AUCTION
            self._next_bidder(seat)
            return
        if self.caller >= 0 and (k == self.c or self.with_w[seat] >= 0):
            self.with_w[seat] = k  # EST-4, or a choosing With player's pick (EST-2)
        self.est[seat] = k
        self.total += k
        self.pos += 1
        if self.pos < len(self.order):
            self.cur = self.order[self.pos]
        elif self.caller < 0:
            self._end_fixed_estimation()
        else:
            self._start_play()

    def _end_fixed_estimation(self) -> None:
        """FIX-3: the highest estimate is Caller (first said wins ties); equal ones are With."""
        best = max(self.est)
        caller = next(seat for seat in self.order if self.est[seat] == best)
        self.caller = caller
        self.c = best
        for seat in self.order:
            if seat != caller and self.est[seat] == best:
                self.with_w[seat] = best
        self._start_play()

    def _start_play(self) -> None:
        self.phase = Phase.PLAY  # PLY-1: the Caller leads
        self.cur = self.caller

    def _skip(self) -> None:
        self.phase = Phase.DONE  # GAM-3
        self.skipped = True
        self.cur = -1

    def _take_top(self, seat: int, action: int) -> None:
        old = self.top_seat
        if old >= 0:
            for s in range(4):
                if self.joined[s] == old:
                    self.joined[s] = -1
        self.top = action
        self.top_seat = seat
        self._call(seat, action)
        self.joined[seat] = -1
        self.out[seat] = False

    def _next_bidder(self, seat: int) -> None:
        for k in (1, 2, 3, 4):
            nxt = (seat + k) & 3
            if not self.out[nxt] and nxt != self.top_seat:
                self.cur = nxt
                return
        if self.top_seat < 0:
            self._skip()  # AUC-3, GAM-3: nobody bid
            return
        self.final = True
        self.cur = self.top_seat

    def _end_auction(self) -> None:
        caller = self.top_seat
        tricks, trump = divmod(self.top - FIRST_BID, 5)
        tricks += 4
        self.caller = caller
        self.c = tricks
        self.trump = trump
        self.final = False
        after = ((caller + 1) & 3, (caller + 2) & 3, (caller + 3) & 3)
        choosers: list[int] = []
        for s in after:
            if self.dash[s]:
                continue
            call = self._with_call(s, trump)
            if call < 0:
                continue
            w = (call - FIRST_BID) // 5 + 4
            self.with_w[s] = w
            if self._keeps(call, tricks, self.stayed[s]):
                self.est[s] = w
                self.total += w
            else:
                choosers.append(s)
        self.est[caller] = tricks
        self.total += tricks
        self.order = (
            *choosers,
            *(s for s in after if not self.dash[s] and self.with_w[s] < 0),
        )
        self.pos = 0
        if self.order:
            self.last = self.order[-1]
            self.phase = Phase.ESTIMATION
            self.cur = self.order[0]
        else:
            self.last = -1  # EST-7: nobody estimates, so nobody takes Risk
            self._start_play()

    def observation(self, seat: int) -> Observation:
        if self.phase is Phase.AUCTION:
            top_trump = (self.top - FIRST_BID) % 5
            calls = (
                (s, self._with_call(s, top_trump))
                for s in range(4)
                if self.top_seat >= 0 and s != self.top_seat and self.out[s] and not self.dash[s]
            )
            with_players = tuple(
                WithInfo(s, (call - FIRST_BID) // 5 + 4, top_trump)
                for s, call in calls
                if call >= 0
            )
        else:
            with_players = tuple(
                WithInfo(s, w, self.trump) for s, w in enumerate(self.with_w) if w >= 0
            )
        e0, e1, e2, e3 = self.est
        settled = self.caller >= 0 and not self.bidding
        total = self.total if settled else None
        last = self.last
        return replace(
            self.obs,
            seat=seat,
            phase=self.phase,
            current_player=self.cur if self.cur >= 0 else None,
            super_call=self.super_call,
            trump=self.trump if self.trump >= 0 else None,
            hand=0,
            log=tuple(self.log),
            dash_callers=tuple(s for s in range(4) if self.dash[s]),
            top_bid=self.top if self.top_seat >= 0 else None,
            top_bidder=self.top_seat if self.top_seat >= 0 else None,
            passed=tuple(
                s
                for s in range(4)
                if self.out[s] and not self.dash[s] and all(w.seat != s for w in with_players)
            ),
            caller=self.caller if self.caller >= 0 else None,
            with_players=with_players,
            estimates=(
                e0 if e0 >= 0 else None,
                e1 if e1 >= 0 else None,
                e2 if e2 >= 0 else None,
                e3 if e3 >= 0 else None,
            ),
            last_estimator=last
            if last >= 0 and (self.phase is Phase.ESTIMATION or settled)
            else None,
            total=total,
            over=None if total is None else total > 13,
            risks=None if total is None else (abs(total - 13) // 2 if last >= 0 else 0),
            leader=self.caller if self.phase is Phase.PLAY else None,
            trick=(),
            tricks=(),
            tricks_won=(0, 0, 0, 0),
        )


def log_likelihoods(
    history: BiddingHistory,
    dealt: npt.NDArray[np.uint64],
    shapes: Floats,
    bid_values: BidNet,
    temperature: float,
    floor: float,
) -> npt.NDArray[np.float64]:
    """Each deal's log likelihood of the other seats' choices, ``(D,)``.

    ``dealt[d, seat]`` is the hand ``seat`` was dealt in deal ``d`` and ``shapes[d, seat]`` its
    shape block. ``floor`` is the share of each choice's probability spread evenly over its legal
    options."""
    deals = dealt.shape[0]
    n = len(history)
    if n == 0 or deals == 0:
        return np.zeros(deals, np.float64)
    seats = np.asarray(history.seats, np.int64)
    x = np.repeat(history.templates[np.newaxis], deals, axis=0)  # (D, n, F)
    hands = dealt[:, seats]  # (D, n)
    hand_bits = bits(hands).astype(np.float32)
    x[:, :, HAND : HAND + CARDS] = hand_bits
    x[:, :, UNSEEN : UNSEEN + CARDS] = 1.0 - hand_bits
    x[:, :, SHAPE : SHAPE + SHAPE_SIZE] = shapes[:, seats]
    values = bid_values(x.reshape(deals * n, NUM_FEATURES)).astype(np.float64)
    values = values.reshape(deals, n, BID_ACTIONS) / temperature
    legal = history.legal[np.newaxis]
    values = np.where(legal, values, -np.inf)
    top = values.max(axis=2, keepdims=True)
    exp = np.exp(values - top)
    soft = exp / exp.sum(axis=2, keepdims=True)
    picked = np.take_along_axis(
        soft, np.broadcast_to(history.chosen[np.newaxis, :, np.newaxis], (deals, n, 1)), axis=2
    )[:, :, 0]
    options = history.legal.sum(axis=1)[np.newaxis]  # (1, n)
    probability = (1.0 - floor) * picked + floor / options
    return np.log(probability).sum(axis=1)
