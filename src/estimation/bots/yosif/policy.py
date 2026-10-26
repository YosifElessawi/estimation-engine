"""Yosif's Dash Calls, bids and estimates (design 5.2-5.8).

Every choice is the number with the best expected points by SPEC §3 [Yosif]: taking too many
tricks is as bad as taking too few, so a hand with no low cards bids up to what it cannot avoid.
When the scoresheet says he needs a jump to win the game, he goes for the call that pays most if
it is made, as long as it can still be made [Yosif]. In the auction he starts low to gather
information and raises on his final turn [Yosif].
"""

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from estimation.engine import (
    DASH_CALL,
    NO_DASH_CALL,
    PASS,
    SANS,
    WITH,
    Observation,
    Phase,
    RulesConfig,
    bid_action,
    bid_of,
    estimate_action,
)
from estimation.engine.actions import ESTIMATE_BASE, is_bid, is_estimate
from estimation.engine.config import DEFAULT_CONFIG

from ..heuristic import STRONGEST_FIRST, lowest_bid
from ..pimc import PIMCConfig
from .calibration import Calibration, load_calibration
from .counting import DEFAULT_COUNT, CountValues, Reads, count_hand, read_bids
from .odds import Calibrated, Outlook, Stake, expected_points, make_chance, outlook, stake

TRUMPS = (0, 1, 2, 3, SANS)
MIN_BID, MAX_TRICKS = 4, 13


@dataclass(frozen=True, slots=True)
class YosifConfig:
    """Yosif's bot. Rules marked [Yosif] are his; numbers marked [Claude] are starting values."""

    count: CountValues = DEFAULT_COUNT
    others_make: float = 0.5  # chance each other player makes their estimate (SCR-5) [Claude]
    dash_break: float = 0.6  # share of each dodge left when opponents try to break a Dash
    # Call [Yosif]; the size [Claude]
    control: float = 0.25  # how reliably dodging lands on a lower number; low because Yosif bids
    # what his hand takes and makes most of his calls [Yosif]; the size [Claude]
    risk_mode: bool = True  # go for the bigger call when he needs a jump to win [Yosif]
    jump_points: float = 25.0  # about one good normal round, per round left [Claude]
    risk_min_make: float = 0.25  # a bigger call must still be makeable [Claude]
    rules: RulesConfig = DEFAULT_CONFIG  # the rules the stakes are priced by
    pimc: PIMCConfig = field(default_factory=lambda: PIMCConfig(leader_weight=True))
    calibrated: bool = False  # price calls by the fit to saved games (``calibration``); off
    # until a play-out measurement backs it: the saved-games fit over-reaches (2026-10-27)
    calibration: Calibration | None = field(default_factory=load_calibration)


def best_number(
    options: Sequence[int],
    outlook_of: Callable[[int], Outlook],
    stake_of: Callable[[int], Stake],
    others_make: float,
    rules: RulesConfig,
    risk: bool = False,
    risk_min_make: float = 0.25,
) -> int:
    """The option with the best expected points; with ``risk``, the one paying most when made
    among those made at least ``risk_min_make`` of the time. Ties go to the lower number."""
    return max(
        number_values(
            options, outlook_of, stake_of, others_make, rules, risk, risk_min_make
        ).items(),
        key=lambda item: (item[1], -item[0]),
    )[0]


def number_values(
    options: Sequence[int],
    outlook_of: Callable[[int], Outlook],
    stake_of: Callable[[int], Stake],
    others_make: float,
    rules: RulesConfig,
    risk: bool = False,
    risk_min_make: float = 0.25,
) -> dict[int, float]:
    """Each option's value: expected points, or in risk mode the points when made for options
    made often enough (falling back to expected points when none is)."""
    if not options:
        raise ValueError("no number to choose from")
    if risk:
        bold = {
            k: make_chance(outlook_of(k), k) * stake_of(k).win
            for k in options
            if make_chance(outlook_of(k), k) >= risk_min_make
        }
        if bold:
            return bold
    return {k: expected_points(outlook_of(k), k, stake_of(k), others_make, rules) for k in options}


def needs_jump(obs: Observation, jump_points: float) -> bool:
    """True when the gap to the leader is more than ``jump_points`` per round left."""
    open_left = 14 - obs.round_no if obs.fixed_trump is None else 0
    rounds_left = max(open_left, 0) + len(obs.fixed_left)
    me = obs.seat
    return max(obs.scores) - obs.scores[me] > jump_points * rounds_left


class Planner:
    """Chooses every non-card action from the bot's own Observation."""

    __slots__ = ("config",)

    def __init__(self, config: YosifConfig) -> None:
        self.config = config

    def _outlook(
        self,
        hand: int,
        trump: int,
        reads: Reads,
        over: bool | None = None,
        dodge_scale: float = 1.0,
        control: float | None = None,
        calibrate: bool = True,
    ) -> Outlook:
        c = self.config
        steer = c.control if control is None else control
        o = outlook(count_hand(hand, trump, reads, over, c.count), dodge_scale, steer)
        fit = self._fit()
        if fit is None or not calibrate:
            return o
        natural = fit.natural(
            em=sum(k * p for k, p in enumerate(o.most)),
            el=sum(k * p for k, p in enumerate(o.least)),
            trumps=0 if trump == SANS else ((hand >> (13 * trump)) & 0x1FFF).bit_count(),
            sans=trump == SANS,
            fixed=False,
            last_over=0 if over is None else 1 if over else -1,
        )
        return dataclasses.replace(o, calibrated=Calibrated(natural, fit))

    def counting_outlook(
        self, hand: int, trump: int, reads: Reads, over: bool | None = None
    ) -> Outlook:
        """The counting's own outlook, before any calibration (what the fit learns from)."""
        return self._outlook(hand, trump, reads, over, calibrate=False)

    def _fit(self) -> Calibration | None:
        """The saved-games calibration when the bot uses it."""
        return self.config.calibration if self.config.calibrated else None

    def dash_zero_chance(self, hand: int, reads: Reads) -> float:
        """The counting's chance of taking no trick as a Dash caller, under the opponents'
        pressure, averaged over the trumps it is likely to face."""
        outlooks = {t: self._outlook(hand, t, reads, calibrate=False) for t in TRUMPS}
        pressed = self._dash_outlooks(hand, reads, outlooks)
        return sum(o.least[0] for o in pressed) / len(pressed)

    def _chance_points(self, chance: float, s: Stake) -> float:
        """Expected points of a call made with ``chance``, SCR-5 included (``expected_points``
        without the per-trick part)."""
        c = self.config
        bonus = c.rules.only_bonus if s.solo else 0
        return chance * (s.win + bonus * (1 - c.others_make) ** 3) - (1 - chance) * (
            s.lose + bonus * c.others_make**3
        )

    def _risk(self, obs: Observation) -> bool:
        return self.config.risk_mode and needs_jump(obs, self.config.jump_points)

    def _caller_value(self, o: Outlook, options: Sequence[int], risk: bool) -> tuple[int, float]:
        """The best number as Caller among ``options``, and its value."""
        c = self.config

        def stake_of(k: int) -> Stake:
            return stake(k, caller=True, dash=False, last_risks=0, over=None, rules=c.rules)

        values = number_values(
            options, lambda k: o, stake_of, c.others_make, c.rules, risk, c.risk_min_make
        )
        k = max(values.items(), key=lambda item: (item[1], -item[0]))[0]
        return k, values[k]

    def _plain_trump_value(self, o: Outlook) -> float:
        """The best expected points as an ordinary estimator under one trump."""
        c = self.config
        return max(
            expected_points(
                o,
                k,
                stake(k, caller=False, dash=False, last_risks=0, over=None, rules=c.rules),
                c.others_make,
                c.rules,
            )
            for k in range(MAX_TRICKS + 1)
        )

    def _plain_value(self, hand: int, reads: Reads) -> float:
        """What the hand expects as an ordinary estimator, averaged over the five trumps."""
        total = sum(self._plain_trump_value(self._outlook(hand, t, reads)) for t in TRUMPS)
        return total / len(TRUMPS)

    def _dash_outlooks(
        self, hand: int, reads: Reads, outlooks: dict[int, Outlook]
    ) -> list[Outlook]:
        """The hand's outlooks under the trumps a Dash caller is likely to face: every trump but
        its own best suit, which rarely becomes trump. The opponents press every dodge, and
        dodging is the whole of a Dash Call, so it is not discounted by ``control``."""
        c = self.config
        best = max(TRUMPS, key=lambda t: sum(k * p for k, p in enumerate(outlooks[t].most)))
        return [
            self._outlook(hand, t, reads, dodge_scale=c.dash_break, control=1.0, calibrate=False)
            for t in TRUMPS
            if t != best
        ]

    def predicted_make(self, obs: Observation, action: int) -> float | None:
        """The chance of taking exactly the number ``action`` commits to: the contract when
        keeping or raising on the final turn, the top bid's number for With, an estimate's
        number, a super call's number, and 0 for a Dash Call. None for other moves."""
        reads = read_bids(obs)
        top = obs.top_bid
        if obs.phase is Phase.DASH:
            if action != DASH_CALL:
                return None
            zero = self.dash_zero_chance(obs.hand, reads)
            fit = self._fit()
            return fit.dash(zero) if fit is not None else zero
        if obs.phase is Phase.AUCTION:
            if (action == WITH and top is not None) or (
                obs.top_bidder == obs.seat and top is not None and action == PASS
            ):
                committed = top
            elif obs.top_bidder == obs.seat and is_bid(action):
                committed = action
            else:
                return None
            tricks, trump = bid_of(committed)
            return make_chance(self._outlook(obs.hand, trump, reads), tricks)
        if obs.phase is not Phase.ESTIMATION:
            return None
        if is_bid(action):  # a super call (FIX-4)
            tricks, trump = bid_of(action)
            return make_chance(self._outlook(obs.hand, trump, reads), tricks)
        if not is_estimate(action):
            return None
        k = action - ESTIMATE_BASE
        trump = obs.trump if obs.trump is not None else obs.fixed_trump
        if trump is None:
            return None
        before = sum(e for e in obs.estimates if e is not None)
        over = before + k > 13 if obs.last_estimator == obs.seat else None
        return make_chance(self._outlook(obs.hand, trump, reads, over), k)

    def dash(self, obs: Observation, legal: list[int]) -> int:
        """Dash Call when it is worth more than the hand's best alternative (design 5.5)."""
        if DASH_CALL not in legal:
            return NO_DASH_CALL
        c = self.config
        reads = read_bids(obs)
        outlooks = {t: self._outlook(obs.hand, t, reads) for t in TRUMPS}
        s = stake(0, caller=False, dash=True, last_risks=0, over=None, rules=c.rules)
        fit = self._fit()
        if fit is None:
            pressed = self._dash_outlooks(obs.hand, reads, outlooks)
            dash_value = sum(
                expected_points(o, 0, s, c.others_make, c.rules) for o in pressed
            ) / len(pressed)
        else:
            dash_value = self._chance_points(fit.dash(self.dash_zero_chance(obs.hand, reads)), s)
        caller = max(
            self._caller_value(outlooks[t], range(MIN_BID, MAX_TRICKS + 1), False)[1]
            for t in TRUMPS
        )
        alternative = max(caller, self._plain_value(obs.hand, reads))
        return DASH_CALL if dash_value > alternative else NO_DASH_CALL

    def _points(self, o: Outlook, k: int, caller: bool) -> float:
        c = self.config
        s = stake(k, caller=caller, dash=False, last_risks=0, over=None, rules=c.rules)
        return expected_points(o, k, s, c.others_make, c.rules)

    def with_value(self, o: Outlook, tricks: int) -> float:
        """With fixes the estimate at the top bid's number, scored like the Caller (AUC-2,
        EST-2), so it is worth exactly that number's Caller points."""
        return self._points(o, tricks, caller=True)

    def bid(self, obs: Observation, legal: list[int]) -> int:
        """Every legal auction choice priced in expected points, the best one taken (design 5.4).

        On a normal turn a bid starts low, in a trump worth being Caller in, and is valued at the
        number it will be raised to on the final turn [Yosif]. With fixes the estimate at the top
        bid's number. Passing holds the bot's last bid: if that bid's trump wins, it is With at
        that number (AUC-4). On the final turn it keeps or raises; answering a raise it joins the
        new number, stays at its own, or outbids (AUC-3).
        """
        me = obs.seat
        reads = read_bids(obs)
        risk = self._risk(obs)
        lone = sum(1 for s in range(4) if s not in obs.dash_callers) == 1
        options = range(MIN_BID, (MAX_TRICKS - 1 if lone else MAX_TRICKS) + 1)
        outlooks = {t: self._outlook(obs.hand, t, reads) for t in TRUMPS}

        def caller(action: int) -> float:
            tricks, trump = bid_of(action)
            return self._points(outlooks[trump], tricks, caller=True)

        def holding(held: tuple[int, int] | None, trump: int) -> float:
            """Passing: With at the held number if its trump wins, else an ordinary estimator."""
            if held is not None and held[1] == trump:
                return self._points(outlooks[trump], held[0], caller=True)
            return self._plain_trump_value(outlooks[trump])

        values: dict[int, float] = {}
        top = obs.top_bid
        bids = [action for action in legal if is_bid(action)]
        if top is not None and obs.top_bidder == me:  # AUC-3: the final turn
            values[PASS] = caller(top)
            values.update((action, caller(action)) for action in bids)
        elif top is not None and self._out(obs):  # AUC-3: answering the top bidder's raise
            trump = bid_of(top)[1]
            values[WITH] = caller(top)
            mine = next(((w.w, w.trump) for w in obs.with_players if w.seat == me), None)
            values[PASS] = holding(mine, trump)
            values.update((action, caller(action)) for action in bids)
        else:
            for trump in STRONGEST_FIRST:
                target = self._caller_value(outlooks[trump], options, risk)[0]
                tricks = lowest_bid(top, trump)
                if tricks <= target:
                    values[bid_action(tricks, trump)] = self._points(
                        outlooks[trump], target, caller=True
                    )
            if top is None:
                values[PASS] = self._plain_value(obs.hand, reads)
            else:
                values[WITH] = caller(top)
                values[PASS] = holding(self._last_bid(obs), bid_of(top)[1])
        choices = {action: value for action, value in values.items() if action in legal}
        if not choices:  # nothing priced is allowed (AUC-6): price every legal bid
            choices = {action: caller(action) for action in bids}
        return max(choices.items(), key=lambda item: (item[1], item[0] == PASS))[0]

    @staticmethod
    def _out(obs: Observation) -> bool:
        """Out of the auction (passed or With): such a player acts only to answer a raise."""
        me = obs.seat
        return me in obs.passed or any(w.seat == me for w in obs.with_players)

    @staticmethod
    def _last_bid(obs: Observation) -> tuple[int, int] | None:
        """The bot's latest bid this round, as (tricks, trump): what passing holds (AUC-2)."""
        for seat, action in reversed(obs.log):
            if seat == obs.seat and is_bid(action):
                return bid_of(action)
        return None

    def estimate_values(self, obs: Observation, legal: list[int]) -> dict[int, float]:
        """The value of each legal estimate (design 5.6-5.8)."""
        c = self.config
        me = obs.seat
        trump = obs.trump if obs.trump is not None else obs.fixed_trump
        if trump is None:
            raise ValueError("the trump is known before anyone estimates")
        reads = read_bids(obs)
        options = [action - ESTIMATE_BASE for action in legal if is_estimate(action)]
        before = sum(e for e in obs.estimates if e is not None)
        last = obs.last_estimator == me
        with_me = any(w.seat == me for w in obs.with_players)
        contract = bid_of(obs.top_bid)[0] if obs.caller is not None and obs.top_bid else None
        highest = max(
            (e for s, e in enumerate(obs.estimates) if s != me and e is not None), default=0
        )
        outlooks = {
            over: self._outlook(obs.hand, trump, reads, over) for over in (None, True, False)
        }

        def over_of(k: int) -> bool | None:
            return before + k > 13 if last else None

        def is_caller(k: int) -> bool:
            # With a Caller: its number, or a With player (EST-3, EST-4). In a fixed-trump round
            # without one, the highest estimate becomes Caller and ties are With (FIX-3).
            return (with_me or k == contract) if contract is not None else k > 0 and highest <= k

        def stake_of(k: int) -> Stake:
            caller = is_caller(k)
            risks = abs(before + k - 13) // 2 if last else 0
            return stake(
                k, caller=caller, dash=False, last_risks=risks, over=over_of(k), rules=c.rules
            )

        return number_values(
            options,
            lambda k: outlooks[over_of(k)],
            stake_of,
            c.others_make,
            c.rules,
            self._risk(obs),
            c.risk_min_make,
        )

    def estimate(self, obs: Observation, legal: list[int]) -> int:
        """The best estimate, or a super call when another trump is worth more (FIX-4)."""
        values = self.estimate_values(obs, legal)
        k, value = max(values.items(), key=lambda item: (item[1], -item[0]))
        supers: dict[int, list[int]] = {}
        for action in legal:
            if is_bid(action):
                tricks, trump = bid_of(action)
                supers.setdefault(trump, []).append(tricks)
        best_call, best_value = -1, value
        reads = read_bids(obs)
        risk = self._risk(obs)
        for trump, sizes in supers.items():
            tricks, call_value = self._caller_value(
                self._outlook(obs.hand, trump, reads), sizes, risk
            )
            if call_value > best_value:
                best_call, best_value = bid_action(tricks, trump), call_value
        return best_call if best_call >= 0 else estimate_action(k)
