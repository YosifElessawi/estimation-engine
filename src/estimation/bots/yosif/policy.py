"""Yosif's Dash Calls, bids and estimates (design 5.2-5.8).

Every choice is the number with the best expected points by SPEC §3 [Yosif]: taking too many
tricks is as bad as taking too few, so a hand with no low cards bids up to what it cannot avoid.
When the scoresheet says he needs a jump to win the game, he goes for the call that pays most if
it is made, as long as it can still be made [Yosif]. In the auction he starts low to gather
information and raises on his final turn [Yosif].
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from estimation.engine import (
    DASH_CALL,
    NO_DASH_CALL,
    PASS,
    SANS,
    WITH,
    Observation,
    RulesConfig,
    bid_action,
    bid_of,
    estimate_action,
)
from estimation.engine.actions import ESTIMATE_BASE, is_bid, is_estimate
from estimation.engine.config import DEFAULT_CONFIG

from ..heuristic import STRONGEST_FIRST, lowest_bid
from ..pimc import PIMCConfig
from .counting import DEFAULT_COUNT, CountValues, Reads, count_hand, read_bids
from .odds import Outlook, Stake, expected_points, make_chance, outlook, stake

TRUMPS = (0, 1, 2, 3, SANS)
MIN_BID, MAX_TRICKS = 4, 13


@dataclass(frozen=True, slots=True)
class YosifConfig:
    """Yosif's bot. Rules marked [Yosif] are his; numbers marked [Claude] are starting values."""

    count: CountValues = DEFAULT_COUNT
    others_make: float = 0.5  # chance each other player makes their estimate (SCR-5) [Claude]
    dash_break: float = 0.6  # opponents try to break a Dash Call [Yosif]; the size [Claude]
    risk_mode: bool = True  # go for the bigger call when he needs a jump to win [Yosif]
    jump_points: float = 25.0  # about one good normal round, per round left [Claude]
    risk_min_make: float = 0.25  # a bigger call must still be makeable [Claude]
    rules: RulesConfig = DEFAULT_CONFIG  # the rules the stakes are priced by
    pimc: PIMCConfig = field(default_factory=lambda: PIMCConfig(leader_weight=True))


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

    def _outlook(self, hand: int, trump: int, reads: Reads, over: bool | None = None) -> Outlook:
        return outlook(count_hand(hand, trump, reads, over, self.config.count))

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

    def dash(self, obs: Observation, legal: list[int]) -> int:
        """Dash Call when it is worth more than the hand's best alternative (design 5.5)."""
        if DASH_CALL not in legal:
            return NO_DASH_CALL
        c = self.config
        reads = read_bids(obs)
        outlooks = {t: self._outlook(obs.hand, t, reads) for t in TRUMPS}
        best = max(TRUMPS, key=lambda t: sum(k * p for k, p in enumerate(outlooks[t].most)))
        o = outlooks[best]
        s = stake(0, caller=False, dash=True, last_risks=0, over=None, rules=c.rules)
        zero = c.dash_break * o.least[0]
        bonus = c.rules.only_bonus if s.solo else 0
        dash_value = zero * (s.win + bonus * (1 - c.others_make) ** 3) - (1 - zero) * (
            s.lose + bonus * c.others_make**3
        )
        caller = max(
            self._caller_value(outlooks[t], range(MIN_BID, MAX_TRICKS + 1), False)[1]
            for t in TRUMPS
        )
        alternative = max(caller, self._plain_value(obs.hand, reads))
        return DASH_CALL if dash_value > alternative else NO_DASH_CALL

    def bid(self, obs: Observation, legal: list[int]) -> int:
        """Start low, raise to the target on the final turn, else With or Pass (design 5.4)."""
        me = obs.seat
        reads = read_bids(obs)
        risk = self._risk(obs)
        lone = sum(1 for s in range(4) if s not in obs.dash_callers) == 1
        options = range(MIN_BID, (MAX_TRICKS - 1 if lone else MAX_TRICKS) + 1)
        targets: dict[int, tuple[int, float]] = {
            t: self._caller_value(self._outlook(obs.hand, t, reads), options, risk) for t in TRUMPS
        }
        top = obs.top_bid
        if top is not None and obs.top_bidder == me:  # AUC-3: the final turn
            tricks, trump = bid_of(top)
            raised_to = targets[trump][0]
            if raised_to > tricks and bid_action(raised_to, trump) in legal:
                return bid_action(raised_to, trump)
            return PASS
        if top is not None and any(
            w.seat == me and bid_action(w.w, w.trump) == top for w in obs.with_players
        ):
            return PASS  # already With on the top bid: passing keeps it (AUC-2)
        plain = self._plain_value(obs.hand, reads)
        best, best_value = -1, plain
        for trump in STRONGEST_FIRST:
            target, value = targets[trump]
            tricks = lowest_bid(top, trump)
            if tricks <= target and value > best_value and bid_action(tricks, trump) in legal:
                best, best_value = bid_action(tricks, trump), value
        if best >= 0:
            return best
        if top is not None and WITH in legal:
            # With means estimating at least the top bid's number, scored like the Caller
            # (AUC-2, EST-3): say it only when that beats estimating normally in its trump.
            tricks, trump = bid_of(top)
            o = self._outlook(obs.hand, trump, reads)
            with_value = self._caller_value(o, range(tricks, MAX_TRICKS + 1), risk)[1]
            if with_value > self._plain_trump_value(o):
                return WITH
        return PASS

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
