"""Round scoring (SPEC §3) and the carry between rounds (GAM-4).

``score_round`` and ``next_carry`` are pure functions, tested directly by the golden rounds.
"""

from dataclasses import dataclass

from .config import DEFAULT_CONFIG, RulesConfig

Scores = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundOutcome:
    """What happened in a round, as ``score_round`` needs it.

    ``estimates`` are the final estimates (Dash callers 0, the Caller ``c``). ``with_players``
    are the seats scored like the Caller. ``risks`` belong to ``last_estimator``. ``round_no``,
    ``trump`` and ``fixed_trump`` are carried into the result; a fixed-trump round whose final
    trump differs from its fixed trump changed trump by a super call (FIX-5).
    """

    estimates: Scores = (0, 0, 0, 0)
    tricks: Scores = (0, 0, 0, 0)
    caller: int | None = None
    dash_callers: tuple[int, ...] = ()
    with_players: tuple[int, ...] = ()
    last_estimator: int | None = None
    risks: int = 0
    carry: int = 1
    skipped: bool = False
    round_no: int = 0
    trump: int | None = None
    fixed_trump: int | None = None

    def __post_init__(self) -> None:
        if len(self.estimates) != 4 or len(self.tricks) != 4:
            raise ValueError("estimates and tricks must have one entry per seat")
        if self.carry < 1:
            raise ValueError("carry must be at least 1")
        if self.skipped:
            return
        if self.caller is None or not 0 <= self.caller < 4:
            raise ValueError("a played round needs a Caller seat")
        if any(not 0 <= n <= 13 for n in (*self.estimates, *self.tricks)):
            raise ValueError("estimates and tricks must be 0 to 13")
        if sum(self.tricks) != 13:
            raise ValueError("tricks must total 13")
        if sum(self.estimates) == 13:
            raise ValueError("estimates can never total 13 (EST-5)")
        if self.caller in self.dash_callers or self.caller in self.with_players:
            raise ValueError("the Caller must not be a Dash caller or a With player")
        if any(seat in self.dash_callers for seat in self.with_players):
            raise ValueError("a Dash caller must not be With (DSH-4)")
        if any(self.estimates[seat] for seat in self.dash_callers):
            raise ValueError("a Dash caller's estimate must be 0 (DSH-4)")
        if self.risks < 0 or (self.risks and self.last_estimator is None):
            raise ValueError("risks must be 0 or more and need a last estimator")


@dataclass(frozen=True, slots=True)
class SeatResult:
    """One seat's round: roles, the points from SCR-1 to SCR-6, the multiplier and the score."""

    estimate: int
    tricks: int
    won: bool
    is_dash: bool
    is_caller: bool
    is_with: bool
    is_super: bool
    is_last: bool
    risks: int
    base_points: int  # SCR-1
    dash_points: int  # SCR-2
    call_points: int  # SCR-3
    risk_points: int  # SCR-4
    only_points: int  # SCR-5
    super_points: int  # SCR-6
    multiplier: int
    score: int

    @property
    def points(self) -> int:
        """The SCR-1 to SCR-6 points before multipliers; ``points * multiplier == score``."""
        return (
            self.base_points
            + self.dash_points
            + self.call_points
            + self.risk_points
            + self.only_points
            + self.super_points
        )


@dataclass(frozen=True, slots=True)
class RoundResult:
    """A scored round. ``total`` and ``over`` are ``None`` for a skipped round."""

    seats: tuple[SeatResult, SeatResult, SeatResult, SeatResult]
    scores: Scores
    round_no: int
    skipped: bool
    caller: int | None
    trump: int | None
    fixed_trump: int | None
    trump_changed: bool
    total: int | None
    over: bool | None
    calls: int
    all_lose: bool
    carry: int
    multiplier: int
    next_carry: int


def next_carry(carry_in: int, skipped: bool, all_lost: bool, config: RulesConfig) -> int:
    """Multiplier for the next round (GAM-4, SCR-7, SCR-9).

    After a skipped round, or an all-lose round with ``all_lose_doubles_next`` on, the carry
    doubles (``carry_stacks``) or is 2. After any other round it resets to 1.
    """
    if skipped or (all_lost and config.all_lose_doubles_next):
        return carry_in * 2 if config.carry_stacks else 2
    return 1


def score_round(outcome: RoundOutcome, config: RulesConfig = DEFAULT_CONFIG) -> RoundResult:
    """Score a round by SPEC §3."""
    if outcome.skipped:
        return _score_skipped(outcome, config)

    estimates = outcome.estimates
    tricks = outcome.tricks
    caller = outcome.caller
    dash_callers = outcome.dash_callers
    with_players = outcome.with_players
    last = outcome.last_estimator
    risks = outcome.risks

    total = sum(estimates)
    over = total > 13  # EST-6: 14 or more is Over, 12 or less Under
    won = [estimates[s] == tricks[s] for s in range(4)]
    winners = won.count(True)
    all_lose = winners == 0

    calls = 1 + len(with_players)  # SCR-8: the Caller and each With player
    if config.calls_count_risk_and_dash:
        calls += len(dash_callers) + (1 if risks else 0)
    multiplier = outcome.carry * (2 if calls >= config.calls_to_double else 1)  # SCR-7, SCR-8
    if all_lose and config.all_lose_doubles_next:
        multiplier = 0  # SCR-9

    seats: list[SeatResult] = []
    for seat in range(4):
        b = estimates[seat]
        t = tricks[seat]
        win = won[seat]
        sign = 1 if win else -1
        is_dash = seat in dash_callers
        is_caller = seat == caller
        is_with = seat in with_players
        is_super = (is_caller or is_with) and b >= config.super_call_min
        is_last = seat == last

        base = dash = call = risk = only = super_points = 0
        if is_dash:  # SCR-2
            dash = sign * (config.dash_over if over else config.dash_under)
            if not config.dash_replaces_base:
                base = config.win_base + b if win else -abs(b - t)
        elif is_super and config.super_call_scoring:  # SCR-6
            super_points = b * b if win else -(b * b // 2)
        else:  # SCR-1
            base = config.win_base + b if win else -abs(b - t)
        if is_caller or is_with:  # SCR-3
            call = sign * config.caller_with_bonus
        if is_last:  # SCR-4
            risk = sign * config.risk_points * risks
        if (
            (winners == 1 and win) or (winners == 3 and not win)  # SCR-5
        ) and (
            (not is_dash or config.only_bonus_for_dash_calls)
            and (not is_super or config.only_bonus_for_super_calls)
        ):
            only = sign * config.only_bonus

        points = base + dash + call + risk + only + super_points
        seats.append(
            SeatResult(
                estimate=b,
                tricks=t,
                won=win,
                is_dash=is_dash,
                is_caller=is_caller,
                is_with=is_with,
                is_super=is_super,
                is_last=is_last,
                risks=risks if is_last else 0,
                base_points=base,
                dash_points=dash,
                call_points=call,
                risk_points=risk,
                only_points=only,
                super_points=super_points,
                multiplier=multiplier,
                score=points * multiplier,
            )
        )

    fixed = outcome.fixed_trump
    return RoundResult(
        seats=(seats[0], seats[1], seats[2], seats[3]),
        scores=(seats[0].score, seats[1].score, seats[2].score, seats[3].score),
        round_no=outcome.round_no,
        skipped=False,
        caller=caller,
        trump=outcome.trump,
        fixed_trump=fixed,
        trump_changed=fixed is not None and outcome.trump is not None and outcome.trump != fixed,
        total=total,
        over=over,
        calls=calls,
        all_lose=all_lose,
        carry=outcome.carry,
        multiplier=multiplier,
        next_carry=next_carry(outcome.carry, False, all_lose, config),
    )


def _score_skipped(outcome: RoundOutcome, config: RulesConfig) -> RoundResult:
    """A skipped round: everyone scores 0 and the next round gets a carry (GAM-3, GAM-4)."""
    carry = outcome.carry
    seats = tuple(
        SeatResult(
            estimate=0,
            tricks=0,
            won=False,
            is_dash=seat in outcome.dash_callers,
            is_caller=False,
            is_with=False,
            is_super=False,
            is_last=False,
            risks=0,
            base_points=0,
            dash_points=0,
            call_points=0,
            risk_points=0,
            only_points=0,
            super_points=0,
            multiplier=carry,
            score=0,
        )
        for seat in range(4)
    )
    return RoundResult(
        seats=(seats[0], seats[1], seats[2], seats[3]),
        scores=(0, 0, 0, 0),
        round_no=outcome.round_no,
        skipped=True,
        caller=None,
        trump=None,
        fixed_trump=outcome.fixed_trump,
        trump_changed=False,
        total=None,
        over=None,
        calls=0,
        all_lose=False,
        carry=carry,
        multiplier=carry,
        next_carry=next_carry(carry, True, False, config),
    )
