"""HeuristicBot (SPEC §6 M2): a simple, deterministic rule-of-thumb player.

- Dash Call: only with a hopeless hand (no honours worth a trick, no long suit).
- Auction: aims for ``target_tricks`` in its best trump, opening or overcalling with the lowest
  bid in that trump and raising to the target on its final turn. Says With when its hand fits
  the top bid. Bids 8 or more (a super call) once its expected tricks round to 8 (from 7.5;
  until 2026-10-26 it needed 8.5, which rounds to 9, so it never bid exactly 8).
- Estimation: the legal value nearest its expected tricks; the last estimator also weighs Risks.
  In a fixed-trump round it super-calls only when another trump is much stronger.
- Card play: ``card_play.choose_card``: win tricks until the estimate is reached, then duck.

Every threshold lives in ``HeuristicConfig``; they are bot design choices, not rules.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from estimation.engine import (
    DASH_CALL,
    NO_DASH_CALL,
    PASS,
    SANS,
    WITH,
    Observation,
    Phase,
    bid_action,
    bid_of,
    estimate_action,
    play_action,
)
from estimation.engine.actions import is_bid, is_estimate
from estimation.engine.cards import FULL_DECK

from .card_play import choose_card
from .hand_eval import DEFAULT_VALUES, HandValues, expected_tricks, trump_values

STRONGEST_FIRST = (4, 3, 2, 1, 0)  # Sans, Spades, Hearts, Diamonds, Clubs (TBL-5)


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    values: HandValues = DEFAULT_VALUES
    dash_max_tricks: float = 0.3  # Dash Call only if the hand's Sans value is at most this...
    dash_max_suit: int = 4  # ...and no suit is longer than this
    bid_rounding: float = 0.5  # aim for int(expected tricks + bid_rounding)
    super_call_min: int = 8  # the rules' super call size (RulesConfig.super_call_min)
    super_call_tricks: float = 7.5  # aim for a super call only with this many expected tricks
    super_call_gain: float = 2.0  # fixed-trump rounds: super-call only if it adds this many
    risk_aversion: float = 0.25  # tricks of error the last estimator accepts to avoid one Risk


DEFAULT_HEURISTIC = HeuristicConfig()


def target_tricks(expected: float, config: HeuristicConfig) -> int:
    """The trick count to aim for: rounded expected tricks, held below a super call unless the
    hand is very strong."""
    tricks = int(expected + config.bid_rounding + 1e-9)
    if tricks >= config.super_call_min and expected < config.super_call_tricks:
        tricks = config.super_call_min - 1
    return min(tricks, 13)


def pick_estimate(
    expected: float, options: Iterable[int], total_before: int | None, risk_aversion: float
) -> int:
    """The option nearest ``expected``, the lower one on a tie.

    For the last estimator, ``total_before`` is the sum of the other estimates, and each Risk the
    choice would take (EST-7) adds ``risk_aversion`` to its distance.
    """
    best = -1
    best_cost = float("inf")
    for k in sorted(options):
        cost = abs(k - expected)
        if total_before is not None:
            cost += risk_aversion * (abs(total_before + k - 13) // 2)
        if cost < best_cost - 1e-12:
            best, best_cost = k, cost
    if best < 0:
        raise ValueError("no estimate to choose from")
    return best


def lowest_bid(top: int | None, trump: int) -> int:
    """Fewest tricks of a bid in ``trump`` that beats ``top`` (TBL-6); 14 if none does."""
    if top is None:
        return 4
    tricks, top_trump = bid_of(top)
    return tricks if trump > top_trump else tricks + 1


class HeuristicBot:
    """A deterministic rule-of-thumb player. It reads only its own Observation."""

    __slots__ = ("config",)

    def __init__(self, config: HeuristicConfig = DEFAULT_HEURISTIC) -> None:
        self.config = config

    def __repr__(self) -> str:
        return f"HeuristicBot({self.config})"

    def act(self, obs: Observation, legal: list[int]) -> int:
        phase = obs.phase
        if phase is Phase.PLAY:
            action = self.play(obs)
        elif phase is Phase.AUCTION:
            action = self.bid(obs, legal)
        elif phase is Phase.ESTIMATION:
            action = self.estimate(obs, legal)
        elif phase is Phase.DASH:
            action = DASH_CALL if self.hopeless(obs.hand) else NO_DASH_CALL
        else:
            raise ValueError("there is no decision to make in a finished round")
        if action not in legal:
            raise RuntimeError(f"HeuristicBot chose illegal action {action} in {phase.name}")
        return action

    def hopeless(self, hand: int) -> bool:
        """A hand to Dash Call with: almost no high cards and no long suit."""
        config = self.config
        if expected_tricks(hand, SANS, config.values) > config.dash_max_tricks:
            return False
        longest = max(((hand >> (13 * suit)) & 0x1FFF).bit_count() for suit in range(4))
        return longest <= config.dash_max_suit

    def bid(self, obs: Observation, legal: list[int]) -> int:
        config = self.config
        values = trump_values(obs.hand, config.values)
        targets = [target_tricks(value, config) for value in values]
        me = obs.seat
        top = obs.top_bid
        if top is not None and obs.top_bidder == me:  # AUC-3: the top bidder's final turn
            tricks, trump = bid_of(top)
            if targets[trump] > tricks:
                raised = bid_action(targets[trump], trump)
                if raised in legal:
                    return raised
            return _pass_or_cheapest(legal)
        best = -1
        best_value = -1.0
        for trump in STRONGEST_FIRST:
            tricks = lowest_bid(top, trump)
            if tricks <= targets[trump] and values[trump] > best_value:
                action = bid_action(tricks, trump)
                if action in legal:
                    best, best_value = action, values[trump]
        if best >= 0:
            return best
        if top is not None and WITH in legal:
            tricks, trump = bid_of(top)
            if targets[trump] >= tricks:
                return WITH
        return _pass_or_cheapest(legal)

    def estimate(self, obs: Observation, legal: list[int]) -> int:
        config = self.config
        values = trump_values(obs.hand, config.values)
        fixed = obs.fixed_trump
        if fixed is not None and any(is_bid(action) for action in legal):
            call = self._super_call(values, fixed, legal)
            if call >= 0:
                return call
        trump = obs.trump
        if trump is None:
            raise ValueError("the trump is known before anyone estimates")
        options = [action - estimate_action(0) for action in legal if is_estimate(action)]
        total_before = None
        if obs.last_estimator == obs.seat:
            total_before = sum(e for e in obs.estimates if e is not None)
        k = pick_estimate(values[trump], options, total_before, config.risk_aversion)
        return estimate_action(k)

    def _super_call(
        self, values: tuple[float, float, float, float, float], fixed: int, legal: list[int]
    ) -> int:
        """FIX-4: the lowest legal super call in a trump much stronger than the fixed one, or -1."""
        config = self.config
        best = -1
        best_value = -1.0
        for trump in STRONGEST_FIRST:
            if trump == fixed or values[trump] - values[fixed] < config.super_call_gain:
                continue
            target = target_tricks(values[trump], config)
            if target < config.super_call_min or values[trump] <= best_value:
                continue
            for tricks in range(config.super_call_min, target + 1):
                action = bid_action(tricks, trump)
                if action in legal:
                    best, best_value = action, values[trump]
                    break
        return best

    def play(self, obs: Observation) -> int:
        hand = obs.hand
        played = 0
        for record in obs.tricks:
            for card in record.cards:
                played |= 1 << card
        trick = [card for _, card in obs.trick]
        for card in trick:
            played |= 1 << card
        me = obs.seat
        estimate = obs.estimates[me]
        trump = obs.trump
        if estimate is None or trump is None:
            raise ValueError("estimates and trump are known before play")
        want = estimate > obs.tricks_won[me]
        return play_action(choose_card(hand, trick, trump, want, FULL_DECK & ~played & ~hand))


def _pass_or_cheapest(legal: list[int]) -> int:
    """Pass, or when AUC-6 forbids it (passing would leave 13 with nobody to estimate), With,
    else the cheapest legal bid."""
    if PASS in legal:
        return PASS
    if WITH in legal:
        return WITH
    return min(legal)
