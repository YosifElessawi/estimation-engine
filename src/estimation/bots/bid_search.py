"""Search bidding: Dash Call, auction and estimation decisions by sampling and simulation.

The bidding twin of PIMC's card search. For each candidate action it imagines deals of the other
three hands, favouring deals that fit what the bidding so far says about them, and plays each
deal's round to the end: the rest of the bidding with HeuristicBot for every seat, then the card
play with HeuristicBot's fast rollouts. It scores the result by SPEC §3 from its own side (own
score minus the mean of the other three) and picks the best average; ties go to HeuristicBot's
choice.

The imagined rounds run on the real rules engine, rebuilt from the public action log with the
bot's own sampled hands, so they follow the rules exactly without the bot ever seeing a real
hand. A skipped round scores 0 for everyone: the x2 carry it hands the next round is not valued.
"""

import math
import random
from collections.abc import Sequence

from estimation.engine import (
    PASS,
    SANS,
    WITH,
    Observation,
    Phase,
    Round,
    RulesConfig,
    bid_action,
    bid_of,
)
from estimation.engine.actions import is_bid, is_estimate
from estimation.engine.cards import FULL_DECK

from .hand_eval import HandValues, expected_tricks
from .heuristic import HeuristicBot, HeuristicConfig, lowest_bid
from .rollout import finish_round
from .sampling import deal_unseen
from .scoring import RoundScorer

EXTRA_BIDS = 3  # bids searched per trump beyond the lowest legal one


class BidSearch:
    """Chooses non-card actions by simulation. It reads only the Observation it is given."""

    __slots__ = (
        "_heuristic",
        "_rng",
        "estimate_sigma",
        "heuristic_config",
        "oversample",
        "rules",
        "samples",
    )

    def __init__(
        self,
        rng: random.Random,
        samples: int,
        oversample: int,
        estimate_sigma: float,
        heuristic: HeuristicConfig,
        rules: RulesConfig,
    ) -> None:
        if samples < 1 or oversample < 1 or estimate_sigma <= 0:
            raise ValueError("bid search needs samples >= 1, oversample >= 1 and sigma > 0")
        self._rng = rng
        self.samples = samples
        self.oversample = oversample
        self.estimate_sigma = estimate_sigma
        self.heuristic_config = heuristic
        self.rules = rules
        self._heuristic = HeuristicBot(heuristic)

    def choose(self, obs: Observation, legal: list[int]) -> int:
        hint = self._heuristic.act(obs, legal)
        actions = self.candidates(obs, legal, hint)
        if len(actions) == 1:
            return actions[0]
        totals = self.evaluate(obs, actions, self.sample_deals(obs))
        best = 0
        for k in range(1, len(actions)):
            if totals[k] > totals[best]:
                best = k
        return actions[best]

    def candidates(self, obs: Observation, legal: list[int], hint: int) -> list[int]:
        """The actions worth simulating, HeuristicBot's choice first.

        Every Dash Call choice and every estimate; in an auction Pass, With and, per trump, the
        lowest legal bid and the next few; for super calls the lowest legal one per trump.
        """
        picks = [hint]
        if obs.phase is Phase.AUCTION:
            picks += [a for a in (PASS, WITH) if a in legal]
            for trump in range(5):
                low = lowest_bid(obs.top_bid, trump)
                for tricks in range(low, min(low + EXTRA_BIDS, 13) + 1):
                    picks.append(bid_action(tricks, trump))
        elif obs.phase is Phase.ESTIMATION:
            picks += [a for a in legal if is_estimate(a)]
            seen_trumps: set[int] = set()
            for action in legal:
                if is_bid(action) and bid_of(action)[1] not in seen_trumps:
                    seen_trumps.add(bid_of(action)[1])
                    picks.append(action)
        else:
            picks += legal
        allowed = set(legal)
        return [a for a in dict.fromkeys(picks) if a in allowed]

    def sample_deals(self, obs: Observation) -> list[list[int]]:
        """``samples`` deals of the other three hands (own hand included at its seat). With
        ``oversample`` > 1 it keeps those that best fit the bidding: Dash callers' hands should
        be nearly empty, an estimate should match the hand's value under the trump, and the top
        bidder and With players should hold at least their bid's tricks in its trump."""
        me = obs.seat
        others = ((me + 1) & 3, (me + 2) & 3, (me + 3) & 3)
        unseen = FULL_DECK & ~obs.hand
        rng = self._rng
        deals: list[list[int]] = []
        for _ in range(self.samples * self.oversample):
            three = deal_unseen(unseen, [13, 13, 13], [0, 0, 0], rng)
            hands = [0, 0, 0, 0]
            hands[me] = obs.hand
            for i, seat in enumerate(others):
                hands[seat] = three[i]
            deals.append(hands)
        if self.oversample == 1:
            return deals
        values = self.heuristic_config.values
        spread = 2 * self.estimate_sigma**2
        keys: list[float] = []
        for hands in deals:
            misfit = sum(_miss(obs, seat, hands[seat], values) ** 2 for seat in others)
            keys.append(math.log(1.0 - rng.random()) * math.exp(min(misfit / spread, 700.0)))
        kept = sorted(range(len(deals)), key=keys.__getitem__, reverse=True)[: self.samples]
        return [deals[k] for k in sorted(kept)]

    def evaluate(
        self, obs: Observation, actions: Sequence[int], deals: Sequence[list[int]]
    ) -> list[float]:
        """For each action, the sum over ``deals`` of own round score minus the mean of the
        other three, after taking it and playing the round out (HeuristicBot for everything
        after it)."""
        log = [action for _, action in obs.log]
        me = obs.seat
        totals = [0.0] * len(actions)
        for hands in deals:
            dealt = (hands[0], hands[1], hands[2], hands[3])
            start = Round.replay(
                obs.round_no, dealt, obs.fixed_trump, obs.carry, obs.scores, self.rules, log
            )
            for k, action in enumerate(actions):
                world = start.clone()
                world.step(action)
                totals[k] += self._finish(world, me, hands)
        return totals

    def _finish(self, world: Round, me: int, hands: list[int]) -> float:
        heuristic = self._heuristic
        while world.phase is not Phase.PLAY and world.phase is not Phase.DONE:
            seat = world.current_player
            if seat is None:
                raise RuntimeError("an unfinished round always has a player to act")
            world.step(heuristic.act(world.observation(seat), world.legal_actions()))
        if world.phase is Phase.DONE:
            return 0.0  # skipped: every seat scores 0 this round
        view = world.observation(me)
        if view.trump is None or view.leader is None:
            raise RuntimeError("a round in card play has a trump and a leader")
        estimates = [e if e is not None else 0 for e in view.estimates]
        tricks = finish_round(list(hands), [], view.leader, view.trump, estimates, [0, 0, 0, 0])
        return RoundScorer(view, self.rules).value(tricks)


def _miss(obs: Observation, seat: int, hand: int, values: HandValues) -> float:
    """How far a sampled hand is from what ``seat``'s bidding says, in tricks."""
    if seat in obs.dash_callers:
        return expected_tricks(hand, SANS, values)
    estimate = obs.estimates[seat]
    if estimate is not None and obs.trump is not None:
        return expected_tricks(hand, obs.trump, values) - estimate
    if obs.phase is Phase.AUCTION and obs.top_bid is not None:
        if seat == obs.top_bidder:
            tricks, trump = bid_of(obs.top_bid)
            return max(0.0, tricks - expected_tricks(hand, trump, values))
        for w in obs.with_players:
            if w.seat == seat:
                return max(0.0, w.w - expected_tricks(hand, w.trump, values))
    return 0.0
