"""PIMCBot (SPEC §6 M3): Perfect Information Monte Carlo search for card play.

At a card decision with more than one choice, it samples deals of the unseen cards that fit the
hand sizes, voids and played cards, keeping those where each player's hand best fits its
estimate. For each legal card and each deal it plays the round out with HeuristicBot's card play
for all four seats and scores it by SPEC §3, then plays the card with the best mean of its own
round score minus the mean of the other three. Dash Call, bidding and estimation reuse
HeuristicBot.

Search is deterministic for a given seed: the time budget is only measured, unless
``enforce_budget`` cuts the search short, which makes play depend on timing. Even then the budget
is checked between deals, after drawing the deals, so a decision can run a little over it.
"""

import dataclasses
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

from estimation.engine import (
    SANS,
    Observation,
    Phase,
    RoundOutcome,
    RulesConfig,
    play_action,
    score_round,
)
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG

from .card_play import choose_card
from .hand_eval import expected_tricks
from .heuristic import DEFAULT_HEURISTIC, HeuristicBot, HeuristicConfig
from .knowledge import Knowledge, read_knowledge
from .rollout import finish_round
from .sampling import deal_unseen


@dataclass(frozen=True, slots=True)
class PIMCConfig:
    samples: int = 30  # N: deals rolled out per card decision
    oversample: int = 3  # deals drawn per deal kept, to keep those that fit the estimates
    estimate_sigma: float = 1.0  # how closely a sampled hand must fit its estimate, in tricks
    budget_ms: float = 250.0  # time budget per card decision
    enforce_budget: bool = False  # stop rolling out new deals once the budget is spent
    merge_equivalent: bool = False  # roll out one card per run of touching cards
    heuristic: HeuristicConfig = DEFAULT_HEURISTIC  # rollouts, bidding and estimation
    rules: RulesConfig = DEFAULT_CONFIG  # the rules rollouts are scored by


DEFAULT_PIMC = PIMCConfig()


class PIMCBot:
    """Card play by sampling and rollouts; everything else as HeuristicBot. Reads only its own
    Observation: the deals it searches are its own samples."""

    __slots__ = ("_heuristic", "_rng", "config", "seed")

    def __init__(self, seed: int, config: PIMCConfig = DEFAULT_PIMC) -> None:
        if config.samples < 1 or config.oversample < 1 or config.estimate_sigma <= 0:
            raise ValueError("PIMC needs samples >= 1, oversample >= 1 and estimate_sigma > 0")
        self.seed = seed
        self.config = config
        self._heuristic = HeuristicBot(config.heuristic)
        self._rng = random.Random(seed)

    def __repr__(self) -> str:
        return f"PIMCBot(seed={self.seed}, {self.config})"

    def act(self, obs: Observation, legal: list[int]) -> int:
        if obs.phase is not Phase.PLAY:
            return self._heuristic.act(obs, legal)
        if len(legal) == 1:
            return legal[0]
        return play_action(self.choose(obs, [action - PLAY_BASE for action in legal]))

    def choose(self, obs: Observation, cards: list[int]) -> int:
        """The card to play from the legal ``cards``: the best total over the sampled deals,
        HeuristicBot's own card on a tie."""
        config = self.config
        start = time.perf_counter()
        know = read_knowledge(obs)
        trump = obs.trump
        if trump is None:
            raise ValueError("PIMC searches only during card play")
        estimates = _estimates(obs)
        trick = [card for _, card in obs.trick]
        me = obs.seat
        want = estimates[me] > obs.tricks_won[me]
        hint = choose_card(know.hand, trick, trump, want, know.unseen)
        candidates = (
            representatives(cards, know.unseen, hint)
            if config.merge_equivalent
            else sorted(cards, key=lambda card: card != hint)
        )
        if len(candidates) == 1:
            return candidates[0]
        deals = self.sample_deals(obs, know)
        deadline = start + config.budget_ms / 1000 if config.enforce_budget else None
        totals = self.evaluate(obs, candidates, deals, deadline)
        best = 0
        for k in range(1, len(candidates)):
            if totals[k] > totals[best]:
                best = k
        return candidates[best]

    def sample_deals(self, obs: Observation, know: Knowledge | None = None) -> list[list[int]]:
        """``samples`` full deals (four remaining hands, own hand included) that fit the hand
        sizes, voids and played cards. With ``oversample`` > 1 it draws that many times more
        and keeps the ones whose hands best fit the estimates, by weighted sampling without
        replacement: a deal's weight falls with the squared gap, over the other three seats,
        between each player's whole hand valued by HeuristicBot's hand model and its estimate
        (a Dash caller's Sans value against 0)."""
        config = self.config
        rng = self._rng
        if know is None:
            know = read_knowledge(obs)
        trump = obs.trump
        if trump is None:
            raise ValueError("PIMC samples deals only during card play")
        estimates = _estimates(obs)
        me = know.seat
        others = ((me + 1) & 3, (me + 2) & 3, (me + 3) & 3)
        sizes = [know.sizes[seat] for seat in others]
        voids = [know.voids[seat] for seat in others]
        unseen = know.unseen
        deals: list[list[int]] = []
        for _ in range(config.samples * config.oversample):
            three = deal_unseen(unseen, sizes, voids, rng)
            hands = [0, 0, 0, 0]
            hands[me] = know.hand
            for i, seat in enumerate(others):
                hands[seat] = three[i]
            deals.append(hands)
        if config.oversample == 1:
            return deals
        values = config.heuristic.values
        spread = 2 * config.estimate_sigma**2
        dash = obs.dash_callers
        keys: list[float] = []
        for hands in deals:
            misfit = 0.0
            for seat in others:
                original = hands[seat] | know.played_by[seat]
                if seat in dash:
                    miss = expected_tricks(original, SANS, values)
                else:
                    miss = expected_tricks(original, trump, values) - estimates[seat]
                misfit += miss * miss
            # Efraimidis-Spirakis: keep the largest u ** (1 / weight), compared as logarithms.
            keys.append(math.log(1.0 - rng.random()) * math.exp(min(misfit / spread, 700.0)))
        kept = sorted(range(len(deals)), key=keys.__getitem__, reverse=True)[: self.config.samples]
        return [deals[k] for k in sorted(kept)]

    def evaluate(
        self,
        obs: Observation,
        cards: Sequence[int],
        deals: Sequence[list[int]],
        deadline: float | None = None,
    ) -> list[float]:
        """For each card, the sum over ``deals`` of own round score minus the mean of the other
        three, playing the card and then the rest of the round with HeuristicBot card play for
        every seat. Every card is rolled out on the same deals; with a ``deadline`` (a
        ``time.perf_counter`` value) no new deal is started after it passes."""
        trump = obs.trump
        leader = obs.leader
        if trump is None or leader is None:
            raise ValueError("PIMC evaluates cards only during card play")
        estimates = _estimates(obs)
        me = obs.seat
        trick = [card for _, card in obs.trick]
        won = list(obs.tricks_won)
        scorer = RoundScorer(obs, self.config.rules)
        totals = [0.0] * len(cards)
        for sample in deals:
            for k, card in enumerate(cards):
                hands = sample.copy()
                hands[me] ^= 1 << card
                tricks = finish_round(hands, [*trick, card], leader, trump, estimates, won.copy())
                totals[k] += scorer.value(tricks)
            if deadline is not None and time.perf_counter() > deadline:
                break
        return totals


def _estimates(obs: Observation) -> tuple[int, int, int, int]:
    e = obs.estimates
    if e[0] is None or e[1] is None or e[2] is None or e[3] is None:
        raise ValueError("every estimate is known during card play")
    return e[0], e[1], e[2], e[3]


def representatives(cards: list[int], out: int, hint: int) -> list[int]:
    """One card from each run of touching cards, with the hint's run first.

    Two cards of a suit touch when every card ranked between them is played or in the same
    hand (``out`` holds the cards in other hands). They take the same tricks against any
    holding, though a rollout that discards by rank across suits may later play them
    differently. The hint represents its own run, so a tie in search falls back to it.
    """
    runs: list[list[int]] = []
    for card in sorted(cards):
        if runs:
            last = runs[-1][-1]
            between = ((1 << card) - 1) & ~((2 << last) - 1)
            if last // 13 == card // 13 and not out & between:
                runs[-1].append(card)
                continue
        runs.append([card])
    picks: list[int] = []
    for run in runs:
        if hint in run:
            picks.insert(0, hint)
        else:
            picks.append(run[0])
    return picks


class RoundScorer:
    """Scores final trick counts of the round in ``obs`` by SPEC §3, from ``obs.seat``'s side:
    its own round score minus the mean round score of the other three."""

    __slots__ = ("_cache", "_me", "_outcome", "_rules")

    def __init__(self, obs: Observation, rules: RulesConfig = DEFAULT_CONFIG) -> None:
        self._me = obs.seat
        self._rules = rules
        self._cache: dict[tuple[int, ...], float] = {}
        self._outcome = RoundOutcome(
            estimates=_estimates(obs),
            tricks=(13, 0, 0, 0),
            caller=obs.caller,
            dash_callers=obs.dash_callers,
            with_players=tuple(w.seat for w in obs.with_players),
            last_estimator=obs.last_estimator,
            risks=obs.risks or 0,
            carry=obs.carry,
            round_no=obs.round_no,
            trump=obs.trump,
            fixed_trump=obs.fixed_trump,
        )

    def value(self, tricks: Sequence[int]) -> float:
        key = tuple(tricks)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        outcome = dataclasses.replace(self._outcome, tricks=(key[0], key[1], key[2], key[3]))
        scores = score_round(outcome, self._rules).scores
        mine = scores[self._me]
        value = mine - (scores[0] + scores[1] + scores[2] + scores[3] - mine) / 3
        self._cache[key] = value
        return value
