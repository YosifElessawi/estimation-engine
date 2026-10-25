"""Scoring a round's final trick counts from one seat's side (SPEC §3), for search bots."""

import dataclasses
from collections.abc import Sequence

from estimation.engine import Observation, RoundOutcome, RulesConfig, score_round
from estimation.engine.config import DEFAULT_CONFIG


def estimates_of(obs: Observation) -> tuple[int, int, int, int]:
    e = obs.estimates
    if e[0] is None or e[1] is None or e[2] is None or e[3] is None:
        raise ValueError("every estimate is known during card play")
    return e[0], e[1], e[2], e[3]


AHEAD = 1.0  # weight of a player ahead of me in the game [Claude], design 5.9
BEHIND = 0.5  # weight of a player level with or behind me


def standings_weights(
    scores: tuple[int, int, int, int], me: int
) -> tuple[float, float, float, float]:
    """How much each other player's round score counts against mine: players ahead of me in the
    game count double, normalised so the three weights sum to 1. My own weight is 0. Level
    scores give the plain mean."""
    raw = [0.0 if s == me else AHEAD if scores[s] > scores[me] else BEHIND for s in range(4)]
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total, raw[3] / total)


class RoundScorer:
    """Scores final trick counts of the round in ``obs`` by SPEC §3, from ``obs.seat``'s side:
    its own round score minus the mean round score of the other three, or with ``weights``
    (see ``standings_weights``) minus their weighted sum."""

    __slots__ = ("_cache", "_me", "_outcome", "_rules", "_weights")

    def __init__(
        self,
        obs: Observation,
        rules: RulesConfig = DEFAULT_CONFIG,
        weights: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._me = obs.seat
        self._rules = rules
        self._weights = weights
        self._cache: dict[tuple[int, ...], float] = {}
        self._outcome = RoundOutcome(
            estimates=estimates_of(obs),
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
        weights = self._weights
        if weights is None:
            value = mine - (scores[0] + scores[1] + scores[2] + scores[3] - mine) / 3
        else:
            value = mine - sum(w * s for w, s in zip(weights, scores, strict=True))
        self._cache[key] = value
        return value
