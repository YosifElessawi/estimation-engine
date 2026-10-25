"""Standings, match-win probability and the training reward (SPEC §6 M4).

The reward for a seat's round is its own score change minus a standings-weighted change in the
opponents' scores: an opponent ahead of the seat in the game before the round counts
``ahead_weight`` (1.0), one behind it ``behind_weight`` (0.5). In the fixed-trump rounds it is
blended with the change in the seat's match-win probability, so near the end the bot also plays
to finish first, not only for points.

With ``normalize_weights`` (the default) the weights are divided by their sum, so the opponents'
term is a weighted mean of their changes: an opponent ahead still counts twice as much as one
behind. Without it the weights add up to 1.5-3.0, so a round where every seat scores the same
+15 is worth -7.5 to -30 to each of them, and a skipped round (everyone 0) beats an average
played round. In a pilot the self-play bidder learned exactly that: by iteration 8 it passed so
often that 56% of self-play rounds were skipped.

The match-win probability is a simple model: every seat's score over the rounds left moves by an
independent normal step with spread ``round_sd`` per round, and a seat's chance of finishing first
is the chance its final score is the highest. Once the game is over it is the seat's actual share
of first place (ties shared, GAM-5).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

OPEN_ROUNDS = 13  # GAM-1: a Bola opens with 13 open rounds
ROUND_SD = 20.0
"""Spread of one seat's round score in points: HeuristicBot self-play measured 15 in open
rounds and 19 in fixed-trump rounds."""

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(24)
_NODE_WEIGHTS = tuple(
    (float(z), float(w)) for z, w in zip(_NODES, _WEIGHTS / _WEIGHTS.sum(), strict=True)
)

Scores = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RewardConfig:
    ahead_weight: float = 1.0  # an opponent ahead of you in the game
    behind_weight: float = 0.5  # an opponent behind you
    tie_weight: float = 1.0  # an opponent level with you still blocks first place
    normalize_weights: bool = True  # divide the weights by their sum: a weighted mean
    win_blend: float = 0.5  # share of the fixed-trump round reward from match-win probability
    win_points: float = 100.0  # points one whole match win is worth in that blend
    round_sd: float = ROUND_SD  # the match-win model's spread per round

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_blend <= 1.0:
            raise ValueError("win_blend must be between 0 and 1")
        if self.round_sd <= 0:
            raise ValueError("round_sd must be positive")
        if min(self.ahead_weight, self.behind_weight, self.tie_weight) < 0:
            raise ValueError("standings weights must not be negative")
        if (
            self.normalize_weights
            and min(self.ahead_weight, self.behind_weight, self.tie_weight) == 0
        ):
            raise ValueError("normalized standings weights must be positive")


DEFAULT_REWARD = RewardConfig()


def standings_weights(
    scores: Scores, seat: int, config: RewardConfig = DEFAULT_REWARD
) -> tuple[float, float, float, float]:
    """The weight of each opponent's score change in ``seat``'s reward, 0 for ``seat`` itself."""
    mine = scores[seat]
    weights = [0.0, 0.0, 0.0, 0.0]
    for other in range(4):
        if other == seat:
            continue
        theirs = scores[other]
        if theirs > mine:
            weights[other] = config.ahead_weight
        elif theirs < mine:
            weights[other] = config.behind_weight
        else:
            weights[other] = config.tie_weight
    return weights[0], weights[1], weights[2], weights[3]


def opponent_weights(
    scores: Scores, seat: int, config: RewardConfig = DEFAULT_REWARD
) -> tuple[float, float, float, float]:
    """The standings weights, divided by their sum when ``normalize_weights`` is on."""
    weights = standings_weights(scores, seat, config)
    if not config.normalize_weights:
        return weights
    total = sum(weights)
    return weights[0] / total, weights[1] / total, weights[2] / total, weights[3] / total


def points_reward(
    before: Scores, change: Scores, seat: int, config: RewardConfig = DEFAULT_REWARD
) -> float:
    """``seat``'s score change minus the standings-weighted change of the other three, with the
    standings taken from the game scores ``before`` the round."""
    weights = opponent_weights(before, seat, config)
    return change[seat] - sum(weights[other] * change[other] for other in range(4))


def first_place_shares(scores: Scores) -> tuple[float, float, float, float]:
    """Each seat's share of first place: 1, or 1/k when k seats tie on top (GAM-5)."""
    top = max(scores)
    tied = sum(1 for score in scores if score == top)
    shares = [1.0 / tied if score == top else 0.0 for score in scores]
    return shares[0], shares[1], shares[2], shares[3]


@lru_cache(maxsize=4096)
def win_probabilities(
    scores: Scores, rounds_left: int, round_sd: float = ROUND_SD
) -> tuple[float, float, float, float]:
    """Each seat's chance of finishing first with ``rounds_left`` rounds still to play.

    With no rounds left it is the actual share of first place. Otherwise each seat's final score
    is its score plus an independent normal with spread ``round_sd * sqrt(rounds_left)``, and
    ``P(seat first) = E_z[prod over others of Phi((s_seat - s_other) / spread + z)]``, taken by
    24-point Gauss-Hermite quadrature.
    """
    if rounds_left <= 0:
        return first_place_shares(scores)
    spread = round_sd * math.sqrt(rounds_left)
    root2 = math.sqrt(2.0)
    probs = [0.0, 0.0, 0.0, 0.0]
    for seat in range(4):
        gaps = [(scores[seat] - scores[o]) / spread for o in range(4) if o != seat]
        total = 0.0
        for z, weight in _NODE_WEIGHTS:
            product = weight
            for gap in gaps:
                product *= 0.5 * (1.0 + math.erf((gap + z) / root2))
            total += product
        probs[seat] = total
    norm = sum(probs)
    return probs[0] / norm, probs[1] / norm, probs[2] / norm, probs[3] / norm


def rounds_remaining(round_no: int, fixed_left: int, fixed_round: bool) -> int:
    """Rounds still to play counting the current one: the open rounds left (``round_no`` up to
    13) and one per fixed trump not yet played, replays aside."""
    open_left = 0 if fixed_round else max(0, OPEN_ROUNDS - round_no + 1)
    return open_left + fixed_left


def round_reward(
    before: Scores,
    change: Scores,
    seat: int,
    fixed_round: bool,
    rounds_left_before: int,
    rounds_left_after: int,
    config: RewardConfig = DEFAULT_REWARD,
) -> float:
    """The training reward of ``seat``'s round (SPEC §6 M4).

    ``rounds_left_before`` counts the rounds still to play including this one, and
    ``rounds_left_after`` those after it (0 once the game is over). In open rounds the reward is
    ``points_reward``. In fixed-trump rounds it is ``(1 - win_blend)`` times that plus
    ``win_blend * win_points`` times the change in the seat's match-win probability.
    """
    points = points_reward(before, change, seat, config)
    if not fixed_round or config.win_blend == 0.0:
        return points
    after: Scores = (
        before[0] + change[0],
        before[1] + change[1],
        before[2] + change[2],
        before[3] + change[3],
    )
    p_before = win_probabilities(before, rounds_left_before, config.round_sd)[seat]
    p_after = win_probabilities(after, rounds_left_after, config.round_sd)[seat]
    blend = config.win_blend
    return (1.0 - blend) * points + blend * config.win_points * (p_after - p_before)
