"""Arena metrics (SPEC §6 M2).

- **Relative Score Advantage (RSA):** a bot's Bola score minus the mean of the other three seats,
  averaged over all its Bolas, with a 95% Student-t confidence interval over duplicate sets.
- **Elo:** a Bradley-Terry fit over pairwise results with RandomBot fixed at 800. Within each set,
  every pair of different bots is compared by score across the set: a win, a draw or a loss.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .duplicate import ROTATIONS, SetResult

ANCHOR = "random"
ANCHOR_RATING = 800.0
ELO_SCALE = 400.0
DEFAULT_PRIOR = 1.0  # virtual draws per compared pair, so a clean sweep still gets a finite rating


def rsa(scores: Sequence[int], seat: int) -> float:
    """``seat``'s score minus the mean score of the other three seats."""
    return scores[seat] - (sum(scores) - scores[seat]) / 3


def _seat_games(lineup: Sequence[str]) -> list[tuple[str, int, int]]:
    """``(bot, rotation, seat)`` for every Bola each lineup slot plays in a set."""
    return [
        (lineup[slot], rotation, (slot - rotation) % 4)
        for rotation in range(ROTATIONS)
        for slot in range(4)
    ]


def set_rsa(result: SetResult, lineup: Sequence[str]) -> dict[str, float]:
    """Each bot's mean RSA over its Bolas in one set."""
    return _per_bot(
        lineup, ((bot, rsa(result.scores[k], seat)) for bot, k, seat in _seat_games(lineup))
    )


def set_scores(result: SetResult, lineup: Sequence[str]) -> dict[str, float]:
    """Each bot's mean Bola score over its Bolas in one set."""
    return _per_bot(
        lineup, ((bot, float(result.scores[k][seat])) for bot, k, seat in _seat_games(lineup))
    )


def _per_bot(lineup: Sequence[str], values: Iterable[tuple[str, float]]) -> dict[str, float]:
    sums = dict.fromkeys(lineup, 0.0)
    counts = dict.fromkeys(lineup, 0)
    for bot, value in values:
        sums[bot] += value
        counts[bot] += 1
    return {bot: sums[bot] / counts[bot] for bot in sums}


@dataclass(frozen=True, slots=True)
class Interval:
    """A mean with its confidence interval, over ``n`` values with sample deviation ``sd``."""

    mean: float
    low: float
    high: float
    sd: float
    n: int


def mean_ci(values: Sequence[float], level: float = 0.95) -> Interval:
    """The mean of ``values`` with a two-sided Student-t confidence interval."""
    n = len(values)
    if n == 0:
        raise ValueError("a confidence interval needs at least one value")
    mean = math.fsum(values) / n
    if n == 1:
        return Interval(mean, -math.inf, math.inf, 0.0, 1)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    half = t_quantile(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
    return Interval(mean, mean - half, mean + half, sd, n)


def t_quantile(p: float, df: int) -> float:
    """The ``p`` quantile of Student's t distribution with ``df`` degrees of freedom."""
    if not 0 < p < 1 or df < 1:
        raise ValueError("t_quantile needs 0 < p < 1 and df >= 1")
    if p < 0.5:
        return -t_quantile(1 - p, df)
    low, high = 0.0, 1.0
    while _t_cdf(high, df) < p:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        if _t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _t_cdf(t: float, df: int) -> float:
    tail = 0.5 * _beta_inc(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t > 0 else tail


def _beta_inc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_fraction(a, b, x) / a
    return 1 - front * _beta_fraction(b, a, 1 - x) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz's method)."""
    tiny = 1e-300
    c = 1.0
    d = 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        step = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1 + step * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + step / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        step = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + step * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + step / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h


@dataclass(frozen=True, slots=True)
class PairRecord:
    """Sets won, drawn and lost between two bots, from ``a``'s side."""

    a: str
    b: str
    a_wins: int
    draws: int
    b_wins: int


def pair_records(results: Iterable[SetResult], lineup: Sequence[str]) -> list[PairRecord]:
    """Every pair of different bots in the lineup, compared set by set.

    A bot's side of a set is its mean Bola score there. Bots in the same number of slots are
    thereby compared by their total scores across the set; the mean keeps the comparison fair
    when one bot fills more slots than another.
    """
    bots = list(dict.fromkeys(lineup))
    pairs = [(a, b) for i, a in enumerate(bots) for b in bots[i + 1 :]]
    tally = {pair: [0, 0, 0] for pair in pairs}
    for result in results:
        means = set_scores(result, lineup)
        for a, b in pairs:
            diff = means[a] - means[b]
            tally[(a, b)][0 if diff > 0 else 2 if diff < 0 else 1] += 1
    return [PairRecord(a, b, *tally[(a, b)]) for a, b in pairs]


def bradley_terry(
    pairs: Iterable[PairRecord],
    anchor: str = ANCHOR,
    anchor_rating: float = ANCHOR_RATING,
    prior: float = DEFAULT_PRIOR,
) -> dict[str, float]:
    """Elo ratings by Bradley-Terry maximum likelihood, ``anchor`` fixed at ``anchor_rating``.

    ``P(i beats j) = 1 / (1 + 10 ** ((r_j - r_i) / 400))``; a draw counts half a win to each side.
    Each compared pair also gets ``prior`` virtual draws, which keeps ratings finite when one bot
    wins every comparison. Only bots connected to the anchor through comparisons get a rating.
    """
    score: dict[tuple[str, str], float] = {}  # sorted pair -> first bot's points
    games: dict[tuple[str, str], float] = {}
    for record in pairs:
        n = record.a_wins + record.draws + record.b_wins
        if n == 0 or record.a == record.b:
            continue
        points = record.a_wins + record.draws / 2
        key = (record.a, record.b) if record.a < record.b else (record.b, record.a)
        score[key] = score.get(key, 0.0) + (points if key[0] == record.a else n - points)
        games[key] = games.get(key, 0.0) + n
    neighbours: dict[str, dict[str, float]] = {}
    wins: dict[str, float] = {}
    for (x, y), n in games.items():
        x_points = score[(x, y)] + prior / 2
        total = n + prior
        neighbours.setdefault(x, {})[y] = total
        neighbours.setdefault(y, {})[x] = total
        wins[x] = wins.get(x, 0.0) + x_points
        wins[y] = wins.get(y, 0.0) + total - x_points
    if anchor not in neighbours:
        return {}
    players = _component(neighbours, anchor)
    for player in players:
        if wins[player] <= 0 or wins[player] >= sum(neighbours[player].values()):
            raise ValueError(f"{player} won or lost every comparison: use a prior above 0")
    strength = dict.fromkeys(players, 1.0)
    for _ in range(1_000_000):
        updated = {
            i: wins[i] / sum(n / (strength[i] + strength[j]) for j, n in neighbours[i].items())
            for i in players
        }
        scale = updated[anchor]
        updated = {i: s / scale for i, s in updated.items()}
        change = max(abs(math.log(updated[i] / strength[i])) for i in players)
        strength = updated
        if change < 1e-13:
            break
    return {
        player: anchor_rating + ELO_SCALE * math.log10(strength[player])
        for player in sorted(players)
    }


def sweep_gap(n: int, confidence: float = 0.95) -> float:
    """Lower confidence bound on the Elo gap between two bots when one won all ``n`` comparisons.

    A clean sweep has no finite Bradley-Terry fit: its rating comes from the virtual draws. What
    the data do say is that the win rate is above ``(1 - confidence) ** (1 / n)``.
    """
    if n < 1:
        raise ValueError("a sweep needs at least one comparison")
    p = (1 - confidence) ** (1 / n)
    return ELO_SCALE * math.log10(p / (1 - p))


def _component(neighbours: Mapping[str, Mapping[str, float]], start: str) -> list[str]:
    seen = {start}
    stack = [start]
    while stack:
        for other in neighbours[stack.pop()]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return sorted(seen)
