"""From trick chances to the expected points of each number (design 5.2).

``most`` is the trick count when trying to take tricks, ``least`` when trying to lose them. On the
same deal ``least <= most``, so a player aiming at k makes it exactly when ``least <= k <= most``:
``P(most >= k) - P(least >= k + 1)``. Taking too many tricks is as bad as taking too few, which is
why a strong hand with no low cards cannot stop at a low number [Yosif].

``stake`` mirrors SPEC §3's SCR-1 to SCR-4 and SCR-6 for one player; the tests check it against
``score_round``. SCR-5 (only winner, only loser) is added in ``expected_points`` from a prior on how
often the other players make their estimates; multipliers are left out.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from estimation.engine import RulesConfig

from .calibration import Calibration
from .counting import HandCount


@dataclass(frozen=True, slots=True)
class Calibrated:
    """A hand's natural trick count and the fit that prices calls around it (``calibration``)."""

    natural: float
    fit: Calibration


@dataclass(frozen=True, slots=True)
class Outlook:
    """The chance of each trick count, trying to win tricks (``most``) and to lose them
    (``least``). ``control`` is how reliably dodging tricks lands exactly on a lower number."""

    most: tuple[float, ...]
    least: tuple[float, ...]
    control: float = 1.0
    calibrated: Calibrated | None = None  # when set, make chances come from the saved games


@dataclass(frozen=True, slots=True)
class Stake:
    """A number's points: ``win`` when made, ``lose`` (a positive cost) when missed, plus
    ``per_trick`` for each trick of the miss. ``solo_win`` and ``solo_lose`` say whether
    SCR-5's only-winner and only-loser bonuses apply."""

    win: float
    lose: float
    per_trick: float
    solo_win: bool = True
    solo_lose: bool = True


def spread(chances: Iterable[float]) -> list[float]:
    """The distribution of how many of these independent chances come off."""
    odds = [1.0]
    for p in chances:
        nxt = [0.0] * (len(odds) + 1)
        for k, q in enumerate(odds):
            nxt[k] += q * (1 - p)
            nxt[k + 1] += q * p
        odds = nxt
    return odds


def _combine(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return tuple(out)


def outlook(count: HandCount, dodge_scale: float = 1.0, control: float = 1.0) -> Outlook:
    """Trump tricks plus side tricks, wanted and dodged. ``dodge_scale`` weakens every dodge
    (opponents forcing tricks on a player); ``control`` is carried to ``make_chance``."""
    most = _combine(count.trumps, spread(c.win for c in count.side))
    least = _combine(count.trumps, spread(c.win * (1 - c.dodge * dodge_scale) for c in count.side))
    return Outlook(most, least, control)


def _at_least(odds: Sequence[float], k: int) -> float:
    return sum(odds[max(k, 0) :])


def make_chance(o: Outlook, k: int) -> float:
    """The chance of taking exactly ``k`` tricks when aiming at ``k``: from the saved games when
    the outlook is calibrated, else landing on it while taking every trick possible, plus,
    weighted by ``control``, dodging down to it."""
    if o.calibrated is not None:
        return o.calibrated.fit.make(k - o.calibrated.natural, call=k)
    natural = o.most[k] if 0 <= k < len(o.most) else 0.0
    dodged = _at_least(o.most, k + 1) - _at_least(o.least, k + 1)
    return max(0.0, natural + o.control * dodged)


def expected_miss(o: Outlook, k: int) -> float:
    """Expected tricks short of ``k`` plus expected tricks over it."""
    if o.calibrated is not None:
        distance = k - o.calibrated.natural
        fit = o.calibrated.fit
        return (1 - fit.make(distance, call=k)) * fit.miss_size(distance, call=k)
    short = sum((k - n) * p for n, p in enumerate(o.most) if n < k)
    over = sum((n - k) * p for n, p in enumerate(o.least) if n > k)
    return short + over


def stake(
    k: int, *, caller: bool, dash: bool, last_risks: int, over: bool | None, rules: RulesConfig
) -> Stake:
    """The points of estimating ``k`` (SCR-1 to SCR-4, SCR-6). ``caller`` covers With too;
    ``over`` None (unknown) prices a Dash Call at the mean of its Over and Under values."""
    risk = rules.risk_points * last_risks
    if dash:
        if over is None:
            value = (rules.dash_over + rules.dash_under) / 2
        else:
            value = rules.dash_over if over else rules.dash_under
        solo = rules.only_bonus_for_dash_calls
        if rules.dash_replaces_base:
            return Stake(value + risk, value + risk, 0, solo, solo)
        return Stake(value + rules.win_base + risk, value + risk, 1, solo, solo)
    if caller and k >= rules.super_call_min and rules.super_call_scoring:
        square = k * k
        return Stake(
            square + risk,
            square // 2 + risk,
            0,
            rules.only_winner_for_super_calls,
            rules.only_loser_for_super_calls,
        )
    call = rules.caller_with_bonus if caller else 0
    return Stake(rules.win_base + k + call + risk, call + risk, 1)


def expected_points(o: Outlook, k: int, s: Stake, others_make: float, rules: RulesConfig) -> float:
    """Expected points of aiming at ``k``, with SCR-5 from the chance each other player makes
    their estimate."""
    make = make_chance(o, k)
    win_bonus = rules.only_bonus if s.solo_win else 0
    lose_bonus = rules.only_bonus if s.solo_lose else 0
    solo_win = win_bonus * (1 - others_make) ** 3
    solo_lose = lose_bonus * others_make**3
    return (
        make * (s.win + solo_win)
        - (1 - make) * (s.lose + solo_lose)
        - s.per_trick * expected_miss(o, k)
    )
