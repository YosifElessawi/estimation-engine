"""Arena metrics (SPEC §6 M2): Relative Score Advantage with a 95% confidence interval over
duplicate sets, and Bradley-Terry Elo with RandomBot fixed at 400."""

import math

import pytest

from estimation.arena.duplicate import SetResult
from estimation.arena.ratings import (
    PairRecord,
    bradley_terry,
    mean_ci,
    pair_records,
    rsa,
    set_first_places,
    set_rsa,
    set_scores,
    sweep_gap,
    t_quantile,
)

AB = ("a", "b", "a", "b")


def make_set(index: int, scores: list[tuple[int, int, int, int]]) -> SetResult:
    return SetResult(index=index, deal_seed=index, scores=tuple(scores), rounds=(18, 18, 18, 18))


# Relative Score Advantage


def test_rsa_is_own_score_minus_the_mean_of_the_other_three():
    assert rsa((40, 10, 10, -20), 0) == pytest.approx(40 - 0)
    assert rsa((40, 10, 10, -20), 3) == pytest.approx(-20 - 20)
    assert sum(rsa((40, 10, 10, -20), seat) for seat in range(4)) == pytest.approx(0)


def test_set_rsa_averages_a_bot_over_its_seats_and_rotations():
    # Rotation k seats slot (i + k) % 4 at seat i, so slot j sits at seat (j - k) % 4.
    scores = [(30, 0, 30, 0), (0, 30, 0, 30), (12, 0, 0, 0), (0, 0, 0, 12)]
    result = make_set(0, scores)
    per_bot = set_rsa(result, AB)
    a_values = [rsa(scores[k], (j - k) % 4) for k in range(4) for j in (0, 2)]
    assert per_bot["a"] == pytest.approx(sum(a_values) / 8)
    assert per_bot["a"] + per_bot["b"] == pytest.approx(0)


def test_set_scores_average_a_bot_over_its_seat_games():
    scores = [(30, 0, 30, 0), (0, 30, 0, 30), (12, 0, 0, 0), (0, 0, 0, 12)]
    per_bot = set_scores(make_set(0, scores), AB)
    # a sits at seats 0, 2 / 3, 1 / 2, 0 / 1, 3 in rotations 0-3.
    assert per_bot == {"a": pytest.approx((30 + 30 + 30 + 30 + 0 + 12 + 0 + 12) / 8), "b": 0.0}


def test_first_place_share_splits_ties_and_averages_over_seat_games():
    # Rotation k seats slot (i + k) % 4 at seat i; a holds slots 0 and 2, b slots 1 and 3.
    scores = [(30, 0, 30, 0), (30, 0, 30, 0), (12, 0, 0, 0), (0, 12, 9, 0)]
    shares = set_first_places(make_set(0, scores), AB)
    # k=0: a ties at seats 0 and 2 (half each). k=1: b ties at seats 0 and 2. k=2: a wins at
    # seat 0. k=3: a wins at seat 1. So a has 3 first places in its 8 games, b has 1.
    assert shares == {"a": pytest.approx(3 / 8), "b": pytest.approx(1 / 8)}
    assert 2 * shares["a"] + 2 * shares["b"] == pytest.approx(1)  # one first place per game


def test_mean_ci_is_a_student_t_interval():
    interval = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    half = 2.7764451 * math.sqrt(2.5) / math.sqrt(5)
    assert interval.mean == pytest.approx(3.0)
    assert interval.low == pytest.approx(3.0 - half, rel=1e-4)
    assert interval.high == pytest.approx(3.0 + half, rel=1e-4)
    assert interval.n == 5


def test_mean_ci_of_a_single_set_is_unbounded():
    interval = mean_ci([2.0])
    assert interval.mean == 2.0
    assert interval.low == -math.inf
    assert interval.high == math.inf


@pytest.mark.parametrize(
    ("df", "value"),
    [
        (1, 12.7062),
        (2, 4.3027),
        (4, 2.7764),
        (10, 2.2281),
        (30, 2.0423),
        (199, 1.9720),
        (999, 1.9623),
    ],
)
def test_t_quantile_matches_tables(df: int, value: float):
    assert t_quantile(0.975, df) == pytest.approx(value, abs=2e-4)


# Pairwise results and Elo


def test_each_set_compares_every_pair_of_different_bots_by_set_total():
    a_seats = [(10, 0, 10, 0), (0, 10, 0, 10), (10, 0, 10, 0), (0, 10, 0, 10)]
    b_seats = [(0, 5, 0, 5), (5, 0, 5, 0), (0, 5, 0, 5), (5, 0, 5, 0)]
    sets = [make_set(0, a_seats), make_set(1, [(0, 0, 0, 0)] * 4), make_set(2, b_seats)]
    sets_ab = pair_records(sets, AB)
    assert sets_ab == [PairRecord("a", "b", a_wins=1, draws=1, b_wins=1)]


def test_bots_in_more_slots_are_compared_by_their_mean_seat_score():
    lineup = ("a", "b", "b", "b")
    # a scores 9 in each of its 4 Bolas and each b seat 3 in its 12: the totals tie, the means
    # do not.
    scores = [(9, 3, 3, 3), (3, 3, 3, 9), (3, 3, 9, 3), (3, 9, 3, 3)]
    record = pair_records([make_set(0, scores)], lineup)
    assert record == [PairRecord("a", "b", a_wins=1, draws=0, b_wins=0)]


def test_bradley_terry_two_bots_matches_the_closed_form():
    ratings = bradley_terry([PairRecord("a", "random", a_wins=75, draws=0, b_wins=25)], prior=0.0)
    assert ratings["random"] == pytest.approx(400.0)
    assert ratings["a"] == pytest.approx(400 + 400 * math.log10(3), abs=1e-6)


def test_bradley_terry_counts_draws_as_half_wins():
    ratings = bradley_terry([PairRecord("random", "a", a_wins=10, draws=20, b_wins=10)], prior=0.0)
    assert ratings["a"] == pytest.approx(400.0, abs=1e-6)


def test_bradley_terry_stays_finite_when_a_bot_wins_every_set():
    ratings = bradley_terry([PairRecord("a", "random", a_wins=100, draws=0, b_wins=0)], prior=1.0)
    assert ratings["a"] == pytest.approx(400 + 400 * math.log10(100.5 / 0.5), abs=1e-6)


def test_bradley_terry_chains_through_shared_opponents():
    pairs = [
        PairRecord("a", "b", a_wins=30, draws=0, b_wins=10),
        PairRecord("b", "random", a_wins=30, draws=0, b_wins=10),
    ]
    ratings = bradley_terry(pairs, prior=0.0)
    step = 400 * math.log10(3)
    assert ratings["b"] == pytest.approx(400 + step, abs=1e-6)
    assert ratings["a"] == pytest.approx(400 + 2 * step, abs=1e-6)


def test_bradley_terry_merges_records_of_the_same_pair_either_way_round():
    split = [
        PairRecord("a", "random", a_wins=50, draws=0, b_wins=10),
        PairRecord("random", "a", a_wins=15, draws=0, b_wins=25),
    ]
    whole = [PairRecord("a", "random", a_wins=75, draws=0, b_wins=25)]
    assert bradley_terry(split, prior=0.0) == pytest.approx(bradley_terry(whole, prior=0.0))


def test_bots_not_connected_to_the_anchor_are_unrated():
    pairs = [
        PairRecord("a", "random", a_wins=3, draws=0, b_wins=1),
        PairRecord("x", "y", a_wins=3, draws=0, b_wins=1),
    ]
    assert set(bradley_terry(pairs)) == {"a", "random"}
    assert bradley_terry([PairRecord("x", "y", a_wins=3, draws=0, b_wins=1)]) == {}


def test_a_clean_sweep_bounds_the_elo_gap_from_below():
    # Winning all n comparisons puts the win rate above 0.05 ** (1 / n) at 95% confidence.
    p = 0.05 ** (1 / 1000)
    assert sweep_gap(1000) == pytest.approx(400 * math.log10(p / (1 - p)))
    assert sweep_gap(1000) == pytest.approx(1009.2, abs=0.1)
    assert sweep_gap(20) < sweep_gap(200) < sweep_gap(2000)


def _set_won_by(lineup: list[str], winner: str, index: int) -> SetResult:
    """A duplicate set where every seat ``winner`` plays scores 10 and every other seat 0."""
    scores = []
    for rotation in range(4):
        row = [10 if lineup[(seat + rotation) % 4] == winner else 0 for seat in range(4)]
        scores.append((row[0], row[1], row[2], row[3]))
    return SetResult(index, index, tuple(scores), (18, 18, 18, 18))


def _chain(n: int) -> dict[tuple[str, str], tuple[list[SetResult], list[str]]]:
    """random < x < y: x wins 80% of its sets against random, y 70% against x."""
    out: dict[tuple[str, str], tuple[list[SetResult], list[str]]] = {}
    for lower, upper, share in (("random", "x", 0.8), ("x", "y", 0.7)):
        lineup = [lower, upper, lower, upper]
        wins = round(n * share)
        sets = [_set_won_by(lineup, upper if i < wins else lower, i) for i in range(n)]
        out[(lower, upper)] = (sets, lineup)
    return out


def test_bootstrap_range_contains_the_fit_and_narrows_with_more_sets():
    from estimation.arena.ratings import DEFAULT_PRIOR, bootstrap_ranges

    def fit(pairs: dict[tuple[str, str], tuple[list[SetResult], list[str]]]) -> dict[str, float]:
        records = [r for sets, lineup in pairs.values() for r in pair_records(sets, lineup)]
        return bradley_terry(records, prior=DEFAULT_PRIOR)

    few, many = _chain(10), _chain(80)
    ranges_few, ranges_many = bootstrap_ranges(few), bootstrap_ranges(many)
    for bot in ("x", "y"):
        low, high = ranges_few[bot]
        assert low <= fit(few)[bot] <= high
        assert ranges_many[bot][1] - ranges_many[bot][0] < high - low
    assert ranges_few["random"] == (400.0, 400.0)
