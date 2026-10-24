"""Decision-time statistics: mean, maximum and percentiles, mergeable across worker processes."""

import pytest

from estimation.arena.timing import Timing


def test_counts_totals_and_maximum():
    timing = Timing()
    for seconds in (0.010, 0.020, 0.030):
        timing.add(seconds)
    assert timing.count == 3
    assert timing.mean == pytest.approx(0.020)
    assert timing.max == pytest.approx(0.030)


def test_quantiles_are_within_two_percent():
    timing = Timing()
    for ms in range(1, 1001):
        timing.add(ms / 1000)
    assert timing.quantile(0.5) == pytest.approx(0.500, rel=0.03)
    assert timing.quantile(0.99) == pytest.approx(0.990, rel=0.03)
    assert timing.quantile(1.0) == pytest.approx(1.0, rel=0.03)


def test_merging_equals_adding_everything_to_one():
    one, two, both = Timing(), Timing(), Timing()
    for i in range(50):
        (one if i % 2 else two).add(i / 1000)
        both.add(i / 1000)
    one.merge(two)
    assert one == both


def test_round_trips_through_a_dict():
    timing = Timing()
    for seconds in (0.0, 1e-7, 0.004, 0.25):
        timing.add(seconds)
    assert Timing.from_dict(timing.to_dict()) == timing


def test_an_empty_timing_reports_zeroes():
    timing = Timing()
    assert timing.mean == 0.0
    assert timing.quantile(0.99) == 0.0
    assert timing.over == 0


def test_counts_decisions_over_its_budget_exactly():
    timing = Timing(budget=0.25)
    for seconds in (0.1, 0.2, 0.25, 0.2501, 0.4):
        timing.add(seconds)
    assert timing.over == 2


def test_merging_needs_the_same_budget():
    with pytest.raises(ValueError, match="budget"):
        Timing(budget=0.25).merge(Timing(budget=0.1))
