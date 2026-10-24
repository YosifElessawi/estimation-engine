"""Decision-time statistics for arena reports: count, mean, maximum, percentiles and how many
decisions went over a time budget.

Times go into log-spaced buckets 2% wide, so percentiles are good to 2%, any number of decisions
fits in a small dict, and the timings of worker processes merge exactly.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

DEFAULT_BUDGET = 0.25  # seconds per card decision (SPEC §6 M3)

_FLOOR = 1e-6  # decisions faster than a microsecond share the first bucket
_RATIO = 1.02
_LOG_RATIO = math.log(_RATIO)


@dataclass(slots=True)
class Timing:
    budget: float = DEFAULT_BUDGET  # seconds; ``over`` counts decisions slower than this
    count: int = 0
    total_ns: int = 0
    max: float = 0.0
    over: int = 0
    buckets: dict[int, int] = field(default_factory=dict[int, int])

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total_ns += round(seconds * 1e9)
        if seconds > self.max:
            self.max = seconds
        if seconds > self.budget:
            self.over += 1
        index = 0 if seconds <= _FLOOR else int(math.log(seconds / _FLOOR) / _LOG_RATIO) + 1
        self.buckets[index] = self.buckets.get(index, 0) + 1

    def merge(self, other: "Timing") -> None:
        if other.budget != self.budget:
            raise ValueError("cannot merge timings with different budgets")
        self.count += other.count
        self.total_ns += other.total_ns
        self.max = max(self.max, other.max)
        self.over += other.over
        for index, n in other.buckets.items():
            self.buckets[index] = self.buckets.get(index, 0) + n

    @property
    def mean(self) -> float:
        """Mean seconds per decision."""
        return self.total_ns / self.count / 1e9 if self.count else 0.0

    def quantile(self, q: float) -> float:
        """Seconds within which a fraction ``q`` of decisions finished, good to 2%."""
        if not self.count:
            return 0.0
        rank = max(1, math.ceil(q * self.count))
        seen = 0
        for index in sorted(self.buckets):
            seen += self.buckets[index]
            if seen >= rank:
                return min(_FLOOR * _RATIO**index, self.max)
        return self.max

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "count": self.count,
            "total_ns": self.total_ns,
            "max": self.max,
            "over": self.over,
            "buckets": {str(index): n for index, n in sorted(self.buckets.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """The inverse of ``to_dict``, for timings read back from a JSON report."""
        return cls(
            budget=float(data["budget"]),
            count=int(data["count"]),
            total_ns=int(data["total_ns"]),
            max=float(data["max"]),
            over=int(data["over"]),
            buckets={int(index): int(n) for index, n in data["buckets"].items()},
        )
