"""The yosif bot's counting, calibrated on saved games (tuning from the ladder's games).

Yosif's rules still count the hand (``counting``, ``odds``). The calibration turns that count into
what hands really do in games between strong card players, measured on the saved ladder games
by ``estimation.eval.calibrate``:

- the natural trick count: tricks a hand really takes, a fitted line in the counting's own
  numbers (its expected tricks trying to win and trying to lose, trumps held, Sans, a fixed-trump
  round, and the last estimator's Over or Under);
- the make chance of a call, by its distance from that natural count;
- the size of a miss, by the same distance;
- the success of a Dash Call, by the counting's chance of taking no trick under pressure.

The fitted numbers live in ``calibration.json`` next to this file, written by the fitting tool
(never by hand). Without that file the bot uses the counting's own chances.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
Points = tuple[float, ...]


def _interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Piecewise-linear through the points, flat beyond the ends."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            share = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + share * (ys[i] - ys[i - 1])
    return ys[-1]


@dataclass(frozen=True, slots=True)
class Calibration:
    intercept: float
    em: float  # per expected trick trying to win
    el: float  # per expected trick trying to lose
    trumps: float  # per trump held
    sans: float  # in a Sans round
    fixed: float  # in a fixed-trump round
    last_over: float  # the last estimator: +1 going Over, -1 going Under
    offsets: tuple[float, ...]  # a call's distance from the natural count
    made: tuple[float, ...]  # how often calls at that distance are made
    miss: tuple[float, ...]  # mean tricks off when they are missed
    dash_p0: tuple[float, ...]  # the counting's chance of no trick under pressure
    dash_made: tuple[float, ...]  # how often such Dash Calls are made
    spread: float = 1.1  # how far hands stray from the natural count, in tricks (residual sd)
    big_offsets: tuple[float, ...] = ()  # calls of ``big_from`` or more (super calls) have their
    big_made: tuple[float, ...] = ()  # own curve: they are harder to land exactly
    big_miss: tuple[float, ...] = ()
    big_from: int = 8
    source: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def natural(
        self, *, em: float, el: float, trumps: int, sans: bool, fixed: bool, last_over: int
    ) -> float:
        """The tricks a hand really takes, from the counting's numbers."""
        return (
            self.intercept
            + self.em * em
            + self.el * el
            + self.trumps * trumps
            + self.sans * sans
            + self.fixed * fixed
            + self.last_over * last_over
        )

    def _curve(
        self, call: int | None
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        if call is not None and call >= self.big_from and self.big_offsets:
            return self.big_offsets, self.big_made, self.big_miss
        return self.offsets, self.made, self.miss

    def make(self, distance: float, call: int | None = None) -> float:
        """The chance of making ``call``, ``distance`` tricks above the natural count. Inside the
        measured distances, the measured share; beyond them, falling away as hands spread
        (there are almost no calls that far off, so the edge share must not carry on)."""
        offsets, made, _ = self._curve(call)
        gap = max(offsets[0] - distance, distance - offsets[-1], 0.0)
        return _interp(distance, offsets, made) * math.exp(
            -(gap * gap) / (2 * self.spread * self.spread)
        )

    def miss_size(self, distance: float, call: int | None = None) -> float:
        """Mean tricks off when the call is missed: measured, then growing past the edges."""
        offsets, _, miss = self._curve(call)
        gap = max(offsets[0] - distance, distance - offsets[-1], 0.0)
        return _interp(distance, offsets, miss) + gap

    def dash(self, zero_chance: float) -> float:
        """The chance a Dash Call is made, from the counting's pressed chance of no trick."""
        return _interp(zero_chance, self.dash_p0, self.dash_made)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _from_json(data: dict[str, Any]) -> Calibration:
    return Calibration(
        intercept=float(data["intercept"]),
        em=float(data["em"]),
        el=float(data["el"]),
        trumps=float(data["trumps"]),
        sans=float(data["sans"]),
        fixed=float(data["fixed"]),
        last_over=float(data["last_over"]),
        offsets=_floats(data["offsets"]),
        made=_floats(data["made"]),
        miss=_floats(data["miss"]),
        dash_p0=_floats(data["dash_p0"]),
        dash_made=_floats(data["dash_made"]),
        spread=float(data.get("spread", 1.1)),
        big_offsets=_floats(data.get("big_offsets", ())),
        big_made=_floats(data.get("big_made", ())),
        big_miss=_floats(data.get("big_miss", ())),
        big_from=int(data.get("big_from", 8)),
        source=dict(data.get("source", {})),
    )


@cache
def load_calibration(path: Path = CALIBRATION_FILE) -> Calibration | None:
    """The fitted calibration, or None when the file does not exist."""
    if not path.is_file():
        return None
    return _from_json(json.loads(path.read_text()))
