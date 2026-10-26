"""Fit the yosif bot's calibration from saved games (see ``bots/yosif/calibration``).

Each played round of a saved game gives, for every seat held by one of ``--specs``: the counting's
numbers for that seat's hand as the bot would compute them (expected tricks trying to win and to
lose, trumps held, Sans, the last estimator's Over or Under), the estimate, and the tricks taken.
Dash callers give the counting's pressed chance of no trick and whether the Dash Call was made.

The fit is a least-squares line for the natural trick count, then measured shares: how often calls
are made at each distance from that count (half-trick steps), how many tricks off the misses are,
and how often Dash Calls are made by the counting's chance (tenths). Only distances and chances
with enough rows are kept, so the curves follow the data where there is data.

Example: ``uv run python -m estimation.eval.calibrate --from runs/ladder-v2``.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from estimation.arena.games import code_version, games_files, read_games, rules_of
from estimation.bots.yosif import YosifConfig
from estimation.bots.yosif.calibration import CALIBRATION_FILE, Calibration
from estimation.bots.yosif.counting import read_bids
from estimation.bots.yosif.policy import Planner
from estimation.engine import (
    SANS,
    Bola,
    IllegalActionError,
    Observation,
    Phase,
    RoundResult,
    RulesConfig,
)
from estimation.eval.measure import Trial, read_trials

DEFAULT_SPECS = ("pimc", "pimc:samples=16", "pimc:samples=8", "yosif")  # strong card players
PER_FILE = 400  # games read per games file
STEP = 0.5  # width of a distance bin, in tricks
MIN_ROWS = 200  # rows a distance bin needs to be kept
MIN_DASH_ROWS = 30  # rows a Dash Call bin needs to be kept


@dataclass(frozen=True, slots=True)
class Row:
    spec: str
    estimate: int
    tricks: int
    won: bool
    em: float  # the counting's expected tricks trying to win
    el: float  # trying to lose
    trumps: int
    sans: bool
    last_over: int  # +1 the last estimator going Over, -1 Under, 0 for anyone else
    dash: bool
    zero: float  # for a Dash caller: the counting's pressed chance of no trick


def extract_rows(
    paths: Iterable[Path], specs: set[str], per_file: int, rules: RulesConfig | None = None
) -> list[Row]:
    """Rows from the saved games at ``paths``, for seats held by ``specs``."""
    planner = Planner(YosifConfig(calibrated=False))
    rows: list[Row] = []
    for path in paths:
        header, records, _ = read_games(path)
        if header is not None and not specs & set(header.lineup):
            continue  # none of these players sat in this run
        game_rules = rules or rules_of(header)
        for record in records[:per_file]:
            try:
                rows += _game_rows(
                    record.deal_seed, record.actions, record.seats, specs, planner, game_rules
                )
            except IllegalActionError:
                continue  # a game played by other rules: skipped
    return rows


def _game_rows(
    seed: int,
    actions: Sequence[int],
    seats: Sequence[str],
    specs: set[str],
    planner: Planner,
    rules: RulesConfig,
) -> list[Row]:
    bola = Bola(seed, rules)
    rows: list[Row] = []
    snapshot: list[Observation] | None = None  # every seat at the round's first card
    rounds = 0
    for action in actions:
        if bola.phase is Phase.PLAY and snapshot is None:
            snapshot = [bola.observation(s) for s in range(4)]
        bola.step(action)
        if len(bola.history) > rounds:
            rounds = len(bola.history)
            result = bola.history[-1]
            if not result.skipped and snapshot is not None and result.trump is not None:
                for seat in range(4):
                    if seats[seat] in specs:
                        rows.append(_row(seats[seat], result, seat, snapshot[seat], planner))
            snapshot = None
    return rows


def _row(spec: str, result: RoundResult, seat: int, obs: Observation, planner: Planner) -> Row:
    s = result.seats[seat]
    trump = result.trump
    if trump is None:
        raise ValueError("a played round has a trump")
    reads = read_bids(obs)
    hand = obs.hand
    over = bool(result.over) if s.is_last else None
    o = planner.counting_outlook(hand, trump, reads, over)
    return Row(
        spec=spec,
        estimate=s.estimate,
        tricks=s.tricks,
        won=s.won,
        em=sum(k * p for k, p in enumerate(o.most)),
        el=sum(k * p for k, p in enumerate(o.least)),
        trumps=0 if trump == SANS else ((hand >> (13 * trump)) & 0x1FFF).bit_count(),
        sans=trump == SANS,
        last_over=0 if over is None else 1 if over else -1,
        dash=s.is_dash,
        zero=planner.dash_zero_chance(hand, reads) if s.is_dash else 0.0,
    )


def fit(
    rows: Sequence[Row],
    source: dict[str, object] | None = None,
    min_rows: int = MIN_ROWS,
    min_dash_rows: int = MIN_DASH_ROWS,
    trials: Sequence[Trial] | None = None,
) -> Calibration:
    """The calibration that best describes ``rows`` (see the module docstring). With play-out
    ``trials`` (``measure``), the make, miss and Dash curves come from them instead: their calls
    were assigned, not chosen, so calls nobody would choose are measured too. The natural count
    always comes from ``rows``."""
    plays = [r for r in rows if not r.dash]
    if not plays:
        raise ValueError("no estimating seats to fit")
    x = np.array([[1.0, r.em, r.el, r.trumps, float(r.sans), float(r.last_over)] for r in plays])
    tricks = np.array([float(r.tricks) for r in plays])
    weights, *_ = np.linalg.lstsq(x, tricks, rcond=None)
    natural = x @ weights
    spread = float(np.std(tricks - natural))
    distance = np.array([r.estimate for r in plays]) - natural
    won = np.array([float(r.won) for r in plays])
    off = np.abs(tricks - np.array([r.estimate for r in plays]))
    dash_points = [(r.zero, r.won) for r in rows if r.dash]
    if trials:
        played = [t for t in trials if t.scenario != "dash"]
        tx = np.array([[1.0, t.em, t.el, t.trumps, float(t.sans), 0.0] for t in played])
        calls = np.array([float(t.call) for t in played])
        distance = calls - tx @ weights
        won = np.array([float(t.made) for t in played])
        off = np.abs(np.array([float(t.tricks) for t in played]) - calls)
        dash_points = [(t.zero, t.made) for t in trials if t.scenario == "dash"]
    offsets, made, miss = [], [], []
    for centre in np.arange(-4.0, 4.0 + STEP, STEP):
        inside = np.abs(distance - centre) <= STEP / 2
        if inside.sum() >= min_rows:
            offsets.append(float(centre))
            made.append(float(won[inside].mean()))
            missed = inside & (won == 0)
            miss.append(float(off[missed].mean()) if missed.any() else 1.0)
    if not offsets:
        raise ValueError("too few rows to measure make chances")
    dash_p0, dash_made = [], []
    for low in np.arange(0.0, 1.0, 0.1):
        inside = [
            won
            for zero, won in dash_points
            if low <= zero < low + 0.1 or (low >= 0.9 and zero == 1)
        ]
        if len(inside) >= min_dash_rows:
            dash_p0.append(float(low + 0.05))
            dash_made.append(sum(inside) / len(inside))
    if not dash_p0:  # too few Dash Calls to measure: the counting's chance, as before
        dash_p0, dash_made = [0.0, 1.0], [0.0, 1.0]
    w = [float(v) for v in weights]
    return Calibration(
        intercept=w[0],
        em=w[1],
        el=w[2],
        trumps=w[3],
        sans=w[4],
        fixed=0.0,
        last_over=w[5],
        offsets=tuple(offsets),
        made=tuple(made),
        miss=tuple(miss),
        dash_p0=tuple(dash_p0),
        dash_made=tuple(dash_made),
        spread=spread,
        source={
            **(source or {}),
            "rows": len(plays),
            "dash_rows": sum(r.dash for r in rows),
            "trials": len(trials or ()),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.eval.calibrate",
        description="Fit the yosif bot's calibration from saved games.",
    )
    parser.add_argument("--from", dest="source", type=Path, required=True, help="a run or file")
    parser.add_argument(
        "--specs", default=",".join(DEFAULT_SPECS), help="seats to learn from, comma-separated"
    )
    parser.add_argument("--per-file", type=int, default=PER_FILE, help="games per file (400)")
    parser.add_argument("--out", type=Path, default=CALIBRATION_FILE, help="where to write it")
    parser.add_argument(
        "--playouts", type=Path, help="play-out trials (measure) for the make and Dash curves"
    )
    parser.add_argument(
        "--min-rows", type=int, default=MIN_ROWS, help="rows a distance step needs (200)"
    )
    args = parser.parse_args(argv)
    files = games_files(args.source)
    if not files:
        parser.error(f"no saved games under {args.source}")
    specs = {spec.strip() for spec in args.specs.split(",") if spec.strip()}
    rows = extract_rows(files, specs, args.per_file)
    if not rows:
        parser.error("no rows for those specs in these games")
    source = {
        "games": str(args.source),
        "specs": sorted(specs),
        "per_file": args.per_file,
        "code_version": code_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    trials = None
    if args.playouts is not None:
        playout_header, trials = read_trials(args.playouts)
        source["playouts"] = {"file": str(args.playouts), **playout_header}
    try:
        calibration = fit(rows, source, args.min_rows, max(1, args.min_rows // 7), trials)
    except ValueError as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration.to_json(), indent=1) + "\n")
    print(
        f"Natural tricks = {calibration.intercept:.2f} + {calibration.em:.2f} x count trying to win"
        f" + {calibration.el:.2f} x count trying to lose + {calibration.trumps:.2f} per trump"
        f" + {calibration.sans:.2f} in Sans, from {calibration.source['rows']:,} seat-rounds."
    )
    made = ", ".join(
        f"{o:+.1f}: {m:.0%}" for o, m in zip(calibration.offsets, calibration.made, strict=True)
    )
    print(f"Calls made, by distance from the natural count: {made}")
    dash = ", ".join(
        f"{p:.2f}: {m:.0%}" for p, m in zip(calibration.dash_p0, calibration.dash_made, strict=True)
    )
    print(f"Dash Calls made, by the counting's chance: {dash}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
