"""The Elo ladder: rate bots that are too far apart to compare directly.

Bots are listed weakest first. Each neighbouring pair plays its own duplicate run (lineup A, B, A,
B), and one Bradley-Terry fit over every pair's set results rates them all, with RandomBot fixed
at 400 (SPEC §6 M2). Neighbours close enough that the weaker one wins some sets give each step a
finite, measured size, so ratings far from the anchor are measured too.

Each pair's report is saved under ``<dir>/pairs/``, and a pair already played with the same sets
and seed is reused, so the ladder can be extended or re-rated without replaying it.

Example: ``uv run python -m estimation.arena.ladder --sets 200 --workers 5``.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.bots import make_bot

from .duplicate import SetResult, run_arena
from .ratings import (
    ANCHOR,
    ANCHOR_RATING,
    DEFAULT_PRIOR,
    PairRecord,
    bradley_terry,
    set_first_places,
    sweep_gap,
)
from .report import build_report, write_json

DEFAULT_LADDER = (
    "random",
    "heuristic:noise=0.75",
    "heuristic:noise=0.5",
    "heuristic:noise=0.35",
    "heuristic:noise=0.27",
    "heuristic:noise=0.2",
    "heuristic:noise=0.1",
    "heuristic:noise=0.05",
    "heuristic:noise=0.02",
    "heuristic:noise=0.01",
    "heuristic",
    "pimc:samples=1:play_noise=0.3",
    "pimc:samples=1",
    "pimc:samples=2",
    "pimc:samples=4",
    "pimc:samples=8",
    "pimc:samples=16",
    "pimc",
)
"""Calibrated by 20- and 50-set pilots so that each bot wins roughly 60-95% of its sets against
the one below it."""


@dataclass(frozen=True, slots=True)
class PairResult:
    """One pair's run: its set results (from the lower bot's side) and the upper bot's RSA."""

    lower: str
    upper: str
    record: PairRecord
    upper_rsa: float
    upper_rsa_low: float | None
    upper_rsa_high: float | None
    upper_first: float  # share of games one of the stronger bot's two seats finished first
    seconds: float  # 0 when reused


def ladder_pairs(
    bots: Sequence[str], extras: Sequence[tuple[str, str]] = ()
) -> list[tuple[str, str]]:
    """Every neighbouring pair, weaker bot first, then each extra pair not already listed."""
    pairs = [(bots[i], bots[i + 1]) for i in range(len(bots) - 1)]
    for a, b in extras:
        if (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))
    return pairs


def _slug(bot: str) -> str:
    return bot.replace(":", "_").replace("=", "-")


def _pair_file(folder: Path, lower: str, upper: str) -> Path:
    return folder / "pairs" / f"{_slug(lower)}--{_slug(upper)}.json"


def _from_report(lower: str, upper: str, report: dict[str, Any], seconds: float) -> PairResult:
    (pair,) = report["pairs"]
    record = PairRecord(pair["a"], pair["b"], pair["a_wins"], pair["draws"], pair["b_wins"])
    rsa = next(bot["rsa"] for bot in report["bots"] if bot["name"] == upper)
    lineup = report["lineup"]
    shares = [
        set_first_places(
            SetResult(r["index"], r["deal_seed"], tuple(tuple(s) for s in r["scores"]), ()),
            lineup,
        )[upper]
        for r in report["set_results"]
    ]
    first = 2 * sum(shares) / len(shares)  # the stronger bot holds two of the four seats
    return PairResult(lower, upper, record, rsa["mean"], rsa["low"], rsa["high"], first, seconds)


def _saved(path: Path, lineup: list[str], sets: int, seed: int) -> dict[str, Any] | None:
    """The saved report of this pair, if it was played with the same lineup, sets and seed."""
    if not path.exists():
        return None
    report: dict[str, Any] = json.loads(path.read_text())["report"]
    same = report["lineup"] == lineup and report["sets"] == sets and report["seed"] == seed
    return report if same and not report["rules"] else None


def play_pair(
    lower: str, upper: str, sets: int, seed: int, workers: int, folder: Path
) -> tuple[PairResult, bool]:
    """Rate one pair, reusing its saved report when it matches. Returns the result and whether
    it was reused."""
    lineup = [lower, upper, lower, upper]
    path = _pair_file(folder, lower, upper)
    saved = _saved(path, lineup, sets, seed)
    if saved is not None:
        return _from_report(lower, upper, saved, 0.0), True
    run = run_arena(tuple(lineup), sets, seed, workers=workers)
    report = build_report(run)
    write_json(path, report, run)
    return _from_report(
        lower, upper, json.loads(path.read_text())["report"], run.wall_seconds
    ), False


def _signed(value: float) -> str:
    return f"{value:+,.1f}"


def format_ladder(
    bots: Sequence[str],
    results: Sequence[PairResult],
    ratings: dict[str, float],
    sets: int,
    seed: int,
    error: str | None,
) -> str:
    lines = [
        f"Elo ladder: {len(bots)} bots, {len(results)} pairs, {sets:,} sets per pair, seed {seed}",
        "",
    ]
    width = max(len("Bot"), *(len(bot) for bot in bots))
    lines.append(f"Rung  {'Bot':<{width}}  Elo")
    lines.append("-" * (width + 16))
    for rung, bot in enumerate(bots, 1):
        if bot in ratings:
            elo = f"{ratings[bot]:,.0f}" + (" (anchor)" if bot == ANCHOR else "")
        else:
            elo = "n/a"
        lines.append(f"{rung:>4}  {bot:<{width}}  {elo}")
    lines.append("")
    pair_width = max(
        len("Pair (weaker vs stronger)"), *(len(f"{r.lower} vs {r.upper}") for r in results)
    )
    lines.append(
        f"{'Pair (weaker vs stronger)':<{pair_width}}  {'Stronger won':>12}  {'Drew':>4}  "
        f"{'Weaker won':>10}  {'Stronger 1st':>12}  Stronger RSA per game (95% CI)"
    )
    for r in results:
        upper_wins, draws, lower_wins = r.record.b_wins, r.record.draws, r.record.a_wins
        if r.record.a != r.lower:
            upper_wins, lower_wins = r.record.a_wins, r.record.b_wins
        ci = (
            f"[{_signed(r.upper_rsa_low)}, {_signed(r.upper_rsa_high)}]"
            if r.upper_rsa_low is not None and r.upper_rsa_high is not None
            else "n/a"
        )
        lines.append(
            f"{f'{r.lower} vs {r.upper}':<{pair_width}}  {upper_wins:>12,}  {draws:>4,}  "
            f"{lower_wins:>10,}  {f'{100 * r.upper_first:.0f}%':>12}  {_signed(r.upper_rsa)} {ci}"
        )
    lines.append("")
    lines.append(
        f"Elo: one Bradley-Terry fit over every pair's sets, {ANCHOR} fixed at "
        f"{ANCHOR_RATING:.0f}, with 1 virtual draw per pair."
    )
    if error:
        lines.append(f"Elo not fitted: {error}")
    sweeps = [r for r in results if r.record.a_wins == sets or r.record.b_wins == sets]
    for r in sweeps:
        gap = sweep_gap(sets)
        bound = f"at least {gap:,.0f} Elo" if gap > 0 else "too few sets to bound the gap"
        lines.append(
            f"Sweep: one side won all {sets:,} sets in {r.lower} vs {r.upper} ({bound}); that "
            "step's size comes from the virtual draw, not the data."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena.ladder",
        description="Rate bots listed weakest first from duplicate runs of each neighbour pair.",
    )
    parser.add_argument(
        "--bots",
        default=",".join(DEFAULT_LADDER),
        help="bots weakest first, comma-separated (default: random up to pimc)",
    )
    parser.add_argument("--sets", type=int, required=True, help="duplicate sets per pair")
    parser.add_argument("--seed", type=int, default=0, help="run seed for every pair (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument(
        "--dir", type=Path, default=Path("runs/ladder"), help="where pair reports are kept"
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="an extra pair to rate, as weaker,stronger (repeatable)",
    )
    args = parser.parse_args(argv)

    bots = [bot.strip() for bot in args.bots.split(",")]
    if len(bots) < 2 or len(set(bots)) != len(bots) or not all(bots):
        parser.error("--bots needs at least 2 different bots")
    for bot in bots:
        try:
            make_bot(bot, 0)
        except ValueError as error:
            parser.error(str(error))
    extras: list[tuple[str, str]] = []
    for text in args.extra:
        pair = [bot.strip() for bot in text.split(",")]
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(bots):
            parser.error(f"--extra {text!r} must name two different bots from --bots")
        extras.append((pair[0], pair[1]))
    if args.sets < 1 or args.workers < 1 or args.seed < 0:
        parser.error("--sets and --workers must be at least 1, --seed 0 or more")

    pairs = ladder_pairs(bots, extras)
    start = time.perf_counter()
    results: list[PairResult] = []
    played = 0
    for n, (lower, upper) in enumerate(pairs, 1):
        result, reused = play_pair(lower, upper, args.sets, args.seed, args.workers, args.dir)
        results.append(result)
        played += not reused
        note = "reused" if reused else f"{result.seconds:,.0f} s"
        print(f"[ladder] {n}/{len(pairs)} {lower} vs {upper}: {note}", file=sys.stderr, flush=True)

    error = None
    try:
        ratings = bradley_terry([r.record for r in results], prior=DEFAULT_PRIOR)
    except ValueError as exc:
        ratings, error = {}, str(exc)
    print(format_ladder(bots, results, ratings, args.sets, args.seed, error))
    print()
    wall = time.perf_counter() - start
    print(
        f"Run: {wall:,.1f} s wall time on {args.workers} worker"
        f"{'s' if args.workers != 1 else ''}; {played} pairs played, {len(pairs) - played} reused."
    )
    summary = {
        "bots": bots,
        "sets": args.sets,
        "seed": args.seed,
        "ratings": ratings,
        "error": error,
        "pairs": [
            {
                "weaker": r.lower,
                "stronger": r.upper,
                "a": r.record.a,
                "b": r.record.b,
                "a_wins": r.record.a_wins,
                "draws": r.record.draws,
                "b_wins": r.record.b_wins,
                "stronger_rsa": r.upper_rsa,
                "stronger_rsa_low": r.upper_rsa_low,
                "stronger_rsa_high": r.upper_rsa_high,
                "stronger_first_place": r.upper_first,
            }
            for r in results
        ],
    }
    args.dir.mkdir(parents=True, exist_ok=True)
    (args.dir / "ladder.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
