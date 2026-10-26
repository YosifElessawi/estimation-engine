"""The Elo ladder: rate bots that are too far apart to compare directly.

Bots are listed weakest first. Each neighbouring pair plays its own duplicate run (lineup A, B, A,
B), and one Bradley-Terry fit over every pair's set results rates them all, with RandomBot fixed
at 400 (SPEC §6 M2). Neighbours close enough that the weaker one wins some sets give each step a
finite, measured size, so ratings far from the anchor are measured too.

Each pair's report is saved under ``<dir>/pairs/``, and a pair already played with the same sets
and seed is reused, so the ladder can be extended or re-rated without replaying it.

Every pair's games are saved under ``<dir>/games/`` (see ``games``), and a pair is reused only when
it was played with the same lineup, sets, seed and code version and its games are on disk.

Example: ``uv run python -m estimation.arena.ladder --sets 200 --workers 5``, or the full ladder
with ``--preset v2 --workers 5``.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.bots import make_bot

from .duplicate import SetResult, run_arena
from .games import code_version, make_header, write_games
from .ratings import (
    ANCHOR,
    ANCHOR_RATING,
    DEFAULT_PRIOR,
    PairRecord,
    bootstrap_ranges,
    bradley_terry,
    set_first_places,
    sweep_gap,
)
from .report import build_report, write_json
from .roster import GROUPS, entry_for, name_of
from .status import STATUS_FILE, StatusWriter

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

M4 = "rl:checkpoint=runs/m4/checkpoints/"
STAGED_FINAL = "rl:checkpoint=runs/m4-staged/checkpoints/staged-final.pt"
V2_LADDER = (
    *DEFAULT_LADDER[:11],  # random up to HeuristicBot
    M4 + "iter-0010.pt",
    "pimc:samples=1:play_noise=0.3",
    "pimc:samples=1",
    "pimc:samples=2",
    "pimc:samples=4",
    M4 + "warm-start.pt:heuristic_bidding=true",
    M4 + "iter-0020.pt",
    M4 + "iter-0060.pt",
    M4 + "iter-0100.pt",
    M4 + "final.pt",
    "pimc:samples=8",
    "pimc:samples=16",
    STAGED_FINAL,
    "pimc",
    "yosif",
)
"""The full ladder after the With-auction rules (2026-10-26): the default rungs, the M4 checkpoints
where the 2026-10-25 ladder placed them, and the yosif bot last."""
V2_EXTRAS = (
    ("heuristic", "pimc:samples=1:play_noise=0.3"),
    ("pimc:samples=4", "pimc:samples=8"),
    ("pimc:samples=2", M4 + "warm-start.pt:heuristic_bidding=true"),
    (M4 + "final.pt", "pimc:samples=16"),
    (M4 + "final.pt", "pimc"),
    ("pimc:samples=16", "pimc"),
    (M4 + "final.pt", STAGED_FINAL),
    ("pimc:samples=8", STAGED_FINAL),
    ("pimc", "yosif"),  # same card play, so this pair measures the yosif bot's bidding alone
    ("heuristic", "yosif"),
)


@dataclass(frozen=True, slots=True)
class Preset:
    """A named ladder: its bots weakest first, extra pairs, folder, sets per pair and seed."""

    bots: tuple[str, ...]
    extras: tuple[tuple[str, str], ...]
    dir: Path
    sets: int
    seed: int


PRESETS = {"v2": Preset(V2_LADDER, V2_EXTRAS, Path("runs/ladder-v2"), 200, 0)}


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
    sets: tuple[SetResult, ...] = ()  # the pair's set results, for the Elo ranges


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
    """A file name for ``bot``: options and checkpoint paths flattened into one name."""
    return bot.replace(":", "_").replace("=", "-").replace("/", "_").replace("\\", "_")


def _pair_file(folder: Path, lower: str, upper: str) -> Path:
    return folder / "pairs" / f"{_slug(lower)}--{_slug(upper)}.json"


def _games_file(folder: Path, lower: str, upper: str) -> Path:
    return folder / "games" / f"{_slug(lower)}--{_slug(upper)}.jsonl.gz"


def _from_report(lower: str, upper: str, report: dict[str, Any], seconds: float) -> PairResult:
    (pair,) = report["pairs"]
    record = PairRecord(pair["a"], pair["b"], pair["a_wins"], pair["draws"], pair["b_wins"])
    rsa = next(bot["rsa"] for bot in report["bots"] if bot["name"] == upper)
    lineup = report["lineup"]
    sets = tuple(
        SetResult(r["index"], r["deal_seed"], tuple(tuple(s) for s in r["scores"]), ())
        for r in report["set_results"]
    )
    shares = [set_first_places(result, lineup)[upper] for result in sets]
    first = 2 * sum(shares) / len(shares)  # the stronger bot holds two of the four seats
    return PairResult(
        lower, upper, record, rsa["mean"], rsa["low"], rsa["high"], first, seconds, sets
    )


def _saved(
    path: Path, games: Path, lineup: list[str], sets: int, seed: int, version: str
) -> dict[str, Any] | None:
    """The saved report of this pair, if it was played with the same lineup, sets, seed and code
    version, and its games are saved too."""
    if not path.exists() or not games.exists():
        return None
    report: dict[str, Any] = json.loads(path.read_text())["report"]
    same = report["lineup"] == lineup and report["sets"] == sets and report["seed"] == seed
    same = same and report.get("code_version") == version
    return report if same and not report["rules"] else None


def play_pair(
    lower: str,
    upper: str,
    sets: int,
    seed: int,
    workers: int,
    folder: Path,
    on_set: Callable[[SetResult], None] | None = None,
) -> tuple[PairResult, bool]:
    """Rate one pair, reusing its saved report when it matches. Returns the result and whether
    it was reused; ``on_set`` sees each set as it is played."""
    lineup = [lower, upper, lower, upper]
    path = _pair_file(folder, lower, upper)
    games = _games_file(folder, lower, upper)
    version = code_version()
    saved = _saved(path, games, lineup, sets, seed, version)
    if saved is not None:
        return _from_report(lower, upper, saved, 0.0), True
    run = run_arena(tuple(lineup), sets, seed, workers=workers, on_set=on_set, keep_games=True)
    report = build_report(run)
    write_json(path, report, run, version)
    header = make_header(lineup, sets, seed, run.config)
    write_games(games, header, (game for result in run.results for game in result.games))
    return _from_report(
        lower, upper, json.loads(path.read_text())["report"], run.wall_seconds
    ), False


def _summary(result: PairResult) -> dict[str, Any]:
    """A pair's results for the status file (see ``status.unit_results``)."""
    record = result.record
    return {
        "bots": [
            {"bot": result.upper, "rsa": result.upper_rsa, "first": result.upper_first},
            {"bot": result.lower, "rsa": None, "first": None},
        ],
        "record": {
            "a": record.a,
            "b": record.b,
            "a_wins": record.a_wins,
            "draws": record.draws,
            "b_wins": record.b_wins,
        },
    }


def _signed(value: float) -> str:
    return f"{value:+,.1f}"


def _elo(bot: str, ratings: dict[str, float]) -> str:
    if bot not in ratings:
        return "n/a"
    return f"{ratings[bot]:,.0f}" + (" (anchor)" if bot == ANCHOR else "")


def _range(bot: str, ranges: dict[str, tuple[float, float]]) -> str:
    if bot not in ranges or bot == ANCHOR:
        return ""
    low, high = ranges[bot]
    return f"{low:,.0f} to {high:,.0f}"


def format_ladder(
    bots: Sequence[str],
    results: Sequence[PairResult],
    ratings: dict[str, float],
    sets: int,
    seed: int,
    error: str | None,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> str:
    """The ladder as three named tables (see ``roster``), then every pair's results."""
    ranges = ranges or {}
    lines = [
        f"Elo ladder: {len(bots)} bots, {len(results)} pairs, {sets:,} sets per pair, seed {seed}",
    ]
    names = {bot: name_of(bot) for bot in bots}
    width = max(len("Bot"), *(len(name) for name in names.values()))
    for group in GROUPS:
        members = [bot for bot in bots if group in entry_for(bot).groups]
        if not members:
            continue
        members.sort(key=lambda bot: (bot not in ratings, -ratings.get(bot, 0.0)))
        lines += ["", group, f"  {'Bot':<{width}}  {'Elo':>14}  {'95% range':>16}  About"]
        for bot in members:
            lines.append(
                f"  {names[bot]:<{width}}  {_elo(bot, ratings):>14}  {_range(bot, ranges):>16}  "
                f"{entry_for(bot).description}"
            )
    lines.append("")
    label = "Pair (weaker vs stronger)"
    pair_names = [f"{names[r.lower]} vs {names[r.upper]}" for r in results]
    pair_width = max(len(label), *(len(name) for name in pair_names))
    lines.append(
        f"{label:<{pair_width}}  {'Stronger won':>12}  {'Drew':>4}  "
        f"{'Weaker won':>10}  {'Stronger 1st':>12}  Stronger RSA per game (95% CI)"
    )
    for r, pair_name in zip(results, pair_names, strict=True):
        upper_wins, draws, lower_wins = r.record.b_wins, r.record.draws, r.record.a_wins
        if r.record.a != r.lower:
            upper_wins, lower_wins = r.record.a_wins, r.record.b_wins
        ci = (
            f"[{_signed(r.upper_rsa_low)}, {_signed(r.upper_rsa_high)}]"
            if r.upper_rsa_low is not None and r.upper_rsa_high is not None
            else "n/a"
        )
        lines.append(
            f"{pair_name:<{pair_width}}  {upper_wins:>12,}  {draws:>4,}  "
            f"{lower_wins:>10,}  {f'{100 * r.upper_first:.0f}%':>12}  {_signed(r.upper_rsa)} {ci}"
        )
    lines.append("")
    lines.append(
        f"Elo: one Bradley-Terry fit over every pair's sets, {ANCHOR} fixed at "
        f"{ANCHOR_RATING:.0f}, with 1 virtual draw per pair. 95% range: the middle 95% of 200 "
        "refits on resampled sets."
    )
    if error:
        lines.append(f"Elo not fitted: {error}")
    sweeps = [r for r in results if r.record.a_wins == sets or r.record.b_wins == sets]
    for r in sweeps:
        gap = sweep_gap(sets)
        bound = f"at least {gap:,.0f} Elo" if gap > 0 else "too few sets to bound the gap"
        lines.append(
            f"Sweep: one side won all {sets:,} sets in {names[r.lower]} vs {names[r.upper]} "
            f"({bound}); that step's size comes from the virtual draw, not the data."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena.ladder",
        description="Rate bots listed weakest first from duplicate runs of each neighbour pair.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="a named ladder (bots, extra pairs, folder, sets and seed); other flags override it",
    )
    parser.add_argument(
        "--bots", help="bots weakest first, comma-separated (default: random up to pimc)"
    )
    parser.add_argument(
        "--sets", type=int, help="duplicate sets per pair (required without a preset)"
    )
    parser.add_argument("--seed", type=int, help="run seed for every pair (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--dir", type=Path, help="where pair reports and games are kept")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="an extra pair to rate, as weaker,stronger (repeatable)",
    )
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset] if args.preset else None
    if args.bots is None:
        args.bots = ",".join(preset.bots if preset else DEFAULT_LADDER)
    if preset is not None and not args.extra:
        args.extra = [f"{a},{b}" for a, b in preset.extras]
    if args.sets is None:
        if preset is None:
            parser.error("--sets is required without a --preset")
        args.sets = preset.sets
    if args.seed is None:
        args.seed = preset.seed if preset else 0
    if args.dir is None:
        args.dir = preset.dir if preset else Path("runs/ladder")

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
    status = StatusWriter(
        args.dir / STATUS_FILE,
        "ladder",
        [(f"{lower} vs {upper}", (lower, upper, lower, upper)) for lower, upper in pairs],
        args.sets,
        args.seed,
        args.workers,
        ["python -m estimation.arena.ladder", *(argv if argv is not None else sys.argv[1:])],
    )
    results: list[PairResult] = []
    played = 0
    try:
        for n, (lower, upper) in enumerate(pairs, 1):
            status.start(n - 1)
            result, reused = play_pair(
                lower, upper, args.sets, args.seed, args.workers, args.dir, status.set_done
            )
            results.append(result)
            played += not reused
            status.done(n - 1, reused, args.sets, _summary(result))
            note = "reused" if reused else f"{result.seconds:,.0f} s"
            print(
                f"[ladder] {n}/{len(pairs)} {lower} vs {upper}: {note}", file=sys.stderr, flush=True
            )
    except BaseException as exc:  # Ctrl-C included: the watch screen shows why the run stopped
        status.finish(error=f"stopped: {exc!r}")
        raise

    error = None
    try:
        ratings = bradley_terry([r.record for r in results], prior=DEFAULT_PRIOR)
    except ValueError as exc:
        ratings, error = {}, str(exc)
    status.finish(ratings=ratings or None)
    ranges = (
        bootstrap_ranges(
            {(r.lower, r.upper): (r.sets, [r.lower, r.upper, r.lower, r.upper]) for r in results}
        )
        if ratings
        else {}
    )
    print(format_ladder(bots, results, ratings, args.sets, args.seed, error, ranges))
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
        "ranges": {bot: list(span) for bot, span in ranges.items()},
        "names": {bot: name_of(bot) for bot in bots},
        "groups": {bot: list(entry_for(bot).groups) for bot in bots},
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
