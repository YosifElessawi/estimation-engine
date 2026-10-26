"""The arena report: RSA with confidence intervals and Elo per bot, as text and as JSON.

Everything in ``Report`` follows from the lineup, the seed and the rules, so the same seed gives
the same report. Wall time and decision timings depend on the machine; they are kept apart, in
the run section printed after the report.
"""

import dataclasses
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .duplicate import ROTATIONS, ArenaRun, SetResult
from .games import changed_rules, code_version
from .ratings import (
    ANCHOR,
    ANCHOR_RATING,
    DEFAULT_PRIOR,
    Interval,
    PairRecord,
    bradley_terry,
    mean_ci,
    pair_records,
    set_first_places,
    set_rsa,
    set_scores,
    sweep_gap,
)


@dataclass(frozen=True, slots=True)
class BotSummary:
    name: str
    slots: int
    bolas: int  # Bolas played across the run
    mean_score: float  # mean Bola score
    rsa: Interval  # mean RSA per Bola, interval over sets
    first: Interval  # share of its Bolas finished first, interval over sets
    elo: float | None


@dataclass(frozen=True, slots=True)
class Report:
    lineup: tuple[str, ...]
    sets: int
    seed: int
    rules: dict[str, object]  # rules that differ from the defaults
    bolas: int
    rounds: int
    bots: tuple[BotSummary, ...]
    pairs: tuple[PairRecord, ...]
    elo_pairs: tuple[PairRecord, ...]  # this run's pairs plus any loaded from earlier reports
    elo_sources: tuple[str, ...]
    elo_prior: float
    elo_ratings: dict[str, float]
    elo_error: str | None  # why no ratings were fitted, if so
    results: tuple[SetResult, ...]


def build_report(
    run: ArenaRun,
    earlier_pairs: Sequence[PairRecord] = (),
    sources: Sequence[str] = (),
    prior: float = DEFAULT_PRIOR,
) -> Report:
    lineup = run.lineup
    results = run.results
    pairs = tuple(pair_records(results, lineup))
    elo_pairs = (*pairs, *earlier_pairs)
    elo_error = None
    try:
        ratings = bradley_terry(elo_pairs, prior=prior)
    except ValueError as error:
        ratings, elo_error = {}, str(error)
    per_set_rsa = [set_rsa(result, lineup) for result in results]
    per_set_score = [set_scores(result, lineup) for result in results]
    per_set_first = [set_first_places(result, lineup) for result in results]
    bots = tuple(
        BotSummary(
            name=name,
            slots=lineup.count(name),
            bolas=lineup.count(name) * ROTATIONS * len(results),
            mean_score=math.fsum(s[name] for s in per_set_score) / len(results),
            rsa=mean_ci([r[name] for r in per_set_rsa]),
            first=mean_ci([f[name] for f in per_set_first]),
            elo=ratings.get(name),
        )
        for name in dict.fromkeys(lineup)
    )
    rules = changed_rules(run.config)
    return Report(
        lineup=lineup,
        sets=len(results),
        seed=run.seed,
        rules=rules,
        bolas=ROTATIONS * len(results),
        rounds=sum(sum(result.rounds) for result in results),
        bots=bots,
        pairs=pairs,
        elo_pairs=elo_pairs,
        elo_sources=tuple(sources),
        elo_prior=prior,
        elo_ratings=ratings,
        elo_error=elo_error,
        results=results,
    )


def _signed(value: float) -> str:
    return f"{value:+,.1f}"


def _percent(share: float) -> str:
    """A share as a percentage, held to 0-100 (a t-interval over few sets can spill past)."""
    return f"{100 * min(max(share, 0.0), 1.0):.1f}"


def format_report(report: Report) -> str:
    rules = (
        ", ".join(f"{k}={v}" for k, v in report.rules.items()) if report.rules else "default rules"
    )
    lines = [
        f"Duplicate arena: {', '.join(report.lineup)}",
        f"{report.sets:,} sets x {ROTATIONS} rotations = {report.bolas:,} Bolas "
        f"({report.rounds:,} rounds), seed {report.seed}, {rules}",
        "",
    ]
    width = max(len("Bot"), *(len(bot.name) for bot in report.bots))
    header = (
        f"{'Bot':<{width}}  Slots  {'1st place':>9}  {'95% CI':<14}  {'RSA per Bola':>12}  "
        f"{'95% CI':<22}  {'Mean score':>10}  Elo"
    )
    lines += [header, "-" * len(header)]
    for bot in report.bots:
        interval = bot.rsa
        if math.isfinite(interval.low):
            ci = f"[{_signed(interval.low)}, {_signed(interval.high)}]"
        else:
            ci = "n/a (one set)"
        first = bot.first
        if math.isfinite(first.low):
            first_ci = f"[{_percent(first.low)}, {_percent(first.high)}]"
        else:
            first_ci = "n/a"
        elo = "n/a" if bot.elo is None else f"{bot.elo:,.0f}"
        if bot.name == ANCHOR and bot.elo is not None:
            elo += " (anchor)"
        lines.append(
            f"{bot.name:<{width}}  {bot.slots:>5}  {_percent(first.mean) + '%':>9}  "
            f"{first_ci:<14}  {_signed(interval.mean):>12}  {ci:<22}  "
            f"{_signed(bot.mean_score):>10}  {elo}"
        )
    lines.append(
        "1st place: how often one seat of the bot finished its Bola first, ties shared "
        "(a fair share is 25%)."
    )
    lines.append("")
    for pair in report.pairs:
        lines.append(
            f"Sets: {pair.a} beat {pair.b} in {pair.a_wins:,}, drew {pair.draws:,}, "
            f"lost {pair.b_wins:,}."
        )
    lines.append(_elo_note(report))
    lines += _sweep_notes(report)
    return "\n".join(lines)


def _sweep_notes(report: Report) -> list[str]:
    """Explain each pair where one bot won every set: its Elo gap is only bounded below."""
    notes: list[str] = []
    for pair in report.pairs:
        sets = pair.a_wins + pair.draws + pair.b_wins
        if pair.a_wins == sets or pair.b_wins == sets:
            winner, loser = (pair.a, pair.b) if pair.a_wins == sets else (pair.b, pair.a)
            gap = sweep_gap(sets)
            if gap <= 0:
                notes.append(
                    f"{winner} won all {sets:,} sets against {loser}, too few to bound the Elo "
                    "gap; the fitted gap above comes from the virtual draws."
                )
                continue
            notes.append(
                f"{winner} won all {sets:,} sets against {loser}, so the data put {winner} at "
                f"least {gap:,.0f} Elo above {loser} (95% confidence); the fitted gap above "
                "comes from the virtual draws."
            )
    return notes


def _elo_note(report: Report) -> str:
    rated = [bot for bot in report.bots if bot.elo is not None]
    note = (
        f"Elo: Bradley-Terry over each set's pairwise result, {ANCHOR} fixed at "
        f"{ANCHOR_RATING:.0f}, {report.elo_prior:g} virtual draw per pair"
    )
    if report.elo_sources:
        note += f", with sets from {', '.join(report.elo_sources)}"
    if report.elo_error:
        note += f". Elo not fitted: {report.elo_error}"
    elif not rated:
        note += f". No bot here is linked to {ANCHOR} through comparisons, so none is rated"
    return note + "."


def format_run(run: ArenaRun) -> str:
    workers = f"{run.workers} worker" + ("s" if run.workers != 1 else "")
    lines = [f"Run: {run.wall_seconds:,.1f} s wall time on {workers}."]
    timed = {name: t for name, t in run.timings.items() if t.count}
    if timed:
        budget_ms = next(iter(timed.values())).budget * 1000
        width = max(len("Bot"), *(len(name) for name in timed))
        lines.append(
            f"Card decision times (ms): {'Bot':<{width}}  {'count':>9}  {'mean':>7}  "
            f"{'p50':>7}  {'p99':>7}  {'max':>7}  over {budget_ms:g} ms"
        )
        for name, t in timed.items():
            lines.append(
                f"{'':26}{name:<{width}}  {t.count:>9,}  {t.mean * 1e3:>7.2f}  "
                f"{t.quantile(0.5) * 1e3:>7.2f}  {t.quantile(0.99) * 1e3:>7.2f}  "
                f"{t.max * 1e3:>7.2f}  {t.over:,}"
            )
    return "\n".join(lines)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def report_to_json(report: Report) -> dict[str, Any]:
    return {
        "lineup": list(report.lineup),
        "sets": report.sets,
        "seed": report.seed,
        "rules": report.rules,
        "bolas": report.bolas,
        "rounds": report.rounds,
        "bots": [
            {
                "name": bot.name,
                "slots": bot.slots,
                "bolas": bot.bolas,
                "mean_score": bot.mean_score,
                "rsa": {
                    "mean": bot.rsa.mean,
                    "low": _finite(bot.rsa.low),
                    "high": _finite(bot.rsa.high),
                    "sd": bot.rsa.sd,
                    "sets": bot.rsa.n,
                },
                "first_place": {
                    "mean": bot.first.mean,
                    "low": _finite(bot.first.low),
                    "high": _finite(bot.first.high),
                    "sets": bot.first.n,
                },
                "elo": bot.elo,
            }
            for bot in report.bots
        ],
        "pairs": [dataclasses.asdict(pair) for pair in report.pairs],
        "elo": {
            "anchor": ANCHOR,
            "anchor_rating": ANCHOR_RATING,
            "prior_draws": report.elo_prior,
            "sources": list(report.elo_sources),
            "ratings": report.elo_ratings,
            "error": report.elo_error,
        },
        "set_results": [
            {
                "index": result.index,
                "deal_seed": result.deal_seed,
                "scores": [list(scores) for scores in result.scores],
                "rounds": list(result.rounds),
            }
            for result in report.results
        ],
    }


def run_to_json(run: ArenaRun) -> dict[str, Any]:
    return {
        "wall_seconds": run.wall_seconds,
        "workers": run.workers,
        "card_decisions": {name: timing.to_dict() for name, timing in run.timings.items()},
    }


def write_json(path: Path, report: Report, run: ArenaRun, version: str | None = None) -> None:
    """Write the report and run, with the code version that played them (see
    ``games.code_version``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "report": {**report_to_json(report), "code_version": version or code_version()},
        "run": run_to_json(run),
    }
    temporary = path.with_name(path.name + ".tmp")  # written whole, then renamed into place
    temporary.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_pairs(path: Path) -> list[PairRecord]:
    """The pairwise set results stored in an earlier JSON report."""
    data: Mapping[str, Any] = json.loads(path.read_text())
    try:
        return [
            PairRecord(
                str(p["a"]), str(p["b"]), int(p["a_wins"]), int(p["draws"]), int(p["b_wins"])
            )
            for p in data["report"]["pairs"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not an arena report: {error}") from error
