"""The bidding evaluator: how many points each of a bot's bidding decisions cost (design 5.4).

Decisions come from saved games (``--from`` a run folder or games file, or ``--web`` for the web
table's games, where ``--bot human`` scores the player), or from new games (``--lineup``). A sample
spread evenly over the decision types is priced by ``playout.value_table``: each candidate choice
played out on the same re-dealt hands. A decision's points lost are the best candidate's mean
value minus the chosen one's. The report gives the points lost per decision type, the worst
decisions written out, and, for bots that predict their chance of making a call, how those
predictions compare with what happened in the games.

Example: ``uv run python -m estimation.eval.bidding --bot yosif --from runs/ladder-v2
--decisions 300 --workers 5 --out runs/eval/yosif``.
"""

import argparse
import dataclasses
import json
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.arena.games import (
    GameRecord,
    GamesHeader,
    code_version,
    games_files,
    read_games,
    rules_of,
)
from estimation.arena.match import play_bola
from estimation.bots import HeuristicBot, make_bot
from estimation.engine import Bola, IllegalActionError, Phase, RulesConfig

from .decisions import KINDS, Decision, candidates, decisions_of
from .playout import value_table
from .report import action_text, situation
from .work import Progress, print_progress, run_tasks

DEFAULT_DECISIONS = 300
DEFAULT_DEALS = 12  # [Claude]
DEFAULT_CARD_PLAY = "pimc:samples=8"  # [Claude]: PIMC-4 under-punished calls strong players break
WEB_GAMES = Path("runs/web/games.jsonl")
WORST = 20


@dataclass(frozen=True, slots=True)
class Evaluated:
    decision: Decision
    values: dict[int, tuple[float, float]]  # each candidate's mean value and standard error
    best: int
    lost: float  # the best candidate's mean value minus the chosen one's, at least 0


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    low: float
    high: float
    count: int
    predicted: float  # mean predicted chance of making the call
    actual: float  # share of those calls made


def points_lost(values: dict[int, tuple[float, float]], chosen: int) -> tuple[int, float]:
    """The best candidate (the chosen one on a tie) and the points the choice lost to it."""
    best = max(values, key=lambda action: (values[action][0], action == chosen))
    return best, max(0.0, values[best][0] - values[chosen][0])


def sample_decisions(decisions: Sequence[Decision], n: int, seed: int) -> list[Decision]:
    """Up to ``n`` decisions spread evenly over the decision types, the same for a seed."""
    rng = random.Random(seed)
    groups = {kind: [d for d in decisions if d.kind == kind] for kind in KINDS}
    for group in groups.values():
        rng.shuffle(group)
    picked: list[Decision] = []
    while len(picked) < n and any(groups.values()):
        for kind in KINDS:
            if groups[kind] and len(picked) < n:
                picked.append(groups[kind].pop())
    return picked


def _evaluate_one(task: tuple[Decision, int, str, int, RulesConfig]) -> Evaluated:
    d, deals, card_play, seed, rules = task
    if d.game is None:
        raise ValueError("a decision needs the game it came from to know who sat where")
    heuristic = HeuristicBot().act(d.obs, list(d.legal))
    choices = candidates(d, heuristic)
    values = value_table(d, choices, d.game.seats, deals, card_play, seed, rules)
    best, lost = points_lost(values, d.chosen)
    return Evaluated(d, values, best, lost)


def evaluate(
    decisions: Sequence[Decision],
    deals: int = DEFAULT_DEALS,
    card_play: str = DEFAULT_CARD_PLAY,
    seed: int = 0,
    rules: RulesConfig | None = None,
    workers: int = 1,
    progress: Progress | None = None,
) -> list[Evaluated]:
    rules = rules or RulesConfig()
    tasks = [(d, deals, card_play, seed + 7919 * i, rules) for i, d in enumerate(decisions)]
    return run_tasks(_evaluate_one, tasks, workers, "decisions", progress)


def calibration(
    records: Sequence[GameRecord], rules: RulesConfig, target: str
) -> list[CalibrationBin]:
    """For a bot with ``predicted_make``: each round's final call by it, grouped by predicted
    chance in tenths, against how often those calls were made. Empty for other bots."""
    if target == "human":
        return []
    bot = make_bot(target, 0, rules)
    predict = getattr(bot, "predicted_make", None)
    if predict is None:
        return []
    pairs: list[tuple[float, bool]] = []
    for record in records:
        bola = Bola(record.deal_seed, rules)
        pending: dict[int, float] = {}  # seat -> its latest prediction this round
        try:
            for action in record.actions:
                seat = bola.current_player
                if seat is None:
                    break
                if record.seats[seat] == target and bola.phase is not Phase.PLAY:
                    chance = predict(bola.observation(seat), action)
                    if chance is not None:
                        pending[seat] = float(chance)
                rounds = len(bola.history)
                bola.step(action)
                if len(bola.history) > rounds:
                    result = bola.history[-1]
                    if not result.skipped:
                        pairs += [(p, result.seats[s].won) for s, p in pending.items()]
                    pending = {}
        except IllegalActionError:
            continue
    bins: list[CalibrationBin] = []
    for i in range(10):
        inside = [(p, won) for p, won in pairs if min(int(p * 10), 9) == i]
        if inside:
            bins.append(
                CalibrationBin(
                    low=i / 10,
                    high=(i + 1) / 10,
                    count=len(inside),
                    predicted=sum(p for p, _ in inside) / len(inside),
                    actual=sum(won for _, won in inside) / len(inside),
                )
            )
    return bins


def format_evaluation(
    target: str,
    evaluated: Sequence[Evaluated],
    bins: Sequence[CalibrationBin],
    notes: Sequence[str],
) -> str:
    lines = [f"Bidding evaluation: {target}, {len(evaluated)} decisions", *notes, ""]
    lines.append("Points lost per decision, by type (0 means it chose the best candidate)")
    lines.append(f"  {'Type':<14} {'Decisions':>9}  {'Mean lost':>9}  {'Chose best':>10}")
    for kind in KINDS:
        group = [e for e in evaluated if e.decision.kind == kind]
        if group:
            mean = sum(e.lost for e in group) / len(group)
            best = sum(e.best == e.decision.chosen for e in group) / len(group)
            lines.append(f"  {kind:<14} {len(group):>9}  {mean:>9.1f}  {100 * best:>9.0f}%")
    if evaluated:
        mean = sum(e.lost for e in evaluated) / len(evaluated)
        lines.append(f"  {'all':<14} {len(evaluated):>9}  {mean:>9.1f}")
    worst = sorted(evaluated, key=lambda e: -e.lost)[:WORST]
    if worst and worst[0].lost > 0:
        lines += ["", f"The {len(worst)} costliest decisions"]
        for n, e in enumerate(worst, 1):
            if e.lost <= 0:
                break
            d = e.decision
            chosen, best = e.values[d.chosen], e.values[e.best]
            lines += ["", f"{n}. {d.kind}: lost {e.lost:.1f} points"]
            lines += [f"   {line}" for line in situation(d)]
            clear = e.lost > 2 * math.hypot(chosen[1], best[1])
            lines.append(
                f"   Chose {action_text(d.chosen)} ({chosen[0]:+.1f} ± {chosen[1]:.1f}); "
                f"best was {action_text(e.best)} ({best[0]:+.1f} ± {best[1]:.1f}); "
                + ("a clear difference." if clear else "within the noise.")
            )
    if bins:
        lines += ["", "Predicted chance of making a call against what happened"]
        lines.append(f"  {'Predicted':<12} {'Calls':>6}  {'Mean predicted':>14}  {'Made':>6}")
        for b in bins:
            span = f"{100 * b.low:.0f}-{100 * b.high:.0f}%"
            lines.append(
                f"  {span:<12} {b.count:>6}  {100 * b.predicted:>13.0f}%  {100 * b.actual:>5.0f}%"
            )
    return "\n".join(lines)


def _to_json(
    target: str, evaluated: Sequence[Evaluated], bins: Sequence[CalibrationBin]
) -> dict[str, Any]:
    return {
        "bot": target,
        "decisions": [
            {
                "kind": e.decision.kind,
                "seat": e.decision.seat,
                "game": None
                if e.decision.game is None
                else {
                    "deal_seed": e.decision.game.deal_seed,
                    "set_index": e.decision.game.set_index,
                    "rotation": e.decision.game.rotation,
                },
                "step": e.decision.step,
                "chosen": e.decision.chosen,
                "best": e.best,
                "lost": e.lost,
                "values": {str(a): list(v) for a, v in e.values.items()},
            }
            for e in evaluated
        ],
        "calibration": [dataclasses.asdict(b) for b in bins],
    }


def _play_game(task: tuple[tuple[str, ...], int, int]) -> GameRecord:
    lineup, deal, index = task
    seeds = [deal * 4 + seat for seat in range(4)]
    bots = [make_bot(spec, s) for spec, s in zip(lineup, seeds, strict=True)]
    played = play_bola(deal, bots, keep_actions=True)
    return GameRecord(
        set_index=index,
        rotation=0,
        deal_seed=deal,
        seats=(lineup[0], lineup[1], lineup[2], lineup[3]),
        bot_seeds=(seeds[0], seeds[1], seeds[2], seeds[3]),
        scores=played.scores,
        rounds=played.rounds,
        actions=played.actions,
    )


def _play_lineup(
    lineup: Sequence[str], games: int, seed: int, workers: int, progress: Progress | None
) -> list[GameRecord]:
    tasks = [(tuple(lineup), seed + i, i) for i in range(games)]
    return run_tasks(_play_game, tasks, workers, "games", progress)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.eval.bidding",
        description="Points each of a bot's bidding decisions cost against the best candidate.",
    )
    parser.add_argument("--bot", required=True, help="the bot spec to evaluate, or human")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from", dest="source", type=Path, help="a run folder or games file")
    source.add_argument("--lineup", help="play new games with these 4 bots, comma-separated")
    source.add_argument(
        "--web", nargs="?", const=WEB_GAMES, type=Path, help="the web table's saved games"
    )
    parser.add_argument("--games", type=int, default=10, help="new games with --lineup (10)")
    parser.add_argument("--decisions", type=int, default=DEFAULT_DECISIONS, help="(300)")
    parser.add_argument("--deals", type=int, default=DEFAULT_DEALS, help="deals per decision (12)")
    parser.add_argument("--card-play", default=DEFAULT_CARD_PLAY, help="card player in play-outs")
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--out", type=Path, help="write report.txt and evaluation.json here")
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    args = parser.parse_args(argv)
    if min(args.games, args.decisions, args.deals, args.workers) < 1 or args.seed < 0:
        parser.error("--games, --decisions, --deals and --workers must be at least 1")
    groups: list[tuple[GamesHeader | None, list[GameRecord]]]
    notes: list[str] = []
    if args.lineup:
        lineup = [bot.strip() for bot in args.lineup.split(",")]
        if len(lineup) != 4:
            parser.error("--lineup needs 4 bots")
        progress = print_progress if args.progress else None
        groups = [(None, _play_lineup(lineup, args.games, args.seed, args.workers, progress))]
    else:
        path: Path = args.source if args.source is not None else args.web
        files = games_files(path)
        if not files:
            parser.error(f"no saved games under {path}")
        read = [read_games(file) for file in files]
        groups = [(header, records) for header, records, _ in read]
        unreadable = sum(bad for _, _, bad in read)
        if unreadable:
            notes.append(f"Skipped {unreadable} damaged lines in the games files.")
        current = code_version()
        versions = {h.code_version for h, _ in groups if h is not None}
        if any(h is None for h, _ in groups):
            notes.append("Note: web games carry no code version; they are replayed as saved.")
        if versions - {current}:
            notes.append(
                f"Note: games played by code {', '.join(sorted(versions))}, "
                f"evaluated with {current}."
            )
    rules = rules_of(groups[0][0])
    records = [record for _, recs in groups for record in recs]
    decisions: list[Decision] = []
    skipped = 0
    for record in records:
        try:
            decisions += decisions_of(record, rules, args.bot)
        except (IllegalActionError, ValueError):
            skipped += 1
    if skipped:
        notes.append(f"Skipped {skipped} saved games that no longer replay.")
    if not decisions:
        parser.error(f"no decisions by {args.bot} in these games")
    sample = sample_decisions(decisions, args.decisions, args.seed)
    evaluated = evaluate(
        sample,
        args.deals,
        args.card_play,
        args.seed,
        rules,
        args.workers,
        print_progress if args.progress else None,
    )
    bins = calibration(records, rules, args.bot)
    text = format_evaluation(args.bot, evaluated, bins, notes)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.txt").write_text(text + "\n")
        data = _to_json(args.bot, evaluated, bins)
        (args.out / "evaluation.json").write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
