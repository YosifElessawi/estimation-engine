"""Where the evaluators' games come from: saved games (a run folder, a games file or the web
table's games) or new games played by a lineup. Shared by the bidding and card-play evaluators."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from estimation.arena.games import GameRecord, code_version, games_files, read_games, rules_of
from estimation.arena.match import play_bola
from estimation.bots import make_bot
from estimation.engine import RulesConfig

from .work import Progress, print_progress, run_tasks

WEB_GAMES = Path("runs/web/games.jsonl")


@dataclass(frozen=True, slots=True)
class Games:
    rules: RulesConfig
    records: list[GameRecord]
    notes: list[str]  # what the report should say about where the games came from


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


def play_lineup(
    lineup: Sequence[str], games: int, seed: int, workers: int, progress: Progress | None
) -> list[GameRecord]:
    """``games`` new games by the 4 bots of ``lineup``, deal seeds counting up from ``seed``."""
    tasks = [(tuple(lineup), seed + i, i) for i in range(games)]
    return run_tasks(_play_game, tasks, workers, "games", progress)


def saved_games(path: Path) -> Games:
    """Every game saved under ``path``, played by the rules of its first games file."""
    files = games_files(path)
    if not files:
        raise ValueError(f"no saved games under {path}")
    read = [read_games(file) for file in files]
    groups = [(header, records) for header, records, _ in read]
    notes: list[str] = []
    unreadable = sum(bad for _, _, bad in read)
    if unreadable:
        notes.append(f"Skipped {unreadable} damaged lines in the games files.")
    current = code_version()
    versions = {h.code_version for h, _ in groups if h is not None}
    if any(h is None for h, _ in groups):
        notes.append("Note: web games carry no code version; they are replayed as saved.")
    if versions - {current}:
        notes.append(
            f"Note: games played by code {', '.join(sorted(versions))}, evaluated with {current}."
        )
    records = [record for _, recs in groups for record in recs]
    return Games(rules_of(groups[0][0]), records, notes)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """The bot to judge and where its games come from: ``--bot`` and one of ``--from``,
    ``--lineup`` (with ``--games``) or ``--web``."""
    parser.add_argument("--bot", required=True, help="the bot spec to evaluate, or human")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from", dest="source", type=Path, help="a run folder or games file")
    source.add_argument("--lineup", help="play new games with these 4 bots, comma-separated")
    source.add_argument(
        "--web", nargs="?", const=WEB_GAMES, type=Path, help="the web table's saved games"
    )
    parser.add_argument("--games", type=int, default=10, help="new games with --lineup (10)")


def games_from(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Games:
    """The games ``add_source_arguments`` asked for, played or read."""
    if args.lineup:
        lineup = [bot.strip() for bot in args.lineup.split(",")]
        if len(lineup) != 4:
            parser.error("--lineup needs 4 bots")
        progress = print_progress if args.progress else None
        records = play_lineup(lineup, args.games, args.seed, args.workers, progress)
        return Games(RulesConfig(), records, [])
    path: Path = args.source if args.source is not None else args.web
    try:
        return saved_games(path)
    except ValueError as error:
        parser.error(str(error))
