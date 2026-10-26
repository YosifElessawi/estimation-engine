"""How bots bid, from saved games or fresh ones: per bot, per role and per table (design 5.2).

Saved games (an arena or ladder run's ``games/`` folder, one games file, or the web table's
``games.jsonl``) are replayed through the engine, so no bot runs and a whole ladder's games take
seconds. Without ``--from``, tables of four copies of ``--bot`` play ``--games`` new games first.

Per bot, over every seat it sat in: Dash Calls, With, contract sizes as Caller, super calls, how
often each role makes its number, how often it takes more or fewer tricks than it estimated, and
what it does as last estimator. Per table: Over, Under and skipped rounds and the mean estimate
total. "Made" means exactly the estimate.

Examples: ``uv run python -m estimation.arena.bidstats --from runs/ladder-v2`` and
``uv run python -m estimation.arena.bidstats --bot yosif --games 300 --workers 5``.
"""

import argparse
import dataclasses
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

from estimation.bots import make_bot
from estimation.engine import RoundResult, RulesConfig, SeatResult

from .games import (
    GameRecord,
    GamesHeader,
    ReplayStats,
    games_files,
    read_games,
    replay_all,
    rules_of,
)
from .match import play_bola
from .roster import name_of

BIG = "8+"
ROLES = ("Caller", "With", "plain", "last")

Seats = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeatStats:
    """One bot's bidding, over every seat it sat in. Shares are of the rounds that apply."""

    rounds: int  # seat-rounds played (skipped rounds aside)
    dash_share: float  # of open seat-rounds, skipped ones included
    dash_made: float
    with_share: float
    with_made: float
    contract_share: dict[str, float]  # its contracts as Caller in open rounds, by size
    contract_made: dict[str, float]
    super_share: float  # of its Caller and With seat-rounds, scored as a super call (SCR-6)
    super_made: float
    made_by_role: dict[str, float]  # Caller, With, plain (other estimators), last estimator
    over_tricks: float  # took more tricks than it estimated (Dash Calls aside)
    under_tricks: float  # took fewer
    last_over: float  # as last estimator: the round ended Over
    last_under: float
    last_risk: float  # as last estimator: took at least one Risk


@dataclass(frozen=True, slots=True)
class TableStats:
    rounds: int
    over: float  # of played rounds
    under: float
    skipped: float  # of all rounds
    mean_total: float  # of the estimates, in played rounds


@dataclass(slots=True)
class _Tally:
    played: int = 0
    open_rounds: int = 0
    dash: list[int] = field(default_factory=lambda: [0, 0])  # count, made
    with_: list[int] = field(default_factory=lambda: [0, 0])
    supers: list[int] = field(default_factory=lambda: [0, 0, 0])  # count, made, of calls
    sizes: dict[str, list[int]] = field(default_factory=lambda: dict[str, list[int]]())
    roles: dict[str, list[int]] = field(default_factory=lambda: dict[str, list[int]]())
    tricks: list[int] = field(default_factory=lambda: [0, 0, 0])  # estimators, over, under
    last: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # count, over, under, risk


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _role(s: SeatResult) -> str:
    if s.is_caller:
        return "Caller"
    if s.is_with:
        return "With"
    return "last" if s.is_last else "plain"


def stats_from_rounds(
    rounds: Iterable[tuple[Seats, RoundResult]],
) -> tuple[dict[str, SeatStats], dict[Seats, TableStats]]:
    """Per bot and per table stats of scored rounds, each with the bot spec in every seat."""
    tallies: dict[str, _Tally] = {}
    tables: dict[Seats, list[float]] = {}  # rounds, played, over, under, total
    for seats, result in rounds:
        key = tuple(sorted(seats))
        table = tables.setdefault(key, [0, 0, 0, 0, 0])
        table[0] += 1
        is_open = result.fixed_trump is None
        if not result.skipped:
            table[1] += 1
            table[2] += bool(result.over)
            table[3] += not result.over
            table[4] += result.total or 0
        for spec, s in zip(seats, result.seats, strict=True):
            t = tallies.setdefault(spec, _Tally())
            if is_open:
                t.open_rounds += 1
            if result.skipped:
                continue
            t.played += 1
            if s.is_dash:
                t.dash[0] += 1
                t.dash[1] += s.won
                continue
            if s.is_with:
                t.with_[0] += 1
                t.with_[1] += s.won
            if s.is_caller or s.is_with:
                t.supers[2] += 1
                if s.is_super:
                    t.supers[0] += 1
                    t.supers[1] += s.won
            if s.is_caller and is_open:
                size = t.sizes.setdefault(BIG if s.estimate >= 8 else str(s.estimate), [0, 0])
                size[0] += 1
                size[1] += s.won
            role = t.roles.setdefault(_role(s), [0, 0])
            role[0] += 1
            role[1] += s.won
            t.tricks[0] += 1
            t.tricks[1] += s.tricks > s.estimate
            t.tricks[2] += s.tricks < s.estimate
            if s.is_last:
                t.last[0] += 1
                t.last[1] += bool(result.over)
                t.last[2] += not result.over
                t.last[3] += s.risks > 0
    callers = {spec: sum(c for c, _ in t.sizes.values()) for spec, t in tallies.items()}
    bots = {
        spec: SeatStats(
            rounds=t.played,
            dash_share=_share(t.dash[0], t.open_rounds),
            dash_made=_share(t.dash[1], t.dash[0]),
            with_share=_share(t.with_[0], t.played),
            with_made=_share(t.with_[1], t.with_[0]),
            contract_share={k: _share(c, callers[spec]) for k, (c, _) in sorted(t.sizes.items())},
            contract_made={k: _share(m, c) for k, (c, m) in sorted(t.sizes.items())},
            super_share=_share(t.supers[0], t.supers[2]),
            super_made=_share(t.supers[1], t.supers[0]),
            made_by_role={r: _share(t.roles[r][1], t.roles[r][0]) for r in ROLES if r in t.roles},
            over_tricks=_share(t.tricks[1], t.tricks[0]),
            under_tricks=_share(t.tricks[2], t.tricks[0]),
            last_over=_share(t.last[1], t.last[0]),
            last_under=_share(t.last[2], t.last[0]),
            last_risk=_share(t.last[3], t.last[0]),
        )
        for spec, t in tallies.items()
    }
    by_table = {
        key: TableStats(
            rounds=int(n),
            over=_share(int(over), int(played)),
            under=_share(int(under), int(played)),
            skipped=_share(int(n - played), int(n)),
            mean_total=total / played if played else 0.0,
        )
        for key, (n, played, over, under, total) in tables.items()
    }
    return bots, by_table


def stats_from_games(
    groups: Sequence[tuple[GamesHeader | None, Sequence[GameRecord]]],
    rules: RulesConfig | None = None,
    allow_mixed: bool = False,
) -> tuple[dict[str, SeatStats], dict[Seats, TableStats], ReplayStats]:
    """Stats of saved games, grouped by the header of the file they came from. Games from
    different code versions are refused unless ``allow_mixed``; web games (no header) count as
    an unknown version. Games that no longer replay are skipped and counted."""
    versions = {header.code_version for header, _ in groups if header is not None}
    if len(versions) > 1 and not allow_mixed:
        raise ValueError(f"games from different code versions: {', '.join(sorted(versions))}")
    rounds: list[tuple[Seats, RoundResult]] = []
    kept = skipped = 0
    for header, records in groups:
        replayed, stats = replay_all(records, rules or rules_of(header))
        kept += stats.kept
        skipped += stats.skipped
        for record, bola in replayed:
            rounds += [(record.seats, result) for result in bola.history]
    bots, tables = stats_from_rounds(rounds)
    return bots, tables, ReplayStats(kept, skipped)


def _play(task: tuple[str, int, int]) -> GameRecord:
    spec, seed, index = task
    seeds = [seed * 4 + seat for seat in range(4)]
    bots = [make_bot(spec, s) for s in seeds]
    played = play_bola(seed, bots, keep_actions=True)
    return GameRecord(
        set_index=index,
        rotation=0,
        deal_seed=seed,
        seats=(spec, spec, spec, spec),
        bot_seeds=(seeds[0], seeds[1], seeds[2], seeds[3]),
        scores=played.scores,
        rounds=played.rounds,
        actions=played.actions,
    )


def play_games(spec: str, games: int, seed: int, workers: int) -> list[GameRecord]:
    """``games`` new games of four ``spec`` bots, dealt from ``seed``, ``seed + 1``, ..."""
    tasks = [(spec, seed + i, i) for i in range(games)]
    if workers == 1:
        return [_play(task) for task in tasks]
    with Pool(workers) as pool:
        return pool.map(_play, tasks)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def format_stats(
    bots: dict[str, SeatStats], tables: dict[Seats, TableStats], only: str | None = None
) -> str:
    lines: list[str] = []
    for spec in sorted(bots, key=name_of):
        if only is not None and spec != only:
            continue
        s = bots[spec]
        sizes = "  ".join(
            f"{k}: {_pct(v)} ({_pct(s.contract_made[k])} made)" for k, v in s.contract_share.items()
        )
        roles = "  ".join(f"{role} {_pct(v)}" for role, v in s.made_by_role.items())
        lines += [
            f"{name_of(spec)} ({s.rounds:,} seat-rounds)",
            f"  Dash Calls          {_pct(s.dash_share)} of open rounds ({_pct(s.dash_made)} made)",
            f"  With                {_pct(s.with_share)} ({_pct(s.with_made)} made)",
            f"  Contracts as Caller {sizes or 'none'}",
            f"  Super calls         {_pct(s.super_share)} of Caller/With "
            f"({_pct(s.super_made)} made)",
            f"  Made by role        {roles}",
            f"  Took more / fewer   {_pct(s.over_tricks)} / {_pct(s.under_tricks)} than estimated",
            f"  As last estimator   Over {_pct(s.last_over)}, Under {_pct(s.last_under)}, "
            f"took a Risk {_pct(s.last_risk)}",
            "",
        ]
    for key in sorted(tables, key=lambda k: [name_of(b) for b in k]):
        if only is not None and only not in key:
            continue
        t = tables[key]
        names = ", ".join(name_of(bot) for bot in key)
        lines.append(
            f"Table {names}: {t.rounds:,} rounds, Over {_pct(t.over)}, Under {_pct(t.under)}, "
            f"skipped {_pct(t.skipped)}, mean estimate total {t.mean_total:.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena.bidstats",
        description="How bots bid, from saved games (--from) or new games of four copies (--bot).",
    )
    parser.add_argument("--from", dest="source", type=Path, help="a run folder or games file")
    parser.add_argument("--bot", help="the bot to play (without --from) or to show (with it)")
    parser.add_argument("--games", type=int, help="new games to play, without --from")
    parser.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument(
        "--allow-mixed", action="store_true", help="count games from different code versions"
    )
    parser.add_argument("--out", type=Path, help="also write the stats as JSON to this file")
    args = parser.parse_args(argv)
    if args.source is None and (args.bot is None or args.games is None):
        parser.error("give --from, or --bot and --games")
    if args.workers < 1 or args.seed < 0 or (args.games is not None and args.games < 1):
        parser.error("--games and --workers must be at least 1, --seed 0 or more")
    if args.source is not None:
        files = games_files(args.source)
        if not files:
            parser.error(f"no saved games under {args.source}")
        read = [read_games(path) for path in files]
        groups = [(header, records) for header, records, _ in read]
        unreadable = sum(bad for _, _, bad in read)
    else:
        spec, games = str(args.bot), args.games
        if games is None:  # checked above: without --from, --games is given
            parser.error("give --from, or --bot and --games")
        try:
            make_bot(spec, 0)
        except ValueError as error:
            parser.error(str(error))
        groups = [(None, play_games(spec, games, args.seed, args.workers))]
        unreadable = 0
    try:
        bots, tables, replayed = stats_from_games(groups, allow_mixed=args.allow_mixed)
    except ValueError as error:
        parser.error(str(error))
    print(format_stats(bots, tables, args.bot if args.source is not None else None))
    if replayed.skipped:
        print(f"\nSkipped {replayed.skipped:,} saved games that no longer replay.")
    if unreadable:
        print(f"Skipped {unreadable:,} damaged lines in the games files.")
    if any(header is None for header, _ in groups) and args.source is not None:
        print("Note: web games carry no code version.")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bots": {spec: dataclasses.asdict(s) for spec, s in bots.items()},
            "tables": {" | ".join(key): dataclasses.asdict(t) for key, t in tables.items()},
            "games": {"kept": replayed.kept, "skipped": replayed.skipped},
        }
        args.out.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
