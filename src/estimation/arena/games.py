"""Saved games: every game a run plays, kept so stats and evaluations need no new run.

A games file is gzip-compressed JSON lines: a header (the code version, the rules, the lineup,
sets, seed and date), then one line per game with its deal seed, the bot in each seat, each bot's
seed, the final scores, the round count and every action. Replaying the actions through
``Bola(deal_seed)`` rebuilds the game exactly, without running a bot, so a stat over thousands of
games takes seconds. The web table's own ``games.jsonl`` (plain JSON lines, no header) reads too.
"""

import dataclasses
import gzip
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from estimation.engine import Bola, IllegalActionError, RulesConfig

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One played game. ``set_index`` is -1 and ``bot_seeds`` all 0 for a web table game."""

    set_index: int
    rotation: int
    deal_seed: int
    seats: tuple[str, str, str, str]
    bot_seeds: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    rounds: int
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GamesHeader:
    """Where a games file came from. ``rules`` holds only the rules that differ from the
    defaults."""

    code_version: str
    rules: dict[str, Any]
    lineup: tuple[str, ...]
    sets: int
    seed: int
    created: str


@dataclass(frozen=True, slots=True)
class ReplayStats:
    kept: int
    skipped: int


def code_version() -> str:
    """The checked-out commit, with ``+dirty`` when tracked files have changes; ``unknown``
    outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if changes else "")


def changed_rules(config: RulesConfig) -> dict[str, Any]:
    """The rules that differ from the defaults, by field name."""
    defaults = RulesConfig()
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if getattr(config, field.name) != getattr(defaults, field.name)
    }


def rules_of(header: GamesHeader | None) -> RulesConfig:
    """The rules a games file was played by (the defaults for a web game)."""
    return RulesConfig(**header.rules) if header is not None else RulesConfig()


def make_header(lineup: Sequence[str], sets: int, seed: int, config: RulesConfig) -> GamesHeader:
    return GamesHeader(
        code_version=code_version(),
        rules=changed_rules(config),
        lineup=tuple(lineup),
        sets=sets,
        seed=seed,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_games(path: Path, header: GamesHeader, records: Iterable[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"header": dataclasses.asdict(header)}) + "\n")
        for record in records:
            out.write(json.dumps(dataclasses.asdict(record)) + "\n")


def _record(data: dict[str, Any]) -> GameRecord:
    s, b, p = data["seats"], data["bot_seeds"], data["scores"]
    return GameRecord(
        set_index=int(data["set_index"]),
        rotation=int(data["rotation"]),
        deal_seed=int(data["deal_seed"]),
        seats=(str(s[0]), str(s[1]), str(s[2]), str(s[3])),
        bot_seeds=(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
        scores=(int(p[0]), int(p[1]), int(p[2]), int(p[3])),
        rounds=int(data["rounds"]),
        actions=tuple(int(a) for a in data["actions"]),
    )


def _web_record(data: dict[str, Any]) -> GameRecord:
    """A web table game (``server/history.py``): its seats' specs, seed, scores and actions."""
    seats = [str(seat["spec"]) for seat in data["seats"]]
    p = data["scores"]
    return GameRecord(
        set_index=-1,
        rotation=0,
        deal_seed=int(data["seed"]),
        seats=(seats[0], seats[1], seats[2], seats[3]),
        bot_seeds=(0, 0, 0, 0),
        scores=(int(p[0]), int(p[1]), int(p[2]), int(p[3])),
        rounds=int(data["rounds"]),
        actions=tuple(int(a) for a in data["actions"]),
    )


def read_games(path: Path) -> tuple[GamesHeader | None, list[GameRecord]]:
    """A games file's header and records; a web table file has no header."""
    opener = gzip.open if path.suffix == ".gz" else open
    header: GamesHeader | None = None
    records: list[GameRecord] = []
    with opener(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            data = cast(dict[str, Any], json.loads(line))
            if "header" in data:
                h = data["header"]
                header = GamesHeader(
                    code_version=str(h["code_version"]),
                    rules=dict(h["rules"]),
                    lineup=tuple(str(bot) for bot in h["lineup"]),
                    sets=int(h["sets"]),
                    seed=int(h["seed"]),
                    created=str(h["created"]),
                )
            elif "deal_seed" in data:
                records.append(_record(data))
            else:
                records.append(_web_record(data))
    return header, records


def replay(record: GameRecord, rules: RulesConfig) -> Bola | None:
    """The finished game rebuilt from its actions, or None if an action is no longer legal or
    the replay ends on other scores (the game was played by other rules or code)."""
    bola = Bola(record.deal_seed, rules)
    try:
        for action in record.actions:
            bola.step(action)
    except IllegalActionError:
        return None
    if not bola.is_over or bola.scores != record.scores:
        return None
    return bola


def replay_all(
    records: Iterable[GameRecord], rules: RulesConfig
) -> tuple[list[tuple[GameRecord, Bola]], ReplayStats]:
    """Every record that still replays, with how many were kept and skipped."""
    kept: list[tuple[GameRecord, Bola]] = []
    skipped = 0
    for record in records:
        bola = replay(record, rules)
        if bola is None:
            skipped += 1
        else:
            kept.append((record, bola))
    return kept, ReplayStats(len(kept), skipped)


def games_files(path: Path) -> list[Path]:
    """The games files at ``path``: the file itself, or a run folder's ``games/*.jsonl.gz``
    (or the folder's own ``*.jsonl.gz`` when it has no ``games`` folder)."""
    if path.is_file():
        return [path]
    folder = path / "games" if (path / "games").is_dir() else path
    return sorted(folder.glob("*.jsonl.gz"))
