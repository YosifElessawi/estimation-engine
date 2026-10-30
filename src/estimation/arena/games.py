"""Saved games: every game a run plays, kept so stats and evaluations need no new run.

A games file is gzip-compressed JSON lines: a header (the code version, the rules, the lineup,
sets, seed and date), then one line per game with its deal seed, the bot in each seat, each bot's
seed, the final scores, the round count and every action. Replaying the actions through
``Bola(deal_seed)`` rebuilds the game exactly, without running a bot, so a stat over thousands of
games takes seconds. The web table's own ``games.jsonl`` (plain JSON lines, no header) reads too.
"""

import dataclasses
import gzip
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from estimation.engine import Bola, IllegalActionError, RulesConfig

ROOT = Path(__file__).resolve().parents[3]
GZIP_MAGIC = b"\x1f\x8b"


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


def version_of(commit: str, changes: str) -> str:
    """A code version: the commit, plus a fingerprint of uncommitted changes to tracked files, so
    two different sets of edits never share a version."""
    if not changes:
        return commit
    return f"{commit}+dirty.{hashlib.sha1(changes.encode()).hexdigest()[:8]}"


def code_version() -> str:
    """The checked-out commit and any uncommitted changes to tracked files (``version_of``);
    ``unknown`` outside git. Untracked files are not part of it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return version_of(commit, changes)


# Rules added after games files were first written, with the value every file written before
# them was played by. A header always lists them, so a header without one predates it.
LEGACY_RULES: dict[str, Any] = {"with_any_call": False, "with_choose": False}  # 2026-10-30


def changed_rules(config: RulesConfig) -> dict[str, Any]:
    """The rules that differ from the defaults, by field name, and always ``LEGACY_RULES``'
    fields."""
    defaults = RulesConfig()
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.name in LEGACY_RULES
        or getattr(config, field.name) != getattr(defaults, field.name)
    }


def played_by(rules: dict[str, Any]) -> RulesConfig:
    """The rules a saved header or report's ``rules`` stand for: a rule they do not list has its
    default, or its ``LEGACY_RULES`` value when they predate that rule."""
    return RulesConfig(**{**LEGACY_RULES, **rules})


def rules_of(header: GamesHeader | None) -> RulesConfig:
    """The rules a games file was played by (``played_by`` its header). A web game, which has no
    header, is replayed by the current defaults."""
    return played_by(header.rules) if header is not None else RulesConfig()


def make_header(
    lineup: Sequence[str], sets: int, seed: int, config: RulesConfig, version: str | None = None
) -> GamesHeader:
    """A games file's header; ``version`` defaults to the current ``code_version()``."""
    return GamesHeader(
        code_version=version or code_version(),
        rules=changed_rules(config),
        lineup=tuple(lineup),
        sets=sets,
        seed=seed,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_games(path: Path, header: GamesHeader, records: Iterable[GameRecord]) -> None:
    """Write a games file all at once: to a temporary file first, then renamed into place, so a
    stopped run never leaves a half-written file under the real name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"header": dataclasses.asdict(header)}) + "\n")
        for record in records:
            out.write(json.dumps(dataclasses.asdict(record)) + "\n")
    os.replace(temporary, path)


def _header(h: dict[str, Any]) -> GamesHeader:
    return GamesHeader(
        code_version=str(h["code_version"]),
        rules=dict(h["rules"]),
        lineup=tuple(str(bot) for bot in h["lineup"]),
        sets=int(h["sets"]),
        seed=int(h["seed"]),
        created=str(h["created"]),
    )


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


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as raw:
        return raw.read(2) == GZIP_MAGIC


def read_games(path: Path) -> tuple[GamesHeader | None, list[GameRecord], int]:
    """A games file's header, records and how many lines could not be read (a damaged line, or
    the rest of a cut-off file, counts once). A web table file has no header. Gzip is recognised
    by its content, whatever the file is called."""
    header: GamesHeader | None = None
    records: list[GameRecord] = []
    bad = 0
    opener = gzip.open if _is_gzip(path) else open
    try:
        with opener(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = cast(dict[str, Any], json.loads(line))
                    if "header" in data:
                        header = _header(data["header"])
                    elif "deal_seed" in data:
                        records.append(_record(data))
                    else:
                        records.append(_web_record(data))
                except (ValueError, KeyError, TypeError, IndexError):
                    bad += 1
    except (EOFError, OSError, UnicodeDecodeError):  # a cut-off or damaged file
        bad += 1
    return header, records, bad


def read_header(path: Path) -> GamesHeader | None:
    """Only the header of a games file, or None if it has none or cannot be read."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            data = json.loads(lines.readline())
        return _header(data["header"])
    except (EOFError, OSError, ValueError, KeyError, TypeError):
        return None


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
