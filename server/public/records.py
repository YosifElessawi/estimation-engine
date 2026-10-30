"""Every public game, saved move by move in one SQLite file.

A game's row is written when it starts and rewritten after every move, so a crash or a restart
loses nothing: the game is rebuilt from its seed and action log the next time its player asks for
it. A row also keeps who played (the name they typed and a random id their browser made up), when
the game started and ended, and the code version it was played on. Only a hash of the game's
token is stored. No IP address is ever stored.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, TypedDict, cast

DB_FILE = Path("runs/public/games.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    visitor TEXT NOT NULL,
    name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    human_seat INTEGER NOT NULL,
    seats TEXT NOT NULL,
    version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT,
    left_at TEXT,
    rounds INTEGER NOT NULL,
    scores TEXT NOT NULL,
    winners TEXT,
    actions TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS games_by_visitor ON games (visitor, started_at);
CREATE INDEX IF NOT EXISTS games_by_start ON games (started_at);
"""


class GameRow(TypedDict):
    """One public game. ``seats`` are bot specs, ``"human"`` for the player's seat; ``scores``
    are the game scores after the last finished round; ``winners`` is set once it is over."""

    id: str
    visitor: str
    name: str
    seed: int
    human_seat: int
    seats: list[str]
    version: str
    started_at: str
    updated_at: str
    finished_at: str | None
    left_at: str | None
    rounds: int
    scores: list[int]
    winners: list[int] | None
    actions: list[int]


class Progress(TypedDict):
    """What a move changes in a game's row."""

    updated_at: str
    rounds: int
    scores: list[int]
    finished_at: str | None
    winners: list[int] | None
    actions: list[int]


def _row(data: sqlite3.Row) -> GameRow:
    winners = cast(str | None, data["winners"])
    return GameRow(
        id=str(data["id"]),
        visitor=str(data["visitor"]),
        name=str(data["name"]),
        seed=int(data["seed"]),
        human_seat=int(data["human_seat"]),
        seats=[str(spec) for spec in cast(list[Any], json.loads(data["seats"]))],
        version=str(data["version"]),
        started_at=str(data["started_at"]),
        updated_at=str(data["updated_at"]),
        finished_at=cast(str | None, data["finished_at"]),
        left_at=cast(str | None, data["left_at"]),
        rounds=int(data["rounds"]),
        scores=[int(score) for score in cast(list[Any], json.loads(data["scores"]))],
        winners=None
        if winners is None
        else [int(seat) for seat in cast(list[Any], json.loads(winners))],
        actions=[int(action) for action in cast(list[Any], json.loads(data["actions"]))],
    )


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


class Records:
    """The public games in one SQLite file. Safe to use from several threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def create(self, row: GameRow, token_hash: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO games (id, token_hash, visitor, name, seed, human_seat, seats,"
                " version, started_at, updated_at, finished_at, left_at, rounds, scores, winners,"
                " actions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    token_hash,
                    row["visitor"],
                    row["name"],
                    row["seed"],
                    row["human_seat"],
                    _json(row["seats"]),
                    row["version"],
                    row["started_at"],
                    row["updated_at"],
                    row["finished_at"],
                    row["left_at"],
                    row["rounds"],
                    _json(row["scores"]),
                    None if row["winners"] is None else _json(row["winners"]),
                    _json(row["actions"]),
                ),
            )

    def save(self, game_id: str, progress: Progress) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE games SET updated_at = ?, rounds = ?, scores = ?, finished_at = ?,"
                " winners = ?, actions = ? WHERE id = ?",
                (
                    progress["updated_at"],
                    progress["rounds"],
                    _json(progress["scores"]),
                    progress["finished_at"],
                    None if progress["winners"] is None else _json(progress["winners"]),
                    _json(progress["actions"]),
                    game_id,
                ),
            )

    def leave(self, game_id: str, at: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE games SET left_at = ?, updated_at = ? WHERE id = ? AND left_at IS NULL",
                (at, at, game_id),
            )

    def get(self, game_id: str) -> tuple[GameRow, str] | None:
        """The game and its token hash, or None."""
        with self._lock:
            data = self._db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        if data is None:
            return None
        return _row(data), str(data["token_hash"])

    def started_since(self, since: str, visitor: str | None = None) -> int:
        """How many games started at or after ``since`` (an ISO time), by ``visitor`` if given."""
        with self._lock:
            if visitor is None:
                found = self._db.execute(
                    "SELECT COUNT(*) FROM games WHERE started_at >= ?", (since,)
                ).fetchone()
            else:
                found = self._db.execute(
                    "SELECT COUNT(*) FROM games WHERE visitor = ? AND started_at >= ?",
                    (visitor, since),
                ).fetchone()
        return int(found[0])

    def open_games(self, visitor: str) -> list[str]:
        """The ids of ``visitor``'s games still going (not finished, not left), oldest first."""
        with self._lock:
            found = self._db.execute(
                "SELECT id FROM games WHERE visitor = ? AND finished_at IS NULL"
                " AND left_at IS NULL ORDER BY started_at, rowid",
                (visitor,),
            ).fetchall()
        return [str(data["id"]) for data in found]

    def all(self) -> list[GameRow]:
        """Every game, oldest first."""
        with self._lock:
            found = self._db.execute("SELECT * FROM games ORDER BY started_at, rowid").fetchall()
        return [_row(data) for data in found]
