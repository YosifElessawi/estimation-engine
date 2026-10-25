"""Finished games, saved on disk so their seeds and moves outlive the server.

Every played or watched game is appended to a JSON-lines file (by default
``runs/web/games.jsonl`` under the project root, which git ignores) once it is over: its seats,
seed, final scores and full action log. A game still in progress is never written, so a played
game's hidden seed (which fixes every deal) never lands on disk before the end. Reviews of saved
games are not saved again.

From a saved game the page can deal the same cards again (its seats and seed) or review it with
every hand shown (its action log replayed).
"""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

HISTORY_FILE = Path("runs/web/games.jsonl")

logger = logging.getLogger(__name__)


class SavedSeat(TypedDict):
    spec: str
    label: str
    elo: float | None


class SavedGame(TypedDict):
    """One finished game. ``seed_chosen`` says whether the player typed the seed."""

    id: str
    finished_at: str
    mode: str
    seed: int
    seed_chosen: bool
    human_seat: int | None
    seats: list[SavedSeat]
    scores: list[int]
    winners: list[int]
    rounds: int
    actions: list[int]


class SavedGameView(TypedDict):
    """A saved game for the list, without its action log."""

    id: str
    finished_at: str
    mode: str
    seed: int
    seed_chosen: bool
    human_seat: int | None
    seats: list[SavedSeat]
    scores: list[int]
    winners: list[int]
    rounds: int


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _valid(data: object) -> SavedGame | None:
    """``data`` as a saved game, or ``None`` if a field is missing or of the wrong type."""
    if not isinstance(data, dict):
        return None
    game = cast(dict[str, Any], data)
    try:
        seats = [
            SavedSeat(
                spec=str(s["spec"]),
                label=str(s["label"]),
                elo=None if s["elo"] is None else float(s["elo"]),
            )
            for s in game["seats"]
        ]
        saved = SavedGame(
            id=str(game["id"]),
            finished_at=str(game["finished_at"]),
            mode=str(game["mode"]),
            seed=int(game["seed"]),
            seed_chosen=bool(game["seed_chosen"]),
            human_seat=None if game["human_seat"] is None else int(game["human_seat"]),
            seats=seats,
            scores=[int(score) for score in game["scores"]],
            winners=[int(seat) for seat in game["winners"]],
            rounds=int(game["rounds"]),
            actions=[int(action) for action in game["actions"]],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if len(saved["seats"]) != 4 or len(saved["scores"]) != 4:
        return None
    return saved


def list_view(game: SavedGame) -> SavedGameView:
    return SavedGameView(
        id=game["id"],
        finished_at=game["finished_at"],
        mode=game["mode"],
        seed=game["seed"],
        seed_chosen=game["seed_chosen"],
        human_seat=game["human_seat"],
        seats=game["seats"],
        scores=game["scores"],
        winners=game["winners"],
        rounds=game["rounds"],
    )


class GameHistory:
    """The saved games in one JSON-lines file, oldest first on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def save(self, game: SavedGame) -> None:
        """Append ``game``. Raises OSError if the file cannot be written."""
        line = json.dumps(game, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as out:
                out.write(line + "\n")

    def games(self) -> list[SavedGame]:
        """Every saved game, newest first. Lines that do not parse are skipped with a warning
        (a crash mid-write can leave a partial last line)."""
        with self._lock:
            if not self.path.is_file():
                return []
            lines = self.path.read_text(encoding="utf-8").splitlines()
        found: list[SavedGame] = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                game = _valid(json.loads(line))
            except json.JSONDecodeError:
                game = None
            if game is None:
                logger.warning("skipping unreadable line %d of %s", number, self.path)
                continue
            found.append(game)
        found.reverse()
        return found

    def get(self, game_id: str) -> SavedGame | None:
        return next((game for game in self.games() if game["id"] == game_id), None)
