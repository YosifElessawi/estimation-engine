"""The saved public games as web-games files, the format the evaluators read.

``export`` writes, under ``out``:

- ``games.jsonl``: every finished game.
- ``players/<name>.jsonl``: each player's finished games (names that differ only in capitals
  share a file).
- ``unfinished.jsonl``: games given up or still going, with their moves so far. The evaluators
  replay finished games only.

Each line is a game as ``server/history.py`` saves it (seats, seed, scores, actions), plus the
player's name, their visitor id, when it started and the code version it was played on. Run it
with ``uv run python -m server.public export``, then for example
``uv run python -m estimation.eval.bidding --bot human --web runs/public/players/<name>.jsonl``.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app import BOT_NAME
from .records import GameRow, Records

SAFE = frozenset("-_")


@dataclass(frozen=True, slots=True)
class Exported:
    finished: int
    unfinished: int
    players: dict[str, int]  # file name -> finished games


def player_file(name: str) -> str:
    """The file name for ``name``'s games: its letters and digits, other characters as ``-``."""
    slug = "".join(c if c.isalnum() or c in SAFE else "-" for c in name.casefold()).strip("-")
    return f"{slug or 'player'}.jsonl"


def web_game(row: GameRow) -> dict[str, Any]:
    """``row`` as a web-games line."""
    return {
        "id": row["id"],
        "finished_at": row["finished_at"],
        "mode": "play",
        "seed": row["seed"],
        "seed_chosen": False,
        "human_seat": row["human_seat"],
        "seats": [
            {
                "spec": spec,
                "label": row["name"] if seat == row["human_seat"] else BOT_NAME,
                "elo": None,
            }
            for seat, spec in enumerate(row["seats"])
        ],
        "scores": row["scores"],
        "winners": row["winners"],
        "rounds": row["rounds"],
        "actions": row["actions"],
        "player": row["name"],
        "visitor": row["visitor"],
        "started_at": row["started_at"],
        "left_at": row["left_at"],
        "version": row["version"],
    }


def _write(path: Path, rows: list[GameRow]) -> None:
    """Write ``rows`` to ``path`` through a temporary file, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(web_game(row), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def export(records: Records, out: Path) -> Exported:
    rows = records.all()
    finished = [row for row in rows if row["finished_at"] is not None]
    unfinished = [row for row in rows if row["finished_at"] is None]
    _write(out / "games.jsonl", finished)
    _write(out / "unfinished.jsonl", unfinished)
    by_player: dict[str, list[GameRow]] = {}
    for row in finished:
        by_player.setdefault(player_file(row["name"]), []).append(row)
    for file, games in by_player.items():
        _write(out / "players" / file, games)
    return Exported(
        len(finished), len(unfinished), {file: len(games) for file, games in by_player.items()}
    )
