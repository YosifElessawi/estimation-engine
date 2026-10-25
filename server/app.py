"""The HTTP API of the web table (SPEC §6 M6). Every route is under ``/api``.

- ``GET /api/bots``: the bots a seat can hold, with their ladder Elo.
- ``POST /api/games``: start a game from 4 seats (bot specs, or one ``"human"``) and a seed.
- ``GET /api/games/{id}``: the game's seats, mode and progress.
- ``GET /api/games/{id}/frames?start=&count=``: frames to step through. A watched game makes
  them on demand; a played game has those up to the human's next decision.
- ``GET /api/games/{id}/seek?start=&direction=&target=``: the next or previous trick, round
  result, deal or game end.
- ``POST /api/games/{id}/moves``: the human's move in a played game.
- ``POST /api/games/{id}/review``: watch a finished played game again with every hand shown.
- ``GET /api/history``: finished games saved on disk, newest first (seats, seed, scores).
- ``POST /api/history/{id}/review``: review a saved game with every hand shown.

In a played game nothing the browser receives holds another seat's cards: tables come from the
human seat's Observation, the Why panel is left out, and the seed (which fixes the deal) is
withheld until the game ends unless the player chose it. A game is saved only once it is over.
"""

import logging
import secrets
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from estimation.engine import IllegalActionError, RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .catalog import BotEntry, Catalog
from .history import HISTORY_FILE, GameHistory, SavedGame, SavedGameView, SavedSeat, list_view, now
from .session import HUMAN, Frame, GameError, GameSession, Mode, SeekTarget, WaitingForHuman

MAX_GAMES = 32
MAX_HISTORY = 500

logger = logging.getLogger(__name__)
MAX_FRAMES = 500
MAX_SEED = 2**53 - 1  # the largest integer a browser's JSON numbers hold exactly


class NewGame(BaseModel):
    """Four seats, each a bot spec or ``"human"`` (at most one), and an optional seed."""

    seats: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        min_length=4, max_length=4
    )
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)


class Move(BaseModel):
    """The human's action id, and the frame it was chosen on."""

    action: int = Field(ge=0, lt=120)
    frame: int | None = Field(default=None, ge=0)


class BotView(TypedDict):
    spec: str
    label: str
    family: str
    elo: float | None
    explains: bool


class SeatView(TypedDict):
    seat: int
    spec: str
    label: str
    elo: float | None
    human: bool
    explains: bool


class GameView(TypedDict):
    """A game's setup and progress. ``seed`` is ``None`` while a played game hides it."""

    id: str
    mode: Mode
    seed: int | None
    human_seat: int | None
    seats: list[SeatView]
    known: int
    finished: bool
    waiting: bool


class FramesView(TypedDict):
    frames: list[Frame]
    known: int
    finished: bool


class SeekView(TypedDict):
    index: int


def _bot_view(entry: BotEntry) -> BotView:
    return BotView(
        spec=entry.spec,
        label=entry.label,
        family=entry.family,
        elo=None if entry.elo is None else round(entry.elo),
        explains=entry.explains,
    )


@dataclass(slots=True)
class Entry:
    """A game in memory: whether its seed is shown before the end, whether the player chose
    it, and whether the finished game is saved yet."""

    session: GameSession
    seed_shown: bool
    seed_chosen: bool
    saved: bool = False


class GameStore:
    """The games in memory, oldest dropped first beyond ``limit``, and the saved history."""

    def __init__(
        self,
        catalog: Catalog,
        rules: RulesConfig,
        history: GameHistory,
        limit: int = MAX_GAMES,
    ) -> None:
        self.catalog = catalog
        self.rules = rules
        self.history = history
        self.limit = limit
        self._games: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, entry: Entry) -> None:
        with self._lock:
            self._games[entry.session.id] = entry
            while len(self._games) > self.limit:
                self._games.popitem(last=False)

    def get(self, game_id: str) -> Entry:
        with self._lock:
            found = self._games.get(game_id)
            if found is None:
                raise HTTPException(404, f"no game {game_id!r}")
            self._games.move_to_end(game_id)
            return found

    def _seat_label(self, spec: str) -> str:
        return "You" if spec == HUMAN else self.catalog.entry(spec).label

    def _seat_elo(self, spec: str) -> float | None:
        return None if spec == HUMAN else _bot_view(self.catalog.entry(spec))["elo"]

    def view(self, entry: Entry) -> GameView:
        session = entry.session
        seats = [
            SeatView(
                seat=seat,
                spec=spec,
                label=self._seat_label(spec),
                elo=self._seat_elo(spec),
                human=spec == HUMAN,
                explains=session.spectator and seat in session.explained,
            )
            for seat, spec in enumerate(session.seats)
        ]
        return GameView(
            id=session.id,
            mode=session.mode,
            seed=session.seed if entry.seed_shown or session.finished else None,
            human_seat=session.human,
            seats=seats,
            known=len(session.frames),
            finished=session.finished,
            waiting=session.human is not None
            and not session.finished
            and session.bola.current_player == session.human,
        )

    def record(self, entry: Entry) -> None:
        """Save a played or watched game once it is over (reviews are not saved again). A
        failed write is logged and tried again on the next request; the game goes on."""
        session = entry.session
        if entry.saved or not session.finished or session.mode == "review":
            return
        bola = session.bola
        game = SavedGame(
            id=session.id,
            finished_at=now(),
            mode=session.mode,
            seed=session.seed,
            seed_chosen=entry.seed_chosen,
            human_seat=session.human,
            seats=[
                SavedSeat(spec=spec, label=self._seat_label(spec), elo=self._seat_elo(spec))
                for spec in session.seats
            ],
            scores=list(bola.scores),
            winners=list(bola.winners()),
            rounds=len(bola.history),
            actions=list(session.actions),
        )
        try:
            self.history.save(game)
        except OSError:
            logger.exception("could not save game %s to %s", session.id, self.history.path)
            return
        entry.saved = True

    def create(
        self, seats: list[str], seed: int | None, mode: Mode, script: list[int] | None = None
    ) -> GameView:
        chosen = seed is not None
        # A hidden seed must not be guessable from the player's own first hand: the deal is a
        # pure function of the seed (SPEC §5.1), so draw it from the whole range.
        game_seed = seed if seed is not None else secrets.randbelow(MAX_SEED + 1)
        try:
            session = GameSession(
                uuid.uuid4().hex[:12],
                seats,
                game_seed,
                self.catalog,
                mode=mode,
                script=script,
                rules=self.rules,
            )
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        entry = Entry(session, seed_shown=chosen or mode != "play", seed_chosen=chosen)
        self.add(entry)
        with session.lock:
            return self.view(entry)


def create_app(
    catalog: Catalog | None = None,
    rules: RulesConfig = DEFAULT_CONFIG,
    history_file: Path | None = None,
) -> FastAPI:
    """The API, serving ``catalog`` (by default the one under ``$ESTIMATION_ROOT``) and saving
    finished games to ``history_file`` (by default ``runs/web/games.jsonl`` under its root)."""
    catalog = catalog if catalog is not None else Catalog.from_env()
    history = GameHistory(history_file if history_file is not None else catalog.root / HISTORY_FILE)
    store = GameStore(catalog, rules, history)
    app = FastAPI(title="Estimation web table")

    @app.get("/api/bots")
    def list_bots() -> list[BotView]:
        return [_bot_view(entry) for entry in store.catalog.entries]

    @app.post("/api/games")
    def new_game(body: NewGame) -> GameView:
        humans = sum(spec == HUMAN for spec in body.seats)
        if humans > 1:
            raise HTTPException(422, "at most one seat can be human")
        return store.create(body.seats, body.seed, "play" if humans else "watch")

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> GameView:
        entry = store.get(game_id)
        with entry.session.lock:
            return store.view(entry)

    @app.get("/api/games/{game_id}/frames")
    def get_frames(
        game_id: str,
        start: Annotated[int, Query(ge=0)] = 0,
        count: Annotated[int, Query(ge=1, le=MAX_FRAMES)] = 1,
    ) -> FramesView:
        entry = store.get(game_id)
        session = entry.session
        with session.lock:
            try:
                session.ensure(start + count - 1)
            except (GameError, WaitingForHuman) as error:
                raise HTTPException(409, str(error) or "waiting for a move") from error
            store.record(entry)
            return FramesView(
                frames=session.frames[start : start + count],
                known=len(session.frames),
                finished=session.finished,
            )

    @app.get("/api/games/{game_id}/seek")
    def seek(
        game_id: str,
        start: Annotated[int, Query(ge=0)],
        direction: Literal["next", "prev"],
        target: SeekTarget,
    ) -> SeekView:
        entry = store.get(game_id)
        session = entry.session
        with session.lock:
            try:
                index = session.seek(start, direction, target)
            except (GameError, WaitingForHuman) as error:
                raise HTTPException(409, str(error) or "waiting for a move") from error
            store.record(entry)
            return SeekView(index=index)

    @app.post("/api/games/{game_id}/moves")
    def move(game_id: str, body: Move) -> FramesView:
        entry = store.get(game_id)
        session = entry.session
        with session.lock:
            first_new = len(session.frames)
            try:
                session.play(body.action, body.frame)
            except IllegalActionError as error:
                raise HTTPException(422, str(error)) from error
            except GameError as error:
                raise HTTPException(409, str(error)) from error
            store.record(entry)
            return FramesView(
                frames=session.frames[first_new:],
                known=len(session.frames),
                finished=session.finished,
            )

    @app.post("/api/games/{game_id}/review")
    def review(game_id: str) -> GameView:
        session = store.get(game_id).session
        with session.lock:
            if not session.finished:
                raise HTTPException(409, "a game can be reviewed once it is over")
            seats = list(session.seats)
            actions = list(session.actions)
            seed = session.seed
        return store.create(seats, seed, "review", actions)

    @app.get("/api/history")
    def list_history(
        limit: Annotated[int, Query(ge=1, le=MAX_HISTORY)] = 100,
    ) -> list[SavedGameView]:
        return [list_view(game) for game in store.history.games()[:limit]]

    @app.post("/api/history/{game_id}/review")
    def review_saved(game_id: str) -> GameView:
        game = store.history.get(game_id)
        if game is None:
            raise HTTPException(404, f"no saved game {game_id!r}")
        seats = [seat["spec"] for seat in game["seats"]]
        return store.create(seats, game["seed"], "review", game["actions"])

    return app
