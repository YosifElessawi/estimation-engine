"""The public table's HTTP API, and the page it serves.

Only these routes exist; every other path is the page's static files or "not found":

- ``POST /api/games``: start a game from the player's name and their browser's visitor id.
  Returns the game and its token, which every other route needs (``X-Game-Token``).
- ``GET /api/games/{id}``: the game's seats and progress.
- ``GET /api/games/{id}/frames?start=&count=``: frames from the player's last move onwards.
  Earlier frames are refused, so the player can look back at the last trick (every frame carries
  it) but no further.
- ``POST /api/games/{id}/moves``: the player's move.
- ``POST /api/games/{id}/leave``: give the game up; it stays saved.

Frames are built from the player's own Observation only, as at the full table, and the seed is
never sent. The bots are shown as "B-Tig", with no spec or rating. A game lives in memory while
it is played and in ``Records`` always: a game dropped from memory (idle, or a restart) is
rebuilt from its action log when its player comes back.
"""

import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import RequestResponseEndpoint

from estimation.arena.games import code_version
from estimation.engine import IllegalActionError, RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from ..app import MAX_FRAMES, MAX_SEED, FramesView, Move, SeatView
from ..catalog import Catalog
from ..session import HUMAN, Frame, GameError, GameSession
from .records import GameRow, Progress, Records

logger = logging.getLogger(__name__)

BOT_NAME = "B-Tig"
BOT_SPEC = "hybrid"  # the bot behind every B-Tig seat
PUBLIC_SPEC = "b-tig"  # what the page is told a B-Tig seat is
HUMAN_SEAT = 0
NAME_MAX = 20
NAME_EXTRA = frozenset(" .-'_")
VISITOR = re.compile(r"^[A-Za-z0-9-]{8,64}$")
GameToken = Annotated[str | None, Header(alias="X-Game-Token")]


@dataclass(frozen=True, slots=True)
class Limits:
    """How much one visitor, and everyone together, may use the table."""

    games_in_memory: int = 20  # games being played at once
    idle_seconds: float = 600.0  # a game untouched this long may leave memory (it stays saved)
    open_per_visitor: int = 2  # a visitor's unfinished games; a new one leaves the oldest
    games_per_visitor_per_day: int = 30
    games_per_hour: int = 60  # new games by everyone together
    requests_per_window: int = 60  # per visitor or per game
    window_seconds: float = 10.0


def clean_name(raw: str) -> str | None:
    """The name as saved, or None if it is empty, too long or has other characters: letters
    and digits of any script (with their accents), spaces and ``. - ' _``."""
    name = " ".join(unicodedata.normalize("NFC", raw).split())
    if not 1 <= len(name) <= NAME_MAX:
        return None
    for char in name:
        if not (char.isalnum() or char in NAME_EXTRA or unicodedata.category(char) == "Mn"):
            return None
    return name


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def last_decision(frames: list[Frame], human: int) -> int:
    """The frame the human last acted on (0 before their first move): the first one they may
    see again."""
    for frame in reversed(frames):
        event = frame["event"]
        if frame["kind"] == "action" and event is not None and event["seat"] == human:
            return max(frame["index"] - 1, 0)
    return 0


class NewGame(BaseModel):
    name: str = Field(max_length=200)
    visitor: str = Field(max_length=64)


class GameView(TypedDict):
    """A game as the public page sees it. ``start`` is the first frame it may ask for."""

    id: str
    mode: Literal["play"]
    seed: None
    human_seat: int
    seats: list[SeatView]
    known: int
    finished: bool
    waiting: bool
    start: int


class Created(TypedDict):
    game: GameView
    token: str


class RateLimit:
    """At most ``count`` requests per ``seconds`` for each key."""

    def __init__(self, count: int, seconds: float, clock: Callable[[], float]) -> None:
        self.count = count
        self.seconds = seconds
        self._clock = clock
        self._seen: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            if len(self._seen) > 10_000:  # forget quiet keys now and then
                start = now - self.seconds
                self._seen = {k: q for k, q in self._seen.items() if q and q[-1] > start}
            times = self._seen.setdefault(key, deque())
            while times and times[0] <= now - self.seconds:
                times.popleft()
            if len(times) >= self.count:
                return False
            times.append(now)
            return True


@dataclass(slots=True)
class Live:
    """A game in memory. ``floor`` is the first frame its player may see."""

    session: GameSession
    token_hash: str
    visitor: str
    name: str
    floor: int
    seen: float
    left: bool = False


class PublicTable:
    """The games being played, and the rules for starting, reopening and leaving them."""

    def __init__(
        self,
        catalog: Catalog,
        records: Records,
        *,
        rules: RulesConfig = DEFAULT_CONFIG,
        bot: str = BOT_SPEC,
        limits: Limits | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.catalog = catalog
        self.records = records
        self.rules = rules
        self.bot = bot
        self.limits = limits if limits is not None else Limits()
        self.clock = clock
        self.now = now
        self.version = code_version()
        self.requests = RateLimit(
            self.limits.requests_per_window, self.limits.window_seconds, clock
        )
        self._live: OrderedDict[str, Live] = OrderedDict()
        self._lock = threading.Lock()

    # Memory

    def _make_room(self) -> bool:
        """Room for one more game in memory, dropping a finished or idle one if needed (it
        stays saved). Holds ``_lock``."""
        if len(self._live) < self.limits.games_in_memory:
            return True
        idle_before = self.clock() - self.limits.idle_seconds
        by_age = sorted(self._live.values(), key=lambda live: live.seen)
        drop = next((live for live in by_age if live.session.finished), None)
        if drop is None:
            drop = next((live for live in by_age if live.seen <= idle_before), None)
        if drop is None:
            return False
        del self._live[drop.session.id]
        return True

    def _admit(self, live: Live) -> Live:
        """Put ``live`` in memory, or the copy another request got there first."""
        with self._lock:
            found = self._live.get(live.session.id)
            if found is not None:
                return found
            if not self._make_room():
                raise HTTPException(503, "The table is full right now. Try again in a few minutes.")
            self._live[live.session.id] = live
            return live

    def _full(self) -> bool:
        with self._lock:
            return not self._make_room()

    # Saving

    def _progress(self, session: GameSession) -> Progress:
        bola = session.bola
        return Progress(
            updated_at=_iso(self.now()),
            rounds=len(bola.history),
            scores=list(bola.scores),
            finished_at=_iso(self.now()) if bola.is_over else None,
            winners=list(bola.winners()) if bola.is_over else None,
            actions=list(session.actions),
        )

    def _save(self, session: GameSession) -> None:
        self.records.save(session.id, self._progress(session))

    # Games

    def create(self, name: str, visitor: str) -> tuple[Live, str]:
        limits = self.limits
        moment = self.now()
        day = _iso(moment - timedelta(days=1))
        if self.records.started_since(day, visitor) >= limits.games_per_visitor_per_day:
            raise HTTPException(
                429, f"That's {limits.games_per_visitor_per_day} games today. Come back tomorrow."
            )
        if self.records.started_since(_iso(moment - timedelta(hours=1))) >= limits.games_per_hour:
            raise HTTPException(429, "Lots of new games right now. Try again in a few minutes.")
        still_open = self.records.open_games(visitor)
        for game_id in still_open[: max(len(still_open) - limits.open_per_visitor + 1, 0)]:
            self.leave_id(game_id)  # the oldest, so the new game keeps them within the limit
        if self._full():
            raise HTTPException(503, "The table is full right now. Try again in a few minutes.")
        seats = [self.bot if seat != HUMAN_SEAT else HUMAN for seat in range(4)]
        session = GameSession(
            uuid.uuid4().hex[:12],
            seats,
            secrets.randbelow(MAX_SEED + 1),
            self.catalog,
            mode="play",
            rules=self.rules,
        )
        token = secrets.token_urlsafe(24)
        live = Live(session, _hash(token), visitor, name, floor=0, seen=self.clock())
        stamp = _iso(moment)
        progress = self._progress(session)
        self.records.create(
            GameRow(
                id=session.id,
                visitor=visitor,
                name=name,
                seed=session.seed,
                human_seat=HUMAN_SEAT,
                seats=seats,
                version=self.version,
                started_at=stamp,
                updated_at=stamp,
                finished_at=progress["finished_at"],
                left_at=None,
                rounds=progress["rounds"],
                scores=progress["scores"],
                winners=progress["winners"],
                actions=progress["actions"],
            ),
            live.token_hash,
        )
        return self._admit(live), token

    def open(self, game_id: str, token: str | None) -> Live:
        """The game ``token`` belongs to, rebuilt from its saved moves if it is not in memory.
        A wrong or missing token gets the same answer as a game that does not exist."""
        missing = HTTPException(404, "No such game.")
        if token is None:
            raise missing
        with self._lock:
            live = self._live.get(game_id)
        if live is None:
            found = self.records.get(game_id)
            if found is None or not hmac.compare_digest(found[1], _hash(token)):
                raise missing
            row, token_hash = found
            if row["left_at"] is not None:
                raise HTTPException(410, "You left this game.")
            try:
                session = GameSession(
                    row["id"],
                    row["seats"],
                    row["seed"],
                    self.catalog,
                    mode="play",
                    script=row["actions"],
                    rules=self.rules,
                )
            except (GameError, IllegalActionError, ValueError) as error:
                logger.warning("could not rebuild game %s: %s", game_id, error)
                raise HTTPException(410, "This game can't be continued after an update.") from error
            if len(session.actions) != len(row["actions"]):
                self._save(session)  # the bots moved on past the saved moves
            floor = last_decision(session.frames, HUMAN_SEAT)
            live = self._admit(
                Live(session, token_hash, row["visitor"], row["name"], floor, self.clock())
            )
        elif not hmac.compare_digest(live.token_hash, _hash(token)):
            raise missing
        if live.left:
            raise HTTPException(410, "You left this game.")
        if not self.requests.allow(f"game:{game_id}"):
            raise HTTPException(429, "Too many requests. Slow down a little.")
        live.seen = self.clock()
        return live

    def leave_id(self, game_id: str) -> None:
        with self._lock:
            live = self._live.pop(game_id, None)
        if live is not None:
            live.left = True
        self.records.leave(game_id, _iso(self.now()))

    def view(self, live: Live) -> GameView:
        session = live.session
        seats = [
            SeatView(
                seat=seat,
                spec=HUMAN if spec == HUMAN else PUBLIC_SPEC,
                label=live.name if spec == HUMAN else BOT_NAME,
                elo=None,
                human=spec == HUMAN,
                explains=False,
            )
            for seat, spec in enumerate(session.seats)
        ]
        return GameView(
            id=session.id,
            mode="play",
            seed=None,
            human_seat=HUMAN_SEAT,
            seats=seats,
            known=len(session.frames),
            finished=session.finished,
            waiting=not session.finished and session.bola.current_player == HUMAN_SEAT,
            start=live.floor,
        )

    def frames(self, live: Live, start: int, count: int) -> FramesView:
        session = live.session
        with session.lock:
            if start < live.floor:
                raise HTTPException(403, "Only the current trick and the last trick are shown.")
            return FramesView(
                frames=session.frames[start : start + count],
                known=len(session.frames),
                finished=session.finished,
            )

    def move(self, live: Live, body: Move) -> FramesView:
        session = live.session
        with session.lock:
            if live.left:
                raise HTTPException(410, "You left this game.")
            decision = len(session.frames) - 1
            try:
                session.play(body.action, body.frame)
            except IllegalActionError as error:
                raise HTTPException(422, str(error)) from error
            except GameError as error:
                raise HTTPException(409, str(error)) from error
            live.floor = decision
            self._save(session)
            return FramesView(
                frames=session.frames[decision + 1 :],
                known=len(session.frames),
                finished=session.finished,
            )


async def _headers(request: Request, call_next: RequestResponseEndpoint) -> Response:
    response = await call_next(request)
    # The page's own files are named by their content and never change; everything else (the
    # page, the API) is checked again each time, so an update reaches every browser at once.
    if request.url.path.startswith("/_next/static/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def create_public_app(table: PublicTable, static_dir: Path | None = None) -> FastAPI:
    """The public API over ``table``, and the page's static files from ``static_dir`` (the
    public build of web/) when given."""
    app = FastAPI(title=BOT_NAME, docs_url=None, redoc_url=None, openapi_url=None)
    app.middleware("http")(_headers)

    @app.post("/api/games")
    def new_game(body: NewGame) -> Created:
        name = clean_name(body.name)
        if name is None:
            raise HTTPException(
                422, f"Use up to {NAME_MAX} letters, numbers, spaces and . - ' _ for your name."
            )
        if not VISITOR.match(body.visitor):
            raise HTTPException(422, "Reload the page and try again.")
        if not table.requests.allow(f"visitor:{body.visitor}"):
            raise HTTPException(429, "Too many requests. Slow down a little.")
        live, token = table.create(name, body.visitor)
        with live.session.lock:
            return Created(game=table.view(live), token=token)

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str, token: GameToken = None) -> GameView:
        live = table.open(game_id, token)
        with live.session.lock:
            return table.view(live)

    @app.get("/api/games/{game_id}/frames")
    def get_frames(
        game_id: str,
        start: Annotated[int, Query(ge=0)] = 0,
        count: Annotated[int, Query(ge=1, le=MAX_FRAMES)] = 1,
        token: GameToken = None,
    ) -> FramesView:
        return table.frames(table.open(game_id, token), start, count)

    @app.post("/api/games/{game_id}/moves")
    def move(game_id: str, body: Move, token: GameToken = None) -> FramesView:
        return table.move(table.open(game_id, token), body)

    @app.post("/api/games/{game_id}/leave")
    def leave(game_id: str, token: GameToken = None) -> dict[str, bool]:
        table.leave_id(table.open(game_id, token).session.id)
        return {"left": True}

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="page")
    return app
