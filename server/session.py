"""A game at the web table: a Bola, four seats and the frames the browser steps through.

A seat is a bot spec or ``"human"``. With no human seat the game is **watched**: frames show
every hand and the Why panel of RL bots, and are generated on demand as the viewer steps forward.
With one human seat the game is **played**: bots move until it is the human's turn, and frames
are built from the human seat's own Observation only, with its legal actions attached when it is
to act. A **review** replays a finished game's action log as a watched game.

Frames are numbered from 0 and never change once made:

- ``deal``: a round is dealt (its first decision is next)
- ``action``: a seat acted; a card that completes a trick carries the trick's winner
- ``round_end``: the round's result and the game scores after it
- ``game_end``: the final scores and the winners (GAM-5)

Bots are seeded from the game seed and their seat, so a game is replayed by its seats and seed
(plus the human's actions).
"""

import threading
from typing import Literal, TypedDict

import numpy as np

from estimation.bots import Bot
from estimation.engine import Bola, IllegalActionError, Observation, Round, RulesConfig
from estimation.engine.actions import is_play
from estimation.engine.config import DEFAULT_CONFIG

from .catalog import Catalog
from .views import (
    ActionView,
    EventView,
    RoundResultView,
    RoundSummary,
    TableView,
    describe_action,
    result_view,
    summary_view,
    table_view,
)
from .why import WhyView, explain, rl_bot

HUMAN = "human"
Mode = Literal["watch", "play", "review"]
SeekTarget = Literal["trick", "round_end", "deal", "end"]


class Frame(TypedDict):
    """One step of the game as the browser shows it. ``scores`` are the game scores at this
    frame and ``history`` the rounds scored so far. ``legal`` is set only on the frame where
    the human must act."""

    index: int
    kind: str
    round_no: int
    event: EventView | None
    why: WhyView | None
    table: TableView
    scores: list[int]
    history: list[RoundSummary]
    result: RoundResultView | None
    winners: list[int] | None
    legal: list[ActionView] | None


class WaitingForHuman(Exception):
    """The human seat is to act."""


class GameError(ValueError):
    """A request the game cannot take, such as a move out of turn."""


def bot_seed(seed: int, seat: int) -> int:
    """The seed of the bot at ``seat`` in a game seeded ``seed``."""
    return int(np.random.SeedSequence([seed, seat, 6]).generate_state(1)[0])


class GameSession:
    """One game. Every method that touches the game holds ``lock``."""

    def __init__(
        self,
        game_id: str,
        seats: list[str],
        seed: int,
        catalog: Catalog,
        *,
        mode: Mode,
        script: list[int] | None = None,
        rules: RulesConfig = DEFAULT_CONFIG,
    ) -> None:
        if len(seats) != 4:
            raise GameError("a game needs exactly 4 seats")
        humans = [seat for seat, spec in enumerate(seats) if spec == HUMAN]
        if mode == "play" and len(humans) != 1:
            raise GameError("play mode needs exactly one human seat")
        if mode == "watch" and humans:
            raise GameError("a watched game has only bots")
        if mode == "review" and script is None:
            raise GameError("a review needs the game's action log")
        self.id = game_id
        self.seats = list(seats)
        self.seed = seed
        self.mode: Mode = mode
        self.human = humans[0] if mode == "play" else None
        self.lock = threading.Lock()
        self.bots: dict[int, Bot] = {
            seat: catalog.make(spec, bot_seed(seed, seat), rules)
            for seat, spec in enumerate(seats)
            if spec != HUMAN
        }
        self.explained = {seat for seat, bot in self.bots.items() if rl_bot(bot) is not None}
        self.bola = Bola(seed, rules)
        self.actions: list[int] = []
        self.frames: list[Frame] = []
        self._script = list(script or [])
        self._events: list[EventView] = []
        self._history: list[RoundSummary] = []  # rounds shown as scored so far
        self._add(self.bola.round, "deal")
        if mode == "play":
            self._run_bots()

    @property
    def finished(self) -> bool:
        return self.bola.is_over

    @property
    def spectator(self) -> bool:
        return self.mode != "play"

    # Views

    def _table(self, source: Bola | Round) -> TableView:
        """The table of ``source`` for this game's viewer. A player's view reads only the human
        seat's Observation."""
        if self.human is not None:
            return table_view(source.observation(self.human), self._events)
        views = [source.observation(seat) for seat in range(4)]
        return table_view(views[0], self._events, [view.hand for view in views])

    def _add(
        self,
        source: Bola | Round,
        kind: str,
        *,
        event: EventView | None = None,
        why: WhyView | None = None,
        scores: tuple[int, int, int, int] | None = None,
    ) -> None:
        table = self._table(source)
        result = None
        winners = None
        if kind == "round_end":
            self._history.append(summary_view(self.bola.history[-1]))
        if kind in ("round_end", "game_end"):
            result = result_view(self.bola.history[-1])
        if kind == "game_end":
            winners = list(self.bola.winners())
        self.frames.append(
            Frame(
                index=len(self.frames),
                kind=kind,
                round_no=table["round_no"],
                event=event,
                why=why,
                table=table,
                scores=list(scores if scores is not None else table["scores"]),
                history=list(self._history),
                result=result,
                winners=winners,
                legal=None,
            )
        )

    # Moves

    def _step(self) -> None:
        """One decision by a bot, or by the action log in a review."""
        bola = self.bola
        seat = bola.current_player
        if seat is None:
            raise GameError("the game is over")
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        why = None
        if self._script:
            action = self._script.pop(0)
            if action not in legal:
                raise GameError(f"the action log's move {action} is illegal here")
        elif seat == self.human or seat not in self.bots:
            raise WaitingForHuman
        else:
            action = self.bots[seat].act(obs, legal)
        if self.spectator and seat in self.explained:
            why = explain(self.bots[seat], obs, legal, action)
        self._apply(seat, obs, action, why)

    def _apply(self, seat: int, obs: Observation, action: int, why: WhyView | None) -> None:
        """Apply ``seat``'s action (``obs`` is its view just before) and add its frames."""
        bola = self.bola
        rnd = bola.round
        done_tricks = len(obs.tricks)
        view = describe_action(obs, action)
        bola.step(action)
        self.actions.append(action)
        winner = None
        if is_play(action):
            tricks = rnd.observation(seat).tricks
            if len(tricks) > done_tricks:
                winner = tricks[-1].winner
        else:
            self._events.append(EventView(seat=seat, action=view, trick_winner=None))
        event = EventView(seat=seat, action=view, trick_winner=winner)
        self._add(rnd, "action", event=event, why=why)
        if bola.is_over or bola.round is not rnd:
            self._add(rnd, "round_end", scores=bola.scores)
            if bola.is_over:
                self._add(rnd, "game_end", scores=bola.scores)
            self._events = []
            if not bola.is_over:
                self._add(bola.round, "deal")

    def _run_bots(self) -> None:
        """Play bots until the human must act or the game ends, then offer the human's legal
        actions on the last frame."""
        while not self.bola.is_over and self.bola.current_player != self.human:
            self._step()
        if self.human is not None and not self.bola.is_over:
            obs = self.bola.observation(self.human)
            self.frames[-1]["legal"] = [
                describe_action(obs, action) for action in self.bola.legal_actions()
            ]

    def play(self, action: int, frame: int | None = None) -> None:
        """The human's move. ``frame``, when given, must be the latest frame: a move made on an
        older view is refused."""
        if self.human is None:
            raise GameError("nobody plays in a watched game")
        if self.bola.is_over:
            raise GameError("the game is over")
        if self.bola.current_player != self.human:
            raise GameError("it is not your turn")
        if frame is not None and frame != len(self.frames) - 1:
            raise GameError("the table has moved on; reload the latest frame")
        if action not in self.bola.legal_actions():
            raise IllegalActionError(f"action {action} is not legal now")
        self._apply(self.human, self.bola.observation(self.human), action, None)
        self._run_bots()

    def ensure(self, index: int) -> None:
        """Generate frames up to ``index`` in a watched game (fewer if the game ends first)."""
        if self.mode == "play":
            return
        while len(self.frames) <= index and not self.bola.is_over:
            self._step()

    def seek(self, start: int, direction: Literal["next", "prev"], target: SeekTarget) -> int:
        """The index of the next (or previous) frame from ``start`` that is the end of a trick,
        a round's result, a deal, or the end of the game. Generates frames as needed in a
        watched game; stays at the last or first frame when there is none."""
        if direction == "prev":
            for index in range(min(start, len(self.frames)) - 1, -1, -1):
                if _matches(self.frames[index], target):
                    return index
            return 0
        index = start + 1
        while True:
            self.ensure(index)
            if index >= len(self.frames):
                return len(self.frames) - 1
            if _matches(self.frames[index], target):
                return index
            index += 1


def _matches(frame: Frame, target: SeekTarget) -> bool:
    if target == "trick":
        event = frame["event"]
        return event is not None and event["trick_winner"] is not None
    if target == "end":
        return frame["kind"] == "game_end"
    return frame["kind"] == target
