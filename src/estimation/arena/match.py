"""Playing one Bola with four bots (SPEC §6 M2)."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from estimation.bots import Bot
from estimation.engine import Bola, IllegalActionError, Phase, RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .timing import Timing


class BotError(RuntimeError):
    """A bot chose an action that is not legal."""


@dataclass(frozen=True, slots=True)
class BolaRecord:
    """A finished Bola: the game scores by seat, and how many rounds it took."""

    scores: tuple[int, int, int, int]
    rounds: int


def play_bola(
    seed: int,
    bots: Sequence[Bot],
    config: RulesConfig = DEFAULT_CONFIG,
    timers: Sequence[Timing] | None = None,
) -> BolaRecord:
    """Play a whole Bola dealt from ``seed``, ``bots[seat]`` deciding for each seat.

    Each bot is shown only its own seat's Observation and the legal actions. With ``timers``,
    the time of each card decision goes to ``timers[seat]``.
    """
    if len(bots) != 4:
        raise ValueError("a Bola needs exactly four bots")
    bola = Bola(seed, config)
    play = Phase.PLAY
    clock = time.perf_counter
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        bot = bots[seat]
        legal = bola.legal_actions()
        obs = bola.observation(seat)
        if timers is not None and obs.phase is play:
            start = clock()
            action = bot.act(obs, legal)
            timers[seat].add(clock() - start)
        else:
            action = bot.act(obs, legal)
        try:
            bola.step(action)
        except IllegalActionError as error:
            raise BotError(f"seat {seat} ({bot!r}) chose an illegal action: {error}") from error
    return BolaRecord(bola.scores, len(bola.history))
