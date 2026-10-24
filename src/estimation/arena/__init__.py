"""The duplicate arena (SPEC §6 M2): play lineups of bots on the same deals and rate them by
Relative Score Advantage and Elo. CLI: ``python -m estimation.arena``."""

from .duplicate import ArenaRun, SetResult, play_set, run_arena
from .match import BolaRecord, BotError, play_bola
from .timing import Timing

__all__ = [
    "ArenaRun",
    "BolaRecord",
    "BotError",
    "SetResult",
    "Timing",
    "play_bola",
    "play_set",
    "run_arena",
]
