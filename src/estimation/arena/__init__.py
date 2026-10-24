"""The duplicate arena (SPEC §6 M2): play lineups of bots on the same deals and rate them by
Relative Score Advantage and Elo. CLI: ``python -m estimation.arena``."""

from .duplicate import ArenaRun, SetResult, play_set, run_arena
from .match import BolaRecord, BotError, play_bola
from .ratings import Interval, PairRecord, bradley_terry, mean_ci, pair_records, rsa
from .timing import Timing

__all__ = [
    "ArenaRun",
    "BolaRecord",
    "BotError",
    "Interval",
    "PairRecord",
    "SetResult",
    "Timing",
    "bradley_terry",
    "mean_ci",
    "pair_records",
    "play_bola",
    "play_set",
    "rsa",
    "run_arena",
]
