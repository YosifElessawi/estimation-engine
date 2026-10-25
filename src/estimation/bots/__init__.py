"""Bots (SPEC §6 M2-M5). Every bot implements ``Bot.act(obs, legal) -> int`` and sees only its
own Observation."""

from .base import Bot, BotFactory
from .heuristic import HeuristicBot, HeuristicConfig
from .noisy import NoisyBot
from .pimc import PIMCBot, PIMCConfig
from .random_bot import RandomBot
from .registry import BOT_NAMES, make_bot
from .yosif import YosifBot, YosifConfig

__all__ = [
    "BOT_NAMES",
    "Bot",
    "BotFactory",
    "HeuristicBot",
    "HeuristicConfig",
    "NoisyBot",
    "PIMCBot",
    "PIMCConfig",
    "RandomBot",
    "YosifBot",
    "YosifConfig",
    "make_bot",
]
