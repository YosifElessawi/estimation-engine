"""Bots (SPEC §6 M2-M5). Every bot implements ``Bot.act(obs, legal) -> int`` and sees only its
own Observation."""

from .base import Bot, BotFactory
from .heuristic import HeuristicBot, HeuristicConfig
from .random_bot import RandomBot
from .registry import BOT_NAMES, make_bot

__all__ = [
    "BOT_NAMES",
    "Bot",
    "BotFactory",
    "HeuristicBot",
    "HeuristicConfig",
    "RandomBot",
    "make_bot",
]
