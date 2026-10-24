"""Bots (SPEC §6 M2-M5). Every bot implements ``Bot.act(obs, legal) -> int`` and sees only its
own Observation."""

from .base import Bot, BotFactory
from .random_bot import RandomBot

__all__ = ["Bot", "BotFactory", "RandomBot"]
