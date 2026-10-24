"""The bot interface (SPEC §6 M2)."""

from collections.abc import Callable
from typing import Protocol

from estimation.engine import Observation, RulesConfig


class Bot(Protocol):
    """A player. ``act`` gets the acting seat's own Observation and the legal action ids, and
    returns one of them. A bot never sees another seat's cards."""

    def act(self, obs: Observation, legal: list[int]) -> int: ...


BotFactory = Callable[[int, dict[str, str], RulesConfig], Bot]
"""Builds a bot from a seed, its command-line options (``name:key=value``) and the rules of the
game it will play."""
