"""Bots by name, for the arena CLI: ``name`` or ``name:key=value:key=value``.

Options set fields of the bot's config dataclass, so ``heuristic:risk_aversion=0.5`` is a
HeuristicBot with ``HeuristicConfig(risk_aversion=0.5)``.
"""

import dataclasses
from typing import TYPE_CHECKING

from .base import Bot, BotFactory
from .heuristic import HeuristicBot, HeuristicConfig
from .random_bot import RandomBot

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _random(seed: int, options: dict[str, str]) -> Bot:
    if options:
        raise ValueError(f"the random bot takes no options, got {', '.join(sorted(options))}")
    return RandomBot(seed)


def _heuristic(seed: int, options: dict[str, str]) -> Bot:
    return HeuristicBot(configure(HeuristicConfig(), options, "heuristic"))


_FACTORIES: dict[str, BotFactory] = {"random": _random, "heuristic": _heuristic}
BOT_NAMES = tuple(_FACTORIES)


def make_bot(spec: str, seed: int) -> Bot:
    """The bot named by ``spec`` (``name`` or ``name:key=value:...``), seeded with ``seed``."""
    name, *pairs = spec.split(":")
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"unknown bot {name!r}; known bots: {', '.join(_FACTORIES)}")
    options: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"bot option {pair!r} must look like key=value")
        options[key] = value
    return factory(seed, options)


def configure[T: DataclassInstance](config: T, options: dict[str, str], name: str) -> T:
    """``config`` with the given fields replaced, each parsed to the field's current type."""
    fields = {field.name for field in dataclasses.fields(config)}
    changes: dict[str, object] = {}
    for key, text in options.items():
        if key not in fields:
            raise ValueError(f"bot {name} has no option {key!r}")
        current: object = getattr(config, key)
        changes[key] = _parse(text, type(current), f"bot option {key}={text}")
    return dataclasses.replace(config, **changes)


def _parse(text: str, kind: type, label: str) -> object:
    if kind is bool:
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{label} is not a true/false value")
    if kind is int or kind is float:
        try:
            return kind(text)
        except ValueError:
            raise ValueError(f"{label} is not a {kind.__name__}") from None
    raise ValueError(f"{label} cannot be set from the command line")
