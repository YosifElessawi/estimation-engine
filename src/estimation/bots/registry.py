"""Bots by name, for the arena CLI: ``name`` or ``name:key=value:key=value``.

Options set fields of the bot's config dataclass, so ``heuristic:risk_aversion=0.5`` is a
HeuristicBot with ``HeuristicConfig(risk_aversion=0.5)``. Any bot also takes ``noise`` and
``play_noise`` (0 to 1), which wrap it in a NoisyBot that plays that share of its decisions (or
of its card plays) at random. ``rl:checkpoint=<path>`` plays a trained self-play checkpoint (M4);
PyTorch loads only when an rl bot is made.
"""

import dataclasses
from typing import TYPE_CHECKING

from estimation.engine import RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .base import Bot, BotFactory
from .heuristic import HeuristicBot, HeuristicConfig
from .noisy import NoisyBot
from .pimc import PIMCBot, PIMCConfig
from .random_bot import RandomBot

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _random(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    if options:
        raise ValueError(f"the random bot takes no options, got {', '.join(sorted(options))}")
    return RandomBot(seed)


def _heuristic(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    config = HeuristicConfig(super_call_min=rules.super_call_min)
    return HeuristicBot(configure(config, options, "heuristic"))


def _pimc(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    heuristic = HeuristicConfig(super_call_min=rules.super_call_min)
    return PIMCBot(seed, configure(PIMCConfig(heuristic=heuristic, rules=rules), options, "pimc"))


def _rl(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    from .rl import RLBot, RLConfig  # PyTorch loads only when an rl bot is made

    heuristic = HeuristicConfig(super_call_min=rules.super_call_min)
    return RLBot(seed, configure(RLConfig(heuristic=heuristic, rules=rules), options, "rl"))


_FACTORIES: dict[str, BotFactory] = {
    "random": _random,
    "heuristic": _heuristic,
    "pimc": _pimc,
    "rl": _rl,
}
BOT_NAMES = tuple(_FACTORIES)
NOISE_OPTIONS = ("noise", "play_noise")


def make_bot(spec: str, seed: int, rules: RulesConfig = DEFAULT_CONFIG) -> Bot:
    """The bot named by ``spec`` (``name`` or ``name:key=value:...``), seeded with ``seed``, for
    a game played by ``rules``."""
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
    noise = {key: _rate(options.pop(key), key) for key in NOISE_OPTIONS if key in options}
    bot = factory(seed, options, rules)
    if not noise:
        return bot
    return NoisyBot(
        bot, noise=noise.get("noise", 0.0), play_noise=noise.get("play_noise"), seed=seed
    )


def _rate(text: str, key: str) -> float:
    try:
        rate = float(text)
    except ValueError:
        raise ValueError(f"bot option {key}={text} is not a number") from None
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"bot option {key}={text} must be between 0 and 1")
    return rate


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
    if kind is str:
        return text
    if kind is int or kind is float:
        try:
            return kind(text)
        except ValueError:
            raise ValueError(f"{label} is not a {kind.__name__}") from None
    raise ValueError(f"{label} cannot be set from the command line")
