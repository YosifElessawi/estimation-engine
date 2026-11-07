"""Bots by name, for the arena CLI: ``name`` or ``name:key=value:key=value``.

Options set fields of the bot's config dataclass, so ``heuristic:risk_aversion=0.5`` is a
HeuristicBot with ``HeuristicConfig(risk_aversion=0.5)``. Any bot also takes ``noise`` and
``play_noise`` (0 to 1), which wrap it in a NoisyBot that plays that share of its decisions (or
of its card plays) at random. ``rl:checkpoint=<path>`` plays a trained self-play checkpoint (M4);
PyTorch loads only when an rl bot is made.

``search`` is the Search bot: ``bidding=yosif`` (the default) or ``bidding=heuristic``, and any
field of its card search (``PIMCConfig``, for example ``samples=8``) or of Yosif's bidding
(``YosifConfig``, for example ``risk_mode=false``). ``yosif`` is its old name and loads it with
Yosif's bidding, so saved games and ladder runs that name it play as before. ``pimc`` is the
plain M3 search, a frozen measuring stick.
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
from .search import YOSIF, SearchBot, SearchConfig
from .yosif import YosifConfig

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


def _hybrid(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    from .hybrid import HybridBot, HybridConfig  # PyTorch loads only when a hybrid bot is made

    return HybridBot(seed, configure(HybridConfig(rules=rules), options, "hybrid"))


def _onenet(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    """Step 5's one net inside the Hybrid search (estimation.onenet.search): ``net`` is the
    player's file; ``own=true`` plays with no checkpoint, the player guessing the hidden cards
    itself; ``judge`` is ``round`` (round points) or ``king`` (its own King chance); every other
    option a HybridConfig field."""
    from estimation.onenet.search import ROUND, make  # PyTorch loads only with the Hybrid

    from .hybrid import HybridConfig

    net = options.pop("net", "")
    if not net:
        raise ValueError("bot onenet needs net=<the player's .npz file>")
    own = _parse(options.pop("own", "false"), bool, "bot option own") is True
    judge = options.pop("judge", ROUND)
    if own and {"checkpoint", "deals"} & options.keys():
        raise ValueError(
            "bot onenet with own=true takes no checkpoint or deals: the player imagines the deals"
        )
    config = configure(HybridConfig(rules=rules), options, "onenet")
    return make(seed, config, net, own=own, judge=judge)


def _search(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    heuristic = HeuristicConfig(super_call_min=rules.super_call_min)
    config = SearchConfig(
        pimc=PIMCConfig(heuristic=heuristic, rules=rules, leader_weight=True),
        yosif=YosifConfig(rules=rules),
    )
    own: dict[str, str] = {}
    search: dict[str, str] = {}
    bidding: dict[str, str] = {}
    pimc_fields = {field.name for field in dataclasses.fields(PIMCConfig)}
    yosif_fields = {field.name for field in dataclasses.fields(YosifConfig)}
    for key, value in options.items():
        if key in pimc_fields:
            search[key] = value
        elif key in yosif_fields:
            bidding[key] = value
        else:
            own[key] = value
    config = dataclasses.replace(
        configure(config, own, "search"),
        pimc=configure(config.pimc, search, "search"),
        yosif=configure(config.yosif, bidding, "search"),
    )
    if bidding and config.bidding != YOSIF:
        raise ValueError(
            f"bot search only uses {', '.join(sorted(bidding))} with bidding=yosif, "
            f"not bidding={config.bidding}"
        )
    return SearchBot(seed, config)


def _yosif(seed: int, options: dict[str, str], rules: RulesConfig) -> Bot:
    """The Search bot's old name: the Search bot with Yosif's bidding."""
    if "bidding" in options:
        raise ValueError("bot yosif always bids by Yosif's rules; use search:bidding=... instead")
    return _search(seed, {**options, "bidding": YOSIF}, rules)


_FACTORIES: dict[str, BotFactory] = {
    "random": _random,
    "heuristic": _heuristic,
    "pimc": _pimc,
    "search": _search,
    "rl": _rl,
    "hybrid": _hybrid,
    "onenet": _onenet,
    "yosif": _yosif,
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
