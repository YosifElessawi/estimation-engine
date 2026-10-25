"""Training configuration (SPEC §6 M4): one TOML file plus a seed reproduces a training run.

Every section and key is optional; anything left out takes the default below. Unknown sections
or keys are errors, so a typo cannot silently fall back to a default.
"""

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from estimation.engine import RulesConfig

from .standings import RewardConfig


@dataclass(frozen=True, slots=True)
class RecordConfig:
    """The warm start's data: whole Bolas of four PIMC bots, every card decision recorded."""

    games: int = 400
    pimc_samples: int = 30  # deals per PIMC card decision, as the M3 bot


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    """CardPlayNet's warm start: imitate PIMC's card choices and search values."""

    epochs: int = 24
    batch: int = 1024
    lr: float = 1e-3
    value_weight: float = 1.0  # critic: the round reward PIMC's game actually got
    advantage_weight: float = 1.0  # each card's search value relative to PIMC's choice
    choice_weight: float = 0.5  # cross-entropy on PIMC's choice
    temperature: float = 0.1  # advantages (in units of 50 points) per unit of choice logit
    holdout: float = 0.05  # share of games kept out to measure agreement with PIMC
    augment: bool = True  # relabel suits at random in every batch (24 symmetric copies)


@dataclass(frozen=True, slots=True)
class OpponentConfig:
    """Who sits at the table in training games (shares of games)."""

    self_play: float = 0.70  # against itself and recent checkpoints
    heuristic: float = 0.15  # against HeuristicBot and its noisy versions
    pimc: float = 0.15  # against a lighter PIMC
    current_share: float = 0.5  # in self-play games, chance each other seat is the current net
    heuristic_noise: tuple[float, ...] = (0.0, 0.0, 0.02, 0.05, 0.1)
    pimc_samples: tuple[int, ...] = (4, 5, 6, 7, 8)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Deep Monte Carlo self-play: play games with the current nets, then regress each net's
    value of the action taken onto the round reward it got."""

    iterations: int = 60
    games: int = 160  # games per iteration
    chunk: int = 16  # games played in lockstep by one worker, sharing batched inference
    epochs: int = 2  # passes over each iteration's samples
    batch: int = 2048
    bid_lr: float = 5e-4
    card_lr: float = 2e-4
    huber: float = 1.0  # Huber loss threshold, in units of 50 points
    grad_clip: float = 5.0
    card_frozen_iterations: int = 4  # BiddingNet learns alone first, with the warm-start play
    epsilon_bid_start: float = 0.05  # random non-card actions, decaying linearly...
    epsilon_bid_end: float = 0.01  # ...to this by the last iteration
    bid_temperature_start: float = 0.2  # softmax temperature of bidding, in units of 50 points,
    bid_temperature_end: float = 0.05  # ...decaying linearly over the run
    epsilon_card: float = 0.02  # random card plays
    augment_cards: bool = True  # relabel suits at random in card-play batches
    card_anchor_weight: float = 1.0  # keep the PIMC imitation loss in card-play updates
    card_anchor_batch: int = 1024  # recorded PIMC decisions per card-play step
    eval_games: int = 16  # greedy games against three HeuristicBots per iteration, for the log
    snapshot_every: int = 5  # iterations between snapshots joining the opponent pool
    snapshots_kept: int = 5
    checkpoint_every: int = 10
    # Schedules over the run, as shares of the starting value at the last iteration (1: none).
    lr_end_factor: float = 1.0  # both learning rates decay linearly to this share
    card_anchor_end_factor: float = 1.0  # card_anchor_weight decays linearly to this share
    # Speed. Neither changes what is learned from a given set of games.
    overlap: bool = False  # play iteration k+1 with the nets from before k's update, during it
    balance: bool = False  # chunks hold one opponent kind each, the slowest submitted first


@dataclass(frozen=True, slots=True)
class NetConfig:
    hidden: int = 384
    device: str = "auto"  # "auto" picks MPS when available, else the CPU


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 0
    workers: int = 5
    rules: RulesConfig = field(default_factory=RulesConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    nets: NetConfig = field(default_factory=NetConfig)

    def __post_init__(self) -> None:
        mix = self.opponents
        if abs(mix.self_play + mix.heuristic + mix.pimc - 1.0) > 1e-9:
            raise ValueError("opponent shares self_play, heuristic and pimc must add up to 1")
        if not mix.heuristic_noise or not mix.pimc_samples:
            raise ValueError("opponent noise and PIMC sample lists must not be empty")
        if self.workers < 1 or self.selfplay.chunk < 1 or self.selfplay.games < 1:
            raise ValueError("workers, chunk and games must be at least 1")
        if self.selfplay.iterations < 1 or self.record.games < 1 or self.imitation.epochs < 1:
            raise ValueError("iterations, recorded games and imitation epochs must be at least 1")
        play = self.selfplay
        if play.lr_end_factor < 0 or play.card_anchor_end_factor < 0:
            raise ValueError("lr_end_factor and card_anchor_end_factor must be 0 or more")


_SECTIONS = ("rules", "reward", "record", "imitation", "opponents", "selfplay", "nets")

# Stage tags for derived seeds, so no two stages ever share a random stream.
RECORD_STAGE = 1
IMITATION_STAGE = 2
SELFPLAY_STAGE = 3


def linear_schedule(iterations: int, iteration: int, start: float, end: float) -> float:
    """A value going linearly from ``start`` at the first iteration to ``end`` at the last."""
    if iterations <= 1:
        return end
    return start + iteration / (iterations - 1) * (end - start)


def derive_seed(seed: int, *keys: int) -> int:
    """A 32-bit seed for one random choice of a run, from the run seed and its position."""
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1)[0])


def load_config(path: Path, seed: int | None = None) -> TrainConfig:
    """The config in the TOML file at ``path``; ``seed`` overrides the file's seed."""
    data = tomllib.loads(path.read_text())
    config = config_from_dict(data)
    return config if seed is None else dataclasses.replace(config, seed=seed)


def config_from_dict(data: Mapping[str, Any]) -> TrainConfig:
    top = TrainConfig()
    changes: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTIONS:
            if not isinstance(value, Mapping):
                raise ValueError(f"[{key}] must be a table")
            section = cast(Mapping[str, Any], value)
            changes[key] = _replace(getattr(top, key), section, key)
        elif key in ("seed", "workers"):
            changes[key] = _coerce(value, int, key)
        else:
            raise ValueError(f"unknown config key {key!r}")
    return dataclasses.replace(top, **changes)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """The config as plain data (TOML- and JSON-friendly), for checkpoints and logs."""
    return dataclasses.asdict(config)


def _replace(current: Any, values: Mapping[str, Any], section: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(current)}
    changes: dict[str, Any] = {}
    for key, value in values.items():
        if key not in fields:
            raise ValueError(f"unknown key {key!r} in [{section}]")
        old: object = getattr(current, key)
        changes[key] = _coerce(value, type(old), f"{section}.{key}", old)
    return dataclasses.replace(current, **changes)


def _coerce(value: Any, kind: type, label: str, old: Any = None) -> Any:
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be true or false")
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer")
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} must be a number")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string")
        return value
    if kind is tuple:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} must be a non-empty list")
        items = cast(list[object], value)
        first: object = cast(tuple[object, ...], old)[0] if old else 0.0
        return tuple(_coerce(item, type(first), label) for item in items)
    raise ValueError(f"{label} cannot be set from a config file")
