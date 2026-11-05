"""Recorded PIMC games, the data for CardPlayNet's warm start (SPEC §6 M4).

Four PIMC bots (the M3 bot, HeuristicBot bidding) play whole Bolas. At every card decision with a
choice, the record keeps the acting seat's encoded Observation, its legal cards, PIMC's choice,
PIMC's search value of every legal card (the mean over its sampled deals of own round score minus
the mean of the other three) and, once the round ends, the round reward the seat got. Recording
does not change PIMC's play: the game is exactly the one four PIMC bots would play.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from estimation.bots.heuristic import HeuristicConfig
from estimation.bots.pimc import PIMCBot, PIMCConfig, best_index
from estimation.engine import Phase, play_action
from estimation.engine.actions import PLAY_BASE

from .config import RECORD_STAGE, TrainConfig, derive_seed
from .encoder import NUM_FEATURES, encode
from .games import RewardedBola


@dataclass(frozen=True, slots=True)
class CardRecords:
    """Recorded card decisions, one row each."""

    features: npt.NDArray[np.float16]  # (n, NUM_FEATURES) the acting seat's encoded Observation
    legal: npt.NDArray[np.bool_]  # (n, 52) its legal cards
    choice: npt.NDArray[np.int16]  # (n,) the card PIMC played
    values: npt.NDArray[np.float32]  # (n, 52) PIMC's search value per legal card, 0 elsewhere
    reward: npt.NDArray[np.float32]  # (n,) the seat's reward for the round
    game: npt.NDArray[np.int32]  # (n,) the game it came from
    # (n, 52) who held each unseen card (``guesses.guess_labels``), when the recorder kept it
    guesses: npt.NDArray[np.int8] | None = None

    def __len__(self) -> int:
        return len(self.choice)


def empty_records() -> CardRecords:
    return CardRecords(
        features=np.zeros((0, NUM_FEATURES), np.float16),
        legal=np.zeros((0, 52), np.bool_),
        choice=np.zeros(0, np.int16),
        values=np.zeros((0, 52), np.float32),
        reward=np.zeros(0, np.float32),
        game=np.zeros(0, np.int32),
    )


def concat_records(parts: Sequence[CardRecords]) -> CardRecords:
    """``parts`` as one set of records. Guess labels are kept when every part has them."""
    if not parts:
        return empty_records()
    labelled = [p.guesses for p in parts if p.guesses is not None]
    if labelled and len(labelled) != len(parts):
        raise ValueError("some records have guess labels and some do not")
    return CardRecords(
        features=np.concatenate([p.features for p in parts]),
        legal=np.concatenate([p.legal for p in parts]),
        choice=np.concatenate([p.choice for p in parts]),
        values=np.concatenate([p.values for p in parts]),
        reward=np.concatenate([p.reward for p in parts]),
        game=np.concatenate([p.game for p in parts]),
        guesses=np.concatenate(labelled) if labelled else None,
    )


def pimc_bots(index: int, config: TrainConfig) -> list[PIMCBot]:
    rules = config.rules
    pimc = PIMCConfig(
        samples=config.record.pimc_samples,
        heuristic=HeuristicConfig(super_call_min=rules.super_call_min),
        rules=rules,
    )
    return [
        PIMCBot(derive_seed(config.seed, RECORD_STAGE, index, 1 + seat), pimc) for seat in range(4)
    ]


def record_game(index: int, config: TrainConfig) -> CardRecords:
    """Play recorded game ``index`` of a run and return its card decisions."""
    rules = config.rules
    game = RewardedBola(derive_seed(config.seed, RECORD_STAGE, index, 0), rules, config.reward)
    bola = game.bola
    bots = pimc_bots(index, config)
    samples = config.record.pimc_samples
    features: list[npt.NDArray[np.float32]] = []
    legal_rows: list[npt.NDArray[np.bool_]] = []
    choices: list[int] = []
    value_rows: list[npt.NDArray[np.float32]] = []
    seats: list[int] = []
    rewards: list[float] = []
    open_rows: list[int] = []  # rows of the round in progress, waiting for its reward
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        if obs.phase is Phase.PLAY and len(legal) > 1:
            cards = [action - PLAY_BASE for action in legal]
            candidates, totals = bots[seat].search(obs, cards)
            choice = candidates[best_index(totals)]
            mask = np.zeros(52, np.bool_)
            mask[cards] = True
            values = np.zeros(52, np.float32)
            for card, total in zip(candidates, totals, strict=True):
                values[card] = total / samples
            open_rows.append(len(choices))
            features.append(encode(obs, rules))
            legal_rows.append(mask)
            choices.append(choice)
            value_rows.append(values)
            seats.append(seat)
            rewards.append(0.0)
            action = play_action(choice)
        else:
            action = bots[seat].act(obs, legal)
        round_rewards = game.step(action)
        if round_rewards is not None:
            for row in open_rows:
                rewards[row] = round_rewards[seats[row]]
            open_rows.clear()
    if not choices:
        return empty_records()
    n = len(choices)
    return CardRecords(
        features=np.stack(features).astype(np.float16),
        legal=np.stack(legal_rows),
        choice=np.array(choices, np.int16),
        values=np.stack(value_rows),
        reward=np.array(rewards, np.float32),
        game=np.full(n, index, np.int32),
    )


def save_records(path: str, records: CardRecords) -> None:
    if records.guesses is None:
        np.savez(
            path,
            features=records.features,
            legal=records.legal,
            choice=records.choice,
            values=records.values,
            reward=records.reward,
            game=records.game,
        )
        return
    np.savez(
        path,
        features=records.features,
        legal=records.legal,
        choice=records.choice,
        values=records.values,
        reward=records.reward,
        game=records.game,
        guesses=records.guesses,
    )


def load_records(path: str) -> CardRecords:
    """Recorded decisions saved by ``save_records``. Raises ValueError for a file recorded with
    another encoder: its features do not fit the nets, and the games behind it followed the
    rules of that time, so record again."""
    with np.load(path) as data:
        width = int(data["features"].shape[1])
        if width != NUM_FEATURES:
            raise ValueError(
                f"{path} was recorded with {width} features; this encoder has {NUM_FEATURES}. "
                "Delete it and record again."
            )
        return CardRecords(
            features=data["features"],
            legal=data["legal"],
            choice=data["choice"],
            values=data["values"],
            reward=data["reward"],
            game=data["game"],
            guesses=data["guesses"] if "guesses" in data.files else None,
        )
