"""RLBot (SPEC §6 M4): plays a self-play checkpoint.

Card play takes the legal card with the highest CardPlayNet advantage; Dash Calls, bids, With,
Pass, estimates and super calls take the legal action with the highest BiddingNet value. Both read
only the state encoder's features of the bot's own Observation. With ``heuristic_bidding``, the
non-card decisions are HeuristicBot's instead, which is how the warm-start checkpoint (trained
card play, no bidding yet) is played. Play is deterministic: the seed is unused.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from estimation.engine import Observation, Phase, RulesConfig
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG
from estimation.rl.encoder import encode
from estimation.rl.networks import Policy, load_policy

from .heuristic import DEFAULT_HEURISTIC, HeuristicBot, HeuristicConfig


@dataclass(frozen=True, slots=True)
class RLConfig:
    checkpoint: str = ""  # path to a checkpoint saved by the trainer
    heuristic_bidding: bool = False  # HeuristicBot's Dash Calls, bids and estimates
    heuristic: HeuristicConfig = DEFAULT_HEURISTIC  # used with heuristic_bidding
    rules: RulesConfig = DEFAULT_CONFIG  # the rules of the game, for the encoder's stakes


@lru_cache(maxsize=16)
def _cached_policy(path: str, mtime_ns: int) -> Policy:
    return load_policy(Path(path))


def policy_at(checkpoint: str) -> Policy:
    """The checkpoint's nets, loaded once per process (and again if the file changes)."""
    path = Path(checkpoint).resolve()
    return _cached_policy(str(path), path.stat().st_mtime_ns)


class RLBot:
    """Greedy play of a trained checkpoint. Reads only its own Observation."""

    __slots__ = ("_heuristic", "_policy", "config", "seed")

    def __init__(self, seed: int, config: RLConfig) -> None:
        if not config.checkpoint:
            raise ValueError("the rl bot needs a checkpoint, for example rl:checkpoint=<path>")
        self.seed = seed
        self.config = config
        self._policy = policy_at(config.checkpoint)
        self._heuristic = HeuristicBot(config.heuristic) if config.heuristic_bidding else None
        if self._heuristic is None and self._policy.bidding is None:
            raise ValueError(
                f"{config.checkpoint} has no trained BiddingNet; play it with "
                "heuristic_bidding=true"
            )

    def __repr__(self) -> str:
        return f"RLBot({self.config.checkpoint!r}, heuristic_bidding={self._heuristic is not None})"

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        if obs.phase is Phase.PLAY:
            values = self._policy.card_values(encode(obs, self.config.rules)[np.newaxis])[0]
            return max(legal, key=lambda action: values[action - PLAY_BASE])
        if self._heuristic is not None:
            return self._heuristic.act(obs, legal)
        values = self._policy.bid_values(encode(obs, self.config.rules)[np.newaxis])[0]
        return max(legal, key=lambda action: values[action])
