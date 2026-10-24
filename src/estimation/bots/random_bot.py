"""RandomBot (SPEC §6 M2): picks uniformly from the legal actions. It anchors Elo at 800."""

import random

from estimation.engine import Observation


class RandomBot:
    """Uniform random play from its own seeded generator, so a game can be replayed."""

    __slots__ = ("_choice", "seed")

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._choice = random.Random(seed).choice

    def __repr__(self) -> str:
        return f"RandomBot(seed={self.seed})"

    def act(self, obs: Observation, legal: list[int]) -> int:
        return self._choice(legal)
