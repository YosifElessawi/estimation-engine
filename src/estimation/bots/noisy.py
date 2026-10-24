"""NoisyBot: another bot that sometimes plays a random legal move instead.

It gives the Elo ladder bots of in-between strength, so no two neighbours on the ladder are so far
apart that one wins every set. ``noise`` applies to every decision; ``play_noise``, when set,
replaces it for card play, so bidding can stay exact while card play gets sloppier.
"""

import random

from estimation.engine import Observation, Phase

from .base import Bot


class NoisyBot:
    """Plays like ``inner``, except a random share of decisions goes to a uniformly random legal
    action from its own seeded generator."""

    __slots__ = ("_rng", "inner", "noise", "play_noise", "seed")

    def __init__(
        self, inner: Bot, *, noise: float = 0.0, play_noise: float | None = None, seed: int
    ) -> None:
        for label, rate in (("noise", noise), ("play_noise", play_noise)):
            if rate is not None and not 0.0 <= rate <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1, got {rate}")
        self.inner = inner
        self.noise = noise
        self.play_noise = play_noise
        self.seed = seed
        self._rng = random.Random(f"noise-{seed}")

    def __repr__(self) -> str:
        return f"NoisyBot({self.inner!r}, noise={self.noise}, play_noise={self.play_noise})"

    def act(self, obs: Observation, legal: list[int]) -> int:
        rate = self.noise
        if obs.phase is Phase.PLAY and self.play_noise is not None:
            rate = self.play_noise
        if rate and self._rng.random() < rate:
            return self._rng.choice(legal)
        return self.inner.act(obs, legal)
