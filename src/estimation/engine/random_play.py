"""Seeded random play: every decision taken uniformly from the legal actions.

Used by the property tests and the benchmark (SPEC §5.5). Each policy owns its generator, so
the same seeds always give the same game.
"""

import random

from .bola import Bola
from .config import DEFAULT_CONFIG, RulesConfig
from .observation import Phase
from .round import Round
from .scoring import RoundResult

DEFAULT_MAX_STEPS = 100_000


class RandomPolicy:
    """Picks uniformly from the legal actions it is given, from its own seeded generator."""

    __slots__ = ("_choice",)

    def __init__(self, seed: int) -> None:
        self._choice = random.Random(seed).choice

    def __call__(self, legal: list[int]) -> int:
        return self._choice(legal)


def play_random_round(rnd: Round, seed: int) -> RoundResult:
    """Finish ``rnd`` with random play and return its result."""
    choice = random.Random(seed).choice
    while rnd.phase is not Phase.DONE:
        rnd.step(choice(rnd.legal_actions()))
    return rnd.result()


def play_random_bola(
    seed: int,
    policy_seed: int,
    config: RulesConfig = DEFAULT_CONFIG,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Bola:
    """Play a whole Bola dealt from ``seed`` with random play seeded by ``policy_seed``."""
    bola = Bola(seed, config)
    choice = random.Random(policy_seed).choice
    for _ in range(max_steps):
        if bola.is_over:
            return bola
        bola.step(choice(bola.legal_actions()))
    if bola.is_over:
        return bola
    raise RuntimeError(f"the game did not end within {max_steps} steps")
