"""Running training games (SPEC §6 M4): a Bola that reports each seat's reward per round.

The reward is ``round_reward`` divided by the carry the round was played at (x2, x4, ... after
skipped or all-lose rounds, GAM-4). The carry is the same for every decision in the round, so the
division changes no choice between actions, but it keeps the training targets in range: a
bidder that starts from scratch passes often, and a run of skipped rounds would otherwise make
rewards grow without bound.

This is trainer code, outside the bots' isolation guard: it holds a real Bola. Players and nets
are only ever shown ``Bola.observation(seat)`` of the seat to act.
"""

from estimation.engine import Bola, RulesConfig

from .standings import RewardConfig, round_reward, rounds_remaining

Rewards = tuple[float, float, float, float]


class RewardedBola:
    """A Bola whose ``step`` returns every seat's round reward when the step ends a round."""

    __slots__ = ("_before", "_fixed", "_left_before", "_reward", "bola")

    def __init__(self, seed: int, rules: RulesConfig, reward: RewardConfig) -> None:
        self.bola = Bola(seed, rules)
        self._reward = reward
        self._before = self.bola.scores
        self._fixed = False
        self._left_before = 0
        self._start_round()

    def _rounds_left(self) -> int:
        obs = self.bola.observation(0)  # public fields only: round number and fixed trumps
        return rounds_remaining(obs.round_no, len(obs.fixed_left), obs.fixed_trump is not None)

    def _start_round(self) -> None:
        bola = self.bola
        self._before = bola.scores
        self._fixed = bola.round.fixed_trump is not None
        self._left_before = self._rounds_left()

    def step(self, action: int) -> Rewards | None:
        """Apply ``action``; if it finished a round, each seat's reward for that round."""
        bola = self.bola
        played = len(bola.history)
        bola.step(action)
        if len(bola.history) == played:
            return None
        result = bola.history[-1]
        left_after = 0 if bola.is_over else self._rounds_left()
        rewards = tuple(
            round_reward(
                self._before,
                result.scores,
                seat,
                self._fixed,
                self._left_before,
                left_after,
                self._reward,
            )
            / result.carry
            for seat in range(4)
        )
        if not bola.is_over:
            self._start_round()
        return rewards[0], rewards[1], rewards[2], rewards[3]
