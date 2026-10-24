"""A full game, a Bola: 13 open rounds, then the fixed-trump rounds (GAM-1, FIX-1, FIX-5)."""

import numpy as np
import numpy.typing as npt

from .cards import deal
from .config import DEFAULT_CONFIG, RulesConfig
from .observation import Observation, Phase
from .round import FIXED_ORDER, OPEN_ROUNDS, IllegalActionError, Round
from .scoring import RoundResult


def top_seats(scores: tuple[int, int, int, int]) -> tuple[int, ...]:
    """Every seat tied for the highest score (GAM-5)."""
    best = max(scores)
    return tuple(seat for seat in range(4) if scores[seat] == best)


class Bola:
    """A full game. It deals each round from ``seed`` and the round number, picks each round's
    fixed trump, applies the carry, replays fixed-trump rounds whose trump changed (FIX-5) and
    moves to the next round by itself. Once ``is_over`` it behaves like a round in DONE.

    ``round`` is the current Round (the last one once the game is over).
    """

    __slots__ = (
        "_fixed_index",
        "carry",
        "config",
        "history",
        "replays",
        "round",
        "round_no",
        "scores",
        "seed",
    )

    seed: int
    config: RulesConfig
    round_no: int  # counts every round played, skipped and replayed ones included (GAM-2)
    scores: tuple[int, int, int, int]
    history: list[RoundResult]
    carry: int  # multiplier for the current round (GAM-4)
    replays: int  # fixed-trump replays so far (FIX-5)
    round: Round
    _fixed_index: int  # index in FIXED_ORDER of the next fixed trump to complete

    def __init__(self, seed: int, config: RulesConfig = DEFAULT_CONFIG) -> None:
        if seed < 0:
            raise ValueError("seed must be 0 or more")
        self.seed = seed
        self.config = config
        self.round_no = 1
        self.scores = (0, 0, 0, 0)
        self.history = []
        self.carry = 1
        self.replays = 0
        self._fixed_index = 0
        self.round = self._deal_round()

    @property
    def is_over(self) -> bool:
        return self._fixed_index >= len(FIXED_ORDER)

    @property
    def phase(self) -> Phase:
        return self.round.phase

    @property
    def current_player(self) -> int | None:
        return self.round.current_player

    def legal_actions(self) -> list[int]:
        return self.round.legal_actions()

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        return self.round.legal_mask()

    def observation(self, seat: int) -> Observation:
        return self.round.observation(seat)

    def step(self, action: int) -> None:
        """Apply a legal action; when it ends the round, record it and start the next one."""
        if self.is_over:
            raise IllegalActionError("the game is over: no action is legal")
        rnd = self.round
        rnd.step(action)
        if rnd.phase is Phase.DONE:
            self._end_round(rnd)

    def winners(self) -> tuple[int, ...]:
        """Every seat tied for the top game score; after the last round, the winners (GAM-5)."""
        return top_seats(self.scores)

    def _end_round(self, rnd: Round) -> None:
        result = rnd.result()
        self.history.append(result)
        s = self.scores
        r = result.scores
        self.scores = (s[0] + r[0], s[1] + r[1], s[2] + r[2], s[3] + r[3])  # GAM-5
        self.carry = result.next_carry  # GAM-4
        if rnd.fixed_trump is not None:
            if result.trump_changed and self.replays < self.config.max_fixed_replays:
                self.replays += 1  # FIX-5: the same fixed trump is played again
            else:
                self._fixed_index += 1
        if not self.is_over:
            self.round_no += 1
            self.round = self._deal_round()

    def _deal_round(self) -> Round:
        round_no = self.round_no
        fixed_trump = FIXED_ORDER[self._fixed_index] if round_no > OPEN_ROUNDS else None
        return Round.new(
            round_no, deal(self.seed, round_no), fixed_trump, self.carry, self.scores, self.config
        )
