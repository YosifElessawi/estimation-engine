"""YosifBot: Yosif's bidding and estimating, PIMC's card play (design 5.9)."""

from estimation.engine import Observation, Phase

from ..pimc import PIMCBot
from .policy import Planner, YosifConfig

DEFAULT_YOSIF = YosifConfig()


class YosifBot:
    """Dash Calls, bids and estimates by Yosif's rules; cards by PIMC with 30 deals, counting
    the players ahead in the game double. Reads only its own Observation."""

    __slots__ = ("_pimc", "_planner", "config", "seed")

    def __init__(self, seed: int, config: YosifConfig = DEFAULT_YOSIF) -> None:
        self.seed = seed
        self.config = config
        self._pimc = PIMCBot(seed, config.pimc)
        self._planner = Planner(config)

    def __repr__(self) -> str:
        return f"YosifBot(seed={self.seed}, {self.config})"

    def predicted_make(self, obs: Observation, action: int) -> float | None:
        """The chance of making the number ``action`` commits to (see ``Planner``), used by the
        bidding evaluator's calibration check."""
        return self._planner.predicted_make(obs, action)

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        phase = obs.phase
        if phase is Phase.PLAY:
            return self._pimc.act(obs, legal)
        if phase is Phase.DASH:
            action = self._planner.dash(obs, legal)
        elif phase is Phase.AUCTION:
            action = self._planner.bid(obs, legal)
        elif phase is Phase.ESTIMATION:
            action = self._planner.estimate(obs, legal)
        else:
            raise ValueError("there is no decision to make in a finished round")
        if action not in legal:
            raise RuntimeError(f"YosifBot chose illegal action {action} in {phase.name}")
        return action
