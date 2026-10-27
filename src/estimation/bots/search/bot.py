"""SearchBot: PIMC card search with its own play-out rules, and a choice of bidding.

Cards come from PIMC search (``pimc``): 30 sampled deals per decision, each card played out by
the Search bot's own play-out rules (``play_rules``) for every seat, counting the players ahead in
the game double. Dash Calls, bids and estimates follow Yosif's rules (``bots/yosif``) by default,
or HeuristicBot's with ``bidding="heuristic"``.
"""

from dataclasses import dataclass, field

from estimation.engine import Observation, Phase

from ..pimc import PIMCBot, PIMCConfig
from ..yosif.policy import Planner, YosifConfig
from .play_rules import choose_card

YOSIF = "yosif"
HEURISTIC = "heuristic"
BIDDINGS = (YOSIF, HEURISTIC)
CARD_SEARCH = PIMCConfig(leader_weight=True)  # 30 deals, players ahead in the game count double


@dataclass(frozen=True, slots=True)
class SearchConfig:
    bidding: str = YOSIF  # Dash Calls, bids and estimates: Yosif's rules or HeuristicBot's
    pimc: PIMCConfig = CARD_SEARCH  # card search, and HeuristicBot's settings for its bidding
    yosif: YosifConfig = field(default_factory=YosifConfig)  # Yosif's bidding


class SearchBot:
    """Cards by PIMC search with the Search bot's play-out rules; Dash Calls, bids and estimates
    by Yosif's rules or HeuristicBot's. Reads only its own Observation."""

    __slots__ = ("_pimc", "_planner", "config", "seed")

    def __init__(self, seed: int, config: SearchConfig | None = None) -> None:
        config = SearchConfig() if config is None else config
        if config.bidding not in BIDDINGS:
            raise ValueError(
                f"the Search bot's bidding is {' or '.join(BIDDINGS)}, got {config.bidding!r}"
            )
        self.seed = seed
        self.config = config
        self._pimc = PIMCBot(seed, config.pimc, choose_card)
        self._planner = Planner(config.yosif) if config.bidding == YOSIF else None

    def __repr__(self) -> str:
        return f"SearchBot(seed={self.seed}, {self.config})"

    def predicted_make(self, obs: Observation, action: int) -> float | None:
        """The chance of making the number ``action`` commits to (see ``Planner``), used by the
        bidding evaluator's calibration check; None with HeuristicBot bidding."""
        if self._planner is None:
            return None
        return self._planner.predicted_make(obs, action)

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        phase = obs.phase
        planner = self._planner
        if phase is Phase.PLAY or planner is None:
            return self._pimc.act(obs, legal)
        if phase is Phase.DASH:
            action = planner.dash(obs, legal)
        elif phase is Phase.AUCTION:
            action = planner.bid(obs, legal)
        elif phase is Phase.ESTIMATION:
            action = planner.estimate(obs, legal)
        else:
            raise ValueError("there is no decision to make in a finished round")
        if action not in legal:
            raise RuntimeError(f"SearchBot chose illegal action {action} in {phase.name}")
        return action
