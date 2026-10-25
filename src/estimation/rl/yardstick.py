"""The yardstick: a fixed duplicate match that rates the nets the same way every time, so a long
training run shows whether it is still improving.

Every ``yardstick_every`` iterations the trainer plays ``yardstick_sets`` duplicate sets (SPEC §6
M2) of the latest checkpoint against PIMC with ``yardstick_pimc_samples`` deals, lineup A,B,A,B,
and, when ``yardstick_reference`` names one, against that checkpoint the same way. The deals and
every bot's seed come from ``yardstick_seed`` alone, so every yardstick of every run plays the
same games apart from the trained bot's own choices. Training never sees these games.

A set is won when the bot's mean RSA over the set is above zero (with two bots in an A,B,A,B
lineup, one's RSA is minus the other's) and counts half when it is exactly zero.
"""

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.arena.duplicate import SetResult, play_set
from estimation.arena.ratings import Interval, mean_ci, set_first_places, set_rsa
from estimation.engine import RulesConfig

from .config import SelfPlayConfig

PIMC_MATCH = "pimc"
REFERENCE_MATCH = "reference"


@dataclass(frozen=True, slots=True)
class Match:
    """One yardstick match: ``bot`` against ``opponent``, two seats each."""

    name: str
    bot: str
    opponent: str

    @property
    def lineup(self) -> tuple[str, str, str, str]:
        return (self.bot, self.opponent, self.bot, self.opponent)


def rl_spec(checkpoint: Path | str) -> str:
    return f"rl:checkpoint={checkpoint}"


def matches(cfg: SelfPlayConfig, checkpoint: Path) -> list[Match]:
    """The yardstick's matches for the nets saved at ``checkpoint``."""
    bot = rl_spec(checkpoint)
    found = [Match(PIMC_MATCH, bot, f"pimc:samples={cfg.yardstick_pimc_samples}")]
    if cfg.yardstick_reference:
        found.append(Match(REFERENCE_MATCH, bot, rl_spec(cfg.yardstick_reference)))
    return found


def play(
    pool: Executor, found: Sequence[Match], cfg: SelfPlayConfig, rules: RulesConfig
) -> dict[str, dict[str, Any]]:
    """Play every match's sets on ``pool`` and summarize each, by match name."""
    futures = {
        match.name: [
            pool.submit(_play_set, match.lineup, cfg.yardstick_seed, rules, index)
            for index in range(cfg.yardstick_sets)
        ]
        for match in found
    }
    return {
        match.name: summarize([future.result() for future in futures[match.name]], match)
        for match in found
    }


def summarize(results: Sequence[SetResult], match: Match) -> dict[str, Any]:
    """Sets won, RSA per Bola and share of first places of ``match.bot``, with 95% CIs."""
    rsa = [set_rsa(result, match.lineup)[match.bot] for result in results]
    won = [1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in rsa]
    first = [set_first_places(result, match.lineup)[match.bot] for result in results]
    return {
        "opponent": match.opponent,
        "sets": len(results),
        "won": _interval(mean_ci(won)),
        "rsa": _interval(mean_ci(rsa)),
        "first": _interval(mean_ci(first)),
    }


def _interval(interval: Interval) -> dict[str, float]:
    return {"mean": interval.mean, "low": interval.low, "high": interval.high}


def _play_set(lineup: tuple[str, ...], seed: int, rules: RulesConfig, index: int) -> SetResult:
    return play_set(lineup, seed, index, rules)[0]
