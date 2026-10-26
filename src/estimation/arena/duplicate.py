"""The duplicate runner (SPEC §6 M2).

A lineup is 4 bots, one per slot, repeats allowed. A duplicate set plays one seed's deals 4 times,
rotating the lineup so that in rotation ``k`` seat ``i`` gets the bot in slot ``(i + k) % 4``.
Every bot plays every hand, so deal luck cancels out. Sets are independent and seeded by the run
seed and their index, so worker processes give exactly the results of a serial run.
"""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np

from estimation.bots import Bot, make_bot
from estimation.engine import RulesConfig
from estimation.engine.config import DEFAULT_CONFIG

from .match import play_bola
from .timing import DEFAULT_BUDGET, Timing

ROTATIONS = 4

BotMaker = Callable[[str, int, RulesConfig], Bot]  # bot name, seed, the rules being played
Progress = Callable[[int, int, float], None]  # sets done, sets in all, seconds elapsed


def slot_at(seat: int, rotation: int) -> int:
    """The lineup slot playing ``seat`` in ``rotation``."""
    return (seat + rotation) % 4


def deal_seed(seed: int, index: int) -> int:
    """The Bola seed of duplicate set ``index`` in a run seeded ``seed``."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def bot_seed(seed: int, index: int, rotation: int, seat: int) -> int:
    """The seed of the bot at ``seat`` in one rotation of one set."""
    return int(np.random.SeedSequence([seed, index, rotation, seat, 1]).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class SetResult:
    """One duplicate set: its deal seed, and each rotation's game scores by seat and rounds."""

    index: int
    deal_seed: int
    scores: tuple[tuple[int, int, int, int], ...]
    rounds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArenaRun:
    """Every set of a run, in index order, with card-decision timings per bot."""

    lineup: tuple[str, ...]
    seed: int
    config: RulesConfig
    results: tuple[SetResult, ...]
    timings: dict[str, Timing]
    wall_seconds: float
    workers: int


def play_set(
    lineup: Sequence[str],
    seed: int,
    index: int,
    config: RulesConfig = DEFAULT_CONFIG,
    make: BotMaker = make_bot,
    budget: float = DEFAULT_BUDGET,
) -> tuple[SetResult, dict[str, Timing]]:
    """Play duplicate set ``index``: its deals once per rotation of the lineup."""
    if len(lineup) != 4:
        raise ValueError("a lineup has exactly 4 bots")
    seed_of_deals = deal_seed(seed, index)
    timings = {name: Timing(budget) for name in dict.fromkeys(lineup)}
    scores: list[tuple[int, int, int, int]] = []
    rounds: list[int] = []
    for rotation in range(ROTATIONS):
        names = [lineup[slot_at(seat, rotation)] for seat in range(4)]
        bots = [
            make(names[seat], bot_seed(seed, index, rotation, seat), config) for seat in range(4)
        ]
        record = play_bola(seed_of_deals, bots, config, [timings[name] for name in names])
        scores.append(record.scores)
        rounds.append(record.rounds)
    return SetResult(index, seed_of_deals, tuple(scores), tuple(rounds)), timings


def run_arena(
    lineup: Sequence[str],
    sets: int,
    seed: int,
    config: RulesConfig = DEFAULT_CONFIG,
    workers: int = 1,
    budget: float = DEFAULT_BUDGET,
    progress: Progress | None = None,
    on_set: Callable[[SetResult], None] | None = None,
) -> ArenaRun:
    """Play ``sets`` duplicate sets, on ``workers`` processes. Results come back in set order;
    ``on_set`` sees each one as it arrives."""
    names = tuple(lineup)
    if len(names) != 4:
        raise ValueError("a lineup has exactly 4 bots")
    if sets < 1 or workers < 1:
        raise ValueError("a run needs at least one set and one worker")
    for name in dict.fromkeys(names):
        make_bot(name, 0, config)  # fail fast on an unknown bot or a bad option
    task = partial(_play_set_task, names, seed, config, budget)
    timings = {name: Timing(budget) for name in dict.fromkeys(names)}
    results: list[SetResult] = []
    start = time.perf_counter()

    def collect(outcome: tuple[SetResult, dict[str, Timing]]) -> None:
        result, set_timings = outcome
        results.append(result)
        for name, timing in set_timings.items():
            timings[name].merge(timing)
        if on_set is not None:
            on_set(result)
        if progress is not None:
            progress(len(results), sets, time.perf_counter() - start)

    if workers == 1:
        for index in range(sets):
            collect(task(index))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(task, range(sets)):
                collect(outcome)
    return ArenaRun(
        lineup=names,
        seed=seed,
        config=config,
        results=tuple(results),
        timings=timings,
        wall_seconds=time.perf_counter() - start,
        workers=workers,
    )


def _play_set_task(
    lineup: tuple[str, ...], seed: int, config: RulesConfig, budget: float, index: int
) -> tuple[SetResult, dict[str, Timing]]:
    return play_set(lineup, seed, index, config, budget=budget)
