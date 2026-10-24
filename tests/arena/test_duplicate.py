"""The duplicate runner (SPEC §6 M2): one seed's deals played 4 times, the lineup rotated so that
in rotation k seat i gets bot (i + k) % 4, so every bot plays every hand."""

from typing import ClassVar

from estimation.arena.duplicate import (
    ROTATIONS,
    SetResult,
    bot_seed,
    deal_seed,
    play_set,
    run_arena,
    slot_at,
)
from estimation.arena.match import play_bola
from estimation.bots import Bot, make_bot
from estimation.engine import Bola, Observation, RulesConfig

LINEUP = ("heuristic", "random", "heuristic", "random")


class Spy:
    """Plays like the bot it wraps and records the seats it acted from."""

    log: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, name: str, inner: Bot) -> None:
        self.name = name
        self.inner = inner

    def act(self, obs: Observation, legal: list[int]) -> int:
        Spy.log.append((self.name, obs.seat))
        return self.inner.act(obs, legal)


def spy_maker(spec: str, seed: int) -> Bot:
    name, _, inner = spec.partition("=")
    return Spy(name, make_bot(inner, seed))


def test_rotation_k_seats_slot_i_plus_k_at_seat_i():
    for k in range(ROTATIONS):
        assert [slot_at(seat, k) for seat in range(4)] == [(seat + k) % 4 for seat in range(4)]


def test_every_lineup_slot_plays_every_seat_once_per_set():
    lineup = ("a=heuristic", "b=random", "c=random", "d=heuristic")
    Spy.log = []
    play_set(lineup, seed=0, index=0, config=RulesConfig(), make=spy_maker)
    seats = {name: {seat for who, seat in Spy.log if who == name} for name in "abcd"}
    assert seats == {name: {0, 1, 2, 3} for name in "abcd"}


def test_each_rotation_replays_the_same_deals_with_the_rotated_lineup():
    result, _ = play_set(LINEUP, seed=4, index=2, config=RulesConfig())
    seed = deal_seed(4, 2)
    assert result.deal_seed == seed
    for k in range(ROTATIONS):
        bots = [make_bot(LINEUP[slot_at(i, k)], bot_seed(4, 2, k, i)) for i in range(4)]
        record = play_bola(seed, bots)
        assert result.scores[k] == record.scores
        assert result.rounds[k] == record.rounds


def test_sets_use_different_deals():
    seeds = {deal_seed(0, index) for index in range(100)} | {
        deal_seed(1, index) for index in range(100)
    }
    assert len(seeds) == 200


def test_the_same_seed_gives_the_same_results():
    first = run_arena(LINEUP, sets=3, seed=7)
    second = run_arena(LINEUP, sets=3, seed=7)
    assert first.results == second.results
    assert [r.index for r in first.results] == [0, 1, 2]
    assert first.results != run_arena(LINEUP, sets=3, seed=8).results


def test_worker_processes_give_the_same_results():
    serial = run_arena(LINEUP, sets=4, seed=3, workers=1)
    parallel = run_arena(LINEUP, sets=4, seed=3, workers=2)
    assert parallel.results == serial.results


def test_card_decisions_are_timed_per_bot():
    run = run_arena(("heuristic", "heuristic", "heuristic", "random"), sets=1, seed=0)
    assert set(run.timings) == {"heuristic", "random"}
    total = sum(run.timings[name].count for name in run.timings)
    assert total == 52 * sum(run.results[0].rounds) - 52 * skipped_rounds(run.results[0])


def skipped_rounds(result: SetResult) -> int:
    """Skipped rounds have no card play. Recount them by replaying the set's games."""
    skipped = 0
    for k in range(ROTATIONS):
        lineup = ("heuristic", "heuristic", "heuristic", "random")
        bots = [make_bot(lineup[slot_at(i, k)], bot_seed(0, 0, k, i)) for i in range(4)]
        bola = Bola(result.deal_seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
        skipped += sum(1 for r in bola.history if r.skipped)
    return skipped
