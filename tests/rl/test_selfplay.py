"""Training games (SPEC §6 M4): the opponent mix and the lockstep actor."""

import dataclasses
from pathlib import Path

import pytest
import torch

from estimation.arena.match import play_bola
from estimation.bots import HeuristicBot, make_bot
from estimation.engine import RulesConfig
from estimation.rl.config import SELFPLAY_STAGE, config_from_dict
from estimation.rl.networks import BiddingNet, CardPlayNet, save_checkpoint, state_dict_of
from estimation.rl.selfplay import (
    HEURISTIC,
    LEARNER,
    PIMC,
    SNAPSHOT,
    ChunkTask,
    GamePlan,
    NetWeights,
    SeatPlan,
    SeedFn,
    plan_games,
    play_chunk,
)
from estimation.rl.standings import RewardConfig

CONFIG = config_from_dict({"seed": 2, "selfplay": {"games": 40}})


def tiny_weights(seed: int) -> NetWeights:
    torch.manual_seed(seed)
    return NetWeights(16, state_dict_of(CardPlayNet(16)), state_dict_of(BiddingNet(16)))


def test_each_iteration_has_exact_shares_of_each_opponent_kind():
    plans = plan_games(CONFIG, 0, [], SeedFn(2, SELFPLAY_STAGE, 1))
    kinds = [plan.kind for plan in plans]
    assert (kinds.count("self"), kinds.count("heuristic"), kinds.count("pimc")) == (28, 6, 6)
    assert kinds[:6] != ["self"] * 6  # shuffled, so every chunk gets a mix
    for plan in plans:
        seats = [s.kind for s in plan.seats]
        assert LEARNER in seats
        if plan.kind == "self":
            assert set(seats) == {LEARNER}  # no snapshots yet: all four seats learn
        elif plan.kind == "heuristic":
            assert sorted(seats) == [HEURISTIC] * 3 + [LEARNER]
            assert all(s.noise in CONFIG.opponents.heuristic_noise for s in plan.seats)
        else:
            assert sorted(seats) == [LEARNER] + [PIMC] * 3
            assert all(4 <= s.samples <= 8 for s in plan.seats if s.kind == PIMC)


def test_snapshots_join_self_play_games_once_there_are_some():
    plans = plan_games(CONFIG, 3, [5, 10], SeedFn(2, SELFPLAY_STAGE, 4))
    snapshot_seats = [
        s for plan in plans if plan.kind == "self" for s in plan.seats if s.kind == SNAPSHOT
    ]
    assert snapshot_seats
    assert {s.snapshot for s in snapshot_seats} <= {5, 10}
    assert all(LEARNER in [s.kind for s in plan.seats] for plan in plans)


def test_planning_is_seeded():
    one = plan_games(CONFIG, 0, [5], SeedFn(2, SELFPLAY_STAGE, 1))
    two = plan_games(CONFIG, 0, [5], SeedFn(2, SELFPLAY_STAGE, 1))
    other = plan_games(CONFIG, 0, [5], SeedFn(3, SELFPLAY_STAGE, 1))
    assert one == two
    assert one != other


def task(plans: list[GamePlan], weights: NetWeights, epsilon: float = 0.0) -> ChunkTask:
    return ChunkTask(
        plans=tuple(plans),
        learner=weights,
        snapshots={7: tiny_weights(99)},
        epsilon_bid=epsilon,
        epsilon_card=epsilon,
        record_bid=True,
        record_card=True,
        rules=RulesConfig(),
        reward=RewardConfig(),
    )


def test_a_greedy_learner_plays_exactly_like_the_arena_bot(tmp_path: Path):
    weights = tiny_weights(4)
    plan = GamePlan(
        index=0,
        kind="heuristic",
        deal_seed=31,
        seed=8,
        seats=(SeatPlan(HEURISTIC), SeatPlan(LEARNER), SeatPlan(HEURISTIC), SeatPlan(HEURISTIC)),
    )
    with torch.inference_mode():
        result = play_chunk(task([plan], weights))
    path = tmp_path / "net.pt"
    card, bidding = CardPlayNet(16), BiddingNet(16)
    card.load_state_dict(weights.card)
    assert weights.bidding is not None
    bidding.load_state_dict(weights.bidding)
    save_checkpoint(path, card, bidding, {})
    bots = [HeuristicBot(), make_bot(f"rl:checkpoint={path}", 0), HeuristicBot(), HeuristicBot()]
    record = play_bola(31, bots)
    (stats,) = result.games
    scores = record.scores
    assert stats.rsa == pytest.approx(scores[1] - (scores[0] + scores[2] + scores[3]) / 3)
    assert stats.rounds == record.rounds
    assert stats.learner_seats == 1
    assert len(result.card) > 150  # one seat's card decisions with a choice
    assert len(result.bid) > 20


def test_chunks_are_deterministic():
    weights = tiny_weights(5)
    learner = SeatPlan(LEARNER)
    seats = (learner, learner, learner, learner)
    plans = [
        GamePlan(index=i, kind="self", deal_seed=100 + i, seed=200 + i, seats=seats)
        for i in range(2)
    ]
    one = play_chunk(task(plans, weights, epsilon=0.1))
    two = play_chunk(task(plans, weights, epsilon=0.1))
    assert (one.card.features == two.card.features).all()
    assert (one.card.actions == two.card.actions).all()
    assert (one.bid.rewards == two.bid.rewards).all()
    assert one.games == two.games
    assert all(0 <= a < 52 for a in one.card.actions.tolist())
    assert all(0 <= a < 68 for a in one.bid.actions.tolist())


def test_snapshot_seats_play_but_are_not_recorded():
    # A snapshot with the learner's own weights plays exactly like it, so the two games below
    # follow the same course; only the learner seats' decisions may be recorded.
    weights = tiny_weights(6)
    learner, snapshot = SeatPlan(LEARNER), SeatPlan(SNAPSHOT, snapshot=7)
    seats_all = (learner, learner, learner, learner)
    seats_half = (learner, snapshot, learner, snapshot)
    plan = GamePlan(index=0, kind="self", deal_seed=41, seed=42, seats=seats_all)
    every = play_chunk(dataclasses.replace(task([plan], weights), snapshots={7: weights}))
    half = play_chunk(
        dataclasses.replace(
            task([dataclasses.replace(plan, seats=seats_half)], weights), snapshots={7: weights}
        )
    )
    assert every.games[0].rounds == half.games[0].rounds
    assert 0 < len(half.card) < len(every.card)
    rows = {row.tobytes() for row in every.card.features}
    assert all(row.tobytes() in rows for row in half.card.features)
    assert half.games[0].learner_seats == 2
