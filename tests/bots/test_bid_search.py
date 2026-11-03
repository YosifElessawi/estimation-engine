"""Searching the bidding (Hybrid plan step 4): the public bidding finishes rounds exactly like the
engine, many rounds with different contracts play out like RL bots, and the bid search's values
are the engine's."""

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from estimation.bots import Bot, HeuristicBot, HybridBot, HybridConfig, make_bot
from estimation.bots.hybrid.beliefs import PublicBidding
from estimation.bots.hybrid.bid_search import bid_totals
from estimation.bots.hybrid.rollout import play_rounds, seat_templates, shapes_of
from estimation.bots.rl import policy_at
from estimation.bots.scoring import RoundScorer, estimates_of, standings_weights
from estimation.engine import FIXED_ORDER, Bola, Phase, Round, RulesConfig, deal
from estimation.engine.actions import ESTIMATE_BASE, PLAY_BASE
from tests.rl.helpers import make_checkpoint

CONFIGS = (
    RulesConfig(),
    RulesConfig(with_any_call=False, with_choose=False),
    RulesConfig(super_call_min=9, super_caller_may_bid_fixed=False),
)


def new_round(seed: int, rules: RulesConfig) -> Round:
    fixed = FIXED_ORDER[seed % 5] if seed % 3 == 0 else None
    round_no = 14 + seed % 5 if fixed is not None else seed % 13 + 1
    carry = 2 if seed % 7 == 0 else 1
    return Round.new(round_no, deal(seed, round_no), fixed, carry, (seed, -seed, 0, 5), rules)


@pytest.mark.parametrize("rules", CONFIGS)
def test_public_bidding_finishes_every_round_like_the_engine(rules: RulesConfig):
    """Random bidding to its end: card play with the same Caller, contract, With players, total,
    Over or Under, Risks and last estimator (FIX-3 in fixed-trump rounds), or a skipped round."""
    rng = random.Random(1)
    seen: dict[str, int] = {"play": 0, "skipped": 0, "fixed": 0, "super": 0, "nobody": 0}
    for seed in range(400):
        rnd = new_round(seed, rules)
        state = PublicBidding(replace(rnd.observation(0), hand=0), rules)
        while rnd.phase is not Phase.PLAY and rnd.phase is not Phase.DONE:
            seat = rnd.current_player
            assert seat is not None
            assert state.bidding
            assert state.legal_actions() == rnd.legal_actions()
            legal = rnd.legal_actions()
            estimates = [a for a in legal if ESTIMATE_BASE <= a < PLAY_BASE]
            # Random play super-calls in most fixed-trump rounds; estimate mostly instead, so
            # rounds settled by FIX-3 come up too.
            action = rng.choice(estimates if estimates and rng.random() < 0.9 else legal)
            rnd.step(action)
            state.step(seat, action)
        assert state.phase is rnd.phase
        assert not state.bidding
        if rnd.phase is Phase.DONE:
            assert state.skipped
            assert rnd.result().skipped
            seen["skipped"] += 1
            continue
        for seat in range(4):
            assert state.observation(seat) == replace(rnd.observation(seat), hand=0)
        obs = rnd.observation(0)
        seen["play"] += 1
        seen["fixed"] += obs.fixed_trump is not None and not obs.super_call
        seen["super"] += obs.super_call
        seen["nobody"] += obs.last_estimator is None
    assert seen["play"] > 200
    assert seen["skipped"] > 5
    assert seen["fixed"] > 20
    assert seen["super"] > 5


def test_a_clone_steps_on_its_own():
    rnd = new_round(4, RulesConfig())
    state = PublicBidding(replace(rnd.observation(0), hand=0), RulesConfig())
    twin = state.clone()
    seat = state.cur
    action = state.legal_actions()[-1]
    twin.step(seat, action)
    assert state.log == []
    assert twin.log == [(seat, action)]
    assert state.observation(0) != twin.observation(0)


def bid_with(bots: list[Bot], rnd: Round) -> None:
    while rnd.phase is not Phase.PLAY and rnd.phase is not Phase.DONE:
        seat = rnd.current_player
        assert seat is not None
        rnd.step(bots[seat].act(rnd.observation(seat), rnd.legal_actions()))


def test_rounds_with_different_contracts_play_out_like_rl_bots_in_the_engine(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=7)
    rules = RulesConfig()
    bots = [make_bot(f"rl:checkpoint={path}", s, rules) for s in range(4)]
    starts: list[Round] = []
    for seed in range(40):
        rnd = new_round(seed, rules)
        bid_with([HeuristicBot()] * 4, rnd)
        if rnd.phase is Phase.PLAY:
            starts.append(rnd)
    assert len({(r.observation(0).caller, r.observation(0).trump) for r in starts}) > 5
    views = [rnd.observation(0) for rnd in starts]
    dealt = np.array([list(rnd.hands) for rnd in starts], dtype=np.uint64)
    won = play_rounds(
        policy_at(str(path)).card_values,
        np.stack([seat_templates(view, rules) for view in views]),
        np.array([view.leader for view in views], np.int64),
        np.array([view.trump for view in views], np.int64),
        np.array([estimates_of(view) for view in views], np.int64),
        dealt.copy(),
        shapes_of(dealt),
    )
    for rnd, tricks in zip(starts, won.tolist(), strict=True):
        while rnd.phase is Phase.PLAY:
            seat = rnd.current_player
            assert seat is not None
            rnd.step(bots[seat].act(rnd.observation(seat), rnd.legal_actions()))
        assert tricks == [seat.tricks for seat in rnd.result().seats]


def test_the_bid_searchs_values_are_the_engines(tmp_path: Path):
    """On the real deal, each option's value is what the engine gives when the option is made,
    RL bots finish the bidding and play the round, scored from the searching seat's side."""
    path = make_checkpoint(tmp_path / "net.pt", seed=8)
    rules = RulesConfig()
    policy = policy_at(str(path))
    bots = [make_bot(f"rl:checkpoint={path}", s, rules) for s in range(4)]
    compared = skipped = 0
    for seed in range(30):
        rnd = new_round(seed, rules)
        for _ in range(seed % 4):  # start some searches later in the bidding
            if rnd.phase is Phase.PLAY or rnd.phase is Phase.DONE:
                break
            seat = rnd.current_player
            assert seat is not None
            rnd.step(bots[seat].act(rnd.observation(seat), rnd.legal_actions()))
        if rnd.phase is Phase.PLAY or rnd.phase is Phase.DONE:
            continue
        me = rnd.current_player
        assert me is not None
        obs = rnd.observation(me)
        options = rnd.legal_actions()[:3]
        weights = standings_weights(obs.scores, me)
        totals = bid_totals(
            obs, rules, options, [list(rnd.hands)], policy.bid_values, policy.card_values, weights
        )
        for option, total in zip(options, totals, strict=True):
            twin = rnd.clone()
            twin.step(option)
            bid_with(bots, twin)
            if twin.phase is Phase.DONE:
                assert total == 0.0
                skipped += 1
                continue
            scorer = RoundScorer(twin.observation(me), rules, weights)
            while twin.phase is Phase.PLAY:
                seat = twin.current_player
                assert seat is not None
                twin.step(bots[seat].act(twin.observation(seat), twin.legal_actions()))
            tricks = [s.tricks for s in twin.result().seats]
            assert total == pytest.approx(scorer.value(tricks))
            compared += 1
    assert compared > 40


def play_two_rounds(bot: Bot, seed: int) -> list[int]:
    other = HeuristicBot()
    bola = Bola(seed)
    actions: list[int] = []
    while not bola.is_over and bola.round.round_no <= 2:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        action = bot.act(obs, legal) if seat == 0 else other.act(obs, legal)
        assert action in legal
        if seat == 0:
            actions.append(action)
        bola.step(action)
    return actions


def test_a_hybrid_searching_its_bids_plays_seeded_games(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=3)
    spec = f"hybrid:checkpoint={path}:samples=3:bid_search=true:bid_samples=3"
    first = play_two_rounds(make_bot(spec, 4), 8)
    assert first == play_two_rounds(make_bot(spec, 4), 8)
    bot = make_bot(spec, 4)
    assert isinstance(bot, HybridBot)
    play_two_rounds(bot, 8)
    search = bot.last_bid_search
    assert search is not None
    assert 1 <= len(search.actions) <= 3
    assert search.deals in (0, 3)


def test_searching_one_bid_is_bidding_by_the_net(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=5)
    plain = make_bot(f"hybrid:checkpoint={path}:samples=3", 4)
    one = make_bot(f"hybrid:checkpoint={path}:samples=3:bid_search=true:bid_top_k=1", 4)
    assert play_two_rounds(plain, 9) == play_two_rounds(one, 9)


def test_bad_bid_search_settings_are_refused(tmp_path: Path):
    valid = str(make_checkpoint(tmp_path / "ok.pt"))
    for bad in (
        HybridConfig(checkpoint=valid, bid_top_k=0),
        HybridConfig(checkpoint=valid, bid_samples=0),
    ):
        with pytest.raises(ValueError, match="bid search"):
            HybridBot(0, bad)
