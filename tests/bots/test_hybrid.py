"""The hybrid bot (SPEC §6 M5): PIMC deal sampling with BiddingNet deal weights and CardPlayNet
rollouts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from estimation.bots import HeuristicBot, HybridBot, HybridConfig, make_bot
from estimation.bots.hybrid import BOTH, GUESSES
from estimation.bots.hybrid.beliefs import (
    PublicBidding,
    bidding_history,
    log_likelihoods,
)
from estimation.bots.hybrid.bot import _merge_equivalent
from estimation.bots.hybrid.rollout import (
    SHAPE_SIZE,
    Start,
    play_out,
    seat_templates,
    shape_features,
    shapes_of,
)
from estimation.bots.knowledge import read_knowledge
from estimation.bots.rl import RLBot, policy_at
from estimation.bots.scoring import RoundScorer, standings_weights
from estimation.engine import (
    DASH_CALL,
    FIXED_ORDER,
    Bola,
    Phase,
    Round,
    RulesConfig,
    deal,
    play_action,
)
from estimation.engine.actions import ESTIMATE_BASE, FIRST_BID, PLAY_BASE
from estimation.engine.random_play import play_random_round
from estimation.rl.encoder import encode
from tests.rl.helpers import make_checkpoint


def test_public_bidding_reproduces_every_non_card_observation_and_legal_list():
    """``PublicBidding`` rebuilds the exact legal actions and empty-hand Observation of every
    non-card step from ``obs.log`` alone, without touching a Round or any seat's cards."""
    configs = (
        RulesConfig(),
        RulesConfig(super_call_min=9, super_caller_may_bid_fixed=True),
        RulesConfig(with_any_call=False),
        RulesConfig(with_choose=False),
        RulesConfig(with_any_call=False, with_choose=False),
    )
    checked = 0
    saw_super = saw_with = saw_dash = False
    for rules in configs:
        for seed in range(60):
            fixed = FIXED_ORDER[seed % 5] if seed >= 30 else None
            round_no = 14 + seed % 5 if fixed is not None else (seed % 13) + 1
            carry = 2 if seed % 7 == 0 else 1
            scores = (seed * 3, -seed, 10, 0)
            hands = deal(seed, round_no)
            finished = Round.new(round_no, hands, fixed, carry, scores, rules)
            if play_random_round(finished, seed).skipped:
                continue
            rnd = Round.new(round_no, hands, fixed, carry, scores, rules)
            steps: list[tuple[int, int, list[int], object]] = []
            for seat, action in finished.log:
                if rnd.phase is Phase.PLAY:
                    break
                steps.append(
                    (seat, action, rnd.legal_actions(), replace(rnd.observation(seat), hand=0))
                )
                rnd.step(action)
            obs = rnd.observation(0)
            state = PublicBidding(obs, rules)
            for seat, action, expected_legal, expected_obs in steps:
                assert state.legal_actions() == expected_legal
                assert state.observation(seat) == expected_obs
                state.step(seat, action)
                checked += 1
            saw_super = saw_super or obs.super_call
            saw_with = saw_with or bool(obs.with_players)
            saw_dash = saw_dash or bool(obs.dash_callers)
    assert checked > 400
    assert saw_super
    assert saw_with
    assert saw_dash


@pytest.mark.parametrize(
    "rules", [RulesConfig(), RulesConfig(with_any_call=False, with_choose=False)]
)
def test_public_bidding_follows_L12_an_earlier_call_in_the_winning_trump(rules: RulesConfig):
    """Yosif's round (SPEC L12): seat 3 bids 5 Sans, later 6 Hearts, and Sans wins. Under the
    default rules it is With and chooses first, and the mirror must agree step by step."""
    sans, hearts = FIRST_BID + 5 + 4, FIRST_BID + 10 + 2  # 5 Sans, 6 Hearts
    auction = [(1, 0), (2, 0), (3, 0), (0, 0), (1, 53), (2, 53), (3, sans), (0, 52)]
    auction += [(3, hearts), (0, FIRST_BID + 10 + 4), (3, 53), (0, 53)]
    rnd = Round.new(10, deal(5, 10), config=rules)
    for seat, action in auction:
        assert rnd.current_player == seat
        rnd.step(action)
    while rnd.phase is Phase.ESTIMATION:
        rnd.step(rnd.legal_actions()[-1])
    log = list(rnd.log)
    replayed = Round.new(10, deal(5, 10), config=rules)
    state = PublicBidding(replayed.observation(0), rules)
    for seat, action in log:
        assert state.legal_actions() == replayed.legal_actions()
        assert state.observation(seat) == replace(replayed.observation(seat), hand=0)
        state.step(seat, action)
        replayed.step(action)
    order = [seat for seat, action in log[len(auction) :]]
    assert order == ([3, 1, 2] if rules.with_any_call else [1, 2, 3])


def test_bidding_history_matches_encoded_engine_observations():
    for seed in range(20):
        fixed = 2 if seed >= 12 else None
        round_no = 16 if fixed is not None else (seed % 13) + 1
        hands = deal(seed + 100, round_no)
        finished = Round.new(round_no, hands, fixed)
        if play_random_round(finished, seed + 100).skipped:
            continue
        rnd = Round.new(round_no, hands, fixed)
        expected_seats: list[int] = []
        expected_rows = []
        expected_chosen: list[int] = []
        me = seed % 4
        for seat, action in finished.log:
            if rnd.phase is Phase.PLAY:
                break
            legal = rnd.legal_actions()
            if seat != me and len(legal) > 1:
                expected_seats.append(seat)
                expected_rows.append(encode(replace(rnd.observation(seat), hand=0)))
                expected_chosen.append(action)
            rnd.step(action)
        obs = rnd.observation(me)
        history = bidding_history(obs, RulesConfig())
        assert list(history.seats) == expected_seats
        assert history.chosen.tolist() == expected_chosen
        assert np.allclose(history.templates, np.stack(expected_rows))


def test_log_likelihoods_match_direct_per_decision_bidding_net_evaluation(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=7)
    policy = policy_at(str(path))
    rules = RulesConfig()
    hands = deal(12, 1)
    finished = Round.new(1, hands)
    while play_random_round(finished, 12).skipped:
        finished = Round.new(1, hands)
    rnd = Round.new(1, hands)
    decisions: list[tuple[int, int, list[int], Round]] = []
    me = 0
    for seat, action in finished.log:
        if rnd.phase is Phase.PLAY:
            break
        legal = rnd.legal_actions()
        if seat != me and len(legal) > 1:
            decisions.append((seat, action, legal, rnd.clone()))
        rnd.step(action)
    obs = rnd.observation(me)
    history = bidding_history(obs, rules)
    deals = [list(deal(100 + d, 1)) for d in range(4)]
    dealt = np.array(deals, dtype=np.uint64)
    shapes = np.zeros((len(deals), 4, SHAPE_SIZE), np.float32)
    for d, four in enumerate(deals):
        for s in range(4):
            shapes[d, s] = shape_features(four[s])
    temp, floor = 0.15, 0.1
    got = log_likelihoods(history, dealt, shapes, policy.bid_values, temp, floor)
    for d, four in enumerate(deals):
        expected = 0.0
        for seat, action, legal, snap in decisions:
            view = replace(
                snap.observation(seat), hand=four[seat], void_announced=obs.void_announced
            )
            vals = policy.bid_values(encode(view, rules)[np.newaxis])[0].astype(np.float64) / temp
            legal_vals = np.array([vals[a] for a in legal])
            probs = np.exp(legal_vals - legal_vals.max())
            probs /= probs.sum()
            picked = float(probs[legal.index(action)])
            prob = (1.0 - floor) * picked + floor / len(legal)
            expected += float(np.log(prob))
        assert got[d] == pytest.approx(expected, abs=1e-5)


def test_vectorized_rollout_matches_four_rl_bots_in_the_engine(tmp_path: Path):
    """Playing out a batch of deals with ``play_out`` gives the exact same tricks as four
    ``RLBot``s stepping the engine's ``Round``."""
    path = make_checkpoint(tmp_path / "net.pt", seed=3)
    policy = policy_at(str(path))
    rules = RulesConfig(super_call_min=9)
    rl_bots = [RLBot(s, make_bot(f"rl:checkpoint={path}", s, rules).config) for s in range(4)]  # type: ignore[attr-defined]
    heuristic = HeuristicBot()

    # Collect mid-round PLAY positions at several trick lengths (leading and following).
    rounds: list[Round] = []
    for seed in range(6):
        bola = Bola(seed + 40, rules)
        steps_in_play = 0
        target = (seed * 7) % 25
        while not bola.is_over:
            rnd = bola.round
            seat = bola.current_player
            assert seat is not None
            if rnd.phase is Phase.PLAY:
                if steps_in_play == target:
                    rounds.append(rnd.clone())
                    break
                steps_in_play += 1
            bola.step(heuristic.act(bola.observation(seat), bola.legal_actions()))
    assert len(rounds) == 6

    for rnd in rounds:
        seat = rnd.current_player
        assert seat is not None
        obs = rnd.observation(seat)
        know = read_knowledge(obs)
        legal_cards = [a - PLAY_BASE for a in rnd.legal_actions()]
        deals = [[rnd.observation(s).hand for s in range(4)]]
        n_cards = len(legal_cards)
        hands = np.repeat(np.array(deals, dtype=np.uint64), n_cards, axis=0)
        played = np.tile(np.array(know.played_by, dtype=np.uint64), (n_cards, 1))
        voids = np.tile(np.array(know.voids, dtype=np.uint8), (n_cards, 1))
        shapes = np.zeros((1, 4, SHAPE_SIZE), np.float32)
        for s in range(4):
            shapes[0, s] = shape_features(deals[0][s] | know.played_by[s])
        shapes = np.repeat(shapes, n_cards, axis=0)
        first = np.array(legal_cards, dtype=np.int64)
        assert obs.trump is not None
        assert obs.leader is not None
        est = tuple(int(e or 0) for e in obs.estimates)
        start = Start(
            trick=[card for _, card in obs.trick],
            leader=obs.leader,
            won=obs.tricks_won,
            tricks_done=len(obs.tricks),
            estimates=est,
            trump=obs.trump,
        )
        templates = seat_templates(obs, rules)
        got = play_out(policy.card_values, templates, start, hands, played, voids, shapes, first)

        for k, card in enumerate(legal_cards):
            sim = rnd.clone()
            sim.step(play_action(card))
            while sim.phase is Phase.PLAY:
                cur = sim.current_player
                assert cur is not None
                sim.step(rl_bots[cur].act(sim.observation(cur), sim.legal_actions()))
            assert got[k].tolist() == list(sim.observation(0).tricks_won)


def test_single_deal_evaluation_matches_round_scorer_on_engine_playout(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=5)
    rules = RulesConfig()
    bot = HybridBot(
        0, HybridConfig(checkpoint=str(path), samples=1, oversample=1, beliefs=False, rules=rules)
    )
    rl = RLBot(0, make_bot(f"rl:checkpoint={path}", 0, rules).config)  # type: ignore[attr-defined]
    bola = Bola(19, rules)
    while bola.round.phase is not Phase.PLAY:
        seat = bola.current_player or 0
        bola.step(bot.act(bola.observation(seat), bola.legal_actions()))
    rnd = bola.round
    seat = rnd.current_player or 0
    obs = rnd.observation(seat)
    cards = [a - PLAY_BASE for a in rnd.legal_actions()]
    real_deal = [[rnd.observation(s).hand for s in range(4)]]
    totals = bot.evaluate(obs, cards, real_deal)
    scorer = RoundScorer(obs, rules, standings_weights(obs.scores, seat))
    for k, card in enumerate(cards):
        sim = rnd.clone()
        sim.step(play_action(card))
        while sim.phase is Phase.PLAY:
            cur = sim.current_player or 0
            sim.step(rl.act(sim.observation(cur), sim.legal_actions()))
        assert totals[k] == pytest.approx(scorer.value(sim.observation(0).tricks_won))


def test_bidding_decisions_match_the_rl_bot(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=9, favourite_action=DASH_CALL)
    hybrid = make_bot(f"hybrid:checkpoint={path}:samples=2:oversample=1", 0)
    rl = make_bot(f"rl:checkpoint={path}", 0)
    bola = Bola(3)
    checked = 0
    while not bola.is_over and checked < 40:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        if obs.phase is not Phase.PLAY:
            assert hybrid.act(obs, legal) == rl.act(obs, legal)
            checked += 1
        bola.step(rl.act(obs, legal))
    assert checked >= 10


def test_hybrid_search_is_deterministic(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=2)
    spec = f"hybrid:checkpoint={path}:samples=3:oversample=2:top_k=3"
    actions_a: list[int] = []
    actions_b: list[int] = []
    for out in (actions_a, actions_b):
        bot = make_bot(spec, 4)
        other = HeuristicBot()
        bola = Bola(8)
        while not bola.is_over and bola.round.round_no <= 2:
            seat = bola.current_player or 0
            obs = bola.observation(seat)
            legal = bola.legal_actions()
            action = bot.act(obs, legal) if seat == 0 else other.act(obs, legal)
            if seat == 0:
                out.append(action)
            bola.step(action)
    assert len(actions_a) >= 26
    assert actions_a == actions_b


def test_invalid_hybrid_configs_are_rejected(tmp_path: Path):
    card_only = make_checkpoint(tmp_path / "card.pt", bidding=False)
    with pytest.raises(ValueError, match="no trained BiddingNet"):
        HybridBot(0, HybridConfig(checkpoint=str(card_only)))
    valid = str(make_checkpoint(tmp_path / "ok.pt"))
    for bad in (
        HybridConfig(checkpoint=valid, samples=0),
        HybridConfig(checkpoint=valid, oversample=0),
        HybridConfig(checkpoint=valid, belief_temperature=0.0),
        HybridConfig(checkpoint=valid, belief_floor=0.0),
        HybridConfig(checkpoint=valid, belief_floor=1.5),
        HybridConfig(checkpoint=valid, top_k=-1),
    ):
        with pytest.raises(ValueError, match=r"hybrid|top_k"):
            HybridBot(0, bad)
    assert isinstance(
        HybridBot(0, HybridConfig(checkpoint=valid, samples=2, oversample=1)), HybridBot
    )
    assert FIRST_BID < ESTIMATE_BASE


def test_dealing_by_guesses_needs_a_guess_net_and_a_valid_setting(tmp_path: Path):
    plain = str(make_checkpoint(tmp_path / "plain.pt"))
    with pytest.raises(ValueError, match="no GuessNet"):
        HybridBot(0, HybridConfig(checkpoint=plain, deals=GUESSES))
    guessing = str(make_checkpoint(tmp_path / "guess.pt", guess=True))
    for bad in (
        HybridConfig(checkpoint=guessing, deals="random"),
        HybridConfig(checkpoint=guessing, deals=GUESSES, guess_temperature=0.0),
    ):
        with pytest.raises(ValueError, match=r"deals|guess_temperature"):
            HybridBot(0, bad)


def test_guess_chances_are_the_guess_nets_from_the_bots_own_view(tmp_path: Path):
    path = make_checkpoint(tmp_path / "guess.pt", guess=True)
    bot = HybridBot(0, HybridConfig(checkpoint=str(path), deals=GUESSES))
    bola = Bola(5)
    while bola.phase is not Phase.PLAY:
        bola.step(
            HeuristicBot().act(bola.observation(bola.current_player or 0), bola.legal_actions())
        )
    obs = bola.observation(bola.current_player or 0)
    expected = policy_at(str(path)).card_guesses(encode(obs)[np.newaxis])[0]
    assert np.allclose(bot.guess_chances(obs), expected)
    sharp = HybridBot(0, HybridConfig(checkpoint=str(path), deals=GUESSES, guess_temperature=0.5))
    assert np.allclose(sharp.guess_chances(obs), expected**2)


@pytest.mark.parametrize("deals", [GUESSES, BOTH])
def test_a_hybrid_dealing_by_guesses_plays_legal_seeded_games(tmp_path: Path, deals: str):
    path = make_checkpoint(tmp_path / "guess.pt", seed=3, guess=True)
    spec = f"hybrid:checkpoint={path}:samples=3:oversample=2:deals={deals}"
    runs: list[list[int]] = [[], []]
    for out in runs:
        bot = make_bot(spec, 4)
        other = HeuristicBot()
        bola = Bola(8)
        while not bola.is_over and bola.round.round_no <= 2:
            seat = bola.current_player or 0
            obs = bola.observation(seat)
            legal = bola.legal_actions()
            action = bot.act(obs, legal) if seat == 0 else other.act(obs, legal)
            assert action in legal
            if seat == 0:
                out.append(action)
            bola.step(action)
    assert len(runs[0]) >= 26
    assert runs[0] == runs[1]


def test_shapes_of_and_merge_equivalent():
    hands = np.array([list(deal(seed, 1)) for seed in range(8)], dtype=np.uint64)
    got = shapes_of(hands)
    for d in range(8):
        for s in range(4):
            assert np.allclose(got[d, s], shape_features(int(hands[d, s])), atol=1e-6)
    # Cards 2, 3, 4 of clubs touch when card 3 is not in ``out``; card 3 has the highest advantage.
    cards, advs = _merge_equivalent([2, 3, 4, 15], [0.1, 0.5, 0.2, -0.3], out=1 << 8)
    assert cards == [3, 15]
    assert advs == [0.5, -0.3]
