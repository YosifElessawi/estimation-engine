"""Step 5's search bots: the one net inside the Hybrid search (Essam Gouda's bitter-search tests,
from estimation-arena tests/test_bitter_search.py, MIT licence), and the all-in-one bot that also
imagines the deals and judges the cards itself."""

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.arena.duplicate import play_set
from estimation.bots import make_bot
from estimation.bots.hybrid.rollout import Start
from estimation.bots.hybrid.rollout import _beats as beats  # the Hybrid's own trick rule
from estimation.bots.knowledge import read_knowledge
from estimation.engine import Observation, Phase, RulesConfig
from estimation.jaxenv.encode_np import PLAY_BASE, SIZE, encode
from estimation.onenet import nets
from estimation.onenet.nets import PRACTICE, PRACTICE_SIZE
from estimation.onenet.search import (
    OneNetSearch,
    bidding_views,
    load_judge,
    load_net,
    play_out,
    templates_of,
    write_view,
)
from tests.rl.helpers import make_checkpoint

Seen = list[tuple[Observation, list[int], int]]


def random_net(folder: Path, answers: bool = True) -> str:
    """A small random player; with ``answers``, saved with its practice outputs as step 5 saves
    its players."""
    rng = np.random.default_rng(0)
    path = folder / ("player.npz" if answers else "player-moves-only.npz")
    arrays = {
        "w0": rng.normal(size=(SIZE, 32)).astype(np.float32),
        "b0": np.zeros(32, np.float32),
        "pw": rng.normal(size=(32, 120)).astype(np.float32),
        "pb": np.zeros(120, np.float32),
    }
    if answers:
        arrays["qw"] = rng.normal(size=(32, PRACTICE_SIZE)).astype(np.float32)
        arrays["qb"] = np.zeros(PRACTICE_SIZE, np.float32)
        arrays["meta"] = np.array(json.dumps({"practice": [list(p) for p in PRACTICE]}))
    np.savez(path, **arrays)
    return str(path)


@pytest.fixture(scope="module")
def judge_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A small random judge, saved as self-play saves its judge."""
    path = tmp_path_factory.mktemp("judge") / "judge.npz"
    nets.save_judge(path, nets.init_judge(jax.random.PRNGKey(2), 32, 2), {"loop": 0})
    return str(path)


class Recorder:
    """Plays the net's top legal action and keeps every decision it saw."""

    def __init__(self, net: Any, seen: Seen) -> None:
        self.net, self.seen = net, seen

    def act(self, obs: Observation, legal: list[int]) -> int:
        logits = self.net(encode(obs, legal))
        action = max(legal, key=lambda a: logits[a])
        self.seen.append((obs, list(legal), action))
        return action


@pytest.fixture(scope="module")
def game(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, list[Seen]]:
    """Every decision of one duplicate set of the net against itself, by round (a round starts
    at the decision whose public log is empty)."""
    net = load_net(random_net(tmp_path_factory.mktemp("net")))
    seen: Seen = []
    play_set(("a", "b", "c", "d"), 7, 0, RulesConfig(), make=lambda *_: Recorder(net, seen))
    rounds: list[Seen] = []
    for row in seen:
        if not row[0].log:
            rounds.append([])
        rounds[-1].append(row)
    assert rounds
    return net, rounds


def cards_of_round(decisions: Seen) -> Seen:
    return [row for row in decisions if row[0].phase is Phase.PLAY]


def test_the_play_out_view_equals_the_encoder_at_every_decision(game: tuple[Any, list[Seen]]):
    _, rounds = game
    for decisions in filter(None, map(cards_of_round, rounds)):
        templates = templates_of(decisions[0][0])
        for obs, legal, _ in decisions:
            know = read_knowledge(obs)
            cards = [c for _, c in obs.trick] + [0] * (4 - len(obs.trick))
            mask = sum(1 << (a - PLAY_BASE) for a in legal)
            x = np.empty((1, SIZE), np.float32)
            write_view(
                x,
                templates,
                np.array([obs.hand], np.uint64),
                np.array([mask], np.uint64),
                np.array([know.played_by], np.uint64),
                np.array([know.voids], np.uint8),
                np.array([obs.seat]),
                np.array([cards]),
                len(obs.trick),
                np.array([obs.leader]),
                np.array([obs.tricks_won]),
            )
            np.testing.assert_array_equal(x[0], encode(obs, legal))


def test_a_play_out_replays_the_real_round_and_judges_the_seat_at_its_last_card(
    game: tuple[Any, list[Seen]], judge_file: str
):
    net, rounds = game
    judge = load_judge(judge_file)
    for decisions in filter(None, map(cards_of_round, rounds)):
        first, _, action = decisions[0]
        dealt = [0, 0, 0, 0]
        for obs, _, _ in decisions:
            dealt[obs.seat] = dealt[obs.seat] or obs.hand
        trump, leader = first.trump, first.leader
        assert trump is not None
        assert leader is not None
        start = Start(
            trick=[],
            leader=leader,
            won=first.tricks_won,
            tricks_done=0,
            estimates=(0, 0, 0, 0),
            trump=trump,
        )
        out = play_out(
            net,
            templates_of(first),
            start,
            np.array([dealt], np.uint64),
            np.zeros((1, 4), np.uint64),
            np.zeros((1, 4), np.uint8),
            np.array([action - PLAY_BASE]),
            judge_seat=first.seat,
            judge=judge.value,
        )
        last, _, final = decisions[-1]
        assert len(decisions) == 52
        plays = [c for _, c in last.trick] + [final - PLAY_BASE]
        best = 0
        for i, c in enumerate(plays):
            if beats(c, plays[best], trump):
                best = i
        assert last.leader is not None
        expected = list(last.tricks_won)
        expected[(last.leader + best) % 4] += 1
        assert out.won[0].tolist() == expected
        at_last_card = next(
            (obs, legal)
            for obs, legal, _ in decisions
            if len(obs.tricks) == 12 and obs.seat == first.seat
        )
        obs, legal = at_last_card
        left = [dealt[s] & ~read_knowledge(obs).played_by[s] for s in range(4)]
        others = np.array([[left[(obs.seat + k) % 4] for k in (1, 2, 3)]], np.uint64)
        assert out.value is not None
        expected = judge.value(encode(obs, legal)[np.newaxis], others)[0]
        assert out.value[0] == pytest.approx(expected, rel=1e-5)


def test_the_numpy_judge_equals_the_training_judge(judge_file: str):
    rng = np.random.default_rng(1)
    x = (rng.random((6, SIZE)) < 0.1).astype(np.float32)
    cards = rng.integers(0, 2**52, size=(6, 3), dtype=np.uint64)
    hidden = np.unpackbits(cards.astype("<u8").view(np.uint8), bitorder="little")
    hidden = hidden.reshape(6, 3, 64)[:, :, :52].reshape(6, 156)
    net, _ = nets.load_judge(Path(judge_file))
    expected = np.asarray(nets.apply_judge(net, jnp.asarray(x), jnp.asarray(hidden)))
    got = load_judge(judge_file).value(x, cards)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_the_bidding_views_equal_the_encoder_with_the_dealt_hand(game: tuple[Any, list[Seen]]):
    _, rounds = game
    checked = 0
    for decisions in rounds:
        bids = [row for row in decisions if row[0].phase is not Phase.PLAY]
        for obs, _, _ in cards_of_round(decisions)[:4]:  # each seat's first card decision
            views = bidding_views(obs, RulesConfig())
            expected = [row for row in bids if row[0].seat != obs.seat and len(row[1]) > 1]
            assert views.seats == tuple(row[0].seat for row in expected)
            for i, (seen, legal, action) in enumerate(expected):
                x = views.templates[i].copy()
                x[:52] = np.unpackbits(
                    np.array([seen.hand], "<u8").view(np.uint8), bitorder="little"
                )[:52]
                np.testing.assert_array_equal(x, encode(seen, legal))
                assert sorted(np.flatnonzero(views.legal[i])) == sorted(legal)
                assert views.chosen[i] == action
                checked += 1
    assert checked > 100


def test_the_all_in_one_bot_weighs_deals_by_the_players_own_move_chances(
    game: tuple[Any, list[Seen]], tmp_path: Path
):
    net, rounds = game
    path = random_net(tmp_path)
    bot = make_bot(f"onenet:net={path}:own=true:judge=king:samples=2", 0)
    assert isinstance(bot, OneNetSearch)
    floor = bot.config.belief_floor
    for decisions in rounds[:6]:
        dealt = [0, 0, 0, 0]
        for seen, _, _ in decisions:
            dealt[seen.seat] = dealt[seen.seat] or seen.hand
        cards = cards_of_round(decisions)
        if not cards:
            continue
        obs = cards[0][0]
        expected = 0.0
        for seen, legal, action in decisions:
            if seen.phase is Phase.PLAY or seen.seat == obs.seat or len(legal) == 1:
                continue
            logits = net(encode(seen, legal)[np.newaxis])[0].astype(np.float64)[legal]
            chances = np.exp(logits - logits.max())
            chances /= chances.sum()
            expected += math.log((1 - floor) * chances[legal.index(action)] + floor / len(legal))
        got = bot.bidding_log_likelihoods(obs, [dealt], [0, 0, 0, 0])
        if expected == 0.0:
            assert got is None
        else:
            assert got is not None
            assert got[0] == pytest.approx(expected, rel=1e-5)


def test_the_all_in_one_bot_guesses_the_hidden_cards_with_the_players_own_answer(
    game: tuple[Any, list[Seen]], tmp_path: Path
):
    net, rounds = game
    path = random_net(tmp_path)
    obs, legal, _ = next(
        row for row in cards_of_round(rounds[0]) if len(row[1]) > 1 and row[0].tricks
    )
    expected = net.holder_chances(encode(obs, legal)[np.newaxis])[0]
    np.testing.assert_allclose(expected.sum(axis=1), 1.0, rtol=1e-5)
    plain = make_bot(f"onenet:net={path}:own=true", 0)
    assert isinstance(plain, OneNetSearch)
    np.testing.assert_allclose(plain.guess_chances(obs), expected, rtol=1e-5, atol=1e-12)
    sharp = make_bot(f"onenet:net={path}:own=true:guess_temperature=0.5", 0)
    assert isinstance(sharp, OneNetSearch)
    squared = expected.astype(np.float64) ** 2
    np.testing.assert_allclose(sharp.guess_chances(obs), squared, rtol=1e-5, atol=1e-12)


def test_it_plays_a_set_inside_the_hybrid_search(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path / "hybrid.pt", seed=3, guess=True)
    spec = f"onenet:net={random_net(tmp_path)}:checkpoint={checkpoint}:samples=2:top_k=2"
    bot = make_bot(spec, 0)
    assert isinstance(bot, OneNetSearch)
    assert not bot.own
    assert bot.judge == "round"
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig())
    assert len(result.scores) == 4


def test_the_all_in_one_bot_plays_a_set_with_no_checkpoint(tmp_path: Path):
    spec = f"onenet:net={random_net(tmp_path)}:own=true:judge=king:samples=2:top_k=2"
    bot = make_bot(spec, 0)
    assert isinstance(bot, OneNetSearch)
    assert bot.own
    assert bot.judge == "king"
    with pytest.raises(ValueError, match="without a checkpoint"):
        _ = bot.policy
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig())
    assert len(result.scores) == 4


def test_a_king_judged_search_scores_each_card_by_its_mean_king_chance(
    game: tuple[Any, list[Seen]], tmp_path: Path
):
    _, rounds = game
    bot = make_bot(
        f"onenet:net={random_net(tmp_path)}:own=true:judge=king:samples=4:merge_equivalent=false", 0
    )
    assert isinstance(bot, OneNetSearch)
    obs, legal, _ = next(row for row in cards_of_round(rounds[1]) if len(row[1]) > 2)
    search = bot.search(obs, [a - PLAY_BASE for a in legal], [0.0] * len(legal))
    assert search.deals == 4
    assert all(0.0 <= mean <= 1.0 for mean in search.means)


def test_it_needs_the_players_file_and_valid_settings(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path / "hybrid.pt")
    path = random_net(tmp_path)
    with pytest.raises(ValueError, match="net="):
        make_bot(f"onenet:checkpoint={checkpoint}", 0)
    with pytest.raises(ValueError, match="no player file"):
        make_bot(f"onenet:net={tmp_path / 'none.npz'}:checkpoint={checkpoint}", 0)
    with pytest.raises(ValueError, match="no checkpoint or deals"):
        make_bot(f"onenet:net={path}:own=true:checkpoint={checkpoint}", 0)
    with pytest.raises(ValueError, match="judge"):
        make_bot(f"onenet:net={path}:own=true:judge=points", 0)
    moves_only = random_net(tmp_path, answers=False)
    for spec in (f"onenet:net={moves_only}:own=true", f"onenet:net={moves_only}:judge=king"):
        with pytest.raises(ValueError, match="no practice answers"):
            make_bot(spec, 0)
    with pytest.raises(ValueError, match="judge_net"):
        make_bot(f"onenet:net={path}:own=true:judge=place", 0)
    with pytest.raises(ValueError, match="judge_net"):
        make_bot(f"onenet:net={path}:own=true:judge_net={path}", 0)
    with pytest.raises(ValueError, match="not a judge"):
        make_bot(f"onenet:net={path}:own=true:judge=place:judge_net={path}", 0)


def test_a_place_judged_bot_plays_a_set_scoring_each_card_by_the_judge(
    game: tuple[Any, list[Seen]], judge_file: str, tmp_path: Path
):
    _, rounds = game
    spec = (
        f"onenet:net={random_net(tmp_path)}:own=true:judge=place:judge_net={judge_file}"
        ":samples=2:top_k=2"
    )
    bot = make_bot(spec, 0)
    assert isinstance(bot, OneNetSearch)
    assert bot.judge == "place"
    obs, legal, _ = next(row for row in cards_of_round(rounds[1]) if len(row[1]) > 2)
    search = bot.search(obs, [a - PLAY_BASE for a in legal], [0.0] * len(legal))
    assert search.deals == 2
    assert all(math.isfinite(mean) for mean in search.means)
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig())
    assert len(result.scores) == 4
