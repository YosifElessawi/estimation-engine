"""Step 5's check-up bot: the one net inside the Hybrid search (Essam Gouda's bitter-search tests,
from estimation-arena tests/test_bitter_search.py, MIT licence)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from estimation.arena.duplicate import play_set
from estimation.bots import make_bot
from estimation.bots.hybrid.rollout import Start
from estimation.bots.hybrid.rollout import _beats as beats  # the Hybrid's own trick rule
from estimation.bots.knowledge import read_knowledge
from estimation.engine import Observation, Phase, RulesConfig
from estimation.jaxenv.encode_np import PLAY_BASE, SIZE, encode
from estimation.onenet.search import OneNetSearch, load_net, play_out, templates_of, write_view
from tests.rl.helpers import make_checkpoint

Seen = list[tuple[Observation, list[int], int]]


def random_net(folder: Path) -> str:
    rng = np.random.default_rng(0)
    path = folder / "player.npz"
    np.savez(
        path,
        w0=rng.normal(size=(SIZE, 32)).astype(np.float32),
        b0=np.zeros(32, np.float32),
        pw=rng.normal(size=(32, 120)).astype(np.float32),
        pb=np.zeros(120, np.float32),
    )
    return str(path)


class Recorder:
    """Plays the net's top legal action and keeps the card decisions it saw."""

    def __init__(self, net: Any, seen: Seen) -> None:
        self.net, self.seen = net, seen

    def act(self, obs: Observation, legal: list[int]) -> int:
        logits = self.net(encode(obs, legal))
        action = max(legal, key=lambda a: logits[a])
        if obs.phase is Phase.PLAY:
            self.seen.append((obs, list(legal), action))
        return action


@pytest.fixture(scope="module")
def game(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, list[Seen]]:
    """The card decisions of one duplicate set of the net against itself, by round."""
    net = load_net(random_net(tmp_path_factory.mktemp("net")))
    seen: Seen = []
    play_set(("a", "b", "c", "d"), 7, 0, RulesConfig(), make=lambda *_: Recorder(net, seen))
    rounds: list[Seen] = []
    for row in seen:
        if not row[0].tricks and not row[0].trick:
            rounds.append([])
        rounds[-1].append(row)
    assert rounds
    return net, rounds


def test_the_play_out_view_equals_the_encoder_at_every_decision(game: tuple[Any, list[Seen]]):
    _, rounds = game
    for decisions in rounds:
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


def test_a_play_out_replays_the_real_round(game: tuple[Any, list[Seen]]):
    net, rounds = game
    for decisions in rounds:
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
        won = play_out(
            net,
            templates_of(first),
            start,
            np.array([dealt], np.uint64),
            np.zeros((1, 4), np.uint64),
            np.zeros((1, 4), np.uint8),
            np.array([action - PLAY_BASE]),
        )[0]
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
        assert won.tolist() == expected


def test_it_plays_a_set_inside_the_hybrid_search(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path / "hybrid.pt", seed=3, guess=True)
    spec = f"onenet:net={random_net(tmp_path)}:checkpoint={checkpoint}:samples=2:top_k=2"
    assert isinstance(make_bot(spec, 0), OneNetSearch)
    result, _ = play_set((spec, "random", spec, "random"), 1, 0, RulesConfig())
    assert len(result.scores) == 4


def test_it_needs_the_players_file(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path / "hybrid.pt")
    with pytest.raises(ValueError, match="net="):
        make_bot(f"onenet:checkpoint={checkpoint}", 0)
    with pytest.raises(ValueError, match="no player file"):
        make_bot(f"onenet:net={tmp_path / 'none.npz'}:checkpoint={checkpoint}", 0)
