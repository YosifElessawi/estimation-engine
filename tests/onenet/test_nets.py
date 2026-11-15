"""Step 5's nets: the player's outputs, the judge's inputs, and saving both."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.engine.cards import deal
from estimation.jaxenv import rules
from estimation.jaxenv.encode import SIZE
from estimation.jaxenv.selfplay import HIDDEN
from estimation.jaxenv.selfplay import _hidden_hands as jax_hidden_hands  # the JAX games' own
from estimation.onenet.data import card_bits, holders
from estimation.onenet.nets import (
    PRACTICE,
    PRACTICE_SIZE,
    apply_judge,
    apply_player,
    hidden_hands,
    init_judge,
    init_player,
    load_judge,
    load_player,
    parse_size,
    save_judge,
    save_player,
)


def test_net_sizes_are_units_by_layers():
    assert parse_size("384x3") == (384, 3)
    assert parse_size("512X4") == (512, 4)
    for bad in ("384", "0x3", "384x3x2", "big"):
        with pytest.raises(ValueError, match="unit"):
            parse_size(bad)


def test_the_player_answers_the_seven_practice_questions():
    assert [name for name, _ in PRACTICE] == [
        "holder",
        "my_tricks",
        "made",
        "other_tricks",
        "void",
        "king",
        "round_score",
    ]
    assert PRACTICE_SIZE == 156 + 14 + 4 + 42 + 12 + 1 + 4
    net = init_player(jax.random.key(0), 16, 2)
    x = jax.random.bernoulli(jax.random.key(1), 0.2, (5, SIZE)).astype(jnp.float32)
    logits, practice = apply_player(net, x)
    assert logits.shape == (5, rules.NUM_ACTIONS)
    assert practice.shape == (5, PRACTICE_SIZE)
    assert float(jnp.abs(logits).max()) < 0.5  # the move output starts near uniform
    judge = init_judge(jax.random.key(2), 16, 2)
    assert apply_judge(judge, x, jnp.zeros((5, HIDDEN))).shape == (5,)


def test_the_judge_sees_the_hidden_hands_as_the_jax_games_give_them():
    hands = deal(7, 1)
    jax_round = rules.init_round(1, np.stack([card_bits(h) for h in hands]))
    for seat in range(4):
        theirs = jnp.concatenate(jax_hidden_hands(jax_round, seat))
        ours = hidden_hands(jnp.asarray(holders(hands, seat))[None])[0]
        assert bool((theirs == ours).all())


def test_saved_nets_give_the_same_outputs(tmp_path: Path):
    player, judge = init_player(jax.random.key(3), 8, 2), init_judge(jax.random.key(4), 8, 2)
    save_player(tmp_path / "player.npz", player, {"epoch": 3})
    save_judge(tmp_path / "judge.npz", judge, {"epoch": 3})
    player2, meta = load_player(tmp_path / "player.npz")
    judge2, _ = load_judge(tmp_path / "judge.npz")
    assert meta["epoch"] == 3
    x = jax.random.normal(jax.random.key(5), (4, SIZE))
    hidden = jax.random.bernoulli(jax.random.key(6), 0.3, (4, HIDDEN))
    for a, b in zip(apply_player(player, x), apply_player(player2, x), strict=True):
        assert np.allclose(a, b)
    assert np.allclose(apply_judge(judge, x, hidden), apply_judge(judge2, x, hidden))
    with np.load(tmp_path / "player.npz") as saved:  # Essam's names for the body and move output
        assert {"w0", "b0", "w1", "b1", "pw", "pb"} <= set(saved.files)


def test_a_player_with_other_practice_outputs_is_refused(tmp_path: Path):
    save_player(tmp_path / "player.npz", init_player(jax.random.key(7), 8, 1), {})
    with np.load(tmp_path / "player.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["meta"] = np.array(json.dumps({"practice": [["holder", 156]]}))
    np.savez(tmp_path / "other.npz", **arrays)
    with pytest.raises(ValueError, match="other practice outputs"):
        load_player(tmp_path / "other.npz")
