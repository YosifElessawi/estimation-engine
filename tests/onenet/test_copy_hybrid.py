"""Step 5's copy of Hybrid v4: its config, its losses, and a tiny run from end to end."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from estimation.onenet.copy_hybrid import (
    CopyConfig,
    config_of,
    king_share,
    load_config,
    move_loss,
    place_reward,
    played_loss,
    run,
    voids,
)
from estimation.onenet.nets import load_judge, load_player
from estimation.onenet.shards import KIND_CARD, NUM_ACTIONS, VIEW

ROOT = Path(__file__).resolve().parents[2]


def test_the_committed_config_copies_both_sizes_from_the_full_recording():
    config = load_config(ROOT / "configs" / "onenet-copy.toml")
    assert config.data == "runs/onenet/data"
    assert config.sizes == ("384x3", "512x4")
    assert config.held_out == 0.05


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, r"\[copy\] section"),
        ({"copy": {}, "train": {}}, "unknown section"),
        ({"copy": {"sizes": ["384"]}}, "units x layers"),
        ({"copy": {"held_out": 1.0}}, "held_out"),
        ({"copy": {"temperature": 0}}, "above 0"),
        ({"copy": {"played_weight": -1.0}}, "played_weight"),
        ({"copy": {"teacher": "x"}}, "unknown key 'teacher'"),
    ],
)
def test_bad_configs_are_refused(data: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        config_of(data)


def test_out_of_suit_king_and_final_place_answers():
    holder = np.full((1, 52), -1, np.int8)
    holder[0, :13] = 0  # relative seat 1 holds every club
    holder[0, 13] = 2  # relative seat 3 holds one diamond
    out = np.asarray(voids(jnp.asarray(holder)))[0]
    assert out[0].tolist() == [False, True, True, True]
    assert out[1].tolist() == [True, True, True, True]
    assert out[2].tolist() == [True, False, True, True]
    final = jnp.array([[10, 10, 5, 0], [5, 10, 0, 0], [0, 10, 5, 0]])
    assert np.allclose(king_share(final), [0.5, 0.0, 0.0])
    assert np.allclose(place_reward(final), [0.6, 0.2, -0.6])  # ties share their places


def test_the_move_output_learns_the_values_as_chances():
    x = np.zeros((1, VIEW), np.float32)
    x[0, -NUM_ACTIONS + 68] = x[0, -NUM_ACTIONS + 69] = 1  # two legal cards
    values = np.zeros((1, NUM_ACTIONS), np.float32)
    values[0, 68] = 10.0
    mb = {"x": jnp.asarray(x), "values": jnp.asarray(values), "kind": jnp.array([KIND_CARD])}
    even = move_loss(jnp.zeros((1, NUM_ACTIONS)), mb, temperature=2.0)
    assert float(even[0]) == pytest.approx(np.log(2.0))  # equal outputs: ln 2 whatever the target
    sure = jnp.zeros((1, NUM_ACTIONS)).at[0, 68].set(5.0)  # the same chances as the target
    target = np.exp([5.0, 0.0]) / np.exp([5.0, 0.0]).sum()
    assert float(move_loss(sure, mb, temperature=2.0)[0]) == pytest.approx(
        -(target * np.log(target)).sum(), abs=1e-5
    )


def test_the_move_output_learns_the_move_played_where_there_are_no_values():
    x = np.zeros((1, VIEW), np.float32)
    x[0, -NUM_ACTIONS + 68] = x[0, -NUM_ACTIONS + 69] = 1  # two legal cards
    mb = {"x": jnp.asarray(x), "action": jnp.array([69])}
    assert float(played_loss(jnp.zeros((1, NUM_ACTIONS)), mb)[0]) == pytest.approx(np.log(2.0))
    sure = jnp.zeros((1, NUM_ACTIONS)).at[0, 69].set(20.0)
    assert float(played_loss(sure, mb)[0]) == pytest.approx(0.0, abs=1e-6)


def test_a_tiny_copy_learns_scores_and_keeps_its_best_epochs(data: Path, tmp_path: Path):
    config = CopyConfig(
        data=str(data),
        sizes=("16x1",),
        epochs=3,
        batch=64,
        lr=3e-3,
        held_out=0.34,
        played_weight=1.0,
        seed=2,
    )
    summary = run(config, tmp_path / "copy")
    folder = tmp_path / "copy" / "16x1"
    epochs = [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    assert epochs[-1]["train"]["move"] < epochs[0]["train"]["move"]
    size = summary["sizes"][0]
    kept = size["kept"]
    assert kept["held_out"]["move_loss"] == min(e["held_out"]["move_loss"] for e in epochs)
    _, meta = load_player(folder / "player.npz")
    assert meta["epoch"] == kept["epoch"]
    judge_mse = [e["held_out"]["judge_mse"] for e in epochs]
    assert size["judge"] == {
        "epoch": 1 + judge_mse.index(min(judge_mse)),
        "judge_mse": min(judge_mse),
    }
    _, judge_meta = load_judge(folder / "judge.npz")
    assert judge_meta["epoch"] == size["judge"]["epoch"]
    assert epochs[-1]["train"]["played"] < epochs[0]["train"]["played"]
    scores = kept["held_out"]
    assert scores["card_lose1"] >= scores["card_lose3"] >= 0
    assert 0 <= scores["card_best"] <= 1
    assert 1 / 3 <= scores["holder_blind"] <= 1
    assert set(size["card_net"]) == {"card_lose1", "card_best"}
    assert summary["rows"]["train"] > 0
    assert summary["rows"]["held_out"] > 0
    assert json.loads((folder / "metrics.json").read_text()) == size


def test_the_copy_loads_no_pytorch():
    """The home server runs the copy in its JAX env, which has no PyTorch."""
    check = "import sys, estimation.onenet.copy_hybrid; sys.exit('torch' in sys.modules)"
    subprocess.run([sys.executable, "-c", check], check=True)
