"""Step 5's self-play: its config, its answers to the practice questions, and a tiny run."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.engine.cards import deal
from estimation.jaxenv import rules
from estimation.jaxenv.encode import SIZE
from estimation.onenet.data import card_bits, holders
from estimation.onenet.nets import (
    apply_player,
    init_judge,
    init_player,
    load_player,
    save_judge,
    save_player,
)
from estimation.onenet.selfplay import (
    SelfPlayConfig,
    config_of,
    holder_of,
    load_config,
    round_labels,
    run,
    sharpened,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_committed_config_starts_from_the_final_copy_with_the_three_guards():
    config = load_config(ROOT / "configs" / "onenet-selfplay.toml")
    assert config.start == "runs/onenet/copy-final/384x3"
    assert config.loops == 2000
    assert config.entropy < 0.02  # 1: gentler than Essam's from-scratch exploration
    assert config.anchor > 0  # 1: a pull toward the copy
    assert config.anchor_loops < config.loops  # that fades out during the run
    assert 0 < config.copy_share < 1  # 2: the frozen copy among the opponents
    assert config.save_every == 100  # 3: players kept for the check-ups
    assert config.sharpen == 4.0


def test_sharpening_keeps_the_best_move_and_makes_it_surer():
    player = init_player(jax.random.key(4), 16, 1)
    x = jax.random.bernoulli(jax.random.key(5), 0.2, (6, SIZE)).astype(jnp.float32)
    soft, practice = apply_player(player, x)
    sure, same = apply_player(sharpened(player, 4.0), x)
    assert np.allclose(sure, 4.0 * soft, atol=1e-5)
    assert np.allclose(same, practice)  # the practice outputs are untouched


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, r"\[selfplay\] section"),
        ({"selfplay": {}, "copy": {}}, "unknown section"),
        ({"selfplay": {"copy_share": 1.5}}, "copy_share"),
        ({"selfplay": {"batch": 2, "steps": 2, "minibatch": 8}}, "minibatch"),
        ({"selfplay": {"objective": "points"}}, "objective"),
        ({"selfplay": {"sharpen": 0.0}}, "sharpen"),
        ({"selfplay": {"teacher": 1}}, "unknown key 'teacher'"),
    ],
)
def test_bad_configs_are_refused(data: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        config_of(data)


def test_the_games_name_each_cards_holder_as_the_training_set_does():
    hands = deal(11, 1)
    jax_round = rules.init_round(1, np.stack([card_bits(h) for h in hands]))
    for seat in range(4):
        got = np.asarray(holder_of(jax_round, jnp.asarray(seat)))
        assert got.tolist() == holders(hands, seat).tolist()


def trace(**changes: Any) -> dict[str, jax.Array]:
    """Seven decisions of one game slot: a round ends at step 2, a skipped round at step 3, and
    the Bola's last round at step 5; step 6 starts the next Bola."""
    t = {
        "seat": [0, 1, 2, 3, 0, 1, 2],
        "ended": [0, 0, 1, 1, 0, 1, 0],
        "skipped": [0, 0, 0, 1, 0, 0, 0],
        "finished": [0, 0, 0, 0, 0, 1, 0],
        "overflow": [0] * 7,
    }
    t |= changes
    four = np.zeros((7, 4), np.int32)
    won, made, result, scores = four.copy(), four.copy(), four.copy(), four.copy()
    won[2], made[2], result[2] = [3, 4, 5, 1], [1, 0, 1, 0], [13, -1, 15, -3]
    won[5], made[5], result[5] = [2, 2, 4, 5], [0, 0, 1, 1], [-2, -2, 14, 15]
    scores[5] = [100, 50, 20, 10]
    out = {k: jnp.asarray(np.array(v)[:, None]) for k, v in t.items()}
    for name in ("ended", "skipped", "finished", "overflow"):
        out[name] = out[name].astype(bool)
    for name, value in (("won", won), ("made", made), ("result", result), ("scores", scores)):
        out[name] = jnp.asarray(value[:, None, :])
    return out


def rel(values: list[int], seat: int) -> list[int]:
    return [values[(seat + k) % 4] for k in range(4)]


def test_each_decision_gets_its_rounds_and_its_bolas_answers_from_its_own_side():
    labels = {k: np.asarray(v)[:, 0] for k, v in round_labels(trace()).items()}
    seats = [0, 1, 2, 3, 0, 1, 2]
    for step in (0, 1, 2):  # the first round
        assert labels["tricks"][step].tolist() == rel([3, 4, 5, 1], seats[step])
        assert labels["made"][step].tolist() == rel([1, 0, 1, 0], seats[step])
        assert labels["round_score"][step].tolist() == rel([13, -1, 15, -3], seats[step])
    assert labels["tricks"][3].tolist() == [-1] * 4  # a skipped round: no tricks, no calls
    assert labels["made"][3].tolist() == [-1] * 4
    assert labels["round_score"][3].tolist() == [0] * 4
    for step in (4, 5):
        assert labels["tricks"][step].tolist() == rel([2, 2, 4, 5], seats[step])
    assert labels["round_ok"].tolist() == [True] * 6 + [False]  # step 6's round ends later
    assert labels["tricks"][6].tolist() == [-1] * 4
    for step in range(6):
        assert labels["final"][step].tolist() == rel([100, 50, 20, 10], seats[step])
    assert labels["final_ok"].tolist() == [True] * 6 + [False]


def test_a_bola_dropped_past_int32s_range_teaches_no_final_place():
    dropped = trace(finished=[0] * 7, overflow=[0, 0, 0, 0, 0, 1, 0])
    labels = round_labels(dropped)
    assert not np.asarray(labels["final_ok"]).any()
    assert np.asarray(labels["round_ok"])[:6].all()  # its rounds still ended


def tiny(start: Path, loops: int, **changes: Any) -> SelfPlayConfig:
    settings: dict[str, Any] = {
        "start": str(start),
        "loops": loops,
        "batch": 4,
        "steps": 16,
        "minibatch": 32,
        "epochs": 1,
        "anchor_loops": 2,
        "snapshot_every": 1,
        "pool": 2,
        "eval_every": 1,
        "eval_games": 4,
        "eval_steps": 50,
        "save_every": 1,
        "seed": 3,
    }
    return SelfPlayConfig(**{**settings, **changes})


def test_a_tiny_run_plays_learns_keeps_players_and_resumes(tmp_path: Path):
    start = tmp_path / "copy"
    save_player(start / "player.npz", init_player(jax.random.key(0), 16, 1), {"epoch": 5})
    save_judge(start / "judge.npz", init_judge(jax.random.key(1), 16, 1), {"epoch": 1})
    out = tmp_path / "selfplay"
    last = run(tiny(start, 2), out)
    assert last["loop"] == 2
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["loop"] for r in rows] == [1, 2]
    assert {"king_vs_copy", "place_vs_copy", "distance_from_copy", "holders_named"} <= set(rows[0])
    assert rows[0]["anchor_weight"] == 1.0
    assert rows[1]["anchor_weight"] == 0.5  # the pull fades
    assert rows[1]["distance_from_copy"] > 0  # the player has moved off the copy
    _, meta = load_player(out / "snapshots" / "player-00002.npz")
    assert meta["loop"] == 2
    assert any((out / "charts").iterdir())
    run(tiny(start, 3), out)  # a finished run continues when its loops are raised
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["loop"] for r in rows] == [1, 2, 3]
    with pytest.raises(SystemExit, match="other settings"):
        run(dataclasses.replace(tiny(start, 4), seed=9), out)


def test_self_play_loads_no_pytorch():
    """The home server runs self-play in its JAX env, which has no PyTorch."""
    check = "import sys, estimation.onenet.selfplay; sys.exit('torch' in sys.modules)"
    subprocess.run([sys.executable, "-c", check], check=True)
