"""Step 5's training set: every row is its recorded decision, replayed, with the right answers."""

from pathlib import Path

import numpy as np
import pytest

from estimation.bots import HybridBot, make_bot
from estimation.engine import Bola, cards_of
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.onenet.data import build, record_config
from estimation.onenet.shards import (
    DTYPES,
    KIND_BID,
    KIND_CARD,
    KIND_PLAYED,
    load_shard,
    save_shard,
    shard_path,
)
from estimation.rl.encoder import encode as rl_view
from estimation.rl.onenet_record import load_part, part_path
from tests.onenet.conftest import BOLAS


def relative(values: tuple[int, ...] | list[int], seat: int) -> list[int]:
    return [values[(seat + k) % 4] for k in range(4)]


def test_every_row_is_its_replayed_decision_with_the_real_answers(record: Path, data: Path):
    part = load_part(part_path(record, 0))
    shard = load_shard(shard_path(data, 0))
    bot = make_bot(record_config(record).player, 0)
    assert isinstance(bot, HybridBot)
    kept = {int(step): k for k, step in enumerate(part["step"].tolist())}
    actions = part["actions"].tolist()
    row = 0
    for b in range(BOLAS):
        bola = Bola(int(part["seed"][b]), DEFAULT_CONFIG)
        first = row
        in_round: list[int] = []
        for index in range(int(part["log_start"][b]), int(part["log_start"][b + 1])):
            seat = bola.current_player
            assert seat is not None
            legal = bola.legal_actions()
            if len(legal) > 1:
                obs = bola.observation(seat)
                assert shard["bola"][row] == part["game"][b]
                assert shard["seat"][row] == seat
                assert shard["action"][row] == actions[index]
                assert np.array_equal(shard["x"][row], raw_view(obs, legal).astype(np.float16))
                for k in (1, 2, 3):  # each other seat's own view of its hand
                    held = cards_of(bola.observation((seat + k) % 4).hand)
                    assert (shard["holder"][row][held] == k - 1).all()
                hidden = sum(
                    len(cards_of(bola.observation((seat + k) % 4).hand)) for k in (1, 2, 3)
                )
                assert (shard["holder"][row] >= 0).sum() == hidden
                k = kept.get(index)
                values = shard["values"][row]
                if k is None:
                    assert shard["kind"][row] == KIND_PLAYED
                    assert not values.any()
                else:
                    assert shard["kind"][row] == part["kind"][k]
                    stored = part["values"][part["value_start"][k] : part["value_start"][k + 1]]
                    assert values[legal] == pytest.approx(stored)
                    assert not np.delete(values, legal).any()
                if shard["kind"][row] == KIND_CARD:
                    cards = bot.policy.card_values(rl_view(obs, DEFAULT_CONFIG)[np.newaxis])[0]
                    assert shard["card_net"][row] == max(legal, key=lambda a: cards[a - 68])
                else:
                    assert shard["card_net"][row] == -1
                in_round.append(row)
                row += 1
            rounds = len(bola.history)
            bola.step(actions[index])
            if len(bola.history) > rounds:
                result = bola.history[-1]
                for r in in_round:
                    seat_r = int(shard["seat"][r])
                    want_tricks = [-1] * 4 if result.skipped else [s.tricks for s in result.seats]
                    want_made = [-1] * 4 if result.skipped else [int(s.won) for s in result.seats]
                    assert shard["tricks"][r].tolist() == relative(want_tricks, seat_r)
                    assert shard["made"][r].tolist() == relative(want_made, seat_r)
                    assert shard["round_score"][r].tolist() == relative(result.scores, seat_r)
                in_round = []
        for r in range(first, row):
            assert shard["final"][r].tolist() == relative(bola.scores, int(shard["seat"][r]))
    assert row == len(shard["kind"])
    assert (shard["kind"] != KIND_PLAYED).sum() == len(part["step"])


def test_a_build_resumes_and_counts_every_row(record: Path, data: Path):
    shard = shard_path(data, 0)
    stamp = shard.stat().st_mtime_ns
    summary = build(record, data)
    assert shard.stat().st_mtime_ns == stamp  # not built again
    kinds = load_shard(shard)["kind"]
    assert summary == {
        "shards": 1,
        "bolas": BOLAS,
        "rows": len(kinds),
        "bids": int((kinds == KIND_BID).sum()),
        "valued_cards": int((kinds == KIND_CARD).sum()),
        "played_cards": int((kinds == KIND_PLAYED).sum()),
    }
    assert summary["bids"] > 0
    assert summary["valued_cards"] > 0
    assert summary["played_cards"] > 0


def test_a_shard_needs_every_field(tmp_path: Path):
    partial = {name: np.zeros(1, dtype) for name, dtype in DTYPES.items() if name != "holder"}
    with pytest.raises(ValueError, match="holder"):
        save_shard(tmp_path / "shard-00000.npz", partial)
