"""Recorded PIMC games for the warm start (SPEC §6 M4)."""

from pathlib import Path

import numpy as np

from estimation.engine import Bola, Phase
from estimation.engine.actions import PLAY_BASE
from estimation.rl.config import RECORD_STAGE, config_from_dict, derive_seed
from estimation.rl.encoder import NUM_FEATURES
from estimation.rl.record import (
    concat_records,
    load_records,
    pimc_bots,
    record_game,
    save_records,
)

CONFIG = config_from_dict({"seed": 5, "record": {"games": 2, "pimc_samples": 2}})


def plain_pimc_choices(index: int) -> list[int]:
    """The cards four PIMC bots play at decisions with a choice, in an unrecorded game."""
    bola = Bola(derive_seed(CONFIG.seed, RECORD_STAGE, index, 0))
    bots = pimc_bots(index, CONFIG)
    choices: list[int] = []
    while not bola.is_over:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        action = bots[seat].act(obs, legal)
        if obs.phase is Phase.PLAY and len(legal) > 1:
            choices.append(action - PLAY_BASE)
        bola.step(action)
    return choices


def test_recording_does_not_change_pimcs_play():
    records = record_game(0, CONFIG)
    assert records.choice.tolist() == plain_pimc_choices(0)


def test_every_row_has_legal_cards_values_and_the_choice_among_the_best():
    records = record_game(1, CONFIG)
    n = len(records)
    assert n > 500
    assert records.features.shape == (n, NUM_FEATURES)
    assert records.features.dtype == np.float16
    rows = np.arange(n)
    assert records.legal[rows, records.choice].all()
    assert (records.legal.sum(1) > 1).all()
    assert not records.values[~records.legal].any()
    best = np.where(records.legal, records.values, -np.inf).max(1)
    assert np.allclose(records.values[rows, records.choice], best)
    assert (records.game == 1).all()
    assert records.reward.std() > 0  # every row got its round's reward


def test_records_concatenate_and_round_trip_through_a_file(tmp_path: Path):
    parts = [record_game(0, CONFIG), record_game(1, CONFIG)]
    records = concat_records(parts)
    assert len(records) == len(parts[0]) + len(parts[1])
    path = tmp_path / "pimc.npz"
    save_records(str(path), records)
    loaded = load_records(str(path))
    for field in ("features", "legal", "choice", "values", "reward", "game"):
        assert np.array_equal(getattr(loaded, field), getattr(records, field))
