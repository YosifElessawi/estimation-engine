"""CardPlayNet's warm start: imitating recorded PIMC games (SPEC §6 M4)."""

import numpy as np
import pytest
import torch

from estimation.rl.config import config_from_dict
from estimation.rl.imitation import holdout_mask, train_imitation
from estimation.rl.record import CardRecords, concat_records, record_game

CONFIG = config_from_dict(
    {
        "seed": 4,
        "record": {"games": 3, "pimc_samples": 2},
        "imitation": {"epochs": 2, "batch": 256, "holdout": 0.34},
        "selfplay": {"batch": 128, "epochs": 1, "card_anchor_batch": 64},
        "nets": {"hidden": 32, "device": "cpu"},
    }
)


@pytest.fixture(scope="module")
def records() -> CardRecords:
    return concat_records([record_game(index, CONFIG) for index in range(3)])


def test_whole_games_are_held_out():
    games = np.array([0, 0, 1, 2, 19, 19, 20, 39])
    assert holdout_mask(games, 0.05).tolist() == [False] * 4 + [True, True, False, True]
    assert not holdout_mask(games, 0.0).any()


def test_imitation_is_deterministic_and_reports_agreement_on_held_out_games(records: CardRecords):
    one, report = train_imitation(records, CONFIG, torch.device("cpu"), log=False)
    two, _ = train_imitation(records, CONFIG, torch.device("cpu"), log=False)
    for a, b in zip(one.parameters(), two.parameters(), strict=True):
        assert torch.equal(a, b)
    assert report.holdout_rows == int(holdout_mask(records.game, 0.34).sum()) > 0
    assert report.train_rows + report.holdout_rows == len(records)
    assert 0.0 <= report.choice_agreement <= report.best_agreement <= 1.0
    assert report.value_error > 0


def test_imitation_learns_pimcs_choices(records: CardRecords):
    # Without relabelling, enough epochs fit the training games' choices closely.
    config = config_from_dict(
        {
            "seed": 4,
            "nets": {"hidden": 64, "device": "cpu"},
            "imitation": {"epochs": 40, "holdout": 0.0, "augment": False, "lr": 2e-3},
        }
    )
    net, _ = train_imitation(records, config, torch.device("cpu"), log=False)
    rows = np.arange(len(records))
    with torch.no_grad():
        _, advantage = net(torch.as_tensor(records.features.astype(np.float32)))
    picks = np.where(records.legal, advantage.numpy(), -np.inf).argmax(1)
    agreement = (picks == records.choice[rows]).mean()
    chance = (1 / records.legal.sum(1)).mean()  # a random legal card
    assert agreement > 0.8 > 2 * chance
