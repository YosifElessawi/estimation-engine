"""BiddingNet, CardPlayNet and checkpoints (SPEC §6 M4)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from estimation.rl.encoder import NUM_FEATURES
from estimation.rl.networks import (
    BID_ACTIONS,
    CARDS,
    BiddingNet,
    CardPlayNet,
    load_policy,
    parameter_count,
    save_checkpoint,
)
from tests.rl.helpers import make_checkpoint


def test_both_nets_are_mlps_of_about_500k_parameters():
    for net in (BiddingNet(), CardPlayNet()):
        trunk = [m for m in net.trunk.modules() if isinstance(m, torch.nn.Linear)]
        assert len(trunk) == 3  # three hidden layers, then the action outputs
        assert trunk[0].in_features == NUM_FEATURES
        assert 400_000 <= parameter_count(net) <= 600_000


def test_bidding_values_every_non_card_action_and_card_play_has_critic_and_advantages():
    x = torch.zeros(5, NUM_FEATURES)
    assert BiddingNet()(x).shape == (5, BID_ACTIONS) == (5, 68)
    value, advantage = CardPlayNet()(x)
    assert value.shape == (5,)
    assert advantage.shape == (5, CARDS) == (5, 52)


def test_a_checkpoint_round_trips(tmp_path: Path):
    path = make_checkpoint(tmp_path / "a.pt", seed=3)
    policy = load_policy(path)
    assert policy.meta == {"name": "a"}
    torch.manual_seed(3)
    card, bidding = CardPlayNet(32), BiddingNet(32)
    x = np.random.default_rng(0).random((4, NUM_FEATURES), dtype=np.float32)
    with torch.no_grad():
        expected_cards = card(torch.from_numpy(x))[1].numpy()
        expected_bids = bidding(torch.from_numpy(x)).numpy()
    assert np.allclose(policy.card_values(x), expected_cards)
    assert np.allclose(policy.bid_values(x), expected_bids)


def test_a_checkpoint_without_a_bidding_net_says_so(tmp_path: Path):
    policy = load_policy(make_checkpoint(tmp_path / "card-only.pt", bidding=False))
    assert policy.bidding is None
    with pytest.raises(ValueError, match="no trained BiddingNet"):
        policy.bid_values(np.zeros((1, NUM_FEATURES), dtype=np.float32))


def test_a_checkpoint_for_another_encoder_is_refused(tmp_path: Path):
    path = tmp_path / "old.pt"
    save_checkpoint(path, CardPlayNet(16), None, {})
    payload = torch.load(path, weights_only=True)
    payload["encoder"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="encoder version"):
        load_policy(path)


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path):
    path = tmp_path / "junk.pt"
    torch.save({"format": 99}, path)
    with pytest.raises(ValueError, match="not an M4 checkpoint"):
        load_policy(path)


def as_version_1(path: Path) -> None:
    """Rewrite a checkpoint as encoder version 1 saved it: 452 inputs, before the void
    announcements (TBL-7) were appended."""
    payload = torch.load(path, weights_only=True)
    for net in ("card", "bidding"):
        if payload[net] is not None:
            payload[net]["trunk.0.weight"] = payload[net]["trunk.0.weight"][:, :452]
    payload["encoder"], payload["num_features"] = 1, 452
    torch.save(payload, path)


def test_a_version_1_checkpoint_loads_and_ignores_the_new_inputs(tmp_path: Path):
    path = make_checkpoint(tmp_path / "v1.pt")
    as_version_1(path)
    policy = load_policy(path)
    assert policy.meta["upgraded_from_encoder"] == 1
    assert policy.bidding is not None
    first = policy.card.state_dict()["trunk.0.weight"]
    assert first.shape[1] == NUM_FEATURES
    rng = np.random.default_rng(0)
    x = rng.random((5, NUM_FEATURES), dtype=np.float32)
    y = x.copy()
    y[:, 452:] = 1.0 - y[:, 452:]  # any announcement values at all
    assert np.allclose(policy.bid_values(x), policy.bid_values(y))
    assert np.allclose(policy.card_values(x), policy.card_values(y))
    assert not first[:, 452:].any()  # zero weights, learned from there


def test_a_checkpoint_whose_inputs_cannot_be_widened_is_refused(tmp_path: Path):
    path = make_checkpoint(tmp_path / "odd.pt")
    payload = torch.load(path, weights_only=True)
    payload["encoder"], payload["num_features"] = 1, 400
    torch.save(payload, path)
    with pytest.raises(ValueError, match="encoder version 1 with 400 features"):
        load_policy(path)
