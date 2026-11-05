"""BiddingNet, CardPlayNet and checkpoints (SPEC §6 M4)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from estimation.rl.encoder import NUM_FEATURES
from estimation.rl.networks import (
    BID_ACTIONS,
    CARDS,
    GUESS_SEATS,
    BiddingNet,
    CardPlayNet,
    GuessNet,
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


def test_guess_net_guesses_which_other_seat_holds_each_card():
    logits = GuessNet(32)(torch.zeros(5, NUM_FEATURES))
    assert logits.shape == (5, CARDS, GUESS_SEATS) == (5, 52, 3)


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


def test_a_checkpoint_keeps_its_guess_net_and_gives_guesses_as_chances(tmp_path: Path):
    torch.manual_seed(5)
    card, guess = CardPlayNet(32), GuessNet(32)
    path = tmp_path / "g.pt"
    save_checkpoint(path, card, BiddingNet(32), {}, guess)
    policy = load_policy(path)
    assert policy.guess is not None
    x = np.random.default_rng(1).random((3, NUM_FEATURES), dtype=np.float32)
    guesses = policy.card_guesses(x)
    with torch.no_grad():
        expected = torch.softmax(guess(torch.from_numpy(x)), dim=2).numpy()
    assert np.allclose(guesses, expected)
    assert np.allclose(guesses.sum(axis=2), 1.0)


def test_a_checkpoint_without_a_guess_net_says_so(tmp_path: Path):
    policy = load_policy(make_checkpoint(tmp_path / "plain.pt"))
    assert policy.guess is None
    with pytest.raises(ValueError, match="no trained GuessNet"):
        policy.card_guesses(np.zeros((1, NUM_FEATURES), dtype=np.float32))


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


def as_version(path: Path, version: int, features: int) -> None:
    """Rewrite a checkpoint as an older encoder saved it: version 1 had 452 inputs (before the
    void announcements, TBL-7), version 2 had 456 (before each seat's calls by trump, AUC-4)."""
    payload = torch.load(path, weights_only=True)
    for net in ("card", "bidding", "guess"):
        if payload.get(net) is not None:
            payload[net]["trunk.0.weight"] = payload[net]["trunk.0.weight"][:, :features]
    payload["encoder"], payload["num_features"] = version, features
    torch.save(payload, path)


@pytest.mark.parametrize(("version", "features"), [(1, 452), (2, 456)])
def test_an_older_checkpoint_loads_and_ignores_the_new_inputs(
    tmp_path: Path, version: int, features: int
):
    path = make_checkpoint(tmp_path / f"v{version}.pt")
    as_version(path, version, features)
    policy = load_policy(path)
    assert policy.meta["upgraded_from_encoder"] == version
    assert policy.bidding is not None
    first = policy.card.state_dict()["trunk.0.weight"]
    assert first.shape[1] == NUM_FEATURES
    rng = np.random.default_rng(0)
    x = rng.random((5, NUM_FEATURES), dtype=np.float32)
    y = x.copy()
    y[:, features:] = 1.0 - y[:, features:]  # any values at all in the appended inputs
    assert np.allclose(policy.bid_values(x), policy.bid_values(y))
    assert np.allclose(policy.card_values(x), policy.card_values(y))
    assert not first[:, features:].any()  # zero weights, learned from there


def test_a_checkpoint_whose_inputs_cannot_be_widened_is_refused(tmp_path: Path):
    path = make_checkpoint(tmp_path / "odd.pt")
    payload = torch.load(path, weights_only=True)
    payload["encoder"], payload["num_features"] = 1, 400
    torch.save(payload, path)
    with pytest.raises(ValueError, match="encoder version 1 with 400 features"):
        load_policy(path)


def test_an_older_checkpoints_guess_net_is_widened_too(tmp_path: Path):
    torch.manual_seed(6)
    path = tmp_path / "v2-guess.pt"
    save_checkpoint(path, CardPlayNet(32), BiddingNet(32), {}, GuessNet(32))
    as_version(path, 2, 456)
    policy = load_policy(path)
    assert policy.guess is not None
    assert policy.guess.state_dict()["trunk.0.weight"].shape[1] == NUM_FEATURES
