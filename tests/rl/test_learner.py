"""The Deep Monte Carlo update (SPEC §6 M4)."""

import numpy as np
import torch

from estimation.rl.config import config_from_dict
from estimation.rl.encoder import NUM_FEATURES
from estimation.rl.imitation import ImitationData, training_rows
from estimation.rl.learner import Learner
from estimation.rl.networks import REWARD_SCALE, BiddingNet, CardPlayNet
from estimation.rl.record import concat_records, record_game
from estimation.rl.selfplay import Samples

CONFIG = config_from_dict({"seed": 1, "selfplay": {"batch": 64, "epochs": 1}})


def samples(n: int, actions: int, seed: int) -> Samples:
    """Positions where the reward is +40 for action 0 and -40 for every other action."""
    rng = np.random.default_rng(seed)
    features = rng.random((n, NUM_FEATURES), dtype=np.float32).astype(np.float16)
    chosen = rng.integers(0, actions, n).astype(np.int16)
    rewards = np.where(chosen == 0, 40.0, -40.0).astype(np.float32)
    return Samples(features, chosen, rewards)


def learner(seed: int = 0) -> Learner:
    torch.manual_seed(seed)
    return Learner(CardPlayNet(32), BiddingNet(32), CONFIG, torch.device("cpu"))


def test_updates_fit_the_value_of_the_action_taken():
    model = learner()
    bid, card = samples(512, 4, 0), samples(512, 3, 1)
    first = last = model.update(bid, card, 0, train_card=True)
    for iteration in range(1, 30):
        last = model.update(bid, card, iteration, train_card=True)
    assert last.bid_loss < first.bid_loss / 4
    assert last.card_q_loss < first.card_q_loss / 4
    x = torch.as_tensor(bid.features[:64].astype(np.float32))
    with torch.no_grad():
        q = model.bidding(x)
        value, advantage = model.card(x)
    assert (q[:, 0] > q[:, 1]).float().mean() > 0.9
    assert (advantage[:, 0] > advantage[:, 1]).float().mean() > 0.9
    # The critic learns the position's mean reward: one action in three pays +40.
    assert abs(float(value.mean()) * REWARD_SCALE - (40 - 2 * 40) / 3) < 10


def test_card_play_can_stay_frozen():
    model = learner()
    before = {k: v.clone() for k, v in model.card.state_dict().items()}
    report = model.update(samples(128, 4, 0), samples(128, 3, 1), 0, train_card=False)
    assert report.card_steps == 0
    assert all(torch.equal(before[k], v) for k, v in model.card.state_dict().items())
    assert report.bid_steps == 2


def test_updates_are_deterministic():
    one, two = learner(3), learner(3)
    for model in (one, two):
        model.update(samples(256, 4, 0), samples(256, 3, 1), 5, train_card=True)
    for a, b in zip(one.card.parameters(), two.card.parameters(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(one.bidding.parameters(), two.bidding.parameters(), strict=True):
        assert torch.equal(a, b)


def test_state_restores_weights_and_optimizers():
    model = learner(4)
    model.update(samples(128, 4, 0), samples(128, 3, 1), 0, train_card=True)
    copy = learner(9)
    copy.load_state(model.state())
    for target in (model, copy):
        target.update(samples(128, 4, 2), samples(128, 3, 3), 1, train_card=True)
    for a, b in zip(model.card.parameters(), copy.card.parameters(), strict=True):
        assert torch.equal(a, b)


def test_self_play_card_updates_keep_the_imitation_loss_as_an_anchor():
    config = config_from_dict(
        {
            "seed": 4,
            "record": {"pimc_samples": 2},
            "selfplay": {"batch": 128, "epochs": 1, "card_anchor_batch": 64},
        }
    )
    records = concat_records([record_game(0, config)])
    anchor = ImitationData(records, training_rows(records, config), torch.device("cpu"))
    torch.manual_seed(0)
    anchored = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"), anchor)
    report = anchored.update(samples(256, 4, 0), samples(256, 3, 1), 0, train_card=True)
    assert report.anchor_loss > 0
    unanchored_config = config_from_dict({"selfplay": {"card_anchor_weight": 0.0}})
    free = Learner(CardPlayNet(32), BiddingNet(32), unanchored_config, torch.device("cpu"), anchor)
    assert free.anchor is None
    assert free.update(samples(256, 4, 0), samples(256, 3, 1), 0, train_card=True).anchor_loss == 0
