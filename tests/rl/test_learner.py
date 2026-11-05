"""The Deep Monte Carlo update (SPEC §6 M4)."""

import numpy as np
import pytest
import torch

from estimation.rl.config import config_from_dict
from estimation.rl.encoder import NUM_FEATURES
from estimation.rl.guesses import NO_GUESS
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
    return Samples(features, chosen, rewards, np.full((n, 52), NO_GUESS, np.int8))


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


def test_card_updates_can_skip_the_suit_relabelling():
    config = config_from_dict(
        {"seed": 1, "selfplay": {"batch": 64, "epochs": 1, "augment_cards": False}}
    )
    torch.manual_seed(0)
    plain = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"))
    assert plain.augmenter is None
    report = plain.update(samples(128, 4, 0), samples(128, 3, 1), 0, train_card=True)
    assert report.card_steps == 2
    assert learner().augmenter is not None  # on by default


def _step_size(config_changes: dict[str, object], iteration: int) -> float:
    """How far one update at ``iteration`` moves BiddingNet's weights."""
    config = config_from_dict(
        {"seed": 1, "selfplay": {"batch": 512, "epochs": 1, **config_changes}}
    )
    torch.manual_seed(0)
    model = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"))
    before = [p.detach().clone() for p in model.bidding.parameters()]
    model.update(samples(512, 4, 0), samples(512, 3, 1), iteration, train_card=False)
    moved = [
        (p.detach() - b).norm() ** 2
        for p, b in zip(model.bidding.parameters(), before, strict=True)
    ]
    return float(sum(moved) ** 0.5)


def test_learning_rates_and_the_anchor_stay_fixed_by_default():
    model = learner()
    cfg = CONFIG.selfplay
    for iteration in (0, cfg.iterations - 1):
        report = model.update(samples(64, 4, 0), samples(64, 3, 1), iteration, train_card=True)
        assert report.lr_factor == 1.0
        assert report.anchor_weight == cfg.card_anchor_weight
        assert model.bid_optimizer.param_groups[0]["lr"] == cfg.bid_lr
        assert model.card_optimizer.param_groups[0]["lr"] == cfg.card_lr


def test_learning_rates_and_the_anchor_decay_linearly_over_the_run():
    changes = {"iterations": 11, "lr_end_factor": 0.1, "card_anchor_end_factor": 0.25}
    config = config_from_dict({"seed": 1, "selfplay": {"batch": 64, "epochs": 1, **changes}})
    torch.manual_seed(0)
    model = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"))
    cfg = config.selfplay
    for iteration, lr_factor, anchor in ((0, 1.0, 1.0), (5, 0.55, 0.625), (10, 0.1, 0.25)):
        report = model.update(samples(64, 4, 0), samples(64, 3, 1), iteration, train_card=True)
        assert report.lr_factor == pytest.approx(lr_factor)
        assert report.anchor_weight == pytest.approx(anchor * cfg.card_anchor_weight)
        assert model.card_optimizer.param_groups[0]["lr"] == pytest.approx(lr_factor * cfg.card_lr)
        assert model.bid_optimizer.param_groups[0]["lr"] == pytest.approx(lr_factor * cfg.bid_lr)
    # Adam's first step moves the weights in proportion to the learning rate.
    ratio = _step_size(changes, 10) / _step_size({"iterations": 11}, 10)
    assert ratio == pytest.approx(0.1, rel=0.05)


def guess_samples(n: int, seed: int) -> Samples:
    """Positions where the next seat holds every Club, the seat after it every Diamond and the
    last one the rest, and the first input says whether the Hearts went to the next seat."""
    rng = np.random.default_rng(seed)
    base = samples(n, 3, seed)
    flipped = rng.random(n) < 0.5
    features = base.features.copy()
    features[:, 0] = flipped
    labels = np.zeros((n, 52), np.int8)
    labels[:, 13:26] = 1
    labels[:, 26:52] = 2
    labels[flipped, 26:39] = 0
    labels[:, 0] = NO_GUESS  # a card the acting seat can see
    return Samples(features, base.actions, base.rewards, labels)


def test_with_guess_lr_the_guess_net_learns_who_holds_the_cards():
    config = config_from_dict({"seed": 1, "selfplay": {"batch": 64, "epochs": 1, "guess_lr": 1e-3}})
    torch.manual_seed(0)
    model = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"))
    assert model.guess is not None
    data = guess_samples(512, 4)
    first = model.update(data, data, 0, train_card=True)
    last = first
    for iteration in range(1, 30):
        last = model.update(data, data, iteration, train_card=True)
    assert first.guess is not None
    assert last.guess is not None
    assert last.guess.loss < first.guess.loss / 3
    assert last.guess.accuracy > 0.95
    assert last.guess.blind == pytest.approx(1 / 3, abs=0.01)  # nobody has shown out
    state = model.state()
    again = Learner(CardPlayNet(32), BiddingNet(32), config, torch.device("cpu"))
    again.load_state(state)
    assert again.guess is not None
    assert torch.equal(again.guess.out.weight, model.guess.out.weight)  # resumes exactly


def test_without_guess_lr_there_is_no_guess_net():
    model = learner()
    assert model.guess is None
    report = model.update(samples(128, 4, 0), samples(128, 3, 1), 0, train_card=True)
    assert report.guess is None
    assert "guess" not in model.state()
