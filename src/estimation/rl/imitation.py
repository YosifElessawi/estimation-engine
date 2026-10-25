"""CardPlayNet's warm start (SPEC §6 M4): imitate PIMC's card choices from recorded games.

Three losses, all in units of ``REWARD_SCALE`` points:

- **advantage:** each legal card's advantage regresses onto PIMC's search value of that card
  minus the value of the card PIMC chose, so PIMC's choice sits at 0 and worse cards below it;
- **choice:** cross-entropy between the legal cards' advantages (divided by ``temperature``) and
  PIMC's choice, which breaks the many exact ties in search values the way PIMC does;
- **value:** the critic regresses onto the round reward the recorded seat actually got.

Each batch relabels the suits of every row at random (``symmetry``), so each recorded decision
stands for 24. Games are split, not decisions: a share of whole games is held out to measure how
often the net plays a card PIMC rates best.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import IMITATION_STAGE, TrainConfig, derive_seed
from .networks import REWARD_SCALE, CardPlayNet
from .record import CardRecords
from .symmetry import SuitAugmenter
from .torch_typing import backward, seed_torch, seeded_generator, step

NEG = -1e9


@dataclass(frozen=True, slots=True)
class ImitationReport:
    train_rows: int
    holdout_rows: int
    epochs: int
    seconds: float
    loss: float  # last epoch's mean training loss
    choice_agreement: float  # held-out share where the net plays exactly PIMC's card
    best_agreement: float  # held-out share where the net plays a card PIMC rates best
    value_error: float  # held-out critic RMSE in points


def holdout_mask(games: np.ndarray, share: float) -> np.ndarray:
    """Rows from the held-out games: every game whose index falls in the last ``share`` of each
    block of 1 / share games (game 19 of every 20 for 5%)."""
    if share <= 0:
        return np.zeros(len(games), np.bool_)
    period = max(2, round(1 / share))
    return (games % period) == period - 1


@dataclass(frozen=True, slots=True)
class ImitationBatch:
    x: torch.Tensor  # encoded positions, float32
    legal: torch.Tensor  # (rows, 52) legal cards
    choice: torch.Tensor  # PIMC's card
    target: torch.Tensor  # (rows, 52) search value minus the chosen card's, in 50-point units
    reward: torch.Tensor  # the round reward, in 50-point units


class ImitationData:
    """Recorded PIMC decisions (some rows of the records) on the training device."""

    def __init__(self, records: CardRecords, rows: np.ndarray, device: torch.device) -> None:
        self.x = torch.as_tensor(records.features[rows]).to(device)
        legal = torch.as_tensor(records.legal[rows]).to(device)
        choice = torch.as_tensor(records.choice[rows].astype(np.int64)).to(device)
        values = torch.as_tensor(records.values[rows]).to(device)
        chosen = values.gather(1, choice[:, None])
        self.legal = legal
        self.choice = choice
        self.values = values
        self.target = torch.where(legal, (values - chosen) / REWARD_SCALE, 0.0)
        self.reward = torch.as_tensor(records.reward[rows]).to(device) / REWARD_SCALE
        self.n = len(rows)

    def batch(
        self, idx: torch.Tensor, augmenter: SuitAugmenter | None, relabel: torch.Generator
    ) -> ImitationBatch:
        """Rows ``idx``, each with its suits relabelled at random when there is an augmenter."""
        x = self.x[idx].float()
        legal = self.legal[idx]
        choice = self.choice[idx]
        target = self.target[idx]
        if augmenter is not None:
            perms = augmenter.draw(len(idx), relabel)
            x = augmenter.relabel_features(x, perms)
            legal = augmenter.relabel_card_rows(legal, perms)
            target = augmenter.relabel_card_rows(target, perms)
            choice = augmenter.relabel_cards(choice, perms)
        return ImitationBatch(x, legal, choice, target, self.reward[idx])


def imitation_losses(
    net: CardPlayNet, batch: ImitationBatch, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The advantage, choice and value losses of ``net`` on ``batch``."""
    value, advantage = net(batch.x)
    legal = batch.legal
    count = legal.sum(1).clamp(min=1)
    errors = nn.functional.smooth_l1_loss(advantage, batch.target, reduction="none")
    adv_loss = (errors * legal).sum(1).div(count).mean()
    logits = torch.where(legal, advantage / temperature, NEG)
    choice_loss = nn.functional.cross_entropy(logits, batch.choice)
    value_loss = nn.functional.smooth_l1_loss(value, batch.reward)
    return adv_loss, choice_loss, value_loss


def training_rows(records: CardRecords, config: TrainConfig) -> np.ndarray:
    """The rows imitation trains on: every game but the held-out ones."""
    return np.flatnonzero(~holdout_mask(records.game, config.imitation.holdout))


def train_imitation(
    records: CardRecords, config: TrainConfig, device: torch.device, log: bool = True
) -> tuple[CardPlayNet, ImitationReport]:
    """Train a fresh CardPlayNet on ``records``. Deterministic for a given config and device."""
    import time

    cfg = config.imitation
    start = time.perf_counter()
    held = holdout_mask(records.game, cfg.holdout)
    train_rows = training_rows(records, config)
    test_rows = np.flatnonzero(held)
    seed_torch(derive_seed(config.seed, IMITATION_STAGE, 0))
    net = CardPlayNet(config.nets.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    train = ImitationData(records, train_rows, device)
    steps_per_epoch = max(1, math.ceil(train.n / cfg.batch))
    total_steps = steps_per_epoch * cfg.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    )
    rng = np.random.default_rng(derive_seed(config.seed, IMITATION_STAGE, 1))
    augmenter = SuitAugmenter(device) if cfg.augment else None
    relabel = seeded_generator(derive_seed(config.seed, IMITATION_STAGE, 2))
    loss_mean = 0.0
    for epoch in range(cfg.epochs):
        net.train()
        order = torch.as_tensor(rng.permutation(train.n)).to(device)
        total = 0.0
        for batch_no in range(steps_per_epoch):
            idx = order[batch_no * cfg.batch : (batch_no + 1) * cfg.batch]
            batch = train.batch(idx, augmenter, relabel)
            adv_loss, choice_loss, value_loss = imitation_losses(net, batch, cfg.temperature)
            loss = (
                cfg.advantage_weight * adv_loss
                + cfg.choice_weight * choice_loss
                + cfg.value_weight * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            step(optimizer)
            schedule.step()
            total += float(loss.detach())
        loss_mean = total / steps_per_epoch
        if log:
            print(f"[imitation] epoch {epoch + 1}/{cfg.epochs}: loss {loss_mean:.4f}", flush=True)
    net.eval()
    choice_agreement = best_agreement = value_error = float("nan")
    if len(test_rows):
        test = ImitationData(records, test_rows, device)
        choice_agreement, best_agreement, value_error = evaluate(net, test)
    report = ImitationReport(
        train_rows=len(train_rows),
        holdout_rows=len(test_rows),
        epochs=cfg.epochs,
        seconds=time.perf_counter() - start,
        loss=loss_mean,
        choice_agreement=choice_agreement,
        best_agreement=best_agreement,
        value_error=value_error,
    )
    return net.cpu(), report


def evaluate(net: CardPlayNet, data: ImitationData) -> tuple[float, float, float]:
    """Exact agreement with PIMC's choice, share of plays PIMC rates best, critic RMSE."""
    with torch.no_grad():
        exact = best = 0
        squared = 0.0
        for begin in range(0, data.n, 4096):
            idx = torch.arange(begin, min(begin + 4096, data.n), device=data.x.device)
            value, advantage = net(data.x[idx].float())
            legal = data.legal[idx]
            pick = torch.where(legal, advantage, NEG).argmax(1)
            exact += int((pick == data.choice[idx]).sum())
            values = torch.where(legal, data.values[idx], NEG)
            top = values.max(1).values
            best += int((values.gather(1, pick[:, None])[:, 0] >= top - 1e-6).sum())
            squared += float(((value - data.reward[idx]) ** 2).sum())
    return exact / data.n, best / data.n, math.sqrt(squared / data.n) * REWARD_SCALE
