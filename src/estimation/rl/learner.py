"""The Deep Monte Carlo update (SPEC §6 M4).

After each iteration's games, every recorded decision's value regresses onto the reward of the
round it was made in (Huber loss, rewards in units of ``REWARD_SCALE`` points):

- BiddingNet: the value of the action taken.
- CardPlayNet: the critic ``V(s)`` onto the reward, and ``V(s) + A(s, card)`` onto the reward
  with the critic held fixed, so the advantage learns how much better or worse the card was
  than the position's expected reward. Card-play batches relabel suits at random, as in the
  warm start.

Each card-play step also takes a batch of the recorded PIMC decisions and adds the warm start's
imitation loss (``card_anchor_weight``). Without that anchor, a pilot's card play lost about 40
points of RSA against HeuristicBots in 36 self-play iterations: it forgot PIMC-like play while
training on the early, chaotic rounds of a bidder that starts from scratch.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import SELFPLAY_STAGE, TrainConfig, derive_seed
from .imitation import ImitationData, imitation_losses
from .networks import REWARD_SCALE, BiddingNet, CardPlayNet, state_dict_of
from .selfplay import NetWeights, Samples
from .symmetry import SuitAugmenter
from .torch_typing import backward, seeded_generator, step

BID_SHUFFLE = 1
CARD_SHUFFLE = 2
CARD_RELABEL = 3
ANCHOR_ROWS = 4


@dataclass(frozen=True, slots=True)
class UpdateReport:
    bid_loss: float
    card_value_loss: float
    card_q_loss: float
    anchor_loss: float
    bid_steps: int
    card_steps: int


class Learner:
    """Both nets and their optimizers on the training device."""

    def __init__(
        self,
        card: CardPlayNet,
        bidding: BiddingNet,
        config: TrainConfig,
        device: torch.device,
        anchor: ImitationData | None = None,
    ) -> None:
        cfg = config.selfplay
        self.config = config
        self.device = device
        self.card = card.to(device)
        self.bidding = bidding.to(device)
        self.card_optimizer = torch.optim.Adam(self.card.parameters(), lr=cfg.card_lr)
        self.bid_optimizer = torch.optim.Adam(self.bidding.parameters(), lr=cfg.bid_lr)
        self.augmenter = SuitAugmenter(device) if cfg.augment_cards else None
        self.anchor = anchor if cfg.card_anchor_weight > 0 else None

    def weights(self) -> NetWeights:
        """The current nets as CPU state dicts, for the actors."""
        return NetWeights(
            hidden=self.card.hidden,
            card=state_dict_of(self.card),
            bidding=state_dict_of(self.bidding),
        )

    def state(self) -> dict[str, Any]:
        """Everything needed to resume training exactly: weights and optimizer states."""
        return {
            "card": state_dict_of(self.card),
            "bidding": state_dict_of(self.bidding),
            "card_optimizer": copy.deepcopy(self.card_optimizer.state_dict()),
            "bid_optimizer": copy.deepcopy(self.bid_optimizer.state_dict()),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.card.load_state_dict(state["card"])
        self.bidding.load_state_dict(state["bidding"])
        self.card_optimizer.load_state_dict(state["card_optimizer"])
        self.bid_optimizer.load_state_dict(state["bid_optimizer"])

    def update(self, bid: Samples, card: Samples, iteration: int, train_card: bool) -> UpdateReport:
        seed = self.config.seed
        bid_loss, bid_steps = 0.0, 0
        if len(bid):
            rng = np.random.default_rng(derive_seed(seed, SELFPLAY_STAGE, iteration, BID_SHUFFLE))
            bid_loss, bid_steps = self._fit_bidding(bid, rng)
        value_loss = q_loss = anchor_loss = 0.0
        card_steps = 0
        if train_card and len(card):
            rng = np.random.default_rng(derive_seed(seed, SELFPLAY_STAGE, iteration, CARD_SHUFFLE))
            relabel = seeded_generator(derive_seed(seed, SELFPLAY_STAGE, iteration, CARD_RELABEL))
            anchor_rng = np.random.default_rng(
                derive_seed(seed, SELFPLAY_STAGE, iteration, ANCHOR_ROWS)
            )
            value_loss, q_loss, anchor_loss, card_steps = self._fit_card(
                card, rng, relabel, anchor_rng
            )
        return UpdateReport(bid_loss, value_loss, q_loss, anchor_loss, bid_steps, card_steps)

    def _tensors(self, samples: Samples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        x = torch.as_tensor(samples.features).to(device)
        actions = torch.as_tensor(samples.actions.astype(np.int64)).to(device)
        rewards = torch.as_tensor(samples.rewards).to(device) / REWARD_SCALE
        return x, actions, rewards

    def _batches(self, n: int, rng: np.random.Generator) -> list[torch.Tensor]:
        cfg = self.config.selfplay
        batches: list[torch.Tensor] = []
        for _ in range(cfg.epochs):
            order = torch.as_tensor(rng.permutation(n)).to(self.device)
            batches += [order[i : i + cfg.batch] for i in range(0, n, cfg.batch)]
        return batches

    def _fit_bidding(self, samples: Samples, rng: np.random.Generator) -> tuple[float, int]:
        cfg = self.config.selfplay
        x, actions, rewards = self._tensors(samples)
        net, optimizer = self.bidding, self.bid_optimizer
        net.train()
        total = 0.0
        batches = self._batches(len(samples), rng)
        for idx in batches:
            q = net(x[idx].float()).gather(1, actions[idx, None])[:, 0]
            loss = nn.functional.huber_loss(q, rewards[idx], delta=cfg.huber)
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            step(optimizer)
            total += float(loss.detach())
        net.eval()
        return total / max(1, len(batches)), len(batches)

    def _fit_card(
        self,
        samples: Samples,
        rng: np.random.Generator,
        relabel: torch.Generator,
        anchor_rng: np.random.Generator,
    ) -> tuple[float, float, float, int]:
        cfg = self.config.selfplay
        imitation = self.config.imitation
        x, actions, rewards = self._tensors(samples)
        net, optimizer = self.card, self.card_optimizer
        net.train()
        value_total = q_total = anchor_total = 0.0
        batches = self._batches(len(samples), rng)
        for idx in batches:
            features = x[idx].float()
            cards = actions[idx]
            if self.augmenter is not None:
                perms = self.augmenter.draw(len(idx), relabel)
                features = self.augmenter.relabel_features(features, perms)
                cards = self.augmenter.relabel_cards(cards, perms)
            value, advantage = net(features)
            target = rewards[idx]
            q = value.detach() + advantage.gather(1, cards[:, None])[:, 0]
            value_loss = nn.functional.huber_loss(value, target, delta=cfg.huber)
            q_loss = nn.functional.huber_loss(q, target, delta=cfg.huber)
            loss = value_loss + q_loss
            if self.anchor is not None:
                rows = torch.as_tensor(
                    anchor_rng.integers(0, self.anchor.n, cfg.card_anchor_batch)
                ).to(self.device)
                batch = self.anchor.batch(rows, self.augmenter, relabel)
                adv_loss, choice_loss, _ = imitation_losses(net, batch, imitation.temperature)
                anchor = (
                    imitation.advantage_weight * adv_loss + imitation.choice_weight * choice_loss
                )
                loss = loss + cfg.card_anchor_weight * anchor
                anchor_total += float(anchor.detach())
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            step(optimizer)
            value_total += float(value_loss.detach())
            q_total += float(q_loss.detach())
        net.eval()
        steps = max(1, len(batches))
        return value_total / steps, q_total / steps, anchor_total / steps, len(batches)
