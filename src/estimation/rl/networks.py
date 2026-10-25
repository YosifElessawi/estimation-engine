"""BiddingNet and CardPlayNet (SPEC §6 M4): 4-layer MLPs of about 490K parameters each.

- **BiddingNet** predicts the expected round reward of each of the 68 non-card actions (No Dash
  Call, Dash Call, the 50 bids, With, Pass and the 14 estimates; super calls use the bid ids).
- **CardPlayNet** has a critic, the expected reward of the position (deal luck), and an
  advantage head, how much better or worse each of the 52 cards is than that (skill). It plays
  the legal card with the highest advantage.

Rewards are predicted in units of ``REWARD_SCALE`` points.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from estimation.engine.actions import PLAY_BASE

from .encoder import ENCODER_VERSION, NUM_FEATURES

BID_ACTIONS = PLAY_BASE  # action ids 0-67: everything but card play
CARDS = 52
HIDDEN = 384
REWARD_SCALE = 50.0
CHECKPOINT_FORMAT = 1


def _mlp(outputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(NUM_FEATURES, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs),
    )


class BiddingNet(nn.Module):
    """Expected reward of each non-card action, in ``REWARD_SCALE`` units."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.body = _mlp(BID_ACTIONS, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.body(x))


class CardPlayNet(nn.Module):
    """The critic ``V(s)`` and the advantage ``A(s, card)`` of each card, in ``REWARD_SCALE``
    units: ``Q(s, card) = V(s) + A(s, card)``."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.body = _mlp(1 + CARDS, hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = cast(torch.Tensor, self.body(x))
        return out[:, 0], out[:, 1:]


def parameter_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


@dataclass(slots=True)
class Policy:
    """A saved bot: CardPlayNet, and BiddingNet unless it was never trained, with the
    checkpoint's metadata. Runs on the CPU in inference mode."""

    card: CardPlayNet
    bidding: BiddingNet | None
    meta: dict[str, Any]

    def card_values(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """The advantage of each of the 52 cards for a batch of encoded positions."""
        with torch.inference_mode():
            _, advantage = self.card(torch.as_tensor(features))
        return advantage.numpy()

    def bid_values(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """The value of each of the 68 non-card actions for a batch of encoded positions."""
        if self.bidding is None:
            raise ValueError("this checkpoint has no trained BiddingNet")
        with torch.inference_mode():
            values = self.bidding(torch.as_tensor(features))
        return values.numpy()


def state_dict_of(net: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of ``net``'s weights."""
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def save_checkpoint(
    path: Path,
    card: CardPlayNet,
    bidding: BiddingNet | None,
    meta: Mapping[str, Any],
) -> None:
    """Save a bot: both nets' weights, the encoder it reads and JSON-like ``meta``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "encoder": ENCODER_VERSION,
        "num_features": NUM_FEATURES,
        "hidden": card.hidden,
        "card": state_dict_of(card),
        "bidding": None if bidding is None else state_dict_of(bidding),
        "meta": dict(meta),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_policy(path: Path) -> Policy:
    """Load a checkpoint saved by ``save_checkpoint`` for play on the CPU."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an M4 checkpoint (format {payload.get('format')})")
    if payload["encoder"] != ENCODER_VERSION or payload["num_features"] != NUM_FEATURES:
        raise ValueError(
            f"{path} was trained on encoder version {payload['encoder']} with "
            f"{payload['num_features']} features; this code has version {ENCODER_VERSION} with "
            f"{NUM_FEATURES}"
        )
    hidden = int(payload["hidden"])
    card = CardPlayNet(hidden)
    card.load_state_dict(payload["card"])
    card.eval()
    bidding = None
    if payload["bidding"] is not None:
        bidding = BiddingNet(hidden)
        bidding.load_state_dict(payload["bidding"])
        bidding.eval()
    return Policy(card, bidding, dict(payload["meta"]))
