"""Small checkpoints for tests: random weights, or a last layer rigged to prefer one action."""

from pathlib import Path

import torch

from estimation.rl.networks import BiddingNet, CardPlayNet, save_checkpoint


def make_checkpoint(
    path: Path,
    *,
    seed: int = 0,
    hidden: int = 32,
    bidding: bool = True,
    favourite_card: int | None = None,
    favourite_action: int | None = None,
) -> Path:
    """Save a checkpoint with random weights. ``favourite_card`` (a card id) gets the highest
    advantage in every position and ``favourite_action`` (an action id 0-67) the highest
    bidding value, by zeroing the last layer and setting its bias."""
    torch.manual_seed(seed)
    card = CardPlayNet(hidden)
    bid_net = BiddingNet(hidden) if bidding else None
    with torch.no_grad():
        if favourite_card is not None:
            last = card.body[-1]
            assert isinstance(last, torch.nn.Linear)
            last.weight.zero_()
            last.bias.copy_(torch.linspace(0.0, -1.0, 53))
            last.bias[1 + favourite_card] = 5.0
        if favourite_action is not None and bid_net is not None:
            last = bid_net.body[-1]
            assert isinstance(last, torch.nn.Linear)
            last.weight.zero_()
            last.bias.copy_(torch.linspace(0.0, -1.0, 68))
            last.bias[favourite_action] = 5.0
    save_checkpoint(path, card, bid_net, {"name": path.stem})
    return path
