"""Small checkpoints for tests: random weights, or a last layer rigged to prefer one action."""

from pathlib import Path

import torch

from estimation.rl.networks import ActionHead, BiddingNet, CardPlayNet, save_checkpoint


def rig(head: ActionHead, favourite: int) -> None:
    """Make ``head`` value ``favourite`` highest, then lower ids, in every position."""
    head.out.weight.zero_()
    head.out.bias.zero_()
    head.per_action.weight.zero_()
    actions = head.per_action.bias.shape[0]
    head.per_action.bias.copy_(torch.linspace(0.0, -1.0, actions))
    head.per_action.bias[favourite] = 5.0


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
    bidding value, then lower ids, by silencing the shared action head and giving the
    per-action outputs fixed values."""
    torch.manual_seed(seed)
    card = CardPlayNet(hidden)
    bid_net = BiddingNet(hidden) if bidding else None
    with torch.no_grad():
        if favourite_card is not None:
            rig(card.head, favourite_card)
        if favourite_action is not None and bid_net is not None:
            rig(bid_net.head, favourite_action)
    save_checkpoint(path, card, bid_net, {"name": path.stem})
    return path
