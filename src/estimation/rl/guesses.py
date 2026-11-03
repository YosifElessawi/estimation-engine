"""Training the guess head (``networks``): who holds each unseen card.

Training games know the real hands, so every recorded decision gets a label per card: which of
the other three seats holds it, counted from the acting seat (0 for the next seat
counterclockwise, 1 and 2 after it), or ``NO_GUESS`` for a card the acting seat can see (its own
hand, played, or on the table). The labels are a training target only: a playing bot never sees
them.

The head's accuracy is reported next to a blind guess's: an even guess among the seats that have
not shown out of the card's suit. That uses what the inputs show plainly, so the gap between the
two is what the net learned from the bidding, the play and the scores.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .encoder import OFFSETS, VOIDS

NO_GUESS = -1
UNSEEN = OFFSETS["unseen"]
_SUIT_OF_CARD = torch.arange(52) // 13


def guess_labels(
    features: npt.NDArray[np.float32], deal: Sequence[int], seat: int
) -> npt.NDArray[np.int8]:
    """The label of each card for ``seat``'s encoded position ``features``, from the round's
    ``deal`` (the four dealt hands): the holder of each unseen card, ``NO_GUESS`` elsewhere."""
    unseen = features[UNSEEN : UNSEEN + 52] > 0.5
    labels = np.full(52, NO_GUESS, np.int8)
    for r in (1, 2, 3):
        dealt = np.array([deal[(seat + r) & 3]], dtype="<u8")
        held = np.unpackbits(dealt.view(np.uint8), bitorder="little")[:52].astype(bool)
        labels[held & unseen] = r - 1
    return labels


def guess_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The mean cross-entropy over the labelled cards of ``logits`` (``(batch, 52, 3)``) against
    ``labels`` (``(batch, 52)``), the number of cards the head's top guess gets right, and the
    number of labelled cards."""
    flat = labels.reshape(-1).long()
    scores = logits.reshape(-1, logits.shape[-1])
    known = flat != NO_GUESS
    count = known.sum()
    if not bool(count):
        zero = scores.sum() * 0.0
        return zero, count, count
    loss = nn.functional.cross_entropy(scores, flat, ignore_index=NO_GUESS)
    right = ((scores.argmax(1) == flat) & known).sum()
    return loss, right, count


def blind_right(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The expected number of labelled cards an even guess gets right, among the seats that have
    not shown out of each card's suit (the encoder's ``voids`` block)."""
    batch = features.shape[0]
    voids = features[:, VOIDS : VOIDS + 12].reshape(batch, 3, 4)
    suits = _SUIT_OF_CARD.to(features.device)
    possible = (1.0 - voids[:, :, suits]).transpose(1, 2)  # (batch, 52, 3)
    known = labels != NO_GUESS
    chosen = possible.gather(2, labels.clamp(min=0).long()[..., None])[..., 0]
    chance = chosen / possible.sum(2).clamp(min=1.0)
    return (chance * known).sum()
