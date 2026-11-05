"""Suit symmetry of card play, for data augmentation (SPEC §6 M4).

Once the trump is fixed, the card-play rules (PLY-1 to PLY-5) and the scoring (§3) never look at
which suit is which, only at the led suit and the trump. So relabelling the four suits, the trump
included, turns a card decision into an equally valid one with the cards relabelled the same way:
every recorded card decision stands for 24. Bidding is not symmetric (TBL-5 ranks the trumps),
so only card-play data is augmented.

A permutation ``perm`` sends suit ``s`` to suit ``perm[s]``; Sans stays Sans.
"""

import itertools

import numpy as np
import numpy.typing as npt
import torch

from .encoder import AUCTION, CALLS, CONTEXT, NUM_FEATURES, OFFSETS, SHAPE, TRICK, VOIDS

SUIT_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
_CARD_BLOCKS = ("hand", "played", "played", "played", "table", "unseen")


def card_destination(perm: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """The new id of each card id."""
    cards = np.arange(52)
    return np.array([perm[c // 13] * 13 + c % 13 for c in cards], np.int64)


def card_source(perm: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """For each new card id, the card id it came from."""
    source = np.zeros(52, np.int64)
    source[card_destination(perm)] = np.arange(52)
    return source


def feature_source(perm: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """For each feature of the relabelled position, the feature it is read from:
    ``relabelled = features[feature_source(perm)]``."""
    destination = np.arange(NUM_FEATURES)
    cards = card_destination(perm)
    # The six 52-card blocks are contiguous: hand, three opponents' plays, table, unseen.
    start = OFFSETS["hand"]
    for block in range(len(_CARD_BLOCKS)):
        base = start + 52 * block
        destination[base : base + 52] = base + cards
    suits = np.array(perm, np.int64)
    trumps = np.array([*perm, 4], np.int64)  # Sans stays Sans
    for base in (VOIDS, VOIDS + 4, VOIDS + 8, TRICK, SHAPE):
        destination[base : base + 4] = base + suits
    calls = tuple(CALLS + 5 * seat for seat in range(4))
    for base in (CONTEXT + 5, CONTEXT + 12, AUCTION + 2, SHAPE + 4, *calls):
        destination[base : base + 5] = base + trumps
    source = np.zeros(NUM_FEATURES, np.int64)
    source[destination] = np.arange(NUM_FEATURES)
    return source


class SuitAugmenter:
    """Applies a random suit relabelling per row to batches of card-play data on a device."""

    def __init__(self, device: torch.device) -> None:
        self.features = torch.as_tensor(
            np.stack([feature_source(p) for p in SUIT_PERMUTATIONS])
        ).to(device)
        self.card_source = torch.as_tensor(
            np.stack([card_source(p) for p in SUIT_PERMUTATIONS])
        ).to(device)
        self.card_destination = torch.as_tensor(
            np.stack([card_destination(p) for p in SUIT_PERMUTATIONS])
        ).to(device)

    def draw(self, rows: int, generator: torch.Generator) -> torch.Tensor:
        """A permutation index per row, from a CPU ``generator``."""
        draws = torch.randint(len(SUIT_PERMUTATIONS), (rows,), generator=generator)
        return draws.to(self.features.device)

    def relabel_features(self, x: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        return x.gather(1, self.features[perms])

    def relabel_card_rows(self, values: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        """Per-card rows (52 wide), such as legal masks or values, relabelled."""
        return values.gather(1, self.card_source[perms])

    def relabel_cards(self, cards: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        """Card ids relabelled."""
        return self.card_destination[perms].gather(1, cards[:, None])[:, 0]
