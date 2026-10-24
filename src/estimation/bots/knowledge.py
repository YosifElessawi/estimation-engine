"""What a seat knows during card play (SPEC §6 M3): every card played and who played it, and the
suits each seat has shown out of. It all comes from the seat's own Observation."""

from dataclasses import dataclass

from estimation.engine import Observation
from estimation.engine.cards import FULL_DECK


@dataclass(frozen=True, slots=True)
class Knowledge:
    seat: int
    hand: int  # own remaining cards
    played: int  # every card played so far, the current trick included
    played_by: tuple[int, int, int, int]  # the cards each seat has played
    voids: tuple[int, int, int, int]  # per seat, a 4-bit mask of the suits it has none of
    sizes: tuple[int, int, int, int]  # cards left in each hand

    @property
    def unseen(self) -> int:
        """The cards in the other three hands."""
        return FULL_DECK & ~self.played & ~self.hand


def read_knowledge(obs: Observation) -> Knowledge:
    """Played cards, voids and hand sizes from ``obs``.

    A seat that did not follow the led suit has none of it left (PLY-2), so it is void there.
    """
    played_by = [0, 0, 0, 0]
    voids = [0, 0, 0, 0]
    for record in obs.tricks:
        _note(record.leader, record.cards, played_by, voids)
    if obs.trick:
        _note(obs.trick[0][0], [card for _, card in obs.trick], played_by, voids)
    sizes = tuple(13 - mask.bit_count() for mask in played_by)
    return Knowledge(
        seat=obs.seat,
        hand=obs.hand,
        played=played_by[0] | played_by[1] | played_by[2] | played_by[3],
        played_by=(played_by[0], played_by[1], played_by[2], played_by[3]),
        voids=(voids[0], voids[1], voids[2], voids[3]),
        sizes=(sizes[0], sizes[1], sizes[2], sizes[3]),
    )


def _note(
    leader: int, cards: tuple[int, ...] | list[int], played_by: list[int], voids: list[int]
) -> None:
    led = cards[0] // 13
    for i, card in enumerate(cards):
        seat = (leader + i) & 3
        played_by[seat] |= 1 << card
        if card // 13 != led:
            voids[seat] |= 1 << led
