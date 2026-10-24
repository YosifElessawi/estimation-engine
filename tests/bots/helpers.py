"""Shared helpers for bot tests: deals with a chosen hand, and scripted rounds."""

from estimation.engine import Round, mask_of, parse_hand


def deal_with(seat: int, text: str) -> tuple[int, int, int, int]:
    """A deal where ``seat`` holds ``text`` and the other seats share the rest in card order."""
    mine = parse_hand(text)
    rest = [card for card in range(52) if not mine >> card & 1]
    hands = [0, 0, 0, 0]
    hands[seat] = mine
    for i in range(3):
        hands[(seat + 1 + i) % 4] = mask_of(rest[13 * i : 13 * i + 13])
    return hands[0], hands[1], hands[2], hands[3]


def round_with(
    seat: int, text: str, *actions: int, round_no: int = 1, fixed_trump: int | None = None
) -> Round:
    """Round ``round_no`` dealt by ``deal_with(seat, text)``, after the given actions."""
    rnd = Round.new(round_no, deal_with(seat, text), fixed_trump)
    for action in actions:
        rnd.step(action)
    return rnd
