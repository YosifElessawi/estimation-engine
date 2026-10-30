"""Shared helpers for engine tests: short action names and scripted rounds."""

from collections.abc import Iterable

from estimation.engine.actions import (
    DASH_CALL,
    NO_DASH_CALL,
    bid_action,
    estimate_action,
    is_bid,
    is_estimate,
    play_action,
)
from estimation.engine.cards import CLUBS, SANS, deal, parse_card
from estimation.engine.config import DEFAULT_CONFIG, RulesConfig
from estimation.engine.round import Round

NO = NO_DASH_CALL
DASH = DASH_CALL
NO_DASH_ROUND = (NO, NO, NO, NO)


def bid(tricks: int, trump: int) -> int:
    return bid_action(tricks, trump)


def est(k: int) -> int:
    return estimate_action(k)


def play(text: str) -> int:
    return play_action(parse_card(text))


# The auction's With rules before 2026-10-30: only the latest call counts, and a With player
# keeps its number (AUC-4, EST-2 with with_any_call and with_choose off).
LATEST_CALL_ONLY = RulesConfig(with_any_call=False, with_choose=False)


def new_round(
    round_no: int = 1,
    fixed_trump: int | None = None,
    seed: int = 0,
    config: RulesConfig = DEFAULT_CONFIG,
) -> Round:
    return Round.new(round_no, deal(seed, round_no), fixed_trump, config=config)


def run(rnd: Round, *actions: int) -> Round:
    for action in actions:
        rnd.step(action)
    return rnd


def legal_estimates(rnd: Round) -> list[int]:
    return [a - estimate_action(0) for a in rnd.legal_actions() if is_estimate(a)]


def legal_bids(rnd: Round) -> list[int]:
    return [a for a in rnd.legal_actions() if is_bid(a)]


def build_play(
    trump: int, leader: int, targets: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int, int], list[int]]:
    """Hands, and a legal card sequence from them, giving each seat exactly ``targets`` tricks.

    Tricks 1-12 each use four cards of one suit, so following suit is always possible and the
    highest card wins. Each suit's 2 is left over for trick 13, which the seat with the most
    tricks leads with the trump 2 (or the Clubs 2 in Sans), so it wins with the only trump.
    """
    assert sum(targets) == 13
    closer = max(range(4), key=lambda seat: targets[seat])
    counts = list(targets)
    counts[closer] -= 2
    winners = [seat for seat in range(4) for _ in range(counts[seat])] + [closer, closer]

    hands = [0, 0, 0, 0]
    plays: list[int] = []
    lead = leader
    for trick in range(12):
        suit, group = divmod(trick, 3)
        ranks = [1 + 4 * group + i for i in range(4)]
        winner = winners[trick]
        assigned = {winner: ranks[3]}
        assigned.update(zip([s for s in range(4) if s != winner], ranks[:3], strict=True))
        for k in range(4):
            seat = (lead + k) % 4
            card = suit * 13 + assigned[seat]
            plays.append(card)
            hands[seat] |= 1 << card
        lead = winner

    lead_suit = CLUBS if trump == SANS else trump
    last = {closer: lead_suit * 13}
    last.update(
        zip(
            [s for s in range(4) if s != closer],
            [suit * 13 for suit in range(4) if suit != lead_suit],
            strict=True,
        )
    )
    for k in range(4):
        seat = (closer + k) % 4
        plays.append(last[seat])
        hands[seat] |= 1 << last[seat]
    return (hands[0], hands[1], hands[2], hands[3]), plays


def play_cards(rnd: Round, cards: Iterable[int]) -> Round:
    return run(rnd, *(play_action(card) for card in cards))
