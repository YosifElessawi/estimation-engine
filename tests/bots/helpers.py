"""Shared helpers for bot tests: deals with a chosen hand, scripted rounds, and hand-built
card-play observations."""

from estimation.engine import (
    FIXED_ORDER,
    Observation,
    Phase,
    Round,
    TrickRecord,
    bid_action,
    mask_of,
    parse_hand,
)


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


def play_observation(
    *,
    hand: int,
    estimates: tuple[int, int, int, int],
    tricks_won: tuple[int, int, int, int],
    caller: int,
    trump: int,
    tricks: tuple[TrickRecord, ...] = (),
    seat: int = 0,
    leader: int = 0,
    last_estimator: int = 3,
) -> Observation:
    """A card-play Observation of an open round built by hand, for positions too fiddly to
    script through the engine. Only the fields bots read during card play matter."""
    total = sum(estimates)
    return Observation(
        seat=seat,
        round_no=1,
        phase=Phase.PLAY,
        current_player=seat,
        carry=1,
        scores=(0, 0, 0, 0),
        fixed_trump=None,
        fixed_left=FIXED_ORDER,
        replays=0,
        super_call=False,
        trump=trump,
        hand=hand,
        log=(),
        dash_callers=(),
        top_bid=bid_action(estimates[caller], trump),
        top_bidder=caller,
        passed=tuple(s for s in range(4) if s != caller),
        caller=caller,
        with_players=(),
        estimates=estimates,
        last_estimator=last_estimator,
        total=total,
        over=total > 13,
        risks=abs(total - 13) // 2,
        leader=leader,
        trick=(),
        tricks=tricks,
        tricks_won=tricks_won,
    )
