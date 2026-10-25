"""JSON views of a game for the browser: actions, the table, round results.

A table view is built from one seat's ``Observation``. A player sees only that seat's hand: the
builder never gets anything else unless the caller passes every seat's hand, which only a
spectator view does. Cards are card ids (SPEC §5.1) and seats are 0-3.
"""

from collections.abc import Sequence
from typing import TypedDict

from estimation.engine import (
    DASH_CALL,
    NO_DASH_CALL,
    PASS,
    WITH,
    Observation,
    Phase,
    RoundResult,
    bid_of,
    card_of,
    card_str,
    cards_of,
    estimate_of,
)
from estimation.engine.actions import is_bid, is_estimate, is_play
from estimation.engine.cards import TRUMP_NAMES


class ActionView(TypedDict):
    """One action with a readable label. ``kind`` is one of no_dash, dash, bid, super_call,
    with, pass, keep (Pass on the top bidder's final turn), estimate and play."""

    id: int
    kind: str
    label: str
    tricks: int | None
    trump: int | None
    card: int | None


class EventView(TypedDict):
    """A seat's action. ``trick_winner`` is set when the card completed a trick."""

    seat: int
    action: ActionView
    trick_winner: int | None


class BidView(TypedDict):
    tricks: int
    trump: int


class WithView(TypedDict):
    seat: int
    w: int
    trump: int


class PlayedCard(TypedDict):
    seat: int
    card: int


class TrickView(TypedDict):
    leader: int
    cards: list[PlayedCard]
    winner: int


class TableView(TypedDict):
    """The table as one viewer sees it. ``hands`` holds a seat's remaining cards only where the
    viewer may see them, ``None`` elsewhere; ``hand_sizes`` is public. ``events`` are the
    round's Dash Calls, auction calls and estimates so far, and ``calls`` each seat's latest of
    them. ``scores`` are the game scores before this round."""

    round_no: int
    phase: str
    current_player: int | None
    carry: int
    scores: list[int]
    fixed_trump: int | None
    fixed_left: list[int]
    replays: int
    super_call: bool
    trump: int | None
    dash_callers: list[int]
    top_bid: BidView | None
    top_bidder: int | None
    passed: list[int]
    caller: int | None
    with_players: list[WithView]
    estimates: list[int | None]
    last_estimator: int | None
    total: int | None
    over: bool | None
    risks: int | None
    hands: list[list[int] | None]
    hand_sizes: list[int]
    leader: int | None
    trick: list[PlayedCard]
    last_trick: TrickView | None
    tricks_played: int
    tricks_won: list[int]
    events: list[EventView]
    calls: list[ActionView | None]


class SeatResultView(TypedDict):
    estimate: int
    tricks: int
    won: bool
    dash: bool
    caller: bool
    with_player: bool
    super_call: bool
    last: bool
    risks: int
    base_points: int
    dash_points: int
    call_points: int
    risk_points: int
    only_points: int
    super_points: int
    points: int
    multiplier: int
    score: int


class RoundResultView(TypedDict):
    round_no: int
    skipped: bool
    caller: int | None
    trump: int | None
    fixed_trump: int | None
    trump_changed: bool
    total: int | None
    over: bool | None
    calls: int
    all_lose: bool
    carry: int
    multiplier: int
    next_carry: int
    scores: list[int]
    seats: list[SeatResultView]


PHASES = {
    Phase.DASH: "dash",
    Phase.AUCTION: "auction",
    Phase.ESTIMATION: "estimation",
    Phase.PLAY: "play",
    Phase.DONE: "done",
}


def _action(
    action: int,
    kind: str,
    label: str,
    tricks: int | None = None,
    trump: int | None = None,
    card: int | None = None,
) -> ActionView:
    return ActionView(id=action, kind=kind, label=label, tricks=tricks, trump=trump, card=card)


def describe_action(obs: Observation, action: int) -> ActionView:
    """``action`` as the player to act in ``obs`` would take it, labelled in context: a bid in a
    fixed-trump round's estimation is a super call, With names the trick count it joins, and
    Pass on the top bidder's final turn keeps the bid."""
    actor = obs.current_player
    if action == NO_DASH_CALL:
        return _action(action, "no_dash", "No Dash Call")
    if action == DASH_CALL:
        return _action(action, "dash", "Dash Call")
    if is_bid(action):
        tricks, trump = bid_of(action)
        if obs.phase is Phase.ESTIMATION:
            return _action(
                action, "super_call", f"Super call {tricks} {TRUMP_NAMES[trump]}", tricks, trump
            )
        return _action(action, "bid", f"Bid {tricks} {TRUMP_NAMES[trump]}", tricks, trump)
    if action == WITH:
        if obs.top_bid is None:
            return _action(action, "with", "With")
        tricks, trump = bid_of(obs.top_bid)
        return _action(action, "with", f"With {tricks} {TRUMP_NAMES[trump]}", tricks, trump)
    if action == PASS:
        if actor is not None and actor == obs.top_bidder:
            return _action(action, "keep", "Keep bid")
        return _action(action, "pass", "Pass")
    if is_estimate(action):
        k = estimate_of(action)
        return _action(action, "estimate", f"Estimate {k}", tricks=k)
    if is_play(action):
        card = card_of(action)
        return _action(action, "play", f"Play {card_str(card)}", card=card)
    raise ValueError(f"no action with id {action}")


def _played(leader: int, cards: Sequence[int]) -> list[PlayedCard]:
    return [PlayedCard(seat=(leader + i) % 4, card=card) for i, card in enumerate(cards)]


def table_view(
    obs: Observation, events: Sequence[EventView], hands: Sequence[int] | None = None
) -> TableView:
    """The table as ``obs.seat`` sees it, or as a spectator when ``hands`` gives every seat's
    remaining hand. ``events`` are the round's non-card events so far."""
    played = [0, 0, 0, 0]
    for seat, action in obs.log:
        if is_play(action):
            played[seat] += 1
    if hands is None:
        shown: list[list[int] | None] = [
            cards_of(obs.hand) if seat == obs.seat else None for seat in range(4)
        ]
    else:
        shown = [cards_of(hand) for hand in hands]
    calls: list[ActionView | None] = [None, None, None, None]
    for event in events:
        calls[event["seat"]] = event["action"]
    last = obs.tricks[-1] if obs.tricks else None
    return TableView(
        round_no=obs.round_no,
        phase=PHASES[obs.phase],
        current_player=obs.current_player,
        carry=obs.carry,
        scores=list(obs.scores),
        fixed_trump=obs.fixed_trump,
        fixed_left=list(obs.fixed_left),
        replays=obs.replays,
        super_call=obs.super_call,
        trump=obs.trump,
        dash_callers=list(obs.dash_callers),
        top_bid=None if obs.top_bid is None else _bid(obs.top_bid),
        top_bidder=obs.top_bidder,
        passed=list(obs.passed),
        caller=obs.caller,
        with_players=[WithView(seat=w.seat, w=w.w, trump=w.trump) for w in obs.with_players],
        estimates=list(obs.estimates),
        last_estimator=obs.last_estimator,
        total=obs.total,
        over=obs.over,
        risks=obs.risks,
        hands=shown,
        hand_sizes=[13 - n for n in played],
        leader=obs.leader,
        trick=[PlayedCard(seat=seat, card=card) for seat, card in obs.trick],
        last_trick=None
        if last is None
        else TrickView(
            leader=last.leader, cards=_played(last.leader, last.cards), winner=last.winner
        ),
        tricks_played=len(obs.tricks),
        tricks_won=list(obs.tricks_won),
        events=list(events),
        calls=calls,
    )


def _bid(action: int) -> BidView:
    tricks, trump = bid_of(action)
    return BidView(tricks=tricks, trump=trump)


def result_view(result: RoundResult) -> RoundResultView:
    """A scored round: per seat its estimate, tricks, roles and points from SCR-1 to SCR-6."""
    return RoundResultView(
        round_no=result.round_no,
        skipped=result.skipped,
        caller=result.caller,
        trump=result.trump,
        fixed_trump=result.fixed_trump,
        trump_changed=result.trump_changed,
        total=result.total,
        over=result.over,
        calls=result.calls,
        all_lose=result.all_lose,
        carry=result.carry,
        multiplier=result.multiplier,
        next_carry=result.next_carry,
        scores=list(result.scores),
        seats=[
            SeatResultView(
                estimate=seat.estimate,
                tricks=seat.tricks,
                won=seat.won,
                dash=seat.is_dash,
                caller=seat.is_caller,
                with_player=seat.is_with,
                super_call=seat.is_super,
                last=seat.is_last,
                risks=seat.risks,
                base_points=seat.base_points,
                dash_points=seat.dash_points,
                call_points=seat.call_points,
                risk_points=seat.risk_points,
                only_points=seat.only_points,
                super_points=seat.super_points,
                points=seat.points,
                multiplier=seat.multiplier,
                score=seat.score,
            )
            for seat in result.seats
        ],
    )
