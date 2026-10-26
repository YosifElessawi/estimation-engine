"""A bot's bidding decisions in a saved game, and the choices worth trying at each (design 5.4).

A decision is any Dash Call, auction or estimation move where the bot had more than one legal
action. It keeps the deciding seat's Observation, so the evaluator judges it only on what that
player could see, plus what is needed to rebuild the round: its number, fixed trump, carry, the
game scores before it and the round's actions so far.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from estimation.arena.games import GameRecord
from estimation.engine import PASS, WITH, Bola, Observation, Phase, RulesConfig, bid_action, bid_of
from estimation.engine.actions import ESTIMATE_BASE, is_bid, is_estimate

KINDS = (
    "dash",
    "open",
    "bid",
    "final turn",
    "answer",
    "with or pass",
    "estimate",
    "last estimate",
    "super call",
)
MAX_CANDIDATES = 8  # [Claude]
ESTIMATE_REACH = 2  # estimates tried either side of the chosen one [Claude]


@dataclass(frozen=True, slots=True)
class Decision:
    """One non-card decision. ``game`` and ``step`` are None and -1 for a position built by
    hand (``decision_at``)."""

    game: GameRecord | None
    step: int
    seat: int
    kind: str
    obs: Observation
    legal: tuple[int, ...]
    chosen: int
    round_no: int
    fixed_trump: int | None
    carry: int
    scores: tuple[int, int, int, int]
    round_actions: tuple[int, ...]


def _out(obs: Observation) -> bool:
    """Out of the auction (passed or With): such a player acts only to answer a raise."""
    me = obs.seat
    return me in obs.passed or any(w.seat == me for w in obs.with_players)


def kind_of(obs: Observation, chosen: int, legal: Sequence[int]) -> str:
    if obs.phase is Phase.DASH:
        return "dash"
    if obs.phase is Phase.AUCTION:
        if obs.top_bid is None:
            return "open"
        if obs.top_bidder == obs.seat:
            return "final turn"
        if _out(obs):
            return "answer"
        return "with or pass" if chosen in (WITH, PASS) else "bid"
    if is_bid(chosen):
        return "super call"
    return "last estimate" if obs.last_estimator == obs.seat else "estimate"


def decision_at(
    obs: Observation,
    legal: Sequence[int],
    chosen: int,
    game: GameRecord | None = None,
    step: int = -1,
) -> Decision:
    """The decision ``obs.seat`` faces in ``obs``, having chosen ``chosen``."""
    if obs.phase in (Phase.PLAY, Phase.DONE):
        raise ValueError("only Dash Call, auction and estimation moves are bidding decisions")
    return Decision(
        game=game,
        step=step,
        seat=obs.seat,
        kind=kind_of(obs, chosen, legal),
        obs=obs,
        legal=tuple(legal),
        chosen=chosen,
        round_no=obs.round_no,
        fixed_trump=obs.fixed_trump,
        carry=obs.carry,
        scores=obs.scores,
        round_actions=tuple(action for _, action in obs.log),
    )


def decisions_of(record: GameRecord, rules: RulesConfig, target: str) -> list[Decision]:
    """Every decision with a real choice made by a seat holding ``target`` (a bot spec, or
    ``human`` for the player's seat in a web game)."""
    bola = Bola(record.deal_seed, rules)
    found: list[Decision] = []
    for step, action in enumerate(record.actions):
        seat = bola.current_player
        if seat is None:
            raise ValueError("a saved game has more actions than its game had decisions")
        if record.seats[seat] == target and bola.phase is not Phase.PLAY:
            legal = bola.legal_actions()
            if len(legal) > 1:
                found.append(decision_at(bola.observation(seat), legal, action, record, step))
        bola.step(action)
    return found


def _bid_neighbours(action: int) -> list[int]:
    """The same trump one trick lower and one higher."""
    tricks, trump = bid_of(action)
    return [bid_action(t, trump) for t in (tricks - 1, tricks + 1) if 4 <= t <= 13]


def candidates(d: Decision, heuristic_choice: int) -> list[int]:
    """The choices to play out: the bot's own first, then HeuristicBot's, then the ones around
    them (design 5.4), all legal, at most ``MAX_CANDIDATES``."""
    legal = set(d.legal)
    wanted: list[int] = [d.chosen, heuristic_choice]
    if d.obs.phase is Phase.DASH:
        wanted += sorted(legal)
    elif d.obs.phase is Phase.AUCTION:
        wanted += [PASS, WITH]
        for action in (d.chosen, heuristic_choice):
            if is_bid(action):
                wanted += _bid_neighbours(action)
        top = d.obs.top_bid
        if top is not None and not is_bid(d.chosen):
            tricks, trump = bid_of(top)
            if tricks < 13:
                wanted.append(bid_action(tricks + 1, trump))
    else:
        for action in (d.chosen, heuristic_choice):
            if is_estimate(action):
                k = action - ESTIMATE_BASE
                wanted += [
                    ESTIMATE_BASE + k + step
                    for step in range(-ESTIMATE_REACH, ESTIMATE_REACH + 1)
                    if 0 <= k + step <= 13
                ]
            elif is_bid(action):
                wanted += _bid_neighbours(action)
    chosen: list[int] = []
    for action in wanted:
        if action in legal and action not in chosen:
            chosen.append(action)
    return chosen[:MAX_CANDIDATES]
