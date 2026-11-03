"""Searching the bidding (Hybrid plan step 4, ``docs/rl/hybrid-iteration-plan.md``).

For a Dash Call, bid, With, Pass, estimate or super call, the Hybrid tries BiddingNet's favourite
few legal options on imagined deals of the other hands. On each deal it makes the option, finishes
the round's bidding with BiddingNet for every seat (each holding its imagined hand, the bot itself
included for its later choices), plays the round out with CardPlayNet for every seat
(``rollout.play_rounds``) and scores it by SPEC §3 from its own side, as the card search does. The
option with the best mean wins; a round that ends skipped scores 0 for everyone.

Everything comes from the bot's own Observation: the public log rebuilds the bidding so far
(``beliefs.PublicBidding``), and the other hands are the bot's own samples.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from estimation.engine import Observation, Phase, RulesConfig
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode

from ..scoring import RoundScorer, estimates_of
from .beliefs import BidNet, PublicBidding
from .rollout import CardNet, play_rounds, seat_templates, shapes_of


def bid_totals(
    obs: Observation,
    rules: RulesConfig,
    options: Sequence[int],
    deals: Sequence[Sequence[int]],
    bid_net: BidNet,
    card_net: CardNet,
    weights: tuple[float, float, float, float] | None,
) -> list[float]:
    """For each of ``options`` (legal non-card actions of ``obs.seat``), the sum over ``deals``
    (the four hands, as dealt) of ``obs.seat``'s round score minus the others', weighted by
    ``weights`` as ``RoundScorer`` does, after making the option and finishing the round with
    BiddingNet and CardPlayNet for every seat."""
    me = obs.seat
    base = PublicBidding(obs, rules)
    for seat, action in obs.log:
        if action >= PLAY_BASE:
            raise ValueError("the bid search runs before card play")
        base.step(seat, action)
    states: list[PublicBidding] = []
    dealt: list[Sequence[int]] = []
    for hands in deals:
        for option in options:
            state = base.clone()
            state.step(me, option)
            states.append(state)
            dealt.append(hands)
    _finish_bidding(states, dealt, rules, bid_net)
    totals = [0.0] * len(options)
    playing = [g for g, state in enumerate(states) if state.phase is Phase.PLAY]
    if not playing:
        return totals
    views = [states[g].observation(me) for g in playing]
    hands_array = np.array([list(dealt[g]) for g in playing], dtype=np.uint64)
    won = play_rounds(
        card_net,
        np.stack([seat_templates(view, rules) for view in views]),
        np.array([view.leader for view in views], np.int64),
        np.array([view.trump for view in views], np.int64),
        np.array([estimates_of(view) for view in views], np.int64),
        hands_array.copy(),
        shapes_of(hands_array),
    )
    for row, g in enumerate(playing):
        totals[g % len(options)] += RoundScorer(views[row], rules, weights).value(won[row].tolist())
    return totals


def _finish_bidding(
    states: list[PublicBidding],
    dealt: list[Sequence[int]],
    rules: RulesConfig,
    bid_net: BidNet,
) -> None:
    """Step every game's bidding to its end, all games together: each seat takes its legal action
    with the highest BiddingNet value, the first on a tie, as ``RLBot`` does."""
    while True:
        pending: list[int] = []
        legal_lists: list[list[int]] = []
        rows: list[npt.NDArray[np.float32]] = []
        for g, state in enumerate(states):
            if not state.bidding:
                continue
            legal = state.legal_actions()
            if len(legal) == 1:
                state.step(state.cur, legal[0])
                continue
            seat = state.cur
            view = replace(state.observation(seat), hand=dealt[g][seat])
            pending.append(g)
            legal_lists.append(legal)
            rows.append(encode(view, rules))
        if not pending:
            if any(state.bidding for state in states):
                continue  # forced moves only this pass
            return
        values = bid_net(np.stack(rows))
        for g, legal, row in zip(pending, legal_lists, values, strict=True):
            state = states[g]
            state.step(state.cur, max(legal, key=lambda action: row[action]))
