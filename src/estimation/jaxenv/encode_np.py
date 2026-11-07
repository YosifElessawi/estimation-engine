"""The same raw view as jaxenv.encode, from an engine Observation, in numpy (no JAX needed).

An arena bot plays through this. jaxenv.check requires it to equal the GPU encoder at every step.
"""

import math

import numpy as np

from estimation.engine import Phase, cards_of

SIZE = 52 + 208 + 208 + 16 + 20 + 51 + 5 + 4 * 6 + 5 * 3 + 4 + 6 + 6 + 1 + 1 + 1 + 4 + 2 + 3 + 120
FIRST_BID, WITH, PLAY_BASE = 2, 52, 68


def _seat_hot(x, me):
    v = [0.0] * 5
    v[4 if x is None else (x - me) % 4] = 1.0
    return v


def encode(obs, legal) -> np.ndarray:
    me = obs.seat
    rel = [(me + k) % 4 for k in range(4)]
    played = np.zeros((4, 52), np.float32)
    table = np.zeros((4, 52), np.float32)
    shown = np.zeros((4, 4), np.float32)
    for t in obs.tricks:
        for i, c in enumerate(t.cards):
            seat = (t.leader + i) % 4
            played[seat, c] = 1
            if i and c // 13 != t.cards[0] // 13:
                shown[seat, t.cards[0] // 13] = 1
    for i, (seat, c) in enumerate(obs.trick):
        played[seat, c] = table[seat, c] = 1
        if i and c // 13 != obs.trick[0][1] // 13:
            shown[seat, obs.trick[0][1] // 13] = 1
    # calls come from the public log: the last bid or joined bid per seat per trump (AUC-4)
    calls = np.zeros((4, 5), np.float32)
    top = None
    for seat, a in obs.log:
        if FIRST_BID <= a < WITH:
            if obs.fixed_trump is not None and top is None:
                calls[:] = 0
            top = a
            calls[seat, (a - 2) % 5] = ((a - 2) // 5 + 4) / 13
        elif a == WITH:
            calls[seat, (top - 2) % 5] = ((top - 2) // 5 + 4) / 13
    is_with = [0.0] * 4
    level = [0.0] * 4
    for w in obs.with_players:
        is_with[w.seat], level[w.seat] = 1.0, w.w / 13
    passed = [1.0 if s in obs.passed else 0.0 for s in range(4)]
    dash = [1.0 if s in obs.dash_callers else 0.0 for s in range(4)]
    est = [(0 if e is None else e + 1) / 14 for e in obs.estimates]
    scores = [x / 200 for x in obs.scores]
    own = scores[me]
    top_hot = [0.0] * 51
    top_hot[0 if obs.top_bid is None else obs.top_bid - 1] = 1.0
    phase = [0.0] * 4
    phase[int(obs.phase)] = 1.0
    trump = [0.0] * 6
    trump[5 if obs.trump is None else obs.trump] = 1.0
    fixed = [0.0] * 6
    fixed[5 if obs.fixed_trump is None else obs.fixed_trump] = 1.0
    hand = np.zeros(52, np.float32)
    hand[cards_of(obs.hand)] = 1
    mask = np.zeros(120, np.float32)
    mask[list(legal)] = 1
    settled = obs.total is not None
    return np.concatenate([
        hand,
        played[rel].ravel(),
        table[rel].ravel(),
        shown[rel].ravel(),
        calls[rel].ravel(),
        top_hot,
        _seat_hot(obs.top_bidder, me),
        [dash[s] for s in rel],
        [passed[s] for s in rel],
        [is_with[s] for s in rel],
        [level[s] for s in rel],
        [est[s] for s in rel],
        [obs.tricks_won[s] / 13 for s in rel],
        _seat_hot(obs.caller, me),
        _seat_hot(obs.last_estimator, me),
        _seat_hot(obs.leader if obs.phase is Phase.PLAY else None, me),
        phase,
        trump,
        fixed,
        [float(obs.super_call)],
        [obs.round_no / 40],
        [min(max(math.log2(obs.carry) / 10, 0.0), 2.0)],
        [1.0 if s in obs.void_announced else 0.0 for s in rel],
        [min(max(own, -5.0), 5.0), (obs.total - 13) / 13 if settled else 0.0],
        [min(max(scores[s] - own, -5.0), 5.0) for s in rel[1:]],
        mask,
    ]).astype(np.float32)
