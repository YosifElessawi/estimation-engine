"""The state encoder (SPEC §6 M4): one seat's Observation as ``NUM_FEATURES`` floats.

Everything comes from the acting seat's own Observation and the rules being played. Seats are
listed relative to the acting seat (itself, then the next three counterclockwise), so the same
position looks the same from every seat. The blocks, in order:

==========  ====  ===========================================================================
Block       Size  Contents
==========  ====  ===========================================================================
hand          52  own remaining cards
played       156  cards each opponent has played this round, current trick included (3 x 52)
table         52  cards in the current trick
unseen        52  cards in the other three hands
voids         12  suits each opponent has shown out of (3 x 4)
trick         14  led suit, own position in the trick, who is winning it and with what
context       23  phase, fixed-trump round and its trump, super call made, contract of 8 or
                  more, trump, Over/Under delta, carry, round multiplier, Risks, tricks played
auction       13  top bid, top bidder, own final turn, bidders still in
players       56  per seat: estimate, tricks, points if it wins or loses, alive or broken,
                  Caller, With, Dash caller, last estimator, passed, With level (4 x 14)
scores        13  game scores, gap to first place, standings weights, open rounds and fixed
                  trumps left, replays, own match-win probability
shape          9  suit lengths and HeuristicBot's expected tricks under each trump
==========  ====  ===========================================================================

``points_if_win`` and ``points_if_lose`` come from the RulesConfig, so a rule change shows up in
the inputs instead of needing a retrain from scratch.
"""

import math

import numpy as np
import numpy.typing as npt

from estimation.bots.hand_eval import trump_values
from estimation.engine import Observation, Phase, RulesConfig
from estimation.engine.actions import FIRST_BID
from estimation.engine.config import DEFAULT_CONFIG

from .standings import ROUND_SD, rounds_remaining, standings_weights, win_probabilities

ENCODER_VERSION = 1

_BLOCKS = (
    ("hand", 52),
    ("played", 156),
    ("table", 52),
    ("unseen", 52),
    ("voids", 12),
    ("trick", 14),
    ("context", 23),
    ("auction", 13),
    ("players", 56),
    ("scores", 13),
    ("shape", 9),
)
OFFSETS: dict[str, int] = {}
_offset = 0
for _name, _size in _BLOCKS:
    OFFSETS[_name] = _offset
    _offset += _size
NUM_FEATURES = _offset
PLAYER_FEATURES = 14

HAND = OFFSETS["hand"]
VOIDS = OFFSETS["voids"]
TRICK = OFFSETS["trick"]
CONTEXT = OFFSETS["context"]
AUCTION = OFFSETS["auction"]
PLAYERS = OFFSETS["players"]
SCORES = OFFSETS["scores"]
SHAPE = OFFSETS["shape"]

STAKES_SCALE = 100.0  # points_if_win and points_if_lose are given in hundreds of points
CLIP = 5.0  # stakes, scores and multipliers are clipped to +-5 (random play stacks carries)
SCORE_SCALE = 200.0
FULL_DECK = (1 << 52) - 1
_PHASE_SLOT = {Phase.DASH: 0, Phase.AUCTION: 1, Phase.ESTIMATION: 2, Phase.PLAY: 3}
_CARD_MASKS = 6  # hand, three opponents' played cards, table, unseen


def encode(obs: Observation, rules: RulesConfig = DEFAULT_CONFIG) -> npt.NDArray[np.float32]:
    """``obs`` as a float32 vector of ``NUM_FEATURES``."""
    out = np.zeros(NUM_FEATURES, dtype=np.float32)
    encode_into(out, obs, rules)
    return out


def encode_into(out: npt.NDArray[np.float32], obs: Observation, rules: RulesConfig) -> None:
    """Write ``obs``'s features into ``out``, a zeroed float32 vector of ``NUM_FEATURES``."""
    me = obs.seat
    hand = obs.hand

    # Cards: who played what, the table and the unseen cards; voids from failing to follow.
    played_by = [0, 0, 0, 0]
    voids = [0, 0, 0, 0]
    for record in obs.tricks:
        leader = record.leader
        cards = record.cards
        led = cards[0] // 13
        for i in range(4):
            card = cards[i]
            seat = (leader + i) & 3
            played_by[seat] |= 1 << card
            if card // 13 != led:
                voids[seat] |= 1 << led
    table = 0
    trick = obs.trick
    if trick:
        led = trick[0][1] // 13
        for seat, card in trick:
            played_by[seat] |= 1 << card
            table |= 1 << card
            if card // 13 != led:
                voids[seat] |= 1 << led
    played = played_by[0] | played_by[1] | played_by[2] | played_by[3]
    masks = np.array(
        [
            hand,
            played_by[(me + 1) & 3],
            played_by[(me + 2) & 3],
            played_by[(me + 3) & 3],
            table,
            FULL_DECK & ~played & ~hand,
        ],
        dtype="<u8",
    )
    bits = np.unpackbits(masks.view(np.uint8), bitorder="little").reshape(_CARD_MASKS, 64)
    out[HAND : HAND + 52 * _CARD_MASKS] = bits[:, :52].reshape(-1)
    for r in range(1, 4):
        void = voids[(me + r) & 3]
        base = VOIDS + 4 * (r - 1)
        for suit in range(4):
            if void >> suit & 1:
                out[base + suit] = 1.0

    phase = obs.phase
    trump = obs.trump

    # The trick in progress.
    if phase is Phase.PLAY:
        out[TRICK + 4 + len(trick)] = 1.0
    if trick:
        led = trick[0][1] // 13
        out[TRICK + led] = 1.0
        best_seat, best = trick[0]
        for seat, card in trick[1:]:
            suit = card // 13
            if (suit == best // 13 and card > best) or (suit != best // 13 and suit == trump):
                best_seat, best = seat, card
        out[TRICK + 8 + ((best_seat - me) & 3)] = 1.0
        out[TRICK + 12] = float(best // 13 == trump)
        out[TRICK + 13] = (best % 13) / 12.0

    # Round context.
    estimates = obs.estimates
    known = [e for e in estimates if e is not None]
    complete = obs.total is not None
    total = sum(known)
    with_seats = {w.seat: w for w in obs.with_players}
    multiplier = round_multiplier(obs, rules)
    if phase in _PHASE_SLOT:
        out[CONTEXT + _PHASE_SLOT[phase]] = 1.0
    fixed = obs.fixed_trump
    if fixed is not None:
        out[CONTEXT + 4] = 1.0
        out[CONTEXT + 5 + fixed] = 1.0
    out[CONTEXT + 10] = float(obs.super_call)
    out[CONTEXT + 11] = float(contract(obs) >= rules.super_call_min)
    if trump is not None:
        out[CONTEXT + 12 + trump] = 1.0
    out[CONTEXT + 17] = (total - 13) / 13.0
    out[CONTEXT + 18] = float(complete)
    out[CONTEXT + 19] = _clip(math.log2(obs.carry) / 3.0)
    out[CONTEXT + 20] = _clip(math.log2(multiplier) / 3.0)
    out[CONTEXT + 21] = _clip((obs.risks or 0) / 3.0)
    tricks_done = len(obs.tricks)
    out[CONTEXT + 22] = tricks_done / 13.0

    # The auction.
    top = obs.top_bid
    if top is not None:
        tricks, top_trump = divmod(top - FIRST_BID, 5)
        out[AUCTION] = 1.0
        out[AUCTION + 1] = tricks / 9.0
        out[AUCTION + 2 + top_trump] = 1.0
    top_bidder = obs.top_bidder
    if top_bidder is not None:
        out[AUCTION + 7 + ((top_bidder - me) & 3)] = 1.0
    if phase is Phase.AUCTION:
        out[AUCTION + 11] = float(top_bidder == me)
        bidders = 4 - len(obs.dash_callers) - len(obs.passed)
        out[AUCTION + 12] = bidders / 4.0

    # Per player, relative to the acting seat.
    tricks_won = obs.tricks_won
    left = 13 - tricks_done
    for r in range(4):
        seat = (me + r) & 3
        base = PLAYERS + PLAYER_FEATURES * r
        estimate = estimates[seat]
        won = tricks_won[seat]
        out[base + 2] = won / 13.0
        if estimate is not None:
            out[base] = estimate / 13.0
            out[base + 1] = 1.0
            out[base + 3] = (estimate - won) / 13.0
            win, lose = stakes(obs, seat, rules)
            out[base + 4] = _clip(win / STAKES_SCALE)
            out[base + 5] = _clip(lose / STAKES_SCALE)
            alive = won <= estimate <= won + left
            out[base + 6] = float(alive)
            out[base + 7] = float(not alive)
        out[base + 8] = float(seat == obs.caller)
        info = with_seats.get(seat)
        if info is not None:
            out[base + 9] = 1.0
            out[base + 13] = info.w / 13.0
        out[base + 10] = float(seat in obs.dash_callers)
        out[base + 11] = float(seat == obs.last_estimator)
        out[base + 12] = float(seat in obs.passed)

    # The scoreboard.
    scores = obs.scores
    leader_score = max(scores)
    weights = standings_weights(scores, me)
    for r in range(4):
        seat = (me + r) & 3
        out[SCORES + r] = _clip(scores[seat] / SCORE_SCALE)
        if r:
            out[SCORES + 4 + r] = weights[seat]
    out[SCORES + 4] = _clip((leader_score - scores[me]) / STAKES_SCALE)
    out[SCORES + 8] = sum(1 for s in scores if s > scores[me]) / 3.0
    open_round = fixed is None
    out[SCORES + 9] = max(0, 14 - obs.round_no) / 13.0 if open_round else 0.0
    out[SCORES + 10] = len(obs.fixed_left) / 5.0
    out[SCORES + 11] = obs.replays / 5.0
    left_rounds = rounds_remaining(obs.round_no, len(obs.fixed_left), not open_round)
    out[SCORES + 12] = win_probabilities(scores, left_rounds, ROUND_SD)[me]

    # Hand shape and HeuristicBot's hand model, from the full original hand.
    original = hand | played_by[me]
    for suit in range(4):
        out[SHAPE + suit] = ((original >> (13 * suit)) & 0x1FFF).bit_count() / 13.0
    values = trump_values(original)
    for t in range(5):
        out[SHAPE + 4 + t] = values[t] / 13.0


def _clip(value: float) -> float:
    return max(-CLIP, min(CLIP, value))


def contract(obs: Observation) -> int:
    """The round's contract so far: the Caller's ``c`` once known, else the top bid's tricks,
    else (fixed-trump estimation) the highest estimate; 0 if there is none yet."""
    caller = obs.caller
    if caller is not None:
        c = obs.estimates[caller]
        if c is not None:
            return c
    if obs.top_bid is not None:
        return (obs.top_bid - FIRST_BID) // 5 + 4
    return max((e for e in obs.estimates if e is not None), default=0)


def round_multiplier(obs: Observation, rules: RulesConfig) -> int:
    """The round's multiplier as far as it is known (SCR-7, SCR-8): the carry, doubled once the
    Caller and With players (with ``calls_count_risk_and_dash``, also Dash callers and any Risk)
    reach ``calls_to_double``. During an auction only the carry is known."""
    if obs.phase is not Phase.ESTIMATION and obs.phase is not Phase.PLAY:
        return obs.carry
    calls = (obs.caller is not None) + len(obs.with_players)
    if rules.calls_count_risk_and_dash:
        calls += len(obs.dash_callers) + (1 if obs.risks else 0)
    return obs.carry * (2 if calls >= rules.calls_to_double else 1)


def stakes(obs: Observation, seat: int, rules: RulesConfig) -> tuple[int, int]:
    """``(points_if_win, points_if_lose)`` for ``seat``'s estimate: its round score if it takes
    exactly its estimate, and if it misses by one trick, from SCR-1 to SCR-4 and SCR-6 times the
    multiplier known so far. SCR-5 and SCR-9 depend on the other seats and are left out. Before
    estimation is complete, Over/Under is judged on the estimates so far and Risks count 0.
    Both are 0 while ``seat`` has no estimate."""
    b = obs.estimates[seat]
    if b is None:
        return 0, 0
    total = sum(e for e in obs.estimates if e is not None)
    over = total > 13
    is_dash = seat in obs.dash_callers
    called = seat == obs.caller or any(w.seat == seat for w in obs.with_players)
    win = lose = 0
    if is_dash:  # SCR-2
        dash = rules.dash_over if over else rules.dash_under
        win, lose = dash, -dash
        if not rules.dash_replaces_base:
            win += rules.win_base + b
            lose -= 1
    elif called and b >= rules.super_call_min and rules.super_call_scoring:  # SCR-6
        win, lose = b * b, -(b * b // 2)
    else:  # SCR-1
        win, lose = rules.win_base + b, -1
    if called:  # SCR-3
        win += rules.caller_with_bonus
        lose -= rules.caller_with_bonus
    if seat == obs.last_estimator and obs.risks:  # SCR-4
        win += rules.risk_points * obs.risks
        lose -= rules.risk_points * obs.risks
    multiplier = round_multiplier(obs, rules)
    return win * multiplier, lose * multiplier
