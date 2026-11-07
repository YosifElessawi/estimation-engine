"""One round of Estimation as pure JAX functions, mirroring ``estimation.engine.round``.

Every field of ``Round`` is a fixed-shape array, so ``jax.vmap`` steps any number of rounds
at once. ``-1`` means "none" for seats, calls, estimates and trumps, as in the engine. The
comments name the engine's rule ids (SPEC §2-§3) and methods each piece mirrors.
Default rules only (``estimation.engine.config.DEFAULT_CONFIG``).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Action ids (estimation.engine.actions)
NUM_ACTIONS = 120
NO_DASH_CALL, DASH_CALL = 0, 1
FIRST_BID, LAST_BID = 2, 51
WITH, PASS = 52, 53
ESTIMATE_BASE, PLAY_BASE = 54, 68
NO_BID = FIRST_BID - 1
LONE_MAX_BID = FIRST_BID + (12 - 4) * 5 + 4  # AUC-5: a lone bidder may not bid 13

# Phases (estimation.engine.observation.Phase)
DASH, AUCTION, ESTIMATION, PLAY, DONE = 0, 1, 2, 3, 4

SANS = 4
OPEN_ROUNDS = 13
FIXED_ORDER = (4, 3, 2, 1, 0)  # Sans, Spades, Hearts, Diamonds, Clubs (FIX-1)

# Default rules (RulesConfig())
WIN_BASE, CALLER_WITH_BONUS, RISK_POINTS, ONLY_BONUS = 10, 10, 10, 10
DASH_OVER, DASH_UNDER = 25, 33
SUPER_CALL_MIN = 8
CALLS_TO_DOUBLE = 3
MAX_FIXED_REPLAYS = 20

A = jnp.arange(NUM_ACTIONS)
CARDS = jnp.arange(52)
SEATS = jnp.arange(4)
I32 = jnp.int32


def rank_of(call):
    """Tricks of a bid id (4-13)."""
    return (call - FIRST_BID) // 5 + 4


def trump_of(call):
    return (call - FIRST_BID) % 5


IS_BID = (A >= FIRST_BID) & (A <= LAST_BID)
BID_RANK = jnp.where(IS_BID, rank_of(A), 0)
BID_TRUMP = jnp.where(IS_BID, trump_of(A), -1)


class Round(NamedTuple):
    phase: jax.Array
    cur: jax.Array
    first: jax.Array
    round_no: jax.Array
    fixed_trump: jax.Array
    carry: jax.Array
    scores: jax.Array  # [4] game scores before this round
    hands: jax.Array  # [4, 52] bool, remaining hands
    played: jax.Array  # [4, 52] bool, cards each seat has played this round
    shown_out: jax.Array  # [4, 4] bool, suits each seat has failed to follow
    announced: jax.Array  # [4] bool, void at the deal (TBL-7)
    dash: jax.Array  # [4] bool
    chosen: jax.Array
    out: jax.Array  # [4] bool
    top: jax.Array
    top_seat: jax.Array
    hold: jax.Array  # [4]
    calls: jax.Array  # [4, 5] last call per seat per trump (AUC-4)
    stayed: jax.Array  # [4]
    joined: jax.Array  # [4]
    final: jax.Array  # bool
    answers: jax.Array  # [3] queue of seats answering a raise (AUC-3)
    n_answers: jax.Array
    max_bid: jax.Array
    super_call: jax.Array  # bool
    super_seat: jax.Array
    est: jax.Array  # [4]
    total: jax.Array
    order: jax.Array  # [4] estimation order, -1 padded
    order_len: jax.Array
    pos: jax.Array
    caller: jax.Array
    c: jax.Array
    with_w: jax.Array  # [4]
    trump: jax.Array
    last: jax.Array
    leader: jax.Array
    trick: jax.Array  # [4] cards in play order
    tlen: jax.Array
    led: jax.Array
    won: jax.Array  # [4]
    ntricks: jax.Array
    log_len: jax.Array  # actions taken so far
    result: jax.Array  # [4] round scores once DONE
    skipped: jax.Array  # bool
    next_carry: jax.Array
    trump_changed: jax.Array  # bool


def _i(x):
    return jnp.asarray(x, I32)


def init_round(round_no, hands, fixed_trump=-1, carry=1, scores=(0, 0, 0, 0)) -> Round:
    """``Round.new``: ``hands`` is [4, 52] bool; ``fixed_trump`` -1 in open rounds."""
    round_no, fixed_trump, carry = _i(round_no), _i(fixed_trump), _i(carry)
    hands = jnp.asarray(hands, bool)
    first = (round_no - 1) % 4
    fixed = fixed_trump >= 0
    suits = hands.reshape(4, 4, 13).any(axis=2)
    order_fixed = (first + SEATS) % 4
    m1 = _i(-jnp.ones(4))
    return Round(
        phase=jnp.where(fixed, ESTIMATION, DASH).astype(I32),
        cur=first,
        first=first,
        round_no=round_no,
        fixed_trump=fixed_trump,
        carry=carry,
        scores=_i(jnp.asarray(scores)),
        hands=hands,
        played=jnp.zeros((4, 52), bool),
        shown_out=jnp.zeros((4, 4), bool),
        announced=~suits.all(axis=1),
        dash=jnp.zeros(4, bool),
        chosen=_i(0),
        out=jnp.zeros(4, bool),
        top=_i(NO_BID),
        top_seat=_i(-1),
        hold=m1,
        calls=_i(-jnp.ones((4, 5))),
        stayed=m1,
        joined=m1,
        final=jnp.asarray(False),
        answers=_i(-jnp.ones(3)),
        n_answers=_i(0),
        max_bid=_i(LAST_BID),
        super_call=jnp.asarray(False),
        super_seat=_i(-1),
        est=m1,
        total=_i(0),
        order=jnp.where(fixed, order_fixed, -1).astype(I32),
        order_len=jnp.where(fixed, 4, 0).astype(I32),
        pos=_i(0),
        caller=_i(-1),
        c=_i(0),
        with_w=m1,
        trump=jnp.where(fixed, fixed_trump, -1).astype(I32),
        last=jnp.where(fixed, order_fixed[3], -1).astype(I32),
        leader=_i(-1),
        trick=m1,
        tlen=_i(0),
        led=_i(-1),
        won=_i(jnp.zeros(4)),
        ntricks=_i(0),
        log_len=_i(0),
        result=_i(jnp.zeros(4)),
        skipped=jnp.asarray(False),
        next_carry=_i(0),
        trump_changed=jnp.asarray(False),
    )


def _sel(cond, a: Round, b: Round) -> Round:
    """``a`` where ``cond``, else ``b``, field by field."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# ---- auction helpers (Round._ends_on_13, _answerers, _call, _take_top, _next_bidder) ----


def _ends_on_13(s: Round, bid, seat=-1, answer=-1):
    """AUC-6: the auction ending on ``bid`` by the top bidder leaves nobody to estimate and
    the kept numbers total 13 (after ``seat`` answers ``answer``, if given)."""
    c = rank_of(bid)
    trump = trump_of(bid)
    total = c
    ok = jnp.asarray(True)
    for t in range(4):
        skip = (t == s.top_seat) | s.dash[t]
        joins = (t == seat) & (answer == WITH)
        call = s.calls[t, trump]
        stayed = jnp.where((t == seat) & (answer == PASS), s.hold[t], s.stayed[t])
        keeps = (rank_of(call) == c) | (call == stayed)  # EST-2 (with_choose)
        fail = ~skip & ~joins & ((call < 0) | ~keeps)
        ok = ok & ~fail
        total = total + jnp.where(skip | fail, 0, jnp.where(joins, c, rank_of(call)))
    return ok & (total == 13)


def _sort3(keys, values):
    """``values`` and ``keys`` ordered by the three distinct ``keys``, without a sort op (XLA's
    GPU sort of many tiny rows is very slow at large batch sizes). Each item's position is
    the number of keys below it."""
    pos = (keys[None, :] < keys[:, None]).sum(axis=1)
    onehot = pos[None, :] == jnp.arange(3)[:, None]  # [position, item]
    return (onehot * keys[None, :]).sum(axis=1), (onehot * values[None, :]).sum(axis=1)


def _answerers(s: Round, bidder):
    """Seats With on ``bidder``, counterclockwise from it: (queue [3], count)."""
    seats = (bidder + jnp.arange(1, 4)) % 4
    flags = s.joined[seats] == bidder
    key = jnp.where(flags, 0, 1) * 4 + jnp.arange(3)
    srt, ordered = _sort3(key, seats)
    queue = jnp.where(srt < 4, ordered, -1)
    return queue.astype(I32), flags.sum().astype(I32)


def _call(s: Round, seat, action) -> Round:
    return s._replace(
        hold=s.hold.at[seat].set(action),
        calls=s.calls.at[seat, trump_of(action)].set(action),
    )


def _take_top(s: Round, seat, action) -> Round:
    old = s.top_seat
    joined = jnp.where((old >= 0) & (s.joined == old), -1, s.joined)
    s = s._replace(top=_i(action), top_seat=_i(seat), joined=joined)
    s = _call(s, seat, action)
    return s._replace(joined=s.joined.at[seat].set(-1), out=s.out.at[seat].set(False))


def _skip(s: Round) -> Round:
    """GAM-3: no Caller, everyone scores 0, the carry doubles (GAM-4)."""
    return s._replace(
        phase=_i(DONE), result=_i(jnp.zeros(4)), skipped=jnp.asarray(True),
        next_carry=s.carry * 2, trump_changed=jnp.asarray(False),
    )  # fmt: skip


def _next_bidder(s: Round, seat) -> Round:
    """AUC-1: the next seat still in and not the top bidder; else the final turn or a skip."""
    cand = (seat + jnp.arange(1, 5)) % 4
    ok = ~s.out[cand] & (cand != s.top_seat)
    nxt = cand[jnp.argmax(ok)]
    final = s._replace(final=jnp.asarray(True), cur=s.top_seat)
    return _sel(ok.any(), s._replace(cur=nxt.astype(I32)), _sel(s.top_seat < 0, _skip(s), final))


def _start_play(s: Round) -> Round:
    return s._replace(phase=_i(PLAY), leader=s.caller, cur=s.caller)  # PLY-1


def _end_auction(s: Round) -> Round:
    """Caller, trump, With players who keep or choose their number, the estimation order."""
    caller = s.top_seat
    c = rank_of(s.top)
    trump = trump_of(s.top)
    after = (caller + jnp.arange(1, 4)) % 4
    est, with_w, total = s.est, s.with_w, s.total
    group = []
    for k in range(3):
        seat = after[k]
        d = s.dash[seat]
        call = s.calls[seat, trump]
        has = ~d & (call >= 0)
        w = rank_of(call)
        keeps = (w == c) | (call == s.stayed[seat])
        with_w = with_w.at[seat].set(jnp.where(has, w, with_w[seat]))
        est = est.at[seat].set(jnp.where(has & keeps, w, est[seat]))
        total = total + jnp.where(has & keeps, w, 0)
        group.append(jnp.where(has & ~keeps, 0, jnp.where(~d & ~has, 1, 2)))  # choosers first
    est = est.at[caller].set(c)
    total = total + c
    key = jnp.stack(group) * 4 + jnp.arange(3)
    srt, seats = _sort3(key, after)
    order = jnp.concatenate([jnp.where(srt < 8, seats, -1), jnp.array([-1])]).astype(I32)
    order_len = (srt < 8).sum().astype(I32)
    s = s._replace(caller=caller, c=c, trump=trump, final=jnp.asarray(False), est=est,
                   with_w=with_w, total=total, order=order, order_len=order_len, pos=_i(0))  # fmt: skip
    has_order = order_len > 0
    estimating = s._replace(phase=_i(ESTIMATION), cur=order[0], last=order[jnp.maximum(order_len - 1, 0)])
    return _sel(has_order, estimating, _start_play(s._replace(last=_i(-1))))


# ---- legal moves (Round.legal_actions) ----


def _legal_play(s: Round):
    hand = s.hands[s.cur]
    follow = hand & (CARDS // 13 == s.led)
    cards = jnp.where((s.tlen > 0) & follow.any(), follow, hand)  # PLY-2
    return jnp.concatenate([jnp.zeros(PLAY_BASE, bool), cards])


def _legal_estimates(s: Round):
    is_est = (A >= ESTIMATE_BASE) & (A < PLAY_BASE)
    k = A - ESTIMATE_BASE
    # fixed-trump estimation before any super call (FIX-2, FIX-4)
    supers = IS_BID & (BID_RANK >= SUPER_CALL_MIN) & (BID_TRUMP != s.fixed_trump)
    forbid_fixed = (s.pos == 3) & (k == 13 - s.total)
    fixed = supers | (is_est & ~forbid_fixed)
    # after an auction (EST-2, EST-3, EST-5)
    low = jnp.maximum(s.with_w[s.cur], 0)
    forbid = (s.pos == s.order_len - 1) & (k == 13 - s.total)
    normal = is_est & (k >= low) & (k <= s.c) & ~forbid
    return jnp.where(s.caller < 0, fixed, normal)


def _legal_auction(s: Round):
    seat = s.cur
    bids = IS_BID & (A > s.top) & (A <= s.max_bid)
    last_answer = s.n_answers == 1
    answering = (
        bids
        | ((A == WITH) & (~last_answer | ~_ends_on_13(s, s.top, seat, WITH)))
        | ((A == PASS) & (~last_answer | ~_ends_on_13(s, s.top, seat, PASS)))
    )
    _, n_ans = _answerers(s, seat)
    ends = jax.vmap(lambda b: _ends_on_13(s, b))(jnp.clip(A, FIRST_BID, LAST_BID))
    final = (bids & ((n_ans > 0) | ~ends)) | ((A == PASS) & ~_ends_on_13(s, s.top))  # AUC-6
    normal = bids | ((A == WITH) & (s.top_seat >= 0)) | (A == PASS)  # AUC-2
    return jnp.where(s.n_answers > 0, answering, jnp.where(s.final, final, normal))


def legal_mask(s: Round):
    """[120] bool: the legal actions of the player to act (all False once DONE)."""
    masks = jnp.stack([A < 2, _legal_auction(s), _legal_estimates(s), _legal_play(s), A < 0])
    return masks[s.phase]


# ---- steps (Round.step and the _step_* methods) ----


def _step_dash(s: Round, a) -> Round:
    seat = s.cur
    call = a == DASH_CALL
    s = s._replace(
        dash=s.dash.at[seat].set(s.dash[seat] | call),
        est=s.est.at[seat].set(jnp.where(call, 0, s.est[seat])),  # DSH-4
        chosen=s.chosen + 1,
    )
    more = s._replace(cur=(seat + 1) % 4)
    callers = s.dash.sum()
    first_free = (s.first + jnp.arange(4)) % 4
    opener = first_free[jnp.argmax(~s.dash[first_free])]
    auction = s._replace(
        out=s.dash, max_bid=jnp.where(callers == 3, LONE_MAX_BID, LAST_BID).astype(I32),
        phase=_i(AUCTION), cur=opener.astype(I32),
    )  # fmt: skip
    return _sel(s.chosen < 4, more, _sel(callers == 4, _skip(s), auction))


def _step_answer(s: Round, a) -> Round:
    """AUC-3: a With player answers the top bidder's raise."""
    seat = s.cur
    is_bid = IS_BID[a]
    raised = _take_top(s._replace(n_answers=_i(0), answers=_i(-jnp.ones(3))), seat, a)
    raised = _next_bidder(raised, seat)
    with_ = _call(s, seat, s.top)
    stay = s._replace(stayed=s.stayed.at[seat].set(s.hold[seat]), joined=s.joined.at[seat].set(-1))
    t = _sel(a == WITH, with_, stay)
    queue = jnp.concatenate([t.answers[1:], jnp.array([-1])]).astype(I32)
    t = t._replace(answers=queue, n_answers=t.n_answers - 1)
    t = _sel(t.n_answers > 0, t._replace(cur=queue[0]), _end_auction(t))
    return _sel(is_bid, raised, t)


def _step_auction(s: Round, a) -> Round:
    seat = s.cur
    is_bid = IS_BID[a]
    # a raise on the final turn (AUC-3)
    queue, n = _answerers(s, seat)
    raised = _call(s._replace(top=_i(a)), seat, a)
    final_bid = _sel(n > 0, raised._replace(final=jnp.asarray(False), answers=queue, n_answers=n,
                                            cur=queue[0]), _end_auction(raised))  # fmt: skip
    bid = _next_bidder(_take_top(s, seat, a), seat)
    keep = _end_auction(s)  # the top bidder keeps its bid
    passed = _next_bidder(s._replace(out=s.out.at[seat].set(True)), seat)
    joined = _call(s, seat, s.top)
    joined = joined._replace(joined=joined.joined.at[seat].set(s.top_seat), out=joined.out.at[seat].set(True))
    joined = _next_bidder(joined, seat)
    normal = _sel(is_bid, _sel(s.final, final_bid, bid), _sel(a == PASS, _sel(s.final, keep, passed), joined))
    return _sel(s.n_answers > 0, _step_answer(s, a), normal)


def _record(s: Round, seat, k) -> Round:
    return s._replace(est=s.est.at[seat].set(k), total=s.total + k, pos=s.pos + 1)


def _end_fixed_estimation(s: Round) -> Round:
    """FIX-3: the highest estimate is Caller (first said wins ties); equal ones are With."""
    best = s.est.max()
    in_order = s.est[s.order] == best
    caller = s.order[jnp.argmax(in_order)]
    tie = (s.est == best) & (SEATS != caller)
    s = s._replace(caller=caller, c=best, with_w=jnp.where(tie, best, s.with_w).astype(I32))
    return _start_play(s)


def _open_super_call(s: Round, seat, a) -> Round:
    """FIX-4: clear the estimates and open an auction with the super call as top bid."""
    m1 = _i(-jnp.ones(4))
    s = s._replace(
        super_call=jnp.asarray(True), super_seat=seat, est=m1, total=_i(0), pos=_i(0),
        trump=_i(-1), top=_i(a), top_seat=seat, out=jnp.zeros(4, bool), hold=m1,
        calls=_i(-jnp.ones((4, 5))), stayed=m1,
    )  # fmt: skip
    s = _call(s, seat, a)
    s = s._replace(joined=m1, final=jnp.asarray(False), answers=_i(-jnp.ones(3)), n_answers=_i(0),
                   phase=_i(AUCTION))  # fmt: skip
    return _next_bidder(s, seat)


def _step_estimation(s: Round, a) -> Round:
    seat = s.cur
    k = a - ESTIMATE_BASE
    is_est = (a >= ESTIMATE_BASE) & (a < PLAY_BASE)
    # fixed-trump estimation, no Caller yet
    f = _record(s, seat, k)
    f = _sel(f.pos == 4, _end_fixed_estimation(f), f._replace(cur=f.order[jnp.minimum(f.pos, 3)]))
    fixed = _sel(is_est, f, _open_super_call(s, seat, a))
    # after an auction
    w = jnp.where((k == s.c) | (s.with_w[seat] >= 0), k, s.with_w[seat])  # EST-4, EST-2
    n = _record(s._replace(with_w=s.with_w.at[seat].set(w)), seat, k)
    n = _sel(n.pos == n.order_len, _start_play(n), n._replace(cur=n.order[jnp.minimum(n.pos, 3)]))
    return _sel(s.caller < 0, fixed, n)


def _score(s: Round) -> Round:
    """``score_round`` (SPEC §3) with the default rules."""
    est, won = s.est, s.won
    total = est.sum()
    over = total > 13
    win = est == won
    winners = win.sum()
    all_lose = winners == 0
    is_with = s.with_w >= 0
    calls = 1 + is_with.sum()
    mult = s.carry * jnp.where(calls >= CALLS_TO_DOUBLE, 2, 1)  # SCR-7, SCR-8
    mult = jnp.where(all_lose, 0, mult)  # SCR-9
    sign = jnp.where(win, 1, -1)
    is_caller = SEATS == s.caller
    called = is_caller | is_with
    is_super = called & (est >= SUPER_CALL_MIN)
    plain = jnp.where(win, WIN_BASE + est, -jnp.abs(est - won))
    base = jnp.where(s.dash | is_super, 0, plain)  # SCR-1 (Dash replaces base)
    dash = jnp.where(s.dash, sign * jnp.where(over, DASH_OVER, DASH_UNDER), 0)  # SCR-2
    sup = jnp.where(~s.dash & is_super, jnp.where(win, est * est, -((est * est) // 2)), 0)  # SCR-6
    call = jnp.where(called & ~is_super, sign * CALLER_WITH_BONUS, 0)  # SCR-3
    risks = jnp.where(s.last >= 0, jnp.abs(total - 13) // 2, 0)  # EST-7
    risk = jnp.where(SEATS == s.last, sign * RISK_POINTS * risks, 0)  # SCR-4
    only = jnp.where(((winners == 1) & win) | ((winners == 3) & ~win), sign * ONLY_BONUS, 0)  # SCR-5
    points = base + dash + call + risk + only + sup
    changed = (s.fixed_trump >= 0) & (s.trump >= 0) & (s.trump != s.fixed_trump)
    return s._replace(
        phase=_i(DONE), result=(points * mult).astype(I32), skipped=jnp.asarray(False),
        next_carry=jnp.where(all_lose, s.carry * 2, 1).astype(I32), trump_changed=changed,
    )  # fmt: skip


def _step_play(s: Round, a) -> Round:
    seat = s.cur
    card = a - PLAY_BASE
    led = jnp.where(s.tlen == 0, card // 13, s.led)
    off = (s.tlen > 0) & (card // 13 != led)
    s = s._replace(played=s.played.at[seat, card].set(True),
                   shown_out=s.shown_out.at[seat, led].set(s.shown_out[seat, led] | off),
                   hands=s.hands.at[seat, card].set(False), led=led.astype(I32),
                   trick=s.trick.at[s.tlen].set(card), tlen=s.tlen + 1)  # fmt: skip
    # PLY-3: highest trump wins, else the highest card of the led suit
    best, best_suit, offset = s.trick[0], s.led, 0
    for i in (1, 2, 3):
        ci = s.trick[i]
        suit = ci // 13
        beats = ((suit == best_suit) & (ci > best)) | ((suit != best_suit) & (suit == s.trump))
        best = jnp.where(beats, ci, best)
        best_suit = jnp.where(beats, suit, best_suit)
        offset = jnp.where(beats, i, offset)
    winner = ((s.leader + offset) % 4).astype(I32)
    done = s._replace(won=s.won.at[winner].add(1), ntricks=s.ntricks + 1, leader=winner, cur=winner,
                      tlen=_i(0), trick=_i(-jnp.ones(4)))  # fmt: skip
    done = _sel(done.ntricks == 13, _score(done), done)  # PLY-5
    return _sel(s.tlen == 4, done, s._replace(cur=(seat + 1) % 4))


def step(s: Round, a) -> Round:
    """Apply legal action ``a`` for the player to act; a DONE round is left unchanged."""
    a = _i(a)
    branches = [_step_dash, _step_auction, _step_estimation, _step_play, lambda s, a: s]
    out = jax.lax.switch(s.phase, branches, s, a)
    return out._replace(log_len=out.log_len + (s.phase != DONE))
