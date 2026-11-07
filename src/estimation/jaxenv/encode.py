"""One seat's raw view of a round, as floats, on the GPU (see encode_np for the same on CPU).

No hand evaluation and no derived strategy features: cards, the trick, public calls, estimates,
tricks won, scores, the legal moves. Seats are relative to the viewer, itself first.
"""

import jax
import jax.numpy as jnp

from .rules import AUCTION, DONE, PLAY, Round, SEATS, legal_mask, rank_of, trump_of

SIZE = 52 + 208 + 208 + 16 + 20 + 51 + 5 + 4 * 6 + 5 * 3 + 4 + 6 + 6 + 1 + 1 + 1 + 4 + 2 + 3 + 120


def _one_hot_seat(x, seat):
    return jax.nn.one_hot(jnp.where(x >= 0, (x - seat) % 4, 4), 5)


def encode(s: Round, seat) -> jax.Array:
    rel = (seat + SEATS) % 4
    in_auction = s.phase == AUCTION
    top_trump = trump_of(s.top)
    calls_top = s.calls[:, jnp.clip(top_trump, 0, 4)]
    with_a = (s.top_seat >= 0) & (SEATS != s.top_seat) & s.out & ~s.dash & (calls_top >= 0)
    is_with = jnp.where(in_auction, with_a, s.with_w >= 0)
    level = jnp.where(in_auction, jnp.where(with_a, rank_of(calls_top), 0), jnp.maximum(s.with_w, 0))
    passed = s.out & ~s.dash & ~is_with
    table = jnp.zeros((4, 52), bool)
    for i in range(4):
        t = (s.leader + i) % 4
        on = i < s.tlen
        table = table.at[t, jnp.maximum(s.trick[i], 0)].set(table[t, jnp.maximum(s.trick[i], 0)] | on)
    settled = (s.caller >= 0) & ((s.phase == PLAY) | (s.phase == DONE))
    last = jnp.where((s.phase == 2) | settled, s.last, -1)
    calls = jnp.where(s.calls >= 0, rank_of(s.calls) / 13.0, 0.0)
    scores = s.scores.astype(jnp.float32) / 200.0
    own = scores[seat]
    gaps = jnp.clip(scores - own, -5.0, 5.0)
    return jnp.concatenate([
        s.hands[seat].astype(jnp.float32),
        s.played[rel].astype(jnp.float32).ravel(),
        table[rel].astype(jnp.float32).ravel(),
        s.shown_out[rel].astype(jnp.float32).ravel(),
        calls[rel].ravel(),
        jax.nn.one_hot(jnp.where(s.top_seat >= 0, s.top - 1, 0), 51),
        _one_hot_seat(s.top_seat, seat),
        s.dash[rel].astype(jnp.float32),
        passed[rel].astype(jnp.float32),
        is_with[rel].astype(jnp.float32),
        level[rel] / 13.0,
        (s.est[rel] + 1) / 14.0,
        s.won[rel] / 13.0,
        _one_hot_seat(s.caller, seat),
        _one_hot_seat(last, seat),
        _one_hot_seat(jnp.where(s.phase == PLAY, s.leader, -1), seat),
        jax.nn.one_hot(s.phase, 4),
        jax.nn.one_hot(jnp.where(s.trump >= 0, s.trump, 5), 6),
        jax.nn.one_hot(jnp.where(s.fixed_trump >= 0, s.fixed_trump, 5), 6),
        s.super_call.astype(jnp.float32)[None],
        (s.round_no / 40.0)[None],
        jnp.clip(jnp.log2(s.carry.astype(jnp.float32)) / 10.0, 0.0, 2.0)[None],
        s.announced[rel].astype(jnp.float32),
        jnp.stack([jnp.clip(own, -5.0, 5.0), jnp.where(settled, (s.total - 13) / 13.0, 0.0)]),
        gaps[rel][1:],
        legal_mask(s).astype(jnp.float32),
    ]).astype(jnp.float32)
