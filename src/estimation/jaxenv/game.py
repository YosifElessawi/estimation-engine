"""A full game (a Bola) in JAX, mirroring ``estimation.engine.bola``: 13 open rounds, then
the fixed-trump rounds, with the carry between rounds and fixed-trump replays (FIX-5).

``advance`` starts the next round from given hands, so tests can feed the engine's deals;
``deal`` draws fresh hands from a JAX key for training.

Scores are int32 on the GPU, the engine's are unbounded Python ints. The carry doubles with
every skipped or all-lose round (GAM-4), so a game where all four seats keep failing can leave
int32's range; real play never comes near it. Such a game is flagged ``overflow`` (carry above
``CARRY_CAP`` or a score past 2**30): from then on its scores are not the engine's, and
training should drop it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rules import DONE, FIXED_ORDER, MAX_FIXED_REPLAYS, OPEN_ROUNDS, Round, init_round, legal_mask, step

I32 = jnp.int32
_FIXED = jnp.array(FIXED_ORDER, I32)
CARRY_CAP = 1 << 20
SCORE_LIMIT = 1 << 30


class Game(NamedTuple):
    round: Round
    scores: jax.Array  # [4] game scores
    carry: jax.Array
    fixed_index: jax.Array  # index in FIXED_ORDER of the next fixed trump to complete
    replays: jax.Array
    round_no: jax.Array
    over: jax.Array  # bool
    overflow: jax.Array  # bool: left int32's safe range; scores no longer the engine's


def deal(key) -> jax.Array:
    """[4, 52] bool: a uniformly random deal (one sort of 52 uniform keys)."""
    seat_of = jnp.repeat(jnp.arange(4), 13)[jnp.argsort(jax.random.uniform(key, (52,)))]
    return seat_of[None, :] == jnp.arange(4)[:, None]


def deal_pool(key, n: int) -> jax.Array:
    """[n, 4, 52]: ``n`` deals at once. Dealing is the costly part of a step on the GPU, and a
    game needs a new deal only every ~65 moves, so batches draw from a pool."""
    return jax.vmap(deal)(jax.random.split(key, n))


def new_game(hands) -> Game:
    return Game(
        round=init_round(1, hands),
        scores=jnp.zeros(4, I32),
        carry=jnp.asarray(1, I32),
        fixed_index=jnp.asarray(0, I32),
        replays=jnp.asarray(0, I32),
        round_no=jnp.asarray(1, I32),
        over=jnp.asarray(False),
        overflow=jnp.asarray(False),
    )


def advance(g: Game, hands) -> Game:
    """``Bola._end_round`` for a DONE round: add its scores, set the carry, move through the
    fixed trumps, and deal the next round from ``hands`` (ignored once the game is over)."""
    r = g.round
    scores = g.scores + r.result
    fixed = r.fixed_trump >= 0
    replay = fixed & r.trump_changed & (g.replays < MAX_FIXED_REPLAYS)
    replays = g.replays + replay
    fixed_index = g.fixed_index + (fixed & ~replay)
    over = fixed_index >= len(FIXED_ORDER)
    round_no = jnp.where(over, g.round_no, g.round_no + 1)
    fixed_trump = jnp.where(round_no > OPEN_ROUNDS, _FIXED[jnp.minimum(fixed_index, 4)], -1)
    overflow = g.overflow | (r.next_carry > CARRY_CAP) | (jnp.abs(scores) > SCORE_LIMIT).any()
    carry = jnp.minimum(r.next_carry, CARRY_CAP)
    nxt = init_round(round_no, hands, fixed_trump, carry, scores)
    keep = jax.tree.map(lambda a, b: jnp.where(over, a, b), r, nxt)
    return Game(keep, scores, carry, fixed_index, replays, round_no, over, overflow)


def game_step(g: Game, action, next_hands) -> Game:
    """Play ``action``; when it ends the round, start the next one from ``next_hands``."""
    r = step(g.round, action)
    moved = g._replace(round=r)
    ended = (r.phase == DONE) & ~g.over
    return jax.tree.map(lambda a, b: jnp.where(ended, a, b), advance(moved, next_hands), moved)


game_legal = lambda g: legal_mask(g.round) & ~g.over  # noqa: E731
