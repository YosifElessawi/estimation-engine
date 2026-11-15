"""Replay the same games in the engine and in JAX and require identical play.

    uv run python -m estimation.jaxenv.check --rounds 20000 --games 200

Every step of every game must agree on the phase, the player to act and the full set of
legal actions; every round must agree on its scores, next carry, skip and trump change; every
game must agree on its final scores. Moves come from a mix of uniform random play (which
reaches rare rules: four Dash Calls, super calls, AUC-6) and HeuristicBot play.
"""

import argparse
import random
import time

import jax
import jax.numpy as jnp
import numpy as np

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot
from estimation.engine import Bola, Phase, Round
from estimation.engine.cards import deal as engine_deal
from estimation.engine.actions import DASH_CALL, FIRST_BID, LAST_BID, PASS, WITH
from estimation.engine.round import FIXED_ORDER, OPEN_ROUNDS

from . import encode as E
from . import encode_np as EN
from . import game as G
from . import rules as R

_vstep = jax.jit(jax.vmap(R.step))
_vlegal = jax.jit(jax.vmap(R.legal_mask))
_vinit = jax.jit(jax.vmap(R.init_round))
_vencode = jax.jit(jax.vmap(E.encode))
_gstep = jax.jit(jax.vmap(G.game_step))
_glegal = jax.jit(jax.vmap(G.game_legal))
_gnew = jax.jit(jax.vmap(G.new_game))


def bits(hands) -> np.ndarray:
    """[4, 52] bool from four 52-bit hand masks."""
    return np.array([[(h >> c) & 1 for c in range(52)] for h in hands], dtype=bool)


class Mismatch(AssertionError):
    pass


COVERAGE: dict[str, int] = {}


def _seen(name: str) -> None:
    COVERAGE[name] = COVERAGE.get(name, 0) + 1


def _auction_heavy(rng: random.Random):
    """Many Dash Calls, Withs, stays and small raises: reaches the AUC-6 13-total ban."""

    def act(obs, legal):
        if obs.phase is Phase.DASH:
            return DASH_CALL if rng.random() < 0.45 and DASH_CALL in legal else legal[0]
        if obs.phase is Phase.AUCTION:
            bids = [a for a in legal if FIRST_BID <= a <= LAST_BID]
            roll = rng.random()
            if WITH in legal and roll < 0.4:
                return WITH
            if PASS in legal and roll < 0.65:
                return PASS
            if bids:
                return bids[min(int(rng.expovariate(0.7)), len(bids) - 1)]
        return rng.choice(legal)

    return act


def _chooser(rng: random.Random):
    heuristic = HeuristicBot()
    roll = rng.random()
    if roll < 0.35:
        return lambda obs, legal: rng.choice(legal)
    if roll < 0.6:
        return _auction_heavy(rng)
    return lambda obs, legal: heuristic.act(obs, legal) if rng.random() > 0.1 else rng.choice(legal)


def _bid(tricks: int, trump: int) -> int:
    return FIRST_BID + (tricks - 4) * 5 + trump


_H, _S, _C, _N = 2, 3, 0, 4
_D, _ND = DASH_CALL, 0
# Scripted auctions under the default rules (round 1, the engine's deal for seed 0): the Dash
# decisions, then auction moves. They follow the engine's own AUC-3 to AUC-6 and L1-L10 tests.
SCRIPTS = (
    ((_ND, _ND, _ND, _D), (_bid(4, _H), WITH, WITH, _bid(5, _H), PASS)),  # an answer barred (AUC-6)
    ((_ND, _ND, _ND, _D), (_bid(4, _H), WITH, WITH, _bid(5, _H), PASS, WITH)),
    ((_ND, _ND, _D, _D), (_bid(4, _H), _bid(5, _H), PASS, _bid(9, _H))),  # a chooser estimates
    ((_ND, _ND, _ND, _D), (_bid(4, _H), WITH, _bid(5, _H))),  # estimators left: never barred
    ((_ND, _ND, _ND, _D), (_bid(4, _H), WITH, _bid(5, _H), PASS, PASS)),  # L5: choosers off 13
    ((_ND,) * 4, (_bid(4, _H), _bid(5, _H), WITH, PASS, PASS, _bid(6, _H), WITH)),  # L1
    ((_ND,) * 4, (_bid(4, _H), _bid(5, _H), WITH, PASS, PASS, _bid(6, _H), PASS)),  # L1 stay
    ((_ND,) * 4, (_bid(4, _H), WITH, _bid(5, _S), PASS, PASS, PASS)),  # L10: With lost to a trump
    ((_D, _D, _D, _ND), (_bid(6, _H),)),  # L3: lone bidder
    ((_ND,) * 4, (_bid(5, _H), PASS, PASS, PASS)),  # L2
)


def _scripted(prefix, rng: random.Random):
    """Play ``prefix`` first, then uniformly at random."""
    queue = list(prefix)
    return lambda obs, legal: queue.pop(0) if queue else rng.choice(legal)


def random_round(rng: random.Random) -> Round:
    """An open round 1-13, or a fixed-trump round with up to two replays, random carry/scores."""
    if rng.random() < 0.6:
        round_no, fixed = rng.randint(1, OPEN_ROUNDS), None
    else:
        position = rng.randrange(5)
        round_no, fixed = OPEN_ROUNDS + 1 + position + rng.randint(0, 2), FIXED_ORDER[position]
    hands = engine_deal(rng.randrange(2**31), round_no)
    carry = rng.choice((1, 1, 2, 4))
    scores = tuple(rng.randint(-200, 300) for _ in range(4))
    return Round.new(round_no, hands, fixed, carry, scores)


def check_rounds(n: int, seed: int, batch: int = 512, scripts=(), views: bool = False) -> int:
    """``n`` random rounds; with ``scripts``, one batch of those scripted auctions instead
    (each played out at random from several seeds)."""
    rng = random.Random(seed)
    steps = 0
    for start in range(0, n, batch):
        if scripts:
            plan = [scripts[k % len(scripts)] for k in range(n)]
            rounds = [Round.new(1, engine_deal(0, 1)) for _ in plan]
            choose = [_scripted((*dash, *moves), random.Random(seed + k)) for k, (dash, moves) in enumerate(plan)]
        else:
            rounds = [random_round(rng) for _ in range(min(batch, n - start))]
            choose = [_chooser(rng) for _ in rounds]
        s = _vinit(
            jnp.array([r.round_no for r in rounds]),
            jnp.array(np.stack([bits(r.hands) for r in rounds])),
            jnp.array([-1 if r.fixed_trump is None else r.fixed_trump for r in rounds]),
            jnp.array([r.carry for r in rounds]),
            jnp.array([r.scores for r in rounds]),
        )
        while True:
            phase, cur, mask = (np.asarray(x) for x in (s.phase, s.cur, _vlegal(s)))
            if views:
                seen = np.asarray(_vencode(s, s.cur))
            actions = np.zeros(len(rounds), np.int32)
            live = False
            for i, r in enumerate(rounds):
                where = f"round {start + i} (no {r.round_no}, fixed {r.fixed_trump}) step {len(r.log)}"
                if r.phase is Phase.DONE:
                    if phase[i] != R.DONE:
                        raise Mismatch(f"{where}: JAX not done")
                    continue
                live = True
                legal = r.legal_actions()
                if phase[i] != int(r.phase) or cur[i] != r.current_player:
                    raise Mismatch(f"{where}: phase/turn {phase[i]},{cur[i]} vs {int(r.phase)},{r.current_player}")
                if sorted(np.flatnonzero(mask[i]).tolist()) != legal:
                    raise Mismatch(f"{where}: legal {np.flatnonzero(mask[i]).tolist()} vs {legal}")
                if r._answers:
                    _seen("answering a raise (AUC-3)")
                if r.phase is Phase.AUCTION:
                    open_bids = [b for b in range(FIRST_BID, r._max_bid + 1) if b > r._top]
                    if r._answers and (WITH not in legal or PASS not in legal):
                        _seen("an answer barred by the 13 rule (AUC-6)")
                    elif r._final and not r._answers and (
                        PASS not in legal or any(b not in legal for b in open_bids)
                    ):
                        _seen("a keep or raise barred by the 13 rule (AUC-6)")
                obs = r.observation(r.current_player)
                if views:
                    want = EN.encode(obs, legal)
                    if want.shape != seen[i].shape or not np.allclose(seen[i], want, atol=1e-5):
                        bad = np.flatnonzero(np.abs(seen[i] - want) > 1e-5)[:8].tolist()
                        raise Mismatch(f"{where}: encoders differ at {bad}: {seen[i][bad]} vs {want[bad]}")
                a = choose[i](obs, legal)
                r.step(a)
                actions[i] = a
                steps += 1
            if not live:
                break
            s = _vstep(s, jnp.array(actions))
        res, nc, sk, tc = (np.asarray(x) for x in (s.result, s.next_carry, s.skipped, s.trump_changed))
        for i, r in enumerate(rounds):
            out = r.result()
            got = (res[i].tolist(), int(nc[i]), bool(sk[i]), bool(tc[i]))
            want = (list(out.scores), out.next_carry, out.skipped, out.trump_changed)
            if got != want:
                raise Mismatch(f"round {start + i}: result {got} vs {want}")
            _seen("skipped round" if out.skipped else "scored round")
            if r._super:
                _seen("super call (FIX-4)")
            if sum(r._dash) == 4:
                _seen("four Dash Calls (DSH-5)")
            if sum(r._dash) == 3:
                _seen("lone bidder (AUC-5)")
            if out.trump_changed:
                _seen("fixed trump changed (FIX-5)")
            if not out.skipped and out.all_lose:
                _seen("everyone lost (SCR-9)")
            if not out.skipped and any(x.is_super for x in out.seats):
                _seen("super-call scoring (SCR-6)")
            if not out.skipped and out.calls >= 3:
                _seen("doubled by 3+ calls (SCR-8)")
    return steps


def check_games(n: int, seed: int) -> int:
    """Whole Bolas: the engine deals; JAX gets each next round's hands from the engine's deal."""
    rng = random.Random(seed)
    seeds = [rng.randrange(2**31) for _ in range(n)]
    bolas = [Bola(x) for x in seeds]
    choose = [_chooser(rng) for _ in bolas]
    g = _gnew(jnp.array(np.stack([bits(b.round.hands) for b in bolas])))
    steps = 0
    out_of_range = [False] * n  # the engine's carry or a score ever left int32's safe range
    while True:
        mask = np.asarray(_glegal(g))
        actions = np.zeros(n, np.int32)
        nxt = np.zeros((n, 4, 52), bool)
        live = False
        for i, b in enumerate(bolas):
            if b.is_over:
                continue
            live = True
            legal = b.legal_actions()
            if sorted(np.flatnonzero(mask[i]).tolist()) != legal:
                raise Mismatch(f"game {i} round {b.round_no}: legal differs")
            a = choose[i](b.observation(b.current_player), legal)
            before = b.round_no
            b.step(a)
            actions[i] = a
            steps += 1
            if b.round_no != before or b.is_over:
                if b.carry > G.CARRY_CAP or max(abs(x) for x in b.scores) > G.SCORE_LIMIT:
                    out_of_range[i] = True
            if not b.is_over and b.round_no != before:
                nxt[i] = bits(b.round.hands)
        if not live:
            break
        g = _gstep(g, jnp.array(actions), jnp.array(nxt))
    scores, over, rno = np.asarray(g.scores), np.asarray(g.over), np.asarray(g.round_no)
    overflow = np.asarray(g.overflow)
    for i, b in enumerate(bolas):
        if bool(overflow[i]) != out_of_range[i]:
            raise Mismatch(f"game {i}: overflow flag {bool(overflow[i])} vs engine range {out_of_range[i]}")
        if overflow[i]:
            _seen("game past int32 range, flagged (exploding doubles)")
            continue
        _seen("full game compared")
        if not over[i] or scores[i].tolist() != list(b.scores) or rno[i] != b.round_no:
            raise Mismatch(f"game {i}: {scores[i].tolist()} r{rno[i]} vs {list(b.scores)} r{b.round_no}")
    return steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m estimation.jaxenv.check", description=__doc__.split("\n")[0])
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--views", type=int, default=2000, help="rounds also comparing the two encoders")
    a = p.parse_args(argv)
    t = time.perf_counter()
    steps = check_rounds(a.rounds, a.seed)
    print(f"PASS {a.rounds} rounds, {steps} steps identical ({time.perf_counter() - t:.0f} s)", flush=True)
    steps = check_rounds(len(SCRIPTS) * 20, a.seed, batch=10_000, scripts=SCRIPTS)
    print(f"PASS {len(SCRIPTS)} scripted auctions x 20 random play-outs, {steps} steps identical", flush=True)
    steps = check_rounds(a.views, a.seed + 2, views=True)
    print(f"PASS {a.views} rounds with encoders, {steps} views identical", flush=True)
    t = time.perf_counter()
    steps = check_games(a.games, a.seed + 1)
    print(f"PASS {a.games} full games, {steps} steps identical ({time.perf_counter() - t:.0f} s)")
    for name, count in sorted(COVERAGE.items()):
        print(f"  {count:>7}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
