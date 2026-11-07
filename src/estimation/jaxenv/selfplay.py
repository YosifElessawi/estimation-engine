"""Bitter-lesson self-play: one policy for every decision, trained to win the Bola, on the GPU.

    uv run python -m estimation.jaxenv.selfplay --out runs/run2 --logdir runs/tb

- Policy: one MLP over jaxenv.encode (one seat's raw view) for Dash Calls, bids, With, estimates
  and cards. No hand evaluation, imitation or search; it starts from nothing.
- Reward: only at the end of the Bola, zero-sum, chosen by --objective (``objective_rewards``):
  weighted (King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1, ties shared), king (King or nothing)
  or delta (score lead, scaled and clipped). A game past int32's range is dropped, not scored.
- Credit: TD(lambda) along each seat's own decisions (``td_lambda``), so a move is judged by how
  it changed the seat's expected final position, without waiting for the end of the game.
- Critic, training only: the same view plus the three hidden hands, so deal luck stays out of
  the advantage. It is thrown away afterwards; the policy never sees hidden cards.
- Auxiliary head on the policy: predicts the round's relative score, which teaches the shared
  layers about tricks and contracts. It affects neither the reward nor the moves' scores.
- Opponents: half the games are self-play; in the rest two opposite seats play a past version.
- Evaluation: King rate and mean position reward of greedy seats against a random player and
  the starting policy (25% and 0 are an even share). Real strength is measured by the arena (bots/bitter-net).
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")  # before JAX touches the GPU

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

REWARDS = (1.0, 0.2, -0.2, -1.0)  # King, Sub-King, Sub-Kooz, Kooz (estimation-arena's arena.positions)

from . import encode as E  # noqa: E402
from . import game as G  # noqa: E402
from . import rules as R  # noqa: E402

POOL_DEALS = 8  # deals per game per rollout
HIDDEN = 156  # the three other hands, relative to the acting seat


def _orth(scale):
    return jax.nn.initializers.orthogonal(scale)


def _mlp(key, size, hidden, depth):
    sizes = [size] + [hidden] * depth
    keys = jax.random.split(key, depth)
    return [{"w": _orth(jnp.sqrt(2.0))(k, (a, b)), "b": jnp.zeros(b)} for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def _head(key, hidden, out, scale):
    return {"w": _orth(scale)(key, (hidden, out)), "b": jnp.zeros(out)}


def init_nets(key, hidden: int, depth: int):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "policy": {"body": _mlp(k1, E.SIZE, hidden, depth), "policy": _head(k2, hidden, R.NUM_ACTIONS, 0.01),
                   "aux": _head(k3, hidden, 1, 1.0)},
        "critic": {"body": _mlp(k4, E.SIZE + HIDDEN, hidden, depth), "value": _head(k5, hidden, 1, 1.0)},
    }  # fmt: skip


def _body(layers, x):
    for layer in layers:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x


def _out(head, x):
    return x @ head["w"] + head["b"]


def apply_policy(net, x):
    h = _body(net["body"], x)
    return _out(net["policy"], h), _out(net["aux"], h)[..., 0]


def apply_critic(net, x, hidden_hands):
    h = _body(net["body"], jnp.concatenate([x, hidden_hands.astype(x.dtype)], -1))
    return _out(net["value"], h)[..., 0]


def position_rewards(scores):
    """[..., 4]: King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1 (arena.positions.REWARDS). Tied
    seats share the positions they occupy; the four always sum to 0."""
    above = (scores[..., None, :] > scores[..., :, None]).sum(-1)
    tied = (scores[..., None, :] == scores[..., :, None]).sum(-1)
    p = jnp.arange(4)[:, None] if scores.ndim == 1 else jnp.arange(4)[None, :, None]
    held = (above[..., None, :] <= p) & (p < (above + tied)[..., None, :])
    return (jnp.asarray(REWARDS)[..., None] * held).sum(-2) / tied


DELTA_SCALE = 200.0  # a score lead this large is worth the whole reward


def objective_rewards(scores, objective="weighted"):
    """[..., 4]: the end-of-Bola reward that training optimizes, summing to 0 across the seats."""
    if objective == "weighted":
        return position_rewards(scores)
    if objective == "king":
        top = scores == scores.max(-1, keepdims=True)
        return top / top.sum(-1, keepdims=True) - 0.25
    if objective == "delta":
        lead = scores - (scores.sum(-1, keepdims=True) - scores) / 3.0
        clipped = jnp.clip(lead / DELTA_SCALE, -1.0, 1.0)
        return clipped - clipped.mean(-1, keepdims=True)
    raise ValueError(f"unknown objective {objective}")


def round_rewards(r):
    """[B, 4]: each seat's round score minus the mean of the others, over 50, clipped (aux only)."""
    res = r.result.astype(jnp.float32)
    return jnp.clip((res - (res.sum(1, keepdims=True) - res) / 3.0) / 50.0, -5.0, 5.0)


def td_lambda(value, seat, finished, overflow, reward, lam):
    """Per-seat TD(lambda), for [T, B] decisions in time order; [T, B, 4] ``reward``.

    Each seat's decisions form their own chain. A finished game gives each seat its reward as
    the last target. A game past int32's range ends the chains unscored, so its decisions train
    only if a later decision of theirs bootstraps them. A seat's last decision in the rollout has
    no successor, so it is not trained on. Returns (advantage, has_target), both [T, B].
    """
    batch = value.shape[1]

    def back(carry, t):
        next_v, gae, has = carry
        next_v = jnp.where(t["finished"][:, None], t["reward"], next_v)
        gae = jnp.where((t["finished"] | t["overflow"])[:, None], 0.0, gae)
        has = jnp.where(t["finished"][:, None], True, jnp.where(t["overflow"][:, None], False, has))
        s = t["seat"][:, None]
        delta = jnp.take_along_axis(next_v, s, 1)[:, 0] - t["value"]
        adv = delta + lam * jnp.take_along_axis(gae, s, 1)[:, 0]
        adv = jnp.where(jnp.take_along_axis(has, s, 1)[:, 0], adv, 0.0)
        valid = jnp.take_along_axis(has, s, 1)[:, 0]
        next_v = jnp.put_along_axis(next_v, s, t["value"][:, None], 1, inplace=False)
        gae = jnp.put_along_axis(gae, s, adv[:, None], 1, inplace=False)
        has = jnp.put_along_axis(has, s, jnp.ones((batch, 1), bool), 1, inplace=False)
        return (next_v, gae, has), (adv, valid)

    init = (jnp.zeros((batch, 4)), jnp.zeros((batch, 4)), jnp.zeros((batch, 4), bool))
    xs = dict(value=value, seat=seat, finished=finished, overflow=overflow, reward=reward)
    _, (adv, valid) = jax.lax.scan(back, init, xs, reverse=True)
    return adv, valid


def _where_games(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond.reshape(-1, *[1] * (x.ndim - 1)), x, y), a, b)


def _hidden_hands(r, seat):
    return r.hands[(seat + 1) % 4].ravel(), r.hands[(seat + 2) % 4].ravel(), r.hands[(seat + 3) % 4].ravel()


def hidden_of(rounds, seat):
    parts = jax.vmap(_hidden_hands)(rounds, seat)
    return jnp.concatenate(parts, -1)


def choose(policy, opp, g, learner, key, greedy=False):
    seat = g.round.cur
    x = jax.vmap(E.encode)(g.round, seat)
    mask = jax.vmap(G.game_legal)(g)
    logits, _ = apply_policy(policy, x)
    opp_logits, _ = apply_policy(opp, x)
    mine = jnp.take_along_axis(learner, seat[:, None], 1)[:, 0]
    logits = jnp.where(mask, jnp.where(mine[:, None], logits, opp_logits), -1e9)
    action = jnp.argmax(logits, 1) if greedy else jax.random.categorical(key, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
    return action, logp, x, mine, mask.sum(1)


def _advance(g, used, base, pool, action):
    """Play ``action``. Returns the games (finished ones restarted), the deal counter, the round
    just played, whether it ended, and the game before any restart."""
    r = jax.vmap(R.step)(g.round, action)
    ended = (r.phase == R.DONE) & ~g.over
    before = g.round_no
    after = jax.vmap(G.game_step)(g, action, pool[base + used % POOL_DEALS])
    used = used + (after.round_no != before)
    stop = after.over | after.overflow
    fresh = jax.vmap(G.new_game)(pool[base + (used + 1) % POOL_DEALS])
    return _where_games(stop, fresh, after), used + stop, r, ended, after


def make_rollout(batch: int, steps: int, lam: float, objective: str = "weighted"):
    base = jnp.arange(batch) * POOL_DEALS

    def rollout(nets, opp, g, key):
        k_pool, k_seats, k_mix, k_steps = jax.random.split(key, 4)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        pairs = jax.random.bernoulli(k_seats, 0.5, (batch,))
        two = jnp.stack([pairs, ~pairs, pairs, ~pairs], 1)
        learner = jax.random.bernoulli(k_mix, 0.5, (batch,))[:, None] | two  # half pure self-play

        def one(carry, k):
            g, used = carry
            seat = g.round.cur
            phase = g.round.phase
            pre_super = g.round.caller < 0
            action, logp, x, mine, n_legal = choose(nets["policy"], opp, g, learner, k)
            hidden = hidden_of(g.round, seat)
            value = apply_critic(nets["critic"], x, hidden)
            g, used, r, ended, after = _advance(g, used, base, pool, action)
            played = ended & ~r.skipped
            made = jnp.take_along_axis(r.est == r.won, jnp.maximum(r.caller, 0)[:, None], 1)[:, 0]
            return (g, used), dict(
                x=x.astype(jnp.float16), hidden=hidden, action=action, logp=logp, value=value, seat=seat,
                train=mine & (n_legal > 1), ended=ended, round_rew=round_rewards(r),
                finished=after.over & ~after.overflow, overflow=after.overflow,
                reward=objective_rewards(after.scores, objective), phase=phase, pre_super=pre_super,
                skipped=ended & r.skipped, played=played, made=played & made,
                all_lose=played & ~(r.est == r.won).any(1), game_rounds=after.round_no,
            )  # fmt: skip

        (g, _), tr = jax.lax.scan(one, (g, jnp.zeros(batch, jnp.int32)), jax.random.split(k_steps, steps))
        adv, has = td_lambda(tr["value"], tr["seat"], tr["finished"], tr["overflow"], tr["reward"], lam)

        def back(carry, t):  # the round's relative score (aux target) and the game's result
            ret, has_round, final, has_final = carry
            ret = jnp.where(t["ended"][:, None], t["round_rew"], ret)
            has_round = has_round | t["ended"]
            final = jnp.where(t["finished"][:, None], t["reward"], final)
            has_final = (has_final | t["finished"]) & ~t["overflow"]
            s = t["seat"][:, None]
            return (ret, has_round, final, has_final), (
                jnp.take_along_axis(ret, s, 1)[:, 0], has_round,
                jnp.take_along_axis(final, s, 1)[:, 0], has_final,
            )  # fmt: skip

        zeros = (jnp.zeros((batch, 4)), jnp.zeros(batch, bool), jnp.zeros((batch, 4)), jnp.zeros(batch, bool))
        keys = ("ended", "round_rew", "finished", "reward", "overflow", "seat")
        _, (aux_target, aux_ok, final, has_final) = jax.lax.scan(back, zeros, {k: tr[k] for k in keys}, reverse=True)

        a, ph = tr["action"], tr["phase"]
        rate = lambda hit, of: hit.sum() / jnp.maximum(of.sum(), 1)  # noqa: E731
        is_bid = (a >= R.FIRST_BID) & (a <= R.LAST_BID)
        n_fin = tr["finished"].sum()
        target = adv + tr["value"]
        valid = has & tr["train"]
        vw = valid.astype(jnp.float32)
        var = lambda z: ((z - (z * vw).sum() / vw.sum()) ** 2 * vw).sum() / vw.sum()  # noqa: E731
        stats = dict(
            dash=rate((ph == R.DASH) & (a == R.DASH_CALL), ph == R.DASH),
            bid=rate((ph == R.AUCTION) & is_bid, ph == R.AUCTION),
            with_=rate((ph == R.AUCTION) & (a == R.WITH), ph == R.AUCTION),
            pass_=rate((ph == R.AUCTION) & (a == R.PASS), ph == R.AUCTION),
            super_=rate((ph == R.ESTIMATION) & is_bid, (ph == R.ESTIMATION) & tr["pre_super"]),
            skipped=rate(tr["skipped"], tr["ended"]),
            all_lose=rate(tr["all_lose"], tr["played"]),
            caller_made=rate(tr["made"], tr["played"]),
            games_done=n_fin,
            overflow=rate(tr["overflow"], tr["overflow"] | tr["finished"]),
            rounds_per_game=(tr["game_rounds"] * tr["finished"]).sum() / jnp.maximum(n_fin, 1),
            explained_var=1.0 - var(target - tr["value"]) / var(target),
            # how well the critic predicts the final position in a game's last rollout; 0.52 = no idea
            final_mse=((tr["value"] - final) ** 2 * has_final).sum() / jnp.maximum(has_final.sum(), 1),
        )
        data = dict(x=tr["x"], hidden=tr["hidden"], action=a, logp=tr["logp"], value=tr["value"], target=target,
                    valid=valid, aux_target=aux_target, aux_ok=aux_ok & tr["train"])  # fmt: skip
        flat = jax.tree.map(lambda v: v.reshape(batch * steps, *v.shape[2:]), data)
        adv = adv.reshape(batch * steps)
        w = flat["valid"]
        n = jnp.maximum(w.sum(), 1)
        mean = (adv * w).sum() / n
        flat["adv"] = (adv - mean) / (jnp.sqrt((((adv - mean) ** 2) * w).sum() / n) + 1e-6)
        return g, flat, stats

    return jax.jit(rollout)


def make_update(tx_policy, tx_critic, clip: float, aux_weight: float):
    def policy_loss(net, mb, ent):
        x = mb["x"].astype(jnp.float32)
        logits, aux = apply_policy(net, x)
        legal = x[:, -R.NUM_ACTIONS :] > 0.5
        logp_all = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
        logp = jnp.take_along_axis(logp_all, mb["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - mb["logp"])
        w = mb["valid"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        pg = (w * -jnp.minimum(ratio * mb["adv"], jnp.clip(ratio, 1 - clip, 1 + clip) * mb["adv"])).sum() / n
        entropy = (w * -(jnp.exp(logp_all) * jnp.where(legal, logp_all, 0.0)).sum(1)).sum() / n
        aw = mb["aux_ok"].astype(jnp.float32)
        aux_loss = (aw * (aux - mb["aux_target"]) ** 2).sum() / jnp.maximum(aw.sum(), 1.0)
        kl = (w * (mb["logp"] - logp)).sum() / n
        return pg - ent * entropy + aux_weight * aux_loss, dict(pg=pg, entropy=entropy, kl=kl, aux_loss=aux_loss)

    def critic_loss(net, mb):
        value = apply_critic(net, mb["x"].astype(jnp.float32), mb["hidden"])
        w = mb["valid"].astype(jnp.float32)
        return (w * (value - mb["target"]) ** 2).sum() / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def step(nets, opts, mb, ent):
        (_, info), g_p = jax.value_and_grad(policy_loss, has_aux=True)(nets["policy"], mb, ent)
        v_loss, g_c = jax.value_and_grad(critic_loss)(nets["critic"], mb)
        up_p, o_p = tx_policy.update(g_p, opts[0], nets["policy"])
        up_c, o_c = tx_critic.update(g_c, opts[1], nets["critic"])
        nets = {"policy": optax.apply_updates(nets["policy"], up_p), "critic": optax.apply_updates(nets["critic"], up_c)}
        return nets, (o_p, o_c), info | {"value_loss": v_loss}

    return step


def make_eval(batch: int, steps: int):
    """Greedy ``policy`` in two opposite seats vs ``opp``: (King rate, mean reward, games)."""
    base = jnp.arange(batch) * POOL_DEALS

    @jax.jit
    def run(policy, opp, key):
        k_pool, k_steps = jax.random.split(key)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        g = jax.vmap(G.new_game)(pool[base])
        learner = jnp.array([True, False, True, False])[None, :] ^ (jnp.arange(batch) % 2 == 0)[:, None]

        def one(carry, k):
            g, used, firsts, total, games = carry
            action, *_ = choose(policy, opp, g, learner, k, greedy=True)
            g, used, _, _, after = _advance(g, used, base, pool, action)
            done = after.over & ~after.overflow
            top = after.scores == after.scores.max(1, keepdims=True)
            share = (top * learner).sum(1) / top.sum(1)
            mean_r = (position_rewards(after.scores) * learner).sum(1) / 2.0
            return (g, used, firsts + (share * done).sum(), total + (mean_r * done).sum(), games + done.sum()), None

        init = (g, jnp.zeros(batch, jnp.int32), 0.0, 0.0, 0)
        (_, _, firsts, total, games), _ = jax.lax.scan(one, init, jax.random.split(k_steps, steps))
        n = jnp.maximum(games, 1)
        return firsts / n / 2.0, total / n, games

    return run


def save_npz(policy, path: Path, meta: dict) -> None:
    arrays = {}
    for k, layer in enumerate(policy["body"]):
        arrays[f"w{k}"], arrays[f"b{k}"] = np.asarray(layer["w"]), np.asarray(layer["b"])
    arrays["pw"], arrays["pb"] = np.asarray(policy["policy"]["w"]), np.asarray(policy["policy"]["b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part.npz")  # readers never see a half-written net
    np.savez_compressed(partial, **{k: v.astype(np.float32) for k, v in arrays.items()}, meta=json.dumps(meta))
    partial.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m estimation.jaxenv.selfplay", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--batch", type=int, default=4096, help="games at once")
    p.add_argument("--steps", type=int, default=128, help="decisions per game per iteration")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--objective", choices=("weighted", "king", "delta"), default="weighted", help="what the reward pays")
    p.add_argument("--lam", type=float, default=0.95, help="TD(lambda) per own decision")
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--entropy-end", type=float, default=0.01)
    p.add_argument("--aux", type=float, default=0.25, help="weight of the round-score head")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=32768)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--pool", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path)
    a = p.parse_args(argv)
    a.out = a.out.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    print("device:", jax.devices()[0], "| view:", E.SIZE, "| critic adds:", HIDDEN, flush=True)
    key = jax.random.key(a.seed)
    key, k_net = jax.random.split(key)
    nets = init_nets(k_net, a.hidden, a.depth)
    uniform = jax.tree.map(jnp.zeros_like, nets["policy"])  # all-zero logits: uniform over legal
    tx_policy = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opts = (tx_policy.init(nets["policy"]), tx_critic.init(nets["critic"]))
    first_policy = jax.device_get(nets["policy"])
    pool = [first_policy]
    rollout = make_rollout(a.batch, a.steps, a.lam, a.objective)
    update = make_update(tx_policy, tx_critic, a.clip, a.aux)
    evaluate = make_eval(512, 4000)
    key, k = jax.random.split(key)
    g = jax.vmap(G.new_game)(G.deal_pool(k, a.batch))
    writer = None
    if a.logdir:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(str(a.logdir.expanduser() / f"bitter-{a.out.name}"))
    log = (a.out / "log.jsonl").open("a")
    print(f"policy {sum(x.size for x in jax.tree.leaves(nets['policy'])):,} parameters, "
          f"{a.batch * a.steps:,} decisions per iteration", flush=True)  # fmt: skip
    t0, decisions = time.perf_counter(), 0
    for it in range(1, a.iterations + 1):
        key, k_opp, k_roll, k_perm, k_eval = jax.random.split(key, 5)
        opp = pool[int(jax.random.randint(k_opp, (), 0, len(pool)))]
        g, flat, stats = rollout(nets, opp, g, k_roll)
        n = a.batch * a.steps
        frac = (it - 1) / max(a.iterations - 1, 1)
        ent = a.entropy + (a.entropy_end - a.entropy) * frac
        for epoch in range(a.epochs):
            order = jnp.argsort(jax.random.uniform(jax.random.fold_in(k_perm, epoch), (n,)))
            for s in range(0, n - a.minibatch + 1, a.minibatch):
                mb = jax.tree.map(lambda x: x[order[s : s + a.minibatch]], flat)
                nets, opts, info = update(nets, opts, mb, ent)
        decisions += n
        if it % a.snapshot_every == 0:
            pool = (pool + [jax.device_get(nets["policy"])])[-a.pool :]
        row = {"iteration": it, "valid": float(flat["valid"].mean()), **{k: float(v) for k, v in info.items()},
               **{k: float(v) for k, v in stats.items()}, "decisions": decisions,
               "per_s": decisions / (time.perf_counter() - t0)}  # fmt: skip
        if it % a.eval_every == 0 or it == 1:
            king_r, pos_r, games = evaluate(nets["policy"], uniform, k_eval)
            king_s, pos_s, _ = evaluate(nets["policy"], first_policy, jax.random.fold_in(k_eval, 1))
            row |= {"king_vs_random": float(king_r), "pos_vs_random": float(pos_r), "king_vs_start": float(king_s),
                    "pos_vs_start": float(pos_s), "eval_games": int(games)}
            print(f"iter {it:>5}: King/position vs random {row['king_vs_random']:5.1%} {row['pos_vs_random']:+.2f} "
                  f"vs start {row['king_vs_start']:5.1%} {row['pos_vs_start']:+.2f} ({row['eval_games']} games) | "
                  f"entropy {row['entropy']:.2f} final_mse {row['final_mse']:.3f} ev {row['explained_var']:+.2f} kl {row['kl']:.4f} | "
                  f"dash {row['dash']:.2f} bid {row['bid']:.2f} with {row['with_']:.2f} pass {row['pass_']:.2f} super {row['super_']:.2f} | "
                  f"skip {row['skipped']:.2f} alllose {row['all_lose']:.2f} made {row['caller_made']:.2f} "
                  f"rounds/game {row['rounds_per_game']:.1f} ovf {row['overflow']:.2f} | {row['per_s']:,.0f}/s", flush=True)
            save_npz(nets["policy"], a.out / "net.npz", {"iteration": it, "args": {k: str(v) for k, v in vars(a).items()}})
        if it % 200 == 0:
            save_npz(nets["policy"], a.out / f"net-{it:05d}.npz", {"iteration": it})
        log.write(json.dumps(row) + "\n")
        log.flush()
        if writer:
            for k, v in row.items():
                writer.add_scalar(k, v, it)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
