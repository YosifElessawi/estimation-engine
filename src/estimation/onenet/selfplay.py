"""Step 5's self-play: the copied player learns final place from its own games, on the GPU.

    PYTHONPATH=src ~/jax-env/bin/python -m estimation.onenet.selfplay \\
        --config configs/onenet-selfplay.toml --out runs/onenet/selfplay

Essam Gouda's loop (estimation.jaxenv.selfplay, MIT licence): PPO with clipped ratios, credit by
TD(lambda) along each seat's own decisions, the judge (the view plus the three hidden hands) as
its critic, final place as the only reward, and half the games pure self-play while in the
others two opposite seats are played by an opponent. Step 5 adds
(docs/rl/hybrid-iteration-plan.md):

- the start: the player and the judge of the copy of Hybrid v4 (estimation.onenet.copy_hybrid),
  its move chances sharpened ``sharpen`` times: the copy learned them soft, and moves sampled
  from them as they are lose about a point a card more than its best move (0.2 at 4 times);
- the seven practice questions, answered by the games themselves, at ``practice_weight``;
- the three guards that keep the head start:
  1. gentle exploration (a small entropy bonus) and a pull back toward the copy's moves that
     fades out over ``anchor_loops``;
  2. the frozen copy as the opponent in ``copy_share`` of the loops, past versions in the rest;
  3. the player saved every ``save_every`` loops, so the check-ups against Hybrid v4 on the Mac
     pick the best one, not the last;
- the chart page (estimation.onenet.charts), and a checkpoint every ``save_every`` loops that a
  stopped run resumes from. Raising ``loops`` and running again continues a finished run.
"""

import argparse
import dataclasses
import functools
import json
import pickle
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from estimation.jaxenv import encode as E
from estimation.jaxenv import game as G
from estimation.jaxenv import rules as R
from estimation.jaxenv.selfplay import (
    POOL_DEALS,
    _advance,  # Essam's step-and-restart, reused as is
    objective_rewards,
    position_rewards,
    td_lambda,
)
from estimation.rl.config import section_from

from .charts import Charts
from .copy_hybrid import practice_losses
from .nets import (
    SLICES,
    Net,
    apply_judge,
    apply_player,
    hidden_hands,
    load_judge,
    load_player,
    save_player,
)

NUM_ACTIONS = R.NUM_ACTIONS
CHECKPOINT = "checkpoint.pkl"
# Settings a resumed run may change: how long it runs and how often it reports.
RESUMABLE = frozenset({"loops", "eval_every", "eval_games", "eval_steps", "save_every"})


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    start: str = "runs/onenet/copy-final/384x3"  # the copy: player.npz and judge.npz
    sharpen: float = 4.0  # the copy's move chances sharpened, so sampled moves stay near its best
    loops: int = 2000
    batch: int = 4096  # games played at once
    steps: int = 128  # decisions per game per loop
    lr: float = 1e-4
    clip: float = 0.2
    lam: float = 0.95
    entropy: float = 0.003  # guard 1: gentle exploration, falling to entropy_end
    entropy_end: float = 0.001
    anchor: float = 1.0  # guard 1: the pull toward the copy's moves, fading to 0
    anchor_loops: int = 1000
    practice_weight: float = 0.1
    epochs: int = 2
    minibatch: int = 32768
    snapshot_every: int = 25  # a past version joins the opponents
    pool: int = 16  # past versions kept as opponents
    copy_share: float = 0.25  # guard 2: loops whose opponent is the frozen copy
    eval_every: int = 25
    eval_games: int = 512
    eval_steps: int = 4000
    save_every: int = 100  # guard 3: the player kept for the check-ups
    objective: str = "weighted"  # final place: King +1, Sub king +0.2, Sub kooz -0.2, Kooz -1
    seed: int = 0

    def __post_init__(self) -> None:
        counts = (self.loops, self.batch, self.steps, self.epochs, self.minibatch, self.pool)
        if min(counts) < 1 or self.anchor_loops < 1:
            raise ValueError("selfplay counts (loops, batch, steps, ...) must be at least 1")
        if self.minibatch > self.batch * self.steps:
            raise ValueError("selfplay.minibatch must fit in one loop (batch x steps)")
        if min(self.snapshot_every, self.eval_every, self.save_every, self.eval_games) < 1:
            raise ValueError("selfplay.snapshot_every, eval_every, save_every and eval_games >= 1")
        if not 0 <= self.copy_share <= 1:
            raise ValueError("selfplay.copy_share must be from 0 to 1")
        if min(self.entropy, self.entropy_end, self.anchor, self.practice_weight) < 0:
            raise ValueError("selfplay.entropy, entropy_end, anchor and practice_weight >= 0")
        if self.sharpen <= 0:
            raise ValueError("selfplay.sharpen must be above 0")
        if self.objective not in ("weighted", "king", "delta"):
            raise ValueError("selfplay.objective must be weighted, king or delta")


def config_of(data: dict[str, Any]) -> SelfPlayConfig:
    """A self-play config from TOML data: one ``[selfplay]`` section."""
    unknown = set(data) - {"selfplay"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}: self-play has only [selfplay]")
    section = data.get("selfplay")
    if not isinstance(section, dict):
        raise ValueError("a self-play config needs a [selfplay] section")
    return section_from(SelfPlayConfig(), cast(dict[str, Any], section), "selfplay")


def load_config(path: Path) -> SelfPlayConfig:
    return config_of(tomllib.loads(path.read_text()))


# The games.


def sharpened(player: Net, times: float) -> Net:
    """``player`` with its move output times ``times``: the same best move, surer chances."""
    move = {"w": player["policy"]["w"] * times, "b": player["policy"]["b"] * times}
    return {**player, "policy": move}


def holder_of(r: R.Round, seat: jax.Array) -> jax.Array:
    """[52] int8: which other seat (0-2 for relative seats 1-3) holds each card, else -1."""
    hands = r.hands[(seat + jnp.arange(1, 4)) % 4]
    return jnp.where(hands.any(0), jnp.argmax(hands, 0), -1).astype(jnp.int8)


def choose(
    player: Net, opp: Net, g: G.Game, learner: jax.Array, key: jax.Array, greedy: bool = False
) -> tuple[jax.Array, ...]:
    """Each game's move: the player's for the learner's seats, the opponent's for the others."""
    seat = g.round.cur
    x = jax.vmap(E.encode)(g.round, seat)
    mask = jax.vmap(G.game_legal)(g)
    logits, _ = apply_player(player, x)
    opp_logits, _ = apply_player(opp, x)
    mine = jnp.take_along_axis(learner, seat[:, None], 1)[:, 0]
    logits = jnp.where(mask, jnp.where(mine[:, None], logits, opp_logits), -1e9)
    action = jnp.argmax(logits, 1) if greedy else jax.random.categorical(key, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
    return action, logp, x, mine, mask.sum(1)


def round_labels(trace: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """[T, B, ...]: each decision's practice answers, from the decider's side (the decider
    first): its round's ``tricks`` and ``made`` (-1 when unknown: a skipped round or one that
    ended after this loop), ``round_score`` (``round_ok``), and its Bola's ``final`` scores
    (``final_ok``; never for a game dropped past int32's range)."""
    batch = trace["seat"].shape[1]

    def back(carry: tuple[jax.Array, ...], t: dict[str, jax.Array]):
        tricks, made, score, has_round, final, has_final = carry
        end = t["ended"][:, None]
        skip = t["skipped"][:, None]
        tricks = jnp.where(end, jnp.where(skip, -1, t["won"]), tricks)
        made = jnp.where(end, jnp.where(skip, -1, t["made"]), made)
        score = jnp.where(end, t["result"], score)
        has_round = has_round | t["ended"]
        final = jnp.where(t["finished"][:, None], t["scores"], final)
        has_final = (has_final | t["finished"]) & ~t["overflow"]
        order = (t["seat"][:, None] + jnp.arange(4)) % 4

        def rel(a: jax.Array) -> jax.Array:
            return jnp.take_along_axis(a, order, 1)

        known = has_round[:, None]
        out = (
            jnp.where(known, rel(tricks), -1),
            jnp.where(known, rel(made), -1),
            rel(score),
            has_round,
            rel(final),
            has_final,
        )
        return (tricks, made, score, has_round, final, has_final), out

    none = jnp.full((batch, 4), -1, jnp.int32)
    zeros = jnp.zeros((batch, 4), jnp.int32)
    no = jnp.zeros(batch, bool)
    keys = ("ended", "skipped", "won", "made", "result", "finished", "scores", "overflow", "seat")
    _, out = jax.lax.scan(
        back, (none, none, zeros, no, zeros, no), {k: trace[k] for k in keys}, reverse=True
    )
    names = ("tricks", "made", "round_score", "round_ok", "final", "final_ok")
    return dict(zip(names, out, strict=True))


Rollout = Callable[
    [dict[str, Net], Net, G.Game, jax.Array], tuple[G.Game, dict[str, Any], dict[str, Any]]
]


@functools.cache
def make_rollout(config: SelfPlayConfig) -> Rollout:
    batch, steps = config.batch, config.steps
    base = jnp.arange(batch) * POOL_DEALS

    def rollout(nets: dict[str, Net], opp: Net, g: G.Game, key: jax.Array):
        k_pool, k_seats, k_mix, k_steps = jax.random.split(key, 4)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        pairs = jax.random.bernoulli(k_seats, 0.5, (batch,))
        two = jnp.stack([pairs, ~pairs, pairs, ~pairs], 1)
        learner = jax.random.bernoulli(k_mix, 0.5, (batch,))[:, None] | two  # half pure self-play

        def one(carry: tuple[G.Game, jax.Array], k: jax.Array):
            g, used = carry
            seat, phase, pre_super = g.round.cur, g.round.phase, g.round.caller < 0
            action, logp, x, mine, n_legal = choose(nets["player"], opp, g, learner, k)
            holder = jax.vmap(holder_of)(g.round, seat)
            value = apply_judge(nets["judge"], x, hidden_hands(holder))
            g, used, r, ended, after = _advance(g, used, base, pool, action)
            played = ended & ~r.skipped
            hit = r.est == r.won
            caller_hit = jnp.take_along_axis(hit, jnp.maximum(r.caller, 0)[:, None], 1)[:, 0]
            return (g, used), {
                "x": x.astype(jnp.float16),
                "holder": holder,
                "action": action,
                "logp": logp,
                "value": value,
                "seat": seat,
                "train": mine & (n_legal > 1),
                "ended": ended,
                "skipped": ended & r.skipped,
                "won": r.won,
                "made": hit.astype(jnp.int32),
                "result": r.result,
                "finished": after.over & ~after.overflow,
                "overflow": after.overflow,
                "scores": after.scores,
                "reward": objective_rewards(after.scores, config.objective),
                "phase": phase,
                "pre_super": pre_super,
                "played": played,
                "caller_made": played & caller_hit,
            }

        start = (g, jnp.zeros(batch, jnp.int32))
        (g, _), tr = jax.lax.scan(one, start, jax.random.split(k_steps, steps))
        adv, has = td_lambda(
            tr["value"], tr["seat"], tr["finished"], tr["overflow"], tr["reward"], config.lam
        )
        labels = round_labels(tr)
        valid = has & tr["train"]
        target = adv + tr["value"]
        stats = _stats(tr, labels, target, valid, config.objective)
        data = {
            "x": tr["x"],
            "holder": tr["holder"],
            "action": tr["action"],
            "logp": tr["logp"],
            "target": target,
            "valid": valid,
            **labels,
        }
        flat = jax.tree.map(lambda v: v.reshape(batch * steps, *v.shape[2:]), data)
        adv = adv.reshape(batch * steps)
        w = flat["valid"]
        n = jnp.maximum(w.sum(), 1)
        mean = (adv * w).sum() / n
        flat["adv"] = (adv - mean) / (jnp.sqrt((((adv - mean) ** 2) * w).sum() / n) + 1e-6)
        return g, flat, stats

    return cast(Rollout, jax.jit(rollout))


def _stats(
    tr: dict[str, jax.Array],
    labels: dict[str, jax.Array],
    target: jax.Array,
    valid: jax.Array,
    objective: str,
) -> dict[str, jax.Array]:
    def rate(hit: jax.Array, of: jax.Array) -> jax.Array:
        return hit.sum() / jnp.maximum(of.sum(), 1)

    a, ph = tr["action"], tr["phase"]
    is_bid = (a >= R.FIRST_BID) & (a <= R.LAST_BID)
    vw = valid.astype(jnp.float32)

    def var(z: jax.Array) -> jax.Array:
        return ((z - (z * vw).sum() / vw.sum()) ** 2 * vw).sum() / vw.sum()

    place = objective_rewards(labels["final"], objective)[..., 0]
    ok = labels["final_ok"]
    return {
        "dash": rate((ph == R.DASH) & (a == R.DASH_CALL), ph == R.DASH),
        "bid": rate((ph == R.AUCTION) & is_bid, ph == R.AUCTION),
        "with_": rate((ph == R.AUCTION) & (a == R.WITH), ph == R.AUCTION),
        "pass_": rate((ph == R.AUCTION) & (a == R.PASS), ph == R.AUCTION),
        "super_": rate((ph == R.ESTIMATION) & is_bid, (ph == R.ESTIMATION) & tr["pre_super"]),
        "skipped": rate(tr["skipped"], tr["ended"]),
        "caller_made": rate(tr["caller_made"], tr["played"]),
        "games_done": tr["finished"].sum(),
        "overflow": rate(tr["overflow"], tr["overflow"] | tr["finished"]),
        "explained_var": 1.0 - var(target - tr["value"]) / var(target),
        "final_mse": (((tr["value"] - place) ** 2) * ok).sum() / jnp.maximum(ok.sum(), 1),
    }


Update = Callable[..., tuple[dict[str, Net], tuple[Any, Any], dict[str, jax.Array]]]


@functools.cache
def make_update(config: SelfPlayConfig) -> tuple[optax.GradientTransformation, Update]:
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config.lr))

    def player_loss(player: Net, copy: Net, mb: dict[str, Any], ent: jax.Array, pull: jax.Array):
        x = mb["x"].astype(jnp.float32)
        legal = x[:, -NUM_ACTIONS:] > 0.5
        logits, q = apply_player(player, x)
        logp_all = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
        logp = jnp.take_along_axis(logp_all, mb["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - mb["logp"])
        w = mb["valid"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        adv = mb["adv"]
        clipped = jnp.clip(ratio, 1 - config.clip, 1 + config.clip) * adv
        pg = (w * -jnp.minimum(ratio * adv, clipped)).sum() / n
        entropy = (w * -(jnp.exp(logp_all) * jnp.where(legal, logp_all, 0.0)).sum(1)).sum() / n
        copy_logits, _ = apply_player(copy, x)
        copy_logp = jax.nn.log_softmax(jnp.where(legal, copy_logits, -1e9))
        gap = jnp.where(legal, copy_logp - logp_all, 0.0)
        distance = (w * (jnp.exp(copy_logp) * gap).sum(1)).sum() / n  # KL(copy || player)
        parts = practice_losses(q, mb)
        holder = mb["holder"].astype(jnp.int32)
        known = holder >= 0
        guess = q[:, SLICES["holder"]].reshape(-1, 52, 3).argmax(-1)
        named = ((guess == holder) & known).sum() / jnp.maximum(known.sum(), 1)
        practice = sum(parts.values(), jnp.zeros(()))
        total = pg - ent * entropy + pull * distance + config.practice_weight * practice
        info = {
            "pg": pg,
            "entropy": entropy,
            "kl": (w * (mb["logp"] - logp)).sum() / n,
            "distance_from_copy": distance,
            "holders_named": named,
            **{f"practice_{name}": loss for name, loss in parts.items()},
        }
        return total, info

    def judge_loss(judge: Net, mb: dict[str, Any]) -> jax.Array:
        value = apply_judge(judge, mb["x"].astype(jnp.float32), hidden_hands(mb["holder"]))
        w = mb["valid"].astype(jnp.float32)
        return (w * (value - mb["target"]) ** 2).sum() / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def update(
        nets: dict[str, Net], opts: tuple[Any, Any], copy: Net, mb: dict[str, Any], ent, pull
    ):
        grad_player = jax.value_and_grad(player_loss, has_aux=True)
        (_, info), g_p = grad_player(nets["player"], copy, mb, ent, pull)
        v_loss, g_j = jax.value_and_grad(judge_loss)(nets["judge"], mb)
        up_p, o_p = tx.update(g_p, opts[0], nets["player"])
        up_j, o_j = tx.update(g_j, opts[1], nets["judge"])
        nets = {
            "player": cast(Net, optax.apply_updates(nets["player"], up_p)),
            "judge": cast(Net, optax.apply_updates(nets["judge"], up_j)),
        }
        return nets, (o_p, o_j), {**info, "value_loss": v_loss}

    return tx, cast(Update, update)


Evaluate = Callable[[Net, Net, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]


@functools.cache
def make_eval(games: int, steps: int) -> Evaluate:
    """The greedy player in two opposite seats against ``opp`` in the others: its King rate per
    seat (25% is even), its mean final place reward (0 is even) and the games finished."""
    base = jnp.arange(games) * POOL_DEALS

    def run(player: Net, opp: Net, key: jax.Array):
        k_pool, k_steps = jax.random.split(key)
        pool = G.deal_pool(k_pool, games * POOL_DEALS)
        g = jax.vmap(G.new_game)(pool[base])
        seats = jnp.array([True, False, True, False])[None, :]
        learner = seats ^ (jnp.arange(games) % 2 == 0)[:, None]

        def one(carry: tuple[Any, ...], k: jax.Array):
            g, used, firsts, total, done_n = carry
            action, *_ = choose(player, opp, g, learner, k, greedy=True)
            g, used, _, _, after = _advance(g, used, base, pool, action)
            done = after.over & ~after.overflow
            top = after.scores == after.scores.max(1, keepdims=True)
            share = (top * learner).sum(1) / top.sum(1)
            place = (position_rewards(after.scores) * learner).sum(1) / 2.0
            carry = (
                g,
                used,
                firsts + (share * done).sum(),
                total + (place * done).sum(),
                done_n + done.sum(),
            )
            return carry, None

        init = (g, jnp.zeros(games, jnp.int32), 0.0, 0.0, 0)
        (_, _, firsts, total, done_n), _ = jax.lax.scan(one, init, jax.random.split(k_steps, steps))
        n = jnp.maximum(done_n, 1)
        return firsts / n / 2.0, total / n, done_n

    return cast(Evaluate, jax.jit(run))


# The run.


def _say(text: str) -> None:
    print(text, flush=True)


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def _rows(flat: dict[str, Any], rows: jax.Array) -> dict[str, Any]:
    return jax.tree.map(lambda v: v[rows], flat)


def loop_line(row: dict[str, Any]) -> str:
    """One checked loop in a line: strength against the copy and the guards' numbers."""
    return (
        f"King vs the copy {row['king_vs_copy']:.1%} (place {row['place_vs_copy']:+.3f}) | "
        f"distance from the copy {row['distance_from_copy']:.4f}, "
        f"pull {row['anchor_weight']:.2f} | holders {row['holders_named']:.0%} | "
        f"judge {row['final_mse']:.3f} | callers make {row['caller_made']:.0%} | "
        f"{row['decisions_per_s']:,.0f} decisions/s, {_duration(row['minutes_left'] * 60)} left"
    )


def save_checkpoint(out: Path, state: dict[str, Any]) -> None:
    tmp = out / (CHECKPOINT + ".tmp")
    with tmp.open("wb") as file:
        pickle.dump(jax.device_get(state), file)
    tmp.replace(out / CHECKPOINT)


def _changed(saved: dict[str, Any], config: SelfPlayConfig) -> list[str]:
    now = dataclasses.asdict(config)
    return sorted(k for k in now if k not in RESUMABLE and saved.get(k) != now[k])


def _resume(out: Path, config: SelfPlayConfig) -> dict[str, Any] | None:
    path = out / CHECKPOINT
    if not path.exists():
        return None
    with path.open("rb") as file:
        state = pickle.load(file)  # our own checkpoint, written by save_checkpoint
    changed = _changed(state["config"], config)
    if changed:
        raise SystemExit(
            f"{out} holds a run with other settings ({', '.join(changed)}); use a new --out"
        )
    return state


def _numbers(row: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in row.items() if isinstance(v, int | float)}


def run(config: SelfPlayConfig, out: Path) -> dict[str, Any]:
    """Play and learn for ``config.loops`` loops, resuming from ``out`` when it holds a run."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    start = Path(config.start)
    copy, copy_meta = load_player(start / "player.npz")
    copy = sharpened(copy, config.sharpen)  # the start, the frozen opponent and the pull's target
    tx, update = make_update(config)
    rollout = make_rollout(config)
    evaluate = make_eval(config.eval_games, config.eval_steps)
    state = _resume(out, config)
    if state is None:
        judge, _ = load_judge(start / "judge.npz")
        nets = {"player": copy, "judge": judge}
        opts = (tx.init(nets["player"]), tx.init(nets["judge"]))
        key = jax.random.key(config.seed)
        past: list[Net] = []
        first = 1
        (out / "log.jsonl").write_text("")
    else:
        nets = jax.tree.map(jnp.asarray, state["nets"])
        opts = jax.tree.map(jnp.asarray, state["opts"])
        key = jax.random.wrap_key_data(jnp.asarray(state["key"]))
        past = [jax.tree.map(jnp.asarray, p) for p in state["past"]]
        first = state["loop"] + 1
        rows = (out / "log.jsonl").read_text().splitlines()
        kept = [r for r in rows if json.loads(r)["loop"] <= state["loop"]]
        (out / "log.jsonl").write_text("".join(r + "\n" for r in kept))
        _say(f"[selfplay] resuming after loop {state['loop']}")
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    _say(f"[selfplay] device {jax.devices()[0]}; from {start} (copy pass {copy_meta.get('epoch')})")
    key, k_games = jax.random.split(key)
    g = jax.vmap(G.new_game)(G.deal_pool(k_games, config.batch))
    charts = Charts(out / "charts")
    log = (out / "log.jsonl").open("a")
    n = config.batch * config.steps
    started, decisions, row = time.time(), 0, {}
    for loop in range(first, config.loops + 1):
        key, k_opp, k_roll, k_perm, k_eval = jax.random.split(key, 5)
        use_copy = not past or float(jax.random.uniform(k_opp)) < config.copy_share
        if use_copy:
            opp = copy
        else:
            opp = past[int(jax.random.randint(jax.random.fold_in(k_opp, 1), (), 0, len(past)))]
        g, flat, stats = rollout(nets, opp, g, k_roll)
        frac = (loop - 1) / max(config.loops - 1, 1)
        ent = config.entropy + (config.entropy_end - config.entropy) * frac
        pull = config.anchor * max(0.0, 1.0 - (loop - 1) / config.anchor_loops)
        infos = []
        for epoch in range(config.epochs):
            order = jnp.argsort(jax.random.uniform(jax.random.fold_in(k_perm, epoch), (n,)))
            for s in range(0, n - config.minibatch + 1, config.minibatch):
                mb = _rows(flat, order[s : s + config.minibatch])
                nets, opts, info = update(nets, opts, copy, mb, ent, pull)
                infos.append(info)
        decisions += n
        if loop % config.snapshot_every == 0:
            past = ([*past, jax.device_get(nets["player"])])[-config.pool :]
        fetched = jax.device_get(infos)
        elapsed = time.time() - started
        per_loop = elapsed / (loop - first + 1)
        row = {
            "loop": loop,
            "opponent": "copy" if use_copy else "past",
            "entropy_weight": ent,
            "anchor_weight": pull,
            **{k: float(np.mean([i[k] for i in fetched])) for k in fetched[0]},
            **{k: float(v) for k, v in jax.device_get(stats).items()},
            "decisions_per_s": decisions / elapsed,
            "minutes_left": per_loop * (config.loops - loop) / 60,
        }
        if loop % config.eval_every == 0 or loop == first:
            king, place, games = jax.device_get(evaluate(nets["player"], copy, k_eval))
            row |= {
                "king_vs_copy": float(king),
                "place_vs_copy": float(place),
                "eval_games": int(games),
            }
            _say(f"[selfplay] loop {loop}/{config.loops}: {loop_line(row)}")
        log.write(json.dumps(row) + "\n")
        log.flush()
        charts.write(loop, _numbers(row))
        if loop % config.save_every == 0 or loop == config.loops:
            meta = {"loop": loop, "start": str(start), "config": dataclasses.asdict(config)}
            save_player(out / "snapshots" / f"player-{loop:05d}.npz", nets["player"], meta)
            save_checkpoint(
                out,
                {
                    "loop": loop,
                    "nets": nets,
                    "opts": opts,
                    "key": jax.random.key_data(key),
                    "past": past,
                    "config": dataclasses.asdict(config),
                },
            )
    log.close()
    charts.close()
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.selfplay", description=__doc__
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
