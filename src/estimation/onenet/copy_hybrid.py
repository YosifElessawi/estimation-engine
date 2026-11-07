"""Step 5's copy: the player learns to play like Hybrid v4 from the recording, and the judge and
the practice questions learn from the same games (docs/rl/hybrid-iteration-plan.md, step 5).

    uv run python -m estimation.onenet.copy_hybrid \
        --config configs/onenet-copy.toml --out runs/onenet/copy

On the home server's GPU, from the repo, with its JAX env (no PyTorch needed):

    PYTHONPATH=src ~/jax-env/bin/python -m estimation.onenet.copy_hybrid --config ... --out ...

- Move: at every valued decision the values (the teacher's for cards, BiddingNet's for bids)
  become chances, softmax(value / temperature) over the legal moves, and the move output learns
  them. At the card decisions without values, the move output learns Hybrid v4's own move, at
  ``played_weight`` (0 leaves those decisions to the practice questions and the judge).
- Practice questions: each learns its answer from the real game, at ``practice_weight`` each.
- Judge: learns each decision's final place (King +1, Sub king +0.2, Sub kooz -0.2, Kooz -1,
  ties shared) from the view and the three hidden hands.
- A share of the Bolas is held out and scores every epoch: the points the player's top card
  (and the best of its top 3) loses by the teacher's values, next to Hybrid v4's card net's own
  top card; the points its top bid loses by BiddingNet's values; each practice question against
  a blind guess; and the judge against always saying 0.

Each size in ``sizes`` is copied in turn into ``<out>/<size>/``: the player of the epoch whose
top moves lose the fewest points on the held-out decisions (cards and bids), and the judge of
the epoch with its lowest held-out error, which comes early, as it learns the training Bolas'
final places by heart (``player.npz``, ``judge.npz``, ``metrics.json``).
"""

import argparse
import dataclasses
import json
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

from estimation.jaxenv.selfplay import position_rewards
from estimation.rl.config import section_from

from .nets import (
    SLICES,
    Net,
    apply_judge,
    apply_player,
    hidden_hands,
    init_judge,
    init_player,
    parse_size,
    save_judge,
    save_player,
)
from .shards import KIND_BID, KIND_CARD, KIND_PLAYED, NUM_ACTIONS, Shard, load_data

SHOWN = slice(52 + 208 + 208, 52 + 208 + 208 + 16)  # the view's suits each seat showed out of
BATCH_FIELDS = (
    "kind",
    "action",
    "x",
    "values",
    "holder",
    "tricks",
    "made",
    "round_score",
    "final",
)
Batch = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CopyConfig:
    data: str = "runs/onenet/data"
    sizes: tuple[str, ...] = ("384x3", "512x4")
    epochs: int = 12
    batch: int = 4096
    lr: float = 1e-3
    lr_end: float = 3e-5
    temperature: float = 2.0  # points: values become chances by softmax(value / temperature)
    held_out: float = 0.05  # share of the Bolas kept out of training, to score on
    practice_weight: float = 0.1  # each practice question's weight, the move's being 1
    played_weight: float = 0.0  # Hybrid v4's own card moves, at the decisions without values
    seed: int = 0

    def __post_init__(self) -> None:
        for size in self.sizes:
            parse_size(size)
        if self.epochs < 1 or self.batch < 1:
            raise ValueError("copy.epochs and copy.batch must be at least 1")
        if not 0 < self.held_out < 1:
            raise ValueError("copy.held_out must be between 0 and 1")
        if self.temperature <= 0 or self.lr <= 0 or self.lr_end <= 0:
            raise ValueError("copy.temperature, copy.lr and copy.lr_end must be above 0")
        if self.practice_weight < 0 or self.played_weight < 0:
            raise ValueError("copy.practice_weight and copy.played_weight must be 0 or more")


def config_of(data: dict[str, Any]) -> CopyConfig:
    """A copy config from TOML data: one ``[copy]`` section."""
    unknown = set(data) - {"copy"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}: a copy has only [copy]")
    section = data.get("copy")
    if not isinstance(section, dict):
        raise ValueError("a copy config needs a [copy] section")
    return section_from(CopyConfig(), cast(dict[str, Any], section), "copy")


def load_config(path: Path) -> CopyConfig:
    return config_of(tomllib.loads(path.read_text()))


# The losses.


def _mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(jnp.float32)
    return (values * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def _cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Mean over the labels that are not -1."""
    logp = jax.nn.log_softmax(logits)
    picked = jnp.take_along_axis(logp, jnp.maximum(labels, 0)[..., None], -1)[..., 0]
    return _mean(-picked, labels >= 0)


def voids(holder: jax.Array) -> jax.Array:
    """[B, 3, 4]: whether each other seat (relative 1-3) holds no card of each suit."""
    by_suit = holder.reshape(-1, 4, 13)
    held = by_suit[:, None, :, :] == jnp.arange(3, dtype=holder.dtype)[None, :, None, None]
    return ~held.any(-1)


def king_share(final: jax.Array) -> jax.Array:
    """[B]: the player's share of first place (ties shared)."""
    top = final == final.max(-1, keepdims=True)
    return top[:, 0] / top.sum(-1)


def place_reward(final: jax.Array) -> jax.Array:
    """[B]: the player's final place reward, King +1 to Kooz -1, ties shared."""
    return position_rewards(final)[:, 0]


def legal_of(mb: Batch) -> jax.Array:
    return mb["x"][:, -NUM_ACTIONS:] > 0.5


def move_loss(logits: jax.Array, mb: Batch, temperature: float) -> jax.Array:
    """[B]: cross-entropy from the values' chances to the move output, over the legal moves."""
    legal = legal_of(mb)
    logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
    target = jax.nn.softmax(jnp.where(legal, mb["values"] / temperature, -1e9))
    return -(target * jnp.where(legal, logp, 0.0)).sum(-1)


def practice_losses(q: jax.Array, mb: Batch) -> dict[str, jax.Array]:
    """Each practice question's loss. Answers that are -1 are unknown and skipped; the round
    score and the King chance also skip rows whose ``round_ok`` or ``final_ok`` is False (in
    self-play, a round or Bola that ends after the loop). The copy knows every answer."""
    b = q.shape[0]
    round_ok = mb.get("round_ok", jnp.ones(b, bool))
    final_ok = mb.get("final_ok", jnp.ones(b, bool))
    holder = mb["holder"].astype(jnp.int32)
    tricks = mb["tricks"].astype(jnp.int32)
    made = mb["made"].astype(jnp.int32)
    score = jnp.clip(mb["round_score"].astype(jnp.float32) / 50.0, -5.0, 5.0)
    everywhere = jnp.ones((b, 3, 4), bool)
    void_logits = q[:, SLICES["void"]].reshape(b, 3, 4)
    return {
        "holder": _cross_entropy(q[:, SLICES["holder"]].reshape(b, 52, 3), holder),
        "my_tricks": _cross_entropy(q[:, SLICES["my_tricks"]], tricks[:, 0]),
        "made": _mean(
            optax.sigmoid_binary_cross_entropy(
                q[:, SLICES["made"]], (made == 1).astype(jnp.float32)
            ),
            made >= 0,
        ),
        "other_tricks": _cross_entropy(
            q[:, SLICES["other_tricks"]].reshape(b, 3, 14), tricks[:, 1:]
        ),
        "void": _mean(
            optax.sigmoid_binary_cross_entropy(void_logits, voids(holder).astype(jnp.float32)),
            everywhere,
        ),
        "king": _mean(
            optax.sigmoid_binary_cross_entropy(q[:, SLICES["king"]][:, 0], king_share(mb["final"])),
            final_ok,
        ),
        "round_score": _mean(((q[:, SLICES["round_score"]] - score) ** 2).mean(-1), round_ok),
    }


def judge_loss(judge: Net, mb: Batch) -> jax.Array:
    x = mb["x"].astype(jnp.float32)
    value = apply_judge(judge, x, hidden_hands(mb["holder"]))
    return jnp.mean((value - place_reward(mb["final"])) ** 2)


Step = Callable[[Net, Net, Any, Any, Batch], tuple[Net, Net, Any, Any, dict[str, jax.Array]]]


def played_loss(logits: jax.Array, mb: Batch) -> jax.Array:
    """[B]: cross-entropy from the move played to the move output, over the legal moves."""
    logp = jax.nn.log_softmax(jnp.where(legal_of(mb), logits, -1e9))
    return -jnp.take_along_axis(logp, mb["action"].astype(jnp.int32)[:, None], -1)[:, 0]


def make_step(tx: optax.GradientTransformation, config: CopyConfig) -> Step:
    def player_loss(player: Net, mb: Batch) -> tuple[jax.Array, dict[str, jax.Array]]:
        logits, q = apply_player(player, mb["x"].astype(jnp.float32))
        valued = mb["kind"] != KIND_PLAYED
        move = _mean(move_loss(logits, mb, config.temperature), valued)
        played = _mean(played_loss(logits, mb), ~valued)
        parts = practice_losses(q, mb)
        practice = sum(parts.values(), jnp.zeros(()))
        total = move + config.played_weight * played + config.practice_weight * practice
        return total, {"move": move, "played": played, **parts}

    @jax.jit
    def step(player: Net, judge: Net, opt_p: Any, opt_j: Any, mb: Batch):
        (_, info), g_p = jax.value_and_grad(player_loss, has_aux=True)(player, mb)
        j_loss, g_j = jax.value_and_grad(judge_loss)(judge, mb)
        up_p, opt_p = tx.update(g_p, opt_p, player)
        up_j, opt_j = tx.update(g_j, opt_j, judge)
        player = cast(Net, optax.apply_updates(player, up_p))
        judge = cast(Net, optax.apply_updates(judge, up_j))
        return player, judge, opt_p, opt_j, {**info, "judge": j_loss}

    return step


# The held-out scores.


def blind_guesses(data: Shard, rows: np.ndarray) -> dict[str, float]:
    """What a blind guess says, from the training rows: the mean or most common answer."""
    tricks = data["tricks"][rows].astype(np.float64)
    made = data["made"][rows]
    final = data["final"][rows]
    top = final == final.max(1, keepdims=True)
    return {
        "my_tricks": float(tricks[:, 0][tricks[:, 0] >= 0].mean()),
        "other_tricks": float(tricks[:, 1:][tricks[:, 1:] >= 0].mean()),
        "made": float((made[made >= 0] == 1).mean()),
        "king": float((top[:, 0] / top.sum(1)).mean()),
        "round_score": float(np.clip(data["round_score"][rows] / 50.0, -5, 5).mean()),
    }


def make_scores(
    temperature: float,
) -> Callable[[Net, Net, Batch, dict[str, Any]], dict[str, jax.Array]]:
    """Sums over a batch of held-out rows; ``held_out_scores`` turns them into the scores."""

    @jax.jit
    def scores(player: Net, judge: Net, mb: Batch, blind: dict[str, Any]) -> dict[str, jax.Array]:
        x = mb["x"].astype(jnp.float32)
        b = x.shape[0]
        legal = legal_of(mb)
        logits, q = apply_player(player, x)
        valued = mb["kind"] != KIND_PLAYED
        card, bid = mb["kind"] == KIND_CARD, mb["kind"] == KIND_BID
        values = mb["values"]
        best = jnp.where(legal, values, -jnp.inf).max(-1)
        top3 = jnp.argsort(-jnp.where(legal, logits, -jnp.inf), axis=-1)[:, :3]
        top_values = jnp.where(
            jnp.take_along_axis(legal, top3, -1), jnp.take_along_axis(values, top3, -1), -jnp.inf
        )
        lose1 = jnp.where(valued, best - top_values[:, 0], 0.0)
        lose3 = jnp.where(valued, best - top_values.max(-1), 0.0)
        holder = mb["holder"].astype(jnp.int32)
        known = holder >= 0
        shown = x[:, SHOWN].reshape(b, 4, 4)[:, 1:, :] > 0.5  # [B, other seat, suit]
        candidates = (~shown).sum(1)[:, jnp.arange(52) // 13]  # [B, card]
        tricks = mb["tricks"].astype(jnp.int32)
        played = tricks[:, 0] >= 0
        expected = jax.nn.softmax(q[:, SLICES["my_tricks"]]) @ jnp.arange(14.0)
        others = jax.nn.softmax(q[:, SLICES["other_tricks"]].reshape(b, 3, 14)) @ jnp.arange(14.0)
        made = mb["made"].astype(jnp.int32)
        made_p = jax.nn.sigmoid(q[:, SLICES["made"]])
        is_void = voids(holder)
        void_p = jax.nn.sigmoid(q[:, SLICES["void"]].reshape(b, 3, 4))
        share = king_share(mb["final"])
        king_p = jax.nn.sigmoid(q[:, SLICES["king"]][:, 0])
        score = jnp.clip(mb["round_score"].astype(jnp.float32) / 50.0, -5.0, 5.0)
        target = place_reward(mb["final"])
        value = apply_judge(judge, x, hidden_hands(mb["holder"]))
        return {
            "valued": valued.sum(),
            "move_loss": (move_loss(logits, mb, temperature) * valued).sum(),
            "cards": card.sum(),
            "card_lose1": (lose1 * card).sum(),
            "card_lose3": (lose3 * card).sum(),
            "card_best": ((lose1 < 1e-4) & card).sum(),
            "bids": bid.sum(),
            "bid_lose1": (lose1 * bid).sum(),
            "bid_best": ((lose1 < 1e-4) & bid).sum(),
            "holder_n": known.sum(),
            "holder_right": (
                (q[:, SLICES["holder"]].reshape(b, 52, 3).argmax(-1) == holder) & known
            ).sum(),
            "holder_blind": (known / jnp.maximum(candidates, 1)).sum(),
            "my_tricks_n": played.sum(),
            "my_tricks_off": (jnp.abs(expected - tricks[:, 0]) * played).sum(),
            "my_tricks_blind": (jnp.abs(blind["my_tricks"] - tricks[:, 0]) * played).sum(),
            "other_tricks_n": (tricks[:, 1:] >= 0).sum(),
            "other_tricks_off": (jnp.abs(others - tricks[:, 1:]) * (tricks[:, 1:] >= 0)).sum(),
            "other_tricks_blind": (
                jnp.abs(blind["other_tricks"] - tricks[:, 1:]) * (tricks[:, 1:] >= 0)
            ).sum(),
            "made_n": (made >= 0).sum(),
            "made_right": (((made_p > 0.5) == (made == 1)) & (made >= 0)).sum(),
            "made_blind": (((blind["made"] > 0.5) == (made == 1)) & (made >= 0)).sum(),
            "void_n": jnp.asarray(b * 12),
            "void_right": ((void_p > 0.5) == is_void).sum(),
            "void_blind": (shown == is_void).sum(),  # out only where the table showed it
            "king_n": jnp.asarray(b),
            "king_brier": ((king_p - share) ** 2).sum(),
            "king_blind": ((blind["king"] - share) ** 2).sum(),
            "score_n": jnp.asarray(b * 4),
            "score_sq": ((q[:, SLICES["round_score"]] - score) ** 2).sum(),
            "score_blind": ((blind["round_score"] - score) ** 2).sum(),
            "judge_n": jnp.asarray(b),
            "judge_sq": ((value - target) ** 2).sum(),
            "judge_blind": (target**2).sum(),
        }

    return scores


def epoch_line(losses: dict[str, float], held: dict[str, float]) -> str:
    """One epoch in a line: the move's loss and the main held-out scores."""
    return (
        f"move loss {losses['move']:.3f} (held out {held['move_loss']:.3f}) | "
        f"top card loses {held['card_lose1']:.2f} (best of top 3 {held['card_lose3']:.2f}), "
        f"best {held['card_best']:.0%} | top bid loses {held['bid_lose1']:.2f} | "
        f"holders {held['holder']:.0%} (blind {held['holder_blind']:.0%}) | "
        f"judge {held['judge_mse']:.3f} (blind {held['judge_blind']:.3f})"
    )


def _ratio(sums: dict[str, float], top: str, bottom: str) -> float:
    return sums[top] / max(sums[bottom], 1.0)


def held_out_scores(sums: dict[str, float]) -> dict[str, float]:
    return {
        "move_loss": _ratio(sums, "move_loss", "valued"),
        "lose1": (sums["card_lose1"] + sums["bid_lose1"]) / max(sums["cards"] + sums["bids"], 1.0),
        "card_lose1": _ratio(sums, "card_lose1", "cards"),
        "card_lose3": _ratio(sums, "card_lose3", "cards"),
        "card_best": _ratio(sums, "card_best", "cards"),
        "bid_lose1": _ratio(sums, "bid_lose1", "bids"),
        "bid_best": _ratio(sums, "bid_best", "bids"),
        "holder": _ratio(sums, "holder_right", "holder_n"),
        "holder_blind": _ratio(sums, "holder_blind", "holder_n"),
        "my_tricks_off": _ratio(sums, "my_tricks_off", "my_tricks_n"),
        "my_tricks_blind": _ratio(sums, "my_tricks_blind", "my_tricks_n"),
        "other_tricks_off": _ratio(sums, "other_tricks_off", "other_tricks_n"),
        "other_tricks_blind": _ratio(sums, "other_tricks_blind", "other_tricks_n"),
        "made": _ratio(sums, "made_right", "made_n"),
        "made_blind": _ratio(sums, "made_blind", "made_n"),
        "void": _ratio(sums, "void_right", "void_n"),
        "void_blind": _ratio(sums, "void_blind", "void_n"),
        "king_brier": _ratio(sums, "king_brier", "king_n"),
        "king_blind": _ratio(sums, "king_blind", "king_n"),
        "round_score_mse": _ratio(sums, "score_sq", "score_n"),
        "round_score_blind": _ratio(sums, "score_blind", "score_n"),
        "judge_mse": _ratio(sums, "judge_sq", "judge_n"),
        "judge_blind": _ratio(sums, "judge_blind", "judge_n"),
    }


def card_net_scores(data: Shard, rows: np.ndarray) -> dict[str, float]:
    """Hybrid v4's card net's own top card at the held-out valued card decisions."""
    rows = rows[(data["kind"][rows] == KIND_CARD) & (data["card_net"][rows] >= 0)]
    if not len(rows):
        return {}
    values = data["values"][rows]
    legal = data["x"][rows][:, -NUM_ACTIONS:] > 0.5
    best = np.where(legal, values, -np.inf).max(1)
    lose = best - values[np.arange(len(rows)), data["card_net"][rows].astype(np.int64)]
    return {"card_lose1": float(lose.mean()), "card_best": float((lose < 1e-4).mean())}


# The run.


def split(data: Shard, share: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Training and held-out rows, holding out whole Bolas."""
    bolas = np.unique(data["bola"])
    count = min(len(bolas) - 1, max(1, round(share * len(bolas))))
    if count < 1:
        raise ValueError("the copy needs at least 2 recorded Bolas: one to hold out")
    held = np.random.default_rng(seed).choice(bolas, size=count, replace=False)
    is_held = np.isin(data["bola"], held)
    return np.flatnonzero(~is_held), np.flatnonzero(is_held)


def batch_of(data: Shard, rows: np.ndarray) -> Batch:
    return {name: data[name][rows] for name in BATCH_FIELDS}


def _say(text: str) -> None:
    print(text, flush=True)


def copy_size(
    config: CopyConfig, data: Shard, train: np.ndarray, held: np.ndarray, size: str, out: Path
) -> dict[str, Any]:
    """Copy Hybrid v4 into one player size; returns its kept epoch's scores."""
    hidden, depth = parse_size(size)
    k_player, k_judge = jax.random.split(jax.random.key(config.seed))
    player, judge = init_player(k_player, hidden, depth), init_judge(k_judge, hidden, depth)
    per_epoch = max(1, len(train) // config.batch)
    schedule = optax.cosine_decay_schedule(
        config.lr, per_epoch * config.epochs, alpha=config.lr_end / config.lr
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_p, opt_j = tx.init(player), tx.init(judge)
    step = make_step(tx, config)
    scores = make_scores(config.temperature)
    blind = blind_guesses(data, train)
    rng = np.random.default_rng(config.seed)
    folder = out / size
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / "log.jsonl").open("w")
    kept: dict[str, Any] = {}
    judge_kept: dict[str, Any] = {}
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        order = rng.permutation(train)
        infos: list[dict[str, jax.Array]] = []
        for k in range(per_epoch):
            mb = batch_of(data, np.sort(order[k * config.batch : (k + 1) * config.batch]))
            player, judge, opt_p, opt_j, info = step(player, judge, opt_p, opt_j, mb)
            infos.append(info)
        fetched = jax.device_get(infos)
        losses = {key: float(np.mean([i[key] for i in fetched])) for key in fetched[0]}
        sums: dict[str, float] = {}
        for k in range(0, len(held), config.batch):
            batch_sums = jax.device_get(
                scores(player, judge, batch_of(data, held[k : k + config.batch]), blind)
            )
            for key, value in batch_sums.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        held_scores = held_out_scores(sums)
        row = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "train": losses,
            "held_out": held_scores,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        line = epoch_line(losses, held_scores)
        _say(f"[copy {size}] epoch {epoch}/{config.epochs}: {line} | {row['seconds']:.0f} s")
        meta = {"size": size, "epoch": epoch, "config": dataclasses.asdict(config)}
        if not kept or held_scores["lose1"] < kept["held_out"]["lose1"]:
            kept = row
            save_player(folder / "player.npz", player, meta)
        if not judge_kept or held_scores["judge_mse"] < judge_kept["judge_mse"]:
            judge_kept = {"epoch": epoch, "judge_mse": held_scores["judge_mse"]}
            save_judge(folder / "judge.npz", judge, meta)
    log.close()
    result = {
        "size": size,
        "kept": kept,
        "judge": judge_kept,
        "card_net": card_net_scores(data, held),
        "blind": blind,
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(config: CopyConfig, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    _say(f"[copy] device {jax.devices()[0]}; loading {config.data}")
    data = load_data(Path(config.data))
    train, held = split(data, config.held_out, config.seed)
    _say(f"[copy] {len(train):,} training rows, {len(held):,} held out")
    results = [copy_size(config, data, train, held, size, out) for size in config.sizes]
    summary = {"rows": {"train": len(train), "held_out": len(held)}, "sizes": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.copy_hybrid", description=__doc__
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
