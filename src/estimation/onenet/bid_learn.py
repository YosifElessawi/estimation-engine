"""Step 6's learning (docs/rl/hybrid-iteration-plan.md): the player learns its bids from the bid
teacher's lessons and keeps everything else as it is.

    PYTHONPATH=src ~/jax-env/bin/python -m estimation.onenet.bid_learn \\
        --config configs/onenet-bids-1-learn.toml --out runs/onenet/bids-1/learn

- **Bids** (``aimed``). A lesson whose finalist sat its whole exam says by how many points
  that alternative led the player's own move (``estimation.onenet.lessons``). The target is
  the starting player's move chances with the finalist's chance set against the move's: the
  move's chance times exp(tilt), the tilt being the lead over ``temperature``. A finalist that
  led overtakes the move, however sure the player was of it: self-play left the player sure of
  moves it had stopped trying the alternatives to. A finalist that trailed is lowered to that
  level if it stood above it and otherwise left alone. Every other option, and every lesson
  without a whole exam, keeps the starting player's chances.
- **The noise in a lead** (``tilts``). A measured lead is the true one plus the luck of its
  exam's deals. Each is first pulled toward its phase's average lead by its share of noise
  (how the leads spread across the lessons, less their errors, says how much of a lead is
  real), and what noise is left comes off the tilt (minus variance / (2 temperature squared)),
  so that on average an alternative's odds against the move are those of its true lead: one
  no better than the move does not overtake it by luck.
- **Everything else stays.** On card decisions (a training set of the player's own games,
  ``estimation.onenet.shards``) the move chances are held to the starting player's, and on
  every row so are the practice answers, which the search reads for its guesses. Only the bids
  the lessons speak about move.
- **Held out.** A share of the lessons' Bolas, never trained on, scores the start and every
  pass: how often the player's top bid changed, what the changed bids are worth by the
  teacher's own measure on those lessons (``gain``, in points a decision: the exam lead when
  the new bid is the finalist, the first-stage lead when it is another option tried; and in
  points a game, by how many decisions of each kind a seat has in a Bola), and how often the
  top bid is the teacher's careful pick (the finalist when it led its exam by a standard
  error, else the player's own move). Round 2's held-out card decisions give the points the
  top card, and the best of the top 2 and 3, lose against the 256-deal teacher, which should
  not move.
"""

import argparse
import dataclasses
import json
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from estimation.rl.config import section_from

from .lessons import Lessons, load_all
from .nets import Net, apply_player, load_player, save_player
from .shards import KIND_CARD, NUM_ACTIONS, load_shard, shard_paths

PLAY_BASE = 68
CAREFUL = 1.0  # standard errors an exam lead needs for the teacher's careful pick
PHASES = ("Dash Calls", "the auction", "estimates")
Floats = npt.NDArray[np.float32]
Batch = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LearnConfig:
    """The ``[learn]`` section of a learning config."""

    lessons: str = ""  # the lessons' folder (estimation.onenet.bid_lessons)
    cards: str = ""  # a training set of the player's games: its card play is held on these
    start: str = ""  # the player's file to start from
    epochs: int = 10  # passes over the lessons
    batch: int = 2048  # lessons a step, with as many card decisions
    lr: float = 1e-4
    lr_end: float = 1e-5
    temperature: float = 1.0  # points of lead that multiply an alternative's odds by e
    shrink: bool = True  # pull each lead toward the average lead by its share of noise
    clip: float = 5.0  # the most a lesson may move an alternative's log odds
    card_weight: float = 1.0  # holding the card moves to the start's
    practice_weight: float = 1.0  # holding the practice answers to the start's
    held_out: float = 0.05  # share of the lessons' Bolas kept out of training, to score on
    card_shards: int = 40  # card shards trained on (each about 25,000 decisions)
    card_held_out: float = 0.05  # the card set's held-out Bolas: the share and seed of the
    card_seed: int = 20261108  # training that measured them before (round 2's)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch < 1 or self.card_shards < 1:
            raise ValueError("learn.epochs, batch and card_shards must be at least 1")
        if min(self.lr, self.lr_end, self.temperature, self.clip) <= 0:
            raise ValueError("learn.lr, lr_end, temperature and clip must be above 0")
        if min(self.card_weight, self.practice_weight) < 0:
            raise ValueError("learn.card_weight and practice_weight must be 0 or more")
        if not 0 < self.held_out < 1 or not 0 < self.card_held_out < 1:
            raise ValueError("learn.held_out and card_held_out must be between 0 and 1")


def config_of(data: dict[str, Any]) -> LearnConfig:
    """A learning config from TOML data: one ``[learn]`` section naming its three inputs."""
    unknown = set(data) - {"learn"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}: learning has only [learn]")
    section = data.get("learn")
    if not isinstance(section, dict):
        raise ValueError("a learning config needs a [learn] section")
    values = cast(dict[str, Any], section)
    missing = [name for name in ("lessons", "cards", "start") if not values.get(name)]
    if missing:
        raise ValueError(f"learn needs {', '.join(missing)}: the lessons, card set and player")
    return section_from(LearnConfig(), values, "learn")


def load_config(path: Path) -> LearnConfig:
    return config_of(tomllib.loads(path.read_text()))


def _say(text: str) -> None:
    print(f"[bid-learn] {text}", flush=True)


# The lessons as targets.


def examined(lessons: Lessons) -> npt.NDArray[np.bool_]:
    """The lessons whose finalist sat its whole exam, to the search's last deal."""
    rows = np.arange(len(lessons["pick"]))
    total = lessons["deals"][:, 0].max() if len(rows) else 0
    sat = lessons["deals"][rows, lessons["finalist"]]
    return (sat == total) & (lessons["exam_deals"] > 0) & (lessons["finalist"] > 0)


class Leads(NamedTuple):
    """How the examined lessons' leads spread in each phase (Dash Calls, the auction,
    estimates): their mean, and the variance of the true leads behind them (the measured leads'
    variance less their errors')."""

    mean: tuple[float, float, float]
    variance: tuple[float, float, float]


def spread_of(lessons: Lessons) -> Leads:
    """``Leads`` of the examined lessons; a phase with fewer than two has no spread to show."""
    whole = examined(lessons)
    mean, variance = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for phase in range(3):
        rows = whole & (lessons["phase"] == phase)
        if rows.sum() < 2:
            continue
        exam = lessons["exam"][rows].astype(np.float64)
        noise = float((lessons["exam_error"][rows].astype(np.float64) ** 2).mean())
        mean[phase] = float(exam.mean())
        variance[phase] = max(float(exam.var()) - noise, 0.0)
    return Leads((mean[0], mean[1], mean[2]), (variance[0], variance[1], variance[2]))


def tilts(lessons: Lessons, config: LearnConfig, leads: Leads) -> Floats:
    """``[L]``: each lesson's tilt, the log of the odds its finalist is given against the
    player's own move: 0 where the finalist did not sit its whole exam, else its lead over
    ``temperature`` less the noise's share.

    With ``shrink`` the lead is first taken as its likeliest true value given how leads spread
    in its phase (``leads``, measured on the training lessons): the measured lead pulled toward
    the mean lead by variance of its error / (that + variance of the true leads), with what
    uncertainty that leaves. The odds are exp(tilt), and the mean of exp(noise) is above 1, so
    half the variance over temperature squared comes off: on average the odds are
    exp(true lead / temperature), not more. At most ``clip`` either way."""
    whole = examined(lessons)
    lead = lessons["exam"].astype(np.float64)
    variance = lessons["exam_error"].astype(np.float64) ** 2
    if config.shrink:
        mean = np.array(leads.mean)[lessons["phase"]]
        spread = np.array(leads.variance)[lessons["phase"]]
        kept = spread / np.maximum(spread + variance, 1e-12)
        lead = mean + (lead - mean) * kept
        variance = variance * kept
    t = config.temperature
    tilt = np.clip(lead / t - variance / (2 * t * t), -config.clip, config.clip)
    return np.where(whole, tilt, 0.0).astype(np.float32)


def aimed(
    chances: jax.Array, favourite: jax.Array, finalist: jax.Array, tilt: jax.Array
) -> jax.Array:
    """``[B, 120]``: the log chances a lesson aims the player at, from the starting player's
    log chances ``[B, 120]``, its own move and the finalist (action ids) and the tilt ``[B]``.
    Only the finalist's entry differs: with a tilt above 0 it is the move's log chance plus the
    tilt, so the finalist overtakes the move; with a tilt below 0 it is that or the finalist's
    own, whichever is lower, so a finalist is never lifted for trailing by less than the player
    thought; with a tilt of 0 nothing moves. (Not renormalised: a softmax does that.)"""
    rows = jnp.arange(chances.shape[0])
    own = chances[rows, finalist]
    level = chances[rows, favourite] + tilt
    new = jnp.where(tilt > 0, level, jnp.where(tilt < 0, jnp.minimum(own, level), own))
    return chances.at[rows, finalist].set(new)


def careful_pick(lessons: Lessons) -> npt.NDArray[np.int64]:
    """Each lesson's careful pick, as an action id: the finalist where it sat its whole exam
    and led it by ``CAREFUL`` standard errors, else the player's own move."""
    rows = np.arange(len(lessons["pick"]))
    sure = examined(lessons) & (lessons["exam"] >= CAREFUL * lessons["exam_error"])
    sure &= lessons["exam"] > 0
    slot = np.where(sure, lessons["finalist"], 0)
    return lessons["options"][rows, slot].astype(np.int64)


def lesson_batch(lessons: Lessons, tilt: Floats, rows: npt.NDArray[np.intp]) -> Batch:
    at = np.arange(len(rows))
    options = lessons["options"][rows]
    return {
        "x": lessons["x"][rows],
        "favourite": options[:, 0].astype(np.int32),
        "finalist": options[at, lessons["finalist"][rows]].astype(np.int32),
        "tilt": tilt[rows],
    }


# The card set: the rows that hold the card play, and round 2's held-out measure.


class Cards(NamedTuple):
    """Card decisions of the player's own games: ``held`` are views to hold its card moves on
    (training Bolas), ``scored`` and ``values`` the held-out Bolas' decisions the deeper
    teacher valued, with its value of every legal card."""

    held: npt.NDArray[np.float16]  # [C, VIEW]
    scored: npt.NDArray[np.float16]  # [V, VIEW]
    values: Floats  # [V, 120]


def held_out_bolas(bolas: npt.NDArray[Any], share: float, seed: int) -> npt.NDArray[Any]:
    """The Bolas ``estimation.onenet.copy_hybrid.split`` holds out of a training set with these
    Bolas, by the same draw, so a measure taken there can be taken again."""
    bolas = np.unique(bolas)
    count = min(len(bolas) - 1, max(1, round(share * len(bolas))))
    if count < 1:
        raise ValueError("the card set needs at least 2 Bolas: one to hold out")
    return np.random.default_rng(seed).choice(bolas, size=count, replace=False)


def load_cards(folder: Path, config: LearnConfig) -> Cards:
    """Read the card set a shard at a time (the whole set does not fit next to the lessons):
    the held-out Bolas' valued card decisions from every shard, and the other Bolas' card
    decisions from the first ``card_shards`` shards."""
    paths = shard_paths(folder)
    if not paths:
        raise FileNotFoundError(f"no shards in {folder}")
    bolas: list[npt.NDArray[Any]] = []
    for path in paths:
        with np.load(path) as data:
            bolas.append(data["bola"])  # reads only that array
    out = held_out_bolas(np.concatenate(bolas), config.card_held_out, config.card_seed)
    held: list[npt.NDArray[np.float16]] = []
    scored: list[npt.NDArray[np.float16]] = []
    values: list[Floats] = []
    for index, path in enumerate(paths):
        shard = load_shard(path)
        aside = np.isin(shard["bola"], out)
        valued = aside & (shard["kind"] == KIND_CARD)
        scored.append(shard["x"][valued])
        values.append(shard["values"][valued])
        if index < config.card_shards:
            held.append(shard["x"][~aside & (shard["action"] >= PLAY_BASE)])
    return Cards(np.concatenate(held), np.concatenate(scored), np.concatenate(values))


# The losses.


def _legal(x: jax.Array) -> jax.Array:
    return x[:, -NUM_ACTIONS:] > 0.5


def _cross_entropy(target_logits: jax.Array, logits: jax.Array, legal: jax.Array) -> jax.Array:
    """Mean cross-entropy from the chances of ``target_logits`` to those of ``logits``, both
    over the legal moves."""
    target = jax.nn.softmax(jnp.where(legal, target_logits, -1e9))
    logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
    return -(target * jnp.where(legal, logp, 0.0)).sum(-1).mean()


def loss(
    player: Net, start: Net, taught: Batch, cards: jax.Array, config: LearnConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The learning loss of a batch of lessons and a batch of card views, with its parts."""
    x = taught["x"].astype(jnp.float32)
    legal = _legal(x)
    logits, answers = apply_player(player, x)
    base, base_answers = apply_player(start, x)
    chances = jax.nn.log_softmax(jnp.where(legal, base, -1e9))
    aim = aimed(chances, taught["favourite"], taught["finalist"], taught["tilt"])
    bids = _cross_entropy(aim, logits, legal)
    cx = cards.astype(jnp.float32)
    card_logits, card_answers = apply_player(player, cx)
    card_base, card_base_answers = apply_player(start, cx)
    card = _cross_entropy(card_base, card_logits, _legal(cx))
    drift = jnp.mean((answers - base_answers) ** 2) + jnp.mean(
        (card_answers - card_base_answers) ** 2
    )
    total = bids + config.card_weight * card + config.practice_weight * drift
    return total, {"bids": bids, "cards": card, "practice": drift}


# The held-out scores.


@jax.jit
def _order(player: Net, x: jax.Array) -> jax.Array:
    logits, _ = apply_player(player, x.astype(jnp.float32))
    return jnp.argsort(-jnp.where(_legal(x), logits, -jnp.inf), axis=-1)[:, :3]


def _top3(player: Net, x: npt.NDArray[np.float16], batch: int = 8192) -> npt.NDArray[np.int64]:
    """``[N, 3]``: the player's three favourite legal moves at each view, the best first (past
    the legal ones the order is arbitrary)."""
    parts = [np.asarray(_order(player, x[k : k + batch])) for k in range(0, len(x), batch)]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros((0, 3), np.int64)


def top_bid(player: Net, x: npt.NDArray[np.float16]) -> npt.NDArray[np.int64]:
    """``[N]``: the player's favourite legal move at each view, the move it makes."""
    return _top3(player, x)[:, 0]


def bid_scores(
    player: Net, lessons: Lessons, seat_game: Sequence[float] | None = None
) -> dict[str, Any]:
    """How the player's top bids stand on (held-out) lessons: how often the top bid is no longer
    the starting player's move (``changed``); what its changes are worth, in points a decision,
    by the teacher's measures (``gain``: the exam lead where the new bid is the examined
    finalist, the first-stage lead where it is another option the teacher tried, 0 where the
    bid did not change), over the lessons where the new bid was tried at all; how often it was
    not (``untried``); how often the top bid is the teacher's careful pick (``agree``), and how
    often where that pick is not the player's own move (``followed``, of ``overruled``).
    ``phases`` holds ``changed`` and ``gain`` for each kind of decision. With ``seat_game``,
    each kind's decisions a seat has in a Bola (``per_seat_game``), ``game`` is the changes'
    worth in points a game: each kind's gain times its decisions."""
    count = len(lessons["pick"])
    rows = np.arange(count)
    top = top_bid(player, lessons["x"])
    slots = lessons["options"] == top[:, None]
    tried = slots.any(1)
    slot = slots.argmax(1)
    changed = top != lessons["options"][:, 0]
    is_finalist = tried & (slot == lessons["finalist"]) & examined(lessons)
    worth = np.where(is_finalist, lessons["exam"], lessons["opening"][rows, slot])
    worth = np.where(changed, worth, 0.0)
    pick = careful_pick(lessons)
    overruled = pick != lessons["options"][:, 0]

    def gain(of: npt.NDArray[np.bool_]) -> tuple[float, float]:
        w = worth[of & tried]
        mean = float(w.mean()) if len(w) else 0.0
        return mean, float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0.0

    everything = np.ones(count, bool)
    scores: dict[str, Any] = {
        "lessons": float(count),
        "changed": float(changed.mean()) if count else 0.0,
        "gain": gain(everything)[0],
        "gain_error": gain(everything)[1],
        "untried": float((~tried).mean()) if count else 0.0,
        "agree": float((top == pick).mean()) if count else 0.0,
        "overruled": float(overruled.mean()) if count else 0.0,
        "followed": float((top == pick)[overruled].mean()) if overruled.any() else 0.0,
        "phases": {},
    }
    game, game_variance = 0.0, 0.0
    for phase, name in enumerate(PHASES):
        of = lessons["phase"] == phase
        mean, error = gain(of)
        scores["phases"][name] = {
            "lessons": float(of.sum()),
            "changed": float(changed[of].mean()) if of.any() else 0.0,
            "gain": mean,
            "gain_error": error,
        }
        if seat_game is not None:
            game += mean * seat_game[phase]
            game_variance += (error * seat_game[phase]) ** 2
    if seat_game is not None:
        scores["game"] = game
        scores["game_error"] = float(np.sqrt(game_variance))
    return scores


def per_seat_game(lessons: Lessons, folder: Path) -> list[float] | None:
    """Each kind's bidding decisions with a choice that a seat has in a Bola, from the lessons
    and the share of each kind that was taught (the ``label_share`` in the lessons' own
    ``config.json``); None when that file is not there or a kind was not taught."""
    path = folder / "config.json"
    if not path.exists():
        return None
    shares = json.loads(path.read_text()).get("label_share")
    bolas = len(np.unique(lessons["game"]))
    if not isinstance(shares, list) or len(shares) != 3 or not all(shares) or not bolas:
        return None
    return [
        float((lessons["phase"] == phase).sum()) / shares[phase] / bolas / 4 for phase in range(3)
    ]


def card_scores(player: Net, cards: Cards) -> dict[str, float]:
    """Round 2's measure on its held-out card decisions: the points the player's top card, and
    the best of its top 2 and top 3, lose against the teacher's best card."""
    if not len(cards.scored):
        return {"cards": 0.0}
    order = _top3(player, cards.scored)
    legal = cards.scored[:, -NUM_ACTIONS:] > 0.5
    best = np.where(legal, cards.values, -np.inf).max(1)
    picked = np.take_along_axis(cards.values, order, axis=1)
    picked = np.where(np.take_along_axis(legal, order, axis=1), picked, -np.inf)
    lost = best[:, None] - np.maximum.accumulate(picked, axis=1)
    return {
        "cards": float(len(best)),
        "top_card": float(lost[:, 0].mean()),
        "top_2": float(lost[:, 1].mean()),
        "top_3": float(lost[:, 2].mean()),
    }


# The run.


def split(lessons: Lessons, share: float, seed: int) -> tuple[Lessons, Lessons]:
    """Training and held-out lessons, holding out whole Bolas."""
    bolas = np.unique(lessons["game"])
    count = min(len(bolas) - 1, max(1, round(share * len(bolas))))
    if count < 1:
        raise ValueError("learning needs lessons from at least 2 Bolas: one to hold out")
    aside = np.isin(lessons["game"], np.random.default_rng(seed).choice(bolas, count, False))
    return (
        {name: values[~aside] for name, values in lessons.items()},
        {name: values[aside] for name, values in lessons.items()},
    )


def score_line(bids: dict[str, Any], cards: dict[str, float]) -> str:
    line = (
        f"top bid changed at {bids['changed']:.1%}, worth {bids['gain']:+.3f} points a decision "
        f"(give or take {bids['gain_error']:.3f})"
    )
    if "game" in bids:
        line += f", {bids['game']:+.1f} a game (give or take {bids['game_error']:.1f})"
    line += (
        f"; the teacher's careful pick at {bids['agree']:.1%} "
        f"(where it overrules the player: {bids['followed']:.1%})"
    )
    if cards.get("cards"):
        line += (
            f" | top card loses {cards['top_card']:.3f}, best of top 2 {cards['top_2']:.3f}, "
            f"top 3 {cards['top_3']:.3f}"
        )
    return line


def run(config: LearnConfig, out: Path) -> dict[str, Any]:
    """Learn the lessons from the start; save the last pass's player in ``out`` with the scores
    of the start and of every pass. Returns the summary."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    _say(f"device {jax.devices()[0]}; lessons {config.lessons}; cards {config.cards}")
    lessons = load_all(Path(config.lessons))
    seat_game = per_seat_game(lessons, Path(config.lessons))
    train, held = split(lessons, config.held_out, config.seed)
    cards = load_cards(Path(config.cards), config)
    leads = spread_of(train)
    tilt = tilts(train, config, leads)
    whole = examined(train)
    spreads = ", ".join(
        f"{name} {mean:+.2f} give or take {np.sqrt(variance):.2f}"
        for name, mean, variance in zip(PHASES, leads.mean, leads.variance, strict=True)
    )
    _say(
        f"{len(tilt):,} lessons to learn ({whole.mean():.0%} with a whole exam; true leads: "
        f"{spreads}), {len(held['pick']):,} held out; {len(cards.held):,} card decisions held, "
        f"{len(cards.scored):,} scored"
    )
    start, meta = load_player(Path(config.start))
    own = float((_top3(start, train["x"])[:, 0] == train["options"][:, 0]).mean())
    if own < 0.99:
        raise ValueError(
            f"{config.start} makes the lessons' own move at only {own:.0%} of them: "
            "the lessons were taught for another player"
        )
    player = start
    steps = max(1, len(tilt) // config.batch)
    schedule = optax.cosine_decay_schedule(
        config.lr, steps * config.epochs, alpha=config.lr_end / config.lr
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt = tx.init(player)

    @jax.jit
    def step(player: Net, opt: Any, taught: Batch, held_cards: jax.Array):
        (_, parts), grads = jax.value_and_grad(loss, has_aux=True)(
            player, start, taught, held_cards, config
        )
        updates, opt = tx.update(grads, opt, player)
        return cast(Net, optax.apply_updates(player, updates)), opt, parts

    rng = np.random.default_rng(config.seed)
    before = {"bids": bid_scores(player, held, seat_game), "cards": card_scores(player, cards)}
    _say(f"the start: {score_line(before['bids'], before['cards'])}")
    rows: list[dict[str, Any]] = [{"epoch": 0, **before}]
    log = (out / "log.jsonl").open("w")
    log.write(json.dumps(rows[0]) + "\n")
    size = min(config.batch, len(tilt))
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        order = rng.permutation(len(tilt))
        parts: list[dict[str, jax.Array]] = []
        for k in range(steps):
            taught = lesson_batch(train, tilt, np.sort(order[k * size : (k + 1) * size]))
            picked = np.sort(rng.choice(len(cards.held), size=size, replace=len(cards.held) < size))
            player, opt, info = step(player, opt, taught, cards.held[picked])
            parts.append(info)
        fetched = jax.device_get(parts)
        row = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "train": {name: float(np.mean([p[name] for p in fetched])) for name in fetched[0]},
            "bids": bid_scores(player, held, seat_game),
            "cards": card_scores(player, cards),
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        _say(f"pass {epoch}/{config.epochs}: {score_line(row['bids'], row['cards'])}")
    log.close()
    save_player(
        out / "player.npz",
        player,
        {**meta, "bid_lessons": dataclasses.asdict(config), "epoch": config.epochs},
    )
    summary = {
        "lessons": {"train": len(tilt), "held_out": len(held["pick"])},
        "per_seat_game": seat_game,
        "leads": {"mean": list(leads.mean), "variance": list(leads.variance)},
        "start": before,
        "learned": {"bids": rows[-1]["bids"], "cards": rows[-1]["cards"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.bid_learn",
        description="Learn the player's bids from the bid teacher's lessons.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"--config {args.config}: {error}")
    run(config, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
