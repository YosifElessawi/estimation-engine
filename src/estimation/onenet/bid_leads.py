"""Step 6's lead models (docs/rl/hybrid-iteration-plan.md): what the bid teacher's lessons say
about a decision, read from the view alone.

A lesson is the teacher's verdict on one decision, the luck of its deals included, and no player
meets that decision again. What a player can learn is what the lessons say about decisions that
look alike. A lead model reads exactly that: from a view, the lead over the player's own move of
every option the teacher tried there, in round points.

- **Fit** (``fit``). A fresh net, not the player's (a fresh one reads the lessons better than
  the player's own body does), learns each tried option's measured lead by weighted squared
  error, so on views that look alike it gives their mean lead: an option that is 5 points better
  at one decision in three and 2 worse at the rest comes out ahead, as it should. A lead counts
  by its precision (``targets``): the finalist's exam lead on hundreds of deals counts about ten
  times a dropped option's lead on its first stage.
- **Suit shuffles** (``estimation.onenet.suits``). An estimate is as good with the suits other
  than the trump renamed, so every pass shows each estimate's view under another shuffle: up to
  six views (24 with no trump) for each estimate taught.
- **Folds** (``cross_fit``). A model knows the lessons it was fit on better than it knows the
  game: it has seen their luck. So the training Bolas are cut in folds, each fold's leads are
  read by a model fit on the other folds, and the held-out Bolas' by all of them together. The
  player then learns only leads read by a model that never saw that Bola.

The models are training tools, like step 5's judge: kept out of the player's file and never at
the table.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from estimation.jaxenv.selfplay import _body, _head, _mlp, _out

from .lessons import VIEW, Lessons
from .nets import Net, parse_size
from .shards import NUM_ACTIONS
from .suits import SHUFFLES, shuffle_rows, shuffle_tables

UNIT = 4.0  # round points per unit of a lead model's output, which keeps its numbers near 1
Floats = npt.NDArray[np.float32]
Say = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Reading:
    """How a lead model is fit."""

    size: str = "512x3"  # units x layers
    epochs: int = 40
    batch: int = 2048
    lr: float = 1e-3
    floor: float = 0.5  # points: the least error a measured lead is credited with
    shuffle: bool = True  # show every estimate under a new suit shuffle each pass

    def __post_init__(self) -> None:
        parse_size(self.size)
        if self.epochs < 1 or self.batch < 1:
            raise ValueError("a lead model's epochs and batch must be at least 1")
        if min(self.lr, self.floor) <= 0:
            raise ValueError("a lead model's lr and floor must be above 0")


# The lessons as leads to read.


def tried(lessons: Lessons) -> npt.NDArray[np.bool_]:
    """``[L, K]``: the slots that hold an alternative the teacher tried (not slot 0, the
    player's own move, and not an unused slot)."""
    used = lessons["options"] >= 0
    used[:, 0] = False
    return used


class Targets(NamedTuple):
    """What a lead model learns from lessons: each slot's action id (0 in the slots that do not
    count), its measured lead over the player's own move, and how much that lead counts."""

    options: npt.NDArray[np.int32]  # [L, K]
    lead: Floats  # [L, K]
    weight: Floats  # [L, K]: 0 for the own move and the unused slots


def targets(lessons: Lessons, floor: float) -> Targets:
    """The leads to learn. The finalist's is its exam lead, the fair measure of it; any other
    option's is its lead on the deals it played before it was dropped (a little low, since it
    was dropped on them: these are the options that lost). Each counts by its precision: 1 over
    its squared standard error, which is never taken below ``floor``."""
    count = len(lessons["pick"])
    rows = np.arange(count)
    used = tried(lessons)
    lead = lessons["margin"].astype(np.float32)
    error = lessons["error"].astype(np.float32)
    finalist = lessons["finalist"].astype(np.int64)
    sat = (lessons["exam_deals"] > 0) & (finalist > 0)
    lead[rows[sat], finalist[sat]] = lessons["exam"][sat]
    error[rows[sat], finalist[sat]] = lessons["exam_error"][sat]
    weight = np.where(used, 1.0 / (error**2 + floor**2), 0.0).astype(np.float32)
    options = np.where(lessons["options"] >= 0, lessons["options"], 0).astype(np.int32)
    return Targets(options, np.where(used, lead, 0.0).astype(np.float32), weight)


# The model.


def init(key: jax.Array, size: str) -> Net:
    """A fresh lead model: a body over the view and a lead for each of the 120 moves."""
    hidden, depth = parse_size(size)
    k_body, k_lead = jax.random.split(key)
    return {
        "body": _mlp(k_body, VIEW, hidden, depth),
        "lead": _head(k_lead, hidden, NUM_ACTIONS, 0.01),
    }


def apply(net: Net, x: jax.Array) -> jax.Array:
    """``[B, 120]``: each move's lead over the player's own move at views ``x``, in ``UNIT``
    points. Only the moves the teacher would try there mean anything."""
    return _out(net["lead"], _body(net["body"], x))


def fit(lessons: Lessons, by: Reading, seed: int, say: Say | None = None) -> Net:
    """A lead model fit to ``lessons`` from scratch."""
    aims = targets(lessons, by.floor)
    count = len(aims.lead)
    if not count:
        raise ValueError("a lead model needs lessons to read")
    x = jnp.asarray(lessons["x"])
    options, lead, weight = (
        jnp.asarray(aims.options),
        jnp.asarray(aims.lead / UNIT),
        jnp.asarray(aims.weight),
    )
    tables = shuffle_tables()
    views, moves = jnp.asarray(tables[0]), jnp.asarray(tables[1])
    rows = jnp.asarray(shuffle_rows(lessons["x"]).astype(np.int32))
    size = min(by.batch, count)
    steps = max(1, count // size)
    schedule = optax.cosine_decay_schedule(by.lr, steps * by.epochs, alpha=0.01)
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    net = init(jax.random.PRNGKey(seed), by.size)
    state = tx.init(net)
    shuffle = by.shuffle

    def loss(net: Net, x: jax.Array, o: jax.Array, t: jax.Array, w: jax.Array) -> jax.Array:
        read = jnp.take_along_axis(apply(net, x), o, axis=1)
        return (w * (read - t) ** 2).sum() / jnp.maximum(w.sum(), 1e-9)

    @jax.jit
    def step(
        net: Net,
        state: Any,
        at: jax.Array,
        key: jax.Array,
        x: jax.Array,
        options: jax.Array,
        lead: jax.Array,
        weight: jax.Array,
        rows: jax.Array,
    ):
        seen, o = x[at].astype(jnp.float32), options[at]
        if shuffle:
            which = jax.random.randint(key, (at.shape[0],), 0, SHUFFLES)
            seen = jnp.take_along_axis(seen, views[rows[at], which], axis=1)
            o = jnp.take_along_axis(moves[rows[at], which], o, axis=1)
        value, grads = jax.value_and_grad(loss)(net, seen, o, lead[at], weight[at])
        updates, state = tx.update(grads, state, net)
        return cast(Net, optax.apply_updates(net, updates)), state, value

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    for epoch in range(1, by.epochs + 1):
        order = rng.permutation(count)
        values: list[jax.Array] = []
        for k in range(steps):
            key, sub = jax.random.split(key)
            at = jnp.asarray(order[k * size : (k + 1) * size])
            net, state, value = step(net, state, at, sub, x, options, lead, weight, rows)
            values.append(value)
        if say is not None and (epoch % 10 == 0 or epoch == by.epochs):
            error = float(np.mean(jax.device_get(values))) * UNIT**2
            say(f"pass {epoch}/{by.epochs}: weighted squared error {error:.2f}")
    return net


@jax.jit
def _read(net: Net, x: jax.Array, options: jax.Array) -> jax.Array:
    return jnp.take_along_axis(apply(net, x.astype(jnp.float32)), options, axis=1) * UNIT


def predict(net: Net, lessons: Lessons, batch: int = 8192) -> Floats:
    """``[L, K]``: the lead the model reads for each tried option of ``lessons``, in round
    points; 0 in the slots of the own move and the unused ones."""
    used = tried(lessons)
    options = np.where(lessons["options"] >= 0, lessons["options"], 0).astype(np.int32)
    parts = [
        np.asarray(_read(net, lessons["x"][k : k + batch], options[k : k + batch]))
        for k in range(0, len(options), batch)
    ]
    read = np.concatenate(parts) if parts else np.zeros(options.shape, np.float32)
    return np.where(used, read, 0.0).astype(np.float32)


# Folds.


def fold_of(games: npt.NDArray[Any], folds: int, seed: int) -> npt.NDArray[np.int64]:
    """``[L]``: the fold of each lesson's Bola: the Bolas dealt round the folds in an order drawn
    from ``seed``, a whole Bola in one fold."""
    bolas = np.unique(games)
    if folds < 2 or len(bolas) < folds:
        raise ValueError(f"{folds} folds need at least 2 folds and {folds} Bolas of lessons")
    places = np.random.default_rng(seed).permutation(len(bolas)) % folds
    return places[np.searchsorted(bolas, games)].astype(np.int64)


def cross_fit(
    train: Lessons, held: Lessons, by: Reading, folds: int, seed: int, say: Say | None = None
) -> tuple[Floats, Floats]:
    """The leads read for every training lesson by a model fit on the other folds, never on its
    own Bola, and for the held-out lessons by all the models together (their mean)."""
    fold = fold_of(train["game"], folds, seed)
    read = np.zeros(train["options"].shape, np.float32)
    held_read = np.zeros(held["options"].shape, np.float32)
    for k in range(folds):
        inside = fold != k
        if say is not None:
            say(f"lead model {k + 1}/{folds}: {int(inside.sum()):,} lessons")
        net = fit({name: values[inside] for name, values in train.items()}, by, seed + k, say)
        read[~inside] = predict(net, {name: values[~inside] for name, values in train.items()})
        held_read += predict(net, held) / folds
    return read, held_read


def picks(leads: Floats, lessons: Lessons, margin: float) -> npt.NDArray[np.int64]:
    """``[L]``: the bid the leads pick at each lesson, as an action id: the tried option read
    furthest ahead where that is more than ``margin`` points, else the player's own move."""
    rows = np.arange(len(leads))
    read = np.where(tried(lessons), leads, -np.inf)
    best = read.argmax(1)
    slot = np.where(read[rows, best] > margin, best, 0)
    return lessons["options"][rows, slot].astype(np.int64)
