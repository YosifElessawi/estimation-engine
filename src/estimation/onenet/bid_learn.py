"""Step 6's learning (docs/rl/hybrid-iteration-plan.md): the player learns its bids from the bid
teacher's lessons and keeps everything else as it is.

    PYTHONPATH=src ~/jax-env/bin/python -m estimation.onenet.bid_learn \\
        --config configs/onenet-bids-1-learn.toml --out runs/onenet/bids-1/learn

- **Leads read from the view** (``estimation.onenet.bid_leads``). A lesson is one decision's
  verdict, luck included. Lead models read what the lessons say about decisions that look alike:
  at each lesson, the lead of every tried option over the player's own move, read by a model
  that never saw that lesson's Bola. The player learns these read leads, not the measured ones:
  it is taught only what holds beyond the one decision.
- **Bids** (``aimed``). The target is the starting player's move chances with every tried
  option set against its own move by its tilt (``tilts``): the read lead less ``margin``, over
  ``temperature``. An option read ahead by more than the margin overtakes the move, however sure
  the player was of it: self-play left the player sure of moves it had stopped trying the
  alternatives to. An option read behind is lowered to that level if it stood above it and
  otherwise left alone. The margin is there because the best of several read leads reads high.
- **Which lessons** (``kinds``). Only the lessons of the kinds named are read and learned;
  at the other kinds' lessons the player's bids are held to the starting player's. The
  real-play check (``estimation.onenet.bid_check``) found the teacher's leads real at
  estimates and not at the auction, so round 1 learns its estimates.
- **Everything else stays.** On card decisions (a training set of the player's own games,
  ``estimation.onenet.shards``) the move chances are held to the starting player's, and on
  every row so are the practice answers, which the search reads for its guesses. Only the bids
  the lessons speak about move.
- **Held out.** A share of the lessons' Bolas, never trained on, scores the start and every
  pass (``score_bids``): how often the player's top bid changed and what the changed bids are
  worth by the teacher's own measures on those lessons, in points a decision and a game.
  ``gain`` counts a new bid by its first-stage lead: every option played the first stage
  whatever became of it, so it is fair to any bid, if short (32 deals). ``exam_gain`` counts
  the bid by its long exam where it was the examined finalist and by its first stage elsewhere:
  steadier, but it reads low, since a bid gets its exam only when its first stage went well
  and is left with its first stage when that went badly. The lead models' own bids are scored
  the same way: what the lessons hold for any player that can read them. Round 2's held-out card
  decisions give the points the top card, and the best of the top 2 and 3, lose against the
  256-deal teacher, which should not move.
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

from .bid_leads import Reading, cross_fit, picks, tried
from .lessons import Lessons, careful_pick, examined, load_all
from .nets import Net, apply_player, load_player, parse_size, save_player
from .shards import KIND_CARD, NUM_ACTIONS, load_shard, shard_paths

PLAY_BASE = 68
PHASES = ("Dash Calls", "the auction", "estimates")
KINDS = ("dash", "auction", "estimates")  # the same kinds, as a config names them
Floats = npt.NDArray[np.float32]
Batch = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LearnConfig:
    """The ``[learn]`` section of a learning config."""

    lessons: str = ""  # the lessons' folder (estimation.onenet.bid_lessons)
    cards: str = ""  # a training set of the player's games: its card play is held on these
    start: str = ""  # the player's file to start from
    kinds: tuple[str, ...] = KINDS  # the kinds of decision whose lessons are learned
    folds: int = 5  # lead models: each reads the fold of training Bolas it was not fit on
    lead_size: str = "512x3"  # a lead model's units x layers
    lead_epochs: int = 40  # a lead model's passes over its lessons
    lead_lr: float = 1e-3
    floor: float = 0.5  # points: the least error a measured lead is credited with
    shuffle: bool = True  # the lead models see every estimate under new suit shuffles
    margin: float = 0.5  # points a read lead must clear before the option overtakes the move
    temperature: float = 0.5  # points of read lead, past the margin, that multiply the odds by e
    clip: float = 5.0  # the most a lesson may move an option's log odds
    epochs: int = 40  # the player's passes over the lessons
    batch: int = 2048  # lessons a step, with as many card decisions
    lr: float = 3e-4
    lr_end: float = 1e-5
    card_weight: float = 1.0  # holding the card moves to the start's
    practice_weight: float = 1.0  # holding the practice answers to the start's
    held_out: float = 0.05  # share of the lessons' Bolas kept out of training, to score on
    card_shards: int = 40  # card shards trained on (each about 25,000 decisions)
    card_held_out: float = 0.05  # the card set's held-out Bolas: the share and seed of the
    card_seed: int = 20261108  # training that measured them before (round 2's)
    seed: int = 0

    def __post_init__(self) -> None:
        parse_size(self.lead_size)
        if (
            not self.kinds
            or len(set(self.kinds)) != len(self.kinds)
            or set(self.kinds) - set(KINDS)
        ):
            raise ValueError(f"learn.kinds must name one or more of {', '.join(KINDS)}, each once")
        if min(self.epochs, self.batch, self.card_shards, self.lead_epochs) < 1:
            raise ValueError("learn.epochs, lead_epochs, batch and card_shards must be at least 1")
        if self.folds < 2:
            raise ValueError("learn.folds must be at least 2: a model never reads its own Bolas")
        if min(self.lr, self.lr_end, self.lead_lr, self.temperature, self.clip, self.floor) <= 0:
            raise ValueError(
                "learn.lr, lr_end, lead_lr, temperature, clip and floor must be above 0"
            )
        if min(self.card_weight, self.practice_weight, self.margin) < 0:
            raise ValueError("learn.card_weight, practice_weight and margin must be 0 or more")
        if not 0 < self.held_out < 1 or not 0 < self.card_held_out < 1:
            raise ValueError("learn.held_out and card_held_out must be between 0 and 1")

    @property
    def phases(self) -> tuple[int, ...]:
        """The lessons' phase numbers of the kinds that are learned."""
        return tuple(KINDS.index(kind) for kind in self.kinds)

    @property
    def reading(self) -> Reading:
        """How the lead models are fit."""
        return Reading(
            self.lead_size, self.lead_epochs, self.batch, self.lead_lr, self.floor, self.shuffle
        )


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


def tilts(leads: Floats, config: LearnConfig) -> Floats:
    """``[L, K]``: the log of the odds each option is given against the player's own move, from
    the leads read for it (``bid_leads``): the lead less ``margin``, over ``temperature``, at
    most ``clip`` either way."""
    tilt = (leads.astype(np.float64) - config.margin) / config.temperature
    return np.clip(tilt, -config.clip, config.clip).astype(np.float32)


def aimed(chances: jax.Array, options: jax.Array, used: jax.Array, tilt: jax.Array) -> jax.Array:
    """``[B, 120]``: the log chances a lesson aims the player at, from the starting player's log
    chances ``[B, 120]``, the lesson's ``options`` ``[B, K]`` (action ids, the player's own move
    first), which of them count (``used``: the tried alternatives) and their tilts ``[B, K]``.
    Only those options' entries differ. With a tilt above 0 an option's is the move's log chance
    plus the tilt, so it overtakes the move; with a tilt below 0 it is that or the option's own,
    whichever is lower, so an option is never lifted for trailing by less than the player
    thought. (Not renormalised: a softmax does that.)"""
    own = jnp.take_along_axis(chances, options, axis=1)
    level = own[:, :1] + tilt
    new = jnp.where(tilt > 0, level, jnp.minimum(own, level))
    change = jnp.where(used, new - own, 0.0)
    return chances + (jax.nn.one_hot(options, NUM_ACTIONS) * change[..., None]).sum(1)


def lesson_batch(lessons: Lessons, tilt: Floats, rows: npt.NDArray[np.intp]) -> Batch:
    options = lessons["options"][rows]
    return {
        "x": lessons["x"][rows],
        "options": np.where(options >= 0, options, 0).astype(np.int32),
        "used": tried({"options": options}),
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
    aim = aimed(chances, taught["options"], taught["used"], taught["tilt"])
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


def score_bids(
    top: npt.NDArray[np.int64], lessons: Lessons, seat_game: Sequence[float] | None = None
) -> dict[str, Any]:
    """How bids ``top`` (an action id for each lesson) stand on (held-out) lessons: how often
    the bid is not the starting player's move (``changed``); what the changes are worth, in
    points a decision, by the teacher's measures, over the lessons where the bid was tried at
    all (``gain``: the new bid's first-stage lead, 0 where the bid did not change; ``exam_gain``:
    its exam lead where it is the examined finalist, its first-stage lead elsewhere); how often
    it was not tried (``untried``); how often the bid is the teacher's careful pick (``agree``),
    and how often where that pick is not the player's own move (``followed``, of ``overruled``).
    ``phases`` holds ``changed`` and the gains for each kind of decision. With ``seat_game``,
    each kind's decisions a seat has in a Bola (``per_seat_game``), ``game`` and ``exam_game``
    are the changes' worth in points a game: each kind's gain times its decisions."""
    count = len(lessons["pick"])
    rows = np.arange(count)
    slots = lessons["options"] == top[:, None]
    was_tried = slots.any(1)
    slot = slots.argmax(1)
    changed = top != lessons["options"][:, 0]
    first_stage = lessons["opening"][rows, slot]
    is_finalist = was_tried & (slot == lessons["finalist"]) & examined(lessons)
    fair = np.where(changed, first_stage, 0.0)
    steady = np.where(changed, np.where(is_finalist, lessons["exam"], first_stage), 0.0)
    pick = careful_pick(lessons)
    overruled = pick != lessons["options"][:, 0]

    def gain(worth: npt.NDArray[Any], of: npt.NDArray[np.bool_]) -> tuple[float, float]:
        w = worth[of & was_tried]
        mean = float(w.mean()) if len(w) else 0.0
        return mean, float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0.0

    everything = np.ones(count, bool)
    scores: dict[str, Any] = {
        "lessons": float(count),
        "changed": float(changed.mean()) if count else 0.0,
        "gain": gain(fair, everything)[0],
        "gain_error": gain(fair, everything)[1],
        "exam_gain": gain(steady, everything)[0],
        "untried": float((~was_tried).mean()) if count else 0.0,
        "agree": float((top == pick).mean()) if count else 0.0,
        "overruled": float(overruled.mean()) if count else 0.0,
        "followed": float((top == pick)[overruled].mean()) if overruled.any() else 0.0,
        "phases": {},
    }
    game, game_variance, exam_game = 0.0, 0.0, 0.0
    for phase, name in enumerate(PHASES):
        of = lessons["phase"] == phase
        mean, error = gain(fair, of)
        exam_mean = gain(steady, of)[0]
        scores["phases"][name] = {
            "lessons": float(of.sum()),
            "changed": float(changed[of].mean()) if of.any() else 0.0,
            "gain": mean,
            "gain_error": error,
            "exam_gain": exam_mean,
        }
        if seat_game is not None:
            game += mean * seat_game[phase]
            game_variance += (error * seat_game[phase]) ** 2
            exam_game += exam_mean * seat_game[phase]
    if seat_game is not None:
        scores["game"] = game
        scores["game_error"] = float(np.sqrt(game_variance))
        scores["exam_game"] = exam_game
    return scores


def bid_scores(
    player: Net, lessons: Lessons, seat_game: Sequence[float] | None = None
) -> dict[str, Any]:
    """``score_bids`` of the player's top bids."""
    return score_bids(top_bid(player, lessons["x"]), lessons, seat_game)


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


def bid_line(bids: dict[str, Any]) -> str:
    line = (
        f"top bid changed at {bids['changed']:.1%}, worth {bids['gain']:+.3f} points a decision "
        f"(give or take {bids['gain_error']:.3f})"
    )
    if "game" in bids:
        line += (
            f", {bids['game']:+.1f} a game (give or take {bids['game_error']:.1f}; by the "
            f"exams {bids['exam_game']:+.1f})"
        )
    return line + (
        f"; the teacher's careful pick at {bids['agree']:.1%} "
        f"(where it overrules the player: {bids['followed']:.1%})"
    )


def score_line(bids: dict[str, Any], cards: dict[str, float]) -> str:
    line = bid_line(bids)
    if cards.get("cards"):
        line += (
            f" | top card loses {cards['top_card']:.3f}, best of top 2 {cards['top_2']:.3f}, "
            f"top 3 {cards['top_3']:.3f}"
        )
    return line


def run(config: LearnConfig, out: Path) -> dict[str, Any]:
    """Read the lessons' leads, teach them to the player from the start, and save the last
    pass's player in ``out`` with the scores of the start, of every pass and of the lead
    models' own bids, and those bids on the held-out lessons (``picks.npz``, as
    ``estimation.onenet.bid_check`` takes them). Returns the summary."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    _say(f"device {jax.devices()[0]}; lessons {config.lessons}; cards {config.cards}")
    lessons = load_all(Path(config.lessons))
    seat_game = per_seat_game(lessons, Path(config.lessons))
    train, held = split(lessons, config.held_out, config.seed)
    del lessons
    start, meta = load_player(Path(config.start))
    own = float((top_bid(start, train["x"]) == train["options"][:, 0]).mean())
    if own < 0.99:
        raise ValueError(
            f"{config.start} makes the lessons' own move at only {own:.0%} of them: "
            "the lessons were taught for another player"
        )
    learned = np.isin(train["phase"], config.phases)
    _say(
        f"{int(learned.sum()):,} lessons to learn ({', '.join(config.kinds)}) of "
        f"{len(learned):,}, {len(held['pick']):,} held out"
    )
    taught = {name: values[learned] for name, values in train.items()}
    read, held_leads = cross_fit(taught, held, config.reading, config.folds, config.seed, _say)
    del taught
    leads = np.zeros(train["options"].shape, np.float32)
    leads[learned] = read
    held_leads[~np.isin(held["phase"], config.phases)] = 0.0  # nothing is read of other kinds
    lead_picks = picks(held_leads, held, config.margin)
    readers = score_bids(lead_picks, held, seat_game)
    _say(f"the lead models' own bids: {bid_line(readers)}")
    np.savez_compressed(
        out / "picks.npz",
        allow_pickle=False,
        game=held["game"],
        step=held["step"],
        action=lead_picks.astype(np.int16),
    )
    cards = load_cards(Path(config.cards), config)
    tilt = np.where(learned[:, None], tilts(leads, config), 0.0).astype(np.float32)
    _say(
        f"{float((tilt.max(1) > 0).mean()):.1%} of the lessons put another bid first; "
        f"{len(cards.held):,} card decisions held, {len(cards.scored):,} scored"
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
        if epoch % 5 == 0 or epoch == config.epochs:
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
        "lead_models": readers,
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
