"""Step 5's recorder with the search on the GPU (docs/rl/hybrid-iteration-plan.md): the records
``estimation.rl.onenet_record`` makes for the one net inside its own search (``onenet`` with
``own=true`` and ``judge=round``), many Bolas at once.

    python -m estimation.onenet.gpu_record run --config configs/onenet-expert-1.toml \\
        --out runs/onenet/expert-1/gpu-record
    python -m estimation.onenet.gpu_record time --config configs/onenet-expert-1.toml --games 64

The engine plays every Bola on the CPU, from the same seeds as the CPU recorder, so the records
replay exactly and the training set is built from them as before (``estimation.onenet.data``).
JAX's copy of each round is stepped alongside for the views and the search
(``estimation.onenet.gpu_search``). At each step every live Bola makes one move: forced moves as
they are, bids by the player's top legal move, and card decisions with a choice by a search: the
teacher (every card, ``teacher_samples`` deals) at ``label_share`` of them, which keeps its
value of every legal card, and the player's own search (its ``top_k`` favourites, ``samples``
deals) at the rest. The record config's ``workers`` is not used: one process drives the GPU.

Trainer code: the record holds whole games, but each search reads only what its seat may see
(its hand, the public table, and the bidding this round), never another seat's cards.
"""

import argparse
import dataclasses
import json
import random
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from estimation.bots import make_bot
from estimation.bots.hybrid.bot import HybridConfig, _merge_equivalent, equivalent_runs  # pyright: ignore[reportPrivateUsage]
from estimation.bots.knowledge import read_knowledge
from estimation.engine import Bola, Phase
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv import game as JG
from estimation.jaxenv.check import bits
from estimation.jaxenv.encode import SIZE, encode
from estimation.rl.config import ONENET_STAGE, derive_seed
from estimation.rl.onenet_record import (
    KIND_CARD,
    SUMMARY,
    GameRecord,
    RecordConfig,
    check_run,
    load_config,
    part_arrays,
    part_path,
    part_starts,
    save_part,
    summarize,
    write_progress,
)

from . import gpu_search as G
from .search import ROUND, OneNetSearch

Mask = npt.NDArray[np.bool_]
_DEAL, _LABELS, _SEARCH = 0, 9, 10  # seed keys: the CPU recorder's deal and labels, then ours


@dataclass(frozen=True, slots=True)
class GpuSettings:
    """How the GPU recorder batches its work."""

    batch: int = 256  # Bolas played at once
    history: int = 48  # bidding decisions with a choice kept per round (more is an error)
    redeals: int = 8  # draws per imagined deal to fit the void announcements
    chunks: int = 3  # play-out batches per search kind, by the cards left in the round


DEFAULT_GPU = GpuSettings()
DECISION_SIZES = (8, 16, 32, *range(64, 1025, 32))  # searched decisions per batch
HISTORY_SIZES = (16, 32)  # bidding rows per decision, then the whole buffer
PAIR_SIZES = tuple(range(32, 1025, 32)) + tuple(range(1152, 16385, 128))  # play-out batch sizes


def player_of(config: RecordConfig) -> tuple[str, HybridConfig]:
    """The player's net file and search settings from the record config's ``onenet`` spec."""
    bot = make_bot(config.player, 0)
    if not isinstance(bot, OneNetSearch) or not bot.own or bot.judge != ROUND:
        raise ValueError(
            "the GPU recorder plays an onenet player with own=true and judge=round, "
            f"not {config.player!r}"
        )
    options: dict[str, str] = {}
    for pair in config.player.split(":")[1:]:
        name, _, value = pair.partition("=")
        options[name] = value
    return options["net"], bot.config


def settings_of(config: HybridConfig, samples: int, redeals: int) -> G.Settings:
    return G.Settings(
        samples=samples,
        oversample=config.oversample if config.beliefs else 1,
        redeals=redeals,
        floor=config.belief_floor,
        guess_temperature=config.guess_temperature,
        leader_weight=config.leader_weight,
    )


def _bucket(n: int, sizes: Sequence[int]) -> int:
    for size in sizes:
        if n <= size:
            return size
    raise ValueError(f"{n} is more than the largest batch {sizes[-1]}")


@dataclass(slots=True)
class Slot:
    """One Bola being played: its number, the engine's game, the record, the label draws and
    this round's bidding decisions with a choice (seat, legal options, choice)."""

    game: int
    bola: Bola
    record: GameRecord
    labels: random.Random
    seats: list[int] = field(default_factory=lambda: list[int]())
    legal: list[Mask] = field(default_factory=lambda: list[Mask]())
    chosen: list[int] = field(default_factory=lambda: list[int]())


@dataclass(slots=True)
class Pending:
    """A card decision waiting for its search: the slot, the legal cards, the cards to search
    (in the order the search ties on), the runs of touching cards, whether the teacher
    searches it, and the cards still to be played in the round."""

    slot: int
    cards: list[int]
    searched: list[int]
    runs: list[list[int]]
    teacher: bool
    left: int


class Recorder:
    """Plays a recording config's Bolas ``batch`` at a time, the searches on the GPU."""

    def __init__(self, config: RecordConfig, gpu: GpuSettings = DEFAULT_GPU) -> None:
        net, hybrid = player_of(config)
        self.config = config
        self.gpu = gpu
        self.hybrid = hybrid
        self.player = G.load_player(net)
        self.player_settings = settings_of(hybrid, hybrid.samples, gpu.redeals)
        self.teacher_settings = settings_of(hybrid, config.teacher_samples, gpu.redeals)
        player = self.player
        self._views = jax.jit(
            lambda g: G.move_logits(player, jax.vmap(encode)(g.round, g.round.cur))
        )
        self._step = jax.jit(jax.vmap(JG.game_step))
        self._new = jax.jit(jax.vmap(JG.new_game))
        self._note = jax.jit(_note_templates)
        self._imagine = jax.jit(partial(G.imagine_all, player), static_argnums=0)
        self._evaluate = jax.jit(partial(G.evaluate, player), static_argnums=5)
        self._gather = jax.jit(_gather)
        self._restart = jax.jit(_restart)
        self.clock = Clock()

    def play(self, games: list[int], done: Callable[[int, GameRecord], None]) -> None:
        """Play Bolas ``games`` to the end, calling ``done`` with each finished one."""
        queue = list(reversed(games))
        b = min(self.gpu.batch, len(games))
        slots: list[Slot | None] = [self._start(queue.pop()) for _ in range(b)]
        hands = np.stack([bits(s.bola.round.hands) for s in slots if s is not None])
        g = self._new(jnp.asarray(hands))
        templates = jnp.zeros((b, self.gpu.history, SIZE), jnp.float32)
        key = jax.random.PRNGKey(derive_seed(self.config.seed, ONENET_STAGE, _SEARCH, games[0]))
        clock = self.clock = Clock()
        while any(slot is not None for slot in slots):
            clock.tick("advance")
            logits = np.asarray(self._views(g))
            clock.tick("views")
            actions = np.zeros(b, np.int32)
            pending: list[Pending] = []
            noted: list[tuple[int, int]] = []
            for i, slot in enumerate(slots):
                if slot is None:
                    continue
                bola = slot.bola
                seat = bola.current_player
                if seat is None:
                    raise RuntimeError("a Bola that is not over always has a player to act")
                legal = bola.legal_actions()
                if len(legal) == 1:
                    actions[i] = legal[0]
                    continue
                obs = bola.observation(seat)
                if obs.phase is not Phase.PLAY:
                    action = max(legal, key=lambda a: logits[i, a])
                    actions[i] = action
                    if len(slot.seats) >= self.gpu.history:
                        raise ValueError(f"more than {self.gpu.history} bidding choices in a round")
                    noted.append((i, len(slot.seats)))
                    mask = np.zeros(G.BID_ACTIONS, np.bool_)
                    mask[legal] = True
                    slot.seats.append(seat)
                    slot.legal.append(mask)
                    slot.chosen.append(action)
                    continue
                teacher = slot.labels.random() < self.config.label_share
                cards = [a - PLAY_BASE for a in legal]
                advantages = [float(logits[i, PLAY_BASE + c]) for c in cards]
                unseen = read_knowledge(obs).unseen
                merged, merged_adv = cards, advantages
                if self.hybrid.merge_equivalent:
                    merged, merged_adv = _merge_equivalent(cards, advantages, unseen)
                order = sorted(range(len(merged)), key=lambda k: (-merged_adv[k], merged[k]))
                top_k = 0 if teacher else self.hybrid.top_k
                searched = [merged[k] for k in (order[:top_k] if top_k else order)]
                if len(searched) == 1:  # nothing to search: the CPU plays its one candidate
                    actions[i] = searched[0] + PLAY_BASE
                    continue
                runs = equivalent_runs(cards, unseen)
                left = 52 - 4 * len(obs.tricks) - len(obs.trick)
                pending.append(Pending(i, cards, searched, runs, teacher, left))
            if noted:
                rows = np.full((b, 2), b, np.int32)  # one fixed size: unused rows fall off
                rows[: len(noted)] = noted
                templates = self._note(templates, g, jnp.asarray(rows))
            clock.tick("host")
            key, k_teacher, k_player = jax.random.split(key, 3)
            for teacher, k in ((True, k_teacher), (False, k_player)):
                group = [p for p in pending if p.teacher == teacher]
                if group:
                    self._search(group, teacher, k, g, templates, slots, actions)
                    clock.tick(("teacher" if teacher else "player") + " play-outs")
                    clock.count["teacher cards" if teacher else "player cards"] += sum(
                        len(p.searched) for p in group
                    )
            g = self._advance(g, slots, actions, queue, done)
            clock.step(sum(slot is not None for slot in slots))

    def _start(self, game: int) -> Slot:
        config = self.config
        bola = Bola(derive_seed(config.seed, ONENET_STAGE, game, _DEAL), DEFAULT_CONFIG)
        labels = random.Random(derive_seed(config.seed, ONENET_STAGE, game, _LABELS))
        return Slot(game, bola, GameRecord(bola.seed), labels)

    def _search(
        self,
        group: list[Pending],
        teacher: bool,
        key: jax.Array,
        g: JG.Game,
        templates: jax.Array,
        slots: list[Slot | None],
        actions: npt.NDArray[np.int32],
    ) -> None:
        """Search the pending decisions of one kind and play (and for the teacher, keep) the
        best card of each."""
        settings = self.teacher_settings if teacher else self.player_settings
        n = _bucket(len(group), DECISION_SIZES)
        most = max(len(slot.seats) for p in group if (slot := slots[p.slot]) is not None)
        cap = self.gpu.history
        h = _bucket(max(most, 1), (*(k for k in HISTORY_SIZES if k < cap), cap))
        idx = np.zeros(n, np.int32)
        seats = np.full((n, h), -1, np.int32)
        legal = np.zeros((n, h, G.BID_ACTIONS), np.bool_)
        chosen = np.zeros((n, h), np.int32)
        for row, p in enumerate(group):
            slot = slots[p.slot]
            assert slot is not None
            idx[row] = p.slot
            count = len(slot.seats)
            if count:
                seats[row, :count] = slot.seats
                legal[row, :count] = np.stack(slot.legal)
                chosen[row, :count] = slot.chosen
        idx[len(group) :] = group[0].slot
        rounds, me, kept = self._gather(g, templates, jnp.asarray(idx))
        kept = kept[:, :h]  # the bid check runs over this round's bidding only
        bidding = G.Bidding(
            kept,
            jnp.asarray(seats),
            jnp.asarray(legal),
            jnp.asarray(chosen),
            jnp.asarray(seats >= 0),
        )
        hands = self._imagine(settings, key, rounds, me, bidding)
        hands.block_until_ready()
        self.clock.tick(("teacher" if teacher else "player") + " deals")
        which = np.array([row for row, p in enumerate(group) for _ in p.searched], np.int32)
        cards = np.array([c for p in group for c in p.searched], np.int32)
        left = np.array([p.left for p in group for _ in p.searched])
        # Play-outs run until their longest is over, so they go in chunks of similar length.
        order = np.argsort(left, kind="stable")
        chunks = np.array_split(order, max(1, min(self.gpu.chunks, len(order) // 64)))
        queued: list[tuple[npt.NDArray[np.intp], jax.Array]] = []
        for part in chunks:
            size = _bucket(len(part), PAIR_SIZES)
            w = np.zeros(size, np.int32)
            c = np.full(size, cards[part[0]], np.int32)
            w[: len(part)], c[: len(part)] = which[part], cards[part]
            values = self._evaluate(
                rounds, me, hands, jnp.asarray(w), jnp.asarray(c), settings.leader_weight
            )
            queued.append((part, values.mean(-1)))
        means = np.zeros(len(cards))
        for part, mean in queued:
            means[part] = np.asarray(mean)[: len(part)]
        start = 0
        for p in group:
            mine = means[start : start + len(p.searched)].tolist()
            start += len(p.searched)
            best = int(np.argmax(mine))  # the first on a tie, as best_index
            actions[p.slot] = p.searched[best] + PLAY_BASE
            if teacher:
                value_of = dict(zip(p.searched, mine, strict=True))
                card_values = np.zeros(52, np.float32)
                for run in p.runs:
                    card_values[run] = next(value_of[c] for c in run if c in value_of)
                slot = slots[p.slot]
                assert slot is not None
                slot.record.keep(KIND_CARD, [float(card_values[c]) for c in p.cards])

    def _advance(
        self,
        g: JG.Game,
        slots: list[Slot | None],
        actions: npt.NDArray[np.int32],
        queue: list[int],
        done: Callable[[int, GameRecord], None],
    ) -> JG.Game:
        """Play every live Bola's action in the engine and in JAX; start the next round's JAX
        copy from the engine's deal, and the next Bola in a finished one's place."""
        nxt = np.zeros((len(slots), 4, 52), np.bool_)
        restart: list[int] = []
        for i, slot in enumerate(slots):
            if slot is None:
                continue
            bola = slot.bola
            rounds = len(bola.history)
            slot.record.actions.append(int(actions[i]))
            bola.step(int(actions[i]))
            if len(bola.history) > rounds:
                slot.seats.clear()
                slot.legal.clear()
                slot.chosen.clear()
                if not bola.is_over:
                    nxt[i] = bits(bola.round.hands)
            if bola.is_over:
                slot.record.scores = bola.scores
                done(slot.game, slot.record)
                if queue:
                    slots[i] = self._start(queue.pop())
                    restart.append(i)
                else:
                    slots[i] = None
        g = self._step(g, jnp.asarray(actions), jnp.asarray(nxt))
        if restart:
            b = len(slots)
            hands = np.zeros((b, 4, 52), np.bool_)
            rows = np.full(b, b, np.int32)  # one fixed size: unused rows fall off
            for k, i in enumerate(restart):
                slot = slots[i]
                assert slot is not None
                hands[k] = bits(slot.bola.round.hands)
                rows[k] = i
            g = self._restart(g, jnp.asarray(hands), jnp.asarray(rows))
        return g


class Clock:
    """Where a recording's time goes, printed every ``every`` steps."""

    def __init__(self, every: int = 200) -> None:
        self.every = every
        self.steps = 0
        self.last = time.perf_counter()
        self.seconds: dict[str, float] = {}
        self.count: dict[str, int] = {"teacher cards": 0, "player cards": 0}

    def tick(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = self.seconds.get(name, 0.0) + now - self.last
        self.last = now

    def step(self, live: int) -> None:
        self.steps += 1
        if self.steps % self.every:
            return
        total = sum(self.seconds.values())
        parts = ", ".join(f"{k} {v / total:.0%}" for k, v in self.seconds.items())
        cards = ", ".join(f"{k} {v / self.every:.0f}" for k, v in self.count.items())
        print(
            f"[gpu-record] step {self.steps}: {total / self.every * 1000:.0f} ms a step "
            f"({parts}); per step {cards}; {live} Bolas live",
            file=sys.stderr,
            flush=True,
        )
        self.seconds = {}
        self.count = dict.fromkeys(self.count, 0)


def _note_templates(templates: jax.Array, g: JG.Game, rows: jax.Array) -> jax.Array:
    """Keep the views of bidding decisions ``rows`` ``[k, 2]`` (slot, index), hand cleared."""
    slot, index = rows[:, 0], rows[:, 1]
    r = jax.tree.map(lambda a: a[jnp.minimum(slot, a.shape[0] - 1)], g.round)
    x = jax.vmap(encode)(r, r.cur).at[:, :52].set(0.0)
    return templates.at[slot, index].set(x, mode="drop")  # rows past the batch are padding


def _restart(g: JG.Game, hands: jax.Array, rows: jax.Array) -> JG.Game:
    """New Bolas in slots ``rows`` (rows past the batch are padding), dealt ``hands``."""
    new = jax.vmap(JG.new_game)(hands)
    return jax.tree.map(lambda a, b: a.at[rows].set(b, mode="drop"), g, new)


def _gather(g: JG.Game, templates: jax.Array, idx: jax.Array) -> tuple[Any, jax.Array, jax.Array]:
    r = jax.tree.map(lambda a: a[idx], g.round)
    return r, r.cur, templates[idx]


# The run.


def run(config: RecordConfig, out: Path, gpu: GpuSettings = DEFAULT_GPU) -> dict[str, Any]:
    """Record every missing part into ``out/parts``, as the CPU recorder does; return the
    recording's summary."""
    check_run(config, out)
    gpu_path = out / "gpu.json"
    wanted = dataclasses.asdict(gpu)
    if gpu_path.exists() and json.loads(gpu_path.read_text()) != wanted:
        raise SystemExit(f"{out} was recorded with other GPU settings; use a new --out")
    gpu_path.write_text(json.dumps(wanted, indent=2) + "\n")
    parts_dir = out / "parts"
    parts_dir.mkdir(exist_ok=True)
    missing = [(first, n) for first, n in part_starts(config) if not part_path(out, first).exists()]
    if missing:
        games = [game for first, n in missing for game in range(first, first + n)]
        part_of = {game: first for first, n in missing for game in range(first, first + n)}
        size = dict(missing)
        finished: dict[int, GameRecord] = {}
        got = dict.fromkeys(size, 0)
        progress: dict[str, float] = {"games": config.games}
        progress["resumed_at"] = config.games - len(games)
        progress["started"] = time.time()
        state = {"done": progress["resumed_at"]}
        write_progress(out, {**progress, "done": state["done"]})
        start = time.perf_counter()

        def done(game: int, record: GameRecord) -> None:
            first = part_of[game]
            finished[game] = record
            got[first] += 1
            counter = parts_dir / f"part-{first:05d}.games"
            if got[first] < size[first]:
                counter.write_text(f"{got[first]}\n")
                return
            records = [(g, finished.pop(g)) for g in range(first, first + size[first])]
            save_part(part_path(out, first), part_arrays(records))
            counter.unlink(missing_ok=True)
            state["done"] += size[first]
            write_progress(out, {**progress, "done": state["done"]})
            elapsed = time.perf_counter() - start
            fresh = state["done"] - progress["resumed_at"]
            left = elapsed / fresh * (config.games - state["done"])
            print(
                f"[gpu-record] {state['done']}/{config.games} Bolas, {elapsed / 60:.1f} min, "
                f"about {left / 60:.1f} min left",
                file=sys.stderr,
                flush=True,
            )

        Recorder(config, gpu).play(games, done)
        if finished:
            raise RuntimeError(f"{len(finished)} finished Bolas were never saved in a part")
        write_progress(out, {**progress, "done": state["done"], "finished": time.time()})
    summary = summarize(out, config)
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def time_run(config: RecordConfig, games: int, gpu: GpuSettings) -> dict[str, float]:
    """Record ``games`` Bolas into a scratch folder: the wall time, and the hours the config's
    whole recording would take at that pace."""
    trial = dataclasses.replace(config, games=games, part_games=games)
    with tempfile.TemporaryDirectory() as folder:
        start = time.perf_counter()
        summary = run(trial, Path(folder), gpu)
        seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "hours": seconds / games * config.games / 3600,
        "actions_per_bola": summary["actions"] / games,
        "cards_per_bola": summary["cards"] / games,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.gpu_record",
        description="Record the one net's own search with the searches on the GPU.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "record (or resume) a config's Bolas"), ("time", "time a few")):
        cmd = commands.add_parser(name, help=text)
        cmd.add_argument("--config", type=Path, required=True, help="TOML recording config")
        cmd.add_argument("--batch", type=int, default=GpuSettings.batch, help="Bolas at once")
        if name == "run":
            cmd.add_argument("--out", type=Path, required=True, help="the recording's folder")
        else:
            cmd.add_argument("--games", type=int, default=64, help="Bolas to time")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"--config {args.config}: {error}")
    gpu = GpuSettings(batch=args.batch)
    # Compiled searches are kept between runs: each batch size compiles once.
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "gpu-record"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    print(f"[gpu-record] JAX devices: {jax.devices()}", file=sys.stderr, flush=True)
    if args.command == "time":
        timing = time_run(config, args.games, gpu)
        print(
            f"{args.games} Bolas, {gpu.batch} at once, in {timing['seconds'] / 60:.1f} min: "
            f"{config.games} Bolas in about {timing['hours']:.1f} h. Per Bola: "
            f"{timing['actions_per_bola']:.0f} actions and {timing['cards_per_bola']:.0f} "
            "teacher-valued cards kept"
        )
        return 0
    print(json.dumps(run(config, args.out, gpu)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
