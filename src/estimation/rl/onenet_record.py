"""Hybrid v4's games for step 5's one-net copy (``docs/rl/hybrid-iteration-plan.md``).

    uv run python -m estimation.rl.onenet_record run --config configs/onenet-record.toml \
        --out runs/onenet/record
    uv run python -m estimation.rl.onenet_record watch runs/onenet/record
    uv run python -m estimation.rl.onenet_record time --config configs/onenet-record.toml

Hybrid v4 (``player``) plays whole Bolas in all four seats. At a share of its card decisions with
a choice (``label_share``) the teacher, the same Hybrid searching deeper (``teacher_samples``
deals, every legal card), values every legal card and plays its best one, as in step 2. At every
Dash Call, bid, With, Pass, estimate and super call with a choice, BiddingNet's value of each legal
option is kept: Hybrid v4 bids by it.

A record is compact: each Bola's seed and every action played, so the engine replays any position
exactly (the raw view, the real hands, tricks, round scores and final places all come from the
replay), plus the values above for the decisions they belong to, in points and in the order of
``legal_actions()``. ``part_games`` Bolas are saved at a time, so a stopped run resumes where it
stopped, and the same config and code give the same records.

Trainer code: the record holds whole games, but every bot sees only its own Observation.
"""

import argparse
import dataclasses
import json
import random
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from estimation.bots import HybridBot, make_bot
from estimation.engine import Bola, Observation, Phase, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG

from .config import ONENET_STAGE, derive_seed, section_from
from .encoder import encode
from .expert import teach
from .networks import REWARD_SCALE
from .train import worker_pool

KIND_BID, KIND_CARD = 0, 1  # whose values a decision keeps: BiddingNet's, or the teacher's
PROGRESS = "progress.json"
SUMMARY = "summary.json"
_DEAL, _LABELS = 0, 9  # keys of a game's seeds; seats take 1-4 (players) and 5-8 (teachers)

Arrays = dict[str, npt.NDArray[Any]]


@dataclass(frozen=True, slots=True)
class RecordConfig:
    """The ``[record]`` section of a recording config."""

    player: str = ""  # the Hybrid that plays every seat, as a bot spec (Hybrid v4)
    games: int = 2000  # Bolas recorded
    part_games: int = 25  # Bolas per saved part
    label_share: float = 0.5  # share of card decisions with a choice that the teacher values
    teacher_samples: int = 64  # the teacher's deals per card decision; it searches every card
    seed: int = 0
    workers: int = 5

    def __post_init__(self) -> None:
        if not self.player.startswith("hybrid"):
            raise ValueError("record.player must be a Hybrid bot spec, such as Hybrid v4's")
        if self.games < 1 or self.part_games < 1 or self.workers < 1:
            raise ValueError("record.games, part_games and workers must be at least 1")
        if not 0.0 <= self.label_share <= 1.0:
            raise ValueError("record.label_share must be from 0 to 1")
        if self.teacher_samples < 1:
            raise ValueError("record.teacher_samples must be at least 1")


def config_of(data: dict[str, Any]) -> RecordConfig:
    """A recording config from TOML data: one ``[record]`` section."""
    unknown = set(data) - {"record"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}: a recording has only [record]")
    section = data.get("record")
    if not isinstance(section, dict):
        raise ValueError("a recording config needs a [record] section")
    values = cast(dict[str, Any], section)
    if "player" not in values:
        raise ValueError("record.player must name the Hybrid that plays (a bot spec)")
    return section_from(RecordConfig(player=str(values["player"])), values, "record")


def load_config(path: Path) -> RecordConfig:
    return config_of(tomllib.loads(path.read_text()))


def _say(text: str) -> None:
    print(f"[record] {text}", file=sys.stderr, flush=True)


# Playing and recording one Bola.


def bots(config: RecordConfig, game: int) -> tuple[list[HybridBot], list[HybridBot]]:
    """Each seat's playing Hybrid and teacher for Bola ``game``, seeded from the game."""
    players: list[HybridBot] = []
    teachers: list[HybridBot] = []
    for seat in range(4):
        player = make_bot(config.player, derive_seed(config.seed, ONENET_STAGE, game, 1 + seat))
        if not isinstance(player, HybridBot):
            raise ValueError(f"record.player {config.player!r} is not a plain Hybrid")
        deeper = dataclasses.replace(player.config, samples=config.teacher_samples, top_k=0)
        players.append(player)
        teachers.append(HybridBot(derive_seed(config.seed, ONENET_STAGE, game, 5 + seat), deeper))
    return players, teachers


@dataclass(slots=True)
class GameRecord:
    """One recorded Bola: its seed, every action, its final scores, and the values kept at its
    decisions (each decision's index in ``actions``, kind and values)."""

    seed: int
    actions: list[int] = field(default_factory=lambda: list[int]())
    scores: tuple[int, int, int, int] = (0, 0, 0, 0)
    steps: list[int] = field(default_factory=lambda: list[int]())
    kinds: list[int] = field(default_factory=lambda: list[int]())
    values: list[list[float]] = field(default_factory=lambda: list[list[float]]())

    def keep(self, kind: int, values: list[float]) -> None:
        """Keep ``values`` for the decision about to be played."""
        self.steps.append(len(self.actions))
        self.kinds.append(kind)
        self.values.append(values)


def record_game(config: RecordConfig, game: int) -> GameRecord:
    """Play Bola ``game`` with the config's Hybrid in every seat and record it."""
    bola = Bola(derive_seed(config.seed, ONENET_STAGE, game, _DEAL), DEFAULT_CONFIG)
    players, teachers = bots(config, game)
    labels = random.Random(derive_seed(config.seed, ONENET_STAGE, game, _LABELS))
    record = GameRecord(bola.seed)
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        legal = bola.legal_actions()
        action = legal[0]
        if len(legal) > 1:
            obs = bola.observation(seat)
            action = _decide(config, obs, legal, players[seat], teachers[seat], labels, record)
        record.actions.append(action)
        bola.step(action)
    record.scores = bola.scores
    return record


def _decide(
    config: RecordConfig,
    obs: Observation,
    legal: list[int],
    player: HybridBot,
    teacher: HybridBot,
    labels: random.Random,
    record: GameRecord,
) -> int:
    """The action played at a decision with a choice, keeping its values when it has some."""
    features = encode(obs, DEFAULT_CONFIG)
    if obs.phase is not Phase.PLAY:
        values = player.policy.bid_values(features[np.newaxis])[0] * REWARD_SCALE
        record.keep(KIND_BID, [float(values[a]) for a in legal])
        return player.act(obs, legal)
    if labels.random() < config.label_share:
        cards = [a - PLAY_BASE for a in legal]
        taught = teach(teacher, features, obs, cards)
        if taught is not None:
            choice, card_values = taught
            record.keep(KIND_CARD, [float(card_values[c]) for c in cards])
            return play_action(choice)
    return player.act(obs, legal)


def part_arrays(records: Sequence[tuple[int, GameRecord]]) -> Arrays:
    """Recorded Bolas (each with its game number) as one part's arrays. ``step`` indexes the
    part's ``actions``; ``log_start`` and ``value_start`` hold each Bola's and each decision's
    first index."""
    lengths = [len(record.actions) for _, record in records]
    log_start = np.zeros(len(records) + 1, np.int64)
    log_start[1:] = np.cumsum(lengths)
    steps: list[int] = []
    kinds: list[int] = []
    flat: list[float] = []
    value_start = [0]
    for (_, record), offset in zip(records, log_start[:-1].tolist(), strict=True):
        for step, kind, values in zip(record.steps, record.kinds, record.values, strict=True):
            steps.append(offset + step)
            kinds.append(kind)
            flat.extend(values)
            value_start.append(len(flat))
    actions = [a for _, record in records for a in record.actions]
    return {
        "game": np.array([game for game, _ in records], np.int32),
        "seed": np.array([record.seed for _, record in records], np.int64),
        "scores": np.array([record.scores for _, record in records], np.int64).reshape(-1, 4),
        "log_start": log_start,
        "actions": np.array(actions, np.uint8),
        "step": np.array(steps, np.int64),
        "kind": np.array(kinds, np.int8),
        "value_start": np.array(value_start, np.int64),
        "values": np.array(flat, np.float32),
    }


def save_part(path: Path, arrays: Arrays) -> None:
    """Write a part so that a reader never sees it half written."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        game=arrays["game"],
        seed=arrays["seed"],
        scores=arrays["scores"],
        log_start=arrays["log_start"],
        actions=arrays["actions"],
        step=arrays["step"],
        kind=arrays["kind"],
        value_start=arrays["value_start"],
        values=arrays["values"],
    )
    tmp.replace(path)


def load_part(path: Path) -> Arrays:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


@dataclass(frozen=True, slots=True)
class Decision:
    """A recorded decision, replayed: the Bola just before it, the acting seat, its legal
    actions, the action played, and the values kept (``kind`` says whose)."""

    bola: Bola
    seat: int
    legal: list[int]
    action: int
    kind: int
    values: npt.NDArray[np.float32]


def replay(part: Arrays, row: int) -> Bola:
    """Bola ``row`` of a part, replayed from its seed and actions to its end."""
    bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
    start, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
    for action in part["actions"][start:end].tolist():
        bola.step(action)
    return bola


def decisions(part: Arrays) -> Iterator[Decision]:
    """Every recorded decision of a part, in order, replayed from the Bolas' seeds. The Bola
    yielded is the live replay: read it before asking for the next decision."""
    kept = dict(zip(part["step"].tolist(), range(len(part["step"])), strict=True))
    actions = part["actions"].tolist()
    for row in range(len(part["seed"])):
        bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
        for index in range(int(part["log_start"][row]), int(part["log_start"][row + 1])):
            k = kept.get(index)
            if k is not None:
                seat = bola.current_player
                if seat is None:
                    raise ValueError(f"part decision {k} is at a step with nobody to act")
                values = part["values"][part["value_start"][k] : part["value_start"][k + 1]]
                kind = int(part["kind"][k])
                yield Decision(bola, seat, bola.legal_actions(), actions[index], kind, values)
            bola.step(actions[index])


# The run.


@dataclass(frozen=True, slots=True)
class PartTask:
    """Bolas ``first`` to ``first + count - 1``; ``progress`` is a file the part keeps its
    finished Bola count in, for ``watch``."""

    first: int
    count: int
    config: RecordConfig
    progress: str = ""


def record_part(task: PartTask) -> Arrays:
    """Every Bola of ``task``, in order, as one part's arrays."""
    records: list[tuple[int, GameRecord]] = []
    for done, game in enumerate(range(task.first, task.first + task.count), 1):
        records.append((game, record_game(task.config, game)))
        if task.progress:
            Path(task.progress).write_text(f"{done}\n")
    return part_arrays(records)


def part_starts(config: RecordConfig) -> list[tuple[int, int]]:
    """Each part's first Bola and number of Bolas."""
    return [
        (first, min(config.part_games, config.games - first))
        for first in range(0, config.games, config.part_games)
    ]


def part_path(out: Path, first: int) -> Path:
    return out / "parts" / f"part-{first:05d}.npz"


def write_progress(out: Path, data: dict[str, Any]) -> None:
    tmp = out / (PROGRESS + ".tmp")
    tmp.write_text(json.dumps(data) + "\n")
    tmp.replace(out / PROGRESS)


def check_run(config: RecordConfig, out: Path) -> None:
    """Start ``out`` with the config, or check that it holds the same config (a resume)."""
    path = out / "config.json"
    wanted = dataclasses.asdict(config)
    if path.exists():
        if json.loads(path.read_text()) != wanted:
            raise SystemExit(f"{out} holds a recording with a different config; use a new --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wanted, indent=2) + "\n")


def run(config: RecordConfig, out: Path) -> dict[str, Any]:
    """Record every missing part into ``out/parts``; return the recording's summary."""
    check_run(config, out)
    (out / "parts").mkdir(exist_ok=True)
    parts = part_starts(config)
    missing = [(first, count) for first, count in parts if not part_path(out, first).exists()]
    if missing:
        done = config.games - sum(count for _, count in missing)
        progress = {"games": config.games, "resumed_at": done, "started": time.time()}
        write_progress(out, {**progress, "done": done})
        _say(f"{len(missing)} of {len(parts)} parts to record on {config.workers} workers")
        start = time.perf_counter()
        with worker_pool(config.workers) as pool:
            futures = {
                pool.submit(
                    record_part,
                    PartTask(first, count, config, str(out / "parts" / f"part-{first:05d}.games")),
                ): first
                for first, count in missing
            }
            for future in as_completed(futures):
                first = futures[future]
                arrays = future.result()
                save_part(part_path(out, first), arrays)
                (out / "parts" / f"part-{first:05d}.games").unlink(missing_ok=True)
                done += len(arrays["game"])
                write_progress(out, {**progress, "done": done})
                elapsed = time.perf_counter() - start
                left = elapsed / (done - progress["resumed_at"]) * (config.games - done)
                _say(
                    f"{done}/{config.games} Bolas, {elapsed / 60:.1f} min, "
                    f"about {left / 60:.1f} min left"
                )
        write_progress(out, {**progress, "done": done, "finished": time.time()})
    summary = summarize(out, config)
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def part_counts(path: Path) -> dict[str, int]:
    """A saved part's Bolas, actions, and decisions kept with BiddingNet's and the teacher's
    values."""
    with np.load(path) as data:
        kind = data["kind"]
        return {
            "bolas": len(data["game"]),
            "actions": len(data["actions"]),
            "bids": int((kind == KIND_BID).sum()),
            "cards": int((kind == KIND_CARD).sum()),
        }


def summarize(out: Path, config: RecordConfig) -> dict[str, Any]:
    totals = {"bolas": 0, "actions": 0, "bids": 0, "cards": 0}
    for first, _ in part_starts(config):
        for key, value in part_counts(part_path(out, first)).items():
            totals[key] += value
    return totals


def time_run(config: RecordConfig, games: int) -> dict[str, float]:
    """Record ``games`` Bolas on the config's workers, one Bola a part, into a scratch folder:
    the wall time, and the hours the config's whole recording would take at that pace."""
    trial = dataclasses.replace(config, games=games, part_games=1)
    with tempfile.TemporaryDirectory() as folder:
        start = time.perf_counter()
        summary = run(trial, Path(folder))
        seconds = time.perf_counter() - start
    return {
        "games": float(games),
        "seconds": seconds,
        "hours": seconds / games * config.games / 3600,
        "actions_per_bola": summary["actions"] / games,
        "bids_per_bola": summary["bids"] / games,
        "cards_per_bola": summary["cards"] / games,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.rl.onenet_record",
        description="Record Hybrid v4's games for step 5's one-net copy.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="record (or resume) a config's Bolas")
    run_cmd.add_argument("--config", type=Path, required=True, help="TOML recording config")
    run_cmd.add_argument("--out", type=Path, required=True, help="the recording's folder")
    time_cmd = commands.add_parser("time", help="time a few Bolas on the config's workers")
    time_cmd.add_argument("--config", type=Path, required=True, help="TOML recording config")
    time_cmd.add_argument("--games", type=int, default=10, help="Bolas to time")
    watch_cmd = commands.add_parser("watch", help="watch a recording live")
    watch_cmd.add_argument("out", type=Path, help="the recording's folder")
    watch_cmd.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)
    if args.command == "watch":
        from .onenet_watch import watch

        return watch(args.out, once=args.once)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"--config {args.config}: {error}")
    if args.command == "time":
        timing = time_run(config, args.games)
        print(
            f"{args.games} Bolas on {config.workers} workers in {timing['seconds'] / 60:.1f} min: "
            f"{config.games} Bolas in about {timing['hours']:.1f} h. Per Bola: "
            f"{timing['actions_per_bola']:.0f} actions, {timing['bids_per_bola']:.0f} bids and "
            f"{timing['cards_per_bola']:.0f} teacher-valued cards kept"
        )
        return 0
    print(json.dumps(run(config, args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
