"""Expert iteration for the Hybrid (Hybrid plan step 2, ``docs/rl/hybrid-iteration-plan.md``).

    uv run python -m estimation.rl.expert run --config configs/expert.toml --out runs/expert

Each round improves the nets the Hybrid is built from, in three stages:

1. **record:** the round's Hybrid plays whole Bolas in all four seats. At a share of its card
   decisions (``label_share``) a deeper Hybrid search, the teacher (``teacher_samples`` deals,
   every legal card), values every legal card and plays its best one. The record keeps the acting
   seat's encoded Observation, its legal cards, the teacher's value of each (touching cards share
   their run's value), the teacher's choice, who held each unseen card and the round's reward.
2. **copy:** CardPlayNet, from the round's card net, learns the teacher's values as the PIMC warm
   start learned PIMC's (``[imitation]``: each card's value against the chosen card's, the
   choice, and the critic onto the reward), and GuessNet learns the recorded positions' real
   holders. Both train on the records of the last ``replay_rounds`` rounds. Held-out games
   measure how much the net's top 1, 2 and 3 cards lose under the teacher's values, before and
   after, and GuessNet's accuracy.
3. **refit:** a short self-play run (``train``) in which only BiddingNet learns, playing with the
   new card net, so the bidding fits the card play it bids for (``[selfplay]`` of the config).

``round-K/hybrid.pt`` holds the copied card net, the refit BiddingNet and GuessNet (trained further
by the refit's self-play), and the next round's
Hybrid plays it. Every stage is skipped when its output exists, and the recording is saved in
parts, so an interrupted run resumes; ``watch`` shows a run live. The same config, seed and code
give the same records and checkpoints.

Trainer code: it holds real Bolas to label the guess head, but every bot sees only its own
Observation.
"""

import argparse
import copy
import dataclasses
import json
import math
import random
import shutil
import sys
import time
import tomllib
from collections.abc import Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from estimation.bots import HybridBot, HybridConfig
from estimation.bots.hybrid.bot import equivalent_runs
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import best_index
from estimation.engine import Observation, Phase, play_action
from estimation.engine.actions import PLAY_BASE

from .config import (
    EXPERT_STAGE,
    TrainConfig,
    config_from_dict,
    config_to_dict,
    derive_seed,
    section_from,
)
from .encoder import encode
from .games import RewardedBola
from .guesses import blind_right, guess_labels, guess_loss
from .imitation import NEG, ImitationData, holdout_mask, imitation_losses
from .networks import CardPlayNet, GuessNet, load_policy, save_checkpoint
from .record import CardRecords, concat_records, empty_records, load_records, save_records
from .symmetry import SuitAugmenter
from .torch_typing import backward, seed_torch, seeded_generator, step
from .train import resolve_device, train, worker_pool

GAME_ID = 1_000_000  # record game ids are round * GAME_ID + game, unique across rounds
PROGRESS = "progress.json"
COPY_REPORT = "copy.json"
ROUND_CHECKPOINT = "hybrid.pt"
STAGES = ("record", "copy", "refit")
_COPY_SHUFFLE, _COPY_RELABEL, _COPY_INIT, _GUESS_SHUFFLE = 1, 2, 3, 4
_DEAL, _LABELS = 0, 9  # keys of a game's seeds; seats take 1-4 (players) and 5-8 (teachers)


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """The ``[expert]`` section: what each round records and how the card net copies it."""

    start_from: str = ""  # the checkpoint round 1's Hybrid plays (RL v3)
    rounds: int = 3
    games: int = 2000  # Hybrid games recorded per round
    part_games: int = 25  # games per saved part of a recording
    label_share: float = 0.25  # share of card decisions with a choice that the teacher labels
    play_samples: int = 16  # the playing Hybrid: deals per card decision...
    play_top_k: int = 3  # ...and cards searched
    teacher_samples: int = 64  # the teacher: deals per card decision, every legal card
    copy_epochs: int = 4
    copy_batch: int = 1024
    copy_lr: float = 1e-4  # falls to 0 over the copy by a cosine
    replay_rounds: int = 2  # the copy trains on this many rounds of records, the latest included
    holdout: float = 0.05  # share of each round's games kept out to measure the copy
    guess_lr: float = 3e-4  # GuessNet's learning rate in the copy; 0 leaves it as it is
    augment: bool = True  # relabel suits at random in every copy batch

    def __post_init__(self) -> None:
        if not self.start_from:
            raise ValueError("expert.start_from must name the checkpoint to start from")
        if self.rounds < 1 or self.games < 1 or self.part_games < 1 or self.copy_epochs < 1:
            raise ValueError("rounds, games, part_games and copy_epochs must be at least 1")
        if not 0.0 < self.label_share <= 1.0:
            raise ValueError("label_share must be above 0 and at most 1")
        if self.play_samples < 1 or self.teacher_samples < 1 or self.play_top_k < 0:
            raise ValueError("the Hybrids need samples >= 1 and top_k >= 0")
        if self.replay_rounds < 1 or self.guess_lr < 0 or self.copy_lr <= 0:
            raise ValueError("replay_rounds >= 1, guess_lr >= 0 and copy_lr > 0, please")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """An expert run: ``[expert]``, and everything else as a training config, whose
    ``[selfplay]`` is the bidding refit and whose ``[imitation]`` weighs the copy's losses."""

    expert: ExpertConfig
    train: TrainConfig = field(default_factory=TrainConfig)


def config_of(data: dict[str, Any]) -> RunConfig:
    """A run config from TOML data: ``[expert]`` plus a training config's keys."""
    rest = dict(data)
    section = rest.pop("expert", None)
    if not isinstance(section, dict):
        raise ValueError("an expert config needs an [expert] section")
    values = cast(dict[str, Any], section)
    if "start_from" not in values:
        raise ValueError("expert.start_from must name the checkpoint to start from")
    expert = section_from(ExpertConfig(start_from=str(values["start_from"])), values, "expert")
    return RunConfig(expert, config_from_dict(rest))


def load_run_config(path: Path) -> RunConfig:
    return config_of(tomllib.loads(path.read_text()))


def refit_config(config: RunConfig, start: Path) -> TrainConfig:
    """The bidding refit's training config: from ``start``, card play frozen throughout (so no
    card anchor and no recording), no yardstick (the round is measured as a Hybrid instead)."""
    play = dataclasses.replace(
        config.train.selfplay,
        start_from=str(start),
        card_frozen_iterations=config.train.selfplay.iterations,
        card_anchor_weight=0.0,
        yardstick_every=0,
        yardstick_reference="",
    )
    return dataclasses.replace(config.train, selfplay=play)


def _say(stage: str, text: str) -> None:
    print(f"[{stage}] {text}", file=sys.stderr, flush=True)


def write_progress(folder: Path, data: dict[str, Any]) -> None:
    """The round's stage and progress, for ``watch``."""
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (PROGRESS + ".tmp")
    tmp.write_text(json.dumps(data) + "\n")
    tmp.replace(folder / PROGRESS)


# The recording.


@dataclass(frozen=True, slots=True)
class PartTask:
    """Games ``first`` to ``first + count - 1`` of round ``round_no``, played by the Hybrid on
    ``checkpoint``."""

    round_no: int
    first: int
    count: int
    checkpoint: str
    config: RunConfig
    progress: str = ""  # a file the part keeps its finished game count in, for ``watch``


def hybrids(task: PartTask, game: int) -> tuple[list[HybridBot], list[HybridBot]]:
    """Each seat's playing Hybrid and teacher for one game, seeded from the game."""
    expert, rules = task.config.expert, task.config.train.rules
    seed = task.config.train.seed
    play = HybridConfig(
        checkpoint=task.checkpoint,
        samples=expert.play_samples,
        top_k=expert.play_top_k,
        rules=rules,
    )
    teach = dataclasses.replace(play, samples=expert.teacher_samples, top_k=0)
    players = [
        HybridBot(derive_seed(seed, EXPERT_STAGE, task.round_no, game, 1 + s), play)
        for s in range(4)
    ]
    teachers = [
        HybridBot(derive_seed(seed, EXPERT_STAGE, task.round_no, game, 5 + s), teach)
        for s in range(4)
    ]
    return players, teachers


@dataclass(slots=True)
class _Rows:
    """A game's labelled decisions so far."""

    features: list[npt.NDArray[np.float32]] = field(
        default_factory=lambda: list[npt.NDArray[np.float32]]()
    )
    legal: list[npt.NDArray[np.bool_]] = field(
        default_factory=lambda: list[npt.NDArray[np.bool_]]()
    )
    choice: list[int] = field(default_factory=lambda: list[int]())
    values: list[npt.NDArray[np.float32]] = field(
        default_factory=lambda: list[npt.NDArray[np.float32]]()
    )
    guesses: list[npt.NDArray[np.int8]] = field(
        default_factory=lambda: list[npt.NDArray[np.int8]]()
    )
    seats: list[int] = field(default_factory=lambda: list[int]())
    rewards: list[float] = field(default_factory=lambda: list[float]())


def teach(
    teacher: HybridBot, obs: Observation, cards: list[int]
) -> tuple[int, npt.NDArray[np.float32]] | None:
    """The teacher's card and its value of every legal card (touching cards share their run's
    value), or None when every legal card is equivalent and nothing is searched."""
    search = teacher.search(obs, cards, teacher.card_advantages(obs, cards))
    if len(search.cards) < 2:
        return None
    value_of = dict(zip(search.cards, search.means, strict=True))
    values = np.zeros(52, np.float32)
    for run in equivalent_runs(cards, read_knowledge(obs).unseen):
        values[run] = next(value_of[c] for c in run if c in value_of)
    return search.cards[best_index(search.means)], values


def record_game(task: PartTask, game: int) -> CardRecords:
    """Play game ``game`` of the round and return the teacher's labelled decisions."""
    config = task.config
    rules, seed = config.train.rules, config.train.seed
    played = RewardedBola(
        derive_seed(seed, EXPERT_STAGE, task.round_no, game, _DEAL), rules, config.train.reward
    )
    bola = played.bola
    players, teachers = hybrids(task, game)
    labels = random.Random(derive_seed(seed, EXPERT_STAGE, task.round_no, game, _LABELS))
    rows = _Rows()
    open_rows: list[int] = []
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        action = -1
        choosing = obs.phase is Phase.PLAY and len(legal) > 1
        if choosing and labels.random() < config.expert.label_share:
            features = encode(obs, rules)
            cards = [a - PLAY_BASE for a in legal]
            taught = teach(teachers[seat], obs, cards)
            if taught is not None:
                choice, values = taught
                mask = np.zeros(52, np.bool_)
                mask[cards] = True
                open_rows.append(len(rows.choice))
                rows.features.append(features)
                rows.legal.append(mask)
                rows.choice.append(choice)
                rows.values.append(values)
                rows.guesses.append(guess_labels(features, bola.round.hands, seat))
                rows.seats.append(seat)
                rows.rewards.append(0.0)
                action = play_action(choice)
        if action < 0:
            action = players[seat].act(obs, legal)
        round_rewards = played.step(action)
        if round_rewards is not None:
            for row in open_rows:
                rows.rewards[row] = round_rewards[rows.seats[row]]
            open_rows.clear()
    if not rows.choice:
        return dataclasses.replace(empty_records(), guesses=np.zeros((0, 52), np.int8))
    return CardRecords(
        features=np.stack(rows.features).astype(np.float16),
        legal=np.stack(rows.legal),
        choice=np.array(rows.choice, np.int16),
        values=np.stack(rows.values),
        reward=np.array(rows.rewards, np.float32),
        game=np.full(len(rows.choice), task.round_no * GAME_ID + game, np.int32),
        guesses=np.stack(rows.guesses),
    )


def record_part(task: PartTask) -> CardRecords:
    """Every game of ``task``, in order, counting finished games in ``task.progress``."""
    parts: list[CardRecords] = []
    for done, game in enumerate(range(task.first, task.first + task.count), 1):
        parts.append(record_game(task, game))
        if task.progress:
            Path(task.progress).write_text(f"{done}\n")
    return concat_records(parts)


def games_in_progress(folder: Path) -> int:
    """Games finished in parts still being recorded in round ``folder``."""
    total = 0
    for path in (folder / "records").glob("part-*.games"):
        try:
            total += int(path.read_text())
        except (OSError, ValueError):
            continue  # being written or just removed; the next look counts it
    return total


def part_starts(expert: ExpertConfig) -> list[tuple[int, int]]:
    """Each part's first game and number of games."""
    return [
        (first, min(expert.part_games, expert.games - first))
        for first in range(0, expert.games, expert.part_games)
    ]


def load_round_records(folder: Path) -> CardRecords:
    parts = sorted((folder / "records").glob("part-*.npz"))
    return concat_records([load_records(str(path)) for path in parts])


def record_stage(config: RunConfig, round_no: int, folder: Path, checkpoint: Path) -> CardRecords:
    """Round ``round_no``'s records, recorded part by part into ``folder/records``."""
    expert = config.expert
    parts_dir = folder / "records"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = part_starts(expert)
    path_of = {first: parts_dir / f"part-{first:05d}.npz" for first, _ in parts}
    missing = [(first, count) for first, count in parts if not path_of[first].exists()]
    if missing:
        _say("record", f"round {round_no}: {len(missing)} of {len(parts)} parts to record")
        done = expert.games - sum(count for _, count in missing)
        start = time.perf_counter()
        resumed_at = done
        started = time.time()
        progress = {"stage": "record", "resumed_at": done, "games": expert.games}
        write_progress(folder, {**progress, "done": done, "started": started})
        with worker_pool(config.train.workers) as pool:
            futures = {
                pool.submit(
                    record_part,
                    PartTask(
                        round_no,
                        first,
                        count,
                        str(checkpoint),
                        config,
                        str(parts_dir / f"part-{first:05d}.games"),
                    ),
                ): (first, count)
                for first, count in missing
            }
            for future in as_completed(futures):
                first, count = futures[future]
                tmp = parts_dir / f"part-{first:05d}.tmp.npz"
                save_records(str(tmp), future.result())
                tmp.replace(path_of[first])
                (parts_dir / f"part-{first:05d}.games").unlink(missing_ok=True)
                done += count
                elapsed = time.perf_counter() - start
                left = elapsed / (done - resumed_at) * (expert.games - done)
                write_progress(folder, {**progress, "done": done, "started": started})
                _say(
                    "record",
                    f"round {round_no}: {done}/{expert.games} games, {elapsed / 60:.1f} min, "
                    f"about {left / 60:.1f} min left",
                )
    records = concat_records([load_records(str(path_of[first])) for first, _ in parts])
    _say("record", f"round {round_no}: {len(records):,} labelled card decisions")
    return records


# The copy.


@dataclass(frozen=True, slots=True)
class Regret:
    """Points the best of the net's top 1, 2 and 3 cards loses to the teacher's best card, by
    the teacher's values: the held-out mean."""

    top1: float
    top2: float
    top3: float


def regret(net: CardPlayNet, data: ImitationData) -> Regret:
    """``net``'s ``Regret`` over ``data``: the search's own view, since the Hybrid searches the
    net's favourite cards."""
    totals = [0.0, 0.0, 0.0]
    net.eval()
    with torch.no_grad():
        for begin in range(0, data.n, 4096):
            idx = torch.arange(begin, min(begin + 4096, data.n), device=data.x.device)
            _, advantage = net(data.x[idx].float())
            legal = data.legal[idx]
            values = torch.where(legal, data.values[idx], NEG)
            best = values.max(1).values
            order = torch.where(legal, advantage, NEG).argsort(1, descending=True)
            for k in range(3):
                picked = values.gather(1, order[:, : k + 1]).max(1).values
                totals[k] += float((best - picked).sum())
    n = max(1, data.n)
    return Regret(totals[0] / n, totals[1] / n, totals[2] / n)


def copy_stage(
    config: RunConfig,
    round_no: int,
    records: CardRecords,
    start: CardPlayNet,
    device: torch.device,
    progress: Path | None = None,
) -> tuple[CardPlayNet, dict[str, Any]]:
    """``start`` trained to copy the teacher on ``records``, and a report of the copy."""
    expert, imitation = config.expert, config.train.imitation
    seed = config.train.seed
    begin = time.perf_counter()
    held = holdout_mask(records.game, expert.holdout)
    train_rows, test_rows = np.flatnonzero(~held), np.flatnonzero(held)
    seed_torch(derive_seed(seed, EXPERT_STAGE, round_no, _COPY_INIT))
    net = copy.deepcopy(start).to(device)
    data = ImitationData(records, train_rows, device)
    test = ImitationData(records, test_rows, device)
    before = regret(net, test)
    steps_per_epoch = max(1, math.ceil(data.n / expert.copy_batch))
    total_steps = steps_per_epoch * expert.copy_epochs
    optimizer = torch.optim.Adam(net.parameters(), lr=expert.copy_lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total_steps) / total_steps))
    )
    rng = np.random.default_rng(derive_seed(seed, EXPERT_STAGE, round_no, _COPY_SHUFFLE))
    relabel = seeded_generator(derive_seed(seed, EXPERT_STAGE, round_no, _COPY_RELABEL))
    augmenter = SuitAugmenter(device) if expert.augment else None
    epochs: list[dict[str, float]] = []
    for epoch in range(expert.copy_epochs):
        net.train()
        order = torch.as_tensor(rng.permutation(data.n)).to(device)
        sums = dict.fromkeys(("loss", "advantage", "choice", "value"), 0.0)
        for k in range(steps_per_epoch):
            if progress is not None and k % 100 == 0:
                write_progress(
                    progress,
                    {
                        "stage": "copy",
                        "done": epoch + k / steps_per_epoch,
                        "epochs": expert.copy_epochs,
                        "seconds": time.perf_counter() - begin,
                    },
                )
            idx = order[k * expert.copy_batch : (k + 1) * expert.copy_batch]
            batch = data.batch(idx, augmenter, relabel)
            adv_loss, choice_loss, value_loss = imitation_losses(net, batch, imitation.temperature)
            loss = (
                imitation.advantage_weight * adv_loss
                + imitation.choice_weight * choice_loss
                + imitation.value_weight * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            step(optimizer)
            schedule.step()
            parts = (loss, adv_loss, choice_loss, value_loss)
            for name, part in zip(sums, parts, strict=True):
                sums[name] += float(part.detach())
        means = {name: total / steps_per_epoch for name, total in sums.items()}
        epochs.append(means)
        _say(
            "copy",
            f"round {round_no}: epoch {epoch + 1}/{expert.copy_epochs}, loss {means['loss']:.4f}",
        )
        if progress is not None:
            write_progress(
                progress,
                {
                    "stage": "copy",
                    "done": epoch + 1,
                    "epochs": expert.copy_epochs,
                    "seconds": time.perf_counter() - begin,
                },
            )
    after = regret(net, test)
    report = {
        "round": round_no,
        "train_rows": len(train_rows),
        "holdout_rows": len(test_rows),
        "regret_before": dataclasses.asdict(before),
        "regret_after": dataclasses.asdict(after),
        "epochs": epochs,
        "seconds": time.perf_counter() - begin,
    }
    return net.cpu().eval(), report


def copy_guesses(
    config: RunConfig,
    round_no: int,
    records: CardRecords,
    start: GuessNet,
    device: torch.device,
) -> tuple[GuessNet, dict[str, float]]:
    """``start`` trained on the recorded positions' real holders, and its held-out accuracy before
    and after, next to a blind guess's."""
    expert = config.expert
    if records.guesses is None:
        raise ValueError("these records have no guess labels")
    held = holdout_mask(records.game, expert.holdout)
    x = torch.as_tensor(records.features).to(device)
    labels = torch.as_tensor(records.guesses).to(device)
    train_rows = torch.as_tensor(np.flatnonzero(~held)).to(device)
    test_rows = torch.as_tensor(np.flatnonzero(held)).to(device)
    net = copy.deepcopy(start).to(device)

    def accuracy() -> tuple[float, float]:
        net.eval()
        right = blind = count = 0.0
        with torch.no_grad():
            for begin in range(0, len(test_rows), 4096):
                rows = test_rows[begin : begin + 4096]
                features = x[rows].float()
                _, hits, labelled = guess_loss(net(features), labels[rows])
                right += float(hits)
                blind += float(blind_right(features, labels[rows]))
                count += float(labelled)
        return right / max(1.0, count), blind / max(1.0, count)

    before, blind = accuracy()
    optimizer = torch.optim.Adam(net.parameters(), lr=expert.guess_lr)
    rng = np.random.default_rng(
        derive_seed(config.train.seed, EXPERT_STAGE, round_no, _GUESS_SHUFFLE)
    )
    net.train()
    for _ in range(expert.copy_epochs):
        order = train_rows[torch.as_tensor(rng.permutation(len(train_rows))).to(device)]
        for begin in range(0, len(order), expert.copy_batch):
            rows = order[begin : begin + expert.copy_batch]
            loss, _, _ = guess_loss(net(x[rows].float()), labels[rows])
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            step(optimizer)
    after, _ = accuracy()
    return net.cpu().eval(), {"before": before, "after": after, "blind": blind}


# A round and a run.


def round_dir(out: Path, round_no: int) -> Path:
    return out / f"round-{round_no}"


def round_checkpoint(config: RunConfig, out: Path, round_no: int) -> Path:
    """The checkpoint round ``round_no``'s Hybrid plays: the start, or the last round's."""
    if round_no == 1:
        return Path(config.expert.start_from)
    return round_dir(out, round_no - 1) / ROUND_CHECKPOINT


def copy_round(config: RunConfig, out: Path, round_no: int, playing: Path) -> Path:
    """The round's copy checkpoint: the copied card net and the round's BiddingNet."""
    folder = round_dir(out, round_no)
    copied = folder / "copy.pt"
    if copied.exists():
        return copied
    first = max(1, round_no - config.expert.replay_rounds + 1)
    records = concat_records(
        [load_round_records(round_dir(out, r)) for r in range(first, round_no + 1)]
    )
    device = resolve_device(config.train.nets.device)
    start = load_policy(playing)
    _say("copy", f"round {round_no}: copying the teacher on {len(records):,} decisions")
    write_progress(folder, {"stage": "copy", "done": 0, "epochs": config.expert.copy_epochs})
    net, report = copy_stage(config, round_no, records, start.card, device, folder)
    guess = start.guess
    if guess is not None and config.expert.guess_lr > 0:
        guess, report["guess_accuracy"] = copy_guesses(config, round_no, records, guess, device)
    meta = {"stage": "expert-copy", "round": round_no, "plays": str(playing), **report}
    save_checkpoint(copied, net, start.bidding, meta, guess)
    (folder / COPY_REPORT).write_text(json.dumps(report, indent=1) + "\n")
    before, after = report["regret_before"], report["regret_after"]
    _say(
        "copy",
        f"round {round_no}: held-out regret of the net's top 1/2/3 cards "
        f"{before['top1']:.2f}/{before['top2']:.2f}/{before['top3']:.2f} points before, "
        f"{after['top1']:.2f}/{after['top2']:.2f}/{after['top3']:.2f} after",
    )
    return copied


def run_round(config: RunConfig, out: Path, round_no: int, until: str = "refit") -> Path:
    """Run round ``round_no``'s stages up to ``until`` and return the round's checkpoint (or the
    last output, after ``record`` or ``copy``)."""
    folder = round_dir(out, round_no)
    final = folder / ROUND_CHECKPOINT
    if final.exists():
        _say("round", f"round {round_no}: reusing {final}")
        return final
    playing = round_checkpoint(config, out, round_no)
    if not playing.is_file():
        raise SystemExit(f"round {round_no} plays {playing}, which is not a file")
    record_stage(config, round_no, folder, playing)
    if until == "record":
        return folder / "records"
    copied = copy_round(config, out, round_no, playing)
    if until == "copy":
        return copied
    write_progress(folder, {"stage": "refit"})
    refit = train(refit_config(config, copied), folder / "refit")
    shutil.copyfile(refit, final)
    write_progress(folder, {"stage": "done"})
    _say("round", f"round {round_no}: done, {final}")
    return final


def check_run(config: RunConfig, out: Path) -> None:
    """Write the config the first time; refuse to resume a folder run differently."""
    path = out / "config.json"
    data = {"expert": dataclasses.asdict(config.expert), "train": config_to_dict(config.train)}
    text = json.dumps(data, indent=1, sort_keys=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(text):
            raise SystemExit(f"{path} holds a different config; use a new --out folder")
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def run(config: RunConfig, out: Path, rounds: int | None = None, until: str = "refit") -> Path:
    """Rounds 1 to ``rounds`` (default the config's), the last one up to ``until``."""
    check_run(config, out)
    last = config.expert.rounds if rounds is None else min(rounds, config.expert.rounds)
    path = Path(config.expert.start_from)
    for round_no in range(1, last + 1):
        path = run_round(config, out, round_no, until if round_no == last else "refit")
    return path


# Timing, to size label_share before a run.


def time_decisions(config: RunConfig, decisions: int, seed: int) -> dict[str, float]:
    """Mean seconds per card decision with a choice of the playing Hybrid and of the teacher, on
    the positions of one game the Hybrid plays, and the share of card decisions with a choice."""
    task = PartTask(0, 0, 1, config.expert.start_from, config)
    players, teachers = hybrids(task, seed)
    played = RewardedBola(seed, config.train.rules, config.train.reward)
    bola = played.bola
    play_seconds: list[float] = []
    teach_seconds: list[float] = []
    cards = choices = 0
    while not bola.is_over and len(teach_seconds) < decisions:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        choosing = obs.phase is Phase.PLAY and len(legal) > 1
        if obs.phase is Phase.PLAY:
            cards += 1
            choices += choosing
        tick = time.perf_counter()
        action = players[seat].act(obs, legal)
        if choosing:
            play_seconds.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            card_ids = [a - PLAY_BASE for a in legal]
            teach(teachers[seat], obs, card_ids)
            teach_seconds.append(time.perf_counter() - tick)
        played.step(action)
    return {
        "play": float(np.mean(play_seconds)),
        "teacher": float(np.mean(teach_seconds)),
        "choice_share": choices / max(1, cards),
        "decisions": float(len(teach_seconds)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.rl.expert", description="Expert iteration for the Hybrid."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="run the rounds of a config")
    run_cmd.add_argument("--config", type=Path, required=True, help="TOML expert config")
    run_cmd.add_argument("--out", type=Path, required=True, help="the run's folder")
    run_cmd.add_argument("--rounds", type=int, help="stop after this many rounds")
    run_cmd.add_argument(
        "--until",
        choices=STAGES,
        default="refit",
        help="stop the last round after this stage (default: all three)",
    )
    time_cmd = commands.add_parser("time", help="time the playing Hybrid and the teacher")
    time_cmd.add_argument("--config", type=Path, required=True, help="TOML expert config")
    time_cmd.add_argument("--decisions", type=int, default=60, help="card decisions to time")
    time_cmd.add_argument("--seed", type=int, default=0, help="the game's seed")
    watch_cmd = commands.add_parser("watch", help="watch a run live")
    watch_cmd.add_argument("out", type=Path, help="the run's folder")
    watch_cmd.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)
    if args.command == "watch":
        from .expert_watch import watch

        return watch(args.out, once=args.once)
    try:
        config = load_run_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"--config {args.config}: {error}")
    if args.command == "time":
        timing = time_decisions(config, args.decisions, args.seed)
        expert = config.expert
        per_game = 18 * 52 * timing["choice_share"]  # card decisions with a choice per game
        seconds = per_game * (timing["play"] + expert.label_share * timing["teacher"])
        hours = expert.games * seconds / config.train.workers / 3600
        print(
            f"playing Hybrid {1000 * timing['play']:.0f} ms and teacher "
            f"{1000 * timing['teacher']:.0f} ms per card decision ({timing['decisions']:.0f} "
            f"timed); about {seconds:.0f} s of card play per game at label_share "
            f"{expert.label_share}: {expert.games} games in about {hours:.1f} h on "
            f"{config.train.workers} workers"
        )
        return 0
    print(run(config, args.out, args.rounds, args.until))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
