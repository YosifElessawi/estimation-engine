"""Train the M4 bot from a config file and a seed (SPEC §6 M4).

    uv run python -m estimation.rl.train run --config configs/m4.toml --out runs/m4

Three stages, each skipped when its output already exists, so an interrupted run resumes:

1. **record:** whole Bolas of four PIMC bots, every card decision recorded
   (``<out>/warmstart/pimc.npz``).
2. **imitate:** CardPlayNet imitates PIMC's card play (``<out>/checkpoints/warm-start.pt``,
   played with ``heuristic_bidding=true``).
3. **selfplay:** Deep Monte Carlo self-play from the warm start, BiddingNet from scratch, with a
   checkpoint every ``checkpoint_every`` iterations and ``final.pt`` at the end.
   ``<out>/selfplay/state.pt`` resumes it exactly.

The same config, seed and code give the same checkpoints, on any number of workers.
``<out>/checkpoints/index.json`` lists the checkpoints; ``record-elo`` adds each one's Elo from a
ladder run's ``ladder.json``.
"""

import argparse
import json
import multiprocessing
import shutil
import sys
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from .config import (
    SELFPLAY_STAGE,
    TrainConfig,
    config_to_dict,
    derive_seed,
    linear_schedule,
    load_config,
)
from .imitation import ImitationData, train_imitation, training_rows
from .learner import Learner, UpdateReport
from .memory import gpu_mb, peak_rss_mb
from .networks import BiddingNet, CardPlayNet, load_policy, save_checkpoint
from .record import CardRecords, concat_records, load_records, record_game, save_records
from .selfplay import (
    LEARNER,
    SNAPSHOT,
    ChunkResult,
    ChunkTask,
    GameStats,
    NetWeights,
    SeedFn,
    concat_samples,
    plan_eval_games,
    plan_games,
    play_chunk,
)
from .torch_typing import seed_torch

WARM_START = "warm-start"
FINAL = "final"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _init_worker() -> None:
    torch.set_num_threads(1)  # many small batches per worker: one thread each, no contention


@contextmanager
def worker_pool(workers: int) -> Generator[ProcessPoolExecutor]:
    """Worker processes for games. Actors always run in workers, so their results never depend
    on the main process's thread settings."""
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker)
    try:
        yield pool
    finally:
        pool.shutdown(cancel_futures=True)


def _say(stage: str, text: str) -> None:
    print(f"[{stage}] {text}", file=sys.stderr, flush=True)


class Run:
    """One training run's folder."""

    def __init__(self, out: Path, config: TrainConfig) -> None:
        self.out = out
        self.config = config
        self.checkpoints = out / "checkpoints"
        self.index_path = self.checkpoints / "index.json"

    def check_config(self) -> None:
        """Write the config the first time; refuse to resume a folder trained differently."""
        path = self.out / "config.json"
        data = config_to_dict(self.config)
        text = json.dumps(data, indent=1, sort_keys=True)
        if path.exists():
            if json.loads(path.read_text()) != json.loads(text):
                raise SystemExit(f"{path} holds a different config; use a new --out folder")
            return
        self.out.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    def index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return cast(list[dict[str, Any]], json.loads(self.index_path.read_text())["checkpoints"])

    def add_to_index(self, entry: dict[str, Any]) -> None:
        entries = [e for e in self.index() if e["name"] != entry["name"]]
        entries.append(entry)
        self.write_index(entries)

    def write_index(self, entries: list[dict[str, Any]]) -> None:
        """Save the index in training order: the warm start, each iteration, then final."""
        entries = sorted(
            entries, key=lambda e: (e["name"] == FINAL, int(e.get("iteration", 0)), e["name"])
        )
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps({"checkpoints": entries}, indent=1) + "\n")

    def save(
        self, name: str, card: CardPlayNet, bidding: BiddingNet | None, meta: dict[str, Any]
    ) -> Path:
        path = self.checkpoints / f"{name}.pt"
        full = {"name": name, "seed": self.config.seed, **meta}
        save_checkpoint(path, card, bidding, full)
        self.add_to_index({"name": name, "file": path.name, "elo": {}, **meta})
        return path


def record_stage(run: Run, pool: ProcessPoolExecutor) -> CardRecords:
    path = run.out / "warmstart" / "pimc.npz"
    if path.exists():
        _say("record", f"reusing {path}")
        return load_records(str(path))
    games = run.config.record.games
    start = time.perf_counter()
    parts: list[CardRecords] = []
    task = partial(record_game, config=run.config)
    for done, part in enumerate(pool.map(task, range(games)), 1):
        parts.append(part)
        if done % 20 == 0 or done == games:
            elapsed = time.perf_counter() - start
            left = elapsed / done * (games - done)
            _say(
                "record",
                f"{done}/{games} PIMC games, {sum(map(len, parts)):,} card decisions, "
                f"{elapsed / 60:.1f} min elapsed, about {left / 60:.1f} min left",
            )
    records = concat_records(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_records(str(path), records)
    seconds = time.perf_counter() - start
    (run.out / "warmstart" / "record.json").write_text(
        json.dumps({"games": games, "rows": len(records), "seconds": seconds}, indent=1) + "\n"
    )
    return records


def imitation_stage(run: Run, records: CardRecords, device: torch.device) -> Path:
    path = run.checkpoints / f"{WARM_START}.pt"
    if path.exists():
        _say("imitate", f"reusing {path}")
        return path
    _say("imitate", f"training CardPlayNet on {len(records):,} PIMC card decisions ({device})")
    net, report = train_imitation(records, run.config, device)
    _say(
        "imitate",
        f"held-out agreement with PIMC: {100 * report.choice_agreement:.1f}% exact, "
        f"{100 * report.best_agreement:.1f}% a card PIMC rates best; "
        f"critic error {report.value_error:.1f} points; {report.seconds:.0f} s",
    )
    record_info = run.out / "warmstart" / "record.json"
    record_seconds = json.loads(record_info.read_text())["seconds"] if record_info.exists() else 0
    meta = {
        "stage": "imitate",
        "iteration": 0,
        "games": run.config.record.games,
        "seconds": record_seconds + report.seconds,
        "imitation": {
            "train_rows": report.train_rows,
            "holdout_rows": report.holdout_rows,
            "choice_agreement": report.choice_agreement,
            "best_agreement": report.best_agreement,
            "value_error": report.value_error,
            "loss": report.loss,
        },
    }
    return run.save(WARM_START, net, None, meta)


def _summary(stats: Iterable[GameStats]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[GameStats]] = {}
    for game in stats:
        groups.setdefault(game.kind, []).append(game)
    return {
        kind: {
            "games": len(games),
            "rsa": sum(g.rsa for g in games) / len(games),
            "first": sum(g.first for g in games) / len(games),
            "skipped": sum(g.skipped for g in games) / max(1, sum(g.rounds for g in games)),
        }
        for kind, games in sorted(groups.items())
    }


def _weights_to_payload(weights: NetWeights) -> dict[str, Any]:
    return {"hidden": weights.hidden, "card": weights.card, "bidding": weights.bidding}


def _weights_from_payload(data: dict[str, Any]) -> NetWeights:
    return NetWeights(hidden=int(data["hidden"]), card=data["card"], bidding=data["bidding"])


def selfplay_stage(
    run: Run,
    warm_start: Path,
    records: CardRecords,
    pool: ProcessPoolExecutor,
    device: torch.device,
    on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
) -> Path:
    config = run.config
    cfg = config.selfplay
    final = run.checkpoints / f"{FINAL}.pt"
    if final.exists():
        _say("selfplay", f"reusing {final}")
        return final
    card = load_policy(warm_start).card
    seed_torch(derive_seed(config.seed, SELFPLAY_STAGE, 0, 0))
    bidding = BiddingNet(config.nets.hidden)
    anchor = None
    if cfg.card_anchor_weight > 0:
        anchor = ImitationData(records, training_rows(records, config), device)
    learner = Learner(card, bidding, config, device, anchor)
    state_path = run.out / "selfplay" / "state.pt"
    snapshots: dict[int, NetWeights] = {}
    start_iteration = 0
    totals = {"games": 0, "seconds": 0.0, "bid_samples": 0, "card_samples": 0}
    warm_meta = load_policy(warm_start).meta
    base_seconds = float(warm_meta.get("seconds", 0.0))
    if state_path.exists():
        state = cast(dict[str, Any], torch.load(state_path, weights_only=True))
        learner.load_state(state["learner"])
        snapshots = {int(k): _weights_from_payload(v) for k, v in state["snapshots"].items()}
        start_iteration = int(state["iteration"])
        totals = state["totals"]
        _say("selfplay", f"resuming after iteration {start_iteration}")
    log_path = run.out / "selfplay" / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path = final
    for iteration in range(start_iteration, cfg.iterations):
        tick = time.perf_counter()
        epsilon = linear_schedule(
            cfg.iterations, iteration, cfg.epsilon_bid_start, cfg.epsilon_bid_end
        )
        temperature = linear_schedule(
            cfg.iterations, iteration, cfg.bid_temperature_start, cfg.bid_temperature_end
        )
        train_card = iteration >= cfg.card_frozen_iterations
        seed_of = SeedFn(config.seed, SELFPLAY_STAGE, iteration + 1)
        plans = plan_games(config, iteration, sorted(snapshots), seed_of)
        weights = learner.weights()
        tasks: list[ChunkTask] = []
        for begin in range(0, len(plans), cfg.chunk):
            chunk = tuple(plans[begin : begin + cfg.chunk])
            used = {s.snapshot for p in chunk for s in p.seats if s.kind == SNAPSHOT}
            tasks.append(
                ChunkTask(
                    plans=chunk,
                    learner=weights,
                    snapshots={key: snapshots[key] for key in sorted(used)},
                    epsilon_bid=epsilon,
                    epsilon_card=cfg.epsilon_card,
                    bid_temperature=temperature,
                    record_bid=True,
                    record_card=train_card,
                    rules=config.rules,
                    reward=config.reward,
                )
            )
        evals = plan_eval_games(config, seed_of)
        for begin in range(0, len(evals), cfg.chunk):
            tasks.append(
                ChunkTask(
                    plans=tuple(evals[begin : begin + cfg.chunk]),
                    learner=weights,
                    snapshots={},
                    epsilon_bid=0.0,
                    epsilon_card=0.0,
                    bid_temperature=0.0,
                    record_bid=False,
                    record_card=False,
                    rules=config.rules,
                    reward=config.reward,
                )
            )
        results: list[ChunkResult] = list(pool.map(play_chunk, tasks))
        play_seconds = time.perf_counter() - tick
        bid = concat_samples([r.bid for r in results])
        card_samples = concat_samples([r.card for r in results])
        update: UpdateReport = learner.update(bid, card_samples, iteration, train_card)
        stats = [game for r in results for game in r.games]
        seconds = time.perf_counter() - tick
        totals["games"] += len(plans) + len(evals)
        totals["seconds"] += seconds
        totals["bid_samples"] += len(bid)
        totals["card_samples"] += len(card_samples)
        summary = _summary(stats)
        entry = {
            "iteration": iteration + 1,
            "seconds": seconds,
            "play_seconds": play_seconds,
            "games": len(plans),
            "learner_seats": sum(1 for p in plans for s in p.seats if s.kind == LEARNER),
            "bid_samples": len(bid),
            "card_samples": len(card_samples),
            "epsilon_bid": epsilon,
            "bid_temperature": temperature,
            "train_card": train_card,
            "lr_factor": update.lr_factor,
            "anchor_weight": update.anchor_weight,
            "bid_loss": update.bid_loss,
            "card_value_loss": update.card_value_loss,
            "card_q_loss": update.card_q_loss,
            "anchor_loss": update.anchor_loss,
            "memory_mb": {
                "trainer_peak": peak_rss_mb(),
                "trainer_gpu": gpu_mb(device),
                "worker_peak": max(r.peak_rss_mb for r in results),
            },
            "results": summary,
        }
        with log_path.open("a") as log:
            log.write(json.dumps(entry) + "\n")
        parts = [
            f"{kind} RSA {s['rsa']:+.1f} first {100 * s['first']:.0f}% "
            f"skipped {100 * s['skipped']:.0f}%"
            for kind, s in summary.items()
        ]
        _say(
            "selfplay",
            f"iteration {iteration + 1}/{cfg.iterations}: {len(plans)} games in "
            f"{seconds:.1f} s, {len(card_samples):,} card + {len(bid):,} bid samples, "
            f"bid loss {update.bid_loss:.3f}, card V {update.card_value_loss:.3f} "
            f"Q {update.card_q_loss:.3f} anchor {update.anchor_loss:.3f}; " + "; ".join(parts),
        )
        if on_iteration is not None:
            on_iteration(iteration + 1, entry)
        done = iteration + 1
        if done % cfg.snapshot_every == 0:
            snapshots[done] = learner.weights()
            for key in sorted(snapshots)[: -cfg.snapshots_kept]:
                del snapshots[key]
        if done % cfg.checkpoint_every == 0 or done == cfg.iterations:
            meta = {
                "stage": "selfplay",
                "iteration": done,
                "games": totals["games"],
                "seconds": base_seconds + totals["seconds"],
                "bid_samples": totals["bid_samples"],
                "card_samples": totals["card_samples"],
            }
            path = run.save(f"iter-{done:04d}", learner.card, learner.bidding, meta)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_suffix(".tmp")
            torch.save(
                {
                    "iteration": done,
                    "learner": learner.state(),
                    "snapshots": {str(k): _weights_to_payload(v) for k, v in snapshots.items()},
                    "totals": totals,
                },
                tmp,
            )
            tmp.replace(state_path)
    shutil.copyfile(path, final)
    entry = next(e for e in run.index() if e["file"] == path.name)
    run.add_to_index({**entry, "name": FINAL, "file": final.name, "copy_of": path.name})
    return final


def train(
    config: TrainConfig,
    out: Path,
    until: str = "selfplay",
    on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
) -> Path:
    """Run the stages up to ``until`` in ``out`` and return the last checkpoint (or the
    recorded data's path after ``record``)."""
    run = Run(out, config)
    run.check_config()
    device = resolve_device(config.nets.device)
    with worker_pool(config.workers) as pool:
        records = record_stage(run, pool)
        if until == "record":
            return out / "warmstart" / "pimc.npz"
        warm_start = imitation_stage(run, records, device)
        if until == "imitate":
            return warm_start
        return selfplay_stage(run, warm_start, records, pool, device, on_iteration)


def record_elo(out: Path, ladder: Path) -> list[tuple[str, dict[str, float]]]:
    """Write each checkpoint's ladder Elo (by bot spec) into ``<out>/checkpoints/index.json``.
    A copy (``final.pt``) and the checkpoint it copies share their ratings."""
    data = json.loads(ladder.read_text())
    ratings: dict[str, float] = data["ratings"]
    run = Run(out, TrainConfig())
    entries = run.index()
    by_file: dict[str, dict[str, float]] = {}
    for entry in entries:
        target = (run.checkpoints / entry["file"]).resolve()
        found = dict(entry.get("elo", {}))
        for spec, rating in ratings.items():
            if _checkpoint_of(spec) == target:
                found[spec] = rating
        by_file[entry["file"]] = found
    for entry in entries:
        original = entry.get("copy_of")
        if original in by_file:
            merged = {**by_file[original], **by_file[entry["file"]]}
            by_file[original] = by_file[entry["file"]] = merged
    updated = [{**entry, "elo": by_file[entry["file"]]} for entry in entries]
    run.write_index(updated)
    return [(str(entry["name"]), by_file[entry["file"]]) for entry in entries]


def _checkpoint_of(spec: str) -> Path | None:
    name, *options = spec.split(":")
    if name != "rl":
        return None
    for option in options:
        key, _, value = option.partition("=")
        if key == "checkpoint":
            return Path(value).resolve()
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.rl.train", description="Train the M4 self-play bot."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="train from a config file and a seed")
    run_cmd.add_argument("--config", type=Path, required=True, help="TOML training config")
    run_cmd.add_argument("--seed", type=int, help="run seed (default: the config's seed)")
    run_cmd.add_argument("--out", type=Path, required=True, help="the run's folder")
    run_cmd.add_argument(
        "--until",
        choices=("record", "imitate", "selfplay"),
        default="selfplay",
        help="stop after this stage (default: run all three)",
    )
    elo_cmd = commands.add_parser("record-elo", help="save ladder Elo next to each checkpoint")
    elo_cmd.add_argument("--out", type=Path, required=True, help="the run's folder")
    elo_cmd.add_argument("--ladder", type=Path, required=True, help="a ladder run's ladder.json")
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            config = load_config(args.config, args.seed)
        except (OSError, ValueError) as error:
            parser.error(f"--config {args.config}: {error}")
        path = train(config, args.out, args.until)
        print(path)
        return 0
    for name, elo in record_elo(args.out, args.ladder):
        rated = ", ".join(f"{spec}: {rating:,.0f}" for spec, rating in elo.items()) or "unrated"
        print(f"{name}: {rated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
