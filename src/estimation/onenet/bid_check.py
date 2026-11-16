"""Step 6's real-play check (docs/rl/hybrid-iteration-plan.md): what a different bid is worth in
rounds played for real.

    uv run python -m estimation.onenet.bid_check changes --lessons runs/onenet/bids-1/lessons \\
        --out runs/onenet/bids-1/check --player trained=runs/onenet/bids-1/learn/player.npz
    uv run python -m estimation.onenet.bid_check run --out runs/onenet/bids-1/check --workers 5
    uv run python -m estimation.onenet.bid_check report --out runs/onenet/bids-1/check

The bid teacher values a bid in rounds it imagines: guessed hands, the other seats' bids drawn
from the player's chances, every card the player's top card with no search. This check asks the
games themselves. At a recorded bidding decision it plays the rest of the round twice on the deal
as it was really dealt, the table player in every seat with its real search: once with the bid
that was made and once with another bid. Both plays use the same seeds, so they differ only in
what the bid changes. The other bid's real lead is the difference in what the round is worth to
the deciding seat, counted as the teacher and the search count it: own round score less the
others', players ahead counting double (``worth``).

- ``changes``: which other bids to check, on the lessons' held-out Bolas (``bid_learn.split``:
  the Bolas the learning never trained on). Each source names a bid at every lesson: the teacher
  its careful pick (``bid_learn.careful_pick``), a player its top bid (``--player name=file``),
  a picks file whatever it holds (``--picks name=file``, with ``game``, ``step`` and ``action``
  arrays). A source's changes are the lessons where its bid is not the bid that was made.
- ``run``: plays them, one Bola's decisions a task, on worker processes; a stopped run resumes,
  and ``--bolas`` plays only so many Bolas (drawn by the seed) for a shorter run.
- ``report``: each source's changes in real play, next to the teacher's own measure of the same
  changes: its first-stage lead, which is fair to every option it tried.
"""

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from estimation.bots import Bot, make_bot
from estimation.bots.scoring import standings_weights
from estimation.engine import Bola
from estimation.engine.config import DEFAULT_CONFIG
from estimation.rl.config import CHECK_STAGE, derive_seed
from estimation.rl.onenet_record import RecordConfig, load_part, part_path
from estimation.rl.train import worker_pool

from .lessons import Lessons, load_all

Arrays = dict[str, npt.NDArray[Any]]
Scores = tuple[int, int, int, int]
PHASES = ("Dash Calls", "the auction", "estimates")
TEACHER = "teacher"
ACTIONS = 120
CHANGES = "changes.npz"
CONFIG = "config.json"
SUMMARY = "summary.json"


def _say(text: str) -> None:
    print(f"[bid-check] {text}", file=sys.stderr, flush=True)


# The changes to check.


@dataclass(frozen=True, slots=True)
class Source:
    """A bidder to check: the bid it makes at each lesson (an action id)."""

    name: str
    picks: npt.NDArray[np.int64]


def picks_from(lessons: Lessons, picks: Arrays) -> npt.NDArray[np.int64]:
    """A picks file's bids at the lessons: its ``action`` where it names the lesson's decision
    (``game`` and ``step``), the bid that was made everywhere else."""
    out = lessons["options"][:, 0].astype(np.int64)
    key = picks["game"].astype(np.int64) * (1 << 32) + picks["step"].astype(np.int64)
    if not len(key):
        return out
    order = np.argsort(key)
    wanted = lessons["game"].astype(np.int64) * (1 << 32) + lessons["step"].astype(np.int64)
    at = np.clip(np.searchsorted(key[order], wanted), 0, len(key) - 1)
    found = key[order][at] == wanted
    out[found] = picks["action"].astype(np.int64)[order][at][found]
    return out


def find_changes(lessons: Lessons, sources: Sequence[Source]) -> Arrays:
    """One row for every lesson and other bid that some source makes there: the decision
    (``game``, ``step``, ``seat``, ``phase``), the bid made (``own``) and the ``other`` one,
    which sources make it (``source``, a column each), and the teacher's measures of it where it
    tried it (``tried``): its first-stage lead (``opening``) and, where the bid was the finalist
    that sat its whole exam (``examined``), its exam lead (``exam``)."""
    from .bid_learn import examined  # the learner imports JAX: only this command needs it

    own = lessons["options"][:, 0].astype(np.int64)
    keys = [
        np.flatnonzero(source.picks != own) * ACTIONS + source.picks[source.picks != own]
        for source in sources
    ]
    pair = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    at, other = pair // ACTIONS, pair % ACTIONS
    slots = lessons["options"][at] == other[:, None]
    tried = slots.any(1)
    slot = slots.argmax(1)
    whole = tried & (slot == lessons["finalist"][at]) & examined(lessons)[at]
    flags = [np.isin(pair, key) for key in keys]
    return {
        "game": lessons["game"][at].astype(np.int32),
        "step": lessons["step"][at].astype(np.int32),
        "seat": lessons["seat"][at].astype(np.int8),
        "phase": lessons["phase"][at].astype(np.int8),
        "own": own[at].astype(np.int16),
        "other": other.astype(np.int16),
        "source": np.stack(flags, 1) if flags else np.zeros((0, 0), bool),
        "tried": tried,
        "opening": np.where(tried, lessons["opening"][at, slot], np.nan).astype(np.float32),
        "examined": whole,
        "exam": np.where(whole, lessons["exam"][at], np.nan).astype(np.float32),
    }


def _load(path: Path) -> Arrays:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def decisions(changes: Arrays) -> int:
    """How many decisions the changes are at."""
    return len(np.unique(changes["game"].astype(np.int64) * (1 << 32) + changes["step"]))


# Playing a round for real.


def play_round(
    seed: int, actions: Sequence[int], step: int, action: int, bots: Sequence[Bot]
) -> tuple[Scores, Scores]:
    """The Bola dealt from ``seed``, replayed through its first ``step`` recorded ``actions``,
    then ``action``, then the rest of that round by ``bots`` (one a seat). Returns the standings
    the round began at and the round's scores."""
    bola = Bola(seed, DEFAULT_CONFIG)
    for played in actions[:step]:
        bola.step(played)
    if action not in bola.legal_actions():
        raise ValueError(f"move {action} is not legal at step {step} of the Bola dealt from {seed}")
    index = len(bola.history)
    standings = bola.scores
    bola.step(action)
    while len(bola.history) == index:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a round that is not over always has a player to act")
        bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
    return standings, bola.history[index].scores


@dataclass(frozen=True, slots=True)
class BolaTask:
    """One Bola's changes to play: its recording part (``record`` folder, the part's ``first``
    Bola), the table ``player`` (a bot spec), the run's ``seed``, and each change's ``steps``,
    ``seats``, bids made (``owns``) and ``others``."""

    record: str
    first: int
    game: int
    player: str
    seed: int
    steps: tuple[int, ...]
    seats: tuple[int, ...]
    owns: tuple[int, ...]
    others: tuple[int, ...]


def check_bola(task: BolaTask) -> Arrays:
    """Play every change of ``task``: at each decision the round with the bid made, and with
    each other bid, by four table players seeded from the decision, the same four seeds in each
    play. One row a play: its ``step`` and ``action``, the ``standings`` the round began at and
    the round's ``result``."""
    part = load_part(part_path(Path(task.record), task.first))
    found = np.flatnonzero(part["game"] == task.game)
    if not len(found):
        raise ValueError(f"Bola {task.game} is not in the part starting at Bola {task.first}")
    row = int(found[0])
    start, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
    actions: list[int] = part["actions"][start:end].tolist()
    seed = int(part["seed"][row])
    plays: dict[int, list[int]] = {}
    for step, seat, own, other in zip(task.steps, task.seats, task.owns, task.others, strict=True):
        if step >= len(actions) or actions[step] != own:
            raise ValueError(f"Bola {task.game} did not play move {own} at step {step}")
        replay = Bola(seed, DEFAULT_CONFIG)
        for played in actions[:step]:
            replay.step(played)
        if replay.current_player != seat:
            raise ValueError(f"Bola {task.game}: seat {seat} does not decide at step {step}")
        tries = plays.setdefault(step, [own])
        if other not in tries:
            tries.append(other)
    steps: list[int] = []
    tried: list[int] = []
    standings: list[Scores] = []
    results: list[Scores] = []
    for step, tries in sorted(plays.items()):
        for action in tries:
            bots = [
                make_bot(task.player, derive_seed(task.seed, CHECK_STAGE, task.game, step, seat))
                for seat in range(4)
            ]
            before, result = play_round(seed, actions, step, action, bots)
            steps.append(step)
            tried.append(action)
            standings.append(before)
            results.append(result)
    return {
        "step": np.array(steps, np.int32),
        "action": np.array(tried, np.int16),
        "standings": np.array(standings, np.int64).reshape(-1, 4),
        "result": np.array(results, np.int64).reshape(-1, 4),
    }


def check_path(out: Path, game: int) -> Path:
    return out / "parts" / f"check-{game:05d}.npz"


def save_check(path: Path, arrays: Arrays) -> None:
    """Write a Bola's plays so that a reader never sees them half written."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    with tmp.open("wb") as file:
        np.savez_compressed(file, allow_pickle=False, **arrays)
    tmp.replace(path)


def tasks_of(changes: Arrays, config: dict[str, Any], seed: int) -> list[BolaTask]:
    """A task for every Bola with changes, in a random order drawn by ``seed``."""
    record = Path(config["record"])
    recording = RecordConfig(**json.loads((record / CONFIG).read_text()))
    games = np.unique(changes["game"])
    tasks: list[BolaTask] = []
    for game in np.random.default_rng(seed).permutation(games).tolist():
        rows = np.flatnonzero(changes["game"] == game)
        tasks.append(
            BolaTask(
                str(record),
                game - game % recording.part_games,
                game,
                recording.player,
                seed,
                tuple(changes["step"][rows].tolist()),
                tuple(changes["seat"][rows].tolist()),
                tuple(changes["own"][rows].tolist()),
                tuple(changes["other"][rows].tolist()),
            )
        )
    return tasks


def run(out: Path, workers: int = 5, bolas: int = 0, seed: int = 0) -> dict[str, Any]:
    """Play the changes kept in ``out`` (``changes``): every Bola's, or the first ``bolas`` of
    the seed's random order, skipping the Bolas already played. Returns the report."""
    changes = _load(out / CHANGES)
    config = json.loads((out / CONFIG).read_text())
    tasks = tasks_of(changes, config, seed)
    if bolas:
        tasks = tasks[:bolas]
    (out / "parts").mkdir(exist_ok=True)
    missing = [task for task in tasks if not check_path(out, task.game).exists()]
    if missing:
        plays = sum(len(set(task.steps)) + len(task.steps) for task in missing)
        _say(f"{len(missing)} of {len(tasks)} Bolas to play on {workers} workers: {plays:,} rounds")
        started = time.perf_counter()
        done = 0
        with worker_pool(workers) as pool:
            futures = {pool.submit(check_bola, task): task for task in missing}
            for future in as_completed(futures):
                task = futures[future]
                save_check(check_path(out, task.game), future.result())
                done += len(set(task.steps)) + len(task.steps)
                elapsed = time.perf_counter() - started
                _say(
                    f"{done:,}/{plays:,} rounds, {elapsed / 60:.1f} min, "
                    f"about {elapsed / done * (plays - done) / 60:.1f} min left"
                )
    return report(out)


# The report.


def worth(result: Sequence[int], standings: Sequence[int], seat: int) -> float:
    """What a round's scores are worth to ``seat``, whose Bola stood at ``standings``: its own
    round score less the others', players ahead of it counting double (``standings_weights``)."""
    weights = standings_weights(_four(standings), seat)
    return float(result[seat]) - sum(w * float(r) for w, r in zip(weights, result, strict=True))


def _four(values: Sequence[int]) -> Scores:
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def played(out: Path) -> dict[tuple[int, int, int], tuple[Scores, Scores]]:
    """Every play saved in ``out``: by Bola, step and bid, the standings and the round's scores."""
    found: dict[tuple[int, int, int], tuple[Scores, Scores]] = {}
    for path in sorted((out / "parts").glob("check-*.npz")):
        if ".tmp" in path.name:
            continue
        game = int(path.stem.split("-")[1])
        with np.load(path) as data:
            rows = zip(
                data["step"].tolist(),
                data["action"].tolist(),
                data["standings"].tolist(),
                data["result"].tolist(),
                strict=True,
            )
            for step, action, standings, result in rows:
                found[(game, step, action)] = (_four(standings), _four(result))
    return found


def real_leads(
    changes: Arrays, plays: Mapping[tuple[int, int, int], tuple[Scores, Scores]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Each change's real lead: the round's worth to the deciding seat with the other bid less
    its worth with the bid made (nan where the change was not played), and the same lead in the
    seat's own round score alone."""
    count = len(changes["game"])
    lead = np.full(count, np.nan)
    plain = np.full(count, np.nan)
    rows = zip(
        changes["game"].tolist(),
        changes["step"].tolist(),
        changes["seat"].tolist(),
        changes["own"].tolist(),
        changes["other"].tolist(),
        strict=True,
    )
    for k, (game, step, seat, own, other) in enumerate(rows):
        made, tried = plays.get((game, step, own)), plays.get((game, step, other))
        if made is None or tried is None:
            continue
        lead[k] = worth(tried[1], tried[0], seat) - worth(made[1], made[0], seat)
        plain[k] = tried[1][seat] - made[1][seat]
    return lead, plain


def _mean(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """A mean and its standard error (0 with fewer than two values)."""
    if not len(values):
        return 0.0, 0.0
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), error


def summarize(
    changes: Arrays, lead: npt.NDArray[np.float64], plain: npt.NDArray[np.float64], config: Any
) -> dict[str, Any]:
    """Each source's changes that were played: how many, their real lead and the teacher's
    first-stage lead of the same changes (where it tried the bid), in all and by kind of
    decision, and both in points a game: each kind's lead times how often the source changes a
    decision of that kind (over the lessons the changes were found in) times that kind's
    decisions a seat has in a Bola."""
    done = ~np.isnan(lead)
    seat_game = config.get("per_seat_game")
    summary: dict[str, Any] = {"played": int(done.sum()), "changes": len(lead), "sources": {}}
    for k, name in enumerate(config["sources"]):
        of = changes["source"][:, k]
        rows = of & done
        both = rows & changes["tried"]
        real, real_error = _mean(lead[rows])
        entry: dict[str, Any] = {
            "changes": int(of.sum()),
            "played": int(rows.sum()),
            "real": real,
            "real_error": real_error,
            "own_score": _mean(plain[rows])[0],
            "imagined": _mean(changes["opening"][both].astype(np.float64))[0],
            "real_where_imagined": _mean(lead[both])[0],
            "phases": {},
        }
        game = imagined_game = variance = 0.0
        for phase, phase_name in enumerate(PHASES):
            at = changes["phase"] == phase
            mean, error = _mean(lead[rows & at])
            imagined = _mean(changes["opening"][both & at].astype(np.float64))[0]
            lessons = config["lessons_by_phase"][phase]
            rate = float((of & at).sum()) / lessons if lessons else 0.0
            entry["phases"][phase_name] = {
                "played": int((rows & at).sum()),
                "rate": rate,
                "real": mean,
                "real_error": error,
                "imagined": imagined,
            }
            if seat_game:
                game += rate * seat_game[phase] * mean
                imagined_game += rate * seat_game[phase] * imagined
                variance += (rate * seat_game[phase] * error) ** 2
        if seat_game:
            entry["game"] = game
            entry["game_error"] = float(np.sqrt(variance))
            entry["imagined_game"] = imagined_game
        summary["sources"][name] = entry
    return summary


def lines(summary: dict[str, Any]) -> list[str]:
    out = [f"{summary['played']:,} of {summary['changes']:,} changes played for real"]
    for name, entry in summary["sources"].items():
        line = (
            f"{name}: {entry['played']:,} changes played; a change is worth "
            f"{entry['real']:+.2f} points for real (give or take {entry['real_error']:.2f}), "
            f"{entry['imagined']:+.2f} in the teacher's rounds"
        )
        if "game" in entry:
            line += (
                f"; a game {entry['game']:+.1f} for real (give or take "
                f"{entry['game_error']:.1f}), {entry['imagined_game']:+.1f} in the teacher's"
            )
        out.append(line)
        for phase, row in entry["phases"].items():
            if row["played"]:
                out.append(
                    f"  {phase}: {row['played']:,} played, changed at {row['rate']:.1%} of "
                    f"decisions; {row['real']:+.2f} for real (give or take "
                    f"{row['real_error']:.2f}), {row['imagined']:+.2f} in the teacher's rounds"
                )
    return out


def report(out: Path) -> dict[str, Any]:
    """The summary of the plays saved in ``out`` so far, written next to them."""
    changes = _load(out / CHANGES)
    config = json.loads((out / CONFIG).read_text())
    lead, plain = real_leads(changes, played(out))
    summary = summarize(changes, lead, plain, config)
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# The commands.


def keep_changes(
    out: Path,
    lessons_folder: Path,
    players: Sequence[tuple[str, Path]],
    picks: Sequence[tuple[str, Path]],
    held_out: float,
    split_seed: int,
    teacher: bool = True,
) -> Arrays:
    """Find the changes of the teacher, the ``players`` and the ``picks`` files on the lessons'
    held-out Bolas (every lesson with ``held_out`` 0) and keep them in ``out``."""
    from . import bid_learn as learn  # the learner imports JAX: only this command needs it
    from .nets import load_player

    lessons = load_all(lessons_folder)
    seat_game = learn.per_seat_game(lessons, lessons_folder)
    if held_out:
        lessons = learn.split(lessons, held_out, split_seed)[1]
    sources: list[Source] = []
    if teacher:
        sources.append(Source(TEACHER, learn.careful_pick(lessons)))
    for name, path in players:
        sources.append(Source(name, learn.top_bid(load_player(path)[0], lessons["x"])))
    for name, path in picks:
        with np.load(path) as data:
            sources.append(Source(name, picks_from(lessons, {k: data[k] for k in data.files})))
    names = [source.name for source in sources]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"the sources need one or more different names, not {names}")
    changes = find_changes(lessons, sources)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / CHANGES, allow_pickle=False, **changes)
    config = {
        "lessons": str(lessons_folder),
        "record": json.loads((lessons_folder / CONFIG).read_text())["record"],
        "held_out": held_out,
        "split_seed": split_seed,
        "sources": names,
        "lessons_by_phase": [int((lessons["phase"] == phase).sum()) for phase in range(3)],
        "per_seat_game": seat_game,
    }
    (out / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return changes


def _named(text: str) -> tuple[str, Path]:
    name, _, path = text.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=file, not {text!r}")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.bid_check",
        description="What other bids are worth in rounds played for real.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("changes", help="find the changes to check on held-out lessons")
    find.add_argument("--lessons", type=Path, required=True)
    find.add_argument("--out", type=Path, required=True)
    find.add_argument("--player", type=_named, action="append", default=[], metavar="NAME=FILE")
    find.add_argument("--picks", type=_named, action="append", default=[], metavar="NAME=FILE")
    find.add_argument("--no-teacher", action="store_true", help="leave the teacher's picks out")
    find.add_argument("--held-out", type=float, default=0.05)
    find.add_argument("--split-seed", type=int, default=0)
    play = commands.add_parser("run", help="play the changes for real")
    play.add_argument("--out", type=Path, required=True)
    play.add_argument("--workers", type=int, default=5)
    play.add_argument("--bolas", type=int, default=0, help="play only this many Bolas")
    play.add_argument("--seed", type=int, default=0)
    show = commands.add_parser("report", help="the plays saved so far")
    show.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "changes":
        changes = keep_changes(
            args.out,
            args.lessons,
            args.player,
            args.picks,
            args.held_out,
            args.split_seed,
            not args.no_teacher,
        )
        count = decisions(changes)
        print(
            f"{len(changes['game']):,} changes at {count:,} decisions of "
            f"{len(np.unique(changes['game'])):,} Bolas: {count + len(changes['game']):,} rounds "
            "to play"
        )
        return 0
    summary = run(args.out, args.workers, args.bolas, args.seed) if args.command == "run" else None
    for line in lines(summary if summary is not None else report(args.out)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
