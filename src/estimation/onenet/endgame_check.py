"""Step 7's yardstick (docs/rl/hybrid-iteration-plan.md): the cards the table player played in a
round's last tricks, against the exact answer.

    uv run python -m estimation.onenet.endgame_check run --record runs/onenet/bids-1/games \\
        --out runs/onenet/endgame-1 --bolas 10 --workers 1
    uv run python -m estimation.onenet.endgame_check play --out runs/onenet/endgame-1 --workers 5
    uv run python -m estimation.onenet.endgame_check report --out runs/onenet/endgame-1

The table player finishes an imagined deal roughly, every seat playing the player's top card.
``estimation.onenet.endgame`` finishes it exactly. This check asks what that is worth, on games
the table player recorded:

- ``run``: at every recorded card decision with a choice and at most ``--cards`` cards in the
  hand, the table player imagines ``--deals`` deals as it does at the table. Every legal card is
  valued on each of them twice: by the exact finish and by the rough one. The deals are kept as
  two halves, so a card picked on one half is judged on the other and no pick is flattered by
  the luck of the deals that picked it. One Bola is a task; a Bola that is saved is not done
  again, so a stopped run resumes.
- ``play``: the real-play check of step 6 (``bid_check``), for cards. Where the exact answer over
  all the deals is another card than the one played, the round is finished twice on the deal as
  it was really dealt, the table player in every seat and the same seeds in both plays: once
  with the card played, once with the exact answer's card. Hidden hands are what the exact
  answer cannot see, so this is what says whether it would have helped at the table.
- ``report``: by cards in the hand, how often the exact answer is another card, what the card
  played loses by the exact finish (picked on one half of the deals, judged on the other), the
  same for the player's top card and for the rough finish's own pick on the same deals with
  every card, and the real leads played so far.
"""

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from estimation.engine import Bola, Observation, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv.encode_np import PLAY_BASE
from estimation.rl.config import ENDGAME_STAGE, derive_seed
from estimation.rl.onenet_record import RecordConfig, load_part, part_path
from estimation.rl.train import worker_pool

from . import bid_check
from .endgame import MAX_CARDS, solve, worths_of

Arrays = dict[str, npt.NDArray[Any]]
Floats = npt.NDArray[np.float64]
CONFIG = "config.json"
SUMMARY = "summary.json"
PLAYS = "plays"  # the real plays' folder inside the run's, laid out as bid_check's
LEVEL = 1e-4  # points: two cards closer than this by the exact finish are the same to it
DTYPES = {
    "game": np.int32,
    "step": np.int32,
    "seat": np.int8,
    "round": np.int8,
    "left": np.int8,  # cards in the deciding seat's hand
    "place": np.int8,  # cards already on the table in its trick
    "legal": np.int8,  # [MAX_CARDS] its legal cards, lowest first, then -1
    "played": np.int8,  # the card the table player played
    "net_top": np.int8,  # the player's top card with no search
    "exact_a": np.float32,  # [MAX_CARDS] each legal card's worth by the exact finish, the
    "exact_b": np.float32,  # first and the second half of the deals
    "rough_a": np.float32,  # the same by the rough finish
    "rough_b": np.float32,
}
WIDE = ("legal", "exact_a", "exact_b", "rough_a", "rough_b")


def _say(text: str) -> None:
    print(f"[endgame-check] {text}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class CheckConfig:
    """A run of the check: the recording, its table player (a bot spec), the deals imagined at
    a decision (two halves), the most cards in a hand that is checked, and the seed."""

    record: str
    player: str
    games: int
    part_games: int
    deals: int = 64
    cards: int = MAX_CARDS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.deals < 2 or self.deals % 2:
            raise ValueError("deals must be an even number, 2 or more: they are used as two halves")
        if not 2 <= self.cards <= MAX_CARDS:
            raise ValueError(f"cards must be 2 to {MAX_CARDS}")


@dataclass(frozen=True, slots=True)
class BolaTask:
    """One Bola to check: its number, the first Bola of its recording part, and the run."""

    game: int
    first: int
    config: CheckConfig


def table_player(spec: str, seed: int) -> Any:
    """The table player of ``spec``, which must be the one net's search judging by round
    points: the exact finish counts a round as that search does."""
    from estimation.bots import make_bot  # PyTorch loads with the search, not with this module
    from estimation.onenet.search import ROUND, OneNetSearch

    bot = make_bot(spec, seed)
    if not isinstance(bot, OneNetSearch) or bot.judge != ROUND:
        raise ValueError("the endgame check needs an onenet player with judge=round")
    return bot


def judge(obs: Observation, legal: list[int], action: int, bot: Any, deals: int) -> dict[str, Any]:
    """One card decision's row (``DTYPES``, less ``game`` and ``step``): ``bot`` imagines
    ``deals`` deals from ``obs``, and every legal card is valued on each half of them by the
    exact finish and by ``bot``'s own rough one."""
    if obs.leader is None or obs.trump is None:
        raise ValueError("cards are judged during card play")
    cards = [move - PLAY_BASE for move in legal]
    half = deals // 2
    drawn = bot.sample_deals(obs, deals)
    worths = worths_of(obs, bot.config.rules, bot.config.leader_weight)
    solved = solve(drawn, obs.trick, obs.leader, obs.tricks_won, obs.trump, worths)
    if solved.cards != cards:
        raise RuntimeError("the solver and the engine disagree on the legal cards")
    exact = solved.worths[:, :, obs.seat]
    rough = [np.array(bot.evaluate(obs, cards, drawn[at : at + half])) / half for at in (0, half)]
    net = bot.card_advantages(obs, cards)

    def wide(values: Any, fill: float) -> list[float]:
        return [*np.asarray(values).tolist(), *([fill] * (MAX_CARDS - len(cards)))]

    return {
        "seat": obs.seat,
        "round": obs.round_no,
        "left": obs.hand.bit_count(),
        "place": len(obs.trick),
        "legal": wide(cards, -1),
        "played": action - PLAY_BASE,
        "net_top": cards[int(np.argmax(net))],
        "exact_a": wide(exact[:half].mean(0), np.nan),
        "exact_b": wide(exact[half:].mean(0), np.nan),
        "rough_a": wide(rough[0], np.nan),
        "rough_b": wide(rough[1], np.nan),
    }


def stack(rows: list[dict[str, Any]]) -> Arrays:
    """Rows as arrays of ``DTYPES``."""
    return {
        name: np.array([row[name] for row in rows], dtype).reshape(
            (len(rows), MAX_CARDS) if name in WIDE else (len(rows),)
        )
        for name, dtype in DTYPES.items()
    }


def check_bola(task: BolaTask) -> Arrays:
    """Every card decision of the Bola with a choice and at most ``cards`` cards in the hand,
    judged (``judge``) by a table player seeded from the decision."""
    config = task.config
    part = load_part(part_path(Path(config.record), task.first))
    found = np.flatnonzero(part["game"] == task.game)
    if not len(found):
        raise ValueError(f"Bola {task.game} is not in the part starting at Bola {task.first}")
    row = int(found[0])
    start, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
    actions: list[int] = part["actions"][start:end].tolist()
    bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
    rows: list[dict[str, Any]] = []
    for step, action in enumerate(actions):
        seat = bola.current_player
        if seat is not None and bola.phase is Phase.PLAY:
            legal = bola.legal_actions()
            obs = bola.observation(seat)
            if len(legal) > 1 and obs.hand.bit_count() <= config.cards:
                seed = derive_seed(config.seed, ENDGAME_STAGE, task.game, step, seat)
                bot = table_player(config.player, seed)
                rows.append(
                    {
                        "game": task.game,
                        "step": step,
                        **judge(obs, legal, action, bot, config.deals),
                    }
                )
        bola.step(action)
    return stack(rows)


def part_file(out: Path, game: int) -> Path:
    return out / "parts" / f"endgame-{game:05d}.npz"


def load_config(out: Path) -> CheckConfig:
    return CheckConfig(**json.loads((out / CONFIG).read_text()))


def start(out: Path, record: Path, deals: int, cards: int, seed: int) -> CheckConfig:
    """The run's config, written to ``out``, or the one already there if it is the same run."""
    recording = RecordConfig(**json.loads((record / CONFIG).read_text()))
    config = CheckConfig(
        str(record), recording.player, recording.games, recording.part_games, deals, cards, seed
    )
    path = out / CONFIG
    if path.exists():
        if load_config(out) != config:
            raise ValueError(f"{out} holds another run of the check: {path.read_text().strip()}")
    else:
        (out / "parts").mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
    return config


def run(out: Path, bolas: int, workers: int = 1) -> dict[str, Any]:
    """Check the first ``bolas`` Bolas of the run's order (a shuffle of the recording by the
    run's seed) that are not saved yet, on ``workers`` processes. Returns the report."""
    config = load_config(out)
    order = np.random.default_rng(config.seed).permutation(config.games).tolist()[:bolas]
    tasks = [
        BolaTask(game, game - game % config.part_games, config)
        for game in order
        if not part_file(out, game).exists()
    ]
    if tasks:
        _say(f"{len(tasks)} of {len(order)} Bolas to check on {workers} workers")
        started = time.perf_counter()
        decisions = 0
        with worker_pool(workers) as pool:
            futures = {pool.submit(check_bola, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                bid_check.save_check(part_file(out, futures[future].game), rows)
                decisions += len(rows["game"])
                elapsed = time.perf_counter() - started
                _say(
                    f"{done}/{len(tasks)} Bolas, {decisions:,} decisions, {elapsed / 60:.1f} min, "
                    f"about {elapsed / done * (len(tasks) - done) / 60:.1f} min left"
                )
    return report(out)


def load_rows(out: Path) -> Arrays:
    """Every decision saved in ``out``."""
    parts: list[Arrays] = []
    for path in sorted((out / "parts").glob("endgame-*.npz")):
        if ".tmp" in path.name:
            continue
        with np.load(path) as data:
            parts.append({name: data[name] for name in DTYPES})
    if not parts:
        return stack([])
    return {name: np.concatenate([part[name] for part in parts]) for name in DTYPES}


# What the rows say.


def _slot(rows: Arrays, cards: npt.NDArray[Any]) -> npt.NDArray[np.intp]:
    """Where each row's ``cards`` entry sits among its legal cards."""
    return (rows["legal"] == cards[:, None]).argmax(1)


def _at(values: Floats, slot: npt.NDArray[np.intp]) -> Floats:
    return np.take_along_axis(values, slot[:, None], 1)[:, 0]


def losses(rows: Arrays) -> dict[str, Floats]:
    """For every decision, by the exact finish and with the pick made on one half of the deals
    and judged on the other (both ways round, the mean of the two): what the card played
    (``played``), the player's top card (``net_top``) and the rough finish's pick with every
    card (``rough``) lose against the exact finish's pick. ``rough_played`` is what the card
    played loses by the rough finish's own pick and measure. ``lead`` is how far the exact
    finish's best card over all the deals leads the card played, and ``differs`` says whether it
    leads it at all (by more than ``LEVEL``): whether the exact answer is another card."""
    exact = [rows["exact_a"].astype(np.float64), rows["exact_b"].astype(np.float64)]
    rough = [rows["rough_a"].astype(np.float64), rows["rough_b"].astype(np.float64)]
    played, top = _slot(rows, rows["played"]), _slot(rows, rows["net_top"])
    out = {name: np.zeros(len(played)) for name in ("played", "net_top", "rough", "rough_played")}
    for pick_on, judge_on in ((0, 1), (1, 0)):
        best = _at(exact[judge_on], np.nanargmax(exact[pick_on], 1))
        out["played"] += (best - _at(exact[judge_on], played)) / 2
        out["net_top"] += (best - _at(exact[judge_on], top)) / 2
        rough_pick = np.nanargmax(rough[pick_on], 1)
        out["rough"] += (best - _at(exact[judge_on], rough_pick)) / 2
        out["rough_played"] += (_at(rough[judge_on], rough_pick) - _at(rough[judge_on], played)) / 2
    whole = (exact[0] + exact[1]) / 2
    out["lead"] = _at(whole, np.nanargmax(whole, 1)) - _at(whole, played)
    out["differs"] = (out["lead"] > LEVEL).astype(np.float64)
    return out


def exact_card(rows: Arrays) -> npt.NDArray[np.int8]:
    """The exact answer's card over all the deals of each decision: the card played where no
    card leads it (``LEVEL``)."""
    whole = (rows["exact_a"].astype(np.float64) + rows["exact_b"].astype(np.float64)) / 2
    best = np.take_along_axis(rows["legal"], np.nanargmax(whole, 1)[:, None], 1)[:, 0]
    return np.where(losses(rows)["differs"] > 0, best, rows["played"]).astype(np.int8)


def _mean(values: Floats) -> tuple[float, float]:
    """A mean and its standard error (0 with fewer than two values)."""
    if not len(values):
        return 0.0, 0.0
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), error


def real_leads(rows: Arrays, plays: Any) -> Floats:
    """Each decision's real lead of the exact answer's card over the card played: the round's
    worth to the deciding seat with the one less its worth with the other, both played for
    real (nan where the two are the same card or were not both played)."""
    lead = np.full(len(rows["game"]), np.nan)
    keys = zip(
        rows["game"].tolist(),
        rows["step"].tolist(),
        rows["seat"].tolist(),
        (rows["played"].astype(np.int64) + PLAY_BASE).tolist(),
        (exact_card(rows).astype(np.int64) + PLAY_BASE).tolist(),
        strict=True,
    )
    for k, (game, step, seat, made, other) in enumerate(keys):
        was, tried = plays.get((game, step, made)), plays.get((game, step, other))
        if made != other and was is not None and tried is not None:
            lead[k] = bid_check.worth(tried[1], tried[0], seat) - bid_check.worth(
                was[1], was[0], seat
            )
    return lead


def summarize(rows: Arrays, real: Floats) -> dict[str, Any]:
    """The decisions in all and by cards in the hand: how many, and a seat's in a Bola; how
    often the exact answer is another card than the one played; each of ``losses`` in points a
    decision with its standard error; and the real plays so far: how many of the changed cards
    were played, and the exact card's real lead a change."""
    loss = losses(rows)
    bolas = len(np.unique(rows["game"]))
    summary: dict[str, Any] = {"bolas": bolas, "decisions": len(rows["game"]), "groups": {}}
    groups = [("all", np.ones(len(rows["game"]), bool))]
    groups += [(str(left), rows["left"] == left) for left in np.unique(rows["left"]).tolist()]
    for name, of in groups:
        count = int(of.sum())
        seat_game = count / (4 * bolas) if bolas else 0.0
        entry: dict[str, Any] = {
            "decisions": count,
            "per_seat_game": seat_game,
            "differs": float(loss["differs"][of].mean()) if count else 0.0,
        }
        for key in ("played", "net_top", "rough", "rough_played"):
            entry[key], entry[key + "_error"] = _mean(loss[key][of])
        changed = of & (loss["differs"] > 0)
        done = changed & ~np.isnan(real)
        entry["lead"] = _mean(loss["lead"][changed])[0]
        entry["real_played"] = int(done.sum())
        entry["real"], entry["real_error"] = _mean(real[done])
        entry["lead_where_played"] = _mean(loss["lead"][done])[0]
        summary["groups"][name] = entry
    return summary


def lines(summary: dict[str, Any]) -> list[str]:
    """The summary as text."""
    out = [f"{summary['decisions']:,} card decisions of {summary['bolas']} Bolas"]
    for name, entry in summary["groups"].items():
        per_game = entry["per_seat_game"]
        title = "every decision" if name == "all" else f"{name} cards in the hand"
        out.append(
            f"{title}: {entry['decisions']:,} ({per_game:.1f} a seat a Bola); the exact answer is "
            f"another card at {entry['differs']:.1%}"
        )
        for key, label in (
            ("played", "the card played loses"),
            ("net_top", "the player's top card loses"),
            ("rough", "the rough finish with every card loses"),
        ):
            out.append(
                f"  {label} {entry[key]:.3f} points a decision (+-{entry[key + '_error']:.3f}), "
                f"{entry[key] * per_game:+.2f} a game"
            )
        out.append(
            f"  by the rough finish's own measure the card played loses "
            f"{entry['rough_played']:.3f} (+-{entry['rough_played_error']:.3f})"
        )
        if entry["real_played"]:
            share = entry["differs"] * per_game
            out.append(
                f"  played for real, {entry['real_played']:,} changed cards: the exact card leads "
                f"by {entry['real']:+.2f} points a change (+-{entry['real_error']:.2f}), where the "
                f"exact finish gave it {entry['lead_where_played']:+.2f}; "
                f"{entry['real'] * share:+.2f} a game (+-{entry['real_error'] * share:.2f})"
            )
    return out


def report(out: Path) -> dict[str, Any]:
    """The summary of what is saved in ``out`` so far, written next to it."""
    rows = load_rows(out)
    plays = bid_check.played(out / PLAYS) if (out / PLAYS / "parts").exists() else {}
    summary = summarize(rows, real_leads(rows, plays))
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# The real plays.


def play_tasks(rows: Arrays, config: CheckConfig, margin: float) -> list[bid_check.BolaTask]:
    """A ``bid_check`` task for every Bola with a changed card: each decision where the exact
    answer's card is not the one played and leads it by at least ``margin`` points over all the
    deals."""
    other = exact_card(rows)
    loss = losses(rows)
    changed = np.flatnonzero((loss["differs"] > 0) & (loss["lead"] >= margin))
    tasks: list[bid_check.BolaTask] = []
    for game in np.unique(rows["game"][changed]).tolist():
        at = changed[rows["game"][changed] == game]
        tasks.append(
            bid_check.BolaTask(
                config.record,
                game - game % config.part_games,
                game,
                config.player,
                config.seed,
                tuple(rows["step"][at].tolist()),
                tuple(rows["seat"][at].tolist()),
                tuple((rows["played"][at].astype(np.int64) + PLAY_BASE).tolist()),
                tuple((other[at].astype(np.int64) + PLAY_BASE).tolist()),
            )
        )
    return tasks


def play(out: Path, workers: int = 5, margin: float = 0.0) -> dict[str, Any]:
    """Play the changed cards of the decisions saved in ``out`` for real, both ways, skipping
    every play already saved (``bid_check.check_bola``). Returns the report."""
    config = load_config(out)
    folder = out / PLAYS
    (folder / "parts").mkdir(parents=True, exist_ok=True)
    saved = bid_check.played(folder)
    missing: list[tuple[bid_check.BolaTask, int]] = []
    for task in play_tasks(load_rows(out), config, margin):
        done = {(step, action) for game, step, action in saved if game == task.game}
        left = len(task.plays() - done)
        if left:
            missing.append((dataclasses.replace(task, done=tuple(sorted(done))), left))
    if missing:
        rounds = sum(left for _, left in missing)
        _say(f"{len(missing)} Bolas to play on {workers} workers: {rounds:,} rounds")
        started = time.perf_counter()
        done_rounds = 0
        with worker_pool(workers) as pool:
            futures = {pool.submit(bid_check.check_bola, task): (task, n) for task, n in missing}
            for future in as_completed(futures):
                task, left = futures[future]
                path = bid_check.check_path(folder, task.game)
                new = future.result()
                if path.exists():
                    with np.load(path) as data:
                        new = {name: np.concatenate([data[name], new[name]]) for name in new}
                bid_check.save_check(path, new)
                done_rounds += left
                elapsed = time.perf_counter() - started
                _say(
                    f"{done_rounds:,}/{rounds:,} rounds, {elapsed / 60:.1f} min, about "
                    f"{elapsed / done_rounds * (rounds - done_rounds) / 60:.1f} min left"
                )
    return report(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="The cards played in a round's last tricks, against the exact answer."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("run", help="value the recorded last cards exactly")
    check.add_argument("--record", type=Path, required=True, help="the recording's folder")
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--bolas", type=int, required=True, help="Bolas checked, in all")
    check.add_argument("--workers", type=int, default=1)
    check.add_argument("--deals", type=int, default=64, help="deals imagined at a decision")
    check.add_argument("--cards", type=int, default=MAX_CARDS, help="most cards in the hand")
    check.add_argument("--seed", type=int, default=0)
    real = commands.add_parser("play", help="play the changed cards for real")
    real.add_argument("--out", type=Path, required=True)
    real.add_argument("--workers", type=int, default=5)
    real.add_argument("--margin", type=float, default=0.0, help="least exact lead of a change")
    show = commands.add_parser("report", help="what is saved so far")
    show.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        start(args.out, args.record, args.deals, args.cards, args.seed)
        summary = run(args.out, args.bolas, args.workers)
    elif args.command == "play":
        summary = play(args.out, args.workers, args.margin)
    else:
        summary = report(args.out)
    for line in lines(summary):
        print(line)


if __name__ == "__main__":
    main()
