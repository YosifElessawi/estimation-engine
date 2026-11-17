"""Step 6's bid lessons (docs/rl/hybrid-iteration-plan.md): the bid teacher's verdicts at the
bidding decisions of recorded games.

    python -m estimation.onenet.bid_lessons run --config configs/onenet-bids-1.toml \\
        --out runs/onenet/bids-1/lessons
    python -m estimation.onenet.bid_lessons time --config configs/onenet-bids-1.toml

The games are a recording of the player inside its own search (``estimation.rl.onenet_record``
or ``estimation.onenet.gpu_record``: each Bola's seed and moves). The engine replays every Bola
on the CPU; at a share of its bidding decisions with a choice (``label_share``: of the Dash
Calls, of the auction's and of the estimates) the teacher (``estimation.onenet.bid_teacher``)
tries the player's options on imagined deals and keeps each one's lead over the move the player
made (the fields are in ``estimation.onenet.lessons``). One lesson part is saved per recording
part, so a stopped run resumes where it stopped, and two machines share a run as they share a
recording: one takes the parts from the first up, the other from the last down, and each saved
part is copied to the other (``rsync --ignore-existing``). With ``--wait`` a run teaches a
recording that is still being made, part by part as its parts arrive, and ``--hours`` ends it at
a deadline.

``time`` teaches the same decisions twice, on fresh deals each time: the pace, how often the
teacher picks the same option both times, and how far its picks are from the player's own.

Trainer code: the replay holds whole games, but the teacher reads only what the deciding seat
may see (its hand, the public table and the bidding so far), never another seat's cards.
"""

import argparse
import dataclasses
import functools
import json
import random
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from estimation.engine import Bola, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv import game as JG
from estimation.jaxenv.check import bits
from estimation.jaxenv.rules import NUM_ACTIONS
from estimation.rl.config import LESSON_STAGE, derive_seed, section_from
from estimation.rl.onenet_record import (
    DOWN,
    UP,
    Arrays,
    RecordConfig,
    load_part,
    part_path,
    part_starts,
)

from . import bid_teacher as T
from .gpu_record import player_of
from .lessons import Lessons, empty, join, lesson_path, load_lessons, save_lessons

HISTORY = (8, 16, 32, 48)  # earlier bidding choices a decision may hold, by compiled size
GATES = (1.0, 1.5, 2.0, 3.0)  # leads, in standard errors, the noise report also counts from
PHASES = ("Dash Call", "auction", "estimate")
SUMMARY = "summary.json"
Ints = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class LessonConfig:
    """The ``[lessons]`` section of a lessons config."""

    record: str = ""  # the games: a recording of the player inside its own search
    # The share of the decisions with a choice that are taught: of the Dash Calls, of the
    # auction's (bids, With, Pass) and of the estimates (and super calls).
    label_share: tuple[float, ...] = (0.25, 0.5, 1.0)
    coverage: float = 0.97  # the options tried hold this share of the player's move chances,
    min_options: int = 5  # are at least this many
    max_options: int = 8  # and at most this many
    hand_margin: float = 3.0  # tricks a call or estimate may be from the player's expected tricks
    deals: tuple[int, ...] = (32, 64, 128, 256)  # deals each stage of the narrowing adds
    opponent_temperature: float = 1.0  # the other seats' imagined bids: 1 their chances, 0 the top
    hopeless: float = 3.0  # an option this many standard errors behind the player's move is dropped
    redeals: int = 8  # draws per imagined deal to fit the void announcements
    batch: int = 64  # decisions searched at once
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.record:
            raise ValueError("lessons.record must name the recording of the games")
        shares = self.label_share
        if len(shares) != 3 or min(shares) < 0 or max(shares) > 1 or not any(shares):
            raise ValueError(
                "lessons.label_share must be three shares from 0 to 1, not all 0: "
                "of the Dash Calls, the auction's decisions and the estimates"
            )
        if self.batch < 1:
            raise ValueError("lessons.batch must be at least 1")
        settings = T.Settings(
            deals=self.deals,
            redeals=self.redeals,
            opponent_temperature=self.opponent_temperature,
            hopeless=self.hopeless,
        )
        T.check(settings, self.pruning)

    @property
    def pruning(self) -> T.Pruning:
        return T.Pruning(self.coverage, self.min_options, self.max_options, self.hand_margin)


def config_of(data: dict[str, Any]) -> LessonConfig:
    """A lessons config from TOML data: one ``[lessons]`` section."""
    unknown = set(data) - {"lessons"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}: lessons have only [lessons]")
    section = data.get("lessons")
    if not isinstance(section, dict):
        raise ValueError("a lessons config needs a [lessons] section")
    values = cast(dict[str, Any], section)
    if "record" not in values:
        raise ValueError("lessons.record must name the recording of the games")
    return section_from(LessonConfig(record=str(values["record"])), values, "lessons")


def load_config(path: Path) -> LessonConfig:
    return config_of(tomllib.loads(path.read_text()))


def _say(text: str) -> None:
    print(f"[bid-lessons] {text}", file=sys.stderr, flush=True)


# The decisions to teach, from a recording.


@dataclass(slots=True)
class Dealt:
    """A round as the engine dealt it, with its bidding moves as they were played."""

    round_no: int
    hands: tuple[int, int, int, int]
    fixed_trump: int  # -1 in an open round
    carry: int
    scores: tuple[int, int, int, int]
    moves: list[int] = field(default_factory=lambda: list[int]())


@dataclass(frozen=True, slots=True)
class Spot:
    """A bidding decision to teach: its Bola and its move's number there, the deciding seat, the
    phase, its round (among the rounds replayed) and its move's number in that round's bidding,
    the places in that bidding of the other seats' earlier choices, and its legal options."""

    game: int
    step: int
    seat: int
    phase: int
    round: int
    at: int
    history: tuple[int, ...]
    legal: tuple[int, ...]


def replay(part: Arrays, config: LessonConfig) -> tuple[list[Dealt], list[Spot], int]:
    """Replay every Bola of a recording part. Returns its rounds, the bidding decisions to teach
    (its phase's ``label_share`` of those with a choice, drawn from the seed and the Bola's
    number) and how many drawn decisions the teacher cannot hold: past a round's ``MOVES``
    bidding moves or the widest history, or in a Bola whose carry or scores have left JAX's
    int32 range."""
    rounds: list[Dealt] = []
    spots: list[Spot] = []
    unheld = 0
    actions = part["actions"].tolist()
    for row in range(len(part["seed"])):
        game = int(part["game"][row])
        bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
        draw = random.Random(derive_seed(config.seed, LESSON_STAGE, game))
        played = -1
        choices: list[tuple[int, int]] = []  # this round's bidding choices: place and seat
        first, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
        for step, action in enumerate(actions[first:end]):
            if len(bola.history) != played:  # a new round
                played = len(bola.history)
                rnd = bola.round
                fixed = -1 if rnd.fixed_trump is None else rnd.fixed_trump
                rounds.append(Dealt(rnd.round_no, rnd.hands, fixed, rnd.carry, rnd.scores))
                choices = []
            if bola.phase is not Phase.PLAY:
                dealt = rounds[-1]
                seat = bola.current_player
                if seat is None:
                    raise ValueError(f"Bola {game} has no player to act at move {step}")
                legal = bola.legal_actions()
                at = len(dealt.moves)
                if len(legal) > 1:
                    if draw.random() < config.label_share[int(bola.phase)]:
                        history = tuple(place for place, who in choices if who != seat)
                        fits = (
                            at < T.MOVES
                            and len(history) <= HISTORY[-1]
                            and dealt.carry <= JG.CARRY_CAP
                            and max(abs(score) for score in dealt.scores) <= JG.SCORE_LIMIT
                        )
                        if fits:
                            index = len(rounds) - 1
                            spots.append(
                                Spot(
                                    game,
                                    step,
                                    seat,
                                    int(bola.phase),
                                    index,
                                    at,
                                    history,
                                    tuple(legal),
                                )
                            )
                        else:
                            unheld += 1
                    choices.append((at, seat))
                dealt.moves.append(action)
            bola.step(action)
        if not bola.is_over or bola.scores != tuple(part["scores"][row].tolist()):
            raise ValueError(f"Bola {game} does not replay to its recorded scores")
    return rounds, spots, unheld


def batch_arrays(
    rounds: list[Dealt], spots: Sequence[Spot], size: int
) -> tuple[T.Rounds, Ints, Ints, Ints]:
    """``spots`` as the teacher takes them (``Teacher.look``): their rounds, each one's round
    and place in its bidding, and its history. Padded to ``size`` decisions (and as many
    rounds) by repeating the first, so every batch compiles to the same shapes."""
    local: dict[int, int] = {}
    for spot in spots:
        local.setdefault(spot.round, len(local))
    used = [rounds[index] for index in local]
    used += [used[0]] * (size - len(used))
    moves = np.zeros((size, T.MOVES), np.int32)
    for row, dealt in enumerate(used):
        kept = dealt.moves[: T.MOVES]
        moves[row, : len(kept)] = kept
    arrays = T.Rounds(
        jnp.asarray([dealt.round_no for dealt in used], jnp.int32),
        jnp.asarray(np.stack([bits(dealt.hands) for dealt in used])),
        jnp.asarray([dealt.fixed_trump for dealt in used], jnp.int32),
        jnp.asarray([dealt.carry for dealt in used], jnp.int32),
        jnp.asarray([dealt.scores for dealt in used], jnp.int32),
        jnp.asarray(moves),
        jnp.asarray([min(len(dealt.moves), T.MOVES) for dealt in used], jnp.int32),
    )
    filled = [*spots, *[spots[0]] * (size - len(spots))]
    where = np.array([local[spot.round] for spot in filled], np.int32)
    at = np.array([spot.at for spot in filled], np.int32)
    longest = max(len(spot.history) for spot in spots)
    history = np.full((size, next(h for h in HISTORY if h >= longest)), -1, np.int32)
    for row, spot in enumerate(filled):
        history[row, : len(spot.history)] = spot.history
    return arrays, where, at, history


# Teaching them.


@functools.cache
def _teacher(net: str, settings: T.Settings) -> T.Teacher:
    """One teacher per player file and settings, so its searches compile once a process."""
    return T.Teacher(T.load_bidder(net), settings)


class Labeler:
    """Teaches a recording's bidding decisions, ``config.batch`` at a time. The player and the
    search's settings (how deals are imagined and scored) are the recording's own."""

    def __init__(self, config: LessonConfig) -> None:
        self.config = config
        self.record = Path(config.record)
        path = self.record / "config.json"
        if not path.exists():
            raise ValueError(f"no recording at {self.record}: its config.json is missing")
        self.record_config = RecordConfig(**json.loads(path.read_text()))
        net, hybrid = player_of(self.record_config)
        self.settings = T.Settings(
            deals=config.deals,
            oversample=hybrid.oversample if hybrid.beliefs else 1,
            redeals=config.redeals,
            floor=hybrid.belief_floor,
            guess_temperature=hybrid.guess_temperature,
            leader_weight=hybrid.leader_weight,
            opponent_temperature=config.opponent_temperature,
            hopeless=config.hopeless,
        )
        self.teacher = _teacher(net, self.settings)

    def teach(
        self,
        rounds: list[Dealt],
        spots: Sequence[Spot],
        salt: int = 0,
        report: Callable[[int], None] | None = None,
    ) -> Lessons:
        """The lessons of ``spots`` (decisions of ``rounds``, as ``replay`` gives them), in
        order. Each batch's deals and draws come from the seed, its first decision and
        ``salt``; ``report`` hears how many decisions are done after each batch."""
        size = self.config.batch
        parts: list[Lessons] = []
        for start in range(0, len(spots), size):
            chunk = spots[start : start + size]
            seed = derive_seed(self.config.seed, LESSON_STAGE, chunk[0].game, chunk[0].step, salt)
            parts.append(self._batch(rounds, chunk, jax.random.PRNGKey(seed)))
            if report is not None:
                report(min(start + size, len(spots)))
        return join(parts) if parts else empty(self.config.max_options)

    def _batch(self, rounds: list[Dealt], spots: Sequence[Spot], key: jax.Array) -> Lessons:
        size, slots = self.config.batch, self.config.max_options
        arrays, where, at, history = batch_arrays(rounds, spots, size)
        batch, logits, tricks = self.teacher.look(arrays, where, at, history)
        x, me = np.asarray(batch.x), np.asarray(batch.me)
        options = np.full((size, slots), -1, np.int32)
        chance = np.zeros((size, slots), np.float32)
        rank = np.zeros((size, slots), np.int16)
        for row, spot in enumerate(spots):
            legal = np.flatnonzero(x[row, -NUM_ACTIONS:] > 0.5).tolist()
            if legal != list(spot.legal) or int(me[row]) != spot.seat:
                raise RuntimeError(
                    f"Bola {spot.game} move {spot.step}: the JAX round has seat {int(me[row])} "
                    f"choosing among {legal}, the engine seat {spot.seat} among {list(spot.legal)}"
                )
            found = T.candidates(logits[row], spot.legal, float(tricks[row]), self.config.pruning)
            count = len(found.actions)
            options[row, :count] = found.actions
            chance[row, :count] = found.chances
            rank[row, :count] = found.ranks
        verdict = self.teacher.teach(key, batch, options)
        taught = [row for row in range(len(spots)) if (options[row] >= 0).sum() > 1]
        return {
            "game": np.array([spots[i].game for i in taught], np.int32),
            "step": np.array([spots[i].step for i in taught], np.int32),
            "seat": np.array([spots[i].seat for i in taught], np.int8),
            "phase": np.array([spots[i].phase for i in taught], np.int8),
            "x": x[taught].astype(np.float16),
            "tricks": tricks[taught].astype(np.float32),
            "options": options[taught].astype(np.int8),
            "chance": chance[taught],
            "rank": rank[taught],
            "value": verdict.value[taught],
            "deals": verdict.deals[taught].astype(np.int16),
            "margin": verdict.margin[taught],
            "error": verdict.error[taught],
            "opening": verdict.opening[taught],
            "finalist": verdict.finalist[taught].astype(np.int8),
            "exam": verdict.exam[taught],
            "exam_error": verdict.exam_error[taught],
            "exam_deals": verdict.exam_deals[taught].astype(np.int16),
            "pick": verdict.pick[taught].astype(np.int8),
        }


# The run.


def check_run(config: LessonConfig, out: Path) -> None:
    """Start ``out`` with the config, or check that it holds the same config (a resume)."""
    path = out / "config.json"
    wanted = json.loads(json.dumps(dataclasses.asdict(config)))
    if path.exists():
        if json.loads(path.read_text()) != wanted:
            raise SystemExit(f"{out} holds lessons made with other settings; use a new --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wanted, indent=2) + "\n")


def part_counts(lessons: Lessons) -> dict[str, int]:
    """A part's lessons, and at how many of them the teacher left the player's favourite."""
    return {"lessons": len(lessons["pick"]), "changed": int((lessons["pick"] != 0).sum())}


def summarize(out: Path, record_config: RecordConfig) -> dict[str, Any]:
    """Totals over the saved lesson parts, and how many parts are still missing."""
    totals = {"lessons": 0, "changed": 0, "parts": 0, "missing_parts": 0}
    for first, _ in part_starts(record_config):
        path = lesson_path(out, first)
        if not path.exists():
            totals["missing_parts"] += 1
            continue
        totals["parts"] += 1
        for name, value in part_counts(load_lessons(path)).items():
            totals[name] += value
    return totals


def _pace(name: str, of: int, every: int) -> Callable[[int], None]:
    """A part's progress line, said each ``every`` decisions."""
    started = time.perf_counter()

    def report(done: int) -> None:
        if done % every == 0 and done < of:
            pace = done / (time.perf_counter() - started)
            _say(f"{name}: {done}/{of} decisions, {pace:.1f} a second")

    return report


def run(
    config: LessonConfig,
    out: Path,
    order: str = UP,
    report_every: int = 10,
    wait: bool = False,
    hours: float = 0.0,
    poll: float = 30.0,
) -> dict[str, Any]:
    """Teach the recording parts that have no lessons in ``out/parts`` yet, one at a time: the
    first such part whose recording is here (``up``) or the last (``down``). A part another
    machine saved meanwhile is kept as it came. Returns the lessons' summary.

    A part whose recording is not here is left missing, unless ``wait``: then the run looks
    again every ``poll`` seconds until every part has lessons, so it can teach a recording that
    is still being made or copied in. After ``hours`` (0: no deadline) no new part is started."""
    if order not in (UP, DOWN):
        raise ValueError(f"order must be {UP} or {DOWN}, not {order!r}")
    check_run(config, out)
    (out / "parts").mkdir(exist_ok=True)
    labeler = Labeler(config)
    parts = [first for first, _ in part_starts(labeler.record_config)]
    if order == DOWN:
        parts.reverse()
    missing = sum(1 for first in parts if not lesson_path(out, first).exists())
    _say(f"{missing} of {len(parts)} parts to teach, {config.batch} decisions at a time")
    started = time.perf_counter()
    mine = unheld = taught = 0
    while True:
        late = hours > 0 and time.perf_counter() - started > hours * 3600
        untaught = [first for first in parts if not lesson_path(out, first).exists()]
        ready = [first for first in untaught if part_path(labeler.record, first).exists()]
        if late or not untaught or not (ready or wait):
            if late and untaught:
                _say(f"{hours:g} h are up: {len(untaught)} parts are left untaught")
            break
        if not ready:
            time.sleep(poll)  # the recording's next part has not arrived
            continue
        first = ready[0]
        target = lesson_path(out, first)
        rounds, spots, skipped = replay(load_part(part_path(labeler.record, first)), config)
        unheld += skipped
        report = _pace(target.stem, len(spots), config.batch * report_every)
        lessons = labeler.teach(rounds, spots, report=report)
        if not target.exists():
            save_lessons(target, lessons)
            mine += 1
            taught += len(lessons["pick"])
        left = sum(1 for other in parts if not lesson_path(out, other).exists())
        elapsed = time.perf_counter() - started
        _say(
            f"{len(parts) - left}/{len(parts)} parts ({mine} here, {taught:,} lessons), "
            f"{elapsed / 60:.1f} min, about {elapsed / max(mine, 1) * left / 3600:.1f} h left "
            "at this machine's pace"
        )
    summary = {**summarize(out, labeler.record_config), "unheld_here": unheld}
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# The timing and noise test.


def gated(lessons: Lessons, gate: float) -> npt.NDArray[np.int64]:
    """Each lesson's pick, counting the finalist only when it led its exam by at least ``gate``
    standard errors (0 keeps every pick)."""
    pick = lessons["pick"].astype(np.int64)
    return np.where(lessons["exam"] >= gate * lessons["exam_error"], pick, 0)


def noise(first: Lessons, second: Lessons) -> dict[str, Any]:
    """How alike two teachings of the same decisions are (the same options, fresh deals): how
    often the pick is the same; how often the teacher leaves the player's favourite, and how
    often it then names the same move again; what following its picks is worth, in points a
    decision, on the other teaching's first-stage deals (``gain``, with its standard error: no
    luck of this teaching's deals is in it); how often its pick is outside the player's top 3;
    how many decisions were searched to the last deal. The same counting only exam leads of at
    least 1 to 3 standard errors (``GATES``), and by phase."""
    if not np.array_equal(first["options"], second["options"]):
        raise ValueError("the two teachings tried different options")
    rows = np.arange(len(first["pick"]))

    def alike(gate: float, of: Any = slice(None)) -> dict[str, Any]:
        a, b = gated(first, gate), gated(second, gate)
        worth = 0.5 * (second["opening"][rows, a] + first["opening"][rows, b])[of]
        a, b = a[of], b[of]
        changed = a != 0
        count = len(a)
        return {
            "decisions": count,
            "same_pick": float((a == b).mean()) if count else None,
            "changed": float(changed.mean()) if count else None,
            "changed_again": float((a[changed] == b[changed]).mean()) if changed.any() else None,
            "gain": float(worth.mean()) if count else None,
            "gain_error": float(worth.std(ddof=1) / np.sqrt(count)) if count > 1 else None,
        }

    report = alike(0.0)
    pick = first["pick"].astype(np.int64)
    changed = pick != 0
    whole = first["deals"][rows, first["finalist"]] == first["deals"][:, 0].max()
    report["outside_top_3"] = float((first["rank"][rows, pick] >= 3).mean())
    report["options"] = float((first["options"] >= 0).sum(1).mean())
    report["play_outs"] = float(first["deals"].sum(1).mean())
    report["searched_to_the_end"] = float(whole.mean())
    report["exam_deals"] = float(first["exam_deals"][whole].mean()) if whole.any() else None
    report["exam_error"] = float(first["exam_error"][whole].mean()) if whole.any() else None
    report["lead_when_changed"] = float(first["exam"][changed].mean()) if changed.any() else None
    report["gates"] = {str(gate): alike(gate) for gate in GATES}
    report["phases"] = {
        name: alike(0.0, first["phase"] == phase) for phase, name in enumerate(PHASES)
    }
    return report


def time_run(config: LessonConfig, decisions: int, keep: Path | None = None) -> dict[str, Any]:
    """Teach the recording's first ``decisions`` drawn decisions twice, on fresh deals each
    time. The second pass is timed, the compiling done; returns ``noise`` of the two with the
    pace (``seconds`` for ``taught`` decisions, ``per_hour``). With ``keep``, both teachings'
    lessons are saved there (``first.npz``, ``second.npz``)."""
    labeler = Labeler(config)
    passes: list[list[Lessons]] = [[], []]
    seconds = 0.0
    left = decisions
    for first, _ in part_starts(labeler.record_config):
        if left <= 0:
            break
        source = part_path(labeler.record, first)
        if not source.exists():
            continue
        rounds, spots, _ = replay(load_part(source), config)
        spots = spots[:left]
        left -= len(spots)
        passes[0].append(labeler.teach(rounds, spots, salt=1))
        started = time.perf_counter()
        passes[1].append(labeler.teach(rounds, spots, salt=2))
        seconds += time.perf_counter() - started
        _say(f"{decisions - left}/{decisions} decisions taught twice")
    if not passes[0]:
        raise ValueError(f"no recorded parts in {labeler.record}")
    taught = decisions - left
    spent = dict(labeler.teacher.spent)
    both = join(passes[0]), join(passes[1])
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        for name, lessons in zip(("first", "second"), both, strict=True):
            save_lessons(keep / f"{name}.npz", lessons)
    report = noise(*both)
    pace = {"taught": taught, "seconds": seconds, "per_hour": taught / seconds * 3600}
    return {**report, **pace, "spent": spent}


def _share(value: float | None) -> str:
    return "none" if value is None else f"{value:.0%}"


def _gain(row: dict[str, Any]) -> str:
    if row["gain"] is None or row["gain_error"] is None:
        return "nothing to measure"
    return f"{row['gain']:+.2f} points a decision (give or take {row['gain_error']:.2f})"


def noise_lines(report: dict[str, Any]) -> list[str]:
    """The timing and noise test in plain lines."""
    total = sum(report["spent"].values()) or 1.0
    spent = ", ".join(f"{name} {seconds / total:.0%}" for name, seconds in report["spent"].items())
    lines = [
        f"{report['taught']} decisions in {report['seconds']:.0f} s: "
        f"{report['per_hour']:,.0f} an hour "
        f"({report['options']:.1f} options and {report['play_outs']:,.0f} play-outs each)",
        f"where the time went, both teachings and their compiling: {spent}",
        f"searched to the last deal: {_share(report['searched_to_the_end'])} of decisions, "
        f"their exam {report['exam_deals'] or 0:.0f} deals, its lead good to "
        f"{report['exam_error'] or 0:.1f} points; the rest stopped early, every alternative "
        "hopeless",
        f"same pick on fresh deals: {_share(report['same_pick'])}",
        f"the teacher leaves the player's own move at {_share(report['changed'])} of decisions "
        f"(by {report['lead_when_changed'] or 0:.1f} points in the exam); "
        f"taught again, it names the same move at {_share(report['changed_again'])} of those",
        f"following its picks is worth {_gain(report)} on the other teaching's deals",
        f"its pick is outside the player's top 3 at {_share(report['outside_top_3'])}",
    ]
    for gate, row in report["gates"].items():
        lines.append(
            f"counting only exam leads of {gate}+ standard errors: same pick "
            f"{_share(row['same_pick'])}, leaves the player's move at {_share(row['changed'])}, "
            f"names the same move again at {_share(row['changed_again'])}, worth {_gain(row)}"
        )
    for name, row in report["phases"].items():
        lines.append(
            f"{name} ({row['decisions']}): same pick {_share(row['same_pick'])}, leaves the "
            f"player's move at {_share(row['changed'])}, worth {_gain(row)}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.onenet.bid_lessons",
        description="Teach the bidding decisions of recorded games with the bid teacher.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "teach (or resume) a recording"), ("time", "time and noise test")):
        cmd = commands.add_parser(name, help=text)
        cmd.add_argument("--config", type=Path, required=True, help="TOML lessons config")
        cmd.add_argument("--record", help="the recording's folder, instead of the config's")
        cmd.add_argument("--batch", type=int, help="decisions at once, instead of the config's")
        if name == "run":
            cmd.add_argument("--out", type=Path, required=True, help="the lessons' folder")
            cmd.add_argument(
                "--order", choices=(UP, DOWN), default=UP, help="from the first part or the last"
            )
            cmd.add_argument("--report-every", type=int, default=10, help="batches between lines")
            cmd.add_argument(
                "--wait", action="store_true", help="wait for recording parts still to arrive"
            )
            cmd.add_argument("--hours", type=float, default=0.0, help="start no part after this")
        else:
            cmd.add_argument("--decisions", type=int, default=512, help="decisions to teach twice")
            cmd.add_argument("--out", type=Path, help="write the report here as JSON")
            cmd.add_argument("--keep", type=Path, help="save both teachings' lessons here")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        changes = {"record": args.record, "batch": args.batch}
        config = dataclasses.replace(config, **{k: v for k, v in changes.items() if v is not None})
    except (OSError, ValueError) as error:
        parser.error(f"--config {args.config}: {error}")
    # Compiled searches are kept between runs: each batch shape compiles once.
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "bid-lessons"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _say(f"JAX devices: {jax.devices()}")
    if args.command == "time":
        report = time_run(config, args.decisions, args.keep)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\n".join(noise_lines(report)))
        return 0
    summary = run(config, args.out, args.order, args.report_every, args.wait, args.hours)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
