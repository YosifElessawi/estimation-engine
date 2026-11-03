"""Watch a training run live: ``uv run python -m estimation.rl.watch runs/m4-staged``.

Redraws every few seconds from the run's ``config.json``, ``selfplay/log.jsonl`` and
``selfplay/yardstick-reference.json``: the loop and the time left, the yardstick so far, how the
greedy nets do against three HeuristicBots, the learning numbers, loop time and memory. While the
run first records PIMC games, a progress bar for that, from ``warmstart/progress.json`` (or the
last ``[record]`` line of ``<run>/train.log``, for runs started before that file existed). It only
reads files, so it can be started and stopped at any time without touching the run. ``--once``
prints one frame and exits.
"""

import argparse
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

BLOCKS = "▁▂▃▄▅▆▇█"
RECENT = 10  # loops averaged for the "now" numbers
SPEED_WINDOW = 25  # loops averaged for the time per loop
STALLED_AFTER = 300.0  # seconds without a new loop before the run is called stalled


_RECORD_LINE = re.compile(
    r"\[record\] (\d+)/(\d+) PIMC games, ([\d,]+) card decisions, ([\d.]+) min elapsed"
)


@dataclass(frozen=True, slots=True)
class Recording:
    """The PIMC recording that starts a run: games done of all, card decisions, seconds."""

    done: int
    games: int
    rows: int
    seconds: float


def recording_of(path: Path) -> Recording | None:
    """The recording's progress while it runs, or None once it is saved or before it starts."""
    if (path / "warmstart" / "pimc.npz").exists():
        return None
    progress = path / "warmstart" / "progress.json"
    if progress.exists():
        data = json.loads(progress.read_text())
        return Recording(int(data["done"]), int(data["games"]), int(data["rows"]), data["seconds"])
    log = path / "train.log"
    if not log.exists():
        return None
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 8192))
        tail = handle.read().decode(errors="replace")
    found = _RECORD_LINE.findall(tail)
    if not found:
        return None
    done, games, rows, minutes = found[-1]
    return Recording(int(done), int(games), int(rows.replace(",", "")), float(minutes) * 60)


@dataclass(frozen=True, slots=True)
class RunState:
    """What a run's files say right now."""

    path: Path
    iterations: int
    entries: list[dict[str, Any]]
    reference: dict[str, Any] | None
    log_age: float | None  # seconds since the log last changed
    finished: bool
    recording: Recording | None = None  # the PIMC recording, while it runs


def load(path: Path, now: float | None = None) -> RunState | None:
    """The run in ``path``, or None before its config is written."""
    config_path = path / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    log_path = path / "selfplay" / "log.jsonl"
    entries: list[dict[str, Any]] = []
    log_age = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                entries.append(cast(dict[str, Any], json.loads(line)))
        log_age = (time.time() if now is None else now) - log_path.stat().st_mtime
    reference_path = path / "selfplay" / "yardstick-reference.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    return RunState(
        path=path,
        iterations=int(config["selfplay"]["iterations"]),
        entries=entries,
        reference=reference,
        log_age=log_age,
        finished=(path / "checkpoints" / "final.pt").exists(),
        recording=None if entries else recording_of(path),
    )


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """``values`` as a row of block characters, averaged into at most ``width`` columns."""
    if not values:
        return ""
    columns = min(width, len(values))
    buckets = [
        values[len(values) * k // columns : len(values) * (k + 1) // columns]
        for k in range(columns)
    ]
    means = [sum(bucket) / len(bucket) for bucket in buckets]
    low, high = min(means), max(means)
    span = high - low or 1.0
    return "".join(BLOCKS[min(7, int((m - low) / span * 8))] for m in means)


def duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _opponent_label(spec: str) -> str:
    name, *options = spec.split(":")
    if name == "pimc":
        samples = next((o.partition("=")[2] for o in options if o.startswith("samples=")), "")
        return f"PIMC-{samples}" if samples else "PIMC"
    if name == "rl":
        checkpoint = next((o.partition("=")[2] for o in options if o.startswith("checkpoint=")), "")
        return f"{Path(checkpoint).stem or 'reference'} (reference)"
    return spec


def _won(match: dict[str, Any]) -> Text:
    """Sets won with its 95% CI, green when the whole interval is above half, red below."""
    won = match["won"]
    low, high = max(0.0, won["low"]), min(1.0, won["high"])
    text = f"{100 * won['mean']:.0f}% ({100 * low:.0f}-{100 * high:.0f})"
    style = "green" if low > 0.5 else "red" if high < 0.5 else ""
    return Text(text, style=style)


def _header(state: RunState) -> Text:
    if state.finished:
        status, style = "finished", "green"
    elif state.log_age is not None and state.log_age > STALLED_AFTER:
        status, style = f"no new loop for {duration(state.log_age)}", "yellow"
    elif state.entries:
        status, style = "running", "cyan"
    elif state.recording is not None:
        status, style = "recording PIMC games for the card anchor", "cyan"
    else:
        status, style = "starting (no loop finished yet)", "cyan"
    text = Text.assemble(("Training run ", "bold"), (str(state.path), "bold"), "  ")
    text.append(status, style=style)
    if state.log_age is not None and not state.finished:
        text.append(f"  (last loop {state.log_age:.0f} s ago)", style="dim")
    return text


def _recording(recording: Recording) -> RenderableType:
    done, games = recording.done, recording.games
    line = Text(f"Recording: game {done:,} of {games:,}  ({100 * done / games:.0f}%)   ")
    line.append(f"{recording.rows:,} card decisions, {duration(recording.seconds)} so far")
    if done:
        line.append(f"   about {duration(recording.seconds / done * (games - done))} left")
    return Group(line, ProgressBar(total=games, completed=done, width=60))


def _progress(state: RunState) -> RenderableType:
    done = state.entries[-1]["iteration"] if state.entries else 0
    total = state.iterations
    elapsed = sum(float(e["seconds"]) for e in state.entries)
    recent = [float(e["seconds"]) for e in state.entries[-SPEED_WINDOW:]]
    per_loop = _mean(recent)
    line = Text(f"Loop {done:,} of {total:,}  ({100 * done / total:.0f}%)   ")
    line.append(f"training time {duration(elapsed)}")
    if state.entries and done < total:
        line.append(f"   about {duration(per_loop * (total - done))} left")
    if state.entries:
        line.append(f"   ({per_loop:.1f} s per loop)", style="dim")
    return Group(line, ProgressBar(total=total, completed=done, width=60))


def _yardstick(state: RunState) -> RenderableType:
    measured = [e for e in state.entries if "yardstick" in e]
    names: list[str] = []
    labels: dict[str, str] = {}
    for entry in measured:
        for name, match in entry["yardstick"]["matches"].items():
            if name not in labels:
                names.append(name)
                labels[name] = _opponent_label(match["opponent"])
    title = Text(
        "Yardstick: the same deals every time (sets won with 95% CI, RSA per game)", style="bold"
    )
    table = Table(expand=False)
    table.add_column("Loop", justify="right")
    if not names:
        names = ["pimc"]
        labels["pimc"] = "PIMC"
    for name in names:
        table.add_column(f"vs {labels[name]}: won")
        table.add_column("RSA", justify="right")
    if state.reference is not None:
        row: list[RenderableType] = [Text(Path(state.reference["checkpoint"]).stem, style="dim")]
        for name in names:
            match = state.reference["matches"].get(name)
            if match is None:
                row += ["", ""]
            else:
                row += [_won(match), f"{match['rsa']['mean']:+.0f}"]
        table.add_row(*row)
    for entry in measured[-12:]:
        row = [str(entry["iteration"])]
        for name in names:
            match = entry["yardstick"]["matches"].get(name)
            row += ["", ""] if match is None else [_won(match), f"{match['rsa']['mean']:+.0f}"]
        table.add_row(*row)
    if not measured:
        table.add_row("-", *(["not measured yet", ""] * len(names)))
    trends = [
        Text(
            f"  trend vs {labels[name]}: "
            + sparkline(
                [
                    float(e["yardstick"]["matches"][name]["won"]["mean"])
                    for e in measured
                    if name in e["yardstick"]["matches"]
                ]
            ),
            style="dim",
        )
        for name in names
        if len(measured) > 1
    ]
    return Group(title, table, *trends)


def _greedy(state: RunState) -> RenderableType:
    evals = [e["results"]["eval"] for e in state.entries if "eval" in e.get("results", {})]
    if not evals:
        return Text("Against 3 HeuristicBots: no games yet", style="dim")
    recent = evals[-RECENT:]
    first = [float(r["first"]) for r in evals]
    text = Text("Against 3 HeuristicBots (greedy games each loop, last ")
    text.append(
        f"{len(recent)} loops): first place {100 * _mean([r['first'] for r in recent]):.0f}%"
    )
    text.append(f", RSA {_mean([r['rsa'] for r in recent]):+.0f} per game")
    trend = Text(f"  first place over the run: {sparkline(first)}", style="dim")
    return Group(text, trend)


def _training(state: RunState) -> RenderableType:
    if not state.entries:
        return Text("")
    last = state.entries[-1]
    recent = state.entries[-RECENT:]
    kinds = Text("Training games (last loops): ")
    parts: list[str] = []
    for kind, label in (("self", "self-play"), ("heuristic", "vs heuristic"), ("pimc", "vs PIMC")):
        found = [e["results"][kind] for e in recent if kind in e["results"]]
        if found:
            parts.append(f"{label} RSA {_mean([f['rsa'] for f in found]):+.0f}")
    kinds.append(", ".join(parts))
    learning = Text(
        f"Learning: bid loss {last['bid_loss']:.3f}, card V {last['card_value_loss']:.3f} "
        f"Q {last['card_q_loss']:.3f}, anchor loss {last['anchor_loss']:.3f}"
    )
    if "anchor_weight" in last:
        learning.append(
            f"; anchor weight {last['anchor_weight']:.2f}, learning rate x{last['lr_factor']:.2f}"
        )
    if not last["train_card"]:
        learning.append("; card play frozen", style="yellow")
    memory = last.get("memory_mb", {})
    speed = Text(
        f"Loop time {last['seconds']:.1f} s (waiting for games {last['play_seconds']:.1f} s"
    )
    if "learn_seconds" in last:
        speed.append(f", learning {last['learn_seconds']:.1f} s")
    speed.append(")")
    lines = [kinds, learning, speed]
    guesses = [e["guess"]["card"] for e in state.entries if "card" in e.get("guess", {})]
    if guesses:
        recent_guesses = guesses[-RECENT:]
        accuracy = [float(g["accuracy"]) for g in guesses]
        lines.append(
            Text(
                "Hidden-card guesses (card net, last loops): "
                f"{100 * _mean([g['accuracy'] for g in recent_guesses]):.1f}% right, blind guess "
                f"{100 * _mean([g['blind'] for g in recent_guesses]):.1f}%; over the run "
                f"{sparkline(accuracy)}"
            )
        )
    if memory:
        lines.append(
            Text(
                f"Memory: trainer {memory['trainer_peak'] / 1024:.1f} GB "
                f"(GPU {memory['trainer_gpu'] / 1024:.1f} GB), each worker up to "
                f"{memory['worker_peak'] / 1024:.1f} GB"
            )
        )
    return Group(*lines)


def render(state: RunState | None, path: Path) -> RenderableType:
    if state is None:
        return Text(f"No training run in {path} yet: waiting for its config.json", style="yellow")
    recording: list[RenderableType] = []
    if state.recording is not None:
        recording = [_recording(state.recording), Text(""), Text("Then self-play:", style="dim")]
    return Group(
        _header(state),
        Text(""),
        *recording,
        _progress(state),
        Text(""),
        _yardstick(state),
        Text(""),
        _greedy(state),
        Text(""),
        _training(state),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.rl.watch", description="Watch a training run live."
    )
    parser.add_argument("run", type=Path, help="the run's folder, for example runs/m4-staged")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="seconds between redraws (default 5)"
    )
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be above 0")
    console = Console()
    if args.once:
        console.print(render(load(args.run), args.run))
        return 0
    try:
        with Live(render(load(args.run), args.run), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(args.interval)
                live.update(render(load(args.run), args.run), refresh=True)
    except KeyboardInterrupt:
        return 0  # Ctrl-C is how the view is closed


if __name__ == "__main__":
    raise SystemExit(main())
