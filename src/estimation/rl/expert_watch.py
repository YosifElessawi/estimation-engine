"""Watch an expert-iteration run live: ``uv run python -m estimation.rl.expert watch runs/expert``.

Redraws every few seconds from the run's files: each round's stages (record, copy, refit) with a
progress bar for the one running, the copy's held-out regret before and after, and the bidding
refit's own training view (``watch``). It only reads files, so it can be started and stopped at
any time without touching the run.
"""

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from . import watch as train_watch
from .expert import COPY_REPORT, PROGRESS, ROUND_CHECKPOINT, round_dir

INTERVAL = 5.0


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _bar(done: float, total: float, line: Text) -> RenderableType:
    return Group(line, ProgressBar(total=max(total, 1), completed=done, width=60))


def _stage_view(folder: Path) -> RenderableType | None:
    """The running stage's progress in round ``folder``, or None when no stage is running."""
    progress = _read(folder / PROGRESS)
    if progress is None or (folder / ROUND_CHECKPOINT).exists():
        return None
    stage = progress.get("stage")
    if stage == "record":
        done, games = int(progress.get("done", 0)), int(progress["games"])
        line = Text(f"Recording Hybrid games: {done:,} of {games:,}  ({100 * done / games:.0f}%)")
        if "left" in progress:
            line.append(f"   about {train_watch.duration(float(progress['left']))} left")
        return _bar(done, games, line)
    if stage == "copy":
        done, epochs = int(progress.get("done", 0)), int(progress["epochs"])
        line = Text(f"Copying the teacher: epoch {done} of {epochs}")
        if done and "seconds" in progress:
            left = float(progress["seconds"]) / done * (epochs - done)
            line.append(f"   about {train_watch.duration(left)} left")
        return _bar(done, epochs, line)
    if stage == "refit":
        state = train_watch.load(folder / "refit")
        if state is None:
            return Text("Bidding refit: starting")
        return Group(Text("Bidding refit (only BiddingNet learns):", style="bold"),
                     train_watch.render(state, folder / "refit"))  # fmt: skip
    return None


def _status(folder: Path) -> tuple[str, str, str]:
    """Each stage's state in round ``folder``: done, running or waiting."""
    progress = _read(folder / PROGRESS) or {}
    stage = progress.get("stage", "")
    if (folder / ROUND_CHECKPOINT).exists():
        return "done", "done", "done"
    record = "done" if stage in ("copy", "refit") or (folder / "copy.pt").exists() else ""
    copy = "done" if (folder / "copy.pt").exists() else ""
    record = record or ("running" if stage == "record" else "waiting")
    copy = copy or ("running" if stage == "copy" else "waiting")
    refit = "running" if stage == "refit" else "waiting"
    return record, copy, refit


def render(out: Path) -> RenderableType:
    config = _read(out / "config.json")
    if config is None:
        return Text(f"No expert run in {out} yet: waiting for its config.json", style="yellow")
    rounds = int(config["expert"]["rounds"])
    table = Table(title="Rounds", title_justify="left")
    columns = ("Round", "Record", "Copy", "Refit", "Regret of top 1/2/3 cards (points)", "Guesses")
    for column in columns:
        table.add_column(column)
    running: RenderableType | None = None
    for round_no in range(1, rounds + 1):
        folder = round_dir(out, round_no)
        if not folder.exists():
            table.add_row(str(round_no), "waiting", "waiting", "waiting", "", "")
            continue
        styled = [
            Text(s, style="green" if s == "done" else "cyan" if s == "running" else "dim")
            for s in _status(folder)
        ]
        report = _read(folder / COPY_REPORT)
        regret = guesses = ""
        if report is not None:
            b, a = report["regret_before"], report["regret_after"]
            regret = (
                f"{b['top1']:.2f}/{b['top2']:.2f}/{b['top3']:.2f} before, "
                f"{a['top1']:.2f}/{a['top2']:.2f}/{a['top3']:.2f} after"
            )
            if "guess_accuracy" in report:
                g = report["guess_accuracy"]
                guesses = (
                    f"{100 * g['before']:.0f}% to {100 * g['after']:.0f}% "
                    f"(blind {100 * g['blind']:.0f}%)"
                )
        table.add_row(str(round_no), *styled, regret, guesses)
        running = running or _stage_view(folder)
    header = Text.assemble(("Expert iteration ", "bold"), (str(out), "bold"))
    parts: list[RenderableType] = [header, Text(""), table]
    if running is not None:
        parts += [Text(""), running]
    return Group(*parts)


def watch(out: Path, once: bool = False) -> int:
    console = Console()
    if once:
        console.print(render(out))
        return 0
    try:
        with Live(render(out), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(INTERVAL)
                live.update(render(out), refresh=True)
    except KeyboardInterrupt:
        return 0  # Ctrl-C is how the view is closed
