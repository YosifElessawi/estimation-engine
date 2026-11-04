"""Watch an expert-iteration run live: ``uv run python -m estimation.rl.expert watch runs/expert``.

Redraws every few seconds from the run's files. Each round lists its four stages, recording the
Hybrid's games, the copy, the bidding refit and the match against the previous Hybrid, each with a
progress bar while it runs and its result once done. Recording counts every finished game, so the
bar moves game by game. It only reads files, so it can be started and stopped at any time without
touching the run.
"""

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.progress_bar import ProgressBar
from rich.text import Text

from .expert import COPY_REPORT, PROGRESS, ROUND_CHECKPOINT, games_in_progress, round_dir
from .watch import duration

INTERVAL = 2.0
BAR = 36
DONE, RUNNING, WAITING = "done", "running", "waiting"
_MARK = {DONE: ("✓", "green"), RUNNING: ("▸", "cyan"), WAITING: ("·", "dim")}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _lines(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def _bar(done: float, total: float) -> ProgressBar:
    return ProgressBar(total=max(total, 1.0), completed=min(done, total), width=BAR)


def _timing(seconds: float, done: float, total: float) -> str:
    text = f"{duration(seconds)} so far"
    if 0 < done < total:
        text += f", about {duration(seconds / done * (total - done))} left"
    return text


def _record(
    folder: Path, progress: dict[str, Any] | None, games: int, now: float
) -> tuple[str, Any, str]:
    stage = None if progress is None else progress.get("stage")
    if stage is None:
        return WAITING, None, "about 2.3 h"
    if stage != "record" or progress is None:
        return DONE, None, f"{games:,} games"
    done = int(progress.get("done", 0)) + games_in_progress(folder)
    fresh = done - int(progress.get("resumed_at", 0))
    elapsed = now - float(progress.get("started", now))
    left = ""
    if fresh > 0:
        left = f", about {duration(elapsed / fresh * (games - done))} left"
    text = f"{done:,} of {games:,} games ({100 * done / games:.0f}%), {duration(elapsed)} so far"
    return RUNNING, _bar(done, games), text + left


def _copy(folder: Path, progress: dict[str, Any] | None) -> tuple[str, Any, str]:
    report = _read(folder / COPY_REPORT)
    if report is not None:
        before, after = report["regret_before"], report["regret_after"]
        text = (
            f"top card loses {before['top1']:.2f} to {after['top1']:.2f} points a move "
            f"by the teacher's values (best of top 3: {before['top3']:.2f} to {after['top3']:.2f})"
        )
        guess = report.get("guess_accuracy")
        if guess is not None:
            text += f"; guesses {100 * guess['before']:.0f}% to {100 * guess['after']:.0f}% right"
        return DONE, None, text
    if progress is None or progress.get("stage") != "copy":
        return WAITING, None, "about 1 h"
    epochs = float(progress["epochs"])
    done = float(progress.get("done", 0.0))
    seconds = float(progress.get("seconds", 0.0))
    text = f"pass {min(int(done) + 1, int(epochs))} of {epochs:.0f}, " + _timing(
        seconds, done, epochs
    )
    return RUNNING, _bar(done, epochs), text


def _refit(folder: Path) -> tuple[str, Any, str]:
    config = _read(folder / "refit" / "config.json")
    entries = _lines(folder / "refit" / "selfplay" / "log.jsonl")
    total = int(config["selfplay"]["iterations"]) if config else 0
    if (folder / ROUND_CHECKPOINT).exists():
        return DONE, None, f"{total} loops of bidding self-play"
    if config is None:
        return WAITING, None, "about 2.5 h"
    done = entries[-1]["iteration"] if entries else 0
    seconds = sum(float(e["seconds"]) for e in entries)
    recent = entries[-10:]
    per_loop = sum(float(e["seconds"]) for e in recent) / len(recent) if recent else 0.0
    text = f"loop {done} of {total}, " + _timing(seconds, done, total)
    if recent:
        text += f" ({per_loop:.0f} s a loop)"
    return RUNNING, _bar(done, total), text


def _match(folder: Path) -> tuple[str, Any, str]:
    result = _read(folder / "match.json")
    if result is not None:
        bots = result["report"]["bots"]
        new = next(b for b in bots if b["name"].endswith(f"{folder.name}/{ROUND_CHECKPOINT}"))
        rsa = new["rsa"]
        verdict = "kept" if rsa["low"] > 0 else "not better: step 2 stops here"
        text = (
            f"{_sets_won(folder)} {rsa['mean']:+.1f} points a game "
            f"({rsa['low']:+.1f} to {rsa['high']:+.1f}): {verdict}"
        )
        return DONE, None, text
    status = _read(folder / "match.status.json")
    if status is None:
        return WAITING, None, "200 sets, about 45 min"
    units: list[dict[str, Any]] = status.get("units") or [{}]
    unit = units[0]
    done = int(unit.get("sets_done", 0))
    total = int(status.get("sets", 200))
    seconds = float(unit.get("seconds", 0.0))
    return RUNNING, _bar(done, total), f"set {done} of {total}, " + _timing(seconds, done, total)


def _sets_won(folder: Path) -> str:
    text = folder / "match.txt"
    try:
        for line in text.read_text().splitlines():
            if line.startswith("Sets: "):
                parts = line.split()
                won, lost = parts[parts.index("in") + 1].rstrip(","), parts[-1].rstrip(".")
                return f"won {won}-{lost} sets,"
    except (OSError, ValueError):
        pass
    return ""


def _stage_lines(name: str, stage: tuple[str, Any, str]) -> list[RenderableType]:
    """A stage's line (mark, name, result or progress) and, while it runs, its progress bar."""
    state, bar, text = stage
    mark, style = _MARK[state]
    line = Text.assemble(
        (f"  {mark} ", style),
        (f"{name:<27}", "dim" if state == WAITING else ""),
        (text, "dim" if state == WAITING else ""),
    )
    return [line] if bar is None else [line, Padding(bar, (0, 0, 0, 31))]


def render(out: Path, now: float | None = None) -> RenderableType:
    config = _read(out / "config.json")
    if config is None:
        return Text(f"No expert run in {out} yet: waiting for its config.json", style="yellow")
    now = time.time() if now is None else now
    rounds = int(config["expert"]["rounds"])
    blocks: list[RenderableType] = [
        Text.assemble(("Step 2: copying the Hybrid's search", "bold"), f"   {out}"),
        Text(
            "Each round: record Hybrid games labelled by a deeper search, copy them into the "
            "card net, refit bidding by self-play, then play the previous Hybrid. A round is "
            "kept only if it wins.",
            style="dim",
        ),
    ]
    for round_no in range(1, rounds + 1):
        folder = round_dir(out, round_no)
        if not folder.exists():
            note = "runs if the round before it wins its match"
            blocks += [
                Text(""),
                Text.assemble((f"Round {round_no}", "bold dim"), "  ", (note, "dim")),
            ]
            continue
        progress = _read(folder / PROGRESS)
        games = int(config["expert"]["games"])
        blocks += [Text(""), Text(f"Round {round_no}", style="bold")]
        blocks += _stage_lines("Record Hybrid games", _record(folder, progress, games, now))
        blocks += _stage_lines("Copy into the card net", _copy(folder, progress))
        blocks += _stage_lines("Refit bidding", _refit(folder))
        blocks += _stage_lines("Match vs previous Hybrid", _match(folder))
    return Group(*blocks)


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
