"""Watch step 5's recording live: ``uv run python -m estimation.rl.onenet_record watch <out>``.

Redraws every few seconds from the recording's files: how many Bolas are recorded (counting
every finished Bola, also in parts still being recorded), the time so far and left, and what the
saved parts hold. It only reads files, so it can be started and stopped at any time without
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

from .onenet_record import PROGRESS, part_counts
from .watch import duration

INTERVAL = 2.0
BAR = 40
LABEL = 20
_counted: dict[Path, dict[str, int]] = {}  # saved parts never change, so each is read once


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _in_progress(parts: Path) -> int:
    """Bolas finished in parts still being recorded."""
    total = 0
    for path in parts.glob("part-*.games"):
        try:
            total += int(path.read_text())
        except (OSError, ValueError):
            continue  # being written or just removed; the next look counts it
    return total


def _saved(parts: Path) -> dict[str, int]:
    totals = {"bolas": 0, "actions": 0, "bids": 0, "cards": 0}
    for path in sorted(parts.glob("part-*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        if path not in _counted:
            try:
                _counted[path] = part_counts(path)
            except (OSError, ValueError, KeyError):
                continue  # not readable yet
        for key, value in _counted[path].items():
            totals[key] += value
    return totals


def _line(label: str, text: str, style: str = "") -> Text:
    return Text.assemble(("  " + f"{label:<{LABEL}}", "bold"), (text, style))


def render(out: Path, now: float | None = None) -> RenderableType:
    config = _read(out / "config.json")
    if config is None:
        return Text(f"No recording in {out} yet: waiting for its config.json", style="yellow")
    now = time.time() if now is None else now
    games = int(config["games"])
    blocks: list[RenderableType] = [
        Text.assemble(("Step 5: recording Hybrid v4 for the one-net copy", "bold"), f"   {out}"),
        Text(
            f"Hybrid v4 plays every seat. At {float(config['label_share']):.0%} of its card "
            f"decisions a deeper search ({config['teacher_samples']} deals, every card) values "
            "every legal card and plays its best; at every bid, BiddingNet's values are kept. "
            f"{config['workers']} workers.",
            style="dim",
        ),
        Text(""),
    ]
    progress = _read(out / PROGRESS)
    parts = out / "parts"
    saved = _saved(parts)
    if progress is None:
        blocks.append(_line("Bolas recorded", f"{saved['bolas']:,} of {games:,}, starting"))
        return Group(*blocks)
    done = int(progress.get("done", 0)) + _in_progress(parts)
    started = float(progress.get("started", now))
    finished = progress.get("finished")
    share = f"{done:,} of {games:,} ({100 * done / games:.0f}%)"
    if finished is not None:
        blocks.append(
            _line("✓ Bolas recorded", f"{share}, in {duration(float(finished) - started)}", "green")
        )
    else:
        blocks.append(_line("▸ Bolas recorded", share))
        bar = ProgressBar(total=games, completed=min(done, games), width=BAR)
        blocks.append(Padding(bar, (0, 0, 0, LABEL + 2)))
        elapsed = now - started
        fresh = done - int(progress.get("resumed_at", 0))
        timing = f"{duration(elapsed)} so far"
        if fresh > 0 and elapsed > 0:
            left = elapsed / fresh * (games - done)
            timing += f", about {duration(left)} left, {3600 * fresh / elapsed:.0f} Bolas an hour"
        blocks.append(_line("Time", timing))
    if saved["bolas"]:
        blocks.append(
            _line(
                "Saved",
                f"{saved['bolas']:,} Bolas ({saved['actions']:,} moves): "
                f"{saved['bids']:,} bids with BiddingNet's values, "
                f"{saved['cards']:,} cards with the teacher's values",
            )
        )
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
