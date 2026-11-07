"""Watch step 5's self-play live in a terminal: ``python -m estimation.onenet.watch <run>``.

One colourful page, redrawn every few seconds from the run's files (it only reads them, so it
starts and stops at any time without touching the run): how far the run is and how fast it
goes, its strength against the copy at every check with a trend line, the guards that keep the
head start, the practice questions, and the check-ups against Hybrid v4 as they come in
(arena reports in ``--checkups``). Green means at or past the mark to beat, red means short of it.

On the home server it needs only ``rich`` (no JAX):

    PYTHONPATH=src ~/jax-env/bin/python -m estimation.onenet.watch runs/onenet/selfplay
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

INTERVAL = 3.0
BLOCKS = "▁▂▃▄▅▆▇█"
STALE = 180.0  # seconds without a new loop before the page warns
EVEN_KING, BLIND_HOLDERS, GUESSNET_HOLDERS, BLIND_JUDGE = 0.25, 0.35, 0.47, 0.52


def duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """The last ``width`` values as a line of block characters, scaled to their own range."""
    values = [v for v in values if math.isfinite(v)][-width:]
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return BLOCKS[3] * len(values)
    return "".join(BLOCKS[round((v - low) / (high - low) * (len(BLOCKS) - 1))] for v in values)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # the line being written; the next redraw reads it whole
    return rows


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _mark(value: float, mark: float, higher_is_better: bool = True) -> str:
    good = value >= mark if higher_is_better else value <= mark
    return "bold green" if good else "bold red"


LABEL = 26


def _label(text: str) -> Text:
    return Text(f"  {text:<{LABEL}}", style="bold")


def _line(label: str, *parts: tuple[str, str] | str) -> Text:
    out = _label(label)
    for part in parts:
        out.append_text(Text(part) if isinstance(part, str) else Text(part[0], style=part[1]))
    return out


def _heading(text: str) -> Text:
    return Text(f"\n{text}", style="bold cyan")


def _progress(
    rows: list[dict[str, Any]], config: dict[str, Any], now: float, run: Path
) -> list[RenderableType]:
    last = rows[-1]
    total = int(config["loops"])
    loop = int(last["loop"])
    out: list[RenderableType] = [
        _line(
            "▸ Loops", (f"{loop:,}", "bold yellow"), f" of {total:,} ({100 * loop / total:.0f}%)"
        ),
        Padding(
            ProgressBar(total=total, completed=loop, width=60, complete_style="yellow"),
            (0, 0, 0, LABEL + 2),
        ),
    ]
    decisions = loop * int(config["batch"]) * int(config["steps"])
    speed = f"{last['decisions_per_s_now']:,.0f} decisions/s · {last['bolas_per_s']:.0f} Bolas/s"
    out.append(
        _line(
            "  Time",
            (duration(float(last["minutes_left"]) * 60), "bold"),
            " left",
            f" · {speed}",
        )
    )
    out.append(
        _line("  Played", f"{last['bolas_total']:,} Bolas · {decisions / 1e6:,.0f} M decisions")
    )
    try:
        quiet = now - (run / "log.jsonl").stat().st_mtime
    except OSError:
        quiet = 0.0
    if quiet > STALE:
        out.append(
            Text(f"  no new loop for {duration(quiet)}: is the run still going?", style="bold red")
        )
    return out


def _strength(rows: list[dict[str, Any]]) -> list[RenderableType]:
    checks = [r for r in rows if "king_vs_copy" in r]
    out: list[RenderableType] = [
        _heading("Strength against the copy (checked every 100 loops, 25% King is even)")
    ]
    if not checks:
        return [*out, Text("  waiting for the first check", style="dim")]
    king = [float(r["king_vs_copy"]) for r in checks]
    place = [float(r["place_vs_copy"]) for r in checks]
    change = king[-1] - king[0]
    out.append(
        _line(
            "King rate",
            (f"{king[-1]:6.1%}", _mark(king[-1], EVEN_KING)),
            "  ",
            (sparkline(king), "magenta"),
            ("  ↑" if change > 0 else "  ↓", "green" if change > 0 else "red"),
            f" {100 * change:+.1f} since loop {checks[0]['loop']}",
        )
    )
    out.append(
        _line(
            "Place (King +1, Kooz -1)",
            (f"{place[-1]:+6.3f}", _mark(place[-1], 0.0)),
            "  ",
            (sparkline(place), "magenta"),
            f"  best {max(place):+.3f} at loop {checks[place.index(max(place))]['loop']}",
        )
    )
    return out


def _guards(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[RenderableType]:
    last = rows[-1]
    distance = [float(r["distance_from_copy"]) for r in rows]
    return [
        _heading("Keeping the head start"),
        _line(
            "Distance from the copy",
            (f"{distance[-1]:.4f}", "bold"),
            "  ",
            (sparkline(distance), "blue"),
            ("  0 is the copy's own moves", "dim"),
        ),
        _line(
            "Pull toward the copy",
            (f"{last['anchor_weight']:.2f}", "bold"),
            (f"  fades to 0 by loop {int(config['anchor_loops']):,}", "dim"),
        ),
    ]


def _practice(rows: list[dict[str, Any]]) -> list[RenderableType]:
    last = rows[-1]
    holders = float(last["holders_named"])
    judged = [r for r in rows if r["final_mse"] > 0]
    out: list[RenderableType] = [
        _heading("Practice questions"),
        _line(
            "Hidden cards named",
            (f"{holders:.0%}", _mark(holders, BLIND_HOLDERS)),
            "  ",
            (sparkline([float(r["holders_named"]) for r in rows]), "blue"),
            ("  blind 35% · GuessNet 47%", "dim"),
        ),
    ]
    if judged:
        error = float(judged[-1]["final_mse"])
        out.append(
            _line(
                "Judge error",
                (f"{error:.3f}", _mark(error, BLIND_JUDGE, higher_is_better=False)),
                "  ",
                (sparkline([float(r["final_mse"]) for r in judged]), "blue"),
                ("  blind 0.52, lower is better", "dim"),
            )
        )
    made = float(last["caller_made"])
    out.append(
        _line(
            "Callers who make it",
            (f"{made:.0%}", "bold"),
            "  ",
            (sparkline([float(r["caller_made"]) for r in rows]), "blue"),
            ("  in the training games", "dim"),
        )
    )
    return out


def _bot(data: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    """The bot whose spec starts with ``prefix`` in an arena report file."""
    bots = data.get("report", {}).get("bots", [])
    return next((b for b in bots if str(b["name"]).startswith(prefix)), None)


def _checkups(folder: Path | None) -> list[RenderableType]:
    if folder is None or not folder.is_dir():
        return []
    table = Table(show_edge=False, header_style="bold", box=None, pad_edge=False, padding=(0, 2))
    for column in ("  Player", "Place", "King", "Hybrid v4 King", "Points per game (95% CI)"):
        table.add_column(column)
    found = False
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if path.name.endswith(".status.json"):
            continue
        report = _json(path)
        ours = _bot(report or {}, "onenet")
        theirs = _bot(report or {}, "hybrid")
        if ours is None or theirs is None:
            continue
        found = True
        place = float(ours.get("place", {}).get("mean", math.nan))
        rsa = ours["rsa"]
        ci = (
            f"{rsa['mean']:+.1f} ({rsa['low']:+.1f} to {rsa['high']:+.1f})"
            if rsa.get("low") is not None
            else f"{rsa['mean']:+.1f}"
        )
        table.add_row(
            f"  {path.stem}",
            Text(f"{place:+.3f}", style=_mark(place, 0.0)),
            Text(
                f"{ours['first_place']['mean']:.0%}",
                style=_mark(ours["first_place"]["mean"], EVEN_KING),
            ),
            f"{theirs['first_place']['mean']:.0%}",
            Text(ci, style=_mark(rsa["mean"], 0.0)),
        )
    for status in sorted(folder.glob("*.status.json")):
        data = _json(status) or {}
        if data.get("finished") is False:
            found = True
            table.add_row(
                f"  {status.name.removesuffix('.status.json')}",
                Text("playing…", style="yellow"),
                "",
                "",
                "",
            )
    if not found:
        return []
    return [_heading("Check-ups against Hybrid v4 (100 sets each, on the Mac)"), table]


def render(run: Path, checkups: Path | None = None, now: float | None = None) -> RenderableType:
    now = time.time() if now is None else now
    config = _json(run / "config.json")
    rows = _rows(run / "log.jsonl")
    if config is None or not rows:
        return Text(f"No self-play in {run} yet: waiting for its first loop", style="yellow")
    title = Text.assemble(
        ("Step 5 self-play: the one net learns from its own games", "bold"),
        (f"   {run}", "dim"),
    )
    about = Text(
        f"From {config['start']} (sharpened {config['sharpen']}x) · {int(config['batch']):,} games "
        f"at once · final place is the reward",
        style="dim",
    )
    parts: list[RenderableType] = [title, about, Text("")]
    parts += _progress(rows, config, now, run)
    parts += _strength(rows)
    parts += _guards(rows, config)
    parts += _practice(rows)
    parts += _checkups(checkups)
    return Group(*parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m estimation.onenet.watch", description=__doc__)
    parser.add_argument("run", type=Path, help="the self-play run's --out folder")
    parser.add_argument("--checkups", type=Path, default=Path("runs/onenet/checkups"))
    parser.add_argument("--interval", type=float, default=INTERVAL)
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    args = parser.parse_args(argv)
    console = Console()
    if args.once:
        console.print(render(args.run, args.checkups))
        return 0
    try:
        with Live(render(args.run, args.checkups), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(args.interval)
                live.update(render(args.run, args.checkups), refresh=True)
    except KeyboardInterrupt:
        return 0  # Ctrl-C is how the page is closed


if __name__ == "__main__":
    raise SystemExit(main())
