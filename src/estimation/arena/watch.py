"""Watch a rating run live: ``uv run python -m estimation.arena.watch runs/ladder-200``.

Redraws every few seconds from the run's status file (see ``status.py``): an Elo ladder's
folder, an arena report path given to ``--out``, or the status file itself. It shows how far the
run got and how long is left, the pair or lineup being played with its results so far, every
finished pair and, once the run ends, the ratings. It only reads the file, so it can be started
and stopped at any time without touching the run. ``--once`` prints one frame and exits.
"""

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rich.console import Console, Group, JustifyMethod, RenderableType
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .roster import name_of
from .status import status_path

STALLED_AFTER = 300.0  # seconds without an update before a running run is called stalled


@dataclass(frozen=True, slots=True)
class RunStatus:
    """What a run's status file says right now."""

    path: Path
    data: dict[str, Any]
    age: float  # seconds since the file was last updated
    alive: bool  # whether the run's process still exists


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, it is just not ours
    return True


def load(target: Path, now: float | None = None) -> RunStatus | None:
    """The status of the run at ``target``, or None before it writes one."""
    path = status_path(target)
    if not path.exists():
        return None
    data = cast(dict[str, Any], json.loads(path.read_text()))
    updated = datetime.fromisoformat(str(data["updated"])).timestamp()
    age = (time.time() if now is None else now) - updated
    return RunStatus(path=path, data=data, age=max(age, 0.0), alive=_alive(int(data["pid"])))


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def short(bot: str) -> str:
    """The roster's name for a bot, or for a bot the roster does not know, its spec with a
    checkpoint path cut to the file's name."""
    known = name_of(bot)
    if known != bot:
        return known
    name, *options = bot.split(":")
    parts = [name]
    for option in options:
        key, _, value = option.partition("=")
        parts.append(Path(value).stem if key == "checkpoint" else option)
    return ":".join(parts)


def _state(status: RunStatus) -> tuple[str, str]:
    data = status.data
    if data["finished"]:
        return "finished", "green"
    if data["error"]:
        return str(data["error"]), "red"
    if not status.alive:
        return "stopped: its process is gone", "red"
    if status.age > STALLED_AFTER:
        return f"no update for {duration(status.age)}", "yellow"
    return "running", "cyan"


def _header(status: RunStatus) -> RenderableType:
    data = status.data
    kind = "Elo ladder" if data["kind"] == "ladder" else "Arena run"
    state, style = _state(status)
    title = Text.assemble((f"{kind} ", "bold"), (str(status.path.parent), "bold"), "  ")
    title.append(state, style=style)
    if not data["finished"]:
        title.append(f"  (updated {duration(status.age)} ago)", style="dim")
    started = datetime.fromisoformat(str(data["started"])).astimezone()
    sets = int(data["sets"])
    what = "sets per pair" if data["kind"] == "ladder" else "sets"
    facts = Text(
        f"{data['workers']} worker{'s' if data['workers'] != 1 else ''} · {sets:,} {what} · "
        f"seed {data['seed']} · started {started:%H:%M}",
        style="dim",
    )
    return Group(title, Text(str(data["command"]), style="dim"), facts)


def _units(data: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], data["units"])


def _progress(status: RunStatus) -> RenderableType:
    data = status.data
    units = _units(data)
    sets = int(data["sets"])
    played = [u for u in units if u["state"] != "reused"]
    done_units = sum(u["state"] in ("done", "reused") for u in units)
    reused = sum(u["state"] == "reused" for u in units)
    played_sets = sum(int(u["sets_done"]) for u in played)
    seconds = sum(float(u["seconds"]) for u in played)
    to_play = sum(sets - int(u["sets_done"]) for u in played if u["state"] != "done")
    line = Text()
    if data["kind"] == "ladder":
        line.append(f"Pairs {done_units} of {len(units)} done")
        if reused:
            line.append(f" ({reused} reused)")
        line.append("   ")
    line.append(f"sets {played_sets:,} played")
    if not data["finished"]:
        line.append(f", {to_play:,} to go")
    line.append(f"   {duration(seconds)} of play")
    if played_sets and to_play and not data["finished"]:
        line.append(f"   about {duration(seconds / played_sets * to_play)} left")
        line.append(f"   ({seconds / played_sets:.1f} s per set)", style="dim")
    total = played_sets + (0 if data["finished"] else to_play)
    return Group(line, ProgressBar(total=max(total, 1), completed=played_sets, width=60))


def _pct(value: Any) -> str:
    return "" if value is None else f"{100 * float(value):.0f}%"


def _signed(value: Any) -> str:
    return "" if value is None else f"{float(value):+,.1f}"


def _record(unit: dict[str, Any]) -> tuple[str, str, str]:
    """Sets won by the stronger bot (the second of the lineup), drawn, and won by the weaker."""
    record = unit.get("record")
    if not record:
        return "", "", ""
    stronger = unit["lineup"][1]
    upper, lower = (
        (record["b_wins"], record["a_wins"])
        if record["b"] == stronger
        else (record["a_wins"], record["b_wins"])
    )
    return str(upper), str(record["draws"]), str(lower)


def _current(status: RunStatus) -> RenderableType:
    data = status.data
    running = [u for u in _units(data) if u["state"] == "running"]
    if not running or data["finished"]:
        return Text("")
    unit = running[0]
    sets = int(data["sets"])
    done = int(unit["sets_done"])
    title = Text("Now playing: ", style="bold")
    title.append(", ".join(short(bot) for bot in dict.fromkeys(unit["lineup"])))
    title.append(f"   set {done:,} of {sets:,}")
    bar = ProgressBar(total=sets, completed=done, width=60)
    lines: list[RenderableType] = [title, bar, Text("")]  # the bar ends without a line break
    if data["kind"] == "ladder":
        upper, draws, lower = _record(unit)
        if upper:
            bots = {b["bot"]: b for b in unit["bots"]}
            stronger = bots.get(unit["lineup"][1], {})
            lines.append(
                Text(
                    f"  stronger won {upper}, drew {draws}, weaker won {lower}; stronger first "
                    f"{_pct(stronger.get('first'))}, RSA {_signed(stronger.get('rsa'))} per game"
                )
            )
    else:
        table = Table(expand=False)
        table.add_column("Bot")
        table.add_column("First place", justify="right")
        table.add_column("RSA per game", justify="right")
        for bot in unit["bots"]:
            table.add_row(short(bot["bot"]), _pct(bot["first"]), _signed(bot["rsa"]))
        lines.append(table)
    return Group(*lines)


def _pairs(status: RunStatus) -> RenderableType:
    data = status.data
    if data["kind"] != "ladder":
        return Text("")
    table = Table(title="Pairs, weaker vs stronger", title_justify="left", expand=False)
    columns: tuple[tuple[str, JustifyMethod], ...] = (
        ("Pair", "left"),
        ("State", "left"),
        ("Stronger won", "right"),
        ("Drew", "right"),
        ("Weaker won", "right"),
        ("Stronger 1st", "right"),
        ("Stronger RSA", "right"),
        ("Time", "right"),
    )
    for column, justify in columns:
        table.add_column(column, justify=justify)
    styles = {"done": "", "reused": "dim", "running": "cyan", "pending": "dim"}
    for unit in _units(data):
        lower, upper = unit["lineup"][0], unit["lineup"][1]
        won, draws, lost = _record(unit)
        bots = {b["bot"]: b for b in unit["bots"]}
        stronger = bots.get(upper, {})
        seconds = float(unit["seconds"])
        table.add_row(
            f"{short(lower)} vs {short(upper)}",
            Text(str(unit["state"]), style=styles.get(str(unit["state"]), "")),
            won,
            draws,
            lost,
            _pct(stronger.get("first")),
            _signed(stronger.get("rsa")),
            duration(seconds) if seconds else "",
        )
    return table


def _ratings(status: RunStatus) -> RenderableType:
    ratings = status.data.get("ratings")
    if not ratings:
        return Text("")
    table = Table(title="Elo (RandomBot 400)", title_justify="left", expand=False)
    table.add_column("Bot")
    table.add_column("Elo", justify="right")
    for bot, elo in sorted(ratings.items(), key=lambda item: -float(item[1])):
        table.add_row(short(str(bot)), f"{float(elo):,.0f}")
    return table


def render(status: RunStatus | None, target: Path) -> RenderableType:
    if status is None:
        return Text(
            f"No status at {status_path(target)} yet: start the run with a ladder --dir or an "
            "arena --out here",
            style="yellow",
        )
    parts: Sequence[RenderableType] = (
        _header(status),
        Text(""),
        _progress(status),
        Text(""),
        _current(status),
        _pairs(status),
        _ratings(status),
    )
    return Group(*parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena.watch", description="Watch a rating run live."
    )
    parser.add_argument(
        "run",
        type=Path,
        help="a ladder's --dir, an arena run's --out report, or its status file",
    )
    parser.add_argument(
        "--interval", type=float, default=3.0, help="seconds between redraws (default 3)"
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
