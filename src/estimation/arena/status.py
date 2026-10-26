"""The live status of a rating run, read by ``python -m estimation.arena.watch``.

Arena and ladder runs rewrite one small JSON file after every duplicate set: what is being
played, how far it got, the results so far and, once the run ends, the ratings. The file is
replaced atomically, so a reader never sees half of it. A ladder keeps it as ``status.json`` in
its folder; an arena run writes it next to its ``--out`` report.

Each *unit* is one duplicate run: the arena's lineup, or one ladder pair. Its results so far are
each bot's mean RSA per Bola and first-place share and, for a lineup of two bots, the sets each
side won.
"""

import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .duplicate import SetResult
from .ratings import pair_records, set_first_places, set_rsa

STATUS_FILE = "status.json"


def status_path(target: Path) -> Path:
    """The status file for ``target``: a folder's ``status.json``, an arena report's
    ``.status.json`` sibling, or ``target`` itself when it already names a status file."""
    if target.is_dir():
        return target / STATUS_FILE
    if target.name.endswith(".status.json") or target.name == STATUS_FILE:
        return target
    return target.with_suffix(".status.json")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def unit_results(results: Sequence[SetResult], lineup: Sequence[str]) -> dict[str, Any]:
    """Each bot's mean RSA and first-place share over ``results`` and, for two bots, the sets
    each side won."""
    bots = list(dict.fromkeys(lineup))
    rsa: dict[str, list[float]] = {bot: [] for bot in bots}
    first: dict[str, list[float]] = {bot: [] for bot in bots}
    for result in results:
        for bot, value in set_rsa(result, lineup).items():
            rsa[bot].append(value)
        for bot, value in set_first_places(result, lineup).items():
            first[bot].append(value)
    summary: dict[str, Any] = {
        "bots": [
            {
                "bot": bot,
                "rsa": sum(rsa[bot]) / len(rsa[bot]) if rsa[bot] else None,
                "first": sum(first[bot]) / len(first[bot]) if first[bot] else None,
            }
            for bot in bots
        ],
        "record": None,
    }
    if len(bots) == 2 and results:
        (record,) = pair_records(results, lineup)
        summary["record"] = {
            "a": record.a,
            "b": record.b,
            "a_wins": record.a_wins,
            "draws": record.draws,
            "b_wins": record.b_wins,
        }
    return summary


class StatusWriter:
    """Keeps a run's status file up to date. Units are the duplicate runs in play order, each a
    name and its lineup."""

    def __init__(
        self,
        path: Path,
        kind: str,
        units: Sequence[tuple[str, Sequence[str]]],
        sets: int,
        seed: int,
        workers: int,
        command: Sequence[str],
    ) -> None:
        self.path = path
        self._start = time.perf_counter()
        self._unit_start = self._start
        self._current = 0
        self._results: list[SetResult] = []
        self.status: dict[str, Any] = {
            "kind": kind,
            "command": " ".join(command),
            "pid": os.getpid(),
            "started": _now(),
            "updated": _now(),
            "sets": sets,
            "seed": seed,
            "workers": workers,
            "finished": False,
            "error": None,
            "ratings": None,
            "seconds": 0.0,
            "units": [
                {
                    "name": name,
                    "lineup": list(lineup),
                    "state": "pending",
                    "sets_done": 0,
                    "seconds": 0.0,
                    **unit_results([], lineup),
                }
                for name, lineup in units
            ],
        }
        self._write()

    def _unit(self, index: int) -> dict[str, Any]:
        unit: dict[str, Any] = self.status["units"][index]
        return unit

    def start(self, index: int) -> None:
        """Unit ``index`` starts playing."""
        self._current = index
        self._results = []
        self._unit_start = time.perf_counter()
        self._unit(index)["state"] = "running"
        self._write()

    def set_done(self, result: SetResult) -> None:
        """One more set of the running unit is played."""
        unit = self._unit(self._current)
        self._results.append(result)
        unit["sets_done"] = len(self._results)
        unit["seconds"] = time.perf_counter() - self._unit_start
        unit.update(unit_results(self._results, unit["lineup"]))
        self._write()

    def done(
        self,
        index: int,
        reused: bool = False,
        sets_done: int | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Unit ``index`` is over: played, or ``reused`` from an earlier run. ``summary``, when
        given, replaces its results (bots and record, as ``unit_results`` makes them)."""
        unit = self._unit(index)
        unit["state"] = "reused" if reused else "done"
        if sets_done is not None:
            unit["sets_done"] = sets_done
        if summary is not None:
            unit.update(summary)
        if not reused:
            unit["seconds"] = time.perf_counter() - self._unit_start
        self._write()

    def finish(self, ratings: dict[str, float] | None = None, error: str | None = None) -> None:
        """The run is over: with its ratings if it fitted any, or with the error that stopped
        it."""
        self.status["finished"] = error is None
        self.status["error"] = error
        self.status["ratings"] = ratings
        self.status["seconds"] = time.perf_counter() - self._start
        self._write()

    def _write(self) -> None:
        self.status["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.status, indent=1, allow_nan=False) + "\n")
        tmp.replace(self.path)
