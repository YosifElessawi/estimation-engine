"""The live view of a rating run: arena and ladder runs keep a status file that the view reads."""

import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console

from estimation.arena.__main__ import main as arena_main
from estimation.arena.duplicate import SetResult, run_arena
from estimation.arena.ladder import main as ladder_main
from estimation.arena.status import STATUS_FILE, StatusWriter, status_path
from estimation.arena.watch import load, main, render

TINY = "random,heuristic:noise=0.5"
PAIR = ("random", "heuristic", "random", "heuristic")


def text_of(target: Path, now: float | None = None) -> str:
    console = Console(file=io.StringIO(), width=400, color_system=None)
    console.print(render(load(target, now=now), target))
    return console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # a StringIO, set above


def test_a_ladder_keeps_a_status_the_view_reads(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert ladder_main(["--bots", TINY, "--sets", "2", "--dir", str(tmp_path)]) == 0
    status = json.loads((tmp_path / STATUS_FILE).read_text())
    assert status["kind"] == "ladder"
    assert status["finished"]
    (unit,) = status["units"]
    assert (unit["state"], unit["sets_done"]) == ("done", 2)
    record = unit["record"]
    assert record["a_wins"] + record["draws"] + record["b_wins"] == 2
    assert status["ratings"]["random"] == 400
    shown = text_of(tmp_path)
    assert "Elo ladder" in shown
    assert "finished" in shown
    assert "random vs heuristic:noise=0.5" in shown
    assert "Elo (RandomBot 400)" in shown


def test_a_reused_pair_shows_as_reused(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    args = ["--bots", TINY, "--sets", "2", "--dir", str(tmp_path)]
    assert ladder_main(args) == 0
    first = json.loads((tmp_path / STATUS_FILE).read_text())["units"][0]["record"]
    assert ladder_main(args) == 0
    (unit,) = json.loads((tmp_path / STATUS_FILE).read_text())["units"]
    assert unit["state"] == "reused"
    assert unit["record"] == first
    assert "reused" in text_of(tmp_path)


def test_an_arena_run_writes_its_status_next_to_its_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "duel.json"
    lineup = "heuristic,random,heuristic,random"
    assert arena_main(["--lineup", lineup, "--sets", "2", "--out", str(out)]) == 0
    assert status_path(out) == tmp_path / "duel.status.json"
    status = json.loads(status_path(out).read_text())
    assert (status["kind"], status["finished"]) == ("arena", True)
    (unit,) = status["units"]
    assert unit["sets_done"] == 2
    assert {bot["bot"] for bot in unit["bots"]} == {"heuristic", "random"}
    assert "Arena run" in text_of(out)  # the view finds the status from the report's path


def test_the_status_counts_every_set_as_it_is_played(tmp_path: Path):
    path = tmp_path / STATUS_FILE
    writer = StatusWriter(path, "ladder", [("random vs heuristic", PAIR)], 3, 0, 1, ["test"])
    writer.start(0)
    seen: list[int] = []

    def on_set(result: SetResult) -> None:
        writer.set_done(result)
        seen.append(json.loads(path.read_text())["units"][0]["sets_done"])

    run_arena(PAIR, 3, 0, on_set=on_set)
    assert seen == [1, 2, 3]


def test_a_running_view_shows_the_pair_in_play_and_the_time_left(tmp_path: Path):
    writer = StatusWriter(
        tmp_path / STATUS_FILE, "ladder", [("random vs heuristic", PAIR)], 4, 0, 1, ["test"]
    )
    writer.start(0)
    run_arena(PAIR, 2, 0, on_set=writer.set_done)  # 2 of the pair's 4 sets
    shown = text_of(tmp_path)
    assert "running" in shown
    assert "Now playing: random, heuristic   set 2 of 4" in shown
    assert "stronger won" in shown
    assert "left" in shown


def test_a_run_whose_process_is_gone_shows_as_stopped(tmp_path: Path):
    path = tmp_path / STATUS_FILE
    StatusWriter(path, "ladder", [("random vs heuristic", PAIR)], 2, 0, 1, ["test"])
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], check=True,
                          capture_output=True, text=True)  # fmt: skip
    data = json.loads(path.read_text())
    data["pid"] = int(gone.stdout)
    path.write_text(json.dumps(data))
    assert "its process is gone" in text_of(tmp_path)


def test_a_run_without_updates_shows_as_stalled(tmp_path: Path):
    StatusWriter(tmp_path / STATUS_FILE, "ladder", [("a vs b", PAIR)], 2, 0, 1, ["test"])
    assert "no update for 10 min" in text_of(tmp_path, now=time.time() + 600)


def test_a_stopped_run_shows_why(tmp_path: Path):
    writer = StatusWriter(tmp_path / STATUS_FILE, "ladder", [("a vs b", PAIR)], 2, 0, 1, ["x"])
    writer.finish(error="stopped: KeyboardInterrupt()")
    assert "stopped: KeyboardInterrupt()" in text_of(tmp_path)


def test_the_view_waits_for_a_status(tmp_path: Path):
    assert "No status at" in text_of(tmp_path)


def test_once_prints_one_frame(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    StatusWriter(tmp_path / STATUS_FILE, "ladder", [("a vs b", PAIR)], 2, 0, 1, ["test"])
    assert main([str(tmp_path), "--once"]) == 0
    assert "Elo ladder" in capsys.readouterr().out
