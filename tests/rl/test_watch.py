"""The live view of a training run reads only the run's files."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from estimation.rl.watch import load, main, render, sparkline


def match(won: float, rsa: float, opponent: str) -> dict[str, Any]:
    return {
        "opponent": opponent,
        "sets": 40,
        "won": {"mean": won, "low": won - 0.1, "high": won + 0.1},
        "rsa": {"mean": rsa, "low": rsa - 20, "high": rsa + 20},
        "first": {"mean": 0.3, "low": 0.2, "high": 0.4},
    }


def entry(iteration: int, yardstick: bool = False) -> dict[str, Any]:
    found: dict[str, Any] = {
        "iteration": iteration,
        "seconds": 10.0,
        "play_seconds": 2.0,
        "learn_seconds": 7.5,
        "train_card": True,
        "lr_factor": 0.9,
        "anchor_weight": 0.8,
        "bid_loss": 0.08,
        "card_value_loss": 0.06,
        "card_q_loss": 0.06,
        "anchor_loss": 0.47,
        "memory_mb": {"trainer_peak": 2048.0, "trainer_gpu": 1024.0, "worker_peak": 512.0},
        "results": {
            "eval": {"games": 32, "rsa": 90.0 + iteration, "first": 0.6, "skipped": 0.0},
            "self": {"games": 168, "rsa": -50.0, "first": 0.2, "skipped": 0.1},
        },
    }
    if yardstick:
        found["yardstick"] = {
            "checkpoint": f"iter-{iteration:04d}.pt",
            "seconds": 60.0,
            "matches": {
                "pimc": match(0.3 + iteration / 100, -40.0, "pimc:samples=8"),
                "reference": match(0.7, 25.0, "rl:checkpoint=runs/ref/m4-final.pt"),
            },
        }
    return found


@pytest.fixture
def run(tmp_path: Path) -> Path:
    (tmp_path / "selfplay").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"selfplay": {"iterations": 100}}))
    lines = [entry(i, yardstick=i % 25 == 0) for i in range(1, 51)]
    (tmp_path / "selfplay/log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    reference = {"checkpoint": "runs/ref/m4-final.pt", "matches": {"pimc": match(0.35, -30, "x")}}
    (tmp_path / "selfplay/yardstick-reference.json").write_text(json.dumps(reference))
    return tmp_path


def text_of(path: Path) -> str:
    console = Console(file=io.StringIO(), width=120, color_system=None)
    console.print(render(load(path, now=0.0), path))
    return console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # a StringIO, set above


def test_the_view_shows_progress_the_yardstick_and_the_greedy_results(run: Path):
    shown = text_of(run)
    assert "Loop 50 of 100  (50%)" in shown
    assert "about 8 min left" in shown  # 50 loops of 10 s
    assert "vs PIMC-8: won" in shown
    assert "vs m4-final (reference): won" in shown
    assert "55% (45-65)" in shown  # loop 25 against PIMC
    assert "70% (60-80)" in shown  # against the reference
    assert "m4-final" in shown  # the reference's own line against PIMC
    assert "first place 60%" in shown
    assert "anchor weight 0.80, learning rate x0.90" in shown
    assert "learning 7.5 s" in shown
    assert "each worker up to 0.5 GB" in shown


def test_a_finished_run_and_a_missing_run(run: Path, tmp_path: Path):
    (run / "checkpoints").mkdir()
    (run / "checkpoints/final.pt").write_bytes(b"")
    assert "finished" in text_of(run)
    empty = tmp_path / "nothing"
    assert "No training run" in text_of(empty)


def test_once_prints_one_frame(run: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(run), "--once"]) == 0
    assert "Loop 50 of 100" in capsys.readouterr().out


def test_sparklines_scale_to_the_range_and_width():
    assert sparkline([]) == ""
    assert sparkline([0, 1]) == "▁█"
    assert len(sparkline(list(range(100)), width=40)) == 40
    assert sparkline([5, 5, 5]) == "▁▁▁"
