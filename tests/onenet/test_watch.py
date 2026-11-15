"""Step 5's self-play page: what it shows from a run's files."""

import io
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from estimation.onenet.watch import render, sparkline


def row(loop: int, **extra: Any) -> dict[str, Any]:
    return {
        "loop": loop,
        "minutes_left": 120.0,
        "decisions_per_s_now": 95000.0,
        "bolas_per_s": 78.0,
        "bolas_total": 400 * loop,
        "distance_from_copy": 0.001 * loop,
        "anchor_weight": 1.0 - loop / 1000,
        "holders_named": 0.44,
        "final_mse": 0.08,
        "caller_made": 0.41,
        **extra,
    }


def shown(run: Path, checkups: Path | None = None) -> str:
    console = Console(file=io.StringIO(), width=160, color_system=None)
    console.print(render(run, checkups, now=0.0))
    return console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # the StringIO above


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "selfplay"
    run.mkdir()
    config = {"loops": 5000, "batch": 4096, "steps": 128, "start": "copy", "sharpen": 4.0}
    (run / "config.json").write_text(json.dumps({**config, "anchor_loops": 1000}))
    rows = [
        row(1, king_vs_copy=0.25, place_vs_copy=0.0),
        row(2),
        row(100, king_vs_copy=0.27, place_vs_copy=0.05),
    ]
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"loop": 10')
    return run


def test_the_page_shows_progress_strength_guards_and_practice(tmp_path: Path):
    text = shown(make_run(tmp_path))
    assert "100 of 5,000 (2%)" in text
    assert "2 h 00 min left" in text
    assert "40,000 Bolas" in text
    assert "27.0%" in text  # the last check's King rate
    assert "+2.0 since loop 1" in text
    assert "best +0.050 at loop 100" in text
    assert "fades to 0 by loop 1,000" in text
    assert "44%" in text
    assert "no new loop" not in text  # now=0: the log is not stale


def test_the_page_lists_the_check_ups_against_hybrid_v4(tmp_path: Path):
    checkups = tmp_path / "checkups"
    checkups.mkdir()
    bots = [
        {
            "name": "onenet:net=x.npz",
            "place": {"mean": -0.234},
            "first_place": {"mean": 0.139},
            "rsa": {"mean": -42.8, "low": -48.7, "high": -36.9},
        },
        {
            "name": "hybrid:checkpoint=y",
            "place": {"mean": 0.234},
            "first_place": {"mean": 0.361},
            "rsa": {"mean": 42.8, "low": 36.9, "high": 48.7},
        },
    ]
    (checkups / "copy.json").write_text(json.dumps({"report": {"bots": bots}}))
    (checkups / "loop-00500.status.json").write_text(json.dumps({"finished": False}))
    text = shown(make_run(tmp_path), checkups)
    assert "Check-ups against Hybrid v4" in text
    assert "-0.234" in text
    assert "-42.8 (-48.7 to -36.9)" in text
    assert "loop-00500" in text
    assert "playing" in text


def test_a_run_with_no_loops_yet_says_so(tmp_path: Path):
    text = shown(tmp_path)
    assert "No self-play in" in text
    assert "first loop" in text


def test_sparklines_scale_to_their_own_range():
    assert sparkline([0.0, 0.5, 1.0]) == "▁▅█"
    assert sparkline([2.0, 2.0]) == "▄▄"
    assert sparkline([]) == ""
