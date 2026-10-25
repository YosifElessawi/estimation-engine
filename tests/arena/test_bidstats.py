"""Bidding statistics per bot (design section 6)."""

import json
from pathlib import Path

import pytest

from estimation.arena.bidstats import RoundRow, SeatRow, collect, format_stats, main, summarize

PLAIN = SeatRow(estimate=3, tricks=3, won=True, dash=False, caller=False, with_=False, super_=False)


def test_summarize_counts_skips_dash_calls_and_super_calls():
    skipped = RoundRow(fixed=False, skipped=True, trump_changed=False, seats=(PLAIN,) * 4)
    super_call = SeatRow(
        estimate=8, tricks=7, won=False, dash=False, caller=True, with_=False, super_=True
    )
    dash = SeatRow(
        estimate=0, tricks=0, won=True, dash=True, caller=False, with_=False, super_=False
    )
    played = RoundRow(
        fixed=False, skipped=False, trump_changed=False, seats=(super_call, dash, PLAIN, PLAIN)
    )
    fixed_caller = SeatRow(
        estimate=5, tricks=5, won=True, dash=False, caller=True, with_=False, super_=False
    )
    fixed = RoundRow(
        fixed=True, skipped=False, trump_changed=False, seats=(PLAIN, PLAIN, fixed_caller, PLAIN)
    )
    stats = summarize([skipped, played, fixed])
    assert stats.open_rounds == 2
    assert stats.skipped_share == 0.5
    assert stats.dash_share == pytest.approx(1 / 8)
    assert stats.dash_made == 1.0
    assert stats.with_round_share == 0.0
    assert stats.contract_share == {"8+": 1.0}
    assert stats.contract_made == {"8+": 0.0}
    assert stats.super_share == 1.0
    assert stats.super_made == 0.0
    assert stats.fixed_rounds == 1
    assert stats.fixed_super_share == 0.0
    assert stats.caller_made == 0.0


def test_collect_plays_whole_games_of_four_copies():
    rows = collect("heuristic", 2, 0, 1)
    assert len(rows) >= 36
    stats = summarize(rows)
    assert sum(stats.contract_share.values()) == pytest.approx(1.0)
    assert "heuristic" in format_stats("heuristic", stats)


def test_the_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "stats.json"
    assert main(["--bot", "heuristic", "--games", "1", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["open_rounds"] == 13
    assert "Caller makes exactly" in capsys.readouterr().out
