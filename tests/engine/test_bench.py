"""Smoke test for the SPEC §5.5 benchmark. The speed itself is measured and reported by hand,
since a test run shares the machine with other work."""

import pytest

from estimation.engine import bench


def test_bench_rounds_and_bolas_report_positive_rates():
    assert bench.bench_rounds(rounds=36, seed=1) > 0
    assert bench.bench_bolas(bolas=1, seed=1) > 0


def test_bench_cli_prints_the_rate_and_passes_a_low_target(capsys: pytest.CaptureFixture[str]):
    code = bench.main(["--rounds", "36", "--bolas", "1", "--repeat", "1", "--target", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "random-play rounds" in out
    assert "rounds/s" in out


def test_bench_cli_exits_nonzero_below_the_target():
    assert bench.main(["--rounds", "18", "--bolas", "1", "--repeat", "1", "--target", "1e12"]) == 1
