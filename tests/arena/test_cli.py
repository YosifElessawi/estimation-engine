"""The arena CLI (SPEC §6 M2): ``python -m estimation.arena --lineup ... --sets ... --seed ...``."""

import json
import math
from pathlib import Path

import pytest

from estimation.arena.__main__ import main

LINEUP = "heuristic,random,heuristic,random"


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    assert main(list(args)) == 0
    return capsys.readouterr().out


def test_prints_rsa_with_a_confidence_interval_and_elo(capsys: pytest.CaptureFixture[str]):
    out = run_cli(capsys, "--lineup", LINEUP, "--sets", "3", "--seed", "0")
    assert "RSA" in out
    assert "95% CI" in out
    assert "Elo" in out
    assert "heuristic" in out
    assert "800" in out


def test_the_same_seed_gives_the_same_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    out1 = run_cli(capsys, "--lineup", LINEUP, "--sets", "3", "--seed", "5", "--out", str(first))
    out2 = run_cli(
        capsys,
        "--lineup",
        LINEUP,
        "--sets",
        "3",
        "--seed",
        "5",
        "--workers",
        "2",
        "--out",
        str(second),
    )
    assert out1.split("Run:")[0] == out2.split("Run:")[0]
    report1 = json.loads(first.read_text())
    report2 = json.loads(second.read_text())
    assert report1["report"] == report2["report"]
    assert report1["run"]["workers"] == 1
    assert report2["run"]["workers"] == 2


def test_elo_can_include_earlier_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    earlier = tmp_path / "earlier.json"
    run_cli(capsys, "--lineup", LINEUP, "--sets", "2", "--seed", "1", "--out", str(earlier))
    later = tmp_path / "later.json"
    lineup = "heuristic:risk_aversion=0,heuristic,heuristic,heuristic"
    run_cli(
        capsys,
        "--lineup",
        lineup,
        "--sets",
        "2",
        "--seed",
        "2",
        "--elo-from",
        str(earlier),
        "--out",
        str(later),
    )
    elo = json.loads(later.read_text())["report"]["elo"]
    assert elo["ratings"]["random"] == 800
    assert set(elo["ratings"]) == {"random", "heuristic", "heuristic:risk_aversion=0"}
    assert elo["sources"] == [str(earlier)]


def test_a_clean_sweep_is_reported_as_a_lower_bound(capsys: pytest.CaptureFixture[str]):
    out = run_cli(capsys, "--lineup", LINEUP, "--sets", "3", "--seed", "0")
    assert "heuristic won all 3 sets against random" in out
    assert "at least" in out


def test_the_elo_prior_is_configurable(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "prior.json"
    run_cli(capsys, "--lineup", LINEUP, "--sets", "2", "--elo-prior", "3", "--out", str(out))
    elo = json.loads(out.read_text())["report"]["elo"]
    assert elo["prior_draws"] == 3
    assert elo["ratings"]["heuristic"] == pytest.approx(800 + 400 * math.log10(3.5 / 1.5))


def test_a_sweep_without_a_prior_reports_elo_as_not_fitted(capsys: pytest.CaptureFixture[str]):
    out = run_cli(capsys, "--lineup", LINEUP, "--sets", "2", "--elo-prior", "0")
    assert "Elo not fitted" in out


def test_rejects_a_lineup_without_four_bots(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        main(["--lineup", "heuristic,random", "--sets", "1"])
    assert "4 bots" in capsys.readouterr().err


def test_rejects_an_unknown_bot(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        main(["--lineup", "heuristic,random,heuristic,wizard", "--sets", "1"])
    assert "unknown bot" in capsys.readouterr().err
