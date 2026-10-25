"""The Elo ladder: bots ordered by strength, each neighbouring pair rated in its own duplicate run,
and one Bradley-Terry fit over all of them with RandomBot fixed at 400."""

import json
from pathlib import Path

import pytest

from estimation.arena import ladder
from estimation.arena.ladder import DEFAULT_LADDER, ladder_pairs, main

TINY = "random,heuristic:noise=0.5,heuristic"


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    assert main(list(args)) == 0
    return capsys.readouterr().out


def test_the_default_ladder_climbs_from_random_to_full_pimc():
    assert DEFAULT_LADDER[0] == "random"
    assert "heuristic" in DEFAULT_LADDER
    assert DEFAULT_LADDER[-1] == "pimc"
    assert len(set(DEFAULT_LADDER)) == len(DEFAULT_LADDER)


def test_pairs_are_the_neighbours_then_any_extras():
    pairs = ladder_pairs(["a", "b", "c"], [("a", "c")])
    assert pairs == [("a", "b"), ("b", "c"), ("a", "c")]
    assert ladder_pairs(["a", "b"], [("a", "b")]) == [("a", "b")]


def test_rates_every_bot_from_one_run_per_pair(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = run_cli(capsys, "--bots", TINY, "--sets", "2", "--dir", str(tmp_path))
    files = sorted((tmp_path / "pairs").glob("*.json"))
    assert len(files) == 2
    lineups = {tuple(json.loads(f.read_text())["report"]["lineup"]) for f in files}
    assert ("random", "heuristic:noise=0.5", "random", "heuristic:noise=0.5") in lineups
    for bot in TINY.split(","):
        assert bot in out
    assert "400 (anchor)" in out
    summary = json.loads((tmp_path / "ladder.json").read_text())
    assert set(summary["ratings"]) == set(TINY.split(","))
    assert "Stronger 1st" in out
    assert all(0 <= pair["stronger_first_place"] <= 1 for pair in summary["pairs"])
    assert summary["ratings"]["random"] == 400


def test_finished_pairs_are_reused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    first = run_cli(capsys, "--bots", TINY, "--sets", "2", "--dir", str(tmp_path))

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished pair was played again")

    monkeypatch.setattr(ladder, "run_arena", refuse)
    second = run_cli(capsys, "--bots", TINY, "--sets", "2", "--dir", str(tmp_path))
    assert second.split("Run:")[0] == first.split("Run:")[0]


def test_a_pair_is_played_again_when_its_settings_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_cli(capsys, "--bots", TINY, "--sets", "2", "--dir", str(tmp_path))
    run_cli(capsys, "--bots", TINY, "--sets", "3", "--dir", str(tmp_path))
    sets = {json.loads(f.read_text())["report"]["sets"] for f in (tmp_path / "pairs").glob("*")}
    assert sets == {3}


def test_extra_pairs_join_the_fit(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    run_cli(
        capsys, "--bots", TINY, "--sets", "2", "--dir", str(tmp_path), "--extra", "random,heuristic"
    )
    assert len(list((tmp_path / "pairs").glob("*.json"))) == 3


@pytest.mark.parametrize(
    "args",
    [
        ["--bots", "random,heuristic,random"],
        ["--bots", "random"],
        ["--bots", TINY, "--extra", "random,pimc"],
        ["--bots", "random,wizard"],
    ],
)
def test_bad_ladders_are_rejected(args: list[str], tmp_path: Path):
    with pytest.raises(SystemExit):
        main([*args, "--sets", "1", "--dir", str(tmp_path)])
