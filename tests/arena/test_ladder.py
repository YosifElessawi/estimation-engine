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


def test_a_checkpoint_path_makes_a_flat_pair_file_name():
    pair_file = ladder._pair_file
    path = pair_file(Path("out"), "heuristic", "rl:checkpoint=runs/m4/final.pt")
    assert path == Path("out/pairs/heuristic--rl_checkpoint-runs_m4_final.pt.json")
    # Names without paths are unchanged, so saved pair reports are still found.
    assert pair_file(Path("out"), "pimc:samples=2", "pimc") == Path(
        "out/pairs/pimc_samples-2--pimc.json"
    )


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
    for name in ("RandomBot", "HeuristicBot, 50% random moves", "HeuristicBot"):
        assert name in out
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


def test_a_pair_from_other_code_is_played_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    run_cli(capsys, "--bots", TINY, "--sets", "1", "--dir", str(tmp_path))
    monkeypatch.setattr(ladder, "code_version", lambda: "other-code")
    run_cli(capsys, "--bots", TINY, "--sets", "1", "--dir", str(tmp_path))
    versions = {
        json.loads(f.read_text())["report"]["code_version"] for f in (tmp_path / "pairs").glob("*")
    }
    assert versions == {"other-code"}


def test_a_ladder_pair_saves_its_games(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from estimation.arena.games import read_games, replay
    from estimation.engine import RulesConfig

    run_cli(capsys, "--bots", "random,heuristic", "--sets", "1", "--dir", str(tmp_path))
    (path,) = (tmp_path / "games").glob("*.jsonl.gz")
    header, records, _ = read_games(path)
    assert header is not None
    assert header.lineup == ("random", "heuristic", "random", "heuristic")
    assert len(records) == 4
    assert all(replay(record, RulesConfig()) is not None for record in records)


def test_a_pair_without_saved_games_is_played_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    run_cli(capsys, "--bots", "random,heuristic", "--sets", "1", "--dir", str(tmp_path))
    for path in (tmp_path / "games").glob("*"):
        path.unlink()
    run_cli(capsys, "--bots", "random,heuristic", "--sets", "1", "--dir", str(tmp_path))
    assert len(list((tmp_path / "games").glob("*.jsonl.gz"))) == 1


def test_the_v2_preset_lists_26_bots_and_10_extras():
    preset = ladder.PRESETS["v2"]
    assert len(preset.bots) == 26
    assert len(set(preset.bots)) == 26
    assert preset.bots[0] == "random"
    assert preset.bots[-1] == "yosif"
    assert len(preset.extras) == 10
    assert all(a in preset.bots and b in preset.bots for a, b in preset.extras)
    assert ("pimc", "yosif") in preset.extras
    assert ("heuristic", "yosif") in preset.extras
    assert preset.dir == Path("runs/ladder-v2")
    assert (preset.sets, preset.seed) == (200, 0)


def test_the_report_shows_named_tables_with_elo_ranges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = run_cli(capsys, "--bots", TINY, "--sets", "4", "--dir", str(tmp_path))
    leaderboard, rungs = out.index("Leaderboard"), out.index("Measuring rungs")
    assert leaderboard < rungs
    assert "RL checkpoints" not in out  # no checkpoint was rated, so no empty table
    assert out.index("HeuristicBot, 50% random moves") > rungs
    assert "95% range" in out
    summary = json.loads((tmp_path / "ladder.json").read_text())
    assert summary["names"]["heuristic:noise=0.5"] == "HeuristicBot, 50% random moves"
    assert summary["groups"]["random"] == ["Leaderboard"]
    low, high = summary["ranges"]["heuristic"]
    assert low <= summary["ratings"]["heuristic"] <= high


def test_a_pair_whose_games_file_is_from_other_code_is_played_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    import dataclasses

    from estimation.arena.games import read_games, write_games

    run_cli(capsys, "--bots", "random,heuristic", "--sets", "1", "--dir", str(tmp_path))
    (path,) = (tmp_path / "games").glob("*.jsonl.gz")
    header, records, _ = read_games(path)
    assert header is not None
    write_games(path, dataclasses.replace(header, code_version="older"), records)

    def played(*args: object, **kwargs: object) -> None:
        raise AssertionError("played again")

    monkeypatch.setattr(ladder, "run_arena", played)
    with pytest.raises(AssertionError, match="played again"):
        run_cli(capsys, "--bots", "random,heuristic", "--sets", "1", "--dir", str(tmp_path))
