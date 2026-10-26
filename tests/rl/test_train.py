"""Training is reproducible from a config file and a seed, and resumes exactly (SPEC §6 M4)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from estimation.bots import make_bot
from estimation.rl.config import TrainConfig, config_from_dict
from estimation.rl.networks import load_policy
from estimation.rl.train import main, record_elo, train
from tests.rl.test_networks import as_version_1

TINY: dict[str, Any] = {
    "seed": 3,
    "workers": 2,
    "record": {"games": 2, "pimc_samples": 2},
    "imitation": {"epochs": 1, "batch": 256},
    "selfplay": {
        "iterations": 3,
        "games": 8,
        "chunk": 4,
        "card_frozen_iterations": 1,
        "snapshot_every": 1,
        "checkpoint_every": 1,
    },
    "nets": {"hidden": 16, "device": "cpu"},
}


def tiny(**changes: Any) -> TrainConfig:
    return config_from_dict({**TINY, **changes})


def tiny_play(workers: int = 2, **selfplay: Any) -> TrainConfig:
    """TINY with changes to its [selfplay] section."""
    return tiny(workers=workers, selfplay={**TINY["selfplay"], **selfplay})


def reuse_records(first_run: Path, out: Path) -> None:
    """Start ``out`` from ``first_run``'s recorded PIMC games, as a staged run reuses M4's."""
    (out / "warmstart").mkdir(parents=True)
    for name in ("pimc.npz", "record.json"):
        shutil.copy(first_run / "warmstart" / name, out / "warmstart" / name)


def same_weights(a: Path, b: Path) -> bool:
    one = torch.load(a, weights_only=True)
    two = torch.load(b, weights_only=True)
    nets = ("card", "bidding")
    return all(
        (one[net] is None and two[net] is None)
        or all(torch.equal(one[net][key], two[net][key]) for key in one[net])
        for net in nets
    )


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("first")
    train(tiny(), out)
    return out


def test_a_run_saves_the_warm_start_every_checkpoint_and_the_final_one(first_run: Path):
    names = [
        entry["name"]
        for entry in json.loads((first_run / "checkpoints/index.json").read_text())["checkpoints"]
    ]
    assert names == ["warm-start", "iter-0001", "iter-0002", "iter-0003", "final"]
    log = [json.loads(line) for line in (first_run / "selfplay/log.jsonl").read_text().splitlines()]
    assert [entry["iteration"] for entry in log] == [1, 2, 3]
    assert [entry["train_card"] for entry in log] == [False, True, True]
    for entry in log:
        memory = entry["memory_mb"]
        assert 50 < memory["trainer_peak"] < 18_000
        assert 50 < memory["worker_peak"] < 18_000
        assert memory["trainer_gpu"] == 0.0  # this test trains on the CPU
    assert same_weights(first_run / "checkpoints/final.pt", first_run / "checkpoints/iter-0003.pt")
    bot = make_bot(f"rl:checkpoint={first_run / 'checkpoints/final.pt'}", 0)
    assert bot.act is not None
    warm = make_bot(
        f"rl:checkpoint={first_run / 'checkpoints/warm-start.pt'}:heuristic_bidding=true", 0
    )
    assert warm.act is not None


def test_the_same_config_and_seed_give_the_same_checkpoints_on_any_worker_count(
    first_run: Path, tmp_path: Path
):
    out = tmp_path / "again"
    train(tiny(workers=1), out)
    for name in ("warm-start", "iter-0001", "iter-0002", "final"):
        assert same_weights(first_run / f"checkpoints/{name}.pt", out / f"checkpoints/{name}.pt")


def test_another_seed_trains_differently(first_run: Path, tmp_path: Path):
    out = tmp_path / "other"
    train(tiny(seed=4), out, until="imitate")
    assert not same_weights(
        first_run / "checkpoints/warm-start.pt", out / "checkpoints/warm-start.pt"
    )


class Crash(RuntimeError):
    pass


def crash_after_two(iteration: int, entry: dict[str, Any]) -> None:
    if iteration == 2:
        raise Crash


def test_an_interrupted_run_resumes_exactly(first_run: Path, tmp_path: Path):
    out = tmp_path / "resumed"
    reuse_records(first_run, out)
    with pytest.raises(Crash):
        train(tiny(), out, on_iteration=crash_after_two)
    assert (out / "selfplay/state.pt").exists()
    assert not (out / "checkpoints/final.pt").exists()
    train(tiny(), out)
    assert same_weights(first_run / "checkpoints/final.pt", out / "checkpoints/final.pt")


@pytest.fixture(scope="module")
def overlap_run(first_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("overlap")
    reuse_records(first_run, out)
    train(tiny_play(overlap=True), out)
    return out


def test_overlap_plays_each_iteration_with_the_nets_from_before_the_last_update(
    first_run: Path, overlap_run: Path
):
    # Iteration 1 plays the starting nets either way; iteration 2 plays them again with overlap
    # (while the learner takes its first update), so the two runs part from there.
    assert same_weights(
        first_run / "checkpoints/iter-0001.pt", overlap_run / "checkpoints/iter-0001.pt"
    )
    assert not same_weights(
        first_run / "checkpoints/iter-0002.pt", overlap_run / "checkpoints/iter-0002.pt"
    )


def test_an_interrupted_overlap_run_resumes_exactly(
    overlap_run: Path, first_run: Path, tmp_path: Path
):
    out = tmp_path / "resumed"
    reuse_records(first_run, out)
    with pytest.raises(Crash):
        train(tiny_play(overlap=True), out, on_iteration=crash_after_two)
    train(tiny_play(overlap=True), out)
    assert same_weights(overlap_run / "checkpoints/final.pt", out / "checkpoints/final.pt")


def test_balanced_overlapping_runs_are_the_same_on_any_worker_count(
    first_run: Path, tmp_path: Path
):
    finals = []
    for workers in (1, 3):
        out = tmp_path / f"workers-{workers}"
        reuse_records(first_run, out)
        finals.append(train(tiny_play(workers, overlap=True, balance=True, chunk=2), out))
    assert same_weights(*finals)


def test_the_yardstick_rates_checkpoints_without_changing_training(first_run: Path, tmp_path: Path):
    out = tmp_path / "yardstick"
    reuse_records(first_run, out)
    reference = tmp_path / "reference.pt"
    shutil.copy(first_run / "checkpoints/final.pt", reference)
    config = tiny_play(
        checkpoint_every=3,
        yardstick_every=2,
        yardstick_sets=2,
        yardstick_pimc_samples=2,
        yardstick_reference=str(reference),
    )
    train(config, out)
    assert same_weights(first_run / "checkpoints/final.pt", out / "checkpoints/final.pt")
    log = [json.loads(line) for line in (out / "selfplay/log.jsonl").read_text().splitlines()]
    assert ["yardstick" in entry for entry in log] == [False, True, False]
    measured = log[1]["yardstick"]
    assert measured["checkpoint"] == "iter-0002.pt"  # saved for the yardstick
    assert (out / "checkpoints/iter-0002.pt").exists()
    assert set(measured["matches"]) == {"pimc", "reference"}
    for match in measured["matches"].values():
        assert match["sets"] == 2
        assert 0.0 <= match["won"]["mean"] <= 1.0
    assert measured["matches"]["reference"]["opponent"] == f"rl:checkpoint={reference}"
    line = json.loads((out / "selfplay/yardstick-reference.json").read_text())
    assert set(line["matches"]) == {"pimc"}


def test_a_missing_yardstick_reference_stops_the_run_before_training(
    first_run: Path, tmp_path: Path
):
    out = tmp_path / "missing"
    reuse_records(first_run, out)
    with pytest.raises(SystemExit, match="is not a file"):
        train(tiny_play(yardstick_reference=str(tmp_path / "nope.pt")), out)
    assert not (out / "selfplay/log.jsonl").exists()


def test_a_folder_refuses_a_different_config(first_run: Path):
    with pytest.raises(SystemExit, match="different config"):
        train(tiny(seed=11), first_run)


def test_ladder_elo_is_saved_with_each_checkpoint(first_run: Path, tmp_path: Path):
    final = first_run / "checkpoints/final.pt"
    ladder = tmp_path / "ladder.json"
    ladder.write_text(json.dumps({"ratings": {"random": 400.0, f"rl:checkpoint={final}": 1234.5}}))
    rated = dict(record_elo(first_run, ladder))
    assert rated["final"] == {f"rl:checkpoint={final}": 1234.5}
    assert rated["iter-0003"] == rated["final"]  # final.pt is a copy of it
    assert rated["warm-start"] == {}
    index = json.loads((first_run / "checkpoints/index.json").read_text())["checkpoints"]
    assert next(e for e in index if e["name"] == "final")["elo"] == {
        f"rl:checkpoint={final}": 1234.5
    }


def test_the_command_line_trains_from_a_config_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = tmp_path / "tiny.toml"
    config.write_text(
        "workers = 1\n[record]\ngames = 1\npimc_samples = 2\n[imitation]\nepochs = 1\n"
        '[nets]\nhidden = 16\ndevice = "cpu"\n'
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--seed",
                "6",
                "--out",
                str(tmp_path / "cli"),
                "--until",
                "imitate",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip().endswith("warm-start.pt")
    saved = json.loads((tmp_path / "cli/config.json").read_text())
    assert saved["seed"] == 6


def test_a_run_can_continue_from_a_trained_checkpoint(first_run: Path, tmp_path: Path):
    """selfplay.start_from: no imitation stage, and both nets start as the checkpoint's, here a
    version 1 checkpoint widened for the void announcements. Held still (card play frozen, no
    bidding steps), they come out exactly as they went in."""
    start = tmp_path / "start.pt"
    shutil.copy(first_run / "checkpoints" / "final.pt", start)
    as_version_1(start)
    out = tmp_path / "run"
    reuse_records(first_run, out)
    config = tiny_play(start_from=str(start), iterations=1, card_frozen_iterations=1, bid_lr=0.0)
    train(config, out)
    assert not (out / "checkpoints" / "warm-start.pt").exists()
    final, begun = load_policy(out / "checkpoints" / "final.pt"), load_policy(start)
    assert begun.bidding is not None
    assert final.bidding is not None
    for net, before in ((final.card, begun.card), (final.bidding, begun.bidding)):
        after = net.state_dict()
        assert all(torch.equal(after[key], value) for key, value in before.state_dict().items())


def test_a_missing_start_checkpoint_stops_the_run_before_any_games(tmp_path: Path):
    with pytest.raises(SystemExit, match="is not a file"):
        train(tiny_play(start_from=str(tmp_path / "nope.pt")), tmp_path / "run")
    assert not (tmp_path / "run" / "warmstart").exists()
