"""Training is reproducible from a config file and a seed, and resumes exactly (SPEC §6 M4)."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from estimation.bots import make_bot
from estimation.rl.config import TrainConfig, config_from_dict
from estimation.rl.train import main, record_elo, train

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


def test_an_interrupted_run_resumes_exactly(first_run: Path, tmp_path: Path):
    out = tmp_path / "resumed"
    (out / "warmstart").mkdir(parents=True)
    for name in ("pimc.npz", "record.json"):
        shutil.copy(first_run / "warmstart" / name, out / "warmstart" / name)

    def crash_after_two(iteration: int, entry: dict[str, Any]) -> None:
        if iteration == 2:
            raise Crash

    with pytest.raises(Crash):
        train(tiny(), out, on_iteration=crash_after_two)
    assert (out / "selfplay/state.pt").exists()
    assert not (out / "checkpoints/final.pt").exists()
    train(tiny(), out)
    assert same_weights(first_run / "checkpoints/final.pt", out / "checkpoints/final.pt")


def test_a_folder_refuses_a_different_config(first_run: Path):
    with pytest.raises(SystemExit, match="different config"):
        train(tiny(seed=11), first_run)


def test_ladder_elo_is_saved_with_each_checkpoint(first_run: Path, tmp_path: Path):
    final = first_run / "checkpoints/final.pt"
    ladder = tmp_path / "ladder.json"
    ladder.write_text(json.dumps({"ratings": {"random": 400.0, f"rl:checkpoint={final}": 1234.5}}))
    rated = dict(record_elo(first_run, ladder))
    assert rated["final"] == {f"rl:checkpoint={final}": 1234.5}
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
