"""Expert iteration for the Hybrid (Hybrid plan step 2): recording, copying and the refit."""

import dataclasses
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from rich.console import Console

from estimation.bots.hybrid.bot import equivalent_runs
from estimation.rl.config import config_from_dict
from estimation.rl.encoder import NUM_FEATURES, OFFSETS, VOIDS
from estimation.rl.expert import (
    GAME_ID,
    ExpertConfig,
    PartTask,
    RunConfig,
    config_of,
    copy_guesses,
    copy_stage,
    load_run_config,
    record_game,
    record_part,
    refit_config,
    regret,
    run,
)
from estimation.rl.expert_watch import render
from estimation.rl.guesses import NO_GUESS
from estimation.rl.imitation import ImitationData
from estimation.rl.networks import GuessNet, load_policy
from estimation.rl.record import CardRecords, concat_records
from estimation.rl.record import record_game as record_pimc
from tests.rl.helpers import make_checkpoint

UNSEEN = OFFSETS["unseen"]
TRAIN: dict[str, Any] = {
    "seed": 4,
    "workers": 2,
    "selfplay": {"iterations": 1, "games": 4, "chunk": 2, "eval_games": 2, "guess_lr": 1e-3},
    "nets": {"hidden": 32, "device": "cpu"},
}


def tiny(checkpoint: Path, **expert: Any) -> RunConfig:
    settings = {
        "start_from": str(checkpoint),
        "rounds": 1,
        "games": 2,
        "part_games": 1,
        "label_share": 0.2,
        "play_samples": 2,
        "play_top_k": 2,
        "teacher_samples": 3,
        "copy_epochs": 2,
        "copy_batch": 64,
        "holdout": 0.5,
        **expert,
    }
    return config_of({**TRAIN, "expert": settings})


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_checkpoint(tmp_path_factory.mktemp("nets") / "start.pt", seed=2)


@pytest.fixture(scope="module")
def recorded(checkpoint: Path) -> Any:
    return record_part(PartTask(1, 0, 2, str(checkpoint), tiny(checkpoint)))


def test_the_committed_config_starts_from_rl_v3_and_refits_bidding_only():
    config = load_run_config(Path(__file__).resolve().parents[2] / "configs" / "expert.toml")
    assert config.expert.start_from == "runs/rl-v3/checkpoints/final.pt"
    assert config.expert.teacher_samples > config.expert.play_samples  # a deeper search
    refit = refit_config(config, Path("round-1/copy.pt")).selfplay
    assert refit.start_from == "round-1/copy.pt"
    assert refit.card_frozen_iterations == refit.iterations  # card play never trains
    assert refit.card_anchor_weight == 0.0
    assert refit.yardstick_every == 0
    assert refit.guess_lr > 0  # GuessNet keeps learning
    mix = config.train.opponents  # strong, varied opponents (Yosif, 2026-11-03)
    assert (mix.heuristic, mix.pimc) == (0.0, 0.0)
    assert mix.hybrid > 0
    assert mix.mixed > 0
    assert mix.hybrid_spec.startswith("hybrid:")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, r"\[expert\] section"),
        ({"expert": {"games": 3}}, "start_from"),
        ({"expert": {"start_from": "a.pt", "label_share": 0.0}}, "label_share"),
        ({"expert": {"start_from": "a.pt", "teachers": 3}}, "unknown key 'teachers'"),
        ({"expert": {"start_from": "a.pt"}, "selfplay": {"oops": 1}}, "unknown key 'oops'"),
    ],
)
def test_bad_configs_are_refused(data: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        config_of(data)


def test_the_teacher_labels_its_share_of_card_decisions_with_every_cards_value(recorded: Any):
    records = recorded
    assert records.guesses is not None
    assert len(records) > 50  # about a fifth of two whole games' card decisions with a choice
    assert set(np.unique(records.game).tolist()) == {GAME_ID, GAME_ID + 1}
    for row in range(len(records)):
        legal = np.flatnonzero(records.legal[row])
        values = records.values[row]
        assert len(legal) > 1
        assert records.legal[row, records.choice[row]]
        assert values[records.choice[row]] == pytest.approx(values[legal].max())
        assert not values[~records.legal[row]].any()
        labels = records.guesses[row]
        assert (labels != NO_GUESS).sum() == records.features[row, UNSEEN : UNSEEN + 52].sum()


def test_touching_cards_share_their_runs_value(recorded: Any):
    records = recorded
    shared = 0
    for row in range(len(records)):
        legal = np.flatnonzero(records.legal[row]).tolist()
        unseen = records.features[row, UNSEEN : UNSEEN + 52]
        out = sum(1 << c for c in range(52) if unseen[c])
        for run_cards in equivalent_runs(legal, out):
            values = {float(records.values[row, c]) for c in run_cards}
            assert len(values) == 1
            shared += len(run_cards) > 1
    assert shared > 0


def test_recording_is_deterministic(checkpoint: Path, recorded: Any):
    again = record_game(PartTask(1, 0, 1, str(checkpoint), tiny(checkpoint)), 0)
    first = recorded.game == GAME_ID
    assert np.array_equal(again.choice, recorded.choice[first])
    assert np.array_equal(again.values, recorded.values[first])
    assert np.array_equal(again.reward, recorded.reward[first])


def test_each_row_gets_its_rounds_reward(recorded: Any):
    assert np.isfinite(recorded.reward).all()
    assert len(set(recorded.reward.tolist())) > 3  # whole Bolas: many rounds, many rewards


@pytest.fixture(scope="module")
def pimc_records() -> CardRecords:
    """Recorded PIMC games: a search's card values on sane positions (a tiny random net's own
    games score wildly, with super calls and stacked carries)."""
    config = config_from_dict({"seed": 4, "record": {"games": 2, "pimc_samples": 2}})
    return concat_records([record_pimc(index, config) for index in range(2)])


def test_the_copy_learns_the_teachers_values(checkpoint: Path, pimc_records: CardRecords):
    config = tiny(checkpoint, copy_epochs=12, holdout=0.0)
    start = load_policy(checkpoint).card
    data = ImitationData(pimc_records, np.arange(len(pimc_records)), torch.device("cpu"))
    before = regret(start, data)
    net, report = copy_stage(config, 1, pimc_records, start, torch.device("cpu"))
    after = regret(net, data)
    assert after.top1 < before.top1
    assert after.top1 >= after.top2 >= after.top3 >= 0
    assert report["epochs"][-1]["advantage"] < report["epochs"][0]["advantage"]
    again, _ = copy_stage(config, 1, pimc_records, start, torch.device("cpu"))
    x = torch.as_tensor(pimc_records.features[:8].astype(np.float32))
    assert torch.equal(net(x)[1], again(x)[1])  # deterministic


def predictable(rows: int, seed: int) -> CardRecords:
    """Records of 40 games whose holders follow the suits, with the Hearts' holder given away
    by the first input."""
    rng = np.random.default_rng(seed)
    features = rng.random((rows, NUM_FEATURES), dtype=np.float32)
    features[:, VOIDS : VOIDS + 12] = 0.0  # nobody has shown out of a suit
    flipped = features[:, 0] < 0.5
    labels = np.zeros((rows, 52), np.int8)
    labels[:, 13:26] = 1
    labels[:, 26:52] = 2
    labels[flipped, 26:39] = 0
    return CardRecords(
        features=features.astype(np.float16),
        legal=np.ones((rows, 52), np.bool_),
        choice=np.zeros(rows, np.int16),
        values=np.zeros((rows, 52), np.float32),
        reward=np.zeros(rows, np.float32),
        game=(np.arange(rows) % 40).astype(np.int32),
        guesses=labels,
    )


def test_the_copy_teaches_the_guess_net_the_recorded_hands(checkpoint: Path):
    config = tiny(checkpoint, copy_epochs=30, guess_lr=1e-3, holdout=0.25)
    torch.manual_seed(1)
    net, accuracy = copy_guesses(config, 1, predictable(800, 3), GuessNet(32), torch.device("cpu"))
    assert accuracy["before"] < 0.5
    assert accuracy["after"] > 0.8  # on held-out games
    assert accuracy["blind"] == pytest.approx(1 / 3, abs=0.01)
    assert not net.training


def test_a_round_records_copies_refits_and_resumes(checkpoint: Path, tmp_path: Path):
    config = tiny(checkpoint)
    out = tmp_path / "expert"
    final = run(config, out)
    assert final == out / "round-1" / "hybrid.pt"
    assert sorted(p.name for p in (out / "round-1" / "records").iterdir()) == [
        "part-00000.npz",
        "part-00001.npz",
    ]
    report = json.loads((out / "round-1" / "copy.json").read_text())
    assert report["holdout_rows"] > 0
    copied, done = load_policy(out / "round-1" / "copy.pt"), load_policy(final)
    for key, value in copied.card.state_dict().items():
        assert torch.equal(value, done.card.state_dict()[key])  # the refit never moves cards
    assert not (out / "round-1" / "refit" / "warmstart").exists()  # no recording needed
    assert done.bidding is not None
    assert done.guess is not None  # a fresh GuessNet, trained in the refit
    shown = Console(file=io.StringIO(), width=120, color_system=None)
    shown.print(render(out))
    text = shown.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # a StringIO, set above
    assert "done" in text
    assert "before" in text
    stamp = final.stat().st_mtime_ns
    assert run(config, out) == final  # resumed: nothing is redone
    assert final.stat().st_mtime_ns == stamp
    changed = dataclasses.replace(config, expert=dataclasses.replace(config.expert, games=3))
    with pytest.raises(SystemExit, match="different config"):
        run(changed, out)


def test_the_watch_shows_the_running_stage(tmp_path: Path):
    out = tmp_path / "expert"
    (out / "round-1").mkdir(parents=True)
    expert = dataclasses.asdict(ExpertConfig(start_from="x.pt", rounds=2))
    (out / "config.json").write_text(json.dumps({"expert": expert, "train": {}}))
    progress = {"stage": "record", "done": 500, "games": 2000, "seconds": 600.0, "left": 1800.0}
    (out / "round-1" / "progress.json").write_text(json.dumps(progress))
    shown = Console(file=io.StringIO(), width=120, color_system=None)
    shown.print(render(out))
    text = shown.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # a StringIO, set above
    assert "Recording Hybrid games: 500 of 2,000  (25%)" in text
    assert "about 30 min left" in text
    assert "running" in text


def test_records_with_and_without_guess_labels_do_not_mix(recorded: Any):
    plain = dataclasses.replace(recorded, guesses=None)
    assert concat_records([plain, plain]).guesses is None
    assert concat_records([recorded, recorded]).guesses is not None
    with pytest.raises(ValueError, match="guess labels"):
        concat_records([recorded, plain])
