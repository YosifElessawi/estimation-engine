"""Recording Hybrid v4's games for the one-net copy, and the one net's own search's games
(Hybrid plan step 5)."""

import dataclasses
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rich.console import Console

from estimation.arena.roster import HYBRID_V4
from estimation.bots.rl import policy_at
from estimation.engine import Bola, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.onenet.search import OneNetSearch
from estimation.rl.encoder import encode
from estimation.rl.networks import REWARD_SCALE
from estimation.rl.onenet_record import (
    KIND_BID,
    KIND_CARD,
    Arrays,
    PartTask,
    RecordConfig,
    bots,
    config_of,
    decisions,
    load_config,
    load_part,
    record_part,
    replay,
    run,
    save_part,
)
from estimation.rl.onenet_watch import render
from tests.onenet.test_search import random_net
from tests.rl.helpers import make_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def tiny(checkpoint: Path, **changes: Any) -> RecordConfig:
    settings: dict[str, Any] = {
        "player": f"hybrid:checkpoint={checkpoint}:deals=both:top_k=2:samples=2",
        "games": 2,
        "part_games": 2,
        "label_share": 0.5,
        "teacher_samples": 3,
        "seed": 4,
        "workers": 1,
    }
    return RecordConfig(**{**settings, **changes})


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_checkpoint(tmp_path_factory.mktemp("nets") / "net.pt", seed=3, guess=True)


@pytest.fixture(scope="module")
def part(checkpoint: Path) -> Arrays:
    return record_part(PartTask(0, 2, tiny(checkpoint)))


def test_the_committed_config_records_hybrid_v4_with_a_deeper_teacher():
    config = load_config(ROOT / "configs" / "onenet-record.toml")
    assert config.player == HYBRID_V4
    assert config.teacher_samples > 32  # deeper than Hybrid v4's own 32 deals
    assert config.label_share == 0.5
    assert config.games == 2000


def test_the_committed_expert_configs_record_the_one_nets_own_search():
    first = load_config(ROOT / "configs" / "onenet-expert-1.toml")
    second = load_config(ROOT / "configs" / "onenet-expert-2.toml")
    for config in (first, second):
        assert config.player.startswith("onenet:")
        assert ":own=true:" in config.player
        assert config.teacher_samples > 32  # deeper than the player's own 32 deals
        assert config.label_share == 0.5
    assert second.player == first.player
    assert (second.teacher_samples, second.games) == (256, 5000)  # round 2 (Yosif, 2026-11-10)
    seeds = {load_config(ROOT / "configs" / "onenet-record.toml").seed, first.seed, second.seed}
    assert len(seeds) == 3  # fresh deals every recording


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, r"\[record\] section"),
        ({"record": {"games": 3}}, "record.player"),
        ({"record": {"player": "pimc"}}, "Hybrid bot spec"),
        ({"record": {"player": "hybrid", "label_share": 1.5}}, "label_share"),
        ({"record": {"player": "hybrid", "teachers": 3}}, "unknown key 'teachers'"),
        ({"record": {"player": "hybrid"}, "selfplay": {}}, "unknown section"),
    ],
)
def test_bad_configs_are_refused(data: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        config_of(data)


def test_replaying_a_bolas_actions_gives_its_recorded_scores(part: Arrays):
    assert part["seed"].tolist() != [0, 0]
    for row in range(2):
        bola = replay(part, row)
        assert bola.is_over
        assert bola.scores == tuple(part["scores"][row].tolist())


def test_every_kept_decision_has_one_value_per_legal_action(checkpoint: Path, part: Arrays):
    policy = policy_at(str(checkpoint))
    bids = cards = 0
    for decision in decisions(part):
        obs = decision.bola.observation(decision.seat)
        assert len(decision.legal) > 1
        assert len(decision.values) == len(decision.legal)
        assert decision.action in decision.legal
        if decision.kind == KIND_BID:
            bids += 1
            assert obs.phase is not Phase.PLAY
            net = policy.bid_values(encode(obs, DEFAULT_CONFIG)[np.newaxis])[0] * REWARD_SCALE
            assert decision.values == pytest.approx([net[a] for a in decision.legal], abs=1e-3)
            assert decision.action == max(decision.legal, key=lambda a: net[a])  # bids by them
        else:
            cards += 1
            assert decision.kind == KIND_CARD
            assert obs.phase is Phase.PLAY
            played = decision.legal.index(decision.action)  # the teacher plays its best card
            assert decision.values[played] == pytest.approx(float(decision.values.max()))
    assert bids > 20
    assert cards > 20


def test_the_one_nets_search_records_with_a_deeper_teacher_and_no_bid_values(tmp_path: Path):
    net = random_net(tmp_path)
    config = tiny(Path(), player=f"onenet:net={net}:own=true:top_k=2:samples=2", games=1)
    players, teachers = bots(config, 0)
    for player, teacher in zip(players, teachers, strict=True):
        assert isinstance(player, OneNetSearch)
        assert isinstance(teacher, OneNetSearch)
        assert teacher.own
        assert (player.config.top_k, player.config.samples) == (2, 2)
        assert (teacher.config.top_k, teacher.config.samples) == (0, 3)  # every card, deeper
    recorded = record_part(PartTask(0, 1, config))
    assert replay(recorded, 0).scores == tuple(recorded["scores"][0].tolist())
    assert not (recorded["kind"] == KIND_BID).any()  # it bids by its own move chances
    cards = 0
    for decision in decisions(recorded):
        cards += 1
        assert decision.kind == KIND_CARD
        assert decision.bola.observation(decision.seat).phase is Phase.PLAY
        assert len(decision.values) == len(decision.legal)
        played = decision.legal.index(decision.action)  # the teacher plays its best card
        assert decision.values[played] == pytest.approx(float(decision.values.max()))
    assert cards > 10


def test_the_teacher_values_about_its_share_of_card_decisions(part: Arrays):
    choices = 0
    for row in range(2):
        bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
        start, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
        for action in part["actions"][start:end].tolist():
            seat = bola.current_player
            assert seat is not None
            if bola.observation(seat).phase is Phase.PLAY and len(bola.legal_actions()) > 1:
                choices += 1
            bola.step(action)
    valued = int((part["kind"] == KIND_CARD).sum())
    assert 0.3 < valued / choices < 0.6  # half, less the moves whose cards are all equivalent


def test_a_run_saves_parts_resumes_and_refuses_another_config(
    checkpoint: Path, part: Arrays, tmp_path: Path
):
    config = tiny(checkpoint, games=1, part_games=1)
    out = tmp_path / "record"
    summary = run(config, out)
    saved = out / "parts" / "part-00000.npz"
    assert sorted(p.name for p in (out / "parts").iterdir()) == [saved.name]
    assert summary["bolas"] == 1
    assert summary["bids"] > 0
    assert summary["cards"] > 0
    again = load_part(saved)  # a worker process records the same Bola as this one
    first = slice(int(part["log_start"][0]), int(part["log_start"][1]))
    assert again["actions"].tolist() == part["actions"][first].tolist()
    stamp = saved.stat().st_mtime_ns
    assert run(config, out) == summary  # resumed: nothing is recorded again
    assert saved.stat().st_mtime_ns == stamp
    with pytest.raises(SystemExit, match="different config"):
        run(dataclasses.replace(config, label_share=0.25), out)


def watched(part: Arrays, tmp_path: Path, progress: dict[str, Any], player: str = "hybrid") -> str:
    out = tmp_path / "record"
    (out / "parts").mkdir(parents=True)
    config = dataclasses.asdict(RecordConfig(player=player))
    (out / "config.json").write_text(json.dumps(config))
    (out / "progress.json").write_text(json.dumps(progress))
    save_part(out / "parts" / "part-00000.npz", part)
    (out / "parts" / "part-00500.games").write_text("20\n")  # a part under way
    shown = Console(file=io.StringIO(), width=200, color_system=None)
    shown.print(render(out, now=600.0))
    return shown.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue]  # a StringIO, set above


def test_the_watch_counts_every_recorded_bola(part: Arrays, tmp_path: Path):
    text = watched(part, tmp_path, {"games": 2000, "done": 500, "resumed_at": 0, "started": 0.0})
    assert "▸ Bolas recorded" in text
    assert "520 of 2,000 (26%)" in text
    assert "10 min so far, about 28 min left, 3120 Bolas an hour" in text
    assert "━" in text  # the bar
    assert "Saved" in text
    assert "2 Bolas" in text


def test_the_watch_shows_a_finished_recording(part: Arrays, tmp_path: Path):
    progress = {"games": 2000, "done": 1980, "resumed_at": 0, "started": 0.0, "finished": 3600.0}
    text = watched(part, tmp_path, progress)
    assert "✓ Bolas recorded" in text
    assert "in 1 h 00 min" in text


def test_the_watch_names_the_one_nets_own_search(part: Arrays, tmp_path: Path):
    progress = {"games": 2000, "done": 500, "resumed_at": 0, "started": 0.0}
    shown = watched(part, tmp_path, progress, player="onenet:net=player.npz:own=true")
    text = " ".join(shown.split())  # the description wraps
    assert "recording the one net's own search" in text
    assert "kept without values" in text
    assert "Hybrid v4" not in text
    assert "BiddingNet" not in text
