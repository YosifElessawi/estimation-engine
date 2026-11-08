"""Step 5's recorder with the search on the GPU: the same records as the CPU recorder's."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from estimation.engine import Bola, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.onenet.gpu_record import GpuSettings, player_of, run
from estimation.rl.onenet_record import (
    KIND_CARD,
    Arrays,
    RecordConfig,
    decisions,
    load_part,
    part_path,
    replay,
)
from tests.onenet.test_search import random_net

GPU = GpuSettings(batch=2, redeals=2)


def tiny(net: str, **changes: Any) -> RecordConfig:
    settings: dict[str, Any] = {
        "player": f"onenet:net={net}:own=true:top_k=2:samples=2",
        "games": 2,
        "part_games": 2,
        "label_share": 0.5,
        "teacher_samples": 3,
        "seed": 4,
        "workers": 1,
    }
    return RecordConfig(**{**settings, **changes})


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[RecordConfig, Path, Arrays]:
    folder = tmp_path_factory.mktemp("gpu-record")
    config = tiny(random_net(folder))
    out = folder / "out"
    summary = run(config, out, GPU)
    assert summary["bolas"] == 2
    return config, out, load_part(part_path(out, 0))


def test_every_bola_replays_to_its_recorded_scores(recorded: tuple[RecordConfig, Path, Arrays]):
    _, _, part = recorded
    for row in range(2):
        bola = replay(part, row)
        assert bola.is_over
        assert bola.scores == tuple(part["scores"][row].tolist())


def test_the_teacher_keeps_a_value_per_legal_card_and_plays_its_best(
    recorded: tuple[RecordConfig, Path, Arrays],
):
    _, _, part = recorded
    cards = 0
    for decision in decisions(part):
        cards += 1
        assert decision.kind == KIND_CARD  # the player bids by its own move chances: no values
        assert decision.bola.observation(decision.seat).phase is Phase.PLAY
        assert len(decision.values) == len(decision.legal)
        assert np.isfinite(decision.values).all()
        played = decision.legal.index(decision.action)
        assert decision.values[played] == pytest.approx(float(decision.values.max()))
    assert cards > 20


def test_the_teacher_values_about_its_share_of_card_decisions(
    recorded: tuple[RecordConfig, Path, Arrays],
):
    _, _, part = recorded
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
    assert 0.3 < valued / choices < 0.6


def test_a_finished_run_resumes_with_nothing_to_do_and_refuses_other_settings(
    recorded: tuple[RecordConfig, Path, Arrays],
):
    config, out, part = recorded
    assert run(config, out, GPU)["bolas"] == 2
    assert load_part(part_path(out, 0))["actions"].tolist() == part["actions"].tolist()
    with pytest.raises(SystemExit, match="other GPU settings"):
        run(config, out, GpuSettings(batch=3, redeals=2))


def test_only_the_one_nets_own_search_is_recorded(tmp_path: Path):
    net = random_net(tmp_path)
    assert player_of(tiny(net))[0] == net
    with pytest.raises(ValueError, match="own=true and judge=round"):
        player_of(tiny(net, player=f"onenet:net={net}:own=true:judge=king"))
