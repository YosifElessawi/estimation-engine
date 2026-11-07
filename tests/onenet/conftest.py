"""A tiny recording of Hybrid play (as step 5 records Hybrid v4) and its training set, shared by
step 5's tests."""

import dataclasses
import json
from pathlib import Path

import pytest

from estimation.onenet.data import build
from estimation.rl.onenet_record import PartTask, RecordConfig, part_path, record_part, save_part
from tests.rl.helpers import make_checkpoint

BOLAS = 3


@pytest.fixture(scope="package")
def record(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A recording folder as ``onenet_record run`` leaves it: its config and one part."""
    folder = tmp_path_factory.mktemp("record")
    checkpoint = make_checkpoint(folder / "net.pt", seed=3, guess=True)
    config = RecordConfig(
        player=f"hybrid:checkpoint={checkpoint}:deals=both:top_k=2:samples=2",
        games=BOLAS,
        part_games=BOLAS,
        label_share=0.5,
        teacher_samples=3,
        seed=4,
        workers=1,
    )
    (folder / "parts").mkdir()
    save_part(part_path(folder, 0), record_part(PartTask(0, BOLAS, config)))
    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    return folder


@pytest.fixture(scope="package")
def data(record: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The recording's training set."""
    out = tmp_path_factory.mktemp("data")
    build(record, out)
    return out
