"""A small project root for server tests: two tiny checkpoints, their index and a ladder report."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.catalog import Catalog
from tests.rl.helpers import make_checkpoint

FINAL = "rl:checkpoint=runs/m4/checkpoints/final.pt"
WARM = "rl:checkpoint=runs/m4/checkpoints/warm-start.pt:heuristic_bidding=true"


@pytest.fixture(scope="session")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("project")
    folder = root / "runs" / "m4" / "checkpoints"
    make_checkpoint(folder / "final.pt", seed=1)
    make_checkpoint(folder / "iter-0010.pt", seed=2)
    make_checkpoint(folder / "warm-start.pt", seed=3, bidding=False)
    index = {
        "checkpoints": [
            {"file": "warm-start.pt", "elo": {WARM: 1900.0}, "stage": "imitate", "iteration": 0},
            {"file": "iter-0010.pt", "elo": {}, "stage": "selfplay", "iteration": 10},
            {"file": "final.pt", "elo": {FINAL: 2000.0}, "stage": "selfplay", "iteration": 150},
        ]
    }
    (folder / "index.json").write_text(json.dumps(index))
    ladder = {
        "bots": ["random", "heuristic", FINAL],
        "ratings": {"random": 400.0, "heuristic": 1000.0, FINAL: 2100.0},
    }
    (root / "runs" / "ladder-200").mkdir(parents=True)
    (root / "runs" / "ladder-200" / "ladder.json").write_text(json.dumps(ladder))
    (root / "outside.pt").write_bytes((folder / "final.pt").read_bytes())
    return root


@pytest.fixture(scope="session")
def catalog(root: Path) -> Catalog:
    return Catalog.load(root)


@pytest.fixture
def history_file(tmp_path: Path) -> Path:
    return tmp_path / "web" / "games.jsonl"


@pytest.fixture
def client(catalog: Catalog, history_file: Path) -> TestClient:
    return TestClient(create_app(catalog, history_file=history_file))
