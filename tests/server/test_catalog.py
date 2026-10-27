"""The bot catalog (SPEC §6 M6): ladder bots and checkpoints, labelled with their Elo."""

from pathlib import Path

import pytest

from estimation.arena.ladder import DEFAULT_LADDER
from estimation.bots import HeuristicBot, NoisyBot
from estimation.bots.rl import RLBot
from estimation.engine.config import DEFAULT_CONFIG
from server.catalog import Catalog, label_of, runs_root
from tests.server.conftest import FINAL, WARM


def test_a_worktree_reads_the_main_checkouts_runs(tmp_path: Path):
    """runs/ is git-ignored and ladders write to the main checkout, so a table started in a
    worktree must read the ratings there."""
    main = tmp_path / "Estimation Engine"
    (main / ".git" / "worktrees" / "side").mkdir(parents=True)
    worktree = main / ".claude" / "worktrees" / "side"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'side'}\n")
    assert runs_root(worktree) == main
    assert runs_root(main) == main  # the main checkout has a .git folder: it is its own root
    plain = tmp_path / "plain"
    plain.mkdir()
    assert runs_root(plain) == plain


def test_estimation_root_overrides_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ESTIMATION_ROOT", str(tmp_path))
    assert Catalog.from_env().root == tmp_path.resolve()


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        ("random", "RandomBot"),
        ("heuristic", "HeuristicBot"),
        ("heuristic:noise=0.75", "HeuristicBot, 75% random moves"),
        ("pimc", "PIMC"),
        ("yosif", "Yosif bot"),
        ("pimc:samples=1", "PIMC, 1 deal"),
        ("pimc:samples=1:play_noise=0.3", "PIMC, 1 deal, 30% random cards"),
        ("pimc:samples=4:oversample=1", "PIMC, 4 deals, oversample=1"),
        ("rl:checkpoint=runs/x/net.pt", "RL net"),
        ("rl:checkpoint=runs/x/net.pt:heuristic_bidding=true", "RL net, HeuristicBot bidding"),
        ("mystery:x=1", "mystery:x=1"),
    ],
)
def test_labels_name_each_bot_family(spec: str, label: str):
    assert label_of(spec) == label


def test_checkpoint_labels_use_the_index(catalog: Catalog):
    labels = {entry.spec: entry.label for entry in catalog.entries}
    assert labels[FINAL] == "RL M4 final"  # the roster's name wins for a known checkpoint
    assert labels[WARM] == "RL warm start (HeuristicBot bidding)"
    assert labels["rl:checkpoint=runs/m4/checkpoints/iter-0010.pt"] == "RL iteration 10"


def test_the_catalog_lists_rated_bots_weakest_first_then_unrated(catalog: Catalog):
    rated = [(entry.spec, entry.elo) for entry in catalog.entries if entry.elo is not None]
    assert rated == [("random", 400.0), ("heuristic", 1000.0), (WARM, 1900.0), (FINAL, 2100.0)]
    assert catalog.entries[-1].spec == "rl:checkpoint=runs/m4/checkpoints/iter-0010.pt"
    assert catalog.entries[-1].elo is None
    assert [entry.explains for entry in catalog.entries] == [False, False, True, True, True]


def test_the_ladder_report_rating_wins_over_the_checkpoint_index(catalog: Catalog):
    assert catalog.entry(FINAL).elo == 2100.0  # the index says 2000, the ladder 2100


def test_without_run_output_the_catalog_is_the_default_ladder_unrated(tmp_path: Path):
    catalog = Catalog.load(tmp_path)
    assert [entry.spec for entry in catalog.entries] == list(DEFAULT_LADDER)
    assert all(entry.elo is None for entry in catalog.entries)


def test_specs_become_bots(catalog: Catalog):
    assert isinstance(catalog.make("heuristic", 0, DEFAULT_CONFIG), HeuristicBot)
    assert isinstance(catalog.make("heuristic:noise=0.5", 0, DEFAULT_CONFIG), NoisyBot)
    final = catalog.make(FINAL, 0, DEFAULT_CONFIG)
    assert isinstance(final, RLBot)
    assert Path(final.config.checkpoint) == catalog.root / "runs/m4/checkpoints/final.pt"
    assert catalog.entry("pimc:samples=3").label == "PIMC, 3 deals"
    assert catalog.entry("pimc:samples=3").elo is None


@pytest.mark.parametrize(
    "spec",
    [
        "rl:checkpoint=outside.pt",
        "rl:checkpoint=runs/../outside.pt",
        "rl:checkpoint=runs/m4/checkpoints/missing.pt",
        "rl:checkpoint=/etc/hosts",
    ],
)
def test_checkpoints_must_be_files_under_runs(catalog: Catalog, spec: str):
    with pytest.raises(ValueError, match="not a file under runs/"):
        catalog.make(spec, 0, DEFAULT_CONFIG)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("nobody", "unknown bot"),
        ("heuristic:noise", "must look like key=value"),
        ("heuristic:noise=2", "must be between 0 and 1"),
        ("pimc:x=1", "has no option 'x'"),
    ],
)
def test_bad_specs_are_refused(catalog: Catalog, spec: str, message: str):
    with pytest.raises(ValueError, match=message):
        catalog.make(spec, 0, DEFAULT_CONFIG)


def test_the_catalog_reads_the_v2_ladder_when_it_exists(tmp_path: Path):
    import json

    folder = tmp_path / "runs" / "ladder-v2"
    folder.mkdir(parents=True)
    ladder = {"bots": ["random", "heuristic", "yosif"], "ratings": {"random": 400, "yosif": 4900}}
    (folder / "ladder.json").write_text(json.dumps(ladder))
    catalog = Catalog.load(tmp_path)
    elos = {entry.spec: entry.elo for entry in catalog.entries}
    assert elos["yosif"] == 4900
    assert {entry.spec: entry.label for entry in catalog.entries}["yosif"] == "Yosif bot"
