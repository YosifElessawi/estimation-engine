"""Plain names and tables for every bot on the ladder (design 5.3)."""

from estimation.arena.ladder import PRESETS
from estimation.arena.roster import CHECKPOINTS, GROUPS, LEADERBOARD, RUNGS, entry_for


def test_every_v2_bot_has_a_plain_name_and_group():
    for spec in PRESETS["v2"].bots:
        entry = entry_for(spec)
        assert entry.name != spec, spec
        assert entry.groups
        assert set(entry.groups) <= set(GROUPS)
        assert entry.description


def test_names_read_plainly():
    assert entry_for("heuristic:noise=0.1").name == "HeuristicBot, 10% random moves"
    assert entry_for("pimc:samples=8").name == "PIMC, 8 deals"
    assert entry_for("pimc:samples=1").name == "PIMC, 1 deal"
    assert entry_for("pimc:samples=1:play_noise=0.3").name == "PIMC, 1 deal, 30% random cards"
    assert entry_for("rl:checkpoint=runs/m4/checkpoints/iter-0020.pt").name == "RL iteration 20"
    assert entry_for("rl:checkpoint=runs/m4/checkpoints/final.pt").name == "RL M4 final"
    assert entry_for("yosif").name == "Yosif bot"
    assert entry_for("random").groups == (LEADERBOARD,)
    assert entry_for("pimc:samples=16").groups == (RUNGS,)


def test_unknown_specs_fall_back_to_the_spec():
    entry = entry_for("wizard:x=1")
    assert entry.name == "wizard:x=1"
    assert entry.groups == (RUNGS,)


def test_the_rl_finals_sit_in_two_tables():
    for spec in (
        "rl:checkpoint=runs/m4/checkpoints/final.pt",
        "rl:checkpoint=runs/m4-staged/checkpoints/staged-final.pt",
    ):
        assert entry_for(spec).groups == (LEADERBOARD, CHECKPOINTS)
