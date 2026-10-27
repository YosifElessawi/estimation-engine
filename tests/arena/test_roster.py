"""Plain names and tables for every bot on the ladder (design 5.3)."""

from estimation.arena.ladder import PRESETS
from estimation.arena.roster import (
    CHECKPOINTS,
    GROUPS,
    LEADERBOARD,
    OFFERED,
    RL_BOT,
    STICKS,
    canonical,
    entry_for,
)


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
    assert entry_for("search").name == "Search bot"
    assert entry_for(RL_BOT).name == "RL bot"
    assert entry_for("random").groups == (STICKS,)
    assert entry_for("pimc").groups == (STICKS,)
    assert entry_for("pimc:samples=16").groups == (STICKS,)


def test_unknown_specs_fall_back_to_the_spec():
    entry = entry_for("wizard:x=1")
    assert entry.name == "wizard:x=1"
    assert entry.groups == (STICKS,)


def test_the_leaderboard_holds_only_the_search_bot_and_the_rl_bot():
    assert OFFERED == ("search", RL_BOT)
    for spec in OFFERED:
        assert entry_for(spec).groups[0] == LEADERBOARD
    assert entry_for(RL_BOT).groups == (LEADERBOARD, CHECKPOINTS)  # also a training checkpoint
    assert entry_for("rl:checkpoint=runs/m4/checkpoints/final.pt").groups == (CHECKPOINTS,)


def test_the_old_name_yosif_reads_as_the_search_bot():
    assert canonical("yosif") == "search"
    assert canonical("yosif:risk_mode=false") == "search:risk_mode=false"
    assert canonical("pimc") == "pimc"
    entry = entry_for("yosif")
    assert entry.spec == "yosif"
    assert (entry.name, entry.groups) == ("Search bot", (LEADERBOARD,))
