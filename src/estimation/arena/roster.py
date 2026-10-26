"""Plain names, tables and one-line descriptions for the bots on the ladder (design 5.3).

The ladder report, the watch view and the web table name bots from here, so a spec such as
``rl:checkpoint=runs/m4/checkpoints/final.pt`` reads as "RL M4 final" everywhere. A bot sits in one
or more tables: the leaderboard of bots worth playing, the RL checkpoints that show training
progress, and the measuring rungs that exist only to measure the gaps between the others.
"""

import re
from dataclasses import dataclass

LEADERBOARD = "Leaderboard"
CHECKPOINTS = "RL checkpoints"
RUNGS = "Measuring rungs"
GROUPS = (LEADERBOARD, CHECKPOINTS, RUNGS)

M4 = "rl:checkpoint=runs/m4/checkpoints/"
STAGED = "rl:checkpoint=runs/m4-staged/checkpoints/"


@dataclass(frozen=True, slots=True)
class RosterEntry:
    spec: str
    name: str
    groups: tuple[str, ...]
    description: str


def _entry(spec: str, name: str, groups: tuple[str, ...], description: str) -> RosterEntry:
    return RosterEntry(spec, name, groups, description)


_KNOWN = {
    e.spec: e
    for e in (
        _entry("random", "RandomBot", (LEADERBOARD,), "Plays any legal move; fixed at 400 Elo"),
        _entry("heuristic", "HeuristicBot", (LEADERBOARD,), "Rule-of-thumb bidding and card play"),
        _entry("pimc", "PIMC", (LEADERBOARD,), "HeuristicBot bidding; searches 30 deals per card"),
        _entry("yosif", "Yosif bot", (LEADERBOARD,), "Yosif's bidding rules, PIMC card play"),
        _entry(
            M4 + "final.pt",
            "RL M4 final",
            (LEADERBOARD, CHECKPOINTS),
            "Self-play nets after 150 iterations (M4)",
        ),
        _entry(
            STAGED + "staged-final.pt",
            "RL staged final",
            (LEADERBOARD, CHECKPOINTS),
            "Self-play nets from the staged training run",
        ),
        _entry(
            M4 + "warm-start.pt:heuristic_bidding=true",
            "RL warm start (HeuristicBot bidding)",
            (CHECKPOINTS,),
            "Card play learned by copying PIMC; HeuristicBot bidding",
        ),
    )
}
_ITERATION = re.compile(r"^rl:checkpoint=runs/m4/checkpoints/iter-0*(\d+)\.pt$")
_NOISE = re.compile(r"^heuristic:noise=([0-9.]+)$")
_PIMC = re.compile(r"^pimc:samples=(\d+)(?::play_noise=([0-9.]+))?$")


def _percent(text: str) -> str:
    return f"{float(text) * 100:g}%"


def entry_for(spec: str) -> RosterEntry:
    """The roster entry of ``spec``; a spec the roster does not know is named by itself."""
    known = _KNOWN.get(spec)
    if known is not None:
        return known
    if match := _ITERATION.match(spec):
        n = match.group(1)
        return _entry(spec, f"RL iteration {n}", (CHECKPOINTS,), f"Self-play nets at iteration {n}")
    if match := _NOISE.match(spec):
        p = _percent(match.group(1))
        return _entry(
            spec, f"HeuristicBot, {p} random moves", (RUNGS,), f"HeuristicBot playing {p} at random"
        )
    if match := _PIMC.match(spec):
        n = int(match.group(1))
        name = f"PIMC, {n} deal{'' if n == 1 else 's'}"
        if match.group(2):
            name += f", {_percent(match.group(2))} random cards"
        return _entry(
            spec, name, (RUNGS,), f"PIMC searching {n} deal{'' if n == 1 else 's'} per card"
        )
    return _entry(spec, spec, (RUNGS,), "")


def name_of(spec: str) -> str:
    return entry_for(spec).name
