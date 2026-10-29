"""Plain names, tables and one-line descriptions for the bots on the ladder (design 5.3).

The ladder report, the watch view and the web table name bots from here, so a spec such as
``rl:checkpoint=runs/m4/checkpoints/final.pt`` reads as "RL M4 final" everywhere. A bot sits in one
or more tables: the leaderboard of the two bots players meet (the Search bot and the RL bot), the
RL checkpoints that show training progress, and the measuring sticks that exist only to measure
the gaps between the others.
"""

import dataclasses
import re
from dataclasses import dataclass

LEADERBOARD = "Leaderboard"
CHECKPOINTS = "RL checkpoints"
STICKS = "Measuring sticks"
GROUPS = (LEADERBOARD, CHECKPOINTS, STICKS)

M4 = "rl:checkpoint=runs/m4/checkpoints/"
STAGED = "rl:checkpoint=runs/m4-staged/checkpoints/"
RULES_V2 = "rl:checkpoint=runs/m4-rules-v2/checkpoints/"
SEARCH_BOT = "search"
RL_BOT = RULES_V2 + "final.pt"  # the best RL checkpoint (RL v2), the RL bot players meet
HYBRID_BOT = "hybrid"
OFFERED = (SEARCH_BOT, RL_BOT, HYBRID_BOT)
"""The bots players meet at the table, the leaderboard's rows."""
ALIASES = {"yosif": SEARCH_BOT}
"""Old names that load the same bot, for saved games and ladder runs that use them."""


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
        _entry(
            SEARCH_BOT,
            "Search bot",
            (LEADERBOARD,),
            "Yosif's bidding rules; searches 30 deals per card",
        ),
        _entry(
            RL_BOT,
            "RL v2",
            (LEADERBOARD, CHECKPOINTS),
            "Self-play from the staged final, 3,000 loops under the current rules",
        ),
        _entry(
            HYBRID_BOT,
            "Hybrid",
            (LEADERBOARD,),
            "RL v2 bidding and deal weights; searches 16 deals per card with CardPlayNet",
        ),
        _entry(
            STAGED + "staged-final.pt",
            "RL staged final",
            (CHECKPOINTS,),
            "Self-play nets from the staged training run",
        ),
        _entry(
            M4 + "final.pt", "RL M4 final", (CHECKPOINTS,), "Self-play nets after 150 iterations"
        ),
        _entry(
            M4 + "warm-start.pt:heuristic_bidding=true",
            "RL warm start (HeuristicBot bidding)",
            (CHECKPOINTS,),
            "Card play learned by copying PIMC; HeuristicBot bidding",
        ),
        _entry("random", "RandomBot", (STICKS,), "Plays any legal move; fixed at 400 Elo"),
        _entry("heuristic", "HeuristicBot", (STICKS,), "Rule-of-thumb bidding and card play"),
        _entry("pimc", "PIMC", (STICKS,), "HeuristicBot bidding; searches 30 deals per card"),
    )
}
_ITERATION = re.compile(r"^rl:checkpoint=runs/m4/checkpoints/iter-0*(\d+)\.pt$")
_NOISE = re.compile(r"^heuristic:noise=([0-9.]+)$")
_PIMC = re.compile(r"^pimc:samples=(\d+)(?::play_noise=([0-9.]+))?$")
_HYBRID = re.compile(r"^hybrid:samples=(\d+)(?::play_noise=([0-9.]+))?$")


def _percent(text: str) -> str:
    return f"{float(text) * 100:g}%"


def canonical(spec: str) -> str:
    """``spec`` with an old bot name replaced by the one it loads as (``ALIASES``)."""
    name, sep, options = spec.partition(":")
    return ALIASES.get(name, name) + sep + options


def entry_for(spec: str) -> RosterEntry:
    """The roster entry of ``spec``; a spec the roster does not know is named by itself. An old
    name (``ALIASES``) gets the entry of the bot it loads as, under its own spec."""
    known = _KNOWN.get(canonical(spec))
    if known is not None:
        return known if known.spec == spec else dataclasses.replace(known, spec=spec)
    if match := _ITERATION.match(spec):
        n = match.group(1)
        return _entry(spec, f"RL iteration {n}", (CHECKPOINTS,), f"Self-play nets at iteration {n}")
    if match := _NOISE.match(spec):
        p = _percent(match.group(1))
        return _entry(
            spec,
            f"HeuristicBot, {p} random moves",
            (STICKS,),
            f"HeuristicBot playing {p} at random",
        )
    if match := _PIMC.match(spec):
        n = int(match.group(1))
        name = f"PIMC, {n} deal{'' if n == 1 else 's'}"
        if match.group(2):
            name += f", {_percent(match.group(2))} random cards"
        return _entry(
            spec, name, (STICKS,), f"PIMC searching {n} deal{'' if n == 1 else 's'} per card"
        )
    if match := _HYBRID.match(spec):
        n = int(match.group(1))
        name = f"Hybrid, {n} deal{'' if n == 1 else 's'}"
        if match.group(2):
            name += f", {_percent(match.group(2))} random cards"
        return _entry(
            spec, name, (STICKS,), f"Hybrid searching {n} deal{'' if n == 1 else 's'} per card"
        )
    return _entry(spec, spec, (STICKS,), "")


def name_of(spec: str) -> str:
    return entry_for(spec).name
