"""The bots a seat can hold, each labelled with its Elo from the ladder (SPEC §6 M6).

Ratings come from the ladder report (``runs/ladder-v2/ladder.json``, else the older
``runs/ladder-200/ladder.json``) and the M4 checkpoints from their index
(``runs/m4/checkpoints/index.json``), both under the project root. Both are
git-ignored run output: without them the catalog lists the default ladder's bots, unrated. The
two bots players meet (``roster.OFFERED``) are always listed, the RL bot when its checkpoint is
on disk. An old name in the ladder report (``roster.ALIASES``, for example ``yosif``) is listed
and rated as the bot it loads as.

A seat may also hold any other spec the bot registry accepts, for example ``pimc:samples=3``. An
``rl`` spec's checkpoint must be a file under the root's ``runs/`` folder; relative paths are
resolved against the root, so specs read the same as in the ladder report.
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from estimation.arena.ladder import DEFAULT_LADDER
from estimation.arena.roster import OFFERED, canonical, name_of
from estimation.bots import Bot, make_bot
from estimation.bots.hybrid import RL_V2_CHECKPOINT
from estimation.engine import RulesConfig

ROOT = Path(__file__).resolve().parents[1]
LADDER_FILE = Path("runs/ladder-200/ladder.json")
V2_LADDER_FILE = Path("runs/ladder-v2/ladder.json")  # read instead of LADDER_FILE once it exists
CHECKPOINT_INDEX = Path("runs/m4/checkpoints/index.json")
RUNS = "runs"


@dataclass(frozen=True, slots=True)
class BotEntry:
    """One bot: its registry spec, a readable label, its family and its ladder Elo (if rated).
    ``explains`` is true for RL bots, whose decisions the Why panel shows."""

    spec: str
    label: str
    family: str
    elo: float | None
    explains: bool


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    """``name`` and options of a ``name:key=value:...`` spec, as the bot registry reads it."""
    name, *pairs = spec.split(":")
    options: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"bot option {pair!r} must look like key=value")
        options[key] = value
    return name, options


def _percent(text: str) -> str:
    return f"{float(text) * 100:g}%"


def label_of(spec: str, checkpoints: dict[str, dict[str, Any]] | None = None) -> str:
    """A readable name for ``spec``, for example "PIMC, 8 deals" or "RL iteration 20": the
    ladder roster's name when the roster knows the spec, else built from its options.
    ``checkpoints`` maps a checkpoint file name to its index entry."""
    known = name_of(spec)
    if known != spec:
        return known
    try:
        name, options = parse_spec(spec)
    except ValueError:
        return spec
    extras: list[str] = []
    name = canonical(name)
    if name == "random" and not options:
        return "RandomBot"
    if name == "heuristic":
        label = "HeuristicBot"
    elif name == "search":
        label = "Search bot"
        if options.pop("bidding", "yosif") == "heuristic":
            extras.append("HeuristicBot bidding")
        if "samples" in options:
            samples = int(options.pop("samples"))
            extras.append(f"{samples} deal{'' if samples == 1 else 's'}")
    elif name == "pimc":
        samples = int(options.pop("samples", "30"))
        label = f"PIMC, {samples} deal{'' if samples == 1 else 's'}"
    elif name == "hybrid":
        label = "Hybrid"
        options.pop("checkpoint", None)
        if "samples" in options:
            samples = int(options.pop("samples"))
            extras.append(f"{samples} deal{'' if samples == 1 else 's'}")
    elif name == "rl" and "checkpoint" in options:
        path = Path(options.pop("checkpoint"))
        heuristic = options.pop("heuristic_bidding", "false").lower() in ("1", "true", "yes", "on")
        entry = (checkpoints or {}).get(path.name, {})
        iteration = entry.get("iteration")
        if entry.get("stage") == "imitate":
            label = "RL warm start"
        elif path.stem == "final" and iteration is not None:
            label = f"RL final (iteration {iteration})"
        elif iteration is not None:
            label = f"RL iteration {iteration}"
        else:
            label = f"RL {path.stem}"
        if heuristic:
            extras.append("HeuristicBot bidding")
    else:
        return spec
    if "noise" in options:
        extras.append(f"{_percent(options.pop('noise'))} random moves")
    if "play_noise" in options:
        extras.append(f"{_percent(options.pop('play_noise'))} random cards")
    extras.extend(f"{key}={value}" for key, value in options.items())
    return ", ".join([label, *extras])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: object = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return cast(dict[str, Any], data)  # a JSON object's keys are always strings


def runs_root(checkout: Path = ROOT) -> Path:
    """The checkout whose ``runs/`` holds the ratings and checkpoints: the main checkout when
    ``checkout`` is a git worktree (``runs/`` is git-ignored, so worktrees start without it and
    ladders and training write to the main one), else ``checkout`` itself."""
    git = checkout / ".git"
    if git.is_file():  # a worktree: "gitdir: <main>/.git/worktrees/<name>"
        text = git.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.removeprefix("gitdir:").strip())
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parent.parent.parent
    return checkout


def _on_disk(root: Path, spec: str) -> bool:
    """Whether ``spec`` can be played under ``root``: an rl or hybrid spec's checkpoint must be a
    file."""
    name, options = parse_spec(spec)
    if name == "rl":
        return (root / options.get("checkpoint", "")).is_file()
    if name == "hybrid":
        return (root / options.get("checkpoint", RL_V2_CHECKPOINT)).is_file()
    return True


class Catalog:
    """The bots on offer, weakest first (unrated bots last), and a factory for seat specs."""

    def __init__(self, root: Path, entries: list[BotEntry], checkpoints: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.entries = tuple(entries)
        self._by_spec = {entry.spec: entry for entry in entries}
        self._checkpoints = checkpoints

    @classmethod
    def load(
        cls,
        root: Path = ROOT,
        ladder_file: Path | None = None,
        checkpoint_index: Path = CHECKPOINT_INDEX,
    ) -> "Catalog":
        """Read the ladder report and the checkpoint index under ``root``, when present. The
        ladder report is the v2 ladder's once it exists, else the older one."""
        root = root.resolve()
        if ladder_file is None:
            ladder_file = V2_LADDER_FILE if (root / V2_LADDER_FILE).is_file() else LADDER_FILE
        ratings: dict[str, float] = {}
        specs: list[str] = list(DEFAULT_LADDER)
        ladder = _read_json(root / ladder_file)
        if ladder is not None:
            rated = {str(spec): float(elo) for spec, elo in dict(ladder["ratings"]).items()}
            # A bot rated under its own name wins over one rated under an old name.
            for spec in sorted(rated, key=lambda spec: canonical(spec) == spec):
                ratings[canonical(spec)] = rated[spec]
            specs = list(dict.fromkeys(canonical(str(spec)) for spec in ladder["bots"]))
        checkpoints: dict[str, dict[str, Any]] = {}
        index = _read_json(root / checkpoint_index)
        if index is not None:
            folder = checkpoint_index.parent
            for item in index["checkpoints"]:
                entry = {str(key): value for key, value in dict(item).items()}
                checkpoints[str(entry["file"])] = entry
                elos = {str(spec): float(elo) for spec, elo in dict(entry.get("elo") or {}).items()}
                for spec, elo in elos.items():
                    ratings.setdefault(spec, elo)  # the ladder report's fit wins
                rated = list(elos)
                spec = rated[0] if rated else f"rl:checkpoint={(folder / entry['file']).as_posix()}"
                if not rated and entry.get("stage") == "imitate":
                    spec += ":heuristic_bidding=true"
                if spec not in specs:
                    specs.append(spec)
        for spec in OFFERED:
            if spec not in specs and _on_disk(root, spec):
                specs.append(spec)
        entries = [
            BotEntry(
                spec=spec,
                label=label_of(spec, checkpoints),
                family=parse_spec(spec)[0],
                elo=ratings.get(spec),
                explains=parse_spec(spec)[0] in ("rl", "hybrid"),
            )
            for spec in specs
        ]
        entries.sort(key=lambda e: (e.elo is None, e.elo or 0.0))
        return cls(root, entries, checkpoints)

    @classmethod
    def from_env(cls) -> "Catalog":
        """The catalog under ``$ESTIMATION_ROOT``, else under ``runs_root()``."""
        env = os.environ.get("ESTIMATION_ROOT")
        return cls.load(Path(env) if env else runs_root())

    def entry(self, spec: str) -> BotEntry:
        """The listed entry for ``spec`` (or for the bot an old name loads as), or an unrated one
        for any other spec."""
        known = self._by_spec.get(canonical(spec))
        if known is not None:
            return known if known.spec == spec else dataclasses.replace(known, spec=spec)
        name = parse_spec(spec)[0]
        return BotEntry(
            spec, label_of(spec, self._checkpoints), name, None, name in ("rl", "hybrid")
        )

    def resolve(self, spec: str) -> str:
        """``spec`` with an rl or hybrid checkpoint as an absolute path, checked to be a file
        under the root's ``runs/`` folder. Raises ValueError for anything else."""
        name, options = parse_spec(spec)
        if name == "hybrid" and "checkpoint" not in options:
            options["checkpoint"] = RL_V2_CHECKPOINT
        if name not in ("rl", "hybrid") or "checkpoint" not in options:
            return spec
        runs = (self.root / RUNS).resolve()
        path = (self.root / options["checkpoint"]).resolve()
        if not path.is_relative_to(runs) or not path.is_file():
            raise ValueError(f"checkpoint {options['checkpoint']!r} is not a file under {RUNS}/")
        options["checkpoint"] = str(path)
        return ":".join([name, *(f"{key}={value}" for key, value in options.items())])

    def make(self, spec: str, seed: int, rules: RulesConfig) -> Bot:
        """The bot for ``spec``, seeded with ``seed``. Raises ValueError for a bad spec."""
        return make_bot(self.resolve(spec), seed, rules)
