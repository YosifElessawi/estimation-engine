"""Bots see only their own Observation (CLAUDE.md hard rule, SPEC §6 M2).

Two guards: the bots package never touches the deal, a Bola or a ``hands`` attribute, which hold
every seat's cards; and a bot's choice never changes when the other seats' hidden cards do.

The RL bot (M4) also reads the state encoder and the networks in ``estimation.rl``; every module
it imports is held to the same static guard, and the encoder's features, like the bot's choices,
must not change when another seat's hidden cards do. The trainer in ``estimation.rl`` runs real
games, so it is outside the guard, but it shows the nets only these features.
"""

import ast
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from estimation.bots import (
    Bot,
    HeuristicBot,
    PIMCBot,
    PIMCConfig,
    RandomBot,
    YosifBot,
    YosifConfig,
    make_bot,
)
from estimation.engine import Bola, Observation, Phase, Round, RulesConfig
from estimation.engine.cards import SUIT_MASKS
from estimation.rl.config import config_from_dict
from estimation.rl.encoder import encode
from estimation.rl.networks import BiddingNet, CardPlayNet, state_dict_of
from estimation.rl.record import record_game
from estimation.rl.selfplay import (
    HEURISTIC,
    LEARNER,
    ChunkTask,
    GamePlan,
    NetWeights,
    SeatPlan,
    play_chunk,
)
from estimation.rl.standings import RewardConfig
from tests.rl.helpers import make_checkpoint

SRC = Path(__file__).resolve().parents[2] / "src"
BOTS = SRC / "estimation" / "bots"
FORBIDDEN_NAMES = {"Round", "Bola", "deal", "play_random_round", "play_random_bola"}
FORBIDDEN_IMPORTS = FORBIDDEN_NAMES | {"round", "bola"}  # names, and the engine's modules
FORBIDDEN_MODULES = {"estimation.engine.round", "estimation.engine.bola", "estimation.arena"}
FORBIDDEN_ATTRIBUTES = FORBIDDEN_IMPORTS | {"hands", "_hands"}


def hidden_card_access(source: str) -> list[str]:
    """Every way ``source`` could reach another seat's cards: importing or naming the deal, a
    Round or a Bola (directly, through a module alias or by a getattr string), or reading a
    ``hands`` attribute."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_MODULES or module.startswith("estimation.arena"):
                found.append(f"imports {module}")
            found += [f"imports {a.name}" for a in node.names if a.name in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.Import):
            found += [f"imports {a.name}" for a in node.names if a.name in FORBIDDEN_MODULES]
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(f"uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            found.append(f"reads .{node.attr}")
        elif isinstance(node, ast.Constant) and node.value in FORBIDDEN_ATTRIBUTES:
            found.append(f"names {node.value!r}")
    return found


def test_the_bots_package_never_reaches_for_hidden_cards():
    files = sorted(BOTS.rglob("*.py"))
    assert files
    found = [
        f"{path.name}: {hit}" for path in files for hit in hidden_card_access(path.read_text())
    ]
    assert found == []


def imported_modules(path: Path) -> set[str]:
    """The ``estimation`` modules imported by the file at ``path``, as dotted names."""
    package = ".".join(path.relative_to(SRC).with_suffix("").parts[:-1])
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{parent}.{base}" if base else parent
            found.add(base)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name for name in found if name.startswith("estimation")}


def rl_modules_the_bot_reads() -> list[Path]:
    """Every ``estimation.rl`` module the RL bot imports, directly or through another one."""
    todo = [BOTS / "rl.py"]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        for name in imported_modules(path):
            if not name.startswith("estimation.rl"):
                continue
            target = SRC.joinpath(*name.split("."))
            module = target / "__init__.py" if target.is_dir() else target.with_suffix(".py")
            if module not in seen:
                seen.add(module)
                todo.append(module)
    return sorted(seen)


def test_the_rl_modules_behind_the_bot_never_reach_for_hidden_cards():
    modules = rl_modules_the_bot_reads()
    names = {path.name for path in modules}
    assert {"encoder.py", "networks.py", "standings.py"} <= names
    assert not names & {"selfplay.py", "record.py", "train.py", "imitation.py"}
    found = [
        f"{path.name}: {hit}" for path in modules for hit in hidden_card_access(path.read_text())
    ]
    assert found == []


@pytest.mark.parametrize(
    "source",
    [
        "import estimation.engine as eng\neng.Round.new(1, eng.deal(0, 1))",
        "from estimation import engine\nengine.Bola(0)",
        "from estimation.engine import round as rd",
        "from estimation.engine.bola import Bola",
        "def peek(obs): return getattr(obs, '_hands')",
        "def peek(game): return game.round.hands",
    ],
)
def test_the_guard_catches_every_route_to_hidden_cards(source: str):
    assert hidden_card_access(source)


def test_the_guard_allows_reading_an_observation():
    source = "from estimation.engine import Observation, play_action\ndef act(obs): obs.hand"
    assert hidden_card_access(source) == []


def swapped_positions(seed: int) -> list[tuple[Round, Round, int]]:
    """Positions from a HeuristicBot game, each paired with the same position after swapping
    one hidden card between two other seats (same suit, so the history stays legal and every
    seat's suit counts, and with them the void announcements of TBL-7, stay the same). Returns
    ``(position, swapped, acting seat)``."""
    rng = random.Random(seed)
    bot = HeuristicBot()
    bola = Bola(seed)
    positions: list[tuple[Round, Round, int]] = []
    while not bola.is_over and len(positions) < 30:
        rnd = bola.round
        me = bola.current_player
        assert me is not None
        if rng.random() < 0.15:
            a, b = rng.sample([seat for seat in range(4) if seat != me], 2)
            hands = list(rnd.hands)
            left_a, left_b = rnd.observation(a).hand, rnd.observation(b).hand
            suits = [s for s in range(4) if left_a & SUIT_MASKS[s] and left_b & SUIT_MASKS[s]]
            if suits:
                suit = rng.choice(suits)
                mine, theirs = left_a & SUIT_MASKS[suit], left_b & SUIT_MASKS[suit]
                swap = (mine & -mine) | (theirs & -theirs)
                hands[a] ^= swap
                hands[b] ^= swap
                other = Round.replay(
                    rnd.round_no,
                    (hands[0], hands[1], hands[2], hands[3]),
                    rnd.fixed_trump,
                    rnd.carry,
                    rnd.scores,
                    rnd.config,
                    [action for _, action in rnd.log],
                )
                positions.append((rnd.clone(), other, me))
        bola.step(bot.act(bola.observation(me), bola.legal_actions()))
    return positions


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("heuristic", lambda: HeuristicBot()),
        ("random", lambda: RandomBot(5)),
        ("pimc", lambda: PIMCBot(5, PIMCConfig(samples=4, oversample=2))),
        (
            "yosif",
            lambda: YosifBot(
                5, YosifConfig(pimc=PIMCConfig(samples=4, oversample=2, leader_weight=True))
            ),
        ),
    ],
)
def test_a_bots_choice_ignores_other_seats_hidden_cards(name: str, make: Callable[[], Bot]):
    checked = 0
    seen: set[Phase] = set()
    for seed in range(12):
        for rnd, other, me in swapped_positions(seed):
            assert other.observation(me) == rnd.observation(me)
            assert make().act(other.observation(me), other.legal_actions()) == make().act(
                rnd.observation(me), rnd.legal_actions()
            ), name
            seen.add(rnd.phase)
            checked += 1
    assert checked > 100
    assert len(seen) >= 3


def test_the_encoder_ignores_other_seats_hidden_cards():
    checked = 0
    for seed in range(12):
        for rnd, other, me in swapped_positions(seed):
            assert np.array_equal(encode(other.observation(me)), encode(rnd.observation(me)))
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("bidding", ["net", "heuristic"])
def test_the_rl_bots_choice_ignores_other_seats_hidden_cards(bidding: str, tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt", seed=4)
    spec = f"rl:checkpoint={path}" + (":heuristic_bidding=true" if bidding == "heuristic" else "")
    bot = make_bot(spec, 5)
    checked = 0
    seen: set[Phase] = set()
    for seed in range(12):
        for rnd, other, me in swapped_positions(seed):
            assert bot.act(other.observation(me), other.legal_actions()) == bot.act(
                rnd.observation(me), rnd.legal_actions()
            )
            seen.add(rnd.phase)
            checked += 1
    assert checked > 100
    assert len(seen) >= 3


def test_the_trainer_shows_each_decision_only_the_acting_seats_observation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Training games (self-play and the PIMC recording) ask a Bola for observations only of the
    seat to act, apart from one read of public fields (round number, fixed trumps left) to
    count the rounds left for the match-win reward."""
    calls: list[tuple[int, int | None, str]] = []
    original = Bola.observation

    def spy(self: Bola, seat: int) -> Observation:
        calls.append((seat, self.current_player, sys._getframe(1).f_code.co_name))
        return original(self, seat)

    monkeypatch.setattr(Bola, "observation", spy)
    torch.manual_seed(0)
    weights = NetWeights(16, state_dict_of(CardPlayNet(16)), state_dict_of(BiddingNet(16)))
    learner, heuristic = SeatPlan(LEARNER), SeatPlan(HEURISTIC)
    plans = (
        GamePlan(0, "self", 5, 6, (learner, learner, learner, learner)),
        GamePlan(1, "heuristic", 7, 8, (heuristic, learner, heuristic, heuristic)),
    )
    task = ChunkTask(plans, weights, {}, 0.1, 0.1, 0.1, True, True, RulesConfig(), RewardConfig())
    play_chunk(task)
    record_game(0, config_from_dict({"record": {"pimc_samples": 2}}))
    assert len(calls) > 1000
    for seat, current, caller in calls:
        assert seat == current or caller == "_rounds_left", (seat, current, caller)
    assert {caller for seat, current, caller in calls if seat != current} == {"_rounds_left"}
