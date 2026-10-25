"""Bots see only their own Observation (CLAUDE.md hard rule, SPEC §6 M2).

Two guards: the bots package never touches the deal, a Bola or a ``hands`` attribute, which hold
every seat's cards; and a bot's choice never changes when the other seats' hidden cards do. The
one exception is the bidding simulator, which builds Rounds of its own from hands it sampled
(never a real game's), so it may name Round; the dynamic test checks it learns nothing hidden.
"""

import ast
import random
from collections.abc import Callable
from pathlib import Path

import pytest

from estimation.bots import Bot, HeuristicBot, PIMCBot, PIMCConfig, RandomBot
from estimation.engine import Bola, Phase, Round
from estimation.engine.cards import SUIT_MASKS

BOTS = Path(__file__).resolve().parents[2] / "src" / "estimation" / "bots"
FORBIDDEN_NAMES = {"Round", "Bola", "deal", "play_random_round", "play_random_bola"}
FORBIDDEN_IMPORTS = FORBIDDEN_NAMES | {"round", "bola"}  # names, and the engine's modules
FORBIDDEN_MODULES = {"estimation.engine.round", "estimation.engine.bola", "estimation.arena"}
FORBIDDEN_ATTRIBUTES = FORBIDDEN_IMPORTS | {"hands", "_hands"}


SIMULATORS = {"bid_search.py"}  # build imagined Rounds from their own sampled hands


def hidden_card_access(source: str, allow_round: bool = False) -> list[str]:
    """Every way ``source`` could reach another seat's cards: importing or naming the deal, a
    Round or a Bola (directly, through a module alias or by a getattr string), or reading a
    ``hands`` attribute. With ``allow_round`` it may import and name ``Round`` itself."""
    found: list[str] = []
    allowed = {"Round"} if allow_round else set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_MODULES or module.startswith("estimation.arena"):
                found.append(f"imports {module}")
            found += [
                f"imports {a.name}" for a in node.names if a.name in FORBIDDEN_IMPORTS - allowed
            ]
        elif isinstance(node, ast.Import):
            found += [f"imports {a.name}" for a in node.names if a.name in FORBIDDEN_MODULES]
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES - allowed:
            found.append(f"uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES - allowed:
            found.append(f"reads .{node.attr}")
        elif isinstance(node, ast.Constant) and node.value in FORBIDDEN_ATTRIBUTES:
            found.append(f"names {node.value!r}")
    return found


def test_the_bots_package_never_reaches_for_hidden_cards():
    files = sorted(BOTS.rglob("*.py"))
    assert files
    found = [
        f"{path.name}: {hit}"
        for path in files
        for hit in hidden_card_access(path.read_text(), allow_round=path.name in SIMULATORS)
    ]
    assert found == []


def test_the_simulator_exception_still_forbids_real_games_and_the_deal():
    source = "from estimation.engine import Round, Bola, deal\nRound.replay\nx.hands"
    assert hidden_card_access(source, allow_round=True) == [
        "imports Bola",
        "imports deal",
        "reads .hands",
    ]


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
    one hidden card between two other seats (same suit once cards are played, so the history
    stays legal). Returns ``(position, swapped, acting seat)``."""
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
        ("bid search", lambda: PIMCBot(5, PIMCConfig(samples=2, bid_search=True, bid_samples=3))),
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
