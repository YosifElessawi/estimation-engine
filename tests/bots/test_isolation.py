"""Bots see only their own Observation (CLAUDE.md hard rule, SPEC §6 M2): the bots package never
touches the deal, a Round or a Bola, which hold every seat's cards."""

import ast
from pathlib import Path

BOTS = Path(__file__).resolve().parents[2] / "src" / "estimation" / "bots"
FORBIDDEN_NAMES = {"Round", "Bola", "deal", "play_random_round", "play_random_bola"}
FORBIDDEN_MODULES = {"estimation.engine.round", "estimation.engine.bola", "estimation.arena"}
FORBIDDEN_ATTRIBUTES = {"hands", "_hands", "round"}


def test_the_bots_package_never_reaches_for_hidden_cards():
    found: list[str] = []
    files = sorted(BOTS.rglob("*.py"))
    assert files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in FORBIDDEN_MODULES or module.startswith("estimation.arena"):
                    found.append(f"{path.name}: imports {module}")
                found += [
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name in FORBIDDEN_NAMES
                ]
            elif isinstance(node, ast.Import):
                found += [
                    f"{path.name}: imports {alias.name}"
                    for alias in node.names
                    if alias.name in FORBIDDEN_MODULES
                ]
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                found.append(f"{path.name}: uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
                found.append(f"{path.name}: reads .{node.attr}")
    assert found == []
