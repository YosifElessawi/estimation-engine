"""Every rule ID and worked example in SPEC.md has a test named after it (M1 Done when 1-2)."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_ID = re.compile(r"\b(TBL|GAM|DSH|AUC|EST|FIX|PLY|SCR)-(\d+)\b")
EXAMPLE_ID = re.compile(r"\*\*([GLT]\d+)[.*]")


def spec_ids() -> set[str]:
    spec = (ROOT / "SPEC.md").read_text()
    rules = spec[spec.index("## 2. Game Rules") : spec.index("## 4. Open Questions")]
    ids = {f"{prefix}-{number}" for prefix, number in RULE_ID.findall(rules)}
    return ids | set(EXAMPLE_ID.findall(rules))


def collect_test_names() -> list[str]:
    names = []
    for path in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        names += [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
    return names


def test_spec_lists_every_rule_family_and_example():
    ids = spec_ids()
    assert len([i for i in ids if "-" in i]) == 6 + 5 + 5 + 6 + 7 + 5 + 5 + 10
    assert {f"G{n}" for n in range(1, 11)} <= ids
    assert {f"L{n}" for n in range(1, 12)} <= ids
    assert {f"T{n}" for n in range(1, 5)} <= ids


def test_every_rule_and_example_has_a_test_named_after_it():
    names = collect_test_names()
    missing = []
    for spec_id in sorted(spec_ids()):
        tag = re.compile(rf"_{spec_id.replace('-', '_')}(?!\d)")
        if not any(tag.search(name) for name in names):
            missing.append(spec_id)
    assert missing == []
