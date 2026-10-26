"""Measuring make chances by play-outs: assigned calls on random deals (no selection bias)."""

from pathlib import Path

from estimation.engine import RulesConfig
from estimation.eval.measure import SCENARIOS, main, read_trials, trials_for_deal

RULES = RulesConfig()


def test_each_deal_gives_trials_for_every_scenario_it_can():
    trials = trials_for_deal(3, "heuristic", RULES)
    scenarios = {t.scenario for t in trials}
    assert scenarios <= set(SCENARIOS)
    assert {"caller", "estimator"} <= scenarios
    for t in trials:
        assert t.made == (t.tricks == t.call)
        assert 0 <= t.tricks <= 13
    callers = [t for t in trials if t.scenario == "caller"]
    assert len({t.call for t in callers}) == len(callers) >= 2  # several calls on one deal
    assert len({(t.em, t.trumps) for t in callers}) == 1  # the same hand each time


def test_the_same_seed_gives_the_same_trials():
    assert trials_for_deal(5, "heuristic", RULES) == trials_for_deal(5, "heuristic", RULES)


def test_the_cli_writes_trials(tmp_path: Path):
    out = tmp_path / "playouts.jsonl.gz"
    args = ["--hands", "2", "--card-play", "heuristic", "--out", str(out)]
    assert main(args) == 0
    header, trials = read_trials(out)
    assert header["hands"] == 2
    assert trials
