"""Running the bidding evaluator and reading its report (design 5.4)."""

import json
from pathlib import Path

from estimation.arena.games import GameRecord
from estimation.arena.match import play_bola
from estimation.bots import make_bot
from estimation.engine import RulesConfig
from estimation.eval.bidding import calibration, main, points_lost, sample_decisions
from estimation.eval.decisions import decisions_of
from tests.eval.test_decisions import recorded_game

RULES = RulesConfig()


def test_sampling_round_robins_over_kinds():
    decisions = decisions_of(recorded_game(), RULES, "heuristic")
    kinds = {d.kind for d in decisions}
    sample = sample_decisions(decisions, len(kinds), 0)
    assert {d.kind for d in sample} == kinds
    assert sample == sample_decisions(decisions, len(kinds), 0)


def test_a_bot_that_always_picks_the_best_loses_nothing():
    values = {1: (5.0, 1.0), 2: (3.0, 1.0)}
    assert points_lost(values, 1) == (1, 0.0)
    assert points_lost(values, 2) == (1, 2.0)


def test_cli_runs_on_one_tiny_game(tmp_path: Path):
    lineup = "heuristic,heuristic,heuristic,heuristic"
    args = ["--bot", "heuristic", "--lineup", lineup, "--games", "1", "--decisions", "3"]
    args += ["--deals", "2", "--card-play", "heuristic", "--out", str(tmp_path)]
    assert main(args) == 0
    report = (tmp_path / "report.txt").read_text()
    assert "Points lost per decision" in report
    data = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(data["decisions"]) == 3
    assert all(d["lost"] >= 0 for d in data["decisions"])


def test_web_mode_scores_the_human_seat(tmp_path: Path):
    record = recorded_game()
    seats = [{"spec": spec, "label": spec, "elo": None} for spec in record.seats]
    seats[1] = {"spec": "human", "label": "You", "elo": None}
    line = {
        "id": "g1",
        "finished_at": "2026-10-26T00:00:00+00:00",
        "mode": "play",
        "seed": record.deal_seed,
        "seed_chosen": False,
        "human_seat": 1,
        "seats": seats,
        "scores": list(record.scores),
        "winners": [0],
        "rounds": record.rounds,
        "actions": list(record.actions),
    }
    web = tmp_path / "games.jsonl"
    web.write_text(json.dumps(line) + "\n")
    args = ["--bot", "human", "--web", str(web), "--decisions", "2", "--deals", "2"]
    args += ["--card-play", "heuristic", "--out", str(tmp_path / "out")]
    assert main(args) == 0
    data = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert {d["seat"] for d in data["decisions"]} == {1}


def test_calibration_bins_the_yosif_bots_calls():
    bots = [make_bot("yosif:risk_mode=false", 1), *(make_bot("heuristic", s) for s in (2, 3, 4))]
    played = play_bola(11, bots, keep_actions=True)
    seats = ("yosif:risk_mode=false", "heuristic", "heuristic", "heuristic")
    record = GameRecord(0, 0, 11, seats, (1, 2, 3, 4), played.scores, played.rounds, played.actions)
    bins = calibration([record], RULES, "yosif:risk_mode=false")
    assert sum(b.count for b in bins) >= 10
    for b in bins:
        assert b.low <= b.predicted <= b.high
        assert 0.0 <= b.actual <= 1.0
