"""Fitting the yosif bot's calibration from saved games."""

import random
from pathlib import Path

import pytest

from estimation.arena.games import write_games
from estimation.bots.yosif.calibration import load_calibration
from estimation.engine import RulesConfig
from estimation.eval.calibrate import Row, extract_rows, fit, main
from tests.arena.test_games import HEADER
from tests.eval.test_decisions import recorded_game


def synthetic(n: int) -> list[Row]:
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        em, trumps = rng.uniform(1, 8), rng.randrange(0, 7)
        natural = 1.0 + em + 0.5 * trumps
        bid = round(natural + rng.choice((-1, 0, 0, 1)))
        tricks = round(natural + rng.choice((-1, 0, 1)))
        rows.append(Row("x", bid, tricks, bid == tricks, em, em / 2, trumps, False, 0, False, 0.0))
    for _ in range(200):
        zero = rng.uniform(0, 1)
        made = rng.random() < 0.6 * zero
        rows.append(Row("x", 0, 0 if made else 1, made, 0.5, 0.1, 2, False, 0, True, zero))
    return rows


def test_fit_recovers_the_natural_count_and_measures_the_curves():
    calibration = fit(synthetic(4000))
    assert calibration.natural(em=4.0, el=2.0, trumps=2, sans=False, fixed=False, last_over=0) == (
        pytest.approx(6.0, abs=0.3)
    )
    assert calibration.make(0.0) > calibration.make(2.0)
    assert calibration.dash(0.9) > calibration.dash(0.1)


def test_rows_come_from_saved_games(tmp_path: Path):
    path = tmp_path / "games.jsonl.gz"
    write_games(path, HEADER, [recorded_game()])
    rows = extract_rows([path], {"heuristic"}, 10, RulesConfig())
    assert len(rows) > 40
    assert all(0 <= row.tricks <= 13 for row in rows)


def test_the_cli_writes_a_calibration_the_bot_loads(tmp_path: Path):
    path = tmp_path / "games.jsonl.gz"
    write_games(path, HEADER, [recorded_game(seed=s) for s in range(3)])
    out = tmp_path / "calibration.json"
    args = ["--from", str(path), "--specs", "heuristic", "--out", str(out), "--min-rows", "5"]
    assert main(args) == 0
    calibration = load_calibration(out)
    assert calibration is not None
    assert calibration.source["rows"] > 0


def test_play_out_trials_set_the_make_and_dash_curves():
    from estimation.eval.measure import Trial

    rows = synthetic(4000)  # the natural count: 1 + em + 0.5 per trump
    trials = []
    for i in range(3000):
        call = 6 + (i % 5) - 2  # assigned calls around the natural count of 6
        made = (i % 5 == 2) or (i % 10 == 1)  # right at it: always; one step under: half
        trials.append(
            Trial("caller", i, call, 3, 4.0, 2.0, 2, False, 0.0, call if made else 9, made)
        )
    for i in range(600):
        zero = (i % 10) / 10 + 0.05
        trials.append(Trial("dash", i, 0, -1, 0.0, 0.0, 0, False, zero, 0, i % 10 >= 5))
    calibration = fit(rows, trials=trials)
    assert calibration.make(0.0) == pytest.approx(1.0, abs=0.05)
    assert calibration.make(-1.0) == pytest.approx(0.5, abs=0.05)
    assert calibration.make(1.0) == pytest.approx(0.0, abs=0.05)
    assert calibration.dash(0.95) == pytest.approx(1.0)
    assert calibration.dash(0.15) == pytest.approx(0.0)
    assert calibration.source["trials"] == 3600


def test_play_outs_give_big_calls_their_own_curve():
    from estimation.eval.measure import Trial

    trials = []
    for i in range(2000):
        small = Trial("caller", i, 6, 3, 4.0, 2.0, 2, False, 0.0, 6 if i % 2 else 7, i % 2 == 1)
        big = Trial("caller", i, 9, 3, 7.0, 3.5, 2, False, 0.0, 9 if i % 5 == 0 else 8, i % 5 == 0)
        trials += [small, big]
    calibration = fit(synthetic(4000), trials=trials, min_rows=40)
    assert calibration.make(0.0, call=6) == pytest.approx(0.5, abs=0.05)
    assert calibration.make(0.0, call=9) == pytest.approx(0.2, abs=0.05)
