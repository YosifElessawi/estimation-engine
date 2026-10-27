"""The data-fitted calibration of the yosif bot's counting (tuning from saved games)."""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from estimation.bots.yosif.calibration import Calibration, load_calibration
from estimation.bots.yosif.odds import Calibrated, Outlook, expected_miss, make_chance

FLAT = Calibration(
    intercept=0.0,
    em=1.0,
    el=0.0,
    trumps=0.0,
    sans=0.0,
    fixed=0.0,
    last_over=0.0,
    offsets=(-2.0, 0.0, 2.0),
    made=(0.2, 0.6, 0.2),
    miss=(2.0, 1.0, 2.0),
    dash_p0=(0.0, 1.0),
    dash_made=(0.1, 0.5),
    source={"test": True},
)


def test_make_interpolates_by_distance_from_the_natural_count():
    assert FLAT.make(0.0) == pytest.approx(0.6)
    assert FLAT.make(1.0) == pytest.approx(0.4)
    assert FLAT.make(-5.0) < 0.01  # beyond the measured distances it falls away
    assert FLAT.dash(0.5) == pytest.approx(0.3)


def test_natural_count_is_the_fitted_line():
    assert FLAT.natural(em=7.0, el=3.0, trumps=5, sans=False, fixed=False, last_over=0) == 7.0
    lifted = dataclasses.replace(FLAT, intercept=1.0, trumps=0.5, sans=2.0)
    assert lifted.natural(em=7.0, el=3.0, trumps=4, sans=False, fixed=False, last_over=0) == 10.0
    assert lifted.natural(em=7.0, el=3.0, trumps=0, sans=True, fixed=False, last_over=0) == 10.0


def test_calibrated_outlooks_use_the_fitted_chances():
    o = Outlook(most=(0.0, 1.0), least=(1.0, 0.0), calibrated=Calibrated(natural=1.0, fit=FLAT))
    assert make_chance(o, 1) == pytest.approx(0.6)
    assert make_chance(o, 3) == pytest.approx(0.2)
    assert expected_miss(o, 1) == pytest.approx((1 - 0.6) * 1.0)


def test_the_calibration_file_round_trips(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(FLAT.to_json()))
    assert load_calibration(path) == FLAT
    assert load_calibration(tmp_path / "missing.json") is None


def test_make_chance_falls_away_beyond_the_measured_distances():
    spread = dataclasses.replace(FLAT, spread=1.0)
    assert spread.make(2.0) == pytest.approx(0.2)
    assert spread.make(3.0) == pytest.approx(0.2 * math.exp(-0.5))
    assert spread.make(6.0) < 0.01
    assert spread.make(-6.0) < 0.01
    assert spread.miss_size(4.0) > spread.miss_size(2.0)


def test_the_shipped_bot_uses_the_calibration():
    from estimation.bots import YosifBot, make_bot

    shipped, switched_off = make_bot("yosif", 0), make_bot("yosif:calibrated=false", 0)
    assert isinstance(shipped, YosifBot)
    assert isinstance(switched_off, YosifBot)
    assert shipped.config.calibrated
    assert shipped.config.calibration is not None
    assert not switched_off.config.calibrated
