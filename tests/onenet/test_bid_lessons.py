"""Step 6's bid lessons: the decisions found in a recording, the lesson parts a run saves, a run
shared by two machines, and the timing and noise test."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from estimation.engine import Bola, Observation, Phase, RulesConfig
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv import game as JG
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.onenet import lessons as L
from estimation.onenet.bid_lessons import (
    Labeler,
    LessonConfig,
    config_of,
    gated,
    noise,
    noise_lines,
    replay,
    run,
    time_run,
)
from estimation.onenet.search import bidding_views
from estimation.rl.onenet_record import (
    Arrays,
    PartTask,
    RecordConfig,
    load_part,
    part_path,
    record_part,
    save_part,
)
from tests.onenet.test_search import random_net

BOLAS = 2
DEALS = (2, 4)


@pytest.fixture(scope="module")
def games(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A recording of a small random player inside its own search, as step 6's games are
    recorded: no teacher, here one Bola a part."""
    folder = tmp_path_factory.mktemp("games")
    config = RecordConfig(
        player=f"onenet:net={random_net(folder)}:own=true:top_k=2:samples=2",
        games=BOLAS,
        part_games=1,
        label_share=0.0,
        teacher_samples=3,
        seed=5,
        workers=1,
    )
    (folder / "parts").mkdir()
    for first in range(BOLAS):
        arrays = record_part(PartTask(first, 1, config))
        assert arrays is not None
        save_part(part_path(folder, first), arrays)
    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    return folder


def tiny(games: Path, **changes: Any) -> LessonConfig:
    settings: dict[str, Any] = {
        "record": str(games),
        "label_share": 0.1,
        "deals": DEALS,
        "min_options": 3,  # a random net is sure of its move: its chances alone would try 2
        "max_options": 3,
        "redeals": 2,
        "batch": 8,
        "seed": 6,
    }
    return LessonConfig(**{**settings, **changes})


class Choice:
    """A bidding decision with a choice, as the engine alone replays it."""

    def __init__(self, step: int, obs: Observation, legal: list[int], action: int) -> None:
        self.step, self.obs, self.legal, self.action = step, obs, legal, action


def choices(part: Arrays, row: int = 0) -> list[Choice]:
    bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
    start, end = int(part["log_start"][row]), int(part["log_start"][row + 1])
    found: list[Choice] = []
    for step, action in enumerate(part["actions"][start:end].tolist()):
        seat = bola.current_player
        assert seat is not None
        legal = bola.legal_actions()
        if bola.phase is not Phase.PLAY and len(legal) > 1:
            found.append(Choice(step, bola.observation(seat), legal, action))
        bola.step(action)
    return found


def holds(obs: Observation) -> bool:
    """Whether JAX's int32 round can hold the Bola at this view."""
    return obs.carry <= JG.CARRY_CAP and max(abs(s) for s in obs.scores) <= JG.SCORE_LIMIT


# The decisions to teach.


def test_replay_finds_every_bidding_decision_with_a_choice(games: Path):
    part = load_part(part_path(games, 0))
    rounds, spots, unheld = replay(part, tiny(games, label_share=1.0))
    found = choices(part)
    # A random player's Bola doubles its carry past JAX's int32 range: those rounds are left out.
    want = [c for c in found if holds(c.obs)]
    assert unheld == len(found) - len(want)
    assert len(want) > 50
    assert [(s.game, s.step, s.seat, list(s.legal)) for s in spots] == [
        (0, c.step, c.obs.seat, c.legal) for c in want
    ]
    assert {s.phase for s in spots} == {int(Phase.DASH), int(Phase.AUCTION), int(Phase.ESTIMATION)}
    for spot, choice in zip(spots, want, strict=True):
        dealt, obs = rounds[spot.round], choice.obs
        assert spot.phase == int(obs.phase)
        assert dealt.moves[spot.at] == choice.action
        assert (dealt.round_no, dealt.carry, dealt.scores) == (obs.round_no, obs.carry, obs.scores)
        assert dealt.fixed_trump == (-1 if obs.fixed_trump is None else obs.fixed_trump)
        assert dealt.hands[obs.seat] == obs.hand  # no card is played before the bidding ends
        # Its history is the other seats' earlier choices this round, as the card search reads them.
        earlier = bidding_views(obs, RulesConfig())
        assert [dealt.moves[place] for place in spot.history] == earlier.chosen.tolist()


def test_replay_draws_its_share_of_the_decisions_from_the_seed(games: Path):
    part = load_part(part_path(games, 0))
    every = [s.step for s in replay(part, tiny(games, label_share=1.0))[1]]
    some = [s.step for s in replay(part, tiny(games, label_share=0.3))[1]]
    assert set(some) < set(every)
    assert 0.15 < len(some) / len(every) < 0.45
    assert some == [s.step for s in replay(part, tiny(games, label_share=0.3))[1]]
    assert some != [s.step for s in replay(part, tiny(games, label_share=0.3, seed=7))[1]]


def test_replay_refuses_a_part_that_does_not_replay_to_its_scores(games: Path):
    part = dict(load_part(part_path(games, 0)))
    part["scores"] = part["scores"] + 1
    with pytest.raises(ValueError, match="does not replay to its recorded scores"):
        replay(part, tiny(games))


# A run.


@pytest.fixture(scope="module")
def taught(games: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[LessonConfig, Path]:
    config = tiny(games)
    out = tmp_path_factory.mktemp("lessons") / "out"
    summary = run(config, out)
    assert summary["parts"] == BOLAS
    assert summary["missing_parts"] == 0
    assert summary["lessons"] > 10
    return config, out


def test_a_run_saves_one_lesson_part_per_recording_part(taught: tuple[LessonConfig, Path]):
    config, out = taught
    assert [path.name for path in L.lesson_paths(out)] == ["lessons-00000.npz", "lessons-00001.npz"]
    total = 0
    for first in range(BOLAS):
        lessons = L.load_lessons(L.lesson_path(out, first))
        count = len(lessons["pick"])
        total += count
        assert set(lessons) == set(L.DTYPES)
        for name, kind in L.DTYPES.items():
            assert lessons[name].dtype == kind
            width = (config.max_options,) if name in L.PER_OPTION else ()
            assert lessons[name].shape == ((count, L.VIEW) if name == "x" else (count, *width))
        assert (lessons["game"] == first).all()
    assert total == json.loads((out / "summary.json").read_text())["lessons"]
    assert json.loads((out / "config.json").read_text())["deals"] == list(DEALS)


def test_each_lesson_is_a_drawn_decision_as_its_seat_saw_it(
    games: Path, taught: tuple[LessonConfig, Path]
):
    config, out = taught
    part = load_part(part_path(games, 0))
    by_step = {choice.step: choice for choice in choices(part)}
    drawn = [spot.step for spot in replay(part, config)[1]]
    lessons = L.load_lessons(L.lesson_path(out, 0))
    assert lessons["step"].tolist() == drawn  # two or more options everywhere: all were taught
    for row, step in enumerate(lessons["step"].tolist()):
        choice = by_step[step]
        assert lessons["seat"][row] == choice.obs.seat
        assert lessons["phase"][row] == int(choice.obs.phase)
        view = raw_view(choice.obs, choice.legal).astype(np.float16)
        assert lessons["x"][row].tolist() == view.tolist()
        options = [a for a in lessons["options"][row].tolist() if a >= 0]
        assert options[0] == choice.action  # slot 0 is the move the player made
        assert len(set(options)) == len(options) >= 2
        assert set(options) <= set(choice.legal)
        assert 0.0 <= lessons["tricks"][row] <= 13.0


def test_each_lesson_holds_the_teachers_verdict_per_option(taught: tuple[LessonConfig, Path]):
    config, out = taught
    lessons = L.load_all(out)
    used = lessons["options"] >= 0
    rows = np.arange(len(lessons["pick"]))
    assert (lessons["deals"][:, 0] == sum(DEALS)).all()  # the favourite plays every deal
    assert (lessons["deals"][used] >= DEALS[0]).all()
    assert not lessons["deals"][~used].any()
    assert (lessons["rank"][:, 0] == 0).all()
    assert (lessons["chance"][:, 0] > 0).all()
    assert (lessons["chance"] >= 0).all()  # a random net's long shots round to 0
    assert (np.diff(lessons["chance"], axis=1)[used[:, 1:]] <= 0).all()  # by the player's chances
    assert not lessons["margin"][:, 0].any()
    assert not lessons["opening"][:, 0].any()
    assert (lessons["error"][used] >= 0).all()
    assert np.isfinite(lessons["value"]).all()
    assert used[rows, lessons["pick"]].all()
    assert (lessons["deals"][rows, lessons["pick"]] == sum(DEALS)).all()
    assert 0 < (lessons["pick"] != 0).sum() < len(rows)  # a random net is often overruled
    assert used.sum(1).max() == config.max_options


def test_a_finished_run_resumes_with_nothing_to_do_and_refuses_other_settings(
    taught: tuple[LessonConfig, Path],
):
    config, out = taught
    before = {path.name: path.stat().st_mtime_ns for path in L.lesson_paths(out)}
    assert run(config, out)["parts"] == BOLAS
    assert {path.name: path.stat().st_mtime_ns for path in L.lesson_paths(out)} == before
    with pytest.raises(SystemExit, match="other settings"):
        run(dataclasses.replace(config, hand_margin=2.0), out)
    with pytest.raises(ValueError, match="order must be"):
        run(config, out, order="sideways")


def test_a_shared_run_keeps_the_other_machines_part_and_leaves_missing_recordings(
    taught: tuple[LessonConfig, Path], tmp_path: Path
):
    """Three one-Bola parts going down: the last has no recording here yet, the middle one's
    lessons came from the other machine (kept as they came), so this machine teaches the first."""
    config, out = taught
    games = tmp_path / "games"
    (games / "parts").mkdir(parents=True)
    source = Path(config.record)
    recorded = json.loads((source / "config.json").read_text())
    (games / "config.json").write_text(json.dumps({**recorded, "games": 3}))
    for first in range(BOLAS):
        part_path(games, first).write_bytes(part_path(source, first).read_bytes())
    shared = tmp_path / "shared"
    (shared / "parts").mkdir(parents=True)
    theirs = L.lesson_path(shared, 1)
    theirs.write_bytes(L.lesson_path(out, 1).read_bytes())
    stamp = theirs.stat().st_mtime_ns
    summary = run(dataclasses.replace(config, record=str(games)), shared, order="down")
    assert (summary["parts"], summary["missing_parts"]) == (2, 1)
    assert theirs.stat().st_mtime_ns == stamp
    mine = L.load_lessons(L.lesson_path(shared, 0))
    assert mine["value"].tolist() == L.load_lessons(L.lesson_path(out, 0))["value"].tolist()


def test_only_a_recording_of_the_one_nets_own_search_is_taught(record: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="own=true and judge=round"):
        Labeler(LessonConfig(record=str(record)))  # the shared fixture: a Hybrid's recording
    with pytest.raises(ValueError, match=r"config\.json is missing"):
        Labeler(LessonConfig(record=str(tmp_path)))


# The config.


def test_a_lessons_config_is_one_checked_section(games: Path):
    section = {"record": str(games), "deals": [8, 16], "min_options": 2, "max_options": 3}
    config = config_of({"lessons": section})
    assert config.deals == (8, 16)
    assert config.pruning.max_options == 3
    with pytest.raises(ValueError, match=r"lessons\.record"):
        config_of({"lessons": {"deals": [8, 16]}})
    with pytest.raises(ValueError, match="only \\[lessons\\]"):
        config_of({"lessons": {"record": "x"}, "copy": {}})
    with pytest.raises(ValueError, match="unknown key"):
        config_of({"lessons": {"record": "x", "samples": 3}})
    with pytest.raises(ValueError, match="label_share"):
        LessonConfig(record="x", label_share=0.0)
    with pytest.raises(ValueError, match="stages narrow at most 3 options"):
        LessonConfig(record="x", deals=(8, 16))  # the default 8 options need 4 stages
    with pytest.raises(ValueError, match="batch"):
        LessonConfig(record="x", batch=0)


# The timing and noise test.


def lessons_of(
    pick: list[int], margin: list[float], error: list[float], opening: list[float]
) -> L.Lessons:
    """Lessons with two slots, the favourite and one alternative with these numbers."""
    count = len(pick)
    blank = L.empty(2)
    lessons = {k: np.zeros((count, *blank[k].shape[1:]), L.DTYPES[k]) for k in L.DTYPES}
    lessons["options"][:] = [3, 7]
    lessons["deals"][:] = [6, 6]
    lessons["pick"][:] = pick
    lessons["margin"][:, 1], lessons["error"][:, 1], lessons["opening"][:, 1] = (
        margin,
        error,
        opening,
    )
    return lessons


def test_noise_compares_two_teachings_of_the_same_decisions():
    first = lessons_of([1, 1, 0, 1], [3.0, 0.5, -1.0, 2.0], [1.0, 1.0, 1.0, 0.5], [4.0, 1.0, -2, 3])
    second = lessons_of([1, 0, 0, 1], [2.0, -0.5, -2.0, 1.0], [1.0] * 4, [2.0, -3.0, -1.0, 1.0])
    first["rank"][:, 1] = [1, 4, 1, 3]
    first["phase"][:] = [0, 1, 1, 2]
    report = noise(first, second)
    assert report["decisions"] == 4
    assert report["same_pick"] == 0.75
    assert report["changed"] == 0.75  # the first teaching leaves the favourite at 3 of 4
    assert report["changed_again"] == pytest.approx(2 / 3)  # and the second agrees at 2 of them
    # A pick's worth is its lead on the other teaching's first deals: (3 - 1.5 + 0 + 2) / 4.
    assert report["gain"] == pytest.approx(0.875)
    assert report["gain_error"] == pytest.approx(np.std([3.0, -1.5, 0.0, 2.0], ddof=1) / 2)
    assert report["outside_top_3"] == 0.5  # its picks' places: 1, 4, 0 and 3
    assert report["lead_when_changed"] == pytest.approx((3.0 + 0.5 + 2.0) / 3)
    assert report["options"] == 2.0
    assert report["play_outs"] == 12.0
    # Counting only leads of 2 standard errors: 1, 0, 0, 1 against 1, 0, 0, 0.
    careful = report["gates"]["2.0"]
    assert (careful["same_pick"], careful["changed"], careful["changed_again"]) == (0.75, 0.5, 0.5)
    assert careful["gain"] == pytest.approx((3.0 + 0.0 + 0.0 + 0.5) / 4)
    assert report["phases"]["Dash Call"]["decisions"] == 1
    assert report["phases"]["auction"]["same_pick"] == 0.5
    assert report["phases"]["estimate"]["changed_again"] == 1.0


def test_gated_counts_an_alternative_only_past_its_standard_errors():
    lessons = lessons_of([1, 1, 0], [3.0, 0.5, -1.0], [1.0, 1.0, 1.0], [0.0] * 3)
    assert gated(lessons, 0.0).tolist() == [1, 1, 0]
    assert gated(lessons, 1.0).tolist() == [1, 0, 0]
    assert gated(lessons, 4.0).tolist() == [0, 0, 0]


def test_noise_refuses_teachings_of_different_options():
    first = lessons_of([0], [0.0], [1.0], [0.0])
    second = lessons_of([0], [0.0], [1.0], [0.0])
    second["options"][0, 1] = 9
    with pytest.raises(ValueError, match="different options"):
        noise(first, second)


def test_the_timing_test_teaches_the_same_decisions_twice(games: Path, tmp_path: Path):
    report = time_run(tiny(games), 12, keep=tmp_path / "kept")
    assert report["taught"] == report["decisions"] == 12
    assert report["seconds"] > 0
    assert report["per_hour"] == pytest.approx(12 / report["seconds"] * 3600)
    assert 0.0 <= report["same_pick"] <= 1.0
    first = L.load_lessons(tmp_path / "kept" / "first.npz")
    second = L.load_lessons(tmp_path / "kept" / "second.npz")
    assert first["options"].tolist() == second["options"].tolist()
    assert first["value"].tolist() != second["value"].tolist()  # fresh deals
    lines = noise_lines(report)
    assert lines[0].startswith("12 decisions in ")
    assert any(line.startswith("same pick on fresh deals: ") for line in lines)


# The layout.


def test_lessons_round_trip_and_join_in_order(tmp_path: Path):
    (tmp_path / "parts").mkdir()
    one = lessons_of([1, 0], [1.0, 2.0], [1.0, 1.0], [0.0, 0.0])
    two = lessons_of([0], [3.0], [1.0], [0.0])
    L.save_lessons(L.lesson_path(tmp_path, 25), two)
    L.save_lessons(L.lesson_path(tmp_path, 0), one)
    loaded = L.load_lessons(L.lesson_path(tmp_path, 0))
    assert all(loaded[name].tolist() == one[name].tolist() for name in L.DTYPES)
    assert L.load_all(tmp_path)["margin"][:, 1].tolist() == [1.0, 2.0, 3.0]
    assert not list((tmp_path / "parts").glob("*.tmp*"))
    with pytest.raises(ValueError, match="missing"):
        L.save_lessons(tmp_path / "bad.npz", {"pick": one["pick"]})
    with pytest.raises(FileNotFoundError, match="no lessons"):
        L.load_all(tmp_path / "parts")
    assert len(L.empty(3)["pick"]) == 0
    assert L.empty(3)["options"].shape == (0, 3)
