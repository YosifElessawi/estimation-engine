"""Step 6's real-play check: the changes it finds, a round played for real from a recorded
decision, and its report."""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from estimation.bots import make_bot
from estimation.engine import Bola, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.onenet import bid_check as C
from estimation.onenet import lessons as L
from estimation.onenet import nets
from estimation.rl.onenet_record import (
    Arrays,
    PartTask,
    RecordConfig,
    part_path,
    record_part,
    save_part,
)
from tests.onenet.test_search import random_net

TOTAL = 6  # deals of a whole search in these lessons


@pytest.fixture(scope="module")
def games(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A recording of one Bola by a small random player in its own search."""
    folder = tmp_path_factory.mktemp("games")
    config = RecordConfig(
        player=f"onenet:net={random_net(folder)}:own=true:top_k=2:samples=2",
        games=1,
        part_games=1,
        label_share=0.0,
        teacher_samples=3,
        seed=5,
        workers=1,
    )
    (folder / "parts").mkdir()
    arrays = record_part(PartTask(0, 1, config))
    assert arrays is not None
    save_part(part_path(folder, 0), arrays)
    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    return folder


def part_of(games: Path) -> Arrays:
    with np.load(part_path(games, 0)) as data:
        return {name: data[name] for name in data.files}


def bidding(part: Arrays) -> list[tuple[int, int, int, list[int]]]:
    """The Bola's bidding decisions with a choice: step, seat, the bid made, the legal bids."""
    bola = Bola(int(part["seed"][0]), DEFAULT_CONFIG)
    found: list[tuple[int, int, int, list[int]]] = []
    for step, action in enumerate(part["actions"][: int(part["log_start"][1])].tolist()):
        legal = bola.legal_actions()
        seat = bola.current_player
        assert seat is not None
        if bola.phase is not Phase.PLAY and len(legal) > 1:
            found.append((step, seat, action, legal))
        bola.step(action)
    return found


def lessons_of(options: list[list[int]]) -> L.Lessons:
    """Lessons with these option slots (the bid made first), whose finalist (slot 1) sat a whole
    exam; Bola 0, steps 10, 20, 30 and so on."""
    count = len(options)
    empty = L.empty(len(options[0]))
    lessons = {k: np.zeros((count, *empty[k].shape[1:]), L.DTYPES[k]) for k in L.DTYPES}
    lessons["options"][:] = options
    lessons["step"][:] = 10 * (1 + np.arange(count))
    lessons["deals"][:] = TOTAL
    lessons["finalist"][:] = 1
    lessons["exam_deals"][:] = 4
    return lessons


# The changes to check.


def test_a_source_changes_the_lessons_where_its_bid_is_not_the_bid_made():
    lessons = lessons_of([[3, 7, 12]] * 4)
    lessons["seat"][:] = [0, 1, 2, 3]
    lessons["phase"][:] = [0, 1, 2, 2]
    lessons["opening"][:] = [[0.0, 1.5, -2.0]] * 4
    lessons["exam"][:] = [4.0, 5.0, 6.0, 7.0]
    lessons["deals"][2] = [2, 2, 2]  # its finalist never sat a whole exam
    first = C.Source("first", np.array([3, 7, 7, 12]))
    second = C.Source("second", np.array([3, 7, 3, 40]))  # 40: a bid the teacher never tried
    changes = C.find_changes(lessons, [first, second])
    # The first lesson changes for nobody; the second for both, to the same bid: one row.
    assert changes["step"].tolist() == [20, 30, 40, 40]
    assert changes["seat"].tolist() == [1, 2, 3, 3]
    assert changes["phase"].tolist() == [1, 2, 2, 2]
    assert changes["own"].tolist() == [3, 3, 3, 3]
    assert changes["other"].tolist() == [7, 7, 12, 40]
    assert changes["source"].tolist() == [[True, True], [True, False], [True, False], [False, True]]
    assert changes["tried"].tolist() == [True, True, True, False]
    assert changes["opening"][:3].tolist() == pytest.approx([1.5, 1.5, -2.0])
    assert np.isnan(changes["opening"][3])
    # The exam lead is the finalist's alone, where it sat its whole exam.
    assert changes["examined"].tolist() == [True, False, False, False]
    assert changes["exam"][0] == pytest.approx(5.0)
    assert np.isnan(changes["exam"][1:]).all()
    assert C.decisions(changes) == 3


def test_no_changes_when_every_source_makes_the_bid_made():
    lessons = lessons_of([[3, 7, 12]] * 2)
    changes = C.find_changes(lessons, [C.Source("same", np.array([3, 3]))])
    assert len(changes["game"]) == 0
    assert C.decisions(changes) == 0


def test_a_picks_file_names_its_bids_by_bola_and_step():
    lessons = lessons_of([[3, 7, 12]] * 3)
    lessons["game"][:] = [4, 4, 5]
    picks = {
        "game": np.array([5, 4, 9]),
        "step": np.array([30, 10, 10]),
        "action": np.array([12, 7, 50]),
    }
    # The lesson at Bola 4, step 20 is not named: its bid stays; Bola 9 has no lesson.
    assert C.picks_from(lessons, picks).tolist() == [7, 3, 12]
    nothing = {name: np.zeros(0, np.int64) for name in picks}
    assert C.picks_from(lessons, nothing).tolist() == [3, 3, 3]


def test_keep_changes_finds_the_teachers_and_a_players_changes_on_the_held_out_bolas(
    tmp_path: Path,
):
    lessons = lessons_of([[54, 55, 56]] * 8)
    lessons["game"][:] = [0, 0, 1, 1, 2, 2, 3, 3]
    lessons["phase"][:] = 2
    lessons["x"][:, L.VIEW - 120 + 54 : L.VIEW - 120 + 57] = 1.0  # the legal bids
    lessons["exam"][:] = [3.0, -3.0] * 4  # the teacher's careful pick: the finalist, then no
    lessons["exam_error"][:] = 1.0
    folder = tmp_path / "lessons"
    (folder / "parts").mkdir(parents=True)
    L.save_lessons(L.lesson_path(folder, 0), lessons)
    (folder / "config.json").write_text(
        json.dumps({"record": "the/games", "label_share": [0.25, 0.5, 1.0]})
    )
    player = tmp_path / "player.npz"
    nets.save_player(player, nets.init_player(jax.random.PRNGKey(1), 16, 1), {})
    picks = tmp_path / "picks.npz"
    np.savez(picks, game=lessons["game"], step=lessons["step"], action=np.full(8, 56))
    out = tmp_path / "check"
    changes = C.keep_changes(out, folder, [("net", player)], [("file", picks)], 0.5, 3)
    config = json.loads((out / C.CONFIG).read_text())
    assert config["sources"] == ["teacher", "net", "file"]
    assert config["record"] == "the/games"
    assert config["lessons_by_phase"] == [0, 0, 4]  # half of the Bolas are held out
    assert config["per_seat_game"] == pytest.approx([0.0, 0.0, 8 / 1.0 / 4 / 4])
    assert len(np.unique(changes["game"])) == 2
    teacher, file = changes["source"][:, 0], changes["source"][:, 2]
    assert changes["other"][teacher].tolist() == [55, 55]  # the lessons whose finalist led
    assert changes["other"][file].tolist() == [56] * 4  # every held-out lesson
    saved = C._load(out / C.CHANGES)  # pyright: ignore[reportPrivateUsage]
    assert saved["other"].tolist() == changes["other"].tolist()
    with pytest.raises(ValueError, match="different names"):
        C.keep_changes(out, folder, [("teacher", player)], [], 0.5, 3)


# Playing a round for real.


def test_a_round_played_twice_from_the_same_seeds_is_the_same_round(games: Path):
    part = part_of(games)
    spec = json.loads((games / "config.json").read_text())["player"]
    actions = part["actions"][: int(part["log_start"][1])].tolist()
    step, _seat, own, legal = bidding(part)[3]
    seed = int(part["seed"][0])

    def play(action: int) -> tuple[C.Scores, C.Scores]:
        bots = [make_bot(spec, 100 + seat) for seat in range(4)]
        return C.play_round(seed, actions, step, action, bots)

    first, again = play(own), play(own)
    assert first == again
    # The standings are the Bola's as the round began; the round is one whole round's scores.
    bola = Bola(seed, DEFAULT_CONFIG)
    for action in actions[:step]:
        bola.step(action)
    assert first[0] == bola.scores
    assert len(first[1]) == 4
    other = next(action for action in legal if action != own)
    assert len(play(other)[1]) == 4  # another bid plays to the round's end too
    with pytest.raises(ValueError, match="not legal"):
        play(next(action for action in range(120) if action not in legal))


def test_check_bola_plays_the_bid_made_once_and_then_each_other_bid(games: Path):
    part = part_of(games)
    spec = json.loads((games / "config.json").read_text())["player"]
    found = bidding(part)
    (step, seat, own, legal), (later, later_seat, later_own, later_legal) = found[0], found[5]
    others = [action for action in legal if action != own][:2]
    later_other = next(action for action in later_legal if action != later_own)
    rows = [(later, later_seat, later_own, later_other)]
    rows += [(step, seat, own, other) for other in others]
    steps, seats, owns, tries = (tuple(column) for column in zip(*rows, strict=True))
    task = C.BolaTask(str(games), 0, 0, spec, 7, steps, seats, owns, tries)
    assert task.plays() == {(later, later_own), (later, later_other), (step, own)} | {
        (step, other) for other in others
    }
    plays = C.check_bola(task)
    # Each decision's bid made first, the decisions in the Bola's order.
    assert plays["step"].tolist() == [step] * (1 + len(others)) + [later, later]
    assert plays["action"].tolist() == [own, *others, later_own, later_other]
    assert plays["result"].shape == plays["standings"].shape == (len(others) + 3, 4)
    again = C.check_bola(task)
    assert again["result"].tolist() == plays["result"].tolist()  # seeded from the decision
    rest = C.check_bola(dataclasses.replace(task, done=((step, own), (later, later_other))))
    assert rest["action"].tolist() == [*others, later_own]  # the saved plays are left out
    wrong = dataclasses.replace(task, owns=(later_other, *owns[1:]))
    with pytest.raises(ValueError, match="did not play move"):
        C.check_bola(wrong)
    with pytest.raises(ValueError, match="does not decide"):
        C.check_bola(dataclasses.replace(task, seats=((later_seat + 1) % 4, *seats[1:])))
    with pytest.raises(ValueError, match="is not in the part"):
        C.check_bola(dataclasses.replace(task, game=3))


# The report.


def test_a_rounds_worth_is_the_own_score_less_the_others_players_ahead_double():
    # Seat 1 is ahead of seat 0 and counts double: weights 1/2, 1/4, 1/4.
    assert C.worth((20, 8, 12, 4), (0, 10, -5, 0), 0) == pytest.approx(20 - (4 + 3 + 1))
    # Nobody is ahead of seat 1: the plain mean of the others.
    assert C.worth((20, 8, 12, 4), (0, 10, -5, 0), 1) == pytest.approx(8 - 12)


def changes_of(rows: list[tuple[int, int, int, int, int, list[bool], float]]) -> Arrays:
    """Changes in Bola 0 from rows of step, seat, phase, own, other, sources and the teacher's
    first-stage lead (nan: a bid it never tried)."""
    opening = np.array([row[6] for row in rows], np.float32)
    return {
        "game": np.zeros(len(rows), np.int32),
        "step": np.array([row[0] for row in rows], np.int32),
        "seat": np.array([row[1] for row in rows], np.int8),
        "phase": np.array([row[2] for row in rows], np.int8),
        "own": np.array([row[3] for row in rows], np.int16),
        "other": np.array([row[4] for row in rows], np.int16),
        "source": np.array([row[5] for row in rows], bool),
        "tried": ~np.isnan(opening),
        "opening": opening,
        "examined": np.zeros(len(rows), bool),
        "exam": np.full(len(rows), np.nan, np.float32),
    }


def test_a_real_lead_is_the_other_bids_round_less_the_made_bids_round():
    changes = changes_of(
        [
            (10, 0, 2, 54, 55, [True], 1.0),
            (20, 2, 2, 56, 55, [True], 1.0),
            (30, 1, 1, 53, 7, [True], 1.0),
        ]
    )
    level: C.Scores = (0, 0, 0, 0)
    plays: dict[tuple[int, int, int], tuple[C.Scores, C.Scores]] = {
        (0, 10, 54): (level, (12, 0, 0, 0)),
        (0, 10, 55): (level, (18, 3, 3, 3)),
        (0, 20, 56): (level, (0, 0, -6, 0)),
        (0, 20, 55): (level, (0, 0, 10, 0)),
        (0, 30, 53): (level, (0, 5, 0, 0)),  # its other bid was never played
    }
    lead, plain = C.real_leads(changes, plays)
    # Seat 0: 18 - 3 against 12. Seat 2: 10 against -6.
    assert lead[:2].tolist() == pytest.approx([3.0, 16.0])
    assert plain[:2].tolist() == pytest.approx([6.0, 16.0])
    assert np.isnan(lead[2])
    assert np.isnan(plain[2])


def test_the_summary_counts_each_sources_changes_for_real_and_in_the_teachers_rounds():
    changes = changes_of(
        [
            (10, 0, 2, 54, 55, [True, False], 2.0),
            (20, 0, 2, 54, 55, [True, True], 4.0),
            (30, 0, 1, 53, 7, [True, False], 1.0),
            (40, 0, 2, 54, 60, [False, True], float("nan")),
            (50, 0, 2, 54, 55, [True, False], 9.0),  # not played
        ]
    )
    lead = np.array([4.0, 0.0, -3.0, 5.0, np.nan])
    plain = np.array([6.0, 2.0, -1.0, 5.0, np.nan])
    config = {
        "sources": ["one", "two"],
        "lessons_by_phase": [10, 20, 30],
        "per_seat_game": [13.0, 20.0, 15.0],
    }
    summary = C.summarize(changes, lead, plain, config)
    assert summary["played"] == 4
    assert summary["changes"] == 5
    one = summary["sources"]["one"]
    assert one["changes"] == 4
    assert one["played"] == 3
    assert one["real"] == pytest.approx(1 / 3)
    assert one["own_score"] == pytest.approx(7 / 3)
    assert one["imagined"] == pytest.approx(7 / 3)
    estimates = one["phases"]["estimates"]
    assert estimates["played"] == 2
    assert estimates["rate"] == pytest.approx(3 / 30)  # the change not played counts in the rate
    assert estimates["real"] == pytest.approx(2.0)
    assert estimates["real_error"] == pytest.approx(2.0)
    assert estimates["imagined"] == pytest.approx(3.0)
    auction = one["phases"]["the auction"]
    assert auction["rate"] == pytest.approx(1 / 20)
    assert auction["real"] == pytest.approx(-3.0)
    # A game: 0.1 of 15 estimates at +2, and 0.05 of 20 auction decisions at -3.
    assert one["game"] == pytest.approx(0.1 * 15 * 2.0 + 0.05 * 20 * -3.0)
    assert one["game_error"] == pytest.approx(0.1 * 15 * 2.0)
    assert one["imagined_game"] == pytest.approx(0.1 * 15 * 3.0 + 0.05 * 20 * 1.0)
    two = summary["sources"]["two"]
    assert two["played"] == 2
    assert two["real"] == pytest.approx(2.5)
    assert two["imagined"] == pytest.approx(4.0)  # only where the teacher tried the bid
    assert two["real_where_imagined"] == pytest.approx(0.0)
    text = "\n".join(C.lines(summary))
    assert "4 of 5 changes played for real" in text
    assert "one: 3 changes played" in text
    assert "estimates: 2 played, changed at 10.0% of decisions" in text


def test_a_run_plays_every_bola_once_and_reports(games: Path, tmp_path: Path):
    part = part_of(games)
    found = [decision for decision in bidding(part) if len(decision[3]) >= 3]
    rows = []
    for step, seat, own, legal in (found[0], found[1]):
        other = next(action for action in legal if action != own)
        rows.append((step, seat, 1, own, other, [True], 0.5))
    out = tmp_path / "check"
    out.mkdir()
    np.savez_compressed(out / C.CHANGES, allow_pickle=False, **changes_of(rows))
    config = {
        "record": str(games),
        "sources": ["one"],
        "lessons_by_phase": [0, 10, 0],
        "per_seat_game": [0.0, 20.0, 0.0],
    }
    (out / C.CONFIG).write_text(json.dumps(config))
    summary = C.run(out, workers=1, seed=3)
    assert summary["played"] == 2
    assert summary["sources"]["one"]["played"] == 2
    assert json.loads((out / C.SUMMARY).read_text()) == summary
    path = C.check_path(out, 0)
    plays = C._load(path)  # pyright: ignore[reportPrivateUsage]
    assert plays["action"].tolist() == [rows[0][3], rows[0][4], rows[1][3], rows[1][4]]
    stamp = path.stat().st_mtime_ns
    assert C.run(out, workers=1, seed=3) == summary  # nothing is played twice
    assert path.stat().st_mtime_ns == stamp
    assert C.main(["report", "--out", str(out)]) == 0
    # A source added later: another bid at the first decision, and a third decision. Only the
    # three new plays are played, after the four that were saved.
    step, seat, own, legal = found[2]
    more = next(action for action in found[0][3] if action not in (rows[0][3], rows[0][4]))
    other = next(action for action in legal if action != own)
    rows += [(rows[0][0], rows[0][1], 1, rows[0][3], more, [True], 0.5)]
    rows += [(step, seat, 1, own, other, [True], 0.5)]
    np.savez_compressed(out / C.CHANGES, allow_pickle=False, **changes_of(rows))
    grown = C.run(out, workers=1, seed=3)
    assert grown["played"] == 4
    again = C._load(path)  # pyright: ignore[reportPrivateUsage]
    assert again["action"][:4].tolist() == plays["action"].tolist()
    assert again["result"][:4].tolist() == plays["result"].tolist()
    assert again["action"][4:].tolist() == [more, own, other]
    assert again["step"][4:].tolist() == [rows[0][0], step, step]


# What really became of the taught estimates.


def test_the_outcomes_are_what_the_taught_estimates_really_took(games: Path, tmp_path: Path):
    part = part_of(games)
    made = C.estimates_made(games)
    actions = part["actions"][: int(part["log_start"][1])].tolist()
    assert len(made["step"]) == sum(54 <= action < 68 for action in actions)  # every estimate
    chosen = [(step, seat, own) for step, seat, own, _ in bidding(part) if own >= 54]
    assert len(chosen) > 8
    # A lesson at every estimate with a choice: the teacher tried one trick more. At every
    # other one that led its exam by 5 standard errors; at the rest it trailed.
    lessons = lessons_of([[own, own + 1] for _, _, own in chosen])
    lessons["step"][:] = [step for step, _, _ in chosen]
    lessons["phase"][:] = 2
    lessons["exam"][:] = np.where(np.arange(len(chosen)) % 2 == 0, 5.0, -5.0)
    lessons["exam_error"][:] = 1.0
    lessons["value"][:, 0] = 2.0
    # The same from the engine alone: each estimate's round, replayed to its end.
    bola = Bola(int(part["seed"][0]), DEFAULT_CONFIG)
    at: dict[int, int] = {}
    for step, action in enumerate(actions):
        at[step] = len(bola.history)
        bola.step(action)
    stood, off, worths = [], [], []
    for step, seat, own in chosen:
        result = bola.history[at[step]]
        mine = result.seats[seat]
        stood.append(not result.skipped and mine.estimate == own - 54)
        off.append(mine.tricks - (own - 54))
        replay = Bola(int(part["seed"][0]), DEFAULT_CONFIG)
        for action in actions[:step]:
            replay.step(action)
        worths.append(C.worth(result.scores, replay.scores, seat))
    stood, off, worths = np.array(stood), np.array(off), np.array(worths)
    led = np.arange(len(chosen)) % 2 == 0
    summary = C.outcomes(lessons, made)
    assert summary["estimates"] == stood.sum() > 4
    higher, kept = summary["groups"]["+1"], summary["groups"]["kept"]
    assert higher["estimates"] == (stood & led).sum()
    assert kept["estimates"] == (stood & ~led).sum()
    assert summary["groups"]["-1"]["estimates"] == 0  # no lesson tried a trick fewer
    assert higher["made"] == pytest.approx((off[stood & led] == 0).mean())
    assert higher["short"] == pytest.approx((off[stood & led] == -1).mean())
    assert higher["over"] == pytest.approx((off[stood & led] == 1).mean())
    assert higher["real"] == pytest.approx(worths[stood & led].mean())
    assert kept["real"] == pytest.approx(worths[stood & ~led].mean())
    assert (higher["imagined"], higher["lead"]) == (pytest.approx(2.0), pytest.approx(5.0))
    assert kept["lead"] == pytest.approx(-5.0)
    text = "\n".join(C.outcome_lines(summary))
    assert f"{stood.sum()} taught estimates stood" in text
    assert "the teacher would go +1" in text
    # The command reads the recording the lessons name.
    folder = tmp_path / "lessons"
    (folder / "parts").mkdir(parents=True)
    L.save_lessons(L.lesson_path(folder, 0), lessons)
    (folder / "config.json").write_text(json.dumps({"record": str(games)}))
    assert C.main(["outcomes", "--lessons", str(folder)]) == 0
    assert C.outcomes(lessons, {name: values[:0] for name, values in made.items()}) == {
        "estimates": 0,
        "groups": {},
    }
