"""Step 6's learning: the read leads as targets, the card set, the held-out scores, and a run
that learns big leads while the card play stays."""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.onenet import bid_learn as B
from estimation.onenet import lessons as L
from estimation.onenet import nets
from estimation.onenet.copy_hybrid import split as copy_split
from estimation.onenet.search import load_net
from estimation.onenet.shards import KIND_CARD, load_data

PASS, PLAY_BASE = 53, 68
TOTAL = 6  # deals of a whole search in these lessons


@pytest.fixture(scope="module")
def start(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small random player, saved as step 5 saves its players. A fresh player's moves are all
    but even; this one's move output is scaled up, so it has favourites to keep or lose."""
    path = tmp_path_factory.mktemp("start") / "player.npz"
    net = nets.init_player(jax.random.PRNGKey(3), 32, 2)
    sure = {name: values * 3000.0 for name, values in net["policy"].items()}
    nets.save_player(path, {**net, "policy": sure}, {"loop": 0})
    return path


def blank(count: int, slots: int = 3) -> L.Lessons:
    empty = L.empty(slots)
    lessons = {k: np.zeros((count, *empty[k].shape[1:]), L.DTYPES[k]) for k in L.DTYPES}
    lessons["options"][:] = -1
    return lessons


def examined_lessons(exam: list[float], error: list[float]) -> L.Lessons:
    """Lessons whose finalist (slot 1) sat a whole exam with these leads and errors."""
    lessons = blank(len(exam))
    lessons["options"][:] = [3, 7, 12]
    lessons["deals"][:] = [TOTAL, TOTAL, 2]
    lessons["finalist"][:] = 1
    lessons["exam_deals"][:] = 4
    lessons["exam"][:], lessons["exam_error"][:] = exam, error
    return lessons


# The lessons as targets.


def test_examined_lessons_are_those_whose_finalist_sat_its_whole_exam():
    lessons = examined_lessons([1.0, 1.0, 1.0, 1.0], [1.0] * 4)
    lessons["deals"][1] = [2, 2, 2]  # stopped early: every alternative hopeless
    lessons["exam_deals"][2] = 0  # never left alone to sit an exam
    assert L.examined(lessons).tolist() == [True, False, False, True]
    assert L.examined(blank(0)).tolist() == []


def test_a_tilt_is_the_read_lead_less_the_margin_over_the_temperature():
    leads = np.array([[0.0, 3.0, -1.0], [0.0, 0.5, 30.0]], np.float32)
    config = B.LearnConfig(margin=0.5, temperature=2.0, clip=5.0)
    # (3 - 0.5) / 2, (-1 - 0.5) / 2; a lead at the margin tilts nothing; then the clip.
    assert B.tilts(leads, config).tolist() == [[-0.25, 1.25, -0.75], [-0.25, 0.0, 5.0]]


def test_a_lesson_aims_every_tried_option_against_the_players_own_move():
    chances = np.full((4, 120), -np.inf, np.float32)
    chances[:, [3, 7, 12]] = np.log([0.9, 0.08, 0.02])  # the move and two options
    options = jnp.asarray([[3, 7, 12], [3, 7, 12], [3, 7, 12], [3, 7, 0]])
    used = jnp.asarray([[False, True, True]] * 3 + [[False, True, False]])
    tilt = np.array(
        [[0.0, 1.0, 2.0], [0.0, -1.0, -4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 3.0]], np.float32
    )
    aim = np.asarray(B.aimed(jnp.asarray(chances), options, used, jnp.asarray(tilt)))
    odds = np.exp(aim[:, [3, 7, 12]])
    # Read ahead: each overtakes the move, at e and e squared times its chance, however
    # unlikely the player thought it.
    assert odds[0].tolist() == pytest.approx([0.9, 0.9 * np.e, 0.9 * np.e**2], rel=1e-5)
    # Read behind. By less than the player thought (0.08 is under 0.9 / e): left alone. By
    # more (0.02 is over 0.9 / e to the 4th): lowered to that.
    assert odds[1].tolist() == pytest.approx([0.9, 0.08, 0.9 * np.exp(-4.0)], rel=1e-5)
    assert odds[2].tolist() == pytest.approx([0.9, 0.08, 0.02], rel=1e-5)  # level: as it was
    # A slot that does not count changes nothing, whatever its tilt.
    assert odds[3].tolist() == pytest.approx([0.9, 0.9 * np.e, 0.02], rel=1e-5)
    assert np.isneginf(aim[:, 0]).all()  # the other moves are untouched


def test_the_careful_pick_is_the_finalist_only_a_standard_error_ahead():
    lessons = examined_lessons([2.0, 0.5, -3.0, 2.0], [1.0, 1.0, 1.0, 1.0])
    lessons["deals"][3] = [2, 2, 2]
    assert L.careful_pick(lessons).tolist() == [7, 3, 3, 3]


# The card set.


def test_the_held_out_bolas_are_the_copys(data: Path):
    whole = load_data(data)
    _, held_rows = copy_split(whole, 0.34, 11)
    ours = B.held_out_bolas(whole["bola"], 0.34, 11)
    assert sorted(ours.tolist()) == np.unique(whole["bola"][held_rows]).tolist()


def test_the_card_set_holds_training_card_views_and_scores_the_held_out_valued_ones(data: Path):
    config = B.LearnConfig(card_held_out=0.34, card_seed=11)
    cards = B.load_cards(data, config)
    whole = load_data(data)
    aside = np.isin(whole["bola"], B.held_out_bolas(whole["bola"], 0.34, 11))
    is_card = whole["action"] >= PLAY_BASE
    assert len(cards.held) == int((~aside & is_card).sum()) > 100
    assert (cards.held[:, -120:][:, :PLAY_BASE] == 0).all()  # card views: no bid is legal
    valued = aside & (whole["kind"] == KIND_CARD)
    assert len(cards.scored) == len(cards.values) == int(valued.sum()) > 10
    assert cards.values.tolist() == whole["values"][valued].tolist()
    with pytest.raises(FileNotFoundError, match="no shards"):
        B.load_cards(data / "nowhere", config)


def test_card_scores_are_the_points_the_top_cards_lose(start: Path, data: Path):
    cards = B.load_cards(data, B.LearnConfig(card_held_out=0.34, card_seed=11))
    player, _ = nets.load_player(start)
    scores = B.card_scores(player, cards)
    logits = load_net(str(start))(cards.scored.astype(np.float32))
    legal = cards.scored[:, -120:] > 0.5
    order = np.argsort(-np.where(legal, logits, -np.inf), axis=1)
    best = np.where(legal, cards.values, -np.inf).max(1)
    lost = np.zeros((len(best), 3))
    for row in range(len(best)):
        tops = [a for a in order[row, :3] if legal[row, a]]
        for k in range(3):
            lost[row, k] = best[row] - cards.values[row, tops[: k + 1]].max()
    assert scores["cards"] == len(best)
    assert scores["top_card"] == pytest.approx(lost[:, 0].mean(), abs=1e-4)
    assert scores["top_2"] == pytest.approx(lost[:, 1].mean(), abs=1e-4)
    assert scores["top_3"] == pytest.approx(lost[:, 2].mean(), abs=1e-4)
    assert scores["top_card"] >= scores["top_2"] >= scores["top_3"] >= 0
    # A player whose own order is the teacher's loses nothing.
    own = np.where(legal, logits, 0.0).astype(np.float32)
    assert B.card_scores(player, cards._replace(values=own))["top_card"] == 0.0
    assert B.card_scores(player, B.Cards(cards.held, cards.scored[:0], cards.values[:0])) == {
        "cards": 0.0
    }


# Lessons on real bidding views.


def bidding_lessons(start: Path, data: Path) -> L.Lessons:
    """Lessons on the training set's bidding views with three or more options: slot 0 is the
    starting player's move there, slots 1 and 2 its next favourites; slot 1 is the finalist and
    sat a whole exam, slot 2 was dropped after the first stage."""
    whole = load_data(data)
    legal = whole["x"][:, -120:] > 0.5
    rows = np.flatnonzero((whole["action"] < PLAY_BASE) & (legal.sum(1) >= 3))
    logits = load_net(str(start))(whole["x"][rows].astype(np.float32))
    order = np.argsort(-np.where(legal[rows], logits, -np.inf), axis=1)[:, :3]
    lessons = blank(len(rows))
    lessons["game"][:] = whole["bola"][rows]
    lessons["x"][:] = whole["x"][rows]
    lessons["options"][:] = order
    lessons["deals"][:] = [TOTAL, TOTAL, 2]
    lessons["finalist"][:] = 1
    lessons["exam_deals"][:] = 4
    lessons["exam_error"][:] = 0.1
    return lessons


def test_bid_scores_value_the_changed_bids_by_the_teachers_measures(start: Path, data: Path):
    lessons = bidding_lessons(start, data)
    count = len(lessons["pick"])
    assert count > 30
    lessons["exam"][:] = 4.0
    lessons["opening"][:, 1] = 3.0
    lessons["opening"][:, 2] = -6.0
    player, _ = nets.load_player(start)
    same = B.bid_scores(player, lessons)  # the player's top bid is slot 0 everywhere
    assert (same["changed"], same["gain"], same["untried"]) == (0.0, 0.0, 0.0)
    assert same["exam_gain"] == 0.0
    assert same["overruled"] == 1.0  # the finalist leads by 40 standard errors everywhere
    assert (same["agree"], same["followed"]) == (0.0, 0.0)
    # Lessons that call another option the starting player's move: the player has "changed" to
    # what is now the finalist (its first-stage lead, or its exam lead by the exams) or the
    # dropped option (its first-stage lead by both).
    moved = {name: values.copy() for name, values in lessons.items()}
    moved["options"][:, [0, 1]] = lessons["options"][:, [1, 0]]
    to_finalist = B.bid_scores(player, moved)
    assert (to_finalist["changed"], to_finalist["gain"]) == (1.0, pytest.approx(3.0))
    assert to_finalist["exam_gain"] == pytest.approx(4.0)
    assert (to_finalist["agree"], to_finalist["followed"]) == (1.0, 1.0)
    moved["options"][:, [1, 2]] = moved["options"][:, [2, 1]]  # the player's bid is slot 2 now
    to_dropped = B.bid_scores(player, moved)
    assert (to_dropped["changed"], to_dropped["gain"]) == (1.0, pytest.approx(-6.0))
    assert to_dropped["exam_gain"] == pytest.approx(-6.0)
    assert to_dropped["gain_error"] == 0.0
    moved["options"][:, 2] = -1  # the player's bid is not among the options the teacher tried
    assert B.bid_scores(player, moved)["untried"] == 1.0
    # Any bids are scored the same way, a player's or not.
    slot_1 = lessons["options"][:, 1].astype(np.int64)
    assert B.score_bids(slot_1, lessons)["gain"] == pytest.approx(3.0)


def test_bid_scores_by_kind_of_decision_and_in_points_a_game(start: Path, data: Path):
    lessons = bidding_lessons(start, data)
    count = len(lessons["pick"])
    lessons["exam"][:] = 4.0
    lessons["opening"][:, 1] = 2.0
    lessons["phase"][:] = np.where(np.arange(count) % 2 == 0, 1, 2)  # auction, estimate, ...
    # The player has "changed" to the finalist at every estimate and nowhere else.
    estimates = lessons["phase"] == 2
    lessons["options"][estimates] = lessons["options"][estimates][:, [1, 0, 2]]
    player, _ = nets.load_player(start)
    scores = B.bid_scores(player, lessons, seat_game=[10.0, 20.0, 15.0])
    assert scores["phases"]["Dash Calls"] == {
        "lessons": 0.0,
        "changed": 0.0,
        "gain": 0.0,
        "gain_error": 0.0,
        "exam_gain": 0.0,
    }
    assert scores["phases"]["the auction"]["lessons"] == (~estimates).sum()
    assert scores["phases"]["the auction"]["changed"] == 0.0
    assert scores["phases"]["estimates"]["changed"] == 1.0
    assert scores["phases"]["estimates"]["gain"] == pytest.approx(2.0)
    assert scores["phases"]["estimates"]["exam_gain"] == pytest.approx(4.0)
    assert scores["gain"] == pytest.approx(2.0 * estimates.mean())
    # 15 estimates a seat a Bola, each worth 2 points by its first stage and 4 by its exam;
    # nothing from the other kinds.
    assert (scores["game"], scores["game_error"]) == (pytest.approx(30.0), 0.0)
    assert scores["exam_game"] == pytest.approx(60.0)
    assert "game" not in B.bid_scores(player, lessons)


def test_a_seats_decisions_a_bola_come_from_the_lessons_and_their_shares(tmp_path: Path):
    lessons = blank(12)
    lessons["game"][:] = [0] * 6 + [1] * 6  # two Bolas
    lessons["phase"][:] = [0, 1, 1, 2, 2, 2] * 2
    assert B.per_seat_game(lessons, tmp_path) is None  # no config.json: the shares are unknown
    (tmp_path / "config.json").write_text(json.dumps({"label_share": [0.25, 0.5, 1.0]}))
    # Per Bola 1 Dash Call at a quarter share, 2 auction lessons at a half, 3 estimates: over
    # four seats that is 1, 1 and 0.75 a seat.
    assert B.per_seat_game(lessons, tmp_path) == pytest.approx([1.0, 1.0, 0.75])
    (tmp_path / "config.json").write_text(json.dumps({"label_share": [0.0, 0.5, 1.0]}))
    assert B.per_seat_game(lessons, tmp_path) is None  # a kind not taught: no count for it


# A run.


def learn_config(lessons: Path, data: Path, start: Path, **changes: Any) -> B.LearnConfig:
    settings: dict[str, Any] = {
        "lessons": str(lessons),
        "cards": str(data),
        "start": str(start),
        "folds": 4,
        "lead_size": "32x2",
        "lead_epochs": 100,
        "lead_lr": 3e-3,
        "epochs": 60,
        "batch": 32,
        "lr": 3e-3,
        "lr_end": 3e-4,
        "clip": 50.0,
        "held_out": 0.34,
        "card_held_out": 0.34,
        "card_seed": 11,
        "seed": 1,
    }
    return B.LearnConfig(**{**settings, **changes})


@pytest.fixture(scope="module")
def learned(
    start: Path, data: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[B.LearnConfig, Path, dict[str, Any], L.Lessons]:
    """A run on lessons that say one thing a view shows: Pass, wherever it is legal and not the
    player's own move, is the finalist and led its exam by 40 points; any other finalist
    trailed by 40, and the dropped option by 30. The lessons are dealt out as twelve Bolas, so
    every fold has every kind of decision."""
    folder = tmp_path_factory.mktemp("learn")
    lessons = bidding_lessons(start, data)
    lessons["game"][:] = np.arange(len(lessons["pick"])) % 12
    options = lessons["options"]
    offered = (lessons["x"][:, -120 + PASS] > 0.5) & (options[:, 0] != PASS)
    third = offered & (options[:, 2] == PASS)
    options[third] = options[third][:, [0, 2, 1]]  # Pass was the dropped option: the finalist
    absent = offered & (options[:, 1] != PASS)
    options[absent] = np.stack(
        [options[absent, 0], np.full(absent.sum(), PASS), options[absent, 1]], 1
    )
    lessons["exam"][:] = np.where(offered, 40.0, -40.0)
    lessons["margin"][:, 2] = -30.0
    lessons["error"][:, 2] = 1.0
    (folder / "lessons" / "parts").mkdir(parents=True)
    L.save_lessons(L.lesson_path(folder / "lessons", 0), lessons)
    config = learn_config(folder / "lessons", data, start)
    out = folder / "out"
    return config, out, B.run(config, out), lessons


def test_a_run_learns_the_bids_that_led_their_exams(
    learned: tuple[B.LearnConfig, Path, dict[str, Any], L.Lessons],
):
    config, out, _, lessons = learned
    train, _ = B.split(lessons, config.held_out, config.seed)
    player = load_net(str(out / "player.npz"))
    assert player.answers  # saved with its practice answers, as the search needs
    logits = player(train["x"].astype(np.float32))
    legal = train["x"][:, -120:] > 0.5
    top = np.where(legal, logits, -np.inf).argmax(1)
    led = train["exam"] > 0
    assert led.sum() > 10
    assert (~led).sum() > 10
    assert (top[led] == train["options"][led, 1]).mean() > 0.9  # the finalist where it led
    assert (top[~led] == train["options"][~led, 0]).mean() > 0.9  # its own move where it trailed


def test_a_run_keeps_the_card_play_and_the_practice_answers(
    start: Path, data: Path, learned: tuple[B.LearnConfig, Path, dict[str, Any], L.Lessons]
):
    config, out, summary, _ = learned
    cards = B.load_cards(data, config)
    before, after = load_net(str(start)), load_net(str(out / "player.npz"))
    x = cards.held.astype(np.float32)
    legal = x[:, -120:] > 0.5
    was = np.where(legal, before(x), -np.inf).argmax(1)
    now = np.where(legal, after(x), -np.inf).argmax(1)
    assert (was == now).mean() > 0.97
    moved = np.abs(after.holder_chances(x) - before.holder_chances(x))
    assert moved.mean() < 0.02  # a toy net learning 40-point leads: close, not equal
    assert moved.max() < 0.25
    for scores in (summary["start"]["cards"], summary["learned"]["cards"]):
        assert scores["cards"] == len(cards.scored)
        assert scores["top_card"] >= scores["top_2"] >= scores["top_3"] >= 0


def test_a_run_scores_the_start_and_every_pass_and_saves_its_summary(
    learned: tuple[B.LearnConfig, Path, dict[str, Any], L.Lessons],
):
    config, out, summary, lessons = learned
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == list(range(config.epochs + 1))
    assert rows[0]["bids"]["changed"] == 0.0  # the start makes its own moves
    assert rows[-1]["train"]["bids"] < rows[1]["train"]["bids"]
    assert summary == json.loads((out / "summary.json").read_text())
    assert summary["lessons"]["train"] + summary["lessons"]["held_out"] == len(lessons["pick"])
    assert 0.1 < summary["start"]["bids"]["overruled"] < 0.9  # the auction decisions held out
    assert json.loads((out / "config.json").read_text())["temperature"] == config.temperature
    # The lead models' own bids on the held-out lessons: the finalist at the auction decisions,
    # as the teacher's careful pick is, kept as picks for the real-play check.
    assert summary["lead_models"]["agree"] > 0.9
    assert summary["lead_models"]["changed"] == pytest.approx(
        summary["start"]["bids"]["overruled"], abs=0.1
    )
    _, held = B.split(lessons, config.held_out, config.seed)
    with np.load(out / "picks.npz") as kept:
        assert kept["step"].tolist() == held["step"].tolist()
        assert (kept["action"] != held["options"][:, 0]).mean() == summary["lead_models"]["changed"]
    with np.load(out / "player.npz") as saved:
        meta = json.loads(str(saved["meta"]))
    assert meta["epoch"] == config.epochs
    assert meta["bid_lessons"]["lessons"] == config.lessons


def test_a_split_holds_out_whole_bolas(start: Path, data: Path):
    lessons = bidding_lessons(start, data)
    train, held = B.split(lessons, 0.34, 1)
    assert len(train["pick"]) + len(held["pick"]) == len(lessons["pick"])
    assert not set(train["game"].tolist()) & set(held["game"].tolist())
    assert len(np.unique(held["game"])) == 1
    one = {name: values[lessons["game"] == lessons["game"][0]] for name, values in lessons.items()}
    with pytest.raises(ValueError, match="at least 2 Bolas"):
        B.split(one, 0.34, 1)


# The config.


def test_a_learning_config_is_one_checked_section():
    section = {"lessons": "a", "cards": "b", "start": "c", "temperature": 2}
    assert B.config_of({"learn": section}).temperature == 2.0
    with pytest.raises(ValueError, match="learn needs cards, start"):
        B.config_of({"learn": {"lessons": "a"}})
    with pytest.raises(ValueError, match="only \\[learn\\]"):
        B.config_of({"learn": section, "copy": {}})
    with pytest.raises(ValueError, match="unknown key"):
        B.config_of({"learn": {**section, "tilt": True}})
    with pytest.raises(ValueError, match="temperature"):
        B.LearnConfig(temperature=0.0)
    with pytest.raises(ValueError, match="held_out"):
        B.LearnConfig(held_out=1.0)
    with pytest.raises(ValueError, match="epochs"):
        B.LearnConfig(epochs=0)
    with pytest.raises(ValueError, match="folds"):
        B.LearnConfig(folds=1)
    with pytest.raises(ValueError, match="units x layers"):
        B.LearnConfig(lead_size="wide")
    with pytest.raises(ValueError, match="margin"):
        B.LearnConfig(margin=-1.0)
    reading = B.LearnConfig(lead_size="64x2", lead_epochs=7, batch=16, lead_lr=0.01).reading
    assert (reading.size, reading.epochs, reading.batch, reading.lr) == ("64x2", 7, 16, 0.01)
