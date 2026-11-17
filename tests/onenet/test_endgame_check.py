"""Step 7's yardstick: the rows it keeps of a recorded Bola's last cards, what they say, and the
changed cards played for real."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from estimation.engine import Bola, Observation, Phase
from estimation.engine.config import DEFAULT_CONFIG
from estimation.jaxenv.encode_np import PLAY_BASE
from estimation.onenet import endgame as E
from estimation.onenet import endgame_check as C
from estimation.rl.config import ENDGAME_STAGE, derive_seed
from estimation.rl.onenet_record import PartTask, RecordConfig, part_path, record_part, save_part
from tests.onenet.test_search import random_net

NAN = float("nan")


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


def config_of(games: Path, deals: int = 4, cards: int = 3) -> C.CheckConfig:
    recording = RecordConfig(**json.loads((games / "config.json").read_text()))
    return C.CheckConfig(str(games), recording.player, 1, 1, deals, cards, seed=9)


@pytest.fixture(scope="module")
def rows(games: Path) -> C.Arrays:
    return C.check_bola(C.BolaTask(0, 0, config_of(games)))


def decisions(games: Path, cards: int) -> dict[int, tuple[Observation, int, list[int]]]:
    """The Bola's card decisions with a choice and at most ``cards`` cards in the hand, by step:
    the deciding seat's view, the card played, the legal moves."""
    with np.load(part_path(games, 0)) as part:
        seed, actions = int(part["seed"][0]), part["actions"][: int(part["log_start"][1])].tolist()
    bola = Bola(seed, DEFAULT_CONFIG)
    found: dict[int, tuple[Observation, int, list[int]]] = {}
    for step, action in enumerate(actions):
        seat = bola.current_player
        if seat is not None and bola.phase is Phase.PLAY:
            legal = bola.legal_actions()
            obs = bola.observation(seat)
            if len(legal) > 1 and obs.hand.bit_count() <= cards:
                found[step] = (obs, action - PLAY_BASE, legal)
        bola.step(action)
    return found


def test_check_bola_keeps_every_late_card_decision_with_a_choice(games: Path, rows: C.Arrays):
    wanted = decisions(games, 3)
    assert rows["step"].tolist() == sorted(wanted)
    assert len(wanted) > 20
    for k, step in enumerate(rows["step"].tolist()):
        obs, played, legal = wanted[step]
        cards = [move - PLAY_BASE for move in legal]
        assert rows["game"][k] == 0
        assert rows["seat"][k] == obs.seat
        assert rows["round"][k] == obs.round_no
        assert rows["left"][k] == obs.hand.bit_count() <= 3
        assert rows["place"][k] == len(obs.trick)
        assert rows["legal"][k].tolist() == cards + [-1] * (E.MAX_CARDS - len(cards))
        assert rows["played"][k] == played
        assert rows["net_top"][k] in cards
        for name in ("exact_a", "exact_b", "rough_a", "rough_b"):
            assert np.isfinite(rows[name][k, : len(cards)]).all()
            assert np.isnan(rows[name][k, len(cards) :]).all()


def test_check_bola_is_the_same_twice(games: Path, rows: C.Arrays):
    again = C.check_bola(C.BolaTask(0, 0, config_of(games)))
    for name in C.DTYPES:
        np.testing.assert_array_equal(rows[name], again[name])


def test_a_row_holds_the_exact_and_the_rough_worth_of_each_card_on_its_deals(
    games: Path, rows: C.Arrays
):
    config = config_of(games)
    wanted = decisions(games, 3)
    for k in (0, len(rows["step"]) // 2, len(rows["step"]) - 1):
        step = int(rows["step"][k])
        obs, _, legal = wanted[step]
        seat = obs.seat
        assert obs.leader is not None
        assert obs.trump is not None
        bot = C.table_player(config.player, derive_seed(9, ENDGAME_STAGE, 0, step, seat))
        deals = bot.sample_deals(obs, 4)
        worths = E.worths_of(obs, bot.config.rules, bot.config.leader_weight)
        solved = E.solve(deals, obs.trick, obs.leader, obs.tricks_won, obs.trump, worths)
        cards = [move - PLAY_BASE for move in legal]
        mine = solved.worths[:, :, seat]
        np.testing.assert_allclose(rows["exact_a"][k, : len(cards)], mine[:2].mean(0), rtol=1e-5)
        np.testing.assert_allclose(rows["exact_b"][k, : len(cards)], mine[2:].mean(0), rtol=1e-5)
        rough = np.array(bot.evaluate(obs, cards, deals[:2])) / 2
        np.testing.assert_allclose(rows["rough_a"][k, : len(cards)], rough, rtol=1e-5)


def test_judge_values_a_hand_of_four_cards(games: Path):
    config = config_of(games)
    step, (obs, played, legal) = next(
        (step, spot)
        for step, spot in decisions(games, 4).items()
        if len(spot[2]) == 4  # four cards to choose from: the lead of the tenth trick
    )
    bot = C.table_player(config.player, derive_seed(9, ENDGAME_STAGE, 0, step, obs.seat))
    row = C.judge(obs, legal, played + PLAY_BASE, bot, 2)
    assert row["left"] == 4
    assert row["place"] == 0
    assert row["played"] == played
    assert np.isfinite(row["exact_a"]).all()
    assert np.isfinite(row["rough_b"]).all()


def made_rows(
    legal: list[list[int]],
    played: list[int],
    net_top: list[int],
    exact_a: list[list[float]],
    exact_b: list[list[float]],
    rough_a: list[list[float]],
    rough_b: list[list[float]],
) -> C.Arrays:
    count = len(played)
    return C.stack(
        [
            {
                "game": k // 2,
                "step": 10 * k,
                "seat": k % 4,
                "round": 0,
                "left": sum(card >= 0 for card in legal[k]),
                "place": 0,
                "legal": legal[k],
                "played": played[k],
                "net_top": net_top[k],
                "exact_a": exact_a[k],
                "exact_b": exact_b[k],
                "rough_a": rough_a[k],
                "rough_b": rough_b[k],
            }
            for k in range(count)
        ]
    )


def test_losses_pick_on_one_half_of_the_deals_and_judge_on_the_other():
    rows = made_rows(
        legal=[[5, 9, 20, -1], [7, 30, -1, -1], [1, 2, -1, -1]],
        played=[9, 7, 2],
        net_top=[20, 7, 2],
        # First decision: half a likes card 5, half b likes card 20; card 9 was played.
        # Third decision: the two cards are level, and the higher one was played.
        exact_a=[[4.0, 1.0, 2.0, NAN], [3.0, 0.0, NAN, NAN], [1.0, 1.0, NAN, NAN]],
        exact_b=[[0.0, 2.0, 6.0, NAN], [5.0, 1.0, NAN, NAN], [2.0, 2.0, NAN, NAN]],
        rough_a=[[0.0, 9.0, 0.0, NAN], [0.0, 2.0, NAN, NAN], [1.0, 1.0, NAN, NAN]],
        rough_b=[[0.0, 8.0, 1.0, NAN], [4.0, 0.0, NAN, NAN], [1.0, 1.0, NAN, NAN]],
    )
    loss = C.losses(rows)
    # Picked on a (card 5) and judged on b: 0 - 2; picked on b (card 20), judged on a: 2 - 1.
    assert loss["played"].tolist() == pytest.approx([(-2.0 + 1.0) / 2, 0.0, 0.0])
    # The player's top card, 20: 0 - 6 on b and 2 - 2 on a.
    assert loss["net_top"].tolist() == pytest.approx([-3.0, 0.0, 0.0])
    # The rough finish picks card 9 on both halves of the first decision, as was played. At the
    # second it picks card 30 on a (judged on b: 5 - 1) and card 7 on b (the exact pick).
    assert loss["rough"].tolist() == pytest.approx([(-2.0 + 1.0) / 2, 2.0, 0.0])
    # By the rough finish's own measure: nothing at the first; at the second card 30, picked on
    # a, is 4 behind the card played on b.
    assert loss["rough_played"].tolist() == pytest.approx([0.0, -2.0, 0.0])
    # Over all the deals: card 20 leads at the first decision (4 against 1.5 for card 9).
    # Level cards are no other answer: the card played stays the exact card.
    assert loss["differs"].tolist() == [1.0, 0.0, 0.0]
    assert loss["lead"].tolist() == pytest.approx([2.5, 0.0, 0.0])
    assert C.exact_card(rows).tolist() == [20, 7, 2]


def test_summarize_counts_by_cards_in_the_hand():
    rows = made_rows(
        legal=[[5, 9, 20, -1], [7, 30, -1, -1], [1, 2, -1, -1], [3, 4, -1, -1]],
        played=[9, 7, 1, 4],
        net_top=[20, 7, 1, 3],
        exact_a=[
            [4.0, 1.0, 2.0, NAN],
            [3.0, 0.0, NAN, NAN],
            [1.0, 0.0, NAN, NAN],
            [2, 0, NAN, NAN],
        ],
        exact_b=[
            [0.0, 2.0, 6.0, NAN],
            [5.0, 1.0, NAN, NAN],
            [1.0, 0.0, NAN, NAN],
            [4, 0, NAN, NAN],
        ],
        rough_a=[
            [0.0, 9.0, 0.0, NAN],
            [0.0, 2.0, NAN, NAN],
            [1.0, 0.0, NAN, NAN],
            [1, 0, NAN, NAN],
        ],
        rough_b=[
            [0.0, 8.0, 1.0, NAN],
            [4.0, 0.0, NAN, NAN],
            [1.0, 0.0, NAN, NAN],
            [1, 0, NAN, NAN],
        ],
    )
    real = np.array([6.0, NAN, NAN, -1.0])
    summary = C.summarize(rows, real)
    assert summary["bolas"] == 2
    assert summary["decisions"] == 4
    assert set(summary["groups"]) == {"all", "2", "3"}
    twos = summary["groups"]["2"]
    assert twos["decisions"] == 3
    assert twos["per_seat_game"] == pytest.approx(3 / 8)
    assert twos["differs"] == pytest.approx(1 / 3)  # the last decision: card 3, not card 4
    assert twos["played"] == pytest.approx((0.0 + 0.0 + 3.0) / 3)
    assert twos["real_played"] == 1
    assert twos["real"] == pytest.approx(-1.0)
    assert twos["lead_where_played"] == pytest.approx(3.0)
    everything = summary["groups"]["all"]
    assert everything["differs"] == pytest.approx(0.5)
    assert everything["real_played"] == 2
    assert everything["real"] == pytest.approx(2.5)
    assert any("played for real, 2 changed cards" in line for line in C.lines(summary))


def test_run_saves_each_bola_once_and_play_plays_its_changed_cards(games: Path, tmp_path: Path):
    out = tmp_path / "check"
    config = C.start(out, games, deals=4, cards=3, seed=9)
    assert config == config_of(games)
    summary = C.run(out, bolas=1, workers=1)
    assert summary["bolas"] == 1
    saved = C.part_file(out, 0)
    stamp = saved.stat().st_mtime_ns
    assert C.run(out, bolas=1, workers=1) == summary  # nothing is checked twice
    assert saved.stat().st_mtime_ns == stamp
    rows = C.load_rows(out)
    assert summary["decisions"] == len(rows["game"]) > 20
    assert summary["groups"]["all"]["real_played"] == 0

    # A random player's rounds leave its cards level nearly everywhere, so three decisions are
    # given another card that leads the one played by 5 points and every other card is level.
    changed = 3
    for name in ("exact_a", "exact_b"):
        rows[name][:] = np.where(rows["legal"] >= 0, 0.0, np.nan)
    for k in (0, 5, 11):
        other = next(c for c in rows["legal"][k].tolist() if c >= 0 and c != rows["played"][k])
        for name in ("exact_a", "exact_b"):
            rows[name][k, rows["legal"][k] == other] = 5.0
    C.bid_check.save_check(saved, rows)
    assert int(C.losses(rows)["differs"].sum()) == changed
    tasks = C.play_tasks(rows, config, margin=0.0)
    assert sum(len(task.steps) for task in tasks) == changed
    assert C.play_tasks(rows, config, margin=1e9) == []
    after = C.play(out, workers=1)
    assert after["groups"]["all"]["real_played"] == changed
    assert np.isfinite(after["groups"]["all"]["real"])
    assert C.play(out, workers=1) == after  # and no round is played twice
    assert json.loads((out / C.SUMMARY).read_text()) == after

    with pytest.raises(ValueError, match="another run"):
        C.start(out, games, deals=8, cards=3, seed=9)


def test_the_check_refuses_what_it_cannot_judge(games: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="even number"):
        C.CheckConfig("x", "y", 1, 1, deals=5)
    with pytest.raises(ValueError, match="cards must be"):
        C.CheckConfig("x", "y", 1, 1, cards=5)
    with pytest.raises(ValueError, match="judge=round"):
        C.table_player("random", 0)
    with pytest.raises(ValueError, match="not in the part"):
        C.check_bola(C.BolaTask(7, 0, config_of(games)))
