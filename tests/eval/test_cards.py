"""Judging card plays by playing every card out on re-dealt hands, and the report."""

import dataclasses
import json
from pathlib import Path

import pytest

from estimation.arena.games import GameRecord, make_header, write_games
from estimation.arena.match import play_bola
from estimation.bots import make_bot
from estimation.bots.knowledge import read_knowledge
from estimation.engine import Round, RulesConfig, TrickRecord
from estimation.engine.cards import FULL_DECK, SUIT_MASKS
from estimation.eval.cards import (
    CELLS,
    Judged,
    Thresholds,
    brilliant,
    choices,
    evaluate,
    game_leader,
    judge,
    label_of,
    main,
    need_of,
    plays_of,
    redeals,
    sample_plays,
    stage_of,
    weighted,
)
from tests.eval.test_decisions import recorded_game

RULES = RulesConfig()
NOISY = "heuristic:play_noise=1"  # bids as HeuristicBot, plays every card at random


def mixed_game(seed: int = 5) -> GameRecord:
    seats = ("heuristic", NOISY, "heuristic", NOISY)
    bots = [make_bot(spec, seed * 4 + seat) for seat, spec in enumerate(seats)]
    played = play_bola(seed, bots, keep_actions=True)
    return GameRecord(0, 0, seed, seats, (0, 0, 0, 0), played.scores, played.rounds, played.actions)


def test_plays_are_the_targets_card_plays_with_a_real_choice():
    record = mixed_game()
    plays = plays_of(record, RULES, NOISY)
    assert plays
    assert {p.seat for p in plays} == {1, 3}
    for p in plays:
        assert p.chosen in p.legal
        assert p.obs.current_player == p.seat
        assert len(choices(p.obs, p.legal, p.chosen)) > 1


def test_choices_put_the_chosen_card_first():
    play = plays_of(recorded_game(), RULES, "heuristic")[0]
    tried = choices(play.obs, play.legal, play.chosen)
    assert tried[0] == play.chosen
    assert set(tried) <= set(play.legal)


def test_stage_follows_the_trick_number():
    obs = plays_of(recorded_game(), RULES, "heuristic")[0].obs
    trick = TrickRecord(0, (0, 1, 2, 3), 0)
    stages = [stage_of(dataclasses.replace(obs, tricks=(trick,) * n)) for n in (0, 3, 4, 8, 9, 12)]
    assert stages == ["tricks 1-4"] * 2 + ["tricks 5-9"] * 2 + ["tricks 10-13"] * 2


def test_need_compares_tricks_won_with_the_estimate():
    obs = plays_of(recorded_game(), RULES, "heuristic")[0].obs
    me = obs.seat
    estimates = tuple(2 if s == me else 3 for s in range(4))

    def need(won: int) -> str:
        tricks_won = tuple(won if s == me else 0 for s in range(4))
        return need_of(dataclasses.replace(obs, estimates=estimates, tricks_won=tricks_won))

    assert [need(1), need(2), need(3)] == ["needs tricks", "has enough", "over"]


def test_the_game_leader_is_another_player_alone_at_the_top():
    assert game_leader((10, 30, 20, 0), 0) == 1
    assert game_leader((10, 30, 20, 0), 1) is None
    assert game_leader((30, 30, 0, 0), 2) is None


def test_sampling_spreads_over_the_cells_and_weights_add_up():
    plays = plays_of(recorded_game(), RULES, "heuristic")
    cells = {(p.stage, p.need) for p in plays}
    sample = sample_plays(plays, len(cells), 0)
    assert {(p.stage, p.need) for p, _ in sample} == cells
    assert sum(w for _, w in sample) == pytest.approx(len(plays))
    assert sample == sample_plays(plays, len(cells), 0)
    everything = sample_plays(plays, len(plays) + 5, 0)
    assert len(everything) == len(plays)
    assert all(w == 1.0 for _, w in everything)
    assert set(CELLS) >= cells


def test_redeals_fit_what_the_player_has_seen():
    plays = plays_of(recorded_game(), RULES, "heuristic")
    play = next(p for p in plays if p.stage == "tricks 10-13")
    obs = play.obs
    know = read_knowledge(obs)
    log = [action for _, action in obs.log]
    for hands in redeals(obs, 10, 0, RULES):
        assert hands[play.seat] == obs.hand | know.played_by[play.seat]
        assert hands[0] | hands[1] | hands[2] | hands[3] == FULL_DECK
        assert all(hand.bit_count() == 13 for hand in hands)
        for seat in range(4):
            left = hands[seat] & ~know.played_by[seat]
            assert know.played_by[seat] & ~hands[seat] == 0
            assert left.bit_count() == know.sizes[seat]
            for suit in range(4):
                if know.voids[seat] >> suit & 1:
                    assert not left & SUIT_MASKS[suit]
        rnd = Round.replay(obs.round_no, hands, obs.fixed_trump, obs.carry, obs.scores, RULES, log)
        assert rnd.observation(play.seat).hand == obs.hand


def test_the_judge_reads_only_the_players_observation():
    play = plays_of(recorded_game(), RULES, "heuristic")[5]
    built_by_hand = dataclasses.replace(play, game=None, step=-1)
    first = judge(play, 3, "heuristic", 7, RULES)
    second = judge(built_by_hand, 3, "heuristic", 7, RULES)
    assert first.values == second.values
    assert (first.best, first.lost, first.lost_se) == (second.best, second.lost, second.lost_se)


def test_judging_is_deterministic_for_a_seed():
    plays = plays_of(recorded_game(), RULES, "heuristic")[:3]
    sample = [(p, 1.0) for p in plays]
    first = evaluate(sample, 2, "heuristic", 4, RULES)
    assert [(j.values, j.lost) for j in first] == [
        (j.values, j.lost) for j in evaluate(sample, 2, "heuristic", 4, RULES)
    ]
    for j in first:
        assert j.lost >= 0
        assert j.values[j.best][0] == max(v[0] for v in j.values.values())


def test_random_card_play_loses_more_points_than_heuristic_card_play():
    record = mixed_game()
    lost = {}
    for target in ("heuristic", NOISY):
        sample = sample_plays(plays_of(record, RULES, target), 18, 1)
        judged = evaluate(sample, 6, "heuristic", 2, RULES)
        lost[target] = weighted(judged, lambda j: j.lost)[0]
    assert lost[NOISY] > lost["heuristic"] + 1


def judged_by_hand(
    chosen: int, values: dict[int, float], leader_scores: dict[int, float]
) -> Judged:
    play = plays_of(recorded_game(), RULES, "heuristic")[0]
    play = dataclasses.replace(play, chosen=chosen, legal=tuple(values))
    best = max(values, key=lambda card: (values[card], card == chosen))
    return Judged(
        play=play,
        weight=1.0,
        values={card: (v, 0.5) for card, v in values.items()},
        best=best,
        lost=values[best] - values[chosen],
        lost_se=0.5,
        leader=1 if leader_scores else None,
        leader_scores=leader_scores,
    )


def test_labels_follow_the_review_thresholds():
    t = Thresholds()
    assert label_of(judged_by_hand(1, {1: 4.0, 2: 3.0}, {}), t) == "best"
    assert label_of(judged_by_hand(2, {1: 4.0, 2: 3.0}, {}), t) == "close"
    assert label_of(judged_by_hand(2, {1: 4.0, 2: 2.0}, {}), t) == "inaccuracy"
    assert label_of(judged_by_hand(2, {1: 4.0, 2: -1.0}, {}), t) == "blunder"
    assert label_of(judged_by_hand(2, {1: 4.0, 2: -1.0}, {}), Thresholds(blunder=6)) == (
        "inaccuracy"
    )


def test_brilliant_sabotage_is_the_best_card_cutting_the_leader_against_the_next_best():
    t = Thresholds()
    cards = {1: 4.0, 2: 3.0, 3: -2.0}
    assert brilliant(judged_by_hand(1, cards, {1: 5.0, 2: 10.0, 3: 30.0}), t)
    assert not brilliant(judged_by_hand(1, cards, {1: 5.0, 2: 9.0, 3: 30.0}), t)
    assert not brilliant(judged_by_hand(2, cards, {1: 30.0, 2: 5.0, 3: 30.0}), t)
    assert not brilliant(judged_by_hand(1, cards, {}), t)


def test_cli_runs_on_one_new_game(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    lineup = "heuristic,heuristic,heuristic,heuristic"
    args = ["--bot", "heuristic", "--lineup", lineup, "--games", "1", "--cards", "3"]
    args += ["--deals", "2", "--card-play", "heuristic", "--progress", "--out", str(tmp_path)]
    assert main(args) == 0
    report = (tmp_path / "report.txt").read_text()
    assert "Points lost per card" in report
    assert "Blunder ??" in report
    data = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(data["cards"]) == 3
    assert all(c["lost"] >= 0 and c["label"] for c in data["cards"])
    assert "cards 3/3" in capsys.readouterr().err


def test_cli_reads_saved_games(tmp_path: Path):
    record = recorded_game()
    games = tmp_path / "run" / "games" / "heuristic.jsonl.gz"
    write_games(games, make_header(record.seats, 1, 0, RULES), [record])
    args = ["--bot", "heuristic", "--from", str(tmp_path / "run"), "--cards", "2"]
    args += ["--deals", "2", "--card-play", "heuristic", "--out", str(tmp_path / "out")]
    assert main(args) == 0
    data = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert {c["game"]["deal_seed"] for c in data["cards"]} == {record.deal_seed}


def test_web_mode_judges_the_human_seat(tmp_path: Path):
    record = recorded_game()
    seats = [{"spec": spec, "label": spec, "elo": None} for spec in record.seats]
    seats[2] = {"spec": "human", "label": "You", "elo": None}
    line = {
        "id": "g1",
        "finished_at": "2026-10-27T00:00:00+00:00",
        "mode": "play",
        "seed": record.deal_seed,
        "seed_chosen": False,
        "human_seat": 2,
        "seats": seats,
        "scores": list(record.scores),
        "winners": [0],
        "rounds": record.rounds,
        "actions": list(record.actions),
    }
    web = tmp_path / "games.jsonl"
    web.write_text(json.dumps(line) + "\n")
    args = ["--bot", "human", "--web", str(web), "--cards", "2", "--deals", "2"]
    args += ["--card-play", "heuristic", "--out", str(tmp_path / "out")]
    assert main(args) == 0
    data = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert {c["seat"] for c in data["cards"]} == {2}
