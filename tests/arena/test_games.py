"""Saved games: written, read back and replayed to the same scores (design 5.1)."""

import dataclasses
import json
from pathlib import Path

from estimation.arena.duplicate import run_arena
from estimation.arena.games import (
    GameRecord,
    GamesHeader,
    ReplayStats,
    code_version,
    read_games,
    replay,
    replay_all,
    write_games,
)
from estimation.arena.match import play_bola
from estimation.bots import make_bot
from estimation.engine import RulesConfig

RULES = RulesConfig()
HEADER = GamesHeader(
    code_version="test",
    rules={},
    lineup=("heuristic",) * 4,
    sets=1,
    seed=0,
    created="2026-10-26T00:00:00+00:00",
)


def one_game() -> GameRecord:
    bots = [make_bot("heuristic", seat) for seat in range(4)]
    played = play_bola(7, bots, keep_actions=True)
    return GameRecord(
        set_index=0,
        rotation=0,
        deal_seed=7,
        seats=("heuristic", "heuristic", "heuristic", "heuristic"),
        bot_seeds=(0, 1, 2, 3),
        scores=played.scores,
        rounds=played.rounds,
        actions=played.actions,
    )


def test_a_saved_game_replays_to_its_scores(tmp_path: Path):
    record = one_game()
    path = tmp_path / "games.jsonl.gz"
    write_games(path, HEADER, [record])
    header, records, _ = read_games(path)
    assert header == HEADER
    assert records == [record]
    bola = replay(records[0], RULES)
    assert bola is not None
    assert bola.scores == record.scores


def test_saving_does_not_change_results():
    plain = run_arena(("heuristic",) * 4, 2, 0)
    saved = run_arena(("heuristic",) * 4, 2, 0, keep_games=True)
    assert [r.scores for r in plain.results] == [r.scores for r in saved.results]
    assert all(len(r.games) == 4 for r in saved.results)
    assert all(r.games == () for r in plain.results)
    game = saved.results[1].games[2]
    assert game.set_index == 1
    assert game.rotation == 2
    assert replay(game, RULES) is not None


def test_an_unreplayable_game_is_skipped():
    broken = dataclasses.replace(one_game(), actions=(119, *one_game().actions[1:]))
    kept, stats = replay_all([broken], RULES)
    assert kept == []
    assert stats == ReplayStats(kept=0, skipped=1)


def test_web_games_are_read(tmp_path: Path):
    record = one_game()
    seat = {"spec": "heuristic", "label": "HeuristicBot", "elo": None}
    line = {
        "id": "abc",
        "finished_at": "2026-10-26T00:00:00+00:00",
        "mode": "watch",
        "seed": 7,
        "seed_chosen": False,
        "human_seat": None,
        "seats": [seat] * 4,
        "scores": list(record.scores),
        "winners": [0],
        "rounds": record.rounds,
        "actions": list(record.actions),
    }
    path = tmp_path / "games.jsonl"
    path.write_text(json.dumps(line) + "\n")
    header, records, _ = read_games(path)
    assert header is None
    assert len(records) == 1
    assert records[0].seats == ("heuristic",) * 4
    assert records[0].set_index == -1
    assert replay(records[0], RULES) is not None


def test_code_version_names_a_commit():
    assert code_version()


def test_the_arena_cli_saves_its_games(tmp_path: Path):
    from estimation.arena.__main__ import main

    path = tmp_path / "run.games.jsonl.gz"
    assert (
        main(
            [
                "--lineup",
                "heuristic,random,heuristic,random",
                "--sets",
                "1",
                "--save-games",
                str(path),
            ]
        )
        == 0
    )
    header, records, _ = read_games(path)
    assert header is not None
    assert header.lineup == ("heuristic", "random", "heuristic", "random")
    assert len(records) == 4
    assert all(replay(record, RULES) is not None for record in records)


def test_dirty_trees_with_different_changes_get_different_versions():
    from estimation.arena.games import version_of

    assert version_of("abc1234", "") == "abc1234"
    one, two = version_of("abc1234", "diff one"), version_of("abc1234", "diff two")
    assert one.startswith("abc1234+dirty")
    assert one != two


def test_damaged_files_and_lines_are_skipped_and_counted(tmp_path: Path):
    record = one_game()
    path = tmp_path / "plain-name.jsonl"  # gzip data under a plain name
    write_games(path, HEADER, [record, record])
    header, records, bad = read_games(path)
    assert (header, len(records), bad) == (HEADER, 2, 0)
    truncated = tmp_path / "cut.jsonl.gz"
    truncated.write_bytes(path.read_bytes()[:-40])
    _, records, bad = read_games(truncated)
    assert bad >= 1
    web = tmp_path / "games.jsonl"
    web.write_text('{"not": "a game"}\n{broken json\n')
    assert read_games(web) == (None, [], 2)
