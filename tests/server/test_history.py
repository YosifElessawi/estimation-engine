"""Finished games are saved on disk with their seeds, so they can be dealt again or reviewed."""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.catalog import Catalog
from server.history import GameHistory
from tests.server.conftest import FINAL
from tests.server.helpers import BOTS, Json, frames, new_game, play_whole_game, seek, whole_game

PLAY = ["heuristic", "human", "random", FINAL]


def history(client: TestClient) -> list[Json]:
    response = client.get("/api/history")
    assert response.status_code == 200, response.text
    return response.json()


def test_a_finished_played_game_is_saved_with_its_seed_and_scores(client: TestClient):
    game, every, _ = play_whole_game(client, PLAY, None)
    (saved,) = history(client)
    assert saved["id"] == game["id"]
    assert saved["mode"] == "play"
    assert saved["seed"] == game["seed"]
    assert saved["seed_chosen"] is False
    assert saved["human_seat"] == 1
    assert [seat["label"] for seat in saved["seats"]] == [
        "HeuristicBot",
        "You",
        "RandomBot",
        "RL M4 final",
    ]
    assert [seat["spec"] for seat in saved["seats"]] == PLAY
    assert saved["scores"] == every[-1]["scores"]
    assert saved["winners"] == every[-1]["winners"]
    assert saved["rounds"] == sum(frame["kind"] == "round_end" for frame in every)
    assert "actions" not in saved  # the list stays light; reviews read the moves from disk


def test_a_game_is_saved_only_once_it_is_over(client: TestClient):
    played = new_game(client, ["random", "random", "human", "random"], seed=31)
    latest = frames(client, played["id"], 0, 500)[-1]
    body = {"action": latest["legal"][0]["id"], "frame": latest["index"]}
    assert client.post(f"/api/games/{played['id']}/moves", json=body).status_code == 200
    watched = new_game(client, BOTS, seed=32)
    frames(client, watched["id"], 0, 200)
    assert history(client) == []  # neither game is over: no seed reaches the disk yet

    seek(client, watched["id"], 0, "next", "end")
    frames(client, watched["id"], 0, 10)
    seek(client, watched["id"], 0, "next", "end")
    (saved,) = history(client)
    assert saved["id"] == watched["id"]
    assert saved["mode"] == "watch"
    assert saved["seed"] == 32
    assert saved["seed_chosen"] is True
    assert saved["human_seat"] is None


def test_saved_games_are_listed_newest_first(client: TestClient):
    first = new_game(client, BOTS, seed=33)
    seek(client, first["id"], 0, "next", "end")
    second = new_game(client, BOTS, seed=34)
    seek(client, second["id"], 0, "next", "end")
    assert [saved["id"] for saved in history(client)] == [second["id"], first["id"]]
    assert [saved["id"] for saved in client.get("/api/history?limit=1").json()] == [second["id"]]


def test_a_saved_game_is_reviewed_after_a_restart_with_every_hand(
    catalog: Catalog, history_file: Path
):
    before = TestClient(create_app(catalog, history_file=history_file))
    game, every, _ = play_whole_game(before, PLAY, None)

    after = TestClient(create_app(catalog, history_file=history_file))  # a fresh server
    (saved,) = history(after)
    response = after.post(f"/api/history/{saved['id']}/review")
    assert response.status_code == 200, response.text
    review = response.json()
    assert review["mode"] == "review"
    assert review["seed"] == game["seed"]
    replay = whole_game(after, review["id"])
    assert [frame["event"] for frame in replay] == [frame["event"] for frame in every]
    assert replay[-1]["scores"] == saved["scores"]
    assert all(None not in frame["table"]["hands"] for frame in replay)
    assert len(history(after)) == 1  # a review is not saved again


def test_dealing_a_saved_game_again_gives_the_same_cards(client: TestClient):
    original = whole_game(client, new_game(client, BOTS, seed=35)["id"])
    (saved,) = history(client)
    seats = [seat["spec"] for seat in saved["seats"]]
    again = whole_game(client, new_game(client, seats, seed=saved["seed"])["id"])
    assert again == original


def test_reviewing_an_unknown_saved_game_is_not_found(client: TestClient):
    assert client.post("/api/history/nope/review").status_code == 404


def test_unreadable_lines_are_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    path = tmp_path / "games.jsonl"
    good = {
        "id": "a",
        "finished_at": "2026-10-25T12:00:00+00:00",
        "mode": "watch",
        "seed": 1,
        "seed_chosen": True,
        "human_seat": None,
        "seats": [{"spec": "random", "label": "RandomBot", "elo": 400}] * 4,
        "scores": [1, 2, 3, 4],
        "winners": [3],
        "rounds": 18,
        "actions": [0, 1],
    }
    lines = ["not json", json.dumps({"id": "b"}), json.dumps(good), '{"id": "c", "fin']
    path.write_text("\n".join(lines))
    with caplog.at_level(logging.WARNING):
        games = GameHistory(path).games()
    assert [game["id"] for game in games] == ["a"]
    assert games[0]["actions"] == [0, 1]
    assert len([r for r in caplog.records if "unreadable line" in r.message]) == 3


def test_a_failed_save_is_logged_and_the_game_goes_on(
    catalog: Catalog, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    blocked = tmp_path / "blocked"
    blocked.mkdir()  # a folder where the file should be: appending to it fails
    client = TestClient(create_app(catalog, history_file=blocked))
    game = new_game(client, BOTS, seed=36)
    with caplog.at_level(logging.ERROR):
        end = seek(client, game["id"], 0, "next", "end")
    assert frames(client, game["id"], end, 1)[0]["kind"] == "game_end"
    assert any("could not save game" in r.message for r in caplog.records)
    assert history(client) == []


def test_old_bot_names_in_saved_games_list_by_todays_names(catalog: Catalog, history_file: Path):
    GameHistory(history_file).save(
        {
            "id": "old",
            "finished_at": "2026-10-26T23:48:49+00:00",
            "mode": "watch",
            "seed": 3,
            "seed_chosen": True,
            "human_seat": None,
            "seats": [
                {"spec": "yosif", "label": "Yosif bot", "elo": 4713.0},
                {"spec": "pimc", "label": "PIMC", "elo": 4925.0},
                {"spec": "heuristic", "label": "HeuristicBot", "elo": 1000.0},
                {"spec": "random", "label": "RandomBot", "elo": 400.0},
            ],
            "scores": [-62, 134, 100, 175],
            "winners": [3],
            "rounds": 18,
            "actions": [],
        }
    )
    (saved,) = history(TestClient(create_app(catalog, history_file=history_file)))
    assert [seat["label"] for seat in saved["seats"]] == [
        "Search bot",
        "PIMC",
        "HeuristicBot",
        "RandomBot",
    ]
    assert saved["seats"][0]["spec"] == "search"  # the bot yosif loads, so it plays as it did
    assert saved["seats"][0]["elo"] == 4713.0  # the rating it had when the game was played
