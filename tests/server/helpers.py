"""Calls to the web table's API shared by the server tests."""

import random
from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

Json = dict[str, Any]
BOTS = ["random", "heuristic", "random", "heuristic"]


def new_game(client: TestClient, seats: list[str], seed: int | None = None) -> Json:
    response = client.post("/api/games", json={"seats": seats, "seed": seed})
    assert response.status_code == 200, response.text
    return response.json()


def frames(client: TestClient, game_id: str, start: int, count: int) -> list[Json]:
    response = client.get(f"/api/games/{game_id}/frames", params={"start": start, "count": count})
    assert response.status_code == 200, response.text
    return response.json()["frames"]


def seek(client: TestClient, game_id: str, start: int, direction: str, target: str) -> int:
    params = {"start": start, "direction": direction, "target": target}
    response = client.get(f"/api/games/{game_id}/seek", params=params)
    assert response.status_code == 200, response.text
    return response.json()["index"]


def whole_game(client: TestClient, game_id: str) -> list[Json]:
    end = seek(client, game_id, 0, "next", "end")
    found: list[Json] = []
    while len(found) <= end:
        found += frames(client, game_id, len(found), 500)
    return found


def walk_cards(value: object, key: str = "") -> Iterator[int]:
    """Every card id in a JSON value: under ``card`` keys and in ``hands`` lists."""
    if isinstance(value, dict):
        for name, inner in value.items():
            yield from walk_cards(inner, name)
    elif isinstance(value, list):
        for inner in value:
            yield from walk_cards(inner, key)
    elif key in ("card", "hands") and isinstance(value, int):
        yield value


def play_whole_game(
    client: TestClient, seats: list[str], seed: int | None
) -> tuple[Json, list[Json], list[Json]]:
    """Play a Bola as the human seat with random legal moves. Returns the game, every frame,
    and every response body the browser got."""
    rng = random.Random(0)
    game = new_game(client, seats, seed)
    responses: list[Json] = [game]
    received = frames(client, game["id"], 0, 500)
    responses.append({"frames": received})
    while received[-1]["kind"] != "game_end":
        latest = received[-1]
        assert latest["legal"], "the human must be offered moves"
        action = rng.choice(latest["legal"])["id"]
        body = {"action": action, "frame": latest["index"]}
        response = client.post(f"/api/games/{game['id']}/moves", json=body)
        assert response.status_code == 200, response.text
        responses.append(response.json())
        received += response.json()["frames"]
    final = client.get(f"/api/games/{game['id']}").json()
    responses.append(final)
    return final, received, responses
