"""The public table (server/public): only its own routes, B-Tig seats, the player's own view,
no looking back past the last trick, every move saved, games resumed after a restart, limits,
and the export the evaluators read."""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from estimation.arena.games import GameRecord, read_games, replay
from estimation.engine import RulesConfig
from server.catalog import Catalog
from server.public.app import Limits, PublicTable, clean_name, create_public_app
from server.public.export import export, player_file
from server.public.records import Records

Json = dict[str, Any]
VISITOR = "visitor-0001"
FREE = 1_000_000  # requests per window: the tests' clock stands still


class Clock:
    """A clock the tests move by hand: monotonic seconds and the wall time with them."""

    def __init__(self) -> None:
        self.seconds = 1000.0
        self.start = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)

    def __call__(self) -> float:
        return self.seconds

    def now(self) -> datetime:
        return self.start + timedelta(seconds=self.seconds - 1000.0)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "public" / "games.db"


def make_client(
    catalog: Catalog, db: Path, limits: Limits | None = None, clock: Clock | None = None
) -> TestClient:
    clock = clock if clock is not None else Clock()
    # The tests' clock stands still, so the request limit is off unless a test sets it.
    limits = limits if limits is not None else Limits(requests_per_window=FREE)
    table = PublicTable(
        catalog, Records(db), bot="random", limits=limits, clock=clock, now=clock.now
    )
    return TestClient(create_public_app(table))


@pytest.fixture
def client(catalog: Catalog, db: Path) -> TestClient:
    return make_client(catalog, db)


def start(client: TestClient, name: str = "Yosif", visitor: str = VISITOR) -> tuple[Json, str]:
    response = client.post("/api/games", json={"name": name, "visitor": visitor})
    assert response.status_code == 200, response.text
    body = response.json()
    return body["game"], body["token"]


def get(client: TestClient, path: str, token: str, **params: int) -> Any:
    response = client.get(path, params=params, headers={"X-Game-Token": token})
    assert response.status_code == 200, response.text
    return response.json()


def status(client: TestClient, game_id: str, token: str) -> int:
    return client.get(f"/api/games/{game_id}", headers={"X-Game-Token": token}).status_code


def move(client: TestClient, game_id: str, token: str, frame: Json, rng: random.Random) -> Json:
    action = rng.choice(frame["legal"])["id"]
    response = client.post(
        f"/api/games/{game_id}/moves",
        json={"action": action, "frame": frame["index"]},
        headers={"X-Game-Token": token},
    )
    assert response.status_code == 200, response.text
    return response.json()


def play(client: TestClient, game: Json, token: str, moves: int | None = None) -> list[Json]:
    """Play random legal moves (all of them, or ``moves``). Returns every body received."""
    rng = random.Random(0)
    first = get(client, f"/api/games/{game['id']}/frames", token, start=game["start"], count=500)
    bodies: list[Json] = [first]
    latest = bodies[0]["frames"][-1]
    made = 0
    while latest["kind"] != "game_end" and (moves is None or made < moves):
        body = move(client, game["id"], token, latest, rng)
        bodies.append(body)
        latest = body["frames"][-1]
        made += 1
    return bodies


# Names


@pytest.mark.parametrize(
    ("raw", "saved"),
    [
        ("Yosif", "Yosif"),
        ("  Ali   B.  ", "Ali B."),
        ("يوسف", "يوسف"),
        ("O'Neil_2", "O'Neil_2"),
        ("x" * 20, "x" * 20),
        ("", None),
        ("   ", None),
        ("x" * 21, None),
        ("<script>", None),
        ("a\u0000b", None),
        ("name@mail", None),
    ],
)
def test_names_are_checked(raw: str, saved: str | None):
    assert clean_name(raw) == saved


def test_a_bad_name_or_visitor_is_refused(client: TestClient):
    response = client.post("/api/games", json={"name": "<b>", "visitor": VISITOR})
    assert response.status_code == 422
    assert "letters, numbers" in response.json()["detail"]
    response = client.post("/api/games", json={"name": "Ali", "visitor": "../x"})
    assert response.status_code == 422


# What the page sees


def test_seats_are_b_tig_with_no_spec_or_rating(client: TestClient):
    game, token = start(client, "Ali")
    assert game["mode"] == "play"
    assert game["seed"] is None
    assert game["human_seat"] == 0
    assert game["start"] == 0
    assert [(s["label"], s["spec"], s["elo"], s["explains"]) for s in game["seats"]] == [
        ("Ali", "human", None, False),
        ("B-Tig", "b-tig", None, False),
        ("B-Tig", "b-tig", None, False),
        ("B-Tig", "b-tig", None, False),
    ]
    assert len(token) >= 30


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/bots"),
        ("get", "/api/history"),
        ("get", "/docs"),
        ("get", "/redoc"),
        ("get", "/openapi.json"),
        ("get", "/api/games/{id}/seek?start=0&direction=next&target=end"),
        ("post", "/api/games/{id}/review"),
        ("post", "/api/history/{id}/review"),
    ],
)
def test_only_the_public_routes_exist(client: TestClient, method: str, path: str):
    game, token = start(client)
    response = client.request(
        method, path.replace("{id}", game["id"]), headers={"X-Game-Token": token}
    )
    assert response.status_code in (404, 405)


def test_routes_are_exactly_the_public_ones(client: TestClient):
    app: Any = client.app
    routes = sorted(
        (sorted(route.methods), route.path) for route in app.routes if hasattr(route, "methods")
    )
    assert routes == [
        (["GET"], "/api/games/{game_id}"),
        (["GET"], "/api/games/{game_id}/frames"),
        (["POST"], "/api/games"),
        (["POST"], "/api/games/{game_id}/leave"),
        (["POST"], "/api/games/{game_id}/moves"),
    ]


def test_a_game_needs_its_token(client: TestClient):
    game, _ = start(client)
    path = f"/api/games/{game['id']}/frames"
    assert client.get(path).status_code == 404
    assert client.get(path, headers={"X-Game-Token": "wrong"}).status_code == 404
    other, other_token = start(client, visitor="visitor-0002")
    assert client.get(path, headers={"X-Game-Token": other_token}).status_code == 404
    assert client.get(f"/api/games/{other['id']}", headers={"X-Game-Token": other_token}).is_success


def test_a_whole_game_shows_only_the_players_hand_and_never_the_seed(client: TestClient):
    game, token = start(client)
    bodies = play(client, game, token)
    frames = [frame for body in bodies for frame in body["frames"]]
    assert frames[-1]["kind"] == "game_end"
    for frame in frames:
        hands = frame["table"]["hands"]
        assert hands[1:] == [None, None, None]
        assert frame["why"] is None
    final = get(client, f"/api/games/{game['id']}", token)
    assert final["finished"] is True
    assert final["seed"] is None
    assert "seed" not in json.dumps(bodies)


def test_frames_before_the_last_move_are_refused(client: TestClient):
    game, token = start(client)
    bodies = play(client, game, token, moves=30)
    last_body = bodies[-1]["frames"]
    first_new = last_body[0]["index"]
    view = get(client, f"/api/games/{game['id']}", token)
    assert view["start"] == first_new - 1
    path = f"/api/games/{game['id']}/frames"
    refused = client.get(path, params={"start": 0, "count": 5}, headers={"X-Game-Token": token})
    assert refused.status_code == 403
    allowed = get(client, path, token, start=view["start"], count=500)
    assert [f["index"] for f in allowed["frames"]] == list(range(view["start"], view["known"]))
    # Every frame carries the last trick, never an older one.
    table = allowed["frames"][-1]["table"]
    assert all(event["action"]["kind"] != "play" for event in table["events"])


# Saving and resuming


def test_every_move_is_saved(client: TestClient, db: Path):
    game, token = start(client, "Ali")
    bodies = play(client, game, token, moves=12)
    row = Records(db).get(game["id"])
    assert row is not None
    saved, token_hash = row
    assert token not in token_hash
    assert saved["name"] == "Ali"
    assert saved["visitor"] == VISITOR
    assert saved["seats"] == ["human", "random", "random", "random"]
    assert saved["finished_at"] is None
    frames = [frame for body in bodies for frame in body["frames"]]
    actions = [f["event"]["action"]["id"] for f in frames if f["kind"] == "action"]
    assert saved["actions"] == actions


def test_a_game_resumes_after_a_restart(catalog: Catalog, db: Path):
    first = make_client(catalog, db)
    game, token = start(first)
    play(first, game, token, moves=40)
    before = get(first, f"/api/games/{game['id']}", token)
    latest = get(first, f"/api/games/{game['id']}/frames", token, start=before["known"] - 1)

    restarted = make_client(catalog, db)  # a new server on the same database
    after = get(restarted, f"/api/games/{game['id']}", token)
    assert after["known"] == before["known"]
    assert after["start"] == before["start"]
    again = get(restarted, f"/api/games/{game['id']}/frames", token, start=after["known"] - 1)
    assert again["frames"][-1] == latest["frames"][-1]
    play(restarted, after, token)
    row = Records(db).get(game["id"])
    assert row is not None
    saved = row[0]
    assert saved["finished_at"] is not None
    assert saved["winners"] is not None
    assert saved["rounds"] >= 18
    scores = saved["scores"]
    record = GameRecord(
        set_index=-1,
        rotation=0,
        deal_seed=saved["seed"],
        seats=("human", "random", "random", "random"),
        bot_seeds=(0, 0, 0, 0),
        scores=(scores[0], scores[1], scores[2], scores[3]),
        rounds=saved["rounds"],
        actions=tuple(saved["actions"]),
    )
    assert replay(record, RulesConfig()) is not None


def test_leaving_ends_the_game_but_keeps_it(client: TestClient, db: Path):
    game, token = start(client)
    play(client, game, token, moves=3)
    headers = {"X-Game-Token": token}
    assert client.post(f"/api/games/{game['id']}/leave", headers=headers).is_success
    assert client.get(f"/api/games/{game['id']}", headers=headers).status_code == 410
    row = Records(db).get(game["id"])
    assert row is not None
    assert row[0]["left_at"] is not None
    assert len(row[0]["actions"]) >= 3


# Limits


def test_a_third_open_game_leaves_the_oldest(client: TestClient):
    first, first_token = start(client)
    second, second_token = start(client)
    start(client)
    assert status(client, first["id"], first_token) == 410
    assert status(client, second["id"], second_token) == 200


def test_games_per_visitor_per_day(catalog: Catalog, db: Path):
    clock = Clock()
    client = make_client(
        catalog, db, Limits(games_per_visitor_per_day=3, requests_per_window=FREE), clock
    )
    for _ in range(3):
        start(client)
    refused = client.post("/api/games", json={"name": "Ali", "visitor": VISITOR})
    assert refused.status_code == 429
    assert "3 games today" in refused.json()["detail"]
    start(client, visitor="visitor-0002")  # someone else still may
    clock.seconds += 24 * 3600 + 1
    start(client)


def test_new_games_per_hour_for_everyone(catalog: Catalog, db: Path):
    client = make_client(catalog, db, Limits(games_per_hour=2, requests_per_window=FREE))
    start(client, visitor="visitor-0001")
    start(client, visitor="visitor-0002")
    refused = client.post("/api/games", json={"name": "Ali", "visitor": "visitor-0003"})
    assert refused.status_code == 429


def test_a_full_table_waits_for_an_idle_game(catalog: Catalog, db: Path):
    clock = Clock()
    client = make_client(
        catalog, db, Limits(games_in_memory=1, idle_seconds=600, requests_per_window=FREE), clock
    )
    first, token = start(client, visitor="visitor-0001")
    full = client.post("/api/games", json={"name": "Ali", "visitor": "visitor-0002"})
    assert full.status_code == 503
    assert "table is full" in full.json()["detail"]
    clock.seconds += 601
    start(client, visitor="visitor-0002")
    # The idle game left memory but not the database: it comes back once there is room.
    assert status(client, first["id"], token) == 503
    clock.seconds += 601
    assert status(client, first["id"], token) == 200


def test_requests_are_rate_limited(catalog: Catalog, db: Path):
    client = make_client(catalog, db, Limits(requests_per_window=3))
    game, token = start(client)
    path = f"/api/games/{game['id']}"
    codes = [client.get(path, headers={"X-Game-Token": token}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


# Export


def test_export_writes_what_the_evaluators_read(client: TestClient, db: Path, tmp_path: Path):
    game, token = start(client, "Ali")
    play(client, game, token)
    other, other_token = start(client, "يوسف", visitor="visitor-0002")
    play(client, other, other_token, moves=5)
    out = tmp_path / "out"
    done = export(Records(db), out)
    assert (done.finished, done.unfinished) == (1, 1)
    assert done.players == {"ali.jsonl": 1}
    header, records, bad = read_games(out / "players" / "ali.jsonl")
    assert header is None
    assert bad == 0
    assert records[0].seats == ("human", "random", "random", "random")
    assert replay(records[0], RulesConfig()) is not None
    line = json.loads((out / "games.jsonl").read_text().splitlines()[0])
    assert line["player"] == "Ali"
    assert line["seats"][1]["label"] == "B-Tig"
    unfinished = json.loads((out / "unfinished.jsonl").read_text().splitlines()[0])
    assert unfinished["player"] == "يوسف"
    assert unfinished["finished_at"] is None


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("Ali", "ali.jsonl"),
        ("Ali B.", "ali-b.jsonl"),
        ("يوسف", "يوسف.jsonl"),
        ("...", "player.jsonl"),
    ],
)
def test_player_files(name: str, file: str):
    assert player_file(name) == file


def test_the_page_is_rechecked_and_its_files_are_cached(catalog: Catalog, db: Path, tmp_path: Path):
    static = tmp_path / "out"
    (static / "_next" / "static").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>B-Tig</title>")
    (static / "_next" / "static" / "app-1a2b.js").write_text("console.log(1)")
    table = PublicTable(catalog, Records(db), bot="random")
    client = TestClient(create_public_app(table, static))
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["Cache-Control"] == "no-cache"
    assert page.headers["X-Frame-Options"] == "DENY"
    asset = client.get("/_next/static/app-1a2b.js")
    assert "immutable" in asset.headers["Cache-Control"]
    assert client.get("/api/bots").status_code == 404
