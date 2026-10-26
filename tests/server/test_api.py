"""The web table's API (SPEC §6 M6): watch a Bola step by step, play one, review it."""

import random
from typing import Any

import pytest
from fastapi.testclient import TestClient

from estimation.engine import cards_of, deal, top_seats
from server import session as session_module
from server.catalog import Catalog
from server.session import GameSession
from tests.server.conftest import FINAL, WARM
from tests.server.helpers import (
    BOTS,
    frames,
    new_game,
    play_whole_game,
    seek,
    walk_cards,
    whole_game,
)

# Bots and games


def test_bots_lists_the_catalog_with_elo(client: TestClient, catalog: Catalog):
    bots = client.get("/api/bots").json()
    assert [bot["spec"] for bot in bots] == [entry.spec for entry in catalog.entries]
    assert bots[0] == {
        "spec": "random",
        "label": "RandomBot",
        "family": "random",
        "elo": 400,
        "explains": False,
        "groups": ["Leaderboard"],
        "description": "Plays any legal move; fixed at 400 Elo",
    }
    groups = {bot["spec"]: bot["groups"] for bot in bots}
    assert groups["rl:checkpoint=runs/m4/checkpoints/final.pt"] == ["Leaderboard", "RL checkpoints"]
    assert groups["rl:checkpoint=runs/m4/checkpoints/iter-0010.pt"] == ["RL checkpoints"]


@pytest.mark.parametrize(
    ("seats", "seed", "message"),
    [
        (["human", "human", "random", "random"], 1, "at most one seat can be human"),
        (["random", "random", "random"], 1, ""),
        (["random", "random", "random", "nobody"], 1, "unknown bot"),
        (["random", "random", "random", "rl:checkpoint=outside.pt"], 1, "not a file under runs/"),
        (["random", "random", "random", "random"], -1, ""),
    ],
)
def test_bad_games_are_refused(client: TestClient, seats: list[str], seed: int, message: str):
    response = client.post("/api/games", json={"seats": seats, "seed": seed})
    assert response.status_code == 422
    assert message in response.text


def test_unknown_games_are_not_found(client: TestClient):
    assert client.get("/api/games/nope").status_code == 404
    assert client.get("/api/games/nope/frames").status_code == 404
    assert client.post("/api/games/nope/moves", json={"action": 0}).status_code == 404


# Watch mode


def test_a_watched_bola_steps_from_the_deal_to_the_winners(client: TestClient):
    game = new_game(client, BOTS, seed=3)
    assert game["mode"] == "watch"
    assert game["seed"] == 3
    assert game["known"] == 1
    every = whole_game(client, game["id"])
    assert [frame["index"] for frame in every] == list(range(len(every)))
    assert every[0]["kind"] == "deal"
    assert every[-1]["kind"] == "game_end"
    results = [frame["result"] for frame in every if frame["kind"] == "round_end"]
    assert len(results) >= 18  # 13 open rounds and 5 fixed trumps, plus replays (GAM-1)
    totals = [sum(result["scores"][seat] for result in results) for seat in range(4)]
    assert every[-1]["scores"] == totals
    assert [summary["scores"] for summary in every[-1]["history"]] == [r["scores"] for r in results]
    for frame in every:
        played = frame["history"]
        assert frame["scores"] == [sum(r["scores"][seat] for r in played) for seat in range(4)]
    assert every[-1]["winners"] == list(top_seats((totals[0], totals[1], totals[2], totals[3])))
    assert client.get(f"/api/games/{game['id']}").json()["finished"]
    kinds = {frame["event"]["action"]["kind"] for frame in every if frame["event"]}
    assert {"no_dash", "bid", "pass", "keep", "estimate", "play"} <= kinds


def test_a_watched_game_shows_every_hand(client: TestClient):
    game = new_game(client, BOTS, seed=4)
    for frame in whole_game(client, game["id"]):
        table = frame["table"]
        hands = table["hands"]
        assert [len(hand) for hand in hands] == table["hand_sizes"]
        if frame["kind"] == "deal":
            dealt = deal(4, frame["round_no"])
            assert hands == [cards_of(hand) for hand in dealt]


def missing_a_suit(seed: int, round_no: int) -> list[int]:
    dealt = deal(seed, round_no)
    return [s for s in range(4) if any(not (dealt[s] >> (13 * suit)) & 0x1FFF for suit in range(4))]


def test_both_modes_show_who_was_announced_void_at_the_deal(client: TestClient):
    """TBL-7: the announcement is public, so a player sees it too; it names seats, not suits."""
    watched = whole_game(client, new_game(client, BOTS, seed=31)["id"])
    for frame in watched:
        if frame["kind"] != "game_end":
            assert frame["table"]["void_announced"] == missing_a_suit(31, frame["round_no"])
    assert any(frame["table"]["void_announced"] for frame in watched)
    _, played, _ = play_whole_game(client, ["random", "human", "heuristic", "random"], seed=32)
    for frame in played:
        if frame["kind"] != "game_end":
            assert frame["table"]["void_announced"] == missing_a_suit(32, frame["round_no"])


def test_the_same_seats_and_seed_replay_the_same_game(client: TestClient):
    first = whole_game(client, new_game(client, BOTS, seed=5)["id"])
    again = whole_game(client, new_game(client, BOTS, seed=5)["id"])
    assert first == again
    other = whole_game(client, new_game(client, BOTS, seed=6)["id"])
    assert first != other


def test_stepping_back_shows_the_same_frames(client: TestClient):
    game_id = new_game(client, BOTS, seed=7)["id"]
    forward = frames(client, game_id, 0, 60)
    assert frames(client, game_id, 10, 1) == forward[10:11]
    assert frames(client, game_id, 0, 60) == forward


def test_seek_finds_tricks_rounds_and_deals(client: TestClient):
    game_id = new_game(client, BOTS, seed=8)["id"]
    trick = seek(client, game_id, 0, "next", "trick")
    (frame,) = frames(client, game_id, trick, 1)
    assert frame["event"]["trick_winner"] is not None
    assert frame["table"]["trick"] == []
    assert len(frame["table"]["last_trick"]["cards"]) == 4
    assert frame["table"]["last_trick"]["winner"] == frame["event"]["trick_winner"]
    round_end = seek(client, game_id, 0, "next", "round_end")
    (frame,) = frames(client, game_id, round_end, 1)
    assert frame["kind"] == "round_end"
    assert frame["result"]["round_no"] == 1
    deal_index = seek(client, game_id, round_end, "next", "deal")
    assert deal_index == round_end + 1
    assert seek(client, game_id, deal_index, "prev", "deal") == 0
    assert seek(client, game_id, 0, "prev", "trick") == 0


def test_rl_bots_explain_each_decision(client: TestClient):
    game = new_game(client, [FINAL, "heuristic", WARM, "random"], seed=9)
    assert [seat["explains"] for seat in game["seats"]] == [True, False, True, False]
    seen = set()
    for frame in frames(client, game["id"], 0, 400):
        event, why = frame["event"], frame["why"]
        if event is None:
            assert why is None
            continue
        seat = event["seat"]
        if seat in (1, 3):
            assert why is None
            continue
        card = event["action"]["kind"] == "play"
        if seat == 2 and not card:  # the warm start bids like HeuristicBot
            assert why["net"] == "HeuristicBot"
            assert why["options"] == []
            continue
        assert why["net"] == ("CardPlayNet" if card else "BiddingNet")
        options = why["options"]
        points = [option["points"] for option in options]
        assert points == sorted(points, reverse=True)
        assert options[0]["action"] == event["action"]  # the bot plays its best option
        assert [option["chosen"] for option in options].count(True) == 1
        if card:
            for option in options:
                expected = why["baseline"] + option["advantage"]
                assert option["points"] == pytest.approx(expected, abs=0.02)
        seen.add((seat, why["net"]))
    assert seen == {(0, "BiddingNet"), (0, "CardPlayNet"), (2, "CardPlayNet")}


def test_bots_get_only_their_own_observation(catalog: Catalog):
    class Recorder:
        def __init__(self, seat: int, inner: Any) -> None:
            self.seat = seat
            self.inner = inner
            self.calls = 0

        def act(self, obs: Any, legal: list[int]) -> int:
            assert obs.seat == self.seat
            assert obs.current_player == self.seat
            self.calls += 1
            return self.inner.act(obs, legal)

    game = GameSession("g", BOTS, 10, catalog, mode="watch")
    recorders = [Recorder(seat, bot) for seat, bot in game.bots.items()]
    game.bots = {recorder.seat: recorder for recorder in recorders}
    game.ensure(10_000)
    assert game.finished
    assert all(recorder.calls > 0 for recorder in recorders)


# Play mode


def test_a_human_plays_a_full_bola(client: TestClient):
    game, every, _ = play_whole_game(client, ["heuristic", "random", "human", FINAL], seed=11)
    assert game["mode"] == "play"
    assert game["human_seat"] == 2
    assert game["finished"]
    assert [frame["index"] for frame in every] == list(range(len(every)))
    results = [frame["result"] for frame in every if frame["kind"] == "round_end"]
    assert len(results) >= 18
    assert every[-1]["scores"] == [sum(r["scores"][seat] for r in results) for seat in range(4)]
    human_moves = [f for f in every if f["event"] and f["event"]["seat"] == 2]
    assert len(human_moves) >= 18 * 13  # at least one card per trick
    waiting = [frame for frame in every if frame["legal"]]
    assert all(frame["table"]["current_player"] == 2 for frame in waiting)


def test_play_mode_never_sends_another_seats_cards(client: TestClient):
    human = 1
    game, every, responses = play_whole_game(client, ["heuristic", "human", FINAL, "random"], None)
    seed = game["seed"]
    assert seed is not None  # shown once the game is over
    assert all(body["seed"] is None for body in responses[:-1] if "seed" in body)

    # Opponents' unplayed cards never appear in anything the browser received.
    played: dict[tuple[int, int], set[int]] = {}
    for frame in every:
        round_no = frame["round_no"]
        event = frame["event"]
        if event and event["action"]["kind"] == "play":
            played.setdefault((round_no, event["seat"]), set()).add(event["action"]["card"])
        dealt = deal(seed, round_no)
        hidden = {
            card
            for seat in range(4)
            if seat != human
            for card in cards_of(dealt[seat])
            if card not in played.get((round_no, seat), set())
        }
        assert not hidden & set(walk_cards(frame)), f"frame {frame['index']} leaks a card"
        assert frame["why"] is None
        hands = frame["table"]["hands"]
        assert [hand is None for hand in hands] == [seat != human for seat in range(4)]
        mine = set(cards_of(dealt[human])) - played.get((round_no, human), set())
        assert set(hands[human]) == mine
        for option in frame["legal"] or []:
            assert option["card"] is None or option["card"] in mine
    for body in responses:
        for frame in body.get("frames", []):
            assert frame in every
    assert all(not seat["explains"] for seat in responses[0]["seats"])


def test_play_views_are_built_from_the_human_observation_only(
    catalog: Catalog, monkeypatch: pytest.MonkeyPatch
):
    human = 3
    real = session_module.table_view
    calls = []

    def spy(obs: Any, events: Any, hands: Any = None) -> Any:
        assert obs.seat == human
        assert hands is None
        calls.append(obs.round_no)
        return real(obs, events, hands)

    monkeypatch.setattr(session_module, "table_view", spy)
    rng = random.Random(1)
    game = GameSession("p", ["heuristic", "random", FINAL, "human"], 12, catalog, mode="play")
    while not game.finished:
        legal = game.frames[-1]["legal"]
        assert legal
        game.play(rng.choice(legal)["id"])
    assert len(calls) == len(game.frames)


def test_moves_are_checked(client: TestClient):
    game = new_game(client, ["random", "random", "human", "random"], seed=13)
    latest = frames(client, game["id"], 0, 500)[-1]
    assert latest["index"] > 0  # seats 0 and 1 moved first
    url = f"/api/games/{game['id']}/moves"
    illegal = next(a for a in range(120) if a not in {o["id"] for o in latest["legal"]})
    assert client.post(url, json={"action": illegal}).status_code == 422
    assert client.post(url, json={"action": 120}).status_code == 422
    stale = {"action": latest["legal"][0]["id"], "frame": latest["index"] - 1}
    assert client.post(url, json=stale).status_code == 409
    fresh = {"action": latest["legal"][0]["id"], "frame": latest["index"]}
    assert client.post(url, json=fresh).status_code == 200
    watched = new_game(client, BOTS, seed=13)
    assert client.post(f"/api/games/{watched['id']}/moves", json={"action": 0}).status_code == 409


def test_play_mode_frames_stop_at_the_human_decision(client: TestClient):
    game = new_game(client, ["random", "random", "human", "random"], seed=14)
    assert game["waiting"]
    known = game["known"]
    got = client.get(f"/api/games/{game['id']}/frames", params={"start": 0, "count": 500}).json()
    assert got["known"] == known
    assert len(got["frames"]) == known
    assert got["frames"][-1]["legal"]
    assert not any(frame["legal"] for frame in got["frames"][:-1])


def test_a_finished_game_can_be_reviewed_with_every_hand(client: TestClient):
    game, every, _ = play_whole_game(client, ["random", "human", "heuristic", WARM], seed=15)
    early = new_game(client, ["random", "human", "heuristic", "random"], seed=15)
    assert client.post(f"/api/games/{early['id']}/review").status_code == 409

    response = client.post(f"/api/games/{game['id']}/review")
    assert response.status_code == 200, response.text
    review = response.json()
    assert review["mode"] == "review"
    assert review["seed"] == 15
    assert [seat["label"] for seat in review["seats"]][1] == "You"
    assert [seat["explains"] for seat in review["seats"]] == [False, False, False, True]
    replay = whole_game(client, review["id"])
    assert [f["event"] for f in replay] == [f["event"] for f in every]
    assert replay[-1]["scores"] == every[-1]["scores"]
    assert all(None not in frame["table"]["hands"] for frame in replay)


def test_a_chosen_seed_is_shown_and_a_random_one_stays_hidden_mid_game(client: TestClient):
    chosen = new_game(client, ["random", "human", "random", "random"], seed=21)
    assert chosen["seed"] == 21
    hidden = new_game(client, ["random", "human", "random", "random"])
    assert hidden["seed"] is None
    latest = frames(client, hidden["id"], 0, 500)[-1]
    url = f"/api/games/{hidden['id']}/moves"
    client.post(url, json={"action": latest["legal"][0]["id"], "frame": latest["index"]})
    assert client.get(f"/api/games/{hidden['id']}").json()["seed"] is None


def test_a_hidden_seed_is_drawn_from_the_whole_range(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """The deal is a pure function of the seed (SPEC §5.1), so a small range would let a player
    find the seed, and every hand, from their own first hand."""
    import server.app as app_module

    asked: list[int] = []

    def draw(below: int) -> int:
        asked.append(below)
        return below - 1

    monkeypatch.setattr(app_module.secrets, "randbelow", draw)
    new_game(client, ["random", "human", "random", "random"])
    assert asked == [2**53]


def test_bots_in_a_played_game_get_only_their_own_observation(catalog: Catalog):
    seats_seen: dict[int, set[int]] = {}

    class Recorder:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.key = len(seats_seen)
            seats_seen[self.key] = set()

        def act(self, obs: Any, legal: list[int]) -> int:
            assert obs.seat == obs.current_player
            seats_seen[self.key].add(obs.seat)
            return self.inner.act(obs, legal)

    class Wrapping:
        def make(self, spec: str, seed: int, rules: Any) -> Any:
            return Recorder(catalog.make(spec, seed, rules))

    rng = random.Random(2)
    game = GameSession("p", ["heuristic", "human", "random", FINAL], 22, Wrapping(), mode="play")  # type: ignore[arg-type]  # a stand-in catalog that only makes bots
    while not game.finished:
        legal = game.frames[-1]["legal"]
        assert legal
        game.play(rng.choice(legal)["id"])
    assert sorted(seats_seen.values(), key=sorted) == [{0}, {2}, {3}]


def test_games_with_search_rl_and_noisy_bots_replay_the_same(client: TestClient):
    seats = [FINAL, "pimc:samples=2", "heuristic:noise=0.3", WARM]
    first = frames(client, new_game(client, seats, seed=23)["id"], 0, 300)
    again = frames(client, new_game(client, seats, seed=23)["id"], 0, 300)
    assert len(first) == 300
    assert first == again


def test_a_changed_fixed_trump_says_whether_it_is_replayed(client: TestClient):
    """FIX-5: random play super-calls often, so this game reaches the replay cap; after it, a
    changed trump is no longer replayed."""
    every = whole_game(client, new_game(client, ["random"] * 4, seed=24)["id"])
    ends = [i for i, frame in enumerate(every) if frame["kind"] == "round_end"]
    replayed = capped = 0
    for i in ends:
        result = every[i]["result"]
        if result["fixed_trump"] is None:
            assert not result["replayed"]
            continue
        following = every[i + 1]  # the next deal, or the game's end
        again = following["kind"] == "deal"
        again = again and following["table"]["fixed_trump"] == result["fixed_trump"]
        assert result["replayed"] == again
        assert result["replayed"] <= result["trump_changed"]
        replayed += result["replayed"]
        capped += result["trump_changed"] and not result["replayed"]
    assert replayed == 20  # max_fixed_replays
    assert capped > 0
