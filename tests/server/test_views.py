"""Views for the browser: action labels in context, and each seat's current call."""

import random

import pytest

from estimation.engine import (
    DASH_CALL,
    HEARTS,
    NO_DASH_CALL,
    PASS,
    SPADES,
    WITH,
    Bola,
    Observation,
    Phase,
    Round,
    bid_action,
    bid_of,
    deal,
    estimate_action,
)
from estimation.engine.actions import is_bid, is_play
from server.views import EventView, describe_action, table_view


def check_label(obs: Observation, action: int) -> str:
    """Describe ``action`` at ``obs`` and check its label against the rules. Returns its kind."""
    view = describe_action(obs, action)
    kind = view["kind"]
    assert view["id"] == action
    assert (kind == "dash") == (action == DASH_CALL)
    assert (kind == "no_dash") == (action == NO_DASH_CALL)
    # AUC-3: Pass on the top bidder's own final turn keeps the bid.
    assert (kind == "keep") == (action == PASS and obs.current_player == obs.top_bidder)
    assert (kind == "pass") == (action == PASS and obs.current_player != obs.top_bidder)
    # FIX-4: a bid made instead of an estimate in a fixed-trump round is a super call.
    assert (kind == "super_call") == (is_bid(action) and obs.phase is Phase.ESTIMATION)
    if kind == "super_call":
        assert obs.fixed_trump is not None
        assert bid_of(action)[1] != obs.fixed_trump
        assert bid_of(action)[0] >= 8
    if kind == "with":  # AUC-2: With joins the top bid's trick count and trump
        assert obs.top_bid is not None
        assert (view["tricks"], view["trump"]) == bid_of(obs.top_bid)
    if is_play(action):
        assert view["card"] == action - 68
    return kind


def test_every_legal_action_in_random_games_is_labelled_by_the_rules():
    rng = random.Random(0)
    seen: set[str] = set()
    for seed in range(6):
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            obs = bola.observation(seat)
            legal = bola.legal_actions()
            for action in legal:
                seen.add(check_label(obs, action))
            bola.step(rng.choice(legal))
    kinds = {"no_dash", "dash", "bid", "super_call", "with", "pass", "keep", "estimate", "play"}
    assert seen == kinds


def test_every_action_id_has_a_label_and_nothing_else_does():
    obs = Bola(0).observation(0)
    labels = {describe_action(obs, action)["label"] for action in range(120)}
    assert len(labels) == 120
    with pytest.raises(ValueError, match="no action with id 120"):
        describe_action(obs, 120)


def test_L7_calls_restart_after_a_super_call():
    """L7: round 16 (Hearts), F = seat 3. Seat 3 estimates 2, seat 0 super-calls 8 Spades and
    keeps it on its final turn; then estimation starts again."""
    rnd = Round.new(16, deal(0, 16), fixed_trump=HEARTS)
    events: list[EventView] = []

    def act(action: int) -> str:
        seat = rnd.current_player
        assert seat is not None
        view = describe_action(rnd.observation(seat), action)
        events.append(EventView(seat=seat, action=view, trick_winner=None))
        rnd.step(action)
        return view["label"]

    def calls() -> list[str | None]:
        found = table_view(rnd.observation(0), events)["calls"]
        return [None if call is None else call["label"] for call in found]

    assert act(estimate_action(2)) == "Estimate 2"
    assert calls() == [None, None, None, "Estimate 2"]
    assert act(bid_action(8, SPADES)) == "Super call 8 Spades"
    assert rnd.phase is Phase.AUCTION
    assert calls() == ["Super call 8 Spades", None, None, None]  # seat 3's estimate is cleared
    for _ in range(3):
        assert act(PASS) == "Pass"
    assert act(PASS) == "Keep bid"
    assert rnd.phase is Phase.ESTIMATION
    assert calls() == [None, None, None, None]
    assert act(estimate_action(2)) == "Estimate 2"  # seat 1 estimates first again
    assert calls() == [None, "Estimate 2", None, None]
    assert len(table_view(rnd.observation(0), events)["events"]) == 7  # the log keeps them all


def test_a_lost_with_is_not_shown_as_a_current_call():
    """AUC-4: seat 1 says With on 4 Hearts, seat 0 then wins at 5 Spades, and seat 1 estimates as
    a normal player; its With bubble is gone once estimation starts."""
    rnd = Round.new(1, deal(0, 1))
    events: list[EventView] = []
    script = [NO_DASH_CALL] * 4 + [
        bid_action(4, HEARTS),
        WITH,
        PASS,
        PASS,
        PASS,
        bid_action(5, SPADES),
    ]
    for action in script:
        seat = rnd.current_player
        assert seat is not None
        view = describe_action(rnd.observation(seat), action)
        events.append(EventView(seat=seat, action=view, trick_winner=None))
        rnd.step(action)
    assert rnd.phase is Phase.ESTIMATION
    assert rnd.observation(0).with_players == ()
    assert table_view(rnd.observation(0), events)["calls"] == [None, None, None, None]
