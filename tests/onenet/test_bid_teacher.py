"""Step 6's bid teacher: its options, its decisions against the engine, its play-outs against
real rounds, and its narrowing."""

import random
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.bots.scoring import standings_weights
from estimation.engine import Observation, Phase, RulesConfig
from estimation.engine.scoring import RoundResult
from estimation.jaxenv import rules as R
from estimation.jaxenv.check import bits, random_round
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.onenet import bid_teacher as T
from estimation.onenet.bid_lessons import Dealt, Spot, batch_arrays
from estimation.onenet.nets import SLICES
from estimation.onenet.search import bidding_views, load_net
from tests.onenet.test_search import random_net

SIZE = 24  # decisions in these tests' one batch


class Seen(NamedTuple):
    """A bidding decision as the engine showed it: the view, the legal options, the move made."""

    obs: Observation
    legal: list[int]
    action: int


class Played(NamedTuple):
    """Rounds played by a net's top move at every decision: the rounds, their bidding decisions
    with a choice (``seen`` holds the engine's side of each), and each round's result."""

    rounds: list[Dealt]
    spots: list[Spot]
    seen: list[Seen]
    results: list[RoundResult]


def play(net_path: str, seed: int, want: int) -> Played:
    """Open and fixed-trump rounds with random carries and scores, every seat making the net's
    top legal move, until ``want`` bidding decisions with a choice are found."""
    net = load_net(net_path)
    rng = random.Random(seed)
    played = Played([], [], [], [])
    while len(played.spots) < want:
        rnd = random_round(rng)
        fixed = -1 if rnd.fixed_trump is None else rnd.fixed_trump
        dealt = Dealt(rnd.round_no, rnd.hands, fixed, rnd.carry, rnd.scores)
        choices: list[tuple[int, int]] = []
        while rnd.phase is not Phase.DONE:
            seat = rnd.current_player
            assert seat is not None
            legal = rnd.legal_actions()
            obs = rnd.observation(seat)
            logits = net(raw_view(obs, legal))
            action = max(legal, key=lambda a: logits[a])
            if rnd.phase is not Phase.PLAY:
                at = len(dealt.moves)
                if len(legal) > 1:
                    history = tuple(place for place, who in choices if who != seat)
                    index = len(played.rounds)
                    played.spots.append(
                        Spot(index, at, seat, int(rnd.phase), index, at, history, tuple(legal))
                    )
                    played.seen.append(Seen(obs, legal, action))
                    choices.append((at, seat))
                dealt.moves.append(action)
            rnd.step(action)
        played.rounds.append(dealt)
        played.results.append(rnd.result())
    return played


@pytest.fixture(scope="module")
def net_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return random_net(tmp_path_factory.mktemp("net"))


@pytest.fixture(scope="module")
def played(net_path: str) -> Played:
    found = play(net_path, 3, SIZE)
    assert any(spot.phase == int(Phase.DASH) for spot in found.spots[:SIZE])
    assert any(dealt.fixed_trump >= 0 for dealt in found.rounds)  # fixed-trump rounds too
    return found


@pytest.fixture(scope="module")
def batch(played: Played) -> T.Batch:
    arrays, where, at, history = batch_arrays(played.rounds, played.spots[:SIZE], SIZE)
    return T.positions(arrays, jnp.asarray(where), jnp.asarray(at), jnp.asarray(history))


# The options.


def bid(tricks: int, trump: int) -> int:
    return R.FIRST_BID + (tricks - 4) * 5 + trump


def spread_logits(favourites: list[int]) -> np.ndarray:
    """Logits falling by 1 along ``favourites``; every other action far below."""
    logits = np.full(120, -50.0)
    for place, action in enumerate(favourites):
        logits[action] = 5.0 - place
    return logits


AUCTION = [R.PASS, R.WITH, bid(4, 0), bid(5, 0), bid(6, 0), bid(4, 1), bid(4, 2)]
# Their chances, by softmax: 63.3%, 23.3%, 8.6%, 3.1%, 1.2%, 0.4%, 0.2%.


def test_candidates_cover_the_players_move_chances_its_favourite_first():
    found = T.candidates(spread_logits(AUCTION), sorted(AUCTION), 5.0, T.Pruning(0.97, 2, 8, 3.0))
    assert found.actions == AUCTION[:4]  # 95.1% after three options, 98.3% after four
    assert found.ranks == [0, 1, 2, 3]
    assert found.chances == pytest.approx([0.6327, 0.2328, 0.0856, 0.0315], abs=1e-4)


def test_candidates_are_at_least_min_options_and_at_most_max_options():
    logits, legal = spread_logits(AUCTION), sorted(AUCTION)
    assert T.candidates(logits, legal, 5.0, T.Pruning(0.97, 6, 8, 3.0)).actions == AUCTION[:6]
    assert T.candidates(logits, legal, 5.0, T.Pruning(0.97, 2, 3, 3.0)).actions == AUCTION[:3]
    two = [R.WITH, R.PASS]  # fewer legal options than min_options: all of them
    assert T.candidates(logits, two, 5.0, T.Pruning(0.97, 5, 8, 3.0)).actions == [R.PASS, R.WITH]


def test_candidates_drop_calls_far_above_the_expected_tricks():
    # A hand expected to take 1 trick: calls of 5 and 6 are more than 3 tricks away.
    found = T.candidates(spread_logits(AUCTION), sorted(AUCTION), 1.0, T.Pruning(1.0, 5, 8, 3.0))
    assert found.actions == [R.PASS, R.WITH, bid(4, 0), bid(4, 1), bid(4, 2)]
    assert found.ranks == [0, 1, 2, 5, 6]  # places among every legal option, dropped ones too
    # The coverage counts the options that fit: 97% of their 95.6% is held by the first three.
    kept = T.candidates(spread_logits(AUCTION), sorted(AUCTION), 1.0, T.Pruning(0.97, 2, 8, 3.0))
    assert kept.actions == [R.PASS, R.WITH, bid(4, 0)]


def test_candidates_drop_estimates_far_from_the_expected_tricks_but_never_the_favourite():
    estimates = [R.ESTIMATE_BASE + k for k in (9, 2, 6, 3, 0, 5, 1, 4)]
    found = T.candidates(
        spread_logits(estimates), sorted(estimates), 2.0, T.Pruning(1.0, 8, 8, 3.0)
    )
    # Estimating 9 is the player's move, so it stays; 6 is 4 tricks from the 2 expected.
    assert found.actions == [R.ESTIMATE_BASE + k for k in (9, 2, 3, 0, 5, 1, 4)]


def test_candidates_never_drop_a_dash_call_with_or_pass():
    assert T.near(R.DASH_CALL, 9.0, 0.0)
    assert T.near(R.WITH, 0.0, 0.0)
    assert T.near(R.PASS, 13.0, 0.0)
    assert T.near(bid(8, 4), 5.0, 3.0)
    assert not T.near(bid(9, 4), 5.0, 3.0)  # a super call 4 tricks above
    assert T.near(R.ESTIMATE_BASE + 2, 5.0, 3.0)
    assert not T.near(R.ESTIMATE_BASE + 1, 5.0, 3.0)


def test_candidates_put_the_lower_action_first_on_a_tie_as_the_player_moves():
    logits = np.zeros(120)
    found = T.candidates(logits, [R.NO_DASH_CALL, R.DASH_CALL], 3.0, T.Pruning())
    assert found.actions == [R.NO_DASH_CALL, R.DASH_CALL]
    assert found.chances == pytest.approx([0.5, 0.5])


def test_check_refuses_settings_the_narrowing_cannot_run():
    T.check(T.Settings(), T.Pruning())
    with pytest.raises(ValueError, match="stages narrow at most 5 options"):
        T.check(T.Settings(deals=(8, 16, 32)), T.Pruning(max_options=6))
    with pytest.raises(ValueError, match="min_options <= max_options"):
        T.check(T.Settings(), T.Pruning(min_options=1))
    with pytest.raises(ValueError, match="at least 2 deals"):
        T.check(T.Settings(deals=(1, 2)), T.Pruning(min_options=2, max_options=2))
    with pytest.raises(ValueError, match="coverage"):
        T.check(T.Settings(), T.Pruning(coverage=0.0))
    with pytest.raises(ValueError, match="opponent_temperature"):
        T.check(T.Settings(opponent_temperature=-1.0), T.Pruning())


# The decisions.


def test_the_decisions_equal_what_each_seat_saw(played: Played, batch: T.Batch):
    x, me = np.asarray(batch.x), np.asarray(batch.me)
    for row, seen in enumerate(played.seen[:SIZE]):
        assert me[row] == seen.obs.seat
        np.testing.assert_allclose(x[row], raw_view(seen.obs, seen.legal), atol=1e-6)


def test_the_bidding_so_far_equals_the_cpu_searchs(played: Played, batch: T.Batch):
    bidding = batch.bidding
    templates, seats = np.asarray(bidding.templates), np.asarray(bidding.seats)
    legal, chosen, valid = (np.asarray(a) for a in (bidding.legal, bidding.chosen, bidding.valid))
    longest = 0
    for row, seen in enumerate(played.seen[:SIZE]):
        want = bidding_views(seen.obs, RulesConfig())
        count = len(want)
        longest = max(longest, count)
        assert valid[row].tolist() == [True] * count + [False] * (valid.shape[1] - count)
        assert seats[row, :count].tolist() == list(want.seats)
        assert (seats[row, count:] == -1).all()
        if count:
            np.testing.assert_allclose(templates[row, :count], want.templates, atol=1e-6)
            assert legal[row, :count].tolist() == want.legal.tolist()
            assert chosen[row, :count].tolist() == want.chosen.tolist()
    assert longest > 2


def test_the_players_opinions_equal_its_saved_outputs(net_path: str, batch: T.Batch):
    logits, tricks = T.opinions(T.load_bidder(net_path), batch.x)
    x = np.asarray(batch.x)
    np.testing.assert_allclose(np.asarray(logits), load_net(net_path)(x), rtol=1e-4, atol=1e-4)
    with np.load(net_path) as data:
        hidden = np.maximum(x @ data["w0"] + data["b0"], 0.0)
        answer = hidden @ data["qw"][:, SLICES["my_tricks"]] + data["qb"][SLICES["my_tricks"]]
    chances = np.exp(answer - answer.max(axis=1, keepdims=True))
    chances /= chances.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(tricks), chances @ np.arange(14.0), rtol=1e-4)


# The deals.


def test_imagined_deals_keep_the_deciding_seats_hand_and_deal_every_card(
    net_path: str, played: Played, batch: T.Batch
):
    settings = T.Settings(deals=(2, 4), redeals=2)
    hands, keys = T.imagine(
        T.load_bidder(net_path).player,
        settings,
        jax.random.PRNGKey(0),
        batch.round,
        batch.me,
        batch.bidding,
    )
    hands = np.asarray(hands)
    assert hands.shape == (SIZE, 6, 4, 52)
    assert keys.shape == (SIZE, 6, 2)
    for row, spot in enumerate(played.spots[:SIZE]):
        mine = bits(played.rounds[spot.round].hands)[spot.seat]
        for deal in hands[row]:
            assert deal[spot.seat].tolist() == mine.tolist()
            assert (deal.sum(0) == 1).all()  # every card, once
            assert (deal.sum(1) == 13).all()


# The play-outs.


def real_deals(played: Played) -> jax.Array:
    """``[SIZE, 1, 4, 52]``: each decision's round as it was really dealt."""
    deals = [bits(played.rounds[spot.round].hands) for spot in played.spots[:SIZE]]
    return jnp.asarray(np.stack(deals))[:, None]


def test_a_play_out_of_the_real_deal_by_top_moves_replays_the_real_round(
    net_path: str, played: Played, batch: T.Batch
):
    """Every seat made the net's top move in the real rounds, so with the other seats' bids not
    drawn (temperature 0) the move really made, on the real deal, must score the real round."""
    made = jnp.asarray([seen.action for seen in played.seen[:SIZE]], jnp.int32)
    values = T.evaluate(
        T.load_bidder(net_path).player,
        T.Settings(opponent_temperature=0.0),
        batch.round,
        batch.me,
        real_deals(played),
        jnp.zeros((SIZE, 1, 2), jnp.uint32),
        jnp.arange(SIZE),
        made,
    )
    skipped = 0
    for row, spot in enumerate(played.spots[:SIZE]):
        result = played.results[spot.round]
        weights = standings_weights(played.rounds[spot.round].scores, spot.seat)
        others = sum(w * score for w, score in zip(weights, result.scores, strict=True))
        assert float(values[row, 0]) == pytest.approx(result.scores[spot.seat] - others, abs=1e-3)
        skipped += result.skipped
    assert skipped < SIZE  # most rounds were played to the end


def test_only_the_other_seats_bids_are_drawn(net_path: str, batch: T.Batch):
    player = T.load_bidder(net_path).player
    x = np.asarray(batch.x)
    logits = load_net(net_path)(x)[:, : T.BID_ACTIONS]
    legal = x[:, -R.NUM_ACTIONS :][:, : T.BID_ACTIONS] > 0.5
    top = np.where(legal, logits, -np.inf).argmax(1)
    keys = jax.random.split(jax.random.PRNGKey(5), SIZE)
    turn = jnp.zeros((), jnp.int32)
    # The deciding seat is the one to move in every round of the batch: its top move, drawn or not.
    mine = T.bid_moves(player, batch.round, batch.me, keys, turn, 30.0)
    assert np.asarray(mine).tolist() == top.tolist()
    # Told another seat decides, the same rounds' moves are drawn; at temperature 0 they are not.
    other = (batch.me + 1) % 4
    drawn = T.bid_moves(player, batch.round, other, keys, turn, 30.0)
    assert np.asarray(drawn).tolist() != top.tolist()
    assert np.take_along_axis(legal, np.asarray(drawn)[:, None], 1).all()
    again = T.bid_moves(player, batch.round, other, keys, turn, 30.0)
    assert np.asarray(again).tolist() == np.asarray(drawn).tolist()  # same keys: same draws
    later = T.bid_moves(player, batch.round, other, keys, turn + 1, 30.0)
    assert np.asarray(later).tolist() != np.asarray(drawn).tolist()  # another move: new draws
    greedy = T.bid_moves(player, batch.round, other, keys, turn, 0.0)
    assert np.asarray(greedy).tolist() == top.tolist()


def test_drawn_bids_follow_the_players_move_chances(net_path: str, batch: T.Batch):
    player = T.load_bidder(net_path).player
    x = np.asarray(batch.x)
    legal = x[:, -R.NUM_ACTIONS :][:, : T.BID_ACTIONS] > 0.5
    row = int(legal.sum(1).argmax())  # the decision with the most options
    assert legal[row].sum() > 3
    temperature = 20.0  # a random net's chances are one-sided: flattened, every option is drawn
    scores = np.where(legal[row], load_net(net_path)(x[row])[: T.BID_ACTIONS], -np.inf)
    chances = np.exp((scores - scores.max()) / temperature)
    chances /= chances.sum()
    draws = 4000
    one = jax.tree.map(lambda a: jnp.broadcast_to(a[row], (draws, *a.shape[1:])), batch.round)
    other = jnp.full(draws, (int(batch.me[row]) + 1) % 4)
    keys = jax.random.split(jax.random.PRNGKey(6), draws)
    moves = np.asarray(T.bid_moves(player, one, other, keys, jnp.zeros((), jnp.int32), temperature))
    rates = np.bincount(moves, minlength=T.BID_ACTIONS) / draws
    assert np.abs(rates - chances).max() < 0.03
    assert (rates > 0).sum() > 3


def test_a_seat_with_no_bidding_after_it_scores_the_same_whatever_is_drawn(
    net_path: str, played: Played, batch: T.Batch
):
    """A move that ends the bidding leaves no other seat's bid to draw, so its play-outs of one
    deal all score the same; a Dash Call's play-outs hold drawn bids, so they do differ."""
    made = jnp.asarray([seen.action for seen in played.seen[:SIZE]], jnp.int32)
    after = np.asarray(jax.vmap(R.step)(batch.round, made).phase)
    last = np.flatnonzero(after >= R.PLAY).tolist()
    dash = [row for row, seen in enumerate(played.seen[:SIZE]) if seen.obs.phase is Phase.DASH]
    assert last
    assert dash
    deals = 16
    hands = jnp.broadcast_to(real_deals(played), (SIZE, deals, 4, 52))
    keys = jax.random.split(jax.random.PRNGKey(7), SIZE * deals).reshape(SIZE, deals, 2)
    values = np.asarray(
        T.evaluate(
            T.load_bidder(net_path).player,
            T.Settings(opponent_temperature=30.0),
            batch.round,
            batch.me,
            hands,
            keys,
            jnp.arange(SIZE),
            made,
        )
    )
    for row in last:
        assert (values[row] == values[row, 0]).all()
    assert any((values[row] != values[row, 0]).any() for row in dash)


# The narrowing.


def test_narrow_keeps_the_favourite_and_the_better_half_of_the_alternatives():
    assert T.narrow([0, 1, 2, 3, 4], [9.0, 1.0, 5.0, 3.0, 4.0], 4) == [0, 2, 4]
    assert T.narrow([0, 1, 2, 3, 4], [-9.0, 1.0, 5.0, 3.0, 4.0], 4) == [0, 2, 4]  # a bad favourite
    assert T.narrow([0, 2, 5, 7], [0.0, 1.0, 3.0, 2.0], 4) == [0, 5, 7]  # 3 alternatives: 2 go on


def test_narrow_keeps_no_more_than_the_next_stage_holds_and_at_least_one():
    assert T.narrow(list(range(8)), [0.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 2) == [0, 1, 2]
    assert T.narrow([0, 3, 6], [0.0, 1.0, 2.0], 1) == [0, 6]
    assert T.narrow([0, 3], [0.0, -5.0], 1) == [0, 3]  # one alternative: it meets the favourite
    assert T.narrow([0, 1, 2], [0.0, 3.0, 3.0], 1) == [0, 1]  # a tie: the more favoured slot


def test_verdicts_measure_each_option_against_the_favourite_on_the_deals_they_shared():
    played = np.full((2, 3, 6), np.nan, np.float32)  # the second decision was not searched
    played[0, 0] = [1, 2, 3, 4, 5, 6]  # the favourite plays every deal
    played[0, 1, :2] = [5, 0]  # dropped after the first stage of 2 deals
    played[0, 2] = [2, 2, 5, 4, 7, 7]  # the last alternative
    found = T.verdicts(played, 2)
    assert found.deals.tolist() == [[6, 2, 6], [0, 0, 0]]
    assert found.value[0].tolist() == pytest.approx([3.5, 2.5, 4.5])
    assert found.margin[0].tolist() == pytest.approx(
        [0.0, 1.0, 1.0]
    )  # leads [4, -2], [1, 0, 2, 0, 2, 1]
    assert found.error[0].tolist() == pytest.approx([0.0, 3.0, 0.3651], abs=1e-4)
    assert found.opening[0].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert found.pick.tolist() == [2, 0]
    assert not found.value[1].any()
    assert not found.margin[1].any()


def test_verdicts_pick_the_favourite_on_a_tie_and_only_an_alternative_that_played_every_deal():
    played = np.full((2, 3, 4), np.nan, np.float32)
    played[0, 0] = [1, 2, 3, 4]
    played[0, 1] = [4, 3, 2, 1]  # level with the favourite over every deal
    played[1, 0] = [1, 1, 1, 1]
    played[1, 1, :2] = [9, 9]  # far ahead, but dropped: the last alternative decides
    played[1, 2] = [0, 0, 0, 0]
    assert T.verdicts(played, 2).pick.tolist() == [0, 0]


@pytest.fixture(scope="module")
def taught(net_path: str, played: Played) -> tuple[T.Teacher, T.Batch, np.ndarray, T.Verdicts]:
    """The batch searched in three stages of 2, 4 and 8 deals: up to five options a decision;
    the first decision is left a single option, which is nothing to search."""
    teacher = T.Teacher(T.load_bidder(net_path), T.Settings(deals=(2, 4, 8), redeals=2))
    spots = played.spots[:SIZE]
    arrays, where, at, history = batch_arrays(played.rounds, spots, SIZE)
    batch, logits, tricks = teacher.look(arrays, where, at, history)
    options = np.full((SIZE, 5), -1, np.int32)
    for row, spot in enumerate(spots):
        found = T.candidates(logits[row], spot.legal, float(tricks[row]), T.Pruning(1.0, 5, 5, 13))
        assert found.actions[0] == played.seen[row].action  # the favourite is the move made
        options[row, : len(found.actions)] = found.actions
    options[0, 1:] = -1
    return teacher, batch, options, teacher.teach(jax.random.PRNGKey(1), batch, options)


def test_every_option_plays_the_deals_its_stage_gives_it(
    taught: tuple[T.Teacher, T.Batch, np.ndarray, T.Verdicts],
):
    _, _, options, found = taught
    by_count = {2: [14], 3: [2, 14], 4: [2, 6, 14], 5: [2, 2, 6, 14]}  # the alternatives' deals
    seen: set[int] = set()
    for row in range(1, SIZE):
        count = int((options[row] >= 0).sum())
        seen.add(count)
        assert found.deals[row, 0] == 14  # the favourite plays every deal
        assert sorted(found.deals[row, 1:count].tolist()) == by_count[count]
        assert not found.deals[row, count:].any()
        assert found.margin[row, 0] == 0.0
        assert found.error[row, 0] == 0.0
        assert np.isfinite(found.value[row]).all()
        last = 1 + int(np.argmax(found.deals[row, 1:count]))
        better = found.value[row, last] > found.value[row, 0]
        assert found.pick[row] == (last if better else 0)
    assert {2, 5} <= seen  # Dash Calls with two options, auctions with five


def test_a_decision_with_one_option_is_not_searched(
    taught: tuple[T.Teacher, T.Batch, np.ndarray, T.Verdicts],
):
    _, _, _, found = taught
    assert not found.deals[0].any()
    assert found.pick[0] == 0


def test_teaching_is_the_same_from_the_same_key_and_differs_on_fresh_deals(
    taught: tuple[T.Teacher, T.Batch, np.ndarray, T.Verdicts],
):
    teacher, batch, options, found = taught
    again = teacher.teach(jax.random.PRNGKey(1), batch, options)
    assert again.value.tolist() == found.value.tolist()
    assert again.pick.tolist() == found.pick.tolist()
    fresh = teacher.teach(jax.random.PRNGKey(2), batch, options)
    assert fresh.value.tolist() != found.value.tolist()
    assert fresh.deals[:, 0].tolist() == found.deals[:, 0].tolist()
