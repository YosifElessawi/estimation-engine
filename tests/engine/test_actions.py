"""The frozen 120-id action space (SPEC §5.2), TBL-5 and TBL-6."""

import itertools

import pytest

from estimation.engine.actions import (
    DASH_CALL,
    NO_DASH_CALL,
    NUM_ACTIONS,
    PASS,
    WITH,
    action_str,
    bid_action,
    bid_of,
    card_of,
    estimate_action,
    estimate_of,
    is_bid,
    is_estimate,
    is_play,
    play_action,
)
from estimation.engine.cards import CLUBS, DIAMONDS, HEARTS, SANS, SPADES, parse_card


def test_action_space_has_120_ids_in_the_spec_layout():
    assert NUM_ACTIONS == 120
    assert (NO_DASH_CALL, DASH_CALL, WITH, PASS) == (0, 1, 52, 53)
    bids = [bid_action(n, t) for n in range(4, 14) for t in range(5)]
    assert bids == list(range(2, 52))
    assert [estimate_action(k) for k in range(14)] == list(range(54, 68))
    assert [play_action(card) for card in range(52)] == list(range(68, 120))


def test_bid_id_formula_matches_spec():
    assert bid_action(4, CLUBS) == 2
    assert bid_action(5, HEARTS) == 2 + 1 * 5 + 2
    assert bid_action(13, SANS) == 51


def test_decoders_round_trip():
    for n, t in itertools.product(range(4, 14), range(5)):
        assert bid_of(bid_action(n, t)) == (n, t)
    for k in range(14):
        assert estimate_of(estimate_action(k)) == k
    for card in range(52):
        assert card_of(play_action(card)) == card


def test_action_kind_predicates_partition_the_space():
    for action in range(NUM_ACTIONS):
        kinds = [is_bid(action), is_estimate(action), is_play(action), action in (0, 1, 52, 53)]
        assert kinds.count(True) == 1


def test_TBL_5_trump_strength_order():
    for n in range(4, 14):
        ids = [bid_action(n, t) for t in (SANS, SPADES, HEARTS, DIAMONDS, CLUBS)]
        assert ids == sorted(ids, reverse=True)


def test_TBL_6_bid_beats_another_with_more_tricks_or_same_tricks_stronger_trump():
    bids = range(2, 52)
    for a, b in itertools.product(bids, bids):
        (na, ta), (nb, tb) = bid_of(a), bid_of(b)
        beats = na > nb or (na == nb and ta > tb)
        assert (a > b) == beats


@pytest.mark.parametrize(
    "call",
    [
        lambda: bid_action(3, CLUBS),
        lambda: bid_action(14, CLUBS),
        lambda: bid_action(4, 5),
        lambda: estimate_action(14),
        lambda: estimate_action(-1),
        lambda: play_action(52),
        lambda: bid_of(52),
        lambda: estimate_of(53),
        lambda: card_of(67),
    ],
)
def test_encoders_reject_values_outside_the_space(call):
    with pytest.raises(ValueError, match=r"action|bid|estimate|card"):
        call()


def test_action_str_names_every_kind():
    assert action_str(NO_DASH_CALL) == "No Dash Call"
    assert action_str(DASH_CALL) == "Dash Call"
    assert action_str(bid_action(5, HEARTS)) == "Bid 5 Hearts"
    assert action_str(WITH) == "With"
    assert action_str(PASS) == "Pass"
    assert action_str(estimate_action(3)) == "Estimate 3"
    assert action_str(play_action(parse_card("JH"))) == "Play JH"
    with pytest.raises(ValueError, match="action"):
        action_str(120)
