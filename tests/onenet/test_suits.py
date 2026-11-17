"""Suit shuffles of the view: once the trump is settled, a round played with its suits renamed
shows every seat the shuffled view of the round as it was, move for move."""

import random

import numpy as np
import pytest

from estimation.engine import Phase
from estimation.engine.cards import cards_of, deal, mask_of
from estimation.engine.round import Round
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.onenet import suits as S

NO_DASH, PASS, ESTIMATES, PLAY_BASE = 0, 53, range(54, 68), 68


def play_shuffled(seed: int, p: tuple[int, int, int, int], opening: int) -> tuple[int, bool]:
    """Play a round and its suit-shuffled twin with the same moves, every bid in one trump (no
    Dash Call, the ``opening`` bid from whoever bids first, a Pass from everyone else, then any
    estimate and any card). From the first estimate on, every seat's view of the twin must be
    the shuffled view of the round. Returns how many decisions were compared and whether cards
    were played."""
    index, moves = S.suit_maps(p)
    hands = deal(seed, 1)
    twin_hands = (
        mask_of(int(moves[PLAY_BASE + c]) - PLAY_BASE for c in cards_of(hands[0])),
        mask_of(int(moves[PLAY_BASE + c]) - PLAY_BASE for c in cards_of(hands[1])),
        mask_of(int(moves[PLAY_BASE + c]) - PLAY_BASE for c in cards_of(hands[2])),
        mask_of(int(moves[PLAY_BASE + c]) - PLAY_BASE for c in cards_of(hands[3])),
    )
    rnd, twin = Round.new(1, hands), Round.new(1, twin_hands)
    rng = random.Random(seed)
    compared, played = 0, False
    opened = False
    while rnd.phase is not Phase.DONE:
        legal, twin_legal = rnd.legal_actions(), twin.legal_actions()
        if rnd.phase in (Phase.ESTIMATION, Phase.PLAY):  # the trump is settled
            assert sorted(int(moves[a]) for a in legal) == sorted(twin_legal)
            for seat in range(4):
                mine = raw_view(rnd.observation(seat), legal)
                theirs = raw_view(twin.observation(seat), twin_legal)
                assert np.array_equal(theirs, mine[index]), f"seat {seat} at decision {compared}"
            compared += 1
        if rnd.phase is Phase.PLAY:
            action = rng.choice(legal)
            played = True
        elif NO_DASH in legal:
            action = NO_DASH
        elif any(a in ESTIMATES for a in legal):
            action = rng.choice([a for a in legal if a in ESTIMATES])
        elif not opened and opening in legal:
            action, opened = opening, True
        else:
            action = PASS if PASS in legal else legal[0]
        rnd.step(action)
        twin.step(int(moves[action]))
    return compared, played


@pytest.mark.parametrize(
    ("p", "trump"),
    [((2, 1, 3, 0), 1), ((1, 3, 2, 0), 2), ((3, 0, 1, 2), 4), ((1, 0, 3, 2), 4)],
)
def test_a_shuffled_round_shows_the_shuffled_view_at_every_decision(
    p: tuple[int, int, int, int], trump: int
):
    opening = 2 + trump  # 4 of that trump: the lowest bid in it
    rounds = 0
    for seed in range(40):
        compared, played = play_shuffled(seed, p, opening)
        assert compared >= 52 or not played  # a played round: its estimates and its 52 cards
        rounds += played
    assert rounds >= 10  # the round was bid, estimated and played to its last card


def test_the_move_map_renames_cards_and_bids_with_their_suit():
    index, moves = S.suit_maps((2, 1, 3, 0))  # suit 2's cards move to suit 0, suit 0's to suit 3
    assert moves[PLAY_BASE + 2 * 13 + 5] == PLAY_BASE + 0 * 13 + 5
    assert moves[PLAY_BASE + 0 * 13 + 12] == PLAY_BASE + 3 * 13 + 12
    assert moves[PLAY_BASE + 13 + 7] == PLAY_BASE + 13 + 7  # suit 1 stays
    assert moves[2 + 5 * 3 + 2] == 2 + 5 * 3 + 0  # 7 of suit 2 is now 7 of suit 0
    assert moves[2 + 5 * 3 + 4] == 2 + 5 * 3 + 4  # no trump is no suit: it stays
    assert moves[:2].tolist() == [0, 1]
    assert moves[52:68].tolist() == list(range(52, 68))  # With, Pass and estimates name no suit
    assert sorted(index.tolist()) == list(range(S.VIEW))  # a view only changes places
    assert sorted(moves.tolist()) == list(range(S.ACTIONS))
    same, kept = S.suit_maps((0, 1, 2, 3))
    assert same.tolist() == list(range(S.VIEW))
    assert kept.tolist() == list(range(S.ACTIONS))
    with pytest.raises(ValueError, match="each of the four suits once"):
        S.suit_maps((0, 1, 1, 3))


def test_the_tables_keep_each_trump_in_place():
    views, moves = S.shuffle_tables()
    assert views.shape == (6, S.SHUFFLES, S.VIEW)
    assert moves.shape == (6, S.SHUFFLES, S.ACTIONS)
    for trump in range(4):
        assert (views[trump, :, S.TRUMP + trump] == S.TRUMP + trump).all()
        assert len({tuple(row) for row in views[trump].tolist()}) == 6  # the other three suits
        assert (moves[trump, :, 2 + trump] == 2 + trump).all()  # a bid in the trump stays
    assert len({tuple(row) for row in views[S.NO_TRUMP].tolist()}) == 24
    assert (views[S.NO_TRUMP_YET] == np.arange(S.VIEW)).all()  # the auction is never shuffled
    assert (moves[S.NO_TRUMP_YET] == np.arange(S.ACTIONS)).all()


def test_only_a_view_with_a_settled_trump_and_no_bid_to_make_is_shuffled():
    views = np.zeros((5, S.VIEW), np.float16)
    views[0, S.TRUMP + 2] = 1.0  # an estimate under suit 2
    views[1, S.TRUMP + S.NO_TRUMP] = 1.0  # an estimate with no trump
    views[2, S.TRUMP + S.NO_TRUMP_YET] = 1.0  # the auction: the suits are ranked
    views[3, S.TRUMP + 1] = 1.0  # a super call is on offer: a bid's suit is ranked too
    views[3, S.LEGAL + 2 + 5 * 6 + 3] = 1.0
    views[4, S.TRUMP + 3] = 1.0  # card play: only cards are legal
    views[4, S.LEGAL + S.PLAY_BASE + 9] = 1.0
    views[:2, S.LEGAL + 54 : S.LEGAL + 60] = 1.0
    assert S.shuffle_rows(views).tolist() == [2, S.NO_TRUMP, S.NO_TRUMP_YET, S.NO_TRUMP_YET, 3]
