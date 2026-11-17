"""Suit shuffles of the player's view (estimation.jaxenv.encode): the same decision with the
suits renamed. Needs only numpy.

Once the trump is settled, a round plays the same whichever of the other suits is which: only the
trump is special (PLY-3). So an estimate's view with the other three suits shuffled (all four in
a round with no trump) is an estimate just as good, and the bid lessons learn from every shuffle
of an estimate's view (``estimation.onenet.bid_leads``). During the auction the suits are not
alike, a higher suit outbids a lower one, so those views are never shuffled.

A shuffle ``p`` moves suit ``p[s]``'s cards to suit ``s``. ``suit_maps`` gives what that does to
a view (``new = old[index]``) and to a move (the new id of each old id: a card or a bid changes
its suit with the cards). ``shuffle_tables`` holds, for each trump, 24 shuffles that leave that
trump in place, and ``shuffle_rows`` says which views may be shuffled at all.
"""

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

VIEW = 747  # estimation.jaxenv.encode.SIZE
ACTIONS = 120
SHUFFLES = 24  # shuffles kept per trump
NO_TRUMP, NO_TRUMP_YET = 4, 5  # the trump's places after the four suits

# Where the view's parts start (estimation.jaxenv.encode.encode, in its order).
HAND = 0  # 52 cards
PLAYED = 52  # 4 seats x 52 cards
TABLE = 260  # 4 seats x 52 cards
SHOWN_OUT = 468  # 4 seats x 4 suits
CALLS = 484  # 4 seats x 5 trumps
TOP_BID = 504  # 51: no bid, then 10 numbers x 5 trumps
TRUMP = 603  # 6: the four suits, no trump, none yet
FIXED_TRUMP = 609  # 6, the same
LEGAL = VIEW - ACTIONS  # the legal moves, by action id
FIRST_BID, PLAY_BASE = 2, 68

Ints = npt.NDArray[np.int32]


def suit_maps(p: Sequence[int]) -> tuple[Ints, Ints]:
    """For the shuffle that moves suit ``p[s]`` to suit ``s``: the view's index map ``[VIEW]``
    (the shuffled view is ``view[index]``) and the move map ``[ACTIONS]`` (the id each move has
    after the shuffle)."""
    if sorted(p) != [0, 1, 2, 3]:
        raise ValueError(f"a shuffle names each of the four suits once, not {tuple(p)}")
    index = np.arange(VIEW, dtype=np.int32)
    for s in range(4):
        cards, from_cards = np.arange(13) + 13 * s, np.arange(13) + 13 * p[s]
        index[HAND + cards] = HAND + from_cards
        index[LEGAL + PLAY_BASE + cards] = LEGAL + PLAY_BASE + from_cards
        numbers = 5 * np.arange(10)
        index[TOP_BID + 1 + numbers + s] = TOP_BID + 1 + numbers + p[s]
        index[LEGAL + FIRST_BID + numbers + s] = LEGAL + FIRST_BID + numbers + p[s]
        index[TRUMP + s] = TRUMP + p[s]
        index[FIXED_TRUMP + s] = FIXED_TRUMP + p[s]
        for seat in range(4):
            index[PLAYED + 52 * seat + cards] = PLAYED + 52 * seat + from_cards
            index[TABLE + 52 * seat + cards] = TABLE + 52 * seat + from_cards
            index[SHOWN_OUT + 4 * seat + s] = SHOWN_OUT + 4 * seat + p[s]
            index[CALLS + 5 * seat + s] = CALLS + 5 * seat + p[s]
    moves = np.arange(ACTIONS, dtype=np.int32)
    for s in range(4):  # suit p[s] becomes suit s
        moves[FIRST_BID + 5 * np.arange(10) + p[s]] = FIRST_BID + 5 * np.arange(10) + s
        moves[PLAY_BASE + 13 * p[s] + np.arange(13)] = PLAY_BASE + 13 * s + np.arange(13)
    return index, moves


def shuffle_tables() -> tuple[Ints, Ints]:
    """``[6, SHUFFLES, VIEW]`` index maps and ``[6, SHUFFLES, ACTIONS]`` move maps: for each
    trump (the four suits, no trump, none yet) the shuffles that leave it in place. A suit trump
    has 6, each kept 4 times; no trump has all 24; with no trump yet the suits are not alike
    (the auction ranks them), so its 24 are all the shuffle that changes nothing."""
    shuffles = list(itertools.permutations(range(4)))
    views = np.zeros((6, SHUFFLES, VIEW), np.int32)
    moves = np.zeros((6, SHUFFLES, ACTIONS), np.int32)
    for trump in range(6):
        if trump < NO_TRUMP:
            kept = [p for p in shuffles if p[trump] == trump] * 4
        elif trump == NO_TRUMP:
            kept = shuffles
        else:
            kept = [(0, 1, 2, 3)] * SHUFFLES
        for k, p in enumerate(kept):
            views[trump, k], moves[trump, k] = suit_maps(p)
    return views, moves


def shuffle_rows(views: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
    """``[N]``: each view's row of ``shuffle_tables``: its trump (a suit 0-3 or ``NO_TRUMP``)
    when its suits may be shuffled, else ``NO_TRUMP_YET``, the row that changes nothing. A view
    may be shuffled when its trump is settled and no bid is legal: a bid's suit is ranked (a
    super call must outbid the call it answers), so a view that offers one is left as it is."""
    trump = np.asarray(views[:, TRUMP : TRUMP + 6]).argmax(1).astype(np.int64)
    bids = np.asarray(views[:, LEGAL + FIRST_BID : LEGAL + FIRST_BID + 50]).any(1)
    return np.where(bids, NO_TRUMP_YET, trump)
