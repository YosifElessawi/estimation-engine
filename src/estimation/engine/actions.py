"""The action space: one fixed list of 120 ids covering every decision (SPEC §5.2).

RL milestones depend on these ids, so they must never change:

======  ==========================================================================
Ids     Action
======  ==========================================================================
0       No Dash Call
1       Dash Call
2-51    Bid ``(n, trump)``, id ``2 + (n - 4) * 5 + trump``; also used for super calls
52      With
53      Pass (on the top bidder's final turn: keep my bid)
54-67   Estimate ``k``, id ``54 + k``
68-119  Play a card, id ``68 + card id``
======  ==========================================================================

A higher bid id is always a stronger bid (TBL-6).
"""

from .cards import NUM_CARDS, SANS, TRUMP_NAMES, card_str

NUM_ACTIONS = 120

NO_DASH_CALL = 0
DASH_CALL = 1
FIRST_BID = 2
LAST_BID = 51
WITH = 52
PASS = 53
ESTIMATE_BASE = 54
PLAY_BASE = 68

MIN_BID_TRICKS = 4
MAX_TRICKS = 13


def bid_action(tricks: int, trump: int) -> int:
    """Action id of the bid ``(tricks, trump)``, for 4-13 tricks and trump 0-4 (Sans is 4)."""
    if not (MIN_BID_TRICKS <= tricks <= MAX_TRICKS and 0 <= trump <= SANS):
        raise ValueError(f"no bid of {tricks} tricks in trump {trump}")
    return FIRST_BID + (tricks - MIN_BID_TRICKS) * 5 + trump


def bid_of(action: int) -> tuple[int, int]:
    """``(tricks, trump)`` of a bid action."""
    if not is_bid(action):
        raise ValueError(f"action {action} is not a bid")
    tricks, trump = divmod(action - FIRST_BID, 5)
    return tricks + MIN_BID_TRICKS, trump


def estimate_action(k: int) -> int:
    """Action id of estimating ``k`` tricks, 0-13."""
    if not 0 <= k <= MAX_TRICKS:
        raise ValueError(f"no estimate of {k} tricks")
    return ESTIMATE_BASE + k


def estimate_of(action: int) -> int:
    if not is_estimate(action):
        raise ValueError(f"action {action} is not an estimate")
    return action - ESTIMATE_BASE


def play_action(card: int) -> int:
    """Action id of playing card id ``card``."""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"no card with id {card}")
    return PLAY_BASE + card


def card_of(action: int) -> int:
    if not is_play(action):
        raise ValueError(f"action {action} is not a card play")
    return action - PLAY_BASE


def is_bid(action: int) -> bool:
    return FIRST_BID <= action <= LAST_BID


def is_estimate(action: int) -> bool:
    return ESTIMATE_BASE <= action < PLAY_BASE


def is_play(action: int) -> bool:
    return PLAY_BASE <= action < NUM_ACTIONS


def action_str(action: int) -> str:
    """Human-readable name of an action, for logs and tests."""
    if action == NO_DASH_CALL:
        return "No Dash Call"
    if action == DASH_CALL:
        return "Dash Call"
    if is_bid(action):
        tricks, trump = bid_of(action)
        return f"Bid {tricks} {TRUMP_NAMES[trump]}"
    if action == WITH:
        return "With"
    if action == PASS:
        return "Pass"
    if is_estimate(action):
        return f"Estimate {action - ESTIMATE_BASE}"
    if is_play(action):
        return f"Play {card_str(action - PLAY_BASE)}"
    raise ValueError(f"no action with id {action}")
