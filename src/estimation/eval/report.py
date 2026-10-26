"""Readable text for the bidding evaluator: hands, moves and situations, as in the quiz."""

from estimation.engine import DASH_CALL, NO_DASH_CALL, PASS, WITH, Phase, bid_of
from estimation.engine.actions import ESTIMATE_BASE, is_bid, is_estimate, is_play

from .decisions import Decision

SUITS = "♣♦♥♠"
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
TRUMPS = ("♣", "♦", "♥", "♠", "Sans")


def hand_text(hand: int) -> str:
    """A hand by suit, Spades first: ``♠ A K 9 | ♥ Q 3 | ♦ - | ♣ A 7``."""
    rows = []
    for suit in (3, 2, 1, 0):
        ranks = [RANKS[r] for r in range(12, -1, -1) if hand >> (13 * suit + r) & 1]
        rows.append(f"{SUITS[suit]} {' '.join(ranks) if ranks else '-'}")
    return "  |  ".join(rows)


def action_text(action: int) -> str:
    if action == DASH_CALL:
        return "Dash Call"
    if action == NO_DASH_CALL:
        return "no Dash Call"
    if action == PASS:
        return "Pass"
    if action == WITH:
        return "With"
    if is_bid(action):
        tricks, trump = bid_of(action)
        return f"{tricks} {TRUMPS[trump]}"
    if is_estimate(action):
        return f"estimate {action - ESTIMATE_BASE}"
    return f"action {action}"


def situation(d: Decision) -> list[str]:
    """The decision as Yosif would see it at the table."""
    obs = d.obs
    me = d.seat

    def who(seat: int) -> str:
        return "you" if seat == me else f"seat {seat}"

    kind = f"fixed trump {TRUMPS[d.fixed_trump]}" if d.fixed_trump is not None else "open"
    lines = [
        f"Round {d.round_no} ({kind}), you are seat {me}; seat {(d.round_no - 1) % 4} acted first.",
        "Scores before this round: " + ", ".join(f"{who(s)} {d.scores[s]}" for s in range(4)) + ".",
        f"Your hand: {hand_text(obs.hand)}",
    ]
    dashes = [who(s) for s, a in obs.log if a == DASH_CALL]
    if obs.phase is not Phase.DASH and d.fixed_trump is None:
        lines.append("Dash Calls: " + (", ".join(dashes) if dashes else "none"))
    moves = [
        f"{who(s)}: {action_text(a)}"
        for s, a in obs.log
        if not is_play(a) and a not in (DASH_CALL, NO_DASH_CALL)
    ]
    lines.append("Bids and estimates so far: " + ("; ".join(moves) if moves else "none yet"))
    if obs.phase is Phase.ESTIMATION and obs.top_bid is not None and obs.caller is not None:
        lines.append(f"Contract: {action_text(obs.top_bid)} by {who(obs.caller)}.")
    return lines
