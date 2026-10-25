"""How many tricks a trump holding takes (design 5.1).

Yosif counts trumps by how the missing ones are split among the other three players: "estimate how
many I could collect from this suit by estimating probable deals on other players" `[Yosif]`. When
he plays his top trumps, everyone who still has trumps must follow, so each round draws up to four.
Voids add no separate tricks: they are why low trumps win, so they are already in this count
`[Yosif]`.

Holdings are 13-bit rank patterns: bit r is rank r, 2 is 0 and A is 12.
"""

import random
from collections.abc import Sequence
from functools import cache

SAMPLES = 400  # sampled splits per holding [Claude]
MAX_DRAWS = 200_000  # splits drawn before a long-opponent condition is given up
LONG = 4  # a "long" opponent holds at least this many trumps [Claude]
UNSEEN = 39  # cards in the other three hands at the start of a round
HAND = 13


def _ranks(pattern: int) -> list[int]:
    """The ranks in ``pattern``, highest first."""
    return [rank for rank in range(12, -1, -1) if pattern >> rank & 1]


def split_tricks(mine: int, opponents: Sequence[int]) -> int:
    """Trump tricks ``mine`` takes against these opponents' trumps when drawn from the top.

    Each round I lead my highest trump while any opponent still has one. In seat order, each
    opponent plays its highest trump if it beats the best card so far and its lowest otherwise;
    I win the round if my card is still the best. Once the opponents are out of trumps, every
    trump I still hold is a trick.
    """
    hand = _ranks(mine)
    others = [_ranks(pattern) for pattern in opponents]
    tricks = 0
    while hand and any(others):
        best = hand.pop(0)
        mine_wins = True
        for held in others:
            if not held:
                continue
            if held[0] > best:
                best = held.pop(0)
                mine_wins = False
            else:
                held.pop()
        tricks += mine_wins
    return tricks + len(hand)


@cache
def trump_odds(mine: int, long_opponents: int = 0) -> tuple[float, ...]:
    """The chance of taking 0, 1, ... ``len(mine)`` trump tricks, over sampled splits.

    The missing trumps are dealt at random among three hands of 13 from the 39 unseen cards. With
    ``long_opponents = j``, only splits where at least j opponents hold 4 or more of them count
    (a player who bid the suit probably holds length there); if too few trumps are missing for
    that, or no split meets it within ``MAX_DRAWS``, the condition is dropped. Sampling is seeded
    from the arguments, so the result is the same on every call.
    """
    held = mine.bit_count()
    missing = [rank for rank in range(13) if not mine >> rank & 1]
    want = min(long_opponents, len(missing) // LONG)
    rng = random.Random(mine * 4 + long_opponents)
    counts = [0] * (held + 1)
    kept = draws = 0
    while kept < SAMPLES:
        draws += 1
        if want and draws > MAX_DRAWS:
            want = 0
        hands = [0, 0, 0]
        for rank, slot in zip(missing, rng.sample(range(UNSEEN), len(missing)), strict=True):
            hands[slot // HAND] |= 1 << rank
        if want and sum(h.bit_count() >= LONG for h in hands) < want:
            continue
        counts[split_tricks(mine, hands)] += 1
        kept += 1
    return tuple(count / SAMPLES for count in counts)
