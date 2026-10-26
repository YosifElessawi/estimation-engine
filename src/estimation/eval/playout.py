"""Pricing one decision: deal the unseen cards again, play each choice out, score it (design 5.4).

Each choice is played on the same deals with the same seeds, so the differences between choices
come from the choices, not from luck. The deciding seat keeps its hand; the other 39 cards are
dealt again to fit what it could see (hand sizes and the void announcements, TBL-7). The round is
then rebuilt from its actions so far, the choice is applied, every seat makes its later Dash
Call, auction and estimation moves with its own bot, and every seat plays its cards with one card
player. A choice is worth the decider's round score minus the others', those ahead in the game
counted double (the weights of PIMC's leader option).
"""

import math
import random
from collections.abc import Sequence

from estimation.bots import make_bot
from estimation.bots.sampling import deal_unseen
from estimation.bots.scoring import standings_weights
from estimation.engine import Phase, Round, RulesConfig
from estimation.engine.cards import FULL_DECK, SUIT_MASKS

from .decisions import Decision

HUMAN = "human"
HUMAN_STAND_IN = "yosif"  # makes a human seat's later decisions in a play-out [Claude]
MAX_REDEALS = 200  # as PIMC


def _announced(hand: int) -> bool:
    return any(not hand & mask for mask in SUIT_MASKS)


def redeal(d: Decision, rng: random.Random) -> tuple[int, int, int, int]:
    """A deal that keeps the decider's hand and fits the void announcements it saw.

    Each player announced void is dealt a hand missing one suit picked at random, as PIMC
    samples; other players must hold every suit. After ``MAX_REDEALS`` misses the last deal is
    kept, as PIMC does.
    """
    mine = d.obs.hand
    unseen = FULL_DECK & ~mine
    others = [seat for seat in range(4) if seat != d.seat]
    announced = set(d.obs.void_announced)
    last: tuple[int, int, int, int] | None = None
    for _ in range(MAX_REDEALS):
        voids = [1 << rng.randrange(4) if seat in announced else 0 for seat in others]
        try:
            three = deal_unseen(unseen, [13, 13, 13], voids, rng)
        except ValueError:  # those voids cannot all hold with these cards: pick again
            continue
        hands = [0, 0, 0, 0]
        hands[d.seat] = mine
        for i, seat in enumerate(others):
            hands[seat] = three[i]
        last = (hands[0], hands[1], hands[2], hands[3])
        if all(_announced(hands[seat]) == (seat in announced) for seat in others):
            return last
    if last is None:
        raise ValueError("no deal fits the void announcements")
    return last


def playout(
    d: Decision,
    hands: tuple[int, int, int, int],
    action: int,
    seats: Sequence[str],
    card_play: str,
    seed: int,
    rules: RulesConfig,
) -> float:
    """The value of ``action`` on one deal, for the deciding seat."""
    rnd = Round.replay(d.round_no, hands, d.fixed_trump, d.carry, d.scores, rules, d.round_actions)
    rnd.step(action)
    deciders = [
        make_bot(HUMAN_STAND_IN if spec == HUMAN else spec, seed * 4 + seat, rules)
        for seat, spec in enumerate(seats)
    ]
    players = [make_bot(card_play, seed * 4 + seat, rules) for seat in range(4)]
    while rnd.phase is not Phase.DONE:
        seat = rnd.current_player
        if seat is None:
            raise RuntimeError("a round that is not done always has a player to act")
        bot = players[seat] if rnd.phase is Phase.PLAY else deciders[seat]
        rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
    scores = rnd.result().scores
    weights = standings_weights(d.scores, d.seat)
    return scores[d.seat] - sum(w * s for w, s in zip(weights, scores, strict=True))


def value_table(
    d: Decision,
    choices: Sequence[int],
    seats: Sequence[str],
    deals: int,
    card_play: str,
    seed: int,
    rules: RulesConfig,
) -> dict[int, tuple[float, float]]:
    """Each choice's mean value over ``deals`` deals, and its standard error."""
    values: dict[int, list[float]] = {choice: [] for choice in choices}
    for deal in range(deals):
        hands = redeal(d, random.Random(seed * 100_003 + deal))
        for choice in choices:
            values[choice].append(playout(d, hands, choice, seats, card_play, seed + deal, rules))
    table: dict[int, tuple[float, float]] = {}
    for choice, vs in values.items():
        mean = sum(vs) / len(vs)
        var = sum((v - mean) ** 2 for v in vs) / (len(vs) - 1) if len(vs) > 1 else 0.0
        table[choice] = (mean, math.sqrt(var / len(vs)))
    return table
