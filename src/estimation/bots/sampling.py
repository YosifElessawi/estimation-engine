"""Dealing the unseen cards (SPEC §6 M3): random deals of the other three hands that fit their
hand sizes and voids.

Cards of suits no hand is void in are shuffled and cut to the hand sizes, which is exactly uniform.
Cards of suits some hand is void in go first, one at a time, to an allowed hand chosen in
proportion to its free places. That keeps every fitting deal possible and is exactly uniform when
at most one suit is limited by voids. With two or more it leans away from uniform: late in a
round, by 0.1 to 0.3 in total variation. A Hall-condition check before each placement keeps every
partial deal completable, so sampling never dead-ends.

``deal_by_guesses`` deals the same way, but places every card, one at a time, with the chances a
guess model gives each hand of holding it (the Hybrid's GuessNet), among the hands that keep a
fitting deal possible.
"""

import random
from collections.abc import Sequence

from estimation.engine.cards import SUIT_MASKS, cards_of

_ALL = 0b111  # the three hands, as a bit mask over local hand indices 0-2
_SUBSETS = tuple(range(1, 8))
# For a card allowed in hands ``allowed`` going to hand ``i``: the hand subsets whose slack it
# uses up (they contain i but cannot absorb the card's suit on their own).
_SPENDS = {
    (allowed, i): tuple(s for s in _SUBSETS if s >> i & 1 and allowed & ~s)
    for allowed in range(1, 8)
    for i in range(3)
    if allowed >> i & 1
}


def deal_unseen(
    unseen: int, sizes: Sequence[int], voids: Sequence[int], rng: random.Random
) -> list[int]:
    """Three hands holding exactly the ``unseen`` cards, of ``sizes`` cards each, where hand
    ``i`` gets no card of a suit in its 4-bit ``voids[i]`` mask."""
    if sum(sizes) != unseen.bit_count():
        raise ValueError("hand sizes must add up to the unseen cards")
    hands = [0, 0, 0]
    free = [sizes[0], sizes[1], sizes[2]]
    shuffled: list[int] = []
    constrained: list[tuple[int, int]] = []  # (allowed hands, cards)
    for suit in range(4):
        cards = unseen & SUIT_MASKS[suit]
        if not cards:
            continue
        allowed = sum(1 << i for i in range(3) if not voids[i] >> suit & 1)
        if allowed == _ALL:
            shuffled += cards_of(cards)
        elif allowed:
            constrained.append((allowed, cards))
        else:
            raise ValueError("every hand is void in a suit that still has unseen cards")
    if constrained:
        slack = _slack(free, constrained, len(shuffled))
        constrained.sort(key=lambda entry: entry[0].bit_count())  # most constrained first
        for allowed, cards in constrained:
            order = cards_of(cards)
            rng.shuffle(order)
            for card in order:
                i = _place(allowed, free, slack, rng)
                hands[i] |= 1 << card
                free[i] -= 1
                for s in _SPENDS[(allowed, i)]:
                    slack[s] -= 1
    rng.shuffle(shuffled)
    start = 0
    for i in range(3):
        for card in shuffled[start : start + free[i]]:
            hands[i] |= 1 << card
        start += free[i]
    return hands


def deal_by_guesses(
    unseen: int,
    sizes: Sequence[int],
    voids: Sequence[int],
    chances: Sequence[Sequence[float]],
    rng: random.Random,
) -> list[int]:
    """Three hands holding exactly the ``unseen`` cards, of ``sizes`` cards each, with no card of
    a suit in hand ``i``'s 4-bit ``voids[i]`` mask, each card placed in proportion to
    ``chances[card][i]`` among the hands that keep the rest of the deal possible (Hall), or in
    proportion to their free places when every such chance is 0. Cards of suits some hand is void
    in go first, most limited first, then the rest in a random order."""
    if sum(sizes) != unseen.bit_count():
        raise ValueError("hand sizes must add up to the unseen cards")
    hands = [0, 0, 0]
    free = [sizes[0], sizes[1], sizes[2]]
    open_cards: list[int] = []
    constrained: list[tuple[int, int]] = []
    for suit in range(4):
        cards = unseen & SUIT_MASKS[suit]
        if not cards:
            continue
        allowed = sum(1 << i for i in range(3) if not voids[i] >> suit & 1)
        if allowed == _ALL:
            open_cards += cards_of(cards)
        elif allowed:
            constrained.append((allowed, cards))
        else:
            raise ValueError("every hand is void in a suit that still has unseen cards")
    slack = _slack(free, constrained, len(open_cards))
    constrained.sort(key=lambda entry: entry[0].bit_count())  # most constrained first
    order: list[tuple[int, int]] = []
    for allowed, cards in constrained:
        group = cards_of(cards)
        rng.shuffle(group)
        order += [(allowed, card) for card in group]
    rng.shuffle(open_cards)
    order += [(_ALL, card) for card in open_cards]
    for allowed, card in order:
        options = [
            i
            for i in range(3)
            if allowed >> i & 1 and free[i] and all(slack[s] > 0 for s in _SPENDS[(allowed, i)])
        ]
        if not options:
            raise ValueError("no deal fits these hand sizes and voids")
        weights = [max(0.0, float(chances[card][i])) for i in options]
        if not sum(weights):
            weights = [float(free[i]) for i in options]
        pick = rng.random() * sum(weights)
        hand = options[-1]
        for i, weight in zip(options, weights, strict=True):
            pick -= weight
            if pick < 0:
                hand = i
                break
        hands[hand] |= 1 << card
        free[hand] -= 1
        for s in _SPENDS[(allowed, hand)]:
            slack[s] -= 1
    return hands


def _slack(free: list[int], constrained: list[tuple[int, int]], unconstrained: int) -> list[int]:
    """Per subset of hands: their free places minus the cards only they can take (Hall)."""
    slack = [0] * 8
    for s in _SUBSETS:
        places = sum(free[i] for i in range(3) if s >> i & 1)
        needed = sum(cards.bit_count() for allowed, cards in constrained if not allowed & ~s)
        if s == _ALL:
            needed += unconstrained
        slack[s] = places - needed
        if slack[s] < 0:
            raise ValueError("no deal fits these hand sizes and voids")
    return slack


def _place(allowed: int, free: list[int], slack: list[int], rng: random.Random) -> int:
    """A hand for one card, chosen in proportion to free places among those keeping a full deal
    possible."""
    options = [
        i
        for i in range(3)
        if allowed >> i & 1 and free[i] and all(slack[s] > 0 for s in _SPENDS[(allowed, i)])
    ]
    total = sum(free[i] for i in options)
    if not total:
        raise ValueError("no deal fits these hand sizes and voids")
    pick = rng.random() * total
    for i in options:
        pick -= free[i]
        if pick < 0:
            return i
    return options[-1]
