"""The exact endgame solver against a plain recursive player on the engine's own trick rules."""

import random
from collections.abc import Sequence

import numpy as np
import pytest

from estimation.bots.scoring import RoundScorer, standings_weights
from estimation.engine import Bola, Phase
from estimation.engine.cards import SANS, legal_plays, parse_card, parse_hand, trick_winner
from estimation.engine.config import DEFAULT_CONFIG
from estimation.onenet import endgame as E

Trick = list[tuple[int, int]]


def targets_worths(targets: Sequence[int]) -> E.Worths:
    """A round where each seat wants exactly its target of tricks, counted against the others."""

    def worths(tricks: E.Tricks) -> list[float]:
        scores = [
            10.0 + 2 * want if got == want else -3.0 * abs(got - want)
            for got, want in zip(tricks, targets, strict=True)
        ]
        return [scores[seat] - (sum(scores) - scores[seat]) / 3 for seat in range(4)]

    return worths


def by_hand(
    hands: Sequence[int],
    trick: Trick,
    leader: int,
    won: Sequence[int],
    trump: int,
    worths: E.Worths,
) -> tuple[list[int], np.ndarray]:
    """The solver's answer the slow way: every line by recursion on ``legal_plays`` and
    ``trick_winner``, each seat playing its best, the mean over the cards it is level on."""

    def after(hands: list[int], trick: Trick, leader: int, won: list[int], card: int) -> np.ndarray:
        seat = (leader + len(trick)) % 4
        hands = list(hands)
        hands[seat] ^= 1 << card
        trick = [*trick, (seat, card)]
        if len(trick) < 4:
            return turn(hands, trick, leader, won)
        winner = trick_winner(trick, trump)
        won = list(won)
        won[winner] += 1
        return turn(hands, [], winner, won)

    def turn(hands: list[int], trick: Trick, leader: int, won: list[int]) -> np.ndarray:
        if not trick and not any(hands):
            return np.array(worths((won[0], won[1], won[2], won[3])), np.float64)
        seat = (leader + len(trick)) % 4
        ends = [after(hands, trick, leader, won, c) for c in legal_plays(hands[seat], trick)]
        best = max(end[seat] for end in ends)
        return np.mean([end for end in ends if end[seat] >= best - E.TIE], axis=0)

    cards = legal_plays(hands[(leader + len(trick)) % 4], trick)
    return cards, np.array([after(list(hands), trick, leader, list(won), c) for c in cards])


def a_spot(rng: random.Random, size: int, played: int) -> tuple[list[int], Trick, int, list[int]]:
    """A random spot with ``size`` cards in each hand and ``played`` cards of the next trick
    already on the table (those seats hold one card fewer), and tricks won that total
    13 less the tricks left."""
    cards = rng.sample(range(52), 4 * size)
    hands = [sum(1 << c for c in cards[seat * size : (seat + 1) * size]) for seat in range(4)]
    leader = rng.randrange(4)
    trick: Trick = []
    for place in range(played):
        seat = (leader + place) % 4
        card = rng.choice(legal_plays(hands[seat], trick))
        hands[seat] ^= 1 << card
        trick.append((seat, card))
    won = [0, 0, 0, 0]
    for _ in range(13 - size):
        won[rng.randrange(4)] += 1
    return hands, trick, leader, won


@pytest.mark.parametrize(
    ("size", "played", "spots"),
    [(1, 0, 6), (1, 2, 6), (2, 0, 25), (2, 1, 25), (2, 3, 25), (3, 0, 12), (3, 2, 12), (4, 1, 2)],
)
def test_solve_is_the_recursive_player(size: int, played: int, spots: int) -> None:
    rng = random.Random(100 * size + played)
    for _ in range(spots):
        hands, trick, leader, won = a_spot(rng, size, played)
        trump = rng.randrange(5)
        worths = targets_worths([rng.randrange(size + 2) + w for w in won])
        cards, values = by_hand(hands, trick, leader, won, trump, worths)
        solved = E.solve([hands], trick, leader, won, trump, worths)
        assert solved.cards == cards
        np.testing.assert_allclose(solved.worths[0], values, atol=1e-9)


def test_solve_plays_four_cards_from_the_lead() -> None:
    rng = random.Random(7)
    hands, trick, leader, won = a_spot(rng, 4, 0)
    worths = targets_worths([w + 1 for w in won])
    cards, values = by_hand(hands, trick, leader, won, 2, worths)
    solved = E.solve([hands], trick, leader, won, 2, worths)
    assert solved.cards == cards
    np.testing.assert_allclose(solved.worths[0], values, atol=1e-9)


def test_solve_plays_every_deal_of_a_batch_as_it_plays_it_alone() -> None:
    rng = random.Random(3)
    hands, trick, leader, won = a_spot(rng, 3, 1)
    mover = (leader + 1) % 4
    others = [seat for seat in range(4) if seat != mover]
    pool = [c for seat in others for c in legal_plays(hands[seat], [])]
    deals = [list(hands)]
    for _ in range(10):  # the other seats' cards dealt again, the seat to play keeps its hand
        rng.shuffle(pool)
        deal, at = list(hands), 0
        for seat in others:
            count = hands[seat].bit_count()
            deal[seat] = sum(1 << c for c in pool[at : at + count])
            at += count
        deals.append(deal)
    worths = targets_worths([w + 1 for w in won])
    together = E.solve(deals, trick, leader, won, SANS, worths, chunk=4)
    assert together.worths.shape == (11, len(together.cards), 4)
    for deal, values in zip(deals, together.worths, strict=True):
        alone = E.solve([deal], trick, leader, won, SANS, worths)
        assert alone.cards == together.cards
        np.testing.assert_allclose(alone.worths[0], values, atol=1e-9)


def test_solve_counts_out_an_exact_estimate() -> None:
    """Seat 0 needs exactly one more trick. The ace of spades first takes it, and seat 1, who
    wants no more, throws its 9 of hearts on it so the last heart trick goes to seat 2. The low
    heart first gives seat 1 the lead and both tricks: the ace is thrown on its diamond."""
    hands = [parse_hand("AS 2H"), parse_hand("9H 5D"), parse_hand("QS 8H"), parse_hand("JS 7H")]
    worths = targets_worths([5, 4, 3, 2])
    solved = E.solve([hands], [], 0, [4, 4, 2, 1], SANS, worths)
    assert solved.cards == [parse_card("2H"), parse_card("AS")]
    low, ace = solved.worths[0, :, 0]
    assert ace > low
    assert ace == pytest.approx(worths((5, 4, 3, 1))[0])
    assert low == pytest.approx(worths((4, 6, 2, 1))[0])


def test_solve_follows_suit_and_trumps() -> None:
    """Seat 1 must follow the club lead with its only club. Seat 2 has none and may trump, but
    it needs exactly one more trick: trumping now wins this trick and the next with its ace, so
    it throws the ace and trumps the last trick."""
    hands = [parse_hand("3D"), parse_hand("5C 2S"), parse_hand("2H AD"), parse_hand("4C KD")]
    trick = [(0, parse_card("9C"))]
    solved = E.solve([hands], trick, 0, [3, 3, 3, 2], 2, targets_worths([3, 3, 4, 3]))
    assert solved.cards == [parse_card("5C")]
    hands = [parse_hand("3D"), parse_hand("2S"), parse_hand("2H AD"), parse_hand("4C KD")]
    trick = [(0, parse_card("9C")), (1, parse_card("5C"))]
    worths = targets_worths([3, 3, 4, 3])
    solved = E.solve([hands], trick, 0, [3, 3, 3, 2], 2, worths)
    assert solved.cards == [parse_card("AD"), parse_card("2H")]
    thrown, trumped = solved.worths[0, :, 2]
    assert thrown == pytest.approx(worths((4, 3, 4, 2))[2])
    assert trumped == pytest.approx(worths((3, 3, 5, 2))[2])
    assert thrown > trumped


def test_solve_refuses_what_it_cannot_play() -> None:
    rng = random.Random(1)
    worths = targets_worths([3, 3, 3, 3])
    hands, trick, leader, won = a_spot(rng, 5, 0)
    with pytest.raises(ValueError, match="1 to 4 cards"):
        E.solve([hands], trick, leader, won, 0, worths)
    hands, trick, leader, won = a_spot(rng, 2, 0)
    short = list(hands)
    short[(leader + 1) % 4] &= short[(leader + 1) % 4] - 1
    with pytest.raises(ValueError, match="every deal must hold"):
        E.solve([short], trick, leader, won, 0, worths)
    other = list(hands)
    other[leader], other[(leader + 1) % 4] = other[(leader + 1) % 4], other[leader]
    with pytest.raises(ValueError, match="same cards in every deal"):
        E.solve([hands, other], trick, leader, won, 0, worths)
    with pytest.raises(ValueError, match="play order"):
        E.solve([hands], [((leader + 2) % 4, 0)], leader, won, 0, worths)


def test_worths_of_is_the_search_value_for_each_seat() -> None:
    bola = Bola(11, DEFAULT_CONFIG)
    rng = random.Random(11)
    while bola.phase is not Phase.PLAY or len(bola.history) < 2:
        bola.step(rng.choice(bola.legal_actions()))
    for seat in range(4):
        obs = bola.observation(seat)
        weighted = RoundScorer(obs, DEFAULT_CONFIG, standings_weights(obs.scores, seat))
        plain = RoundScorer(obs, DEFAULT_CONFIG)
        for tricks in [(13, 0, 0, 0), (3, 4, 3, 3), (0, 5, 6, 2), (2, 2, 2, 7)]:
            assert E.worths_of(obs)(tricks)[seat] == pytest.approx(weighted.value(tricks))
            assert E.worths_of(obs, leader_weight=False)(tricks)[seat] == pytest.approx(
                plain.value(tricks)
            )
