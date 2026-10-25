"""Suit symmetry of card play (SPEC §6 M4): relabelling suits relabels the features exactly."""

import dataclasses

import numpy as np
import pytest
import torch

from estimation.bots import HeuristicBot
from estimation.engine import (
    Bola,
    Observation,
    Phase,
    TrickRecord,
    WithInfo,
    bid_action,
    bid_of,
)
from estimation.engine.actions import PLAY_BASE, is_bid
from estimation.rl.encoder import encode
from estimation.rl.symmetry import (
    SUIT_PERMUTATIONS,
    SuitAugmenter,
    card_destination,
    card_source,
    feature_source,
)


def relabel(obs: Observation, perm: tuple[int, ...]) -> Observation:
    """``obs`` with every suit ``s`` renamed ``perm[s]`` (Sans stays Sans)."""

    def card(c: int) -> int:
        return perm[c // 13] * 13 + c % 13

    def renamed(t: int) -> int:
        return t if t == 4 else perm[t]

    def trump(t: int | None) -> int | None:
        return None if t is None else renamed(t)

    def action(a: int) -> int:
        if a >= PLAY_BASE:
            return PLAY_BASE + card(a - PLAY_BASE)
        if is_bid(a):
            tricks, t = bid_of(a)
            return bid_action(tricks, renamed(t))
        return a

    hand = 0
    for c in range(52):
        if obs.hand >> c & 1:
            hand |= 1 << card(c)
    return dataclasses.replace(
        obs,
        hand=hand,
        log=tuple((seat, action(a)) for seat, a in obs.log),
        top_bid=None if obs.top_bid is None else action(obs.top_bid),
        with_players=tuple(WithInfo(w.seat, w.w, renamed(w.trump)) for w in obs.with_players),
        trump=trump(obs.trump),
        fixed_trump=trump(obs.fixed_trump),
        fixed_left=tuple(renamed(t) for t in obs.fixed_left),
        trick=tuple((seat, card(c)) for seat, c in obs.trick),
        tricks=tuple(
            TrickRecord(
                r.leader,
                (card(r.cards[0]), card(r.cards[1]), card(r.cards[2]), card(r.cards[3])),
                r.winner,
            )
            for r in obs.tricks
        ),
    )


def card_positions(seed: int, count: int) -> list[Observation]:
    bola = Bola(seed)
    bot = HeuristicBot()
    found: list[Observation] = []
    while not bola.is_over and len(found) < count:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        if obs.phase is Phase.PLAY:
            found.append(obs)
        bola.step(bot.act(obs, bola.legal_actions()))
    return found


def test_there_are_24_suit_relabellings():
    assert len(SUIT_PERMUTATIONS) == 24
    assert len(set(SUIT_PERMUTATIONS)) == 24


@pytest.mark.parametrize("perm", [(1, 0, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (0, 1, 2, 3)])
def test_relabelling_a_position_permutes_its_features(perm: tuple[int, ...]):
    source = feature_source(perm)
    positions = card_positions(4, 200) + card_positions(19, 200)
    for obs in positions:
        assert np.array_equal(encode(relabel(obs, perm)), encode(obs)[source])


def test_card_maps_are_inverse():
    for perm in SUIT_PERMUTATIONS:
        assert np.array_equal(card_source(perm)[card_destination(perm)], np.arange(52))


def test_the_augmenter_relabels_features_cards_and_card_rows_together():
    augmenter = SuitAugmenter(torch.device("cpu"))
    obs = card_positions(8, 30)[-1]
    x = torch.as_tensor(encode(obs))[None].repeat(24, 1)
    perms = torch.arange(24)
    relabelled = augmenter.relabel_features(x, perms)
    hand = [c for c in range(52) if obs.hand >> c & 1]
    legal = torch.zeros(24, 52, dtype=torch.bool)
    legal[:, hand] = True
    cards = torch.full((24,), hand[0])
    new_legal = augmenter.relabel_card_rows(legal, perms)
    new_cards = augmenter.relabel_cards(cards, perms)
    for k, perm in enumerate(SUIT_PERMUTATIONS):
        expected = torch.as_tensor(encode(relabel(obs, perm)))
        assert torch.equal(relabelled[k], expected)
        assert set(torch.nonzero(new_legal[k]).flatten().tolist()) == {
            perm[c // 13] * 13 + c % 13 for c in hand
        }
        assert int(new_cards[k]) == perm[hand[0] // 13] * 13 + hand[0] % 13
