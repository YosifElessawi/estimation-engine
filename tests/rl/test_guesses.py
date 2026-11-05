"""Training the guess head: labels from the real deal, the loss and the blind baseline."""

import numpy as np
import pytest
import torch

from estimation.bots import HeuristicBot
from estimation.engine import Bola, Phase
from estimation.engine.cards import cards_of
from estimation.rl.encoder import NUM_FEATURES, VOIDS, encode
from estimation.rl.guesses import NO_GUESS, UNSEEN, blind_right, guess_labels, guess_loss


def test_labels_name_the_real_holder_of_each_unseen_card_from_the_acting_seat():
    bola = Bola(3)
    bot = HeuristicBot()
    checked = 0
    while not bola.is_over and checked < 300:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        features = encode(obs)
        labels = guess_labels(features, bola.round.hands, seat)
        remaining = bola.round._hands  # pyright: ignore[reportPrivateUsage]  # the real remaining hands, to check against
        for r in (1, 2, 3):
            held = cards_of(remaining[(seat + r) & 3])
            assert all(labels[card] == r - 1 for card in held)
        assert int((labels != NO_GUESS).sum()) == int(features[UNSEEN : UNSEEN + 52].sum())
        assert all(labels[card] == NO_GUESS for card in cards_of(obs.hand))
        if obs.phase is Phase.PLAY:
            checked += 1
        bola.step(bot.act(obs, bola.legal_actions()))
    assert checked == 300


def test_the_loss_counts_only_labelled_cards():
    labels = torch.full((2, 52), NO_GUESS, dtype=torch.int8)
    labels[0, :3] = torch.tensor([0, 1, 2], dtype=torch.int8)
    logits = torch.zeros(2, 52, 3)
    logits[0, 0, 0] = 10.0  # right
    logits[0, 1, 0] = 10.0  # wrong: seat 1 holds it
    logits[1, :, 2] = 10.0  # unlabelled, so it never counts
    loss, right, count = guess_loss(logits, labels)
    assert int(count) == 3
    assert int(right) == 1  # card 1 is wrong, and card 2's even logits pick seat 0: wrong
    even = float(torch.log(torch.tensor(3.0)))
    assert float(loss) == pytest.approx((0.0 + 10.0 + even) / 3, abs=1e-3)


def test_a_batch_without_unseen_cards_has_no_loss():
    labels = torch.full((1, 52), NO_GUESS, dtype=torch.int8)
    loss, right, count = guess_loss(torch.zeros(1, 52, 3, requires_grad=True), labels)
    assert float(loss.detach()) == 0.0
    assert int(right) == 0
    assert int(count) == 0
    loss.backward()  # still part of the graph, so a training step can add it


def test_a_blind_guess_splits_evenly_among_the_seats_not_shown_out():
    x = torch.zeros(1, NUM_FEATURES)
    x[0, VOIDS + 4 * 1 + 2] = 1.0  # the second seat after the acting one showed out of Hearts
    labels = torch.full((1, 52), NO_GUESS, dtype=torch.int8)
    labels[0, 0] = 0  # a Club held by the next seat: three possible holders
    labels[0, 26] = 0  # a Heart held by the next seat: two possible holders
    assert float(blind_right(x, labels)) == pytest.approx(1 / 3 + 1 / 2)


def test_in_bidding_every_card_of_the_other_hands_is_labelled():
    bola = Bola(8)
    bot = HeuristicBot()
    while bola.phase is not Phase.PLAY:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        labels = guess_labels(encode(obs), bola.round.hands, seat)
        assert np.bincount(labels[labels != NO_GUESS]).tolist() == [13, 13, 13]
        bola.step(bot.act(obs, bola.legal_actions()))
