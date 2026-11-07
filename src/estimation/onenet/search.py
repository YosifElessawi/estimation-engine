"""The one net inside the Hybrid search: step 5's check-up bot (docs/rl/hybrid-iteration-plan.md).

    onenet:net=runs/onenet/selfplay/snapshots/player-00500.npz:checkpoint=<hybrid.pt>:deals=both

The player bids alone (its top legal move), keeps its ``top_k`` cards to search and, on each
imagined deal, plays all four seats to the end of the round; the Hybrid search supplies the
deals (from ``checkpoint``'s GuessNet and BiddingNet, per ``deals``) and scores each card by its
round points. With Hybrid v4's checkpoint and settings, this is Hybrid v4 with the one net in
place of its card net and BiddingNet: what a check-up plays against Hybrid v4.

This is Essam Gouda's bitter-search (estimation-arena bots/bitter-search, commit e43b15b, MIT
licence, see estimation/jaxenv/LICENSE), which runs his net inside the Hybrid search. Changes
in this copy: it reads the view from estimation.jaxenv, takes the player's file as ``net``, and
drops the option of playing out with estimation-arena's rl-cards net.

The net reads the raw view (estimation.jaxenv.encode_np), not the engine's RL features, so the
play-outs build that view: a per-seat template of what stays fixed during card play, and each
step writes what changes. The tests check it against encode_np at every decision of real games.
"""

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.hybrid.rollout import (
    Start,
    _beats,  # pyright: ignore[reportPrivateUsage]  # the Hybrid's own trick rule, shared
    _only_card,  # pyright: ignore[reportPrivateUsage]  # the Hybrid's own lowest-card pick, shared
    bits,
)
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import best_index
from estimation.bots.scoring import RoundScorer, estimates_of, standings_weights
from estimation.engine import Observation, Phase, play_action
from estimation.engine.cards import SUIT_MASKS
from estimation.jaxenv.encode_np import PLAY_BASE, SIZE, encode

PLAYED, TABLE, SHOWN, WON, LEADER, MASK = 52, 260, 468, 580, 594, SIZE - 120
_ONE = np.uint64(1)
_SUITS = np.array(SUIT_MASKS, dtype=np.uint64)
Floats = npt.NDArray[np.float32]


@cache
def load_net(path: str) -> Any:
    """The player's move output as a numpy function of the raw view: ``(B, SIZE) -> (B, 120)``."""
    data = np.load(path)
    layers, k = [], 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1
    head = data["pw"].astype(np.float32), data["pb"].astype(np.float32)

    def logits(x: Floats) -> Floats:
        for w, b in layers:
            x = np.maximum(x @ w + b, 0.0)
        return x @ head[0] + head[1]

    return logits


def templates_of(obs: Observation) -> Floats:
    """``(4, SIZE)``: each seat's raw view of ``obs``'s public state with an empty hand."""
    views = [encode(replace(obs, seat=seat, hand=0, current_player=seat), []) for seat in range(4)]
    return np.stack(views)


def write_view(x, templates, hand, legal, played, voids, acting, trick_cards, tlen, leaders, won):
    """Write each game's acting seat's view into ``x``, exactly as jaxenv.encode_np would."""
    games = x.shape[0]
    ar = np.arange(games)
    np.take(templates, acting, axis=0, out=x)
    x[:, :52] = bits(hand)
    x[:, TABLE:SHOWN] = 0.0
    for i in range(tlen):
        rel = (leaders + i - acting) & 3
        x[ar, TABLE + 52 * rel + trick_cards[:, i]] = 1.0
    for r in range(4):
        seat = (acting + r) & 3
        x[:, PLAYED + 52 * r : PLAYED + 52 * (r + 1)] = bits(played[ar, seat])
        for suit in range(4):
            x[:, SHOWN + 4 * r + suit] = (voids[ar, seat] >> suit) & 1
        x[:, WON + r] = won[ar, seat] / 13
    x[:, LEADER : LEADER + 5] = 0.0
    x[ar, LEADER + ((leaders - acting) & 3)] = 1.0
    x[:, MASK : MASK + PLAY_BASE] = 0.0
    x[:, MASK + PLAY_BASE :] = bits(legal)


def play_out(net, templates, start: Start, hands, played, voids, first_cards) -> np.ndarray:
    """Every game to the end of the round, every seat playing the net's top legal card; each
    game's tricks won, ``(B, 4)``. As estimation.bots.hybrid.rollout.play_out, on the raw view.
    ``hands``, ``played`` and ``voids`` are used up: pass copies."""
    games = hands.shape[0]
    ar = np.arange(games, dtype=np.int64)
    trump, tlen, tricks_done = start.trump, len(start.trick), start.tricks_done
    trick_cards = np.zeros((games, 4), np.int64)
    best_card, best_at = -1, 0
    for i, c in enumerate(start.trick):
        trick_cards[:, i] = c
        if i == 0 or _beats(c, best_card, trump):
            best_card, best_at = c, i
    best = np.full(games, best_card, np.int64)
    best_pos = np.full(games, best_at, np.int64)
    leaders = np.full(games, start.leader, np.int64)
    won = np.tile(np.asarray(start.won, np.int64), (games, 1))
    buf = np.empty((games, SIZE), np.float32)
    card = first_cards.astype(np.int64)
    while True:
        acting = (leaders + tlen) & 3
        hand = hands[ar, acting]
        led = trick_cards[:, 0] // 13
        if card is None:
            if tlen:
                follow = hand & _SUITS[led]
                legal = np.where(follow != 0, follow, hand)
            else:
                legal = hand
            card = _only_card(legal)
            rows = np.flatnonzero((legal & (legal - _ONE)) != 0)
            if rows.size:
                sub = buf[: rows.size]
                write_view(
                    sub,
                    templates,
                    hand[rows],
                    legal[rows],
                    played[rows],
                    voids[rows],
                    acting[rows],
                    trick_cards[rows],
                    tlen,
                    leaders[rows],
                    won[rows],
                )
                values = net(sub)[:, PLAY_BASE:]
                mask = bits(legal[rows]).astype(bool)
                card[rows] = np.where(mask, values, -np.inf).argmax(axis=1)
        bit = np.left_shift(_ONE, card.astype(np.uint64))
        hands[ar, acting] = hand ^ bit
        played[ar, acting] |= bit
        if tlen:
            off = card // 13 != led
            voids[ar[off], acting[off]] |= np.left_shift(1, led[off]).astype(np.uint8)
            suit, best_suit = card // 13, best // 13
            beats = ((suit == best_suit) & (card > best)) | ((suit != best_suit) & (suit == trump))
            best = np.where(beats, card, best)
            best_pos = np.where(beats, tlen, best_pos)
        else:
            best = card.copy()
            best_pos = np.zeros(games, np.int64)
        trick_cards[:, tlen] = card
        card = None
        tlen += 1
        if tlen == 4:
            winners = (leaders + best_pos) & 3
            won[ar, winners] += 1
            leaders = winners
            tlen = 0
            tricks_done += 1
            if tricks_done == 13:
                return won


class OneNetSearch(HybridBot):
    """The Hybrid search with the one net choosing the cards to search and playing them out."""

    def __init__(self, seed: int, config: HybridConfig, net: str) -> None:
        super().__init__(seed, config)
        self._net = load_net(net)

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        logits = self._net(encode(obs, legal))
        if obs.phase is not Phase.PLAY:
            return max(legal, key=lambda a: logits[a])
        cards = [a - PLAY_BASE for a in legal]
        search = self.search(obs, cards, [float(logits[PLAY_BASE + c]) for c in cards])
        return play_action(search.cards[best_index(search.means)])

    def evaluate(self, obs: Observation, cards: list[int], deals: list[list[int]]) -> list[float]:
        if obs.leader is None or obs.trump is None:
            raise ValueError("the one net's search runs only during card play")
        know = read_knowledge(obs)
        n_cards, n_deals = len(cards), len(deals)
        games = n_cards * n_deals  # game g plays deal g // n_cards with card g % n_cards
        start = Start(
            trick=[card for _, card in obs.trick],
            leader=obs.leader,
            won=obs.tricks_won,
            tricks_done=len(obs.tricks),
            estimates=estimates_of(obs),
            trump=obs.trump,
        )
        won = play_out(
            self._net,
            templates_of(obs),
            start,
            np.repeat(np.array(deals, dtype=np.uint64), n_cards, axis=0),
            np.tile(np.array(know.played_by, dtype=np.uint64), (games, 1)),
            np.tile(np.array(know.voids, dtype=np.uint8), (games, 1)),
            np.tile(np.array(cards, dtype=np.int64), n_deals),
        )
        weights = standings_weights(obs.scores, obs.seat) if self.config.leader_weight else None
        scorer = RoundScorer(obs, self.config.rules, weights)
        totals = [0.0] * n_cards
        for g, tricks in enumerate(won.tolist()):
            totals[g % n_cards] += scorer.value(tricks)
        return totals


def make(seed: int, config: HybridConfig, net: str) -> OneNetSearch:
    if not Path(net).exists():
        raise ValueError(f"onenet: no player file at {net}")
    return OneNetSearch(seed, config, net)
