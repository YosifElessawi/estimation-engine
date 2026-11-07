"""The one net inside the Hybrid search: step 5's bots (docs/rl/hybrid-iteration-plan.md).

    onenet:net=<player.npz>:checkpoint=<hybrid.pt>:deals=both:top_k=2:samples=32
    onenet:net=<player.npz>:own=true:top_k=2:samples=32
    onenet:net=<player.npz>:own=true:judge=place:judge_net=<judge.npz>:top_k=2:samples=32

The player bids alone (its top legal move), keeps its ``top_k`` cards to search and, on each
imagined deal, plays all four seats to the end of the round. Two things are set per bot:

- **Who imagines the deals.** By default the Hybrid search does, from ``checkpoint``'s GuessNet
  and BiddingNet as ``deals`` says. With Hybrid v4's checkpoint and settings this is Hybrid v4
  with the one net in place of its card net and BiddingNet: the check-up bot. With ``own=true``
  no checkpoint is used: the player's own hidden-card guesses (practice question 1) place each
  deal's cards, and its own move chances at the other seats' bidding keep the deals that explain
  that bidding best, as ``deals=both`` does with GuessNet and BiddingNet. Every decision is then
  the player's.
- **How each searched card is judged** (``judge``). ``round``: the play-out's round points, as
  the Hybrid scores them. ``king``: the player's own chance of finishing King (practice question
  6), read from its view at its last card of the round, when twelve tricks are known (tried
  and dropped: it lost about 30 points a game). ``place``: the training judge's expected final
  place (King +1 to Kooz -1, ``judge_net``), read at the same moment from that view and the other
  three hands. The judge sees every hand, so it never plays; it can score an imagined ending
  because every hand of an imagined deal is known: it never sees a real hidden card.

This started as Essam Gouda's bitter-search (estimation-arena bots/bitter-search, commit e43b15b,
MIT licence, see estimation/jaxenv/LICENSE), which runs his net inside the Hybrid search. Changes
in this copy: it reads the view from estimation.jaxenv, takes the player's file as ``net``, drops
the option of playing out with estimation-arena's rl-cards net, and adds ``own`` and ``judge``.

The net reads the raw view (estimation.jaxenv.encode_np), not the engine's RL features, so the
play-outs build that view: a per-seat template of what stays fixed during card play, and each
step writes what changes. The bidding views work the same way: every other seat's view at each of
its bidding choices, encoded once per round with an empty hand, and each deal writes the hand that
seat was dealt. The tests check both against encode_np in real games.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from estimation.bots.hybrid.beliefs import (
    BID_ACTIONS,
    BiddingHistory,
    PublicBidding,
    choice_log_likelihoods,
)
from estimation.bots.hybrid.bot import BOTH, HybridBot, HybridConfig
from estimation.bots.hybrid.rollout import (
    Start,
    _beats,  # pyright: ignore[reportPrivateUsage]  # the Hybrid's own trick rule, shared
    _only_card,  # pyright: ignore[reportPrivateUsage]  # the Hybrid's own lowest-card pick, shared
    bits,
)
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import best_index
from estimation.bots.scoring import RoundScorer, estimates_of, standings_weights
from estimation.engine import Observation, Phase, RulesConfig, legal_plays, play_action
from estimation.engine.cards import SUIT_MASKS
from estimation.jaxenv.encode_np import PLAY_BASE, SIZE, encode

PLAYED, TABLE, SHOWN, WON, LEADER, MASK = 52, 260, 468, 580, 594, SIZE - 120
ROUND, KING, PLACE = "round", "king", "place"
JUDGES = (ROUND, KING, PLACE)
_ONE = np.uint64(1)
_SUITS = np.array(SUIT_MASKS, dtype=np.uint64)
Floats = npt.NDArray[np.float32]


class Player:
    """The player as numpy functions of raw views ``(B, SIZE)``: calling it gives its move logits
    ``(B, 120)``; ``holder_chances`` and ``king_chance`` read its practice answers, which need a
    file saved with them (``answers``)."""

    __slots__ = ("_layers", "_move", "_practice", "_slices")

    def __init__(self, data: Any) -> None:
        layers: list[tuple[Floats, Floats]] = []
        k = 0
        while f"w{k}" in data:
            layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
            k += 1
        self._layers = layers
        self._move = data["pw"].astype(np.float32), data["pb"].astype(np.float32)
        self._practice: tuple[Floats, Floats] | None = None
        self._slices: dict[str, slice] = {}
        if "qw" in data and "meta" in data:
            start = 0
            for name, size in json.loads(str(data["meta"])).get("practice", []):
                self._slices[name] = slice(start, start + size)
                start += size
            if {"holder", "king"} <= self._slices.keys():
                self._practice = data["qw"].astype(np.float32), data["qb"].astype(np.float32)

    @property
    def answers(self) -> bool:
        """Whether the file holds the practice answers this search reads (holders and King)."""
        return self._practice is not None

    def body(self, x: Floats) -> Floats:
        for w, b in self._layers:
            x = np.maximum(x @ w + b, 0.0)
        return x

    def __call__(self, x: Floats) -> Floats:
        w, b = self._move
        return self.body(x) @ w + b

    def _answer(self, x: Floats, name: str) -> Floats:
        if self._practice is None:
            raise ValueError("this player file has no practice answers")
        w, b = self._practice
        part = self._slices[name]
        return self.body(x) @ w[:, part] + b[part]

    def holder_chances(self, x: Floats) -> Floats:
        """``(B, 52, 3)``: for each card, the chance that each other seat (the next one first)
        holds it. Meaningful for the cards the acting seat has not seen."""
        logits = self._answer(x, "holder").reshape(-1, 52, 3)
        exp = np.exp(logits - logits.max(axis=2, keepdims=True))
        return (exp / exp.sum(axis=2, keepdims=True)).astype(np.float32)

    def king_chance(self, x: Floats) -> Floats:
        """``(B,)``: the acting seat's chance of finishing the Bola King."""
        z = self._answer(x, "king")[:, 0]
        return (0.5 * (np.tanh(0.5 * z) + 1.0)).astype(np.float32)  # the sigmoid, never overflowing


@cache
def load_net(path: str) -> Player:
    """The player saved at ``path``."""
    with np.load(path) as data:
        return Player(data)


class Judge:
    """The training judge (estimation.onenet.nets.apply_judge) in numpy: the acting seat's
    expected final place, King +1 to Kooz -1, from its raw view and the three other hands."""

    __slots__ = ("_layers", "_value")

    def __init__(self, data: Any) -> None:
        layers: list[tuple[Floats, Floats]] = []
        k = 0
        while f"w{k}" in data:
            layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
            k += 1
        if not layers or layers[0][0].shape[0] != SIZE + 3 * 52 or "vw" not in data:
            raise ValueError(
                "this file is not a judge (the raw view and three hands in, a value out)"
            )
        self._layers = layers
        self._value = data["vw"].astype(np.float32), data["vb"].astype(np.float32)

    def value(self, x: Floats, others: npt.NDArray[np.uint64]) -> Floats:
        """``(B,)`` from views ``(B, SIZE)`` and ``others`` ``(B, 3)``: the hands of the next three
        seats from the acting one, in that order."""
        h = np.concatenate([x, bits(others).reshape(x.shape[0], 3 * 52)], axis=1, dtype=np.float32)
        for w, b in self._layers:
            h = np.maximum(h @ w + b, 0.0)
        w, b = self._value
        return (h @ w + b)[:, 0]


@cache
def load_judge(path: str) -> Judge:
    """The judge saved at ``path`` (estimation.onenet.selfplay.export_judge)."""
    with np.load(path) as data:
        return Judge(data)


def templates_of(obs: Observation) -> Floats:
    """``(4, SIZE)``: each seat's raw view of ``obs``'s public state with an empty hand."""
    views = [encode(replace(obs, seat=seat, hand=0, current_player=seat), []) for seat in range(4)]
    return np.stack(views)


def bidding_views(obs: Observation, rules: RulesConfig) -> BiddingHistory:
    """This round's choices with more than one legal option by every seat but ``obs.seat``: who
    chose, that seat's raw view at the time with an empty hand (writing the hand it was dealt
    into the first 52 inputs completes it), the legal options and the choice. As
    estimation.bots.hybrid.beliefs.bidding_history does for BiddingNet's features."""
    me = obs.seat
    state = PublicBidding(obs, rules)
    seats: list[int] = []
    rows: list[Floats] = []
    legal_rows: list[npt.NDArray[np.bool_]] = []
    chosen: list[int] = []
    for seat, action in obs.log:
        if action >= PLAY_BASE:
            break
        if seat != me:
            legal = state.legal_actions()
            if len(legal) > 1:
                seats.append(seat)
                rows.append(encode(state.observation(seat), legal))
                mask = np.zeros(BID_ACTIONS, np.bool_)
                mask[legal] = True
                legal_rows.append(mask)
                chosen.append(action)
        state.step(seat, action)
    if not rows:
        return BiddingHistory(
            (),
            np.zeros((0, SIZE), np.float32),
            np.zeros((0, BID_ACTIONS), np.bool_),
            np.zeros(0, np.int64),
        )
    return BiddingHistory(
        tuple(seats), np.stack(rows), np.stack(legal_rows), np.asarray(chosen, np.int64)
    )


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


type EndValue = Callable[[Floats, npt.NDArray[np.uint64]], Floats]
"""A seat's value ``(B,)`` from its views ``(B, SIZE)`` and the next three hands ``(B, 3)``."""


class PlayOut(NamedTuple):
    """The end of each played-out game: tricks won ``(B, 4)``, and with ``judge_seat`` that
    seat's ``judge`` value at its last card of the round ``(B,)``."""

    won: npt.NDArray[np.int64]
    value: Floats | None


def play_out(
    net: Player,
    templates: Floats,
    start: Start,
    hands: npt.NDArray[np.uint64],
    played: npt.NDArray[np.uint64],
    voids: npt.NDArray[np.uint8],
    first_cards: npt.NDArray[np.int64],
    judge_seat: int | None = None,
    judge: EndValue | None = None,
) -> PlayOut:
    """Every game to the end of the round, every seat playing the net's top legal card. As
    estimation.bots.hybrid.rollout.play_out, on the raw view. ``hands``, ``played`` and ``voids``
    are used up: pass copies. With ``judge_seat``, ``judge`` values that seat at its last card of
    the round (its only legal play), from its view and the others' hands at that moment."""
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
    if (judge_seat is None) != (judge is None):
        raise ValueError("play_out takes judge_seat and judge together")
    value = None if judge_seat is None else np.zeros(games, np.float32)
    buf = np.empty((games, SIZE), np.float32)
    card: Any = first_cards.astype(np.int64)
    while True:
        acting = (leaders + tlen) & 3
        hand = hands[ar, acting]
        led = trick_cards[:, 0] // 13
        if value is not None and judge is not None and tricks_done == 12:
            rows = np.flatnonzero(acting == judge_seat)
            if rows.size:  # its last card: the one it holds is its only legal play
                sub = buf[: rows.size]
                write_view(
                    sub,
                    templates,
                    hand[rows],
                    hand[rows],
                    played[rows],
                    voids[rows],
                    acting[rows],
                    trick_cards[rows],
                    tlen,
                    leaders[rows],
                    won[rows],
                )
                others = np.stack([hands[rows, (acting[rows] + k) & 3] for k in (1, 2, 3)], 1)
                value[rows] = judge(sub, others)
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
                return PlayOut(won, value)


class OneNetSearch(HybridBot):
    """The Hybrid search with the one net choosing the cards to search and playing them out; with
    ``own`` it also imagines the deals, and ``judge`` says how each searched card is valued."""

    def __init__(
        self,
        seed: int,
        config: HybridConfig,
        net: str,
        *,
        own: bool = False,
        judge: str = ROUND,
        judge_net: str = "",
    ) -> None:
        if judge not in JUDGES:
            raise ValueError(f"judge must be {ROUND}, {KING} or {PLACE}, not {judge!r}")
        if (judge == PLACE) != bool(judge_net):
            raise ValueError(
                "judge=place takes judge_net=<the judge's .npz file>, and only it does"
            )
        if judge_net and not Path(judge_net).exists():
            raise ValueError(f"onenet: no judge file at {judge_net}")
        self._judge_net = load_judge(judge_net) if judge_net else None
        player = load_net(net)
        if (own or judge == KING) and not player.answers:
            raise ValueError(f"{net} has no practice answers, which own=true and judge=king read")
        super().__init__(seed, replace(config, deals=BOTH) if own else config, nets=not own)
        self._net = player
        self.own = own
        self.judge = judge
        self._views: BiddingHistory | None = None
        self._views_key: tuple[Any, ...] | None = None

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        logits = self._net(encode(obs, legal))
        if obs.phase is not Phase.PLAY:
            return max(legal, key=lambda a: logits[a])
        cards = [a - PLAY_BASE for a in legal]
        search = self.search(obs, cards, [float(logits[PLAY_BASE + c]) for c in cards])
        return play_action(search.cards[best_index(search.means)])

    def guess_chances(self, obs: Observation) -> list[list[float]]:
        """With ``own``, the player's chance of each other seat (the next one first) holding each
        card, from its view at this card decision, raised to ``1 / guess_temperature``."""
        if not self.own:
            return super().guess_chances(obs)
        if obs.phase is not Phase.PLAY:
            raise ValueError("the player's own guesses are read at card decisions")
        legal = [PLAY_BASE + card for card in legal_plays(obs.hand, obs.trick)]
        x = encode(obs, legal)[np.newaxis]
        chances = self._net.holder_chances(x)[0].astype(np.float64)
        if self.config.guess_temperature != 1.0:
            chances = chances ** (1.0 / self.config.guess_temperature)
        return chances.tolist()

    def bidding_log_likelihoods(
        self, obs: Observation, deals: list[list[int]], played_by: Sequence[int]
    ) -> npt.NDArray[np.float64] | None:
        """With ``own``, each deal's log likelihood of the other seats' bidding under the
        player's own move chances (its softmax, mixed with ``belief_floor`` as BiddingNet's is),
        or None when none of them had a choice."""
        if not self.own:
            return super().bidding_log_likelihoods(obs, deals, played_by)
        history = self._bidding_views(obs)
        if not len(history):
            return None
        seats = np.asarray(history.seats, np.int64)
        dealt = np.array(deals, dtype=np.uint64) | np.array(played_by, dtype=np.uint64)
        count, n = dealt.shape[0], len(history)
        x = np.repeat(history.templates[np.newaxis], count, axis=0)  # (D, n, SIZE)
        x[:, :, :52] = bits(dealt[:, seats])
        logits = self._net(x.reshape(count * n, SIZE)).reshape(count, n, -1)
        scores = logits[:, :, :BID_ACTIONS].astype(np.float64)
        return choice_log_likelihoods(scores, history, self.config.belief_floor)

    def _bidding_views(self, obs: Observation) -> BiddingHistory:
        """The other seats' bidding views this round, rebuilt once per round."""
        bidding = tuple(entry for entry in obs.log if entry[1] < PLAY_BASE)
        key = (obs.seat, obs.round_no, obs.scores, obs.carry, obs.void_announced, bidding)
        if self._views is None or self._views_key != key:
            self._views = bidding_views(obs, self.config.rules)
            self._views_key = key
        return self._views

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
        out = play_out(
            self._net,
            templates_of(obs),
            start,
            np.repeat(np.array(deals, dtype=np.uint64), n_cards, axis=0),
            np.tile(np.array(know.played_by, dtype=np.uint64), (games, 1)),
            np.tile(np.array(know.voids, dtype=np.uint8), (games, 1)),
            np.tile(np.array(cards, dtype=np.int64), n_deals),
            **self._end_judge(obs.seat),
        )
        totals = [0.0] * n_cards
        if out.value is not None:
            for g, value in enumerate(out.value.tolist()):
                totals[g % n_cards] += value
            return totals
        weights = standings_weights(obs.scores, obs.seat) if self.config.leader_weight else None
        scorer = RoundScorer(obs, self.config.rules, weights)
        for g, tricks in enumerate(out.won.tolist()):
            totals[g % n_cards] += scorer.value(tricks)
        return totals

    def _end_judge(self, seat: int) -> dict[str, Any]:
        """``play_out``'s judge arguments: none for round points."""
        if self.judge == KING:
            player = self._net
            return {"judge_seat": seat, "judge": lambda x, _others: player.king_chance(x)}
        if self.judge == PLACE and self._judge_net is not None:
            return {"judge_seat": seat, "judge": self._judge_net.value}
        return {}


def make(
    seed: int,
    config: HybridConfig,
    net: str,
    *,
    own: bool = False,
    judge: str = ROUND,
    judge_net: str = "",
) -> OneNetSearch:
    if not Path(net).exists():
        raise ValueError(f"onenet: no player file at {net}")
    return OneNetSearch(seed, config, net, own=own, judge=judge, judge_net=judge_net)
