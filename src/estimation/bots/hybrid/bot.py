"""HybridBot (SPEC §6 M5): the Search bot's deal sampling, with the RL nets in place of rules.

- **Bidding:** Dash Calls, bids, With, Pass, estimates and super calls take the legal action with
  the highest BiddingNet value, exactly as the RL bot plays them. With ``bid_search``, BiddingNet's
  ``bid_top_k`` favourites are searched instead on ``bid_samples`` imagined deals, each finished
  by the nets and scored (``bid_search``), and the best mean is played.
- **Which deals to imagine** (``deals``): at a card decision with more than one choice it deals
  the unseen cards so they fit the hand sizes, voids, played cards and void announcements.
  ``bidding`` (the M5 Hybrid) draws ``samples * oversample`` deals at random, as PIMC does, and
  keeps ``samples`` of them by weighted sampling without replacement, weighted by how well each
  deal explains the other seats' bidding under BiddingNet (``beliefs``). ``guesses`` places each
  card where the checkpoint's GuessNet thinks it is (``guess_temperature`` flattens or sharpens
  its chances), and ``both`` draws ``samples * oversample`` such deals and keeps them by the
  bidding as ``bidding`` does.
- **Judging each card:** for each legal card (or the net's ``top_k`` favourites) and each kept
  deal, it plays the card and then the rest of the round with CardPlayNet's greedy card play for
  all four seats (``rollout``), and scores the result by SPEC §3 like the Search bot: own round
  score minus the other three's, players ahead in the game counting double. It plays the card with
  the best mean; on a tie, CardPlayNet's own favourite.

It reads only its own Observation: the deals it searches are its own samples. Search is
deterministic for a given seed.
"""

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from estimation.engine import Observation, Phase, RulesConfig, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG
from estimation.rl.encoder import encode
from estimation.rl.networks import REWARD_SCALE, Policy

from ..knowledge import read_knowledge
from ..pimc import MAX_REDEALS, best_index, fits_announcements
from ..rl import policy_at
from ..sampling import deal_by_guesses, deal_unseen
from ..scoring import RoundScorer, estimates_of, standings_weights
from .beliefs import BiddingHistory, bidding_history, log_likelihoods
from .bid_search import bid_totals
from .rollout import Start, play_out, seat_templates, shapes_of

RL_V2_CHECKPOINT = "runs/m4-rules-v2/checkpoints/final.pt"
BIDDING, GUESSES, BOTH = "bidding", "guesses", "both"


@dataclass(frozen=True, slots=True)
class HybridConfig:
    checkpoint: str = RL_V2_CHECKPOINT  # the RL checkpoint whose nets it uses (RL v2)
    samples: int = 16  # deals played out per card decision
    oversample: int = 2  # deals drawn per deal kept, to keep those that fit the bidding
    beliefs: bool = True  # weight the drawn deals by BiddingNet; off keeps the first ones drawn
    belief_temperature: float = 0.1  # BiddingNet softmax temperature, in units of 50 points
    belief_floor: float = 0.1  # share of each choice's probability spread over its options
    top_k: int = 3  # cards searched: CardPlayNet's k favourites, 0 for every candidate card
    deals: str = BIDDING  # how deals are imagined: BIDDING, GUESSES or BOTH (see the module)
    bid_search: bool = False  # search BiddingNet's favourite bids instead of taking its first
    bid_top_k: int = 3  # bids searched
    bid_samples: int = 16  # deals per searched bid decision
    guess_temperature: float = 1.0  # GuessNet's chances are raised to 1 / this before dealing
    merge_equivalent: bool = True  # search one card per run of touching equivalent cards
    leader_weight: bool = True  # count players ahead in the game double (standings_weights)
    threads: int = 1  # PyTorch threads: the batches are small, one thread avoids contention
    rules: RulesConfig = DEFAULT_CONFIG  # the rules of the game: stakes and scoring


@dataclass(frozen=True, slots=True)
class BidSearch:
    """One bid search: the actions searched (BiddingNet's favourite first), each one's mean value
    over the deals in points, BiddingNet's value of each in points, and the number of deals."""

    actions: list[int]
    means: list[float]
    net_values: list[float]
    deals: int
    seconds: float


@dataclass(frozen=True, slots=True)
class CardSearch:
    """One card search: the cards searched (CardPlayNet's favourite first), each one's mean
    value over the deals in points, CardPlayNet's predicted value of each in points, and the
    number of deals."""

    cards: list[int]
    means: list[float]
    net_values: list[float]
    deals: int
    seconds: float


@cache
def _set_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


class HybridBot:
    """RL bidding; card play by sampled deals weighted by BiddingNet and played out by
    CardPlayNet. Reads only its own Observation."""

    __slots__ = (
        "_history",
        "_history_key",
        "_policy",
        "_rng",
        "config",
        "last_bid_search",
        "last_search",
        "seed",
    )

    def __init__(self, seed: int, config: HybridConfig | None = None, *, nets: bool = True) -> None:
        """``nets=False`` loads no checkpoint: for a subclass that brings its own nets and
        replaces every method that reads them (``policy`` then raises)."""
        config = HybridConfig() if config is None else config
        if config.samples < 1 or config.oversample < 1:
            raise ValueError("the hybrid needs samples >= 1 and oversample >= 1")
        if config.belief_temperature <= 0 or not 0.0 < config.belief_floor <= 1.0:
            raise ValueError("the hybrid needs belief_temperature > 0 and 0 < belief_floor <= 1")
        if config.top_k < 0:
            raise ValueError("top_k must be 0 (every card) or more")
        if config.deals not in (BIDDING, GUESSES, BOTH):
            raise ValueError(f"deals must be {BIDDING}, {GUESSES} or {BOTH}, not {config.deals!r}")
        if config.guess_temperature <= 0:
            raise ValueError("guess_temperature must be above 0")
        if config.bid_top_k < 1 or config.bid_samples < 1:
            raise ValueError("the hybrid's bid search needs bid_top_k >= 1 and bid_samples >= 1")
        self.seed = seed
        self.config = config
        self._policy: Policy | None = None
        if nets:
            _set_threads(config.threads)
            policy = policy_at(config.checkpoint)
            if policy.bidding is None:
                raise ValueError(f"{config.checkpoint} has no trained BiddingNet")
            if config.deals != BIDDING and policy.guess is None:
                raise ValueError(
                    f"{config.checkpoint} has no GuessNet to deal by (deals={config.deals})"
                )
            self._policy = policy
        self._rng = random.Random(seed)
        self._history: BiddingHistory | None = None
        self._history_key: tuple[Any, ...] | None = None
        self.last_search: CardSearch | None = None
        self.last_bid_search: BidSearch | None = None

    def __repr__(self) -> str:
        return f"HybridBot(seed={self.seed}, {self.config})"

    @property
    def policy(self) -> Policy:
        """The checkpoint's nets this bot plays with."""
        if self._policy is None:
            raise ValueError("this bot was made without a checkpoint's nets (nets=False)")
        return self._policy

    def act(self, obs: Observation, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]
        x = encode(obs, self.config.rules)[np.newaxis]
        if obs.phase is not Phase.PLAY:
            values = self.policy.bid_values(x)[0]
            if self.config.bid_search:
                search = self.search_bids(obs, legal, [float(values[a]) for a in legal])
                return search.actions[best_index(search.means)]
            return max(legal, key=lambda action: values[action])
        advantage = self.policy.card_values(x)[0]
        cards = [action - PLAY_BASE for action in legal]
        search = self.search(obs, cards, [float(advantage[c]) for c in cards])
        return play_action(search.cards[best_index(search.means)])

    def search(self, obs: Observation, cards: list[int], advantages: list[float]) -> CardSearch:
        """Search the legal ``cards`` (CardPlayNet's ``advantages`` of each, to order them)."""
        config = self.config
        start_time = time.perf_counter()
        if config.merge_equivalent and len(cards) > 1:
            cards, advantages = _merge_equivalent(cards, advantages, read_knowledge(obs).unseen)
        order = sorted(range(len(cards)), key=lambda k: (-advantages[k], cards[k]))
        if config.top_k:
            order = order[: config.top_k]
        candidates = [cards[k] for k in order]
        net_values = [advantages[k] * REWARD_SCALE for k in order]
        if len(candidates) == 1:
            return CardSearch(candidates, [0.0], net_values, 0, 0.0)
        deals = self.sample_deals(obs)
        totals = self.evaluate(obs, candidates, deals)
        means = [total / len(deals) for total in totals]
        search = CardSearch(
            candidates, means, net_values, len(deals), time.perf_counter() - start_time
        )
        self.last_search = search
        return search

    def search_bids(self, obs: Observation, legal: list[int], values: list[float]) -> BidSearch:
        """Search BiddingNet's ``bid_top_k`` favourite ``legal`` actions (``values`` of each)."""
        config = self.config
        start_time = time.perf_counter()
        order = sorted(range(len(legal)), key=lambda k: (-values[k], k))[: config.bid_top_k]
        actions = [legal[k] for k in order]
        net_values = [values[k] * REWARD_SCALE for k in order]
        if len(actions) == 1:
            return BidSearch(actions, [0.0], net_values, 0, 0.0)
        deals = self.sample_deals(obs, config.bid_samples)
        weights = standings_weights(obs.scores, obs.seat) if config.leader_weight else None
        policy = self.policy
        totals = bid_totals(
            obs, config.rules, actions, deals, policy.bid_values, policy.card_values, weights
        )
        search = BidSearch(
            actions,
            [total / len(deals) for total in totals],
            net_values,
            len(deals),
            time.perf_counter() - start_time,
        )
        self.last_bid_search = search
        return search

    def sample_deals(self, obs: Observation, count: int | None = None) -> list[list[int]]:
        """``count`` deals (default ``samples``): the four remaining hands, own hand included,
        fitting the hand sizes, voids, played cards and void announcements, imagined as
        ``deals`` says: drawn at random or by GuessNet, and with ``bidding`` or ``both`` kept
        from ``count * oversample`` drawn by BiddingNet's likelihood of the others' bidding."""
        config = self.config
        rng = self._rng
        know = read_knowledge(obs)
        me = know.seat
        others = ((me + 1) & 3, (me + 2) & 3, (me + 3) & 3)
        sizes = [know.sizes[seat] for seat in others]
        voids = [know.voids[seat] for seat in others]
        unseen = know.unseen
        announced = obs.void_announced
        samples = config.samples if count is None else count
        weigh = config.deals != GUESSES and config.beliefs and config.oversample > 1
        drawn = samples * (config.oversample if weigh else 1)
        chances = self.guess_chances(obs) if config.deals != BIDDING else None

        def draw() -> list[int]:
            if chances is None:
                return deal_unseen(unseen, sizes, voids, rng)
            return deal_by_guesses(unseen, sizes, voids, chances, rng)

        deals: list[list[int]] = []
        for _ in range(drawn):
            three = draw()
            for _ in range(MAX_REDEALS):  # TBL-7: re-deal until the announcements fit
                if fits_announcements(three, others, know.played_by, announced):
                    break
                three = draw()
            hands = [0, 0, 0, 0]
            hands[me] = know.hand
            for i, seat in enumerate(others):
                hands[seat] = three[i]
            deals.append(hands)
        if not weigh:
            return deals
        logp = self.bidding_log_likelihoods(obs, deals, know.played_by)
        if logp is None:
            return deals[:samples]
        top = float(logp.max())
        # Efraimidis-Spirakis: keep the largest u ** (1 / weight), compared as logarithms.
        keys = [math.log(1.0 - rng.random()) * math.exp(min(top - float(lp), 700.0)) for lp in logp]
        kept = sorted(range(len(deals)), key=keys.__getitem__, reverse=True)[:samples]
        return [deals[k] for k in sorted(kept)]

    def bidding_log_likelihoods(
        self, obs: Observation, deals: list[list[int]], played_by: Sequence[int]
    ) -> npt.NDArray[np.float64] | None:
        """Each deal's log likelihood of the other seats' bidding this round under BiddingNet
        (``beliefs.log_likelihoods``), or None when none of them had a choice. ``deals`` are the
        remaining hands; ``played_by`` the cards each seat has played this round."""
        history = self._bidding_history(obs)
        if not len(history):
            return None
        dealt = np.array(deals, dtype=np.uint64) | np.array(played_by, dtype=np.uint64)
        return log_likelihoods(
            history,
            dealt,
            shapes_of(dealt),
            self.policy.bid_values,
            self.config.belief_temperature,
            self.config.belief_floor,
        )

    def guess_chances(self, obs: Observation) -> list[list[float]]:
        """GuessNet's chance of each other seat (the next one counterclockwise first) holding
        each card, from ``obs`` alone, raised to ``1 / guess_temperature``."""
        x = encode(obs, self.config.rules)[np.newaxis]
        chances = self.policy.card_guesses(x)[0].astype(np.float64)
        if self.config.guess_temperature != 1.0:
            chances = chances ** (1.0 / self.config.guess_temperature)
        return chances.tolist()

    def _bidding_history(self, obs: Observation) -> BiddingHistory:
        """The other seats' bidding this round, rebuilt once per round."""
        bidding = tuple(entry for entry in obs.log if entry[1] < PLAY_BASE)
        key = (obs.seat, obs.round_no, obs.scores, obs.carry, obs.void_announced, bidding)
        if self._history is None or self._history_key != key:
            self._history = bidding_history(obs, self.config.rules)
            self._history_key = key
        return self._history

    def evaluate(self, obs: Observation, cards: list[int], deals: list[list[int]]) -> list[float]:
        """For each card, the sum over ``deals`` of own round score minus the others' (players
        ahead counting double with ``leader_weight``), playing the card and then the rest of
        the round with CardPlayNet for every seat."""
        trump, leader = obs.trump, obs.leader
        if trump is None or leader is None:
            raise ValueError("the hybrid searches only during card play")
        config = self.config
        know = read_knowledge(obs)
        me = obs.seat
        n_cards, n_deals = len(cards), len(deals)
        games = n_cards * n_deals  # game g plays deal g // n_cards with card g % n_cards
        deal_hands = np.array(deals, dtype=np.uint64)
        played_row = np.array(know.played_by, dtype=np.uint64)
        hands = np.repeat(deal_hands, n_cards, axis=0)
        played = np.tile(played_row, (games, 1))
        voids = np.tile(np.array(know.voids, dtype=np.uint8), (games, 1))
        shapes = np.repeat(shapes_of(deal_hands | played_row), n_cards, axis=0)
        first = np.tile(np.array(cards, dtype=np.int64), n_deals)
        start = Start(
            trick=[card for _, card in obs.trick],
            leader=leader,
            won=obs.tricks_won,
            tricks_done=len(obs.tricks),
            estimates=estimates_of(obs),
            trump=trump,
        )
        templates = seat_templates(obs, config.rules)
        won = play_out(
            self.policy.card_values, templates, start, hands, played, voids, shapes, first
        )
        weights = standings_weights(obs.scores, me) if config.leader_weight else None
        scorer = RoundScorer(obs, config.rules, weights)
        totals = [0.0] * n_cards
        for g, tricks in enumerate(won.tolist()):
            totals[g % n_cards] += scorer.value(tricks)
        return totals


def equivalent_runs(cards: list[int], out: int) -> list[list[int]]:
    """``cards`` split into runs of touching cards: same suit, and ``out`` (the cards still in
    other hands) holds no card ranked between them, so they take the same tricks."""
    runs: list[list[int]] = []
    for card in sorted(cards):
        if runs:
            last = runs[-1][-1]
            between = ((1 << card) - 1) & ~((2 << last) - 1)
            if last // 13 == card // 13 and not out & between:
                runs[-1].append(card)
                continue
        runs.append([card])
    return runs


def _merge_equivalent(
    cards: list[int], advantages: list[float], out: int
) -> tuple[list[int], list[float]]:
    """One card from each run of touching cards (``equivalent_runs``), keeping the card with the
    highest advantage in each run (the lowest card id on a tie)."""
    by_adv = dict(zip(cards, advantages, strict=True))
    kept = [max(run, key=lambda c: (by_adv[c], -c)) for run in equivalent_runs(cards, out)]
    return kept, [by_adv[c] for c in kept]
