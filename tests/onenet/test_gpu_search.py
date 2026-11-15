"""Step 5's search on the GPU against the CPU search it copies (estimation.onenet.search)."""

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.bots.hybrid import HybridConfig
from estimation.bots.knowledge import read_knowledge
from estimation.bots.sampling import deal_by_guesses
from estimation.bots.scoring import RoundScorer, standings_weights
from estimation.engine import Bola, Observation, Phase
from estimation.engine.actions import PLAY_BASE
from estimation.jaxenv import encode as E
from estimation.jaxenv import rules as R
from estimation.jaxenv.check import bits
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.onenet import gpu_search as G
from estimation.onenet.search import OneNetSearch, load_net
from tests.onenet.test_search import random_net

_step = jax.jit(R.step)
_encode = jax.jit(E.encode)


class Position:
    """A card decision with a choice, in the engine and as the JAX round, with the round's
    bidding decisions so far as the GPU search keeps them."""

    def __init__(self, bola: Bola, r: R.Round, bidding: list[tuple[Any, int, Any, int]]) -> None:
        seat = bola.current_player
        assert seat is not None
        self.obs: Observation = bola.observation(seat)
        self.legal = bola.legal_actions()
        self.round = r
        self.me = seat
        rows = bidding or [(np.zeros(E.SIZE, np.float32), -1, np.zeros(120, bool), 0)]
        self.bidding = G.Bidding(
            jnp.asarray(np.stack([x for x, _, _, _ in rows])),
            jnp.asarray([s for _, s, _, _ in rows], jnp.int32),
            jnp.asarray(np.stack([legal[: G.BID_ACTIONS] for _, _, legal, _ in rows])),
            jnp.asarray([a for _, _, _, a in rows], jnp.int32),
            jnp.asarray([bool(bidding)] * len(rows)),
        )


def positions(net_path: str, seed: int, want: int) -> Iterator[Position]:
    """Card decisions with a choice from the first rounds of Bolas played by the net's top
    move, the JAX round stepped alongside the engine."""
    net = load_net(net_path)
    rng = random.Random(seed)
    found = 0
    while found < want:
        bola = Bola(rng.randrange(2**31))
        r = R.init_round(1, jnp.asarray(bits(bola.round.hands)))
        bidding: list[tuple[Any, int, Any, int]] = []
        while bola.round_no == 1 and not bola.is_over:
            seat = bola.current_player
            assert seat is not None
            legal = bola.legal_actions()
            obs = bola.observation(seat)
            if obs.phase is Phase.PLAY and len(legal) > 1 and rng.random() < 0.3:
                yield Position(bola, r, list(bidding))
                found += 1
                if found >= want:
                    return
            logits = net(raw_view(obs, legal))
            action = max(legal, key=lambda a: logits[a])
            if obs.phase is not Phase.PLAY and len(legal) > 1:
                x = np.asarray(_encode(r, seat)).copy()
                x[:52] = 0.0
                mask = np.zeros(120, bool)
                mask[legal] = True
                bidding.append((x, seat, mask, action))
            bola.step(action)
            r = _step(r, action)


@pytest.fixture(scope="module")
def net_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return random_net(tmp_path_factory.mktemp("net"))


@pytest.fixture(scope="module")
def found(net_path: str) -> list[Position]:
    return list(positions(net_path, 5, 12))


def cpu_bot(net_path: str, **changes: Any) -> OneNetSearch:
    config = HybridConfig(top_k=0, samples=8, **changes)
    return OneNetSearch(1, config, net_path, own=True)


def deals_array(deals: list[list[int]]) -> jax.Array:
    return jnp.asarray(np.stack([bits(d) for d in deals]))


def test_the_holder_guesses_equal_the_cpu_players(net_path: str, found: list[Position]):
    player = G.load_player(net_path)
    bot = cpu_bot(net_path)
    for p in found:
        gpu = np.asarray(G.holder_chances(player, E.encode(p.round, p.me)))
        assert gpu == pytest.approx(np.array(bot.guess_chances(p.obs)), abs=1e-5)


def test_imagined_deals_fit_the_sizes_voids_and_announcements(net_path: str, found: list[Position]):
    player = G.load_player(net_path)
    settings = G.Settings(samples=16)
    for k, p in enumerate(found):
        hands = np.asarray(
            G.imagine(
                player, settings, jax.random.PRNGKey(k), p.round, jnp.asarray(p.me), p.bidding
            )
        )
        know = read_knowledge(p.obs)
        for deal in hands:
            assert (deal.sum(0) <= 1).all()  # no card twice
            assert deal[p.me].tolist() == bits([know.hand])[0].tolist()
            for seat in range(4):
                assert deal[seat].sum() == know.sizes[seat]
                assert not (deal[seat] & bits([know.played_by[seat]])[0]).any()
                for suit in range(4):
                    if know.voids[seat] >> suit & 1:
                        assert not deal[seat, 13 * suit : 13 * suit + 13].any()
            assert deal.sum() == 52 - sum(bin(x).count("1") for x in know.played_by)


def test_a_deal_places_cards_as_the_cpu_does():
    """Same constrained hands, guesses favouring one seat per suit: each card's holder rates
    match ``deal_by_guesses`` within sampling noise."""
    rng = np.random.default_rng(3)
    unseen_cards = [c for c in range(52) if c % 3 != 0][:30]
    unseen = np.zeros(52, bool)
    unseen[unseen_cards] = True
    sizes = [12, 10, 8]
    voids = np.zeros((3, 4), bool)
    voids[0, 1] = voids[2, 3] = True
    chances = rng.dirichlet([0.7, 0.7, 0.7], size=52)
    draws = 3000
    keys = jax.random.split(jax.random.PRNGKey(0), draws)
    gpu = np.asarray(
        jax.vmap(G.deal, in_axes=(0, None, None, None, None))(
            keys, jnp.asarray(unseen), jnp.asarray(sizes), jnp.asarray(voids), jnp.asarray(chances)
        )
    )
    mask = sum(1 << c for c in unseen_cards)
    void_masks = [sum(1 << s for s in range(4) if voids[i, s]) for i in range(3)]
    py = random.Random(1)
    cpu = np.full((draws, 52), -1)
    for d in range(draws):
        for i, hand in enumerate(deal_by_guesses(mask, sizes, void_masks, chances.tolist(), py)):
            for c in range(52):
                if hand >> c & 1:
                    cpu[d, c] = i
    for hand in range(3):
        assert ((gpu == hand).sum(1) == sizes[hand]).all()
        gap = np.abs((gpu == hand).mean(0) - (cpu == hand).mean(0))
        assert gap.max() < 0.05


def test_the_bid_weights_equal_the_cpu_players(net_path: str, found: list[Position]):
    player = G.load_player(net_path)
    bot = cpu_bot(net_path)
    checked = 0
    for p in found:
        deals = bot.sample_deals(p.obs, 6)
        know = read_knowledge(p.obs)
        cpu = bot.bidding_log_likelihoods(p.obs, deals, know.played_by)
        gpu, weighed = G.bid_log_likelihoods(
            player, p.bidding, jnp.asarray(p.me), deals_array(deals), p.round.played, 0.1
        )
        assert bool(weighed) == (cpu is not None)
        if cpu is not None:
            assert np.asarray(gpu) == pytest.approx(cpu, rel=1e-4, abs=1e-4)
            checked += 1
    assert checked > 5


def test_play_outs_and_scores_equal_the_cpu_search(net_path: str, found: list[Position]):
    player = G.load_player(net_path)
    bot = cpu_bot(net_path)
    for p in found:
        deals = bot.sample_deals(p.obs, 5)
        cards = [a - PLAY_BASE for a in p.legal]
        cpu = bot.evaluate(p.obs, cards, deals)  # totals over the deals
        values = G.evaluate(
            player,
            jax.tree.map(lambda x: x[None], p.round),
            jnp.asarray([p.me]),
            deals_array(deals)[None],
            jnp.zeros(len(cards), jnp.int32),
            jnp.asarray(cards),
            True,
        )
        assert np.asarray(values.sum(-1)) == pytest.approx(cpu, abs=1e-3)


def test_round_values_equal_the_round_scorer(found: list[Position]):
    rng = random.Random(2)
    for p in found:
        for _ in range(5):
            scores = (rng.randrange(-50, 50), rng.randrange(-50, 50), 0, rng.randrange(-50, 50))
            scorer = RoundScorer(p.obs, weights=standings_weights(scores, p.me))
            tricks = [0, 0, 0, 0]
            for _ in range(13):
                tricks[rng.randrange(4)] += 1
            done = p.round._replace(won=jnp.asarray(tricks, jnp.int32))
            result = R._score(done).result  # pyright: ignore[reportPrivateUsage]
            gpu = G.round_values(result, jnp.asarray(scores), jnp.asarray(p.me))
            assert float(gpu) == pytest.approx(scorer.value(tuple(tricks)), abs=1e-4)


def test_the_search_values_every_searched_card_and_picks_the_best(
    net_path: str, found: list[Position]
):
    player = G.load_player(net_path)
    batch_cards = []
    for p in found:
        cards = [a - PLAY_BASE for a in p.legal][:3]
        batch_cards.append(cards + [-1] * (3 - len(cards)))
    batch = G.Decisions(
        jax.tree.map(lambda *xs: jnp.stack(xs), *[p.round for p in found]),
        jnp.asarray([p.me for p in found]),
        jnp.asarray(batch_cards),
        jax.tree.map(lambda *xs: jnp.stack(xs), *[p.bidding for p in found]),
    )
    means, best = G.search(player, G.Settings(samples=4), jax.random.PRNGKey(0), batch)
    means, best = np.asarray(means), np.asarray(best)
    for row, cards in enumerate(batch_cards):
        real = [c for c in cards if c >= 0]
        assert np.isfinite(means[row, : len(real)]).all()
        assert (means[row, len(real) :] == -np.inf).all()
        assert best[row] == real[int(np.argmax(means[row, : len(real)]))]


def test_a_file_without_holder_answers_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="practice answers"):
        G.load_player(random_net(tmp_path, answers=False))
