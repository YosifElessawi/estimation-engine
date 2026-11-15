"""Step 5's search on the GPU (docs/rl/hybrid-iteration-plan.md): the one net's own search, for
many card decisions at once, in JAX.

It is ``estimation.onenet.search.OneNetSearch`` with ``own=true`` and ``judge=round``, step for
step:

1. **Imagine the deals.** The player's hidden-card guesses (its holder answers at the decision)
   place every unseen card, one at a time, among the other three seats that keep a fitting deal
   possible, exactly as ``deal_by_guesses`` does (``deal``). A deal that breaks the void
   announcements is drawn again, up to ``redeals`` times (the CPU draws up to 200 times; a
   mismatch there is rare, and both keep the last draw).
2. **Keep the deals that explain the bidding.** Each deal's log likelihood of the other seats'
   bidding this round, under the player's own move chances mixed with a floor
   (``bid_log_likelihoods``), and ``samples`` of the drawn deals kept by weighted sampling
   without replacement (Efraimidis-Spirakis, ``keep``).
3. **Play out.** Every searched card on every kept deal, then the player's top legal card for
   every seat to the end of the round (``play_outs``).
4. **Score.** Own round score minus the other three's, players ahead in the Bola counting double
   (``round_values``), averaged over the deals.

Only what the searching seat may see goes in: its own hand, the public table and bidding, and its
guesses. The other three hands of the real round are replaced by imagined ones before anything is
played.
"""

import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from estimation.jaxenv import rules as R
from estimation.jaxenv.encode import encode

I32 = jnp.int32
_ALL = 0b111
_SUBSETS = range(1, 8)


class Player(NamedTuple):
    """The player's weights as JAX arrays: hidden layers, the move head, and the holder answers
    (its part of the practice head)."""

    layers: tuple[tuple[jax.Array, jax.Array], ...]
    move: tuple[jax.Array, jax.Array]
    holder: tuple[jax.Array, jax.Array]


def load_player(path: str) -> Player:
    """The player saved at ``path``, with its holder answers (a file saved with its practice
    answers, as step 5 saves its players)."""
    with np.load(path) as data:
        layers: list[tuple[jax.Array, jax.Array]] = []
        k = 0
        while f"w{k}" in data:
            layers.append((jnp.asarray(data[f"w{k}"]), jnp.asarray(data[f"b{k}"])))
            k += 1
        if "qw" not in data or "meta" not in data:
            raise ValueError(f"{path} has no practice answers, which the GPU search reads")
        start = 0
        holder: slice | None = None
        for name, size in json.loads(str(data["meta"])).get("practice", []):
            if name == "holder":
                holder = slice(start, start + size)
            start += size
        if holder is None:
            raise ValueError(f"{path} has no holder answers")
        qw, qb = data["qw"][:, holder], data["qb"][holder]
        return Player(
            tuple(layers),
            (jnp.asarray(data["pw"]), jnp.asarray(data["pb"])),
            (jnp.asarray(qw), jnp.asarray(qb)),
        )


def body(player: Player, x: jax.Array) -> jax.Array:
    for w, b in player.layers:
        x = jax.nn.relu(x @ w + b)
    return x


def move_logits(player: Player, x: jax.Array) -> jax.Array:
    """``(..., 120)`` move logits of raw views ``(..., SIZE)``."""
    w, b = player.move
    return body(player, x) @ w + b


def holder_chances(player: Player, x: jax.Array) -> jax.Array:
    """``(..., 52, 3)``: each card's chance of being held by each other seat (the next first)."""
    w, b = player.holder
    logits = (body(player, x) @ w + b).reshape((*x.shape[:-1], 52, 3))
    return jax.nn.softmax(logits, axis=-1)


# 1. Imagining a deal (estimation.bots.sampling.deal_by_guesses).

# For a card allowed in hands ``allowed`` going to hand ``i``: the hand subsets whose slack it
# uses up (they contain i but cannot absorb the card's suit on their own).
_SPENDS = np.zeros((8, 3, 8), np.bool_)
for _allowed in range(1, 8):
    for _i in range(3):
        if _allowed >> _i & 1:
            for _s in _SUBSETS:
                if _s >> _i & 1 and _allowed & ~_s:
                    _SPENDS[_allowed, _i, _s] = True
SPENDS = jnp.asarray(_SPENDS)
_IN = jnp.asarray([[s >> i & 1 for i in range(3)] for s in range(8)], jnp.bool_)  # [8, 3]
_POP = jnp.asarray([bin(a).count("1") for a in range(8)], I32)
_SUIT = jnp.arange(52) // 13


def deal(
    key: jax.Array,
    unseen: jax.Array,
    sizes: jax.Array,
    voids: jax.Array,
    chances: jax.Array,
) -> jax.Array:
    """``[52]``: for each card, which of the three other hands (0-2, the next seat first) holds it
    in one imagined deal, -1 for a card not unseen.

    ``unseen`` [52] bool, ``sizes`` [3] cards in each hand, ``voids`` [3, 4] bool suits each hand
    is known to miss, ``chances`` [52, 3] the guess for each card. As ``deal_by_guesses``: cards
    of suits some hand is void in go first, most limited first, then the rest in a random order;
    each goes to a hand in proportion to its chance among the hands with free places that keep
    the rest of the deal possible (Hall), or in proportion to free places when every such chance
    is 0."""
    k_order, k_pick = jax.random.split(key)
    allowed_suit = jnp.asarray([1, 2, 4], I32) @ (~voids).astype(I32)  # [4] 3-bit masks
    allowed = allowed_suit[_SUIT]  # [52]
    constrained = unseen & (allowed != _ALL)
    is_open = unseen & (allowed == _ALL)
    # The CPU order: constrained suits by how few hands may take them (then suit), shuffled
    # within each suit; then every open card, shuffled.
    group = jnp.where(constrained, _POP[allowed] * 4 + _SUIT, jnp.where(is_open, 100, 1000))
    order = jnp.lexsort((jax.random.uniform(k_order, (52,)), group))
    # Hall slack per subset of hands: its free places minus the cards only it can take.
    places = _IN.astype(I32) @ sizes  # [8]
    only = (constrained[None, :] & ((allowed[None, :] & ~jnp.arange(8)[:, None]) == 0)).sum(1)
    needed = only + jnp.where(jnp.arange(8) == _ALL, is_open.sum(), 0)
    slack0 = places - needed
    picks = jax.random.uniform(k_pick, (52,))

    def place(carry: tuple[jax.Array, jax.Array, jax.Array], k: jax.Array):
        free, slack, seat_of = carry
        card = order[k]
        live = unseen[card]
        a = allowed[card]
        spends = SPENDS[a]  # [3, 8]
        fits = ~(spends & (slack[None, :] <= 0)).any(1)
        options = _IN[a] & (free > 0) & fits
        weights = jnp.where(options, jnp.maximum(chances[card], 0.0), 0.0)
        by_free = jnp.where(options, free, 0).astype(weights.dtype)
        weights = jnp.where(weights.sum() > 0, weights, by_free)
        cum = jnp.cumsum(weights)
        hand = jnp.minimum((cum <= picks[k] * cum[-1]).sum(), 2)
        hand = jnp.where(options[hand], hand, jnp.argmax(options))  # rounding at the top edge
        free = jnp.where(live, free.at[hand].add(-1), free)
        slack = jnp.where(live, slack - spends[hand].astype(I32), slack)
        seat_of = seat_of.at[card].set(jnp.where(live, hand, -1))
        return (free, slack, seat_of), None

    start = (sizes.astype(I32), slack0.astype(I32), jnp.full(52, -1, I32))
    (_, _, seat_of), _ = jax.lax.scan(place, start, jnp.arange(52))
    return seat_of


def fits_announcements(
    seat_of: jax.Array, played_others: jax.Array, announced_others: jax.Array
) -> jax.Array:
    """Whether an imagined deal fits the void announcements (TBL-7): a hand announced void misses
    a suit in its dealt cards (imagined plus played this round), any other holds all four.
    ``played_others`` [3, 52] and ``announced_others`` [3] are the other hands' (next first)."""
    dealt = (seat_of[None, :] == jnp.arange(3)[:, None]) | played_others  # [3, 52]
    void = (~dealt.reshape(3, 4, 13).any(-1)).any(-1)
    return (void == announced_others).all()


def deals(
    key: jax.Array,
    count: int,
    redeals: int,
    unseen: jax.Array,
    sizes: jax.Array,
    voids: jax.Array,
    chances: jax.Array,
    played_others: jax.Array,
    announced_others: jax.Array,
) -> jax.Array:
    """``[count, 52]`` imagined deals (``deal``), each the first of ``redeals`` draws that fits the
    void announcements, or the last draw when none does."""
    keys = jax.random.split(key, count * redeals)
    draw = jax.vmap(deal, in_axes=(0, None, None, None, None))
    drawn = draw(keys, unseen, sizes, voids, chances).reshape(count, redeals, 52)
    fit = jax.vmap(jax.vmap(fits_announcements, in_axes=(0, None, None)), in_axes=(0, None, None))(
        drawn, played_others, announced_others
    )
    first = jnp.where(fit.any(1), jnp.argmax(fit, axis=1), redeals - 1)
    return drawn[jnp.arange(count), first]


# 2. Keeping the deals that explain the bidding (estimation.bots.hybrid.beliefs).

BID_ACTIONS = R.PLAY_BASE


class Bidding(NamedTuple):
    """The bidding decisions with a choice so far this round, padded to a fixed number: each
    one's raw view with its hand cleared, its seat, its legal options, the option chosen, and
    whether the row is a real decision (``valid``). Batched with a leading decision axis."""

    templates: jax.Array  # [H, SIZE]
    seats: jax.Array  # [H]
    legal: jax.Array  # [H, BID_ACTIONS] bool
    chosen: jax.Array  # [H]
    valid: jax.Array  # [H] bool


def bid_log_likelihoods(
    player: Player,
    bidding: Bidding,
    me: jax.Array,
    hands: jax.Array,
    played: jax.Array,
    floor: float,
) -> tuple[jax.Array, jax.Array]:
    """Each deal's log likelihood of the other seats' bidding this round, ``[D]``, under the
    player's move chances: a softmax of its logits over the legal options, mixed with an even
    ``floor`` share (``choice_log_likelihoods``). ``hands`` [D, 4, 52] are each deal's remaining
    hands and ``played`` [4, 52] the cards played this round, so a seat's dealt hand is their
    union. Also whether any other seat had a choice (else every weight is even)."""
    use = bidding.valid & (bidding.seats != me)
    seats = jnp.maximum(bidding.seats, 0)
    dealt = hands[:, seats] | played[seats][None]  # [D, H, 52]
    # The first layer of a view is its template's (hand cleared) plus its hand's rows, so the
    # template's share is worked out once for every deal.
    (w0, b0), *rest = player.layers
    x = jax.nn.relu((bidding.templates @ w0 + b0)[None] + dealt.astype(w0.dtype) @ w0[:52])
    for w, b in rest:
        x = jax.nn.relu(x @ w + b)
    pw, pb = player.move
    scores = (x @ pw + pb)[:, :, :BID_ACTIONS]
    values = jnp.where(bidding.legal[None], scores, -jnp.inf)
    soft = jax.nn.softmax(values, axis=-1)
    picked = jnp.take_along_axis(soft, bidding.chosen[None, :, None], axis=-1)[..., 0]
    options = jnp.maximum(bidding.legal.sum(-1), 1)[None]
    probability = (1.0 - floor) * picked + floor / options
    return jnp.where(use[None], jnp.log(probability), 0.0).sum(-1), use.any()


def keep(key: jax.Array, log_weights: jax.Array, weighed: jax.Array, samples: int) -> jax.Array:
    """``[samples]`` indices of deals kept by weighted sampling without replacement
    (Efraimidis-Spirakis, as Gumbel keys), or the first ``samples`` when not ``weighed``."""
    gumbel = -jnp.log(-jnp.log(1.0 - jax.random.uniform(key, log_weights.shape)))
    _, top = jax.lax.top_k(log_weights + gumbel, samples)
    return jnp.where(weighed, top, jnp.arange(samples))


# 3. Playing out (estimation.bots.hybrid.rollout) and 4. scoring (estimation.bots.scoring).

AHEAD, BEHIND = 1.0, 0.5  # standings_weights: players ahead count double


def _select(cond: jax.Array, a: R.Round, b: R.Round) -> R.Round:
    def pick(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.where(cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim)), x, y)

    return jax.tree.map(pick, a, b)


def play_outs(player: Player, rounds: R.Round, first: jax.Array) -> R.Round:
    """Batched rounds in card play, each played from its first move ``first`` (a card) and then
    the player's top legal card for every seat to the end: the finished rounds (``won`` and
    ``result``)."""

    def going(c: tuple[R.Round, jax.Array]) -> jax.Array:
        r, _ = c
        return (r.phase != R.DONE).any()

    def turn(c: tuple[R.Round, jax.Array]) -> tuple[R.Round, jax.Array]:
        r, forced = c
        x = jax.vmap(encode)(r, r.cur)
        legal = jax.vmap(R.legal_mask)(r)
        top = jnp.where(legal, move_logits(player, x), -jnp.inf).argmax(-1)
        action = jnp.where(forced >= 0, forced + R.PLAY_BASE, top)
        nxt = jax.vmap(R._step_play)(r, action)  # pyright: ignore[reportPrivateUsage]
        return _select(r.phase == R.DONE, r, nxt), jnp.full_like(forced, -1)

    r, _ = jax.lax.while_loop(going, turn, (rounds, first))
    return r


def standings(scores: jax.Array, me: jax.Array) -> jax.Array:
    """``[4]`` weights of the others' round scores against mine (``standings_weights``)."""
    raw = jnp.where(scores > scores[me], AHEAD, BEHIND)
    raw = jnp.where(jnp.arange(4) == me, 0.0, raw)
    return raw / raw.sum()


def round_values(result: jax.Array, scores: jax.Array, me: jax.Array) -> jax.Array:
    """Own round score minus the others' weighted by the standings, for results ``[..., 4]``."""
    w = standings(scores, me)
    return result[..., me] - (result * w).sum(-1)


# The whole search for a batch of card decisions.


class Settings(NamedTuple):
    """The search's settings, as ``HybridConfig`` names them (static under jit)."""

    samples: int = 32  # deals kept
    oversample: int = 2  # deals drawn per deal kept
    redeals: int = 8  # draws per deal to fit the void announcements
    floor: float = 0.1  # belief_floor
    guess_temperature: float = 1.0
    leader_weight: bool = True


class Decisions(NamedTuple):
    """A batch of card decisions: each real round (only the acting seat's hand and the public
    table are read), the acting seat, the cards to search (-1 padded) and the round's bidding.
    Every field has a leading decision axis."""

    round: R.Round
    me: jax.Array  # [N]
    cards: jax.Array  # [N, C]
    bidding: Bidding


def imagine(
    player: Player, settings: Settings, key: jax.Array, r: R.Round, me: jax.Array, bidding: Bidding
) -> jax.Array:
    """``[samples, 4, 52]``: the remaining hands of the deals one decision searches, the acting
    seat's own hand in every one."""
    k_draw, k_keep = jax.random.split(key)
    chances = holder_chances(player, encode(r, me))
    if settings.guess_temperature != 1.0:
        chances = chances ** (1.0 / settings.guess_temperature)
    others = (me + 1 + jnp.arange(3)) % 4
    mine = r.hands[me]
    unseen = ~mine & ~r.played.any(0)
    sizes = 13 - r.played[others].sum(-1).astype(I32)
    drawn = deals(
        k_draw,
        settings.samples * settings.oversample,
        settings.redeals,
        unseen,
        sizes,
        r.shown_out[others],
        chances,
        r.played[others],
        r.announced[others],
    )
    holder = jnp.where(drawn >= 0, others[jnp.maximum(drawn, 0)], -1)  # [D, 52] absolute seats
    hands = holder[:, None, :] == jnp.arange(4)[None, :, None]
    hands = hands | ((jnp.arange(4) == me)[None, :, None] & mine[None, None, :])
    if settings.oversample == 1:
        return hands
    logp, weighed = bid_log_likelihoods(player, bidding, me, hands, r.played, settings.floor)
    return hands[keep(k_keep, logp, weighed, settings.samples)]


def evaluate(
    player: Player,
    rounds: R.Round,
    me: jax.Array,
    hands: jax.Array,
    which: jax.Array,
    cards: jax.Array,
    leader_weight: bool,
) -> jax.Array:
    """``[P, S]``: the value of searched card ``cards[p]`` of decision ``which[p]`` on each of
    that decision's deals, played out to the end of the round and scored by round points.
    ``rounds`` and ``me`` have one entry per decision, ``hands`` ``[N, S, 4, 52]`` its deals."""
    p, s = cards.shape[0], hands.shape[1]

    def spread(x: jax.Array) -> jax.Array:  # [N, ...] -> [P * S, ...]
        wide = jnp.broadcast_to(x[which][:, None], (p, s, *x.shape[1:]))
        return wide.reshape(p * s, *x.shape[1:])

    played = jax.tree.map(spread, rounds)._replace(hands=hands[which].reshape(p * s, 4, 52))
    first = jnp.broadcast_to(cards[:, None], (p, s)).reshape(-1)
    result = play_outs(player, played, first).result.reshape(p, s, 4)
    seat, scores = me[which], rounds.scores[which]
    if leader_weight:
        return jax.vmap(round_values)(result, scores, seat)
    mine = jnp.take_along_axis(result, seat[:, None, None], axis=-1)[..., 0]
    return mine - (result.sum(-1) - mine) / 3.0


DEAL_BLOCK = 8192  # imagined deals worked on at once, to bound the bid check's memory


def imagine_all(
    player: Player,
    settings: Settings,
    key: jax.Array,
    rounds: R.Round,
    me: jax.Array,
    bidding: Bidding,
) -> jax.Array:
    """``[N, S, 4, 52]``: ``imagine`` for each of a batch of decisions, a block of decisions at a
    time so the bid check never holds more than about ``DEAL_BLOCK`` drawn deals."""
    keys = jax.random.split(key, me.shape[0])
    drawn = settings.samples * settings.oversample
    block = max(1, DEAL_BLOCK // drawn)

    def one(args: tuple[jax.Array, R.Round, jax.Array, Bidding]) -> jax.Array:
        k, r, m, b = args
        return imagine(player, settings, k, r, m, b)

    return jax.lax.map(one, (keys, rounds, me, bidding), batch_size=block)


def search(
    player: Player, settings: Settings, key: jax.Array, batch: Decisions
) -> tuple[jax.Array, jax.Array]:
    """Each decision's mean value of each searched card, ``[N, C]`` (padded cards score
    ``-inf``), and its best card, the first searched card on a tie."""
    n, c = batch.cards.shape
    hands = imagine_all(player, settings, key, batch.round, batch.me, batch.bidding)
    which = jnp.repeat(jnp.arange(n), c)
    cards = jnp.where(batch.cards >= 0, batch.cards, batch.cards[:, :1]).reshape(-1)
    values = evaluate(player, batch.round, batch.me, hands, which, cards, settings.leader_weight)
    means = jnp.where(batch.cards >= 0, values.mean(-1).reshape(n, c), -jnp.inf)
    return means, jnp.take_along_axis(batch.cards, means.argmax(-1)[:, None], axis=1)[:, 0]
