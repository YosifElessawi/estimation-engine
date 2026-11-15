"""Step 6's bid teacher (docs/rl/hybrid-iteration-plan.md): a slow, careful search of one bidding
decision, for many decisions at once, in JAX. The player learns its bids from the teacher's
verdicts (``estimation.onenet.bid_lessons`` records them, ``bid_learn`` learns them); the teacher
never plays at the table.

At a Dash Call, bid, With, Pass, estimate or super call with a choice:

1. **Options** (``candidates``). The player's legal options, its favourite first, less the calls
   and estimates far from the tricks it expects to take (its ``my_tricks`` answer), until they
   cover ``coverage`` of its move chances: at least ``min_options`` of them, at most
   ``max_options``. Its favourite, the move it makes at the table, always comes first.
2. **Deals** (``imagine``). As the card search imagines them (``estimation.onenet.gpu_search``):
   the player's hidden-card guesses place the unseen cards, and the deals that explain the other
   seats' bidding so far are kept. They come in a random order, so each stage of the narrowing
   below draws a fair share of them.
3. **Play-outs** (``evaluate``). On each deal the option is made, the bidding is finished and the
   round is played out, the player in every seat: the deciding seat bids its top move, as it
   does at the table; the other seats' bids are drawn from the player's move chances
   (``opponent_temperature``), since real opponents do not all make the player's favourite bid;
   every card is the player's top card. A game's draws belong to its deal, not to the option
   tried, so two options meet the same opponents wherever the bidding goes the same way.
4. **Score.** Round points, as the card search scores them: own round score minus the others',
   players ahead in the Bola counting double.
5. **Narrowing** (``Teacher.teach``). Every option plays the first ``deals[0]`` deals and the
   worse half of the alternatives is dropped; the rest play ``deals[1]`` more; and so on, until
   one alternative is left to meet the favourite over every deal. The favourite always stays, so
   a verdict says by how many points, and how surely, each alternative beat the move the player
   makes today over the deals they shared.

Only what the deciding seat may see goes in: its own hand, the public table and bidding, and its
own guesses. The other three hands of the real round are replaced by imagined ones before any
option is tried.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from estimation.jaxenv import rules as R
from estimation.jaxenv.encode import encode

from . import gpu_search as G

I32 = jnp.int32
BID_ACTIONS = R.PLAY_BASE  # every bidding move's id is below the cards'
MOVES = 64  # bidding moves kept per round: Dash Calls, bids, answers and estimates
BIDDING_LIMIT = 256  # no legal bidding lasts this long: a play-out's guard against a bad state
Floats = npt.NDArray[np.float32]
Ints = npt.NDArray[np.int32]


class Settings(NamedTuple):
    """How the teacher searches (static under jit). The first six are the card search's
    (``gpu_search.Settings``); ``deals`` are the deals each stage of the narrowing adds."""

    deals: tuple[int, ...] = (32, 64, 128, 256)
    oversample: int = 2  # deals drawn per deal kept
    redeals: int = 8  # draws per deal to fit the void announcements
    floor: float = 0.1  # belief_floor
    guess_temperature: float = 1.0
    leader_weight: bool = True
    opponent_temperature: float = 1.0  # the other seats' bids: 0 is their top move, 1 their chances


class Pruning(NamedTuple):
    """Which options the teacher tries (``candidates``)."""

    coverage: float = 0.97  # of the player's move chances
    min_options: int = 5
    max_options: int = 8
    hand_margin: float = 3.0  # tricks a call or estimate may be from the player's expected tricks


def check(settings: Settings, pruning: Pruning) -> None:
    """Refuse settings the narrowing cannot run."""
    if not settings.deals or min(settings.deals) < 2:
        raise ValueError("the teacher's deals must be one or more stages of at least 2 deals")
    if settings.oversample < 1 or settings.redeals < 1:
        raise ValueError("the teacher's oversample and redeals must be at least 1")
    if settings.opponent_temperature < 0:
        raise ValueError("the teacher's opponent_temperature must be 0 or more")
    if not 0 < pruning.coverage <= 1:
        raise ValueError("the teacher's coverage must be above 0 and at most 1")
    most = 1 + 2 ** (len(settings.deals) - 1)
    if not 2 <= pruning.min_options <= pruning.max_options <= most:
        raise ValueError(
            f"the teacher's options must be 2 <= min_options <= max_options <= {most}: "
            f"{len(settings.deals)} stages narrow at most {most} options down to 2"
        )
    if pruning.hand_margin < 0:
        raise ValueError("the teacher's hand_margin must be 0 or more")


# The player, as the teacher reads it.


class Bidder(NamedTuple):
    """The search's player (body, move output, holder answers) with its tricks answer
    (``my_tricks``: its chances of taking 0 to 13 tricks this round)."""

    player: G.Player
    tricks: tuple[jax.Array, jax.Array]


def load_bidder(path: str) -> Bidder:
    """The player saved at ``path`` with its practice answers."""
    return Bidder(G.load_player(path), G.practice_answer(path, "my_tricks"))


def opinions(bidder: Bidder, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The player's move logits ``[..., 120]`` and the tricks it expects to take ``[...]`` at
    views ``x``."""
    h = G.body(bidder.player, x)
    pw, pb = bidder.player.move
    tw, tb = bidder.tricks
    return h @ pw + pb, jax.nn.softmax(h @ tw + tb, axis=-1) @ jnp.arange(14.0)


# 1. The options.


class Options(NamedTuple):
    """The options the teacher tries at one decision, the player's favourite first: their action
    ids, the player's chance of each (among every legal option) and each one's place among all
    the legal options by that chance (0 is the favourite)."""

    actions: list[int]
    chances: list[float]
    ranks: list[int]


def near(action: int, tricks: float, margin: float) -> bool:
    """Whether a bidding option fits a hand expected to take ``tricks``: a call (a bid or a
    super call) of at most ``margin`` tricks more, an estimate within ``margin`` tricks either
    way. Dash Calls, With and Pass name no number and always fit."""
    if R.FIRST_BID <= action <= R.LAST_BID:
        return (action - R.FIRST_BID) // 5 + 4 <= tricks + margin
    if R.ESTIMATE_BASE <= action < R.PLAY_BASE:
        return abs(action - R.ESTIMATE_BASE - tricks) <= margin
    return True


def candidates(
    logits: Sequence[float] | npt.NDArray[Any], legal: Sequence[int], tricks: float, by: Pruning
) -> Options:
    """The options to try among ``legal`` (bidding action ids), from the player's move
    ``logits`` (indexed by action id) and its expected ``tricks``. Its favourite first (the
    lowest id on a tie, as it plays), then by its chances: the options that fit the hand
    (``near``) until they hold ``coverage`` of those options' chances, at least ``min_options``
    and at most ``max_options``."""
    scores = np.array([logits[a] for a in legal], np.float64)
    chances = np.exp(scores - scores.max())
    chances /= chances.sum()
    order = sorted(range(len(legal)), key=lambda i: (-scores[i], legal[i]))
    fits = [
        i for place, i in enumerate(order) if place == 0 or near(legal[i], tricks, by.hand_margin)
    ]
    wanted = by.coverage * sum(chances[i] for i in fits)
    chosen: list[int] = []
    held = 0.0
    for i in fits:
        if len(chosen) >= by.max_options or (len(chosen) >= by.min_options and held >= wanted):
            break
        chosen.append(i)
        held += chances[i]
    place_of = {i: place for place, i in enumerate(order)}
    return Options(
        [legal[i] for i in chosen],
        [float(chances[i]) for i in chosen],
        [place_of[i] for i in chosen],
    )


# The decisions: real rounds, replayed up to each bidding decision.


class Rounds(NamedTuple):
    """Rounds as they were dealt, with their bidding as it was played (a leading round axis)."""

    round_no: jax.Array  # [Rn]
    hands: jax.Array  # [Rn, 4, 52] bool: the deal
    fixed_trump: jax.Array  # [Rn]: -1 in an open round
    carry: jax.Array  # [Rn]
    scores: jax.Array  # [Rn, 4]: the Bola's scores before the round
    moves: jax.Array  # [Rn, MOVES]: the round's bidding moves in order, padded
    made: jax.Array  # [Rn]: how many of them are real


class Batch(NamedTuple):
    """A batch of bidding decisions (a leading decision axis): the real round at each decision
    (the search reads only the deciding seat's hand and the public table), the deciding seat,
    the round's earlier bidding choices by the other seats, and the seat's view."""

    round: R.Round
    me: jax.Array  # [N]
    bidding: G.Bidding  # [N, H]
    x: jax.Array  # [N, SIZE]


def trace(rounds: Rounds) -> tuple[R.Round, jax.Array]:
    """Each round's state before each of its bidding moves ``[Rn, MOVES]`` and the view of the
    seat about to make it ``[Rn, MOVES, SIZE]``. Past a round's last bidding move both repeat
    the state the bidding ended in, which nothing reads."""

    def one(
        round_no: jax.Array,
        hands: jax.Array,
        fixed_trump: jax.Array,
        carry: jax.Array,
        scores: jax.Array,
        moves: jax.Array,
        made: jax.Array,
    ) -> tuple[R.Round, jax.Array]:
        def move(s: R.Round, at: tuple[jax.Array, jax.Array]):
            index, action = at
            stepped = R.step(s, action)
            real = index < made
            after = jax.tree.map(lambda new, old: jnp.where(real, new, old), stepped, s)
            return after, (s, encode(s, s.cur))

        # jaxenv's annotations are loose (an "int" that is a traced array), kept as Essam's.
        start = R.init_round(round_no, hands, fixed_trump, carry, scores)  # pyright: ignore[reportArgumentType]
        _, before = jax.lax.scan(move, start, (jnp.arange(moves.shape[0]), moves))
        return before

    return jax.vmap(one)(*rounds)


def positions(rounds: Rounds, where: jax.Array, at: jax.Array, history: jax.Array) -> Batch:
    """The decisions at bidding move ``at[n]`` of round ``where[n]``. ``history`` ``[N, H]``
    lists, for each, the places of that round's earlier bidding choices by the other seats (the
    moves that had more than one legal option), -1 padded."""
    states, views = trace(rounds)
    r = jax.tree.map(lambda a: a[where, at], states)
    rows = jnp.maximum(history, 0)
    seen = views[where[:, None], rows]  # [N, H, SIZE]
    bidding = G.Bidding(
        seen.at[:, :, :52].set(0.0),  # the hand cleared: each imagined deal writes its own
        jnp.where(history >= 0, states.cur[where[:, None], rows], -1),
        seen[:, :, -R.NUM_ACTIONS :][:, :, :BID_ACTIONS] > 0.5,  # a view ends in its legal moves
        rounds.moves[where[:, None], rows],
        history >= 0,
    )
    return Batch(r, r.cur, bidding, views[where, at])


# 2. The deals.


def imagine(
    player: G.Player,
    settings: Settings,
    key: jax.Array,
    rounds: R.Round,
    me: jax.Array,
    bidding: G.Bidding,
) -> tuple[jax.Array, jax.Array]:
    """``[N, S, 4, 52]``: each decision's imagined deals (``gpu_search.imagine``: the deciding
    seat's own hand in every one), all the stages' together and in a random order; and each
    deal's key ``[N, S, 2]`` for its play-outs' draws."""
    k_deal, k_order, k_play = jax.random.split(key, 3)
    total = sum(settings.deals)
    search = G.Settings(
        samples=total,
        oversample=settings.oversample,
        redeals=settings.redeals,
        floor=settings.floor,
        guess_temperature=settings.guess_temperature,
        leader_weight=settings.leader_weight,
    )
    hands = G.imagine_all(player, search, k_deal, rounds, me, bidding)
    n = hands.shape[0]
    # The kept deals come best-fitting first: shuffled, every stage gets a fair share.
    order = jax.vmap(lambda k: jax.random.permutation(k, total))(jax.random.split(k_order, n))
    hands = jnp.take_along_axis(hands, order[:, :, None, None], axis=1)
    return hands, jax.random.split(k_play, n * total).reshape(n, total, 2)


# 3. The play-outs and 4. their scores.


def _select(cond: jax.Array, a: R.Round, b: R.Round) -> R.Round:
    def pick(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.where(cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim)), x, y)

    return jax.tree.map(pick, a, b)


def bid_moves(
    player: G.Player,
    rounds: R.Round,
    me: jax.Array,
    keys: jax.Array,
    turn: jax.Array,
    temperature: float,
) -> jax.Array:
    """``[B]``: the next bidding move of each of a batch of rounds in bidding. Seat ``me``
    ``[B]`` makes the player's top legal move, as at the table; any other seat's move is drawn
    from the player's move chances at ``temperature`` (its top move at 0), by Gumbel noise from
    the game's key ``[B, 2]`` and ``turn``, the move's number in the play-out, so games with
    the same key draw alike."""
    x = jax.vmap(encode)(rounds, rounds.cur)
    legal = x[:, -R.NUM_ACTIONS :][:, :BID_ACTIONS] > 0.5  # a view ends in its legal moves
    scores = jnp.where(legal, G.move_logits(player, x)[:, :BID_ACTIONS], -jnp.inf)
    top = scores.argmax(-1)
    if temperature <= 0:
        return top

    def noise(k: jax.Array) -> jax.Array:
        return jax.random.gumbel(jax.random.fold_in(k, turn), (BID_ACTIONS,))

    drawn = (scores / temperature + jax.vmap(noise)(keys)).argmax(-1)
    return jnp.where(rounds.cur == me, top, drawn)


def finish(
    player: G.Player,
    rounds: R.Round,
    me: jax.Array,
    option: jax.Array,
    keys: jax.Array,
    temperature: float,
) -> R.Round:
    """Batched rounds at a bidding decision of seat ``me``, each with all four hands: make
    ``option``, then finish the bidding move by move (``bid_moves``). Returns the rounds in
    card play, or done where the round was skipped."""
    r = jax.vmap(R.step)(rounds, option)

    def going(c: tuple[R.Round, jax.Array]) -> jax.Array:
        r, turn = c
        return (r.phase < R.PLAY).any() & (turn < BIDDING_LIMIT)

    def move(c: tuple[R.Round, jax.Array]) -> tuple[R.Round, jax.Array]:
        r, turn = c
        action = bid_moves(player, r, me, keys, turn, temperature)
        return _select(r.phase < R.PLAY, jax.vmap(R.step)(r, action), r), turn + 1

    r, _ = jax.lax.while_loop(going, move, (r, jnp.zeros((), I32)))
    return r


def evaluate(
    player: G.Player,
    settings: Settings,
    rounds: R.Round,
    me: jax.Array,
    hands: jax.Array,
    keys: jax.Array,
    which: jax.Array,
    options: jax.Array,
) -> jax.Array:
    """``[P, S]``: what bidding option ``options[p]`` of decision ``which[p]`` is worth on each
    of that decision's deals: the option made, the bidding finished (``finish``), the round
    played out by the player's top cards and scored by round points. ``rounds`` and ``me`` have
    one entry per decision; ``hands`` ``[N, S, 4, 52]`` and ``keys`` ``[N, S, 2]`` its deals."""
    p, s = options.shape[0], hands.shape[1]

    def spread(x: jax.Array) -> jax.Array:  # [N, ...] -> [P * S, ...]
        wide = jnp.broadcast_to(x[which][:, None], (p, s, *x.shape[1:]))
        return wide.reshape(p * s, *x.shape[1:])

    start = jax.tree.map(spread, rounds)._replace(hands=hands[which].reshape(p * s, 4, 52))
    bid = finish(
        player,
        start,
        spread(me),
        jnp.repeat(options, s),
        keys[which].reshape(p * s, 2),
        settings.opponent_temperature,
    )
    done = G.play_outs(player, bid, jnp.full(p * s, -1, I32))
    result = done.result.reshape(p, s, 4)
    return G.values(result, rounds.scores[which], me[which], settings.leader_weight)


# 5. The narrowing.


class Verdicts(NamedTuple):
    """What the teacher found for a batch of decisions, per option slot (slot 0 is the player's
    favourite; padded slots and decisions with nothing to compare hold zeros)."""

    value: Floats  # [N, K]: the option's mean value over the deals it played
    deals: Ints  # [N, K]: how many deals it played
    margin: Floats  # [N, K]: its mean lead over the favourite on the deals it played
    error: Floats  # [N, K]: that lead's standard error
    opening: Floats  # [N, K]: its mean lead on the first stage's deals, which every option plays
    pick: Ints  # [N]: the slot the teacher picks


def verdicts(played: Floats, first: int) -> Verdicts:
    """The verdicts from each option's value on each deal ``[N, K, D]``, NaN where the option
    did not play; the first stage is the first ``first`` deals. An option plays the deals up to
    where it was dropped; the favourite (slot 0) plays them all, as does the last alternative.
    The pick is the better of those two over every deal, the favourite on a tie.

    ``margin`` uses every deal an option played, so it is the best single estimate of its lead,
    but options were kept or dropped on the same deals, which flatters the ones kept;
    ``opening`` is noisier and free of that: no option had been dropped yet."""
    done = ~np.isnan(played)
    deals = done.sum(-1)
    count = np.maximum(deals, 1)
    value = np.where(done, played, 0.0).sum(-1) / count
    lead = np.where(done, played - played[:, :1], 0.0)
    margin = lead.sum(-1) / count
    spread = np.where(done, (lead - margin[..., None]) ** 2, 0.0).sum(-1) / np.maximum(deals - 1, 1)
    error = np.sqrt(spread / count)
    opening = lead[..., :first].sum(-1) / np.maximum(done[..., :first].sum(-1), 1)
    whole = deals == played.shape[-1]
    pick = np.where(whole, value, -np.inf).argmax(-1)
    pick = np.where(whole.any(-1), pick, 0)
    return Verdicts(
        value.astype(np.float32),
        deals.astype(np.int32),
        margin.astype(np.float32),
        error.astype(np.float32),
        opening.astype(np.float32),
        pick.astype(np.int32),
    )


def narrow(alive: list[int], means: Sequence[float], keep: int) -> list[int]:
    """The option slots that go on to the next stage: the favourite (the first of ``alive``) and
    the better half of the alternatives by their ``means`` so far (one per alive slot), at most
    ``keep`` of them and at least one; the more favoured slot on a tie."""
    rest = alive[1:]
    if len(rest) < 2:
        return alive
    count = max(1, min((len(rest) + 1) // 2, keep))
    best = sorted(range(len(rest)), key=lambda i: (-means[i + 1], rest[i]))[:count]
    return [alive[0], *sorted(rest[i] for i in best)]


class Teacher:
    """The bid teacher for one player and one batch size: compiled once, then ``look`` at a
    batch of decisions, choose their options (``candidates``) and ``teach``."""

    def __init__(self, bidder: Bidder, settings: Settings) -> None:
        self.settings = settings
        self._positions = jax.jit(positions)
        self._opinions = jax.jit(partial(opinions, bidder))
        self._imagine = jax.jit(partial(imagine, bidder.player, settings))
        self._evaluate = jax.jit(partial(evaluate, bidder.player, settings))

    def look(
        self, rounds: Rounds, where: Ints, at: Ints, history: Ints
    ) -> tuple[Batch, Floats, Floats]:
        """The decisions (``positions``) with the player's move logits ``[N, 120]`` and
        expected tricks ``[N]`` at each."""
        batch = self._positions(rounds, jnp.asarray(where), jnp.asarray(at), jnp.asarray(history))
        logits, tricks = self._opinions(batch.x)
        return batch, np.asarray(logits), np.asarray(tricks)

    def teach(self, key: jax.Array, batch: Batch, options: Ints) -> Verdicts:
        """Search every decision of ``batch``. ``options`` ``[N, K]`` are each decision's
        options to try, as action ids, its favourite first, -1 padded; a decision with fewer
        than two is not searched."""
        n, slots = options.shape
        stages = self.settings.deals
        hands, keys = self._imagine(key, batch.round, batch.me, batch.bidding)
        played = np.full((n, slots, sum(stages)), np.nan, np.float32)
        alive = [[k for k in range(slots) if options[d, k] >= 0] for d in range(n)]
        alive = [slots_of if len(slots_of) > 1 else [] for slots_of in alive]
        start = 0
        for stage, count in enumerate(stages):
            pairs = [(d, k) for d in range(n) for k in alive[d]]
            if not pairs:
                break
            which = np.array([d for d, _ in pairs], np.int32)
            slot = np.array([k for _, k in pairs], np.int32)
            # One compiled size per whole number of options a decision; the padding repeats
            # the first pair, a real option, so every play-out is a legal game.
            size = max(2, -(-len(pairs) // n)) * n
            w = np.full(size, which[0], np.int32)
            o = np.full(size, options[which[0], slot[0]], np.int32)
            w[: len(pairs)], o[: len(pairs)] = which, options[which, slot]
            end = start + count
            out = self._evaluate(
                batch.round,
                batch.me,
                hands[:, start:end],
                keys[:, start:end],
                jnp.asarray(w),
                jnp.asarray(o),
            )
            played[which, slot, start:end] = np.asarray(out)[: len(pairs)]
            start = end
            if stage + 1 < len(stages):
                keep = 2 ** (len(stages) - 2 - stage)  # alternatives the next stage may hold
                for d in range(n):
                    if len(alive[d]) > 2:
                        means = played[d, alive[d], :start].mean(axis=1).tolist()
                        alive[d] = narrow(alive[d], means, keep)
        return verdicts(played, stages[0])
