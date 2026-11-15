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
   one alternative, the finalist, is left to meet the favourite. The favourite always stays, so
   every alternative is measured against the move the player makes today on the deals they
   shared. An alternative more than ``hopeless`` standard errors behind the favourite is dropped
   at once, and with none left the search ends early: the favourite stands.
6. **The exam** (``verdicts``). The deals the finalist plays once it is the only alternative
   left. Nothing was kept or dropped on those deals, so its lead over them is a fair measure of
   it, where a lead over the earlier deals is flattered by the luck that kept it in. The teacher
   picks the finalist only if it led the exam.

Only what the deciding seat may see goes in: its own hand, the public table and bidding, and its
own guesses. The other three hands of the real round are replaced by imagined ones before any
option is tried.
"""

import time
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
    hopeless: float = 3.0  # standard errors behind the favourite at which an option is dropped


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
    if settings.opponent_temperature < 0 or settings.hopeless < 0:
        raise ValueError("the teacher's opponent_temperature and hopeless must be 0 or more")
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
    """What the teacher found for a batch of decisions. Per option slot (slot 0 is the player's
    favourite; padded slots and decisions with nothing to compare hold zeros): how the option
    did on the deals it played. Per decision: the exam, the finalist against the favourite on
    deals none of the dropping was done on."""

    value: Floats  # [N, K]: the option's mean value over the deals it played
    deals: Ints  # [N, K]: how many deals it played
    margin: Floats  # [N, K]: its mean lead over the favourite on the deals it played
    error: Floats  # [N, K]: that lead's standard error
    opening: Floats  # [N, K]: its mean lead on the first stage's deals, which every option plays
    finalist: Ints  # [N]: the alternative that lasted longest (0: the decision was not searched)
    exam: Floats  # [N]: the finalist's mean lead over the favourite on its exam deals
    exam_error: Floats  # [N]: that lead's standard error
    exam_deals: Ints  # [N]: the exam's deals (0: no alternative was ever left alone to sit one)
    pick: Ints  # [N]: the slot the teacher picks


def verdicts(played: Floats, first: int, exam_from: Ints) -> Verdicts:
    """The verdicts from each option's value on each deal ``[N, K, D]``, NaN where the option
    did not play; the first stage is the first ``first`` deals. An option plays the deals up to
    where it was dropped; the favourite (slot 0) plays every deal any option plays.

    The finalist is the alternative that played the most deals (the one further ahead on a
    tie). Its exam is the deals it played from ``exam_from`` ``[N]`` on, the point where it was
    the only alternative left: no option was kept or dropped on those deals, so its lead over
    them is a fair measure, where ``margin`` (every deal an option played) flatters the options
    that were kept on and ``opening`` is fair but short. The pick is the finalist if it sat the
    whole exam, to the last deal, and led it; otherwise the favourite."""
    count, _, total = played.shape
    rows = np.arange(count)
    done = ~np.isnan(played)
    deals = done.sum(-1)
    many = np.maximum(deals, 1)
    value = np.where(done, played, 0.0).sum(-1) / many
    lead = np.where(done, played - played[:, :1], 0.0)
    margin = lead.sum(-1) / many
    spread = np.where(done, (lead - margin[..., None]) ** 2, 0.0).sum(-1) / np.maximum(deals - 1, 1)
    error = np.sqrt(spread / many)
    opening = lead[..., :first].sum(-1) / np.maximum(done[..., :first].sum(-1), 1)
    ahead = margin[:, 1:] - margin[:, 1:].min(-1, keepdims=True)
    longest = deals[:, 1:] + 0.5 * ahead / (ahead.max(-1, keepdims=True) + 1e-9)
    finalist = np.where(deals[:, 1:].any(-1), 1 + longest.argmax(-1), 0)
    sat = deals[rows, finalist]
    at = np.arange(total)
    window = (at >= exam_from[:, None]) & (at < sat[:, None]) & (finalist > 0)[:, None]
    exam_deals = window.sum(-1)
    exam_lead = np.where(window, lead[rows, finalist], 0.0)
    exam = exam_lead.sum(-1) / np.maximum(exam_deals, 1)
    exam_spread = np.where(window, (exam_lead - exam[:, None]) ** 2, 0.0).sum(-1)
    exam_error = np.sqrt(exam_spread / np.maximum(exam_deals - 1, 1) / np.maximum(exam_deals, 1))
    pick = np.where((sat == total) & (exam_deals > 0) & (exam > 0), finalist, 0)
    return Verdicts(
        value.astype(np.float32),
        deals.astype(np.int32),
        margin.astype(np.float32),
        error.astype(np.float32),
        opening.astype(np.float32),
        finalist.astype(np.int32),
        exam.astype(np.float32),
        exam_error.astype(np.float32),
        exam_deals.astype(np.int32),
        pick.astype(np.int32),
    )


def survivors(leads: Floats, keep: int, hopeless: float) -> list[int]:
    """Which of a decision's alternatives go on to the next stage, from each one's lead over the
    favourite on every deal so far (``[A, deals]``, the more favoured alternative first): the
    better half by mean lead, at most ``keep`` of them and at least one, less any that trails
    the favourite by more than ``hopeless`` standard errors (0: none ever does). The more
    favoured alternative wins a tie. None left ends the search: the favourite stands."""
    count, deals = leads.shape
    mean = leads.mean(axis=1)
    error = leads.std(axis=1, ddof=1) / np.sqrt(deals)
    half = max(1, min((count + 1) // 2, keep))
    best = sorted(range(count), key=lambda i: (-mean[i], i))[:half]
    return sorted(i for i in best if hopeless <= 0 or mean[i] + hopeless * error[i] >= 0)


class Teacher:
    """The bid teacher for one player and one batch size: compiled once, then ``look`` at a
    batch of decisions, choose their options (``candidates``) and ``teach``. ``spent`` keeps
    the seconds each part of the work has taken so far and ``play_outs`` their count."""

    def __init__(self, bidder: Bidder, settings: Settings) -> None:
        self.settings = settings
        self.spent: dict[str, float] = {}
        self.play_outs = 0
        self._positions = jax.jit(positions)
        self._opinions = jax.jit(partial(opinions, bidder))
        self._imagine = jax.jit(partial(imagine, bidder.player, settings))
        self._evaluate = jax.jit(partial(evaluate, bidder.player, settings))

    def _clock(self, name: str, started: float) -> float:
        now = time.perf_counter()
        self.spent[name] = self.spent.get(name, 0.0) + now - started
        return now

    def look(
        self, rounds: Rounds, where: Ints, at: Ints, history: Ints
    ) -> tuple[Batch, Floats, Floats]:
        """The decisions (``positions``) with the player's move logits ``[N, 120]`` and
        expected tricks ``[N]`` at each."""
        started = time.perf_counter()
        batch = self._positions(rounds, jnp.asarray(where), jnp.asarray(at), jnp.asarray(history))
        logits, tricks = self._opinions(batch.x)
        found = batch, np.asarray(logits), np.asarray(tricks)
        self._clock("decisions", started)
        return found

    def teach(self, key: jax.Array, batch: Batch, options: Ints) -> Verdicts:
        """Search every decision of ``batch``. ``options`` ``[N, K]`` are each decision's
        options to try, as action ids, its favourite first, -1 padded; a decision with fewer
        than two is not searched."""
        n, slots = options.shape
        stages = self.settings.deals
        total = sum(stages)
        clock = time.perf_counter()
        hands, keys = self._imagine(key, batch.round, batch.me, batch.bidding)
        hands.block_until_ready()
        clock = self._clock("deals", clock)
        played = np.full((n, slots, total), np.nan, np.float32)
        alive = [[k for k in range(slots) if options[d, k] >= 0] for d in range(n)]
        alive = [slots_of if len(slots_of) > 1 else [] for slots_of in alive]
        # A decision's exam starts once one alternative is left: at once with two options.
        exam_from = np.array([0 if len(slots_of) == 2 else total for slots_of in alive], np.int32)
        step = max(n // 4, 1)  # play-out batches come in quarter-batch sizes, each compiled once
        start = 0
        for stage, count in enumerate(stages):
            pairs = [(d, k) for d in range(n) for k in alive[d]]
            if not pairs:
                break
            which = np.array([d for d, _ in pairs], np.int32)
            slot = np.array([k for _, k in pairs], np.int32)
            # The padding repeats the first pair, a real option: every play-out is a legal game.
            size = -(-len(pairs) // step) * step
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
            self.play_outs += len(pairs) * count
            clock = self._clock(f"stage {stage + 1}", clock)
            start = end
            if stage + 1 == len(stages):
                break
            keep = 2 ** (len(stages) - 2 - stage)  # alternatives the next stage may hold
            for d in range(n):
                slots_of = alive[d]
                if not slots_of:
                    continue
                leads = played[d, slots_of[1:], :start] - played[d, slots_of[0], :start]
                on = [slots_of[1 + i] for i in survivors(leads, keep, self.settings.hopeless)]
                if len(on) == 1 and len(slots_of) > 2:
                    exam_from[d] = start
                alive[d] = [slots_of[0], *on] if on else []
        return verdicts(played, stages[0], exam_from)
