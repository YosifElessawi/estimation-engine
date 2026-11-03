"""Training games (SPEC §6 M4): the current nets against themselves, recent snapshots,
HeuristicBot and a lighter PIMC, played in lockstep for batched inference.

A chunk of games advances together: each game runs its scripted players until a net has to
decide, then every pending net decision of the chunk goes through one forward pass per net. The
current nets ("the learner") explore and record ``(features, action)`` for each decision with a
choice; when the round ends every recorded decision gets that round's reward. Results depend
only on the chunk, never on how chunks are spread over workers.

Learner bidding samples a softmax of its values (``bid_temperature``): with greedy bidding a
pilot's policy flipped between never opening the auction and opening almost every round,
because whichever choice it was avoiding got too few samples to be valued well. Card play is
greedy. Both also take a uniformly random legal action with a small probability (epsilon).

Each recorded decision also keeps who held each unseen card at that moment (``guesses``), the
guess head's training target, read from the real deal: the trainer knows it, the nets never see
it.

Trainer code: it holds real Bolas, but every player and net sees only the acting seat's own
Observation (through the state encoder), exactly as in the arena.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from estimation.bots import Bot, HeuristicBot, NoisyBot, PIMCBot, PIMCConfig
from estimation.bots.heuristic import HeuristicConfig
from estimation.engine import Phase, RulesConfig
from estimation.engine.actions import PLAY_BASE

from .config import TrainConfig
from .encoder import NUM_FEATURES, encode
from .games import RewardedBola
from .guesses import guess_labels
from .memory import peak_rss_mb
from .networks import BiddingNet, CardPlayNet
from .standings import RewardConfig, first_place_shares

LEARNER = "learner"
SNAPSHOT = "snapshot"
HEURISTIC = "heuristic"
PIMC = "pimc"

StateDict = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class SeatPlan:
    """Who plays one seat: the learner, snapshot ``snapshot``, HeuristicBot with ``noise`` or
    PIMC with ``samples`` deals."""

    kind: str
    snapshot: int = -1
    noise: float = 0.0
    samples: int = 0


@dataclass(frozen=True, slots=True)
class GamePlan:
    index: int
    kind: str  # "self", "heuristic", "pimc", or "eval" (greedy, never trained on)
    deal_seed: int
    seed: int  # seeds the scripted bots and the learner's exploration
    seats: tuple[SeatPlan, SeatPlan, SeatPlan, SeatPlan]


@dataclass(frozen=True, slots=True)
class NetWeights:
    """A pair of nets as CPU state dicts; ``bidding`` is None before it is trained."""

    hidden: int
    card: StateDict
    bidding: StateDict | None


@dataclass(frozen=True, slots=True)
class ChunkTask:
    plans: tuple[GamePlan, ...]
    learner: NetWeights
    snapshots: dict[int, NetWeights]
    epsilon_bid: float
    epsilon_card: float
    bid_temperature: float  # learner bidding samples a softmax of its values; 0 is greedy
    record_bid: bool
    record_card: bool
    rules: RulesConfig
    reward: RewardConfig


@dataclass(slots=True)
class Samples:
    """Recorded decisions: encoded Observation, action id (card id for card play), the reward
    of the round it was made in and the guess labels (``guesses.guess_labels``)."""

    features: npt.NDArray[np.float16]
    actions: npt.NDArray[np.int16]
    rewards: npt.NDArray[np.float32]
    guesses: npt.NDArray[np.int8]

    def __len__(self) -> int:
        return len(self.actions)


def empty_samples() -> Samples:
    return Samples(
        np.zeros((0, NUM_FEATURES), np.float16),
        np.zeros(0, np.int16),
        np.zeros(0, np.float32),
        np.zeros((0, 52), np.int8),
    )


def concat_samples(parts: Sequence[Samples]) -> Samples:
    parts = [p for p in parts if len(p)]
    if not parts:
        return empty_samples()
    return Samples(
        np.concatenate([p.features for p in parts]),
        np.concatenate([p.actions for p in parts]),
        np.concatenate([p.rewards for p in parts]),
        np.concatenate([p.guesses for p in parts]),
    )


@dataclass(frozen=True, slots=True)
class GameStats:
    """How the learner did in one game: its seats' mean final score minus the mean of the other
    seats, and its seats' share of first place."""

    index: int
    kind: str
    learner_seats: int
    rsa: float
    first: float
    rounds: int
    skipped: int  # rounds nobody bid in


@dataclass(slots=True)
class ChunkResult:
    bid: Samples
    card: Samples
    games: list[GameStats] = field(default_factory=lambda: list[GameStats]())
    peak_rss_mb: float = 0.0  # the worker process's peak memory so far


@dataclass(slots=True)
class _Nets:
    card: CardPlayNet
    bidding: BiddingNet | None


def build_nets(weights: NetWeights) -> _Nets:
    card = CardPlayNet(weights.hidden)
    card.load_state_dict(weights.card)
    card.eval()
    bidding = None
    if weights.bidding is not None:
        bidding = BiddingNet(weights.hidden)
        bidding.load_state_dict(weights.bidding)
        bidding.eval()
    return _Nets(card, bidding)


def _scripted(plan: SeatPlan, seed: int, rules: RulesConfig) -> Bot:
    heuristic = HeuristicConfig(super_call_min=rules.super_call_min)
    if plan.kind == HEURISTIC:
        bot: Bot = HeuristicBot(heuristic)
        return NoisyBot(bot, noise=plan.noise, seed=seed) if plan.noise else bot
    if plan.kind == PIMC:
        return PIMCBot(seed, PIMCConfig(samples=plan.samples, heuristic=heuristic, rules=rules))
    raise ValueError(f"seat kind {plan.kind!r} is not a scripted bot")


type _Open = tuple[bool, npt.NDArray[np.float32], int, npt.NDArray[np.int8]]
type _Row = tuple[npt.NDArray[np.float32], int, float, npt.NDArray[np.int8]]


class _Game:
    """One game of a chunk and the learner decisions waiting for their round's reward."""

    __slots__ = ("bots", "game", "net_of", "open", "out_bid", "out_card", "plan", "rng")

    def __init__(self, plan: GamePlan, task: ChunkTask) -> None:
        self.plan = plan
        self.game = RewardedBola(plan.deal_seed, task.rules, task.reward)
        self.rng = random.Random(plan.seed)
        self.bots: list[Bot | None] = []
        self.net_of: list[int | None] = []  # -1 for the learner, else the snapshot id
        for seat, seat_plan in enumerate(plan.seats):
            if seat_plan.kind == LEARNER:
                self.bots.append(None)
                self.net_of.append(-1)
            elif seat_plan.kind == SNAPSHOT:
                self.bots.append(None)
                self.net_of.append(seat_plan.snapshot)
            else:
                self.bots.append(_scripted(seat_plan, plan.seed * 4 + seat + 1, task.rules))
                self.net_of.append(None)
        # Per seat, this round's learner decisions: (is card play, features, action, guesses).
        self.open: list[list[_Open]] = [[], [], [], []]
        self.out_bid: list[_Row] = []
        self.out_card: list[_Row] = []

    def step(self, action: int) -> None:
        rewards = self.game.step(action)
        if rewards is None:
            return
        for seat in range(4):
            for is_card, features, act, guesses in self.open[seat]:
                row = (features, act, rewards[seat], guesses)
                (self.out_card if is_card else self.out_bid).append(row)
            self.open[seat].clear()


@dataclass(slots=True)
class _Request:
    game: _Game
    seat: int
    features: npt.NDArray[np.float32]
    legal: list[int]
    guesses: npt.NDArray[np.int8] | None  # the learner's guess labels; None for snapshots


def play_chunk(task: ChunkTask) -> ChunkResult:
    """Play every game of ``task`` to the end and return the learner's recorded decisions."""
    learner = build_nets(task.learner)
    snapshots = {key: build_nets(weights) for key, weights in task.snapshots.items()}
    games = [_Game(plan, task) for plan in task.plans]
    rules = task.rules
    active = list(games)
    with torch.inference_mode():
        while active:
            card_requests: dict[int, list[_Request]] = {}
            bid_requests: dict[int, list[_Request]] = {}
            for game in active:
                request = _advance(game, rules)
                if request is None:
                    continue
                key = game.net_of[request.seat]
                if key is None:
                    raise RuntimeError("only net seats make requests")
                is_card = game.game.bola.phase is Phase.PLAY
                (card_requests if is_card else bid_requests).setdefault(key, []).append(request)
            for key, requests in card_requests.items():
                nets = learner if key == -1 else snapshots[key]
                values = _card_values(nets, requests)
                for request, row in zip(requests, values, strict=True):
                    _apply(request, row, True, key == -1, task)
            for key, requests in bid_requests.items():
                nets = learner if key == -1 else snapshots[key]
                values = _bid_values(nets, requests)
                for request, row in zip(requests, values, strict=True):
                    _apply(request, row, False, key == -1, task)
            active = [game for game in active if not game.game.bola.is_over]
    return ChunkResult(
        bid=_samples([game.out_bid for game in games]),
        card=_samples([game.out_card for game in games]),
        games=[_stats(game) for game in games],
        peak_rss_mb=peak_rss_mb(),
    )


def _advance(game: _Game, rules: RulesConfig) -> _Request | None:
    """Run scripted players and forced moves until a net seat has a real choice, or the game
    ends. Returns that seat's request."""
    bola = game.game.bola
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a Bola that is not over always has a player to act")
        legal = bola.legal_actions()
        bot = game.bots[seat]
        if bot is not None:
            game.step(bot.act(bola.observation(seat), legal))
            continue
        if len(legal) == 1:
            game.step(legal[0])
            continue
        features = encode(bola.observation(seat), rules)
        guesses = (
            guess_labels(features, bola.round.hands, seat) if game.net_of[seat] == -1 else None
        )
        return _Request(game, seat, features, legal, guesses)
    return None


def _card_values(nets: _Nets, requests: list[_Request]) -> npt.NDArray[np.float32]:
    x = torch.as_tensor(np.stack([r.features for r in requests]))
    _, advantage = nets.card(x)
    return advantage.numpy()


def _bid_values(nets: _Nets, requests: list[_Request]) -> npt.NDArray[np.float32]:
    if nets.bidding is None:
        raise ValueError("a net seat needs a BiddingNet to bid")
    x = torch.as_tensor(np.stack([r.features for r in requests]))
    return nets.bidding(x).numpy()


def _apply(
    request: _Request,
    values: npt.NDArray[np.float32],
    is_card: bool,
    is_learner: bool,
    task: ChunkTask,
) -> None:
    legal = request.legal
    offset = PLAY_BASE if is_card else 0
    game = request.game
    if is_learner:
        epsilon = task.epsilon_card if is_card else task.epsilon_bid
        explore = game.rng.random() < epsilon
        if explore:
            action = game.rng.choice(legal)
        elif not is_card and task.bid_temperature > 0:
            action = _softmax_choice(legal, values, task.bid_temperature, game.rng.random())
        else:
            action = max(legal, key=lambda a: values[a - offset])
        if (task.record_card if is_card else task.record_bid) and request.guesses is not None:
            game.open[request.seat].append(
                (is_card, request.features, action - offset, request.guesses)
            )
    else:
        action = max(legal, key=lambda a: values[a - offset])
    game.step(action)


def _softmax_choice(
    legal: list[int], values: npt.NDArray[np.float32], temperature: float, u: float
) -> int:
    """The legal action drawn with probability proportional to ``exp(value / temperature)``,
    by inverting the cumulative distribution at ``u`` (uniform in [0, 1))."""
    q = np.array([float(values[a]) for a in legal], np.float64) / temperature
    cumulative = np.cumsum(np.exp(q - q.max()))
    target = u * float(cumulative[-1])
    index = next(
        (k for k, total in enumerate(cumulative.tolist()) if total > target), len(legal) - 1
    )
    return legal[index]


def _samples(per_game: list[list[_Row]]) -> Samples:
    rows = [row for game in per_game for row in game]
    if not rows:
        return empty_samples()
    return Samples(
        np.stack([features for features, _, _, _ in rows]).astype(np.float16),
        np.array([action for _, action, _, _ in rows], np.int16),
        np.array([reward for _, _, reward, _ in rows], np.float32),
        np.stack([guesses for _, _, _, guesses in rows]),
    )


def _stats(game: _Game) -> GameStats:
    bola = game.game.bola
    scores = bola.scores
    seats = [s for s in range(4) if game.plan.seats[s].kind == LEARNER]
    shares = first_place_shares(scores)
    others = [s for s in range(4) if s not in seats]
    mine = sum(scores[s] for s in seats) / len(seats)
    rest = sum(scores[s] for s in others) / len(others) if others else mine
    return GameStats(
        index=game.plan.index,
        kind=game.plan.kind,
        learner_seats=len(seats),
        rsa=mine - rest,
        first=sum(shares[s] for s in seats) / len(seats),
        rounds=len(bola.history),
        skipped=sum(1 for result in bola.history if result.skipped),
    )


def plan_games(
    config: TrainConfig,
    iteration: int,
    snapshot_ids: Sequence[int],
    seed_of: "SeedFn",
) -> list[GamePlan]:
    """The games of one iteration: exact shares of each opponent kind, in a shuffled order so
    every chunk gets a mix. ``seed_of(*keys)`` derives this iteration's seeds."""
    mix = config.opponents
    total = config.selfplay.games
    n_heuristic = round(mix.heuristic * total)
    n_pimc = round(mix.pimc * total)
    kinds = ["heuristic"] * n_heuristic + ["pimc"] * n_pimc
    kinds += ["self"] * (total - len(kinds))
    rng = random.Random(seed_of(0))
    rng.shuffle(kinds)
    plans: list[GamePlan] = []
    for index, kind in enumerate(kinds):
        game_rng = random.Random(seed_of(1, index))
        seats: list[SeatPlan] = []
        if kind == "self":
            anchor = game_rng.randrange(4)
            for seat in range(4):
                if seat == anchor or not snapshot_ids or game_rng.random() < mix.current_share:
                    seats.append(SeatPlan(LEARNER))
                else:
                    seats.append(SeatPlan(SNAPSHOT, snapshot=game_rng.choice(list(snapshot_ids))))
        else:
            learner_seat = game_rng.randrange(4)
            for seat in range(4):
                if seat == learner_seat:
                    seats.append(SeatPlan(LEARNER))
                elif kind == "heuristic":
                    seats.append(SeatPlan(HEURISTIC, noise=game_rng.choice(mix.heuristic_noise)))
                else:
                    seats.append(SeatPlan(PIMC, samples=game_rng.choice(mix.pimc_samples)))
        plans.append(
            GamePlan(
                index=index,
                kind=kind,
                deal_seed=seed_of(2, index),
                seed=seed_of(3, index),
                seats=(seats[0], seats[1], seats[2], seats[3]),
            )
        )
    return plans


def plan_eval_games(config: TrainConfig, seed_of: "SeedFn") -> list[GamePlan]:
    """Greedy games of the learner against three plain HeuristicBots, kept out of training: the
    log's view of how the nets play the M4 headline lineup."""
    plans: list[GamePlan] = []
    for index in range(config.selfplay.eval_games):
        learner_seat = index % 4
        seats = tuple(SeatPlan(LEARNER if s == learner_seat else HEURISTIC) for s in range(4))
        plans.append(
            GamePlan(
                index=index,
                kind="eval",
                deal_seed=seed_of(4, index),
                seed=seed_of(5, index),
                seats=(seats[0], seats[1], seats[2], seats[3]),
            )
        )
    return plans


def chunk_plans(
    plans: Sequence[GamePlan], size: int, by_kind: bool = False
) -> list[tuple[GamePlan, ...]]:
    """``plans`` cut into chunks of ``size`` games in plan order. With ``by_kind``, each chunk
    holds games of one kind only, so the slow games against PIMC can be started first."""
    groups = [list(plans)]
    if by_kind:
        kinds = sorted({plan.kind for plan in plans})
        groups = [[plan for plan in plans if plan.kind == kind] for kind in kinds]
    return [tuple(group[i : i + size]) for group in groups for i in range(0, len(group), size)]


# Rough seconds per game on one dev-Mac core, to start the slowest chunks first: about 0.04 s of
# rules and scripted play, 0.03 s per net seat, and 0.04 s per deal of each PIMC seat. A game of
# four nets takes about 0.15 s and a game against three PIMC bots of 4-8 deals about 0.75 s.
_GAME_SECONDS = 0.04
_NET_SEAT_SECONDS = 0.03
_PIMC_DEAL_SECONDS = 0.04


def estimated_seconds(plans: Sequence[GamePlan]) -> float:
    """A rough cost of playing ``plans``. It orders work only: no result depends on it."""
    total = 0.0
    for plan in plans:
        total += _GAME_SECONDS
        for seat in plan.seats:
            if seat.kind in (LEARNER, SNAPSHOT):
                total += _NET_SEAT_SECONDS
            elif seat.kind == PIMC:
                total += _PIMC_DEAL_SECONDS * seat.samples
    return total


class SeedFn:
    """Seeds for one iteration: ``SeedFn(run_seed, stage, iteration)(*keys)``."""

    __slots__ = ("prefix",)

    def __init__(self, *prefix: int) -> None:
        self.prefix = prefix

    def __call__(self, *keys: int) -> int:
        return int(np.random.SeedSequence([*self.prefix, *keys]).generate_state(1)[0])
