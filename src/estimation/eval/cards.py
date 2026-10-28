"""The card-play evaluator: how many points each of a bot's card plays cost against the best card.

Card plays come from saved games (``--from`` a run folder or games file, or ``--web`` for the web
table's games, where ``--bot human`` scores the player) or from new games (``--lineup``), as the
bidding evaluator's decisions do. A play counts when the player had a real choice: two legal cards
that do not touch (touching cards take the same tricks, ``representatives``). A sample spread
evenly over the stage of the round (tricks 1-4, 5-9 or 10-13) and where the player stood (still
needing tricks, holding exactly its estimate, or already over it) is judged by a deeper search
than the bots' own:

- the judge deals the hidden cards again ``deals`` times from the player's own Observation, as
  PIMC samples them: hand sizes, cards played, voids shown and the void announcements, leaning
  toward hands that fit each player's estimate. It never sees the real hidden cards;
- each legal card (one per run of touching cards) is played on every deal and the rest of the
  round played out with every seat searching as ``card_play`` (PIMC with 8 deals by default), on
  the same deals and seeds for every card, so the differences come from the cards;
- a card is worth the player's round score minus the others', those ahead in the game counted
  double, as in the bidding evaluator. Its points lost are the best card's mean value minus its
  own.

Means in the report are weighted back to how often each situation comes up in the games, so the
even sample does not skew them. Labels follow the game review (SPEC §6 M6), thresholds as
options: an Inaccuracy (?) loses 2 or more points, a Blunder (??) 5 or more, and a Brilliant
Sabotage (!!) is the best card when it also cuts the game leader's expected round score by 5 or
more against the next-best card.

Example: ``uv run python -m estimation.eval.cards --bot yosif
--from runs/ladder-v2/games/pimc--yosif.jsonl.gz --cards 300 --workers 4 --out runs/eval/cards``.
"""

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.arena.games import GameRecord
from estimation.bots import make_bot
from estimation.bots.knowledge import read_knowledge
from estimation.bots.pimc import PIMCBot, PIMCConfig, representatives
from estimation.bots.scoring import standings_weights
from estimation.engine import (
    Bola,
    IllegalActionError,
    Observation,
    Phase,
    Round,
    RulesConfig,
    play_action,
)
from estimation.engine.actions import card_of

from .bidding import points_lost
from .report import TRUMPS, card_text, hand_text
from .sources import add_source_arguments, games_from
from .work import Progress, print_progress, run_tasks

STAGES = ("tricks 1-4", "tricks 5-9", "tricks 10-13")
NEEDS = ("needs tricks", "has enough", "over")
CELLS = tuple((stage, need) for stage in STAGES for need in NEEDS)
LABELS = ("best", "close", "inaccuracy", "blunder")
DEFAULT_CARDS = 300
DEFAULT_DEALS = 32  # Yosif's pick of the deep play-out judge, 2026-10-27 [Claude]
DEFAULT_CARD_PLAY = "pimc:samples=8"  # as the bidding evaluator's play-outs
WORST = 20


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The game review's label thresholds, in points (SPEC §6 M6)."""

    inaccuracy: float = 2.0
    blunder: float = 5.0
    sabotage: float = 5.0


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True, slots=True)
class CardPlay:
    """One card play with a real choice. ``obs`` is all the judge reads; ``game`` and ``step``
    only say where it came from (None and -1 for a position built by hand)."""

    game: GameRecord | None
    step: int
    obs: Observation
    legal: tuple[int, ...]  # the legal cards, 0-51
    chosen: int  # the card played

    @property
    def seat(self) -> int:
        return self.obs.seat

    @property
    def stage(self) -> str:
        return stage_of(self.obs)

    @property
    def need(self) -> str:
        return need_of(self.obs)


@dataclass(frozen=True, slots=True)
class Judged:
    play: CardPlay
    weight: float  # how many of the bot's card plays this one stands for
    values: dict[int, tuple[float, float]]  # each card tried: mean value and standard error
    best: int
    lost: float  # the best card's mean value minus the chosen one's, at least 0
    lost_se: float  # the standard error of ``lost``, from the per-deal differences
    leader: int | None  # the game leader (``game_leader``), if there is one
    leader_scores: dict[int, float]  # the leader's mean round score after each card tried


def stage_of(obs: Observation) -> str:
    trick = len(obs.tricks) + 1
    return STAGES[0 if trick <= 4 else 1 if trick <= 9 else 2]


def need_of(obs: Observation) -> str:
    estimate = obs.estimates[obs.seat]
    if estimate is None:
        raise ValueError("every estimate is known during card play")
    won = obs.tricks_won[obs.seat]
    return NEEDS[0 if won < estimate else 1 if won == estimate else 2]


def game_leader(scores: tuple[int, int, int, int], me: int) -> int | None:
    """The other player alone at the top of the game, or None when the top is shared or is
    ``me``: there is then no leader to sabotage [Claude]."""
    top = max(scores)
    leaders = [seat for seat in range(4) if scores[seat] == top]
    return leaders[0] if len(leaders) == 1 and leaders[0] != me else None


def choices(obs: Observation, legal: Sequence[int], chosen: int) -> list[int]:
    """The cards worth trying: one per run of touching legal cards, the chosen card first,
    standing for its own run."""
    return representatives(sorted(legal), read_knowledge(obs).unseen, chosen)


def plays_of(record: GameRecord, rules: RulesConfig, target: str) -> list[CardPlay]:
    """Every card play with a real choice by a seat holding ``target`` (a bot spec, or ``human``
    for the player's seat in a web game)."""
    bola = Bola(record.deal_seed, rules)
    found: list[CardPlay] = []
    for step, action in enumerate(record.actions):
        seat = bola.current_player
        if seat is None:
            raise ValueError("a saved game has more actions than its game had decisions")
        if record.seats[seat] == target and bola.phase is Phase.PLAY:
            legal = [card_of(a) for a in bola.legal_actions()]
            if len(legal) > 1:
                obs = bola.observation(seat)
                chosen = card_of(action)
                if len(choices(obs, legal, chosen)) > 1:
                    found.append(CardPlay(record, step, obs, tuple(legal), chosen))
        bola.step(action)
    return found


def sample_plays(plays: Sequence[CardPlay], n: int, seed: int) -> list[tuple[CardPlay, float]]:
    """Up to ``n`` plays spread evenly over the stage and need cells, the same for a seed, each
    with its weight: how many plays of its cell it stands for."""
    rng = random.Random(seed)
    groups = {cell: [p for p in plays if (p.stage, p.need) == cell] for cell in CELLS}
    sizes = {cell: len(group) for cell, group in groups.items()}
    for group in groups.values():
        rng.shuffle(group)
    picked: list[CardPlay] = []
    while len(picked) < n and any(groups.values()):
        for cell in CELLS:
            if groups[cell] and len(picked) < n:
                picked.append(groups[cell].pop())
    taken = {cell: sum((p.stage, p.need) == cell for p in picked) for cell in CELLS}
    return [(p, sizes[(p.stage, p.need)] / taken[(p.stage, p.need)]) for p in picked]


def redeals(
    obs: Observation, deals: int, seed: int, rules: RulesConfig
) -> list[tuple[int, int, int, int]]:
    """``deals`` deals of the round's four hands as dealt that fit what ``obs.seat`` has seen,
    sampled as PIMC samples them (``PIMCBot.sample_deals``), from its Observation alone."""
    know = read_knowledge(obs)
    sampler = PIMCBot(seed, PIMCConfig(samples=deals, rules=rules))
    found: list[tuple[int, int, int, int]] = []
    for hands in sampler.sample_deals(obs, know):
        dealt = [hands[seat] | know.played_by[seat] for seat in range(4)]
        found.append((dealt[0], dealt[1], dealt[2], dealt[3]))
    return found


def card_playout(
    obs: Observation,
    hands: tuple[int, int, int, int],
    card: int,
    card_play: str,
    seed: int,
    rules: RulesConfig,
) -> tuple[int, int, int, int]:
    """The round scores after ``obs.seat`` plays ``card`` on the deal ``hands``, every seat then
    playing its cards as ``card_play``."""
    log = [action for _, action in obs.log]
    rnd = Round.replay(obs.round_no, hands, obs.fixed_trump, obs.carry, obs.scores, rules, log)
    rnd.step(play_action(card))
    players = [make_bot(card_play, seed * 4 + seat, rules) for seat in range(4)]
    while rnd.phase is not Phase.DONE:
        seat = rnd.current_player
        if seat is None:
            raise RuntimeError("a round that is not done always has a player to act")
        rnd.step(players[seat].act(rnd.observation(seat), rnd.legal_actions()))
    return rnd.result().scores


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def judge(
    play: CardPlay,
    deals: int,
    card_play: str,
    seed: int,
    rules: RulesConfig,
    weight: float = 1.0,
) -> Judged:
    """Every card worth trying in ``play``, played out on the same ``deals`` deals."""
    obs = play.obs
    me = obs.seat
    tried = choices(obs, play.legal, play.chosen)
    weights = standings_weights(obs.scores, me)
    leader = game_leader(obs.scores, me)
    per_deal: dict[int, list[float]] = {card: [] for card in tried}
    leader_per_deal: dict[int, list[float]] = {card: [] for card in tried}
    for deal, hands in enumerate(redeals(obs, deals, seed, rules)):
        for card in tried:
            scores = card_playout(obs, hands, card, card_play, seed * 1009 + deal, rules)
            per_deal[card].append(
                scores[me] - sum(w * s for w, s in zip(weights, scores, strict=True))
            )
            if leader is not None:
                leader_per_deal[card].append(scores[leader])
    values = {card: _mean_se(vs) for card, vs in per_deal.items()}
    best, lost = points_lost(values, play.chosen)
    differences = [b - c for b, c in zip(per_deal[best], per_deal[play.chosen], strict=True)]
    leader_scores = {card: _mean_se(vs)[0] for card, vs in leader_per_deal.items() if vs}
    return Judged(play, weight, values, best, lost, _mean_se(differences)[1], leader, leader_scores)


def _judge_task(task: tuple[CardPlay, float, int, str, int, RulesConfig]) -> Judged:
    play, weight, deals, card_play, seed, rules = task
    return judge(play, deals, card_play, seed, rules, weight)


def evaluate(
    sample: Sequence[tuple[CardPlay, float]],
    deals: int = DEFAULT_DEALS,
    card_play: str = DEFAULT_CARD_PLAY,
    seed: int = 0,
    rules: RulesConfig | None = None,
    workers: int = 1,
    progress: Progress | None = None,
) -> list[Judged]:
    rules = rules or RulesConfig()
    tasks = [
        (play, weight, deals, card_play, seed + 7919 * i, rules)
        for i, (play, weight) in enumerate(sample)
    ]
    return run_tasks(_judge_task, tasks, workers, "cards", progress)


def label_of(j: Judged, t: Thresholds) -> str:
    """Best, close (lost less than an inaccuracy), inaccuracy or blunder."""
    if j.best == j.play.chosen:
        return "best"
    if j.lost >= t.blunder:
        return "blunder"
    return "inaccuracy" if j.lost >= t.inaccuracy else "close"


def brilliant(j: Judged, t: Thresholds) -> bool:
    """The best card, and it cuts the game leader's expected round score by ``t.sabotage`` or
    more against the next-best card [Claude: Yosif's default, 2026-10-27]."""
    if j.best != j.play.chosen or not j.leader_scores or len(j.values) < 2:
        return False
    others = [card for card in j.values if card != j.play.chosen]
    runner_up = max(others, key=lambda card: j.values[card][0])
    return j.leader_scores[runner_up] - j.leader_scores[j.play.chosen] >= t.sabotage


def weighted(group: Sequence[Judged], f: Callable[[Judged], float]) -> tuple[float, float]:
    """The weighted mean of ``f`` over ``group`` and its standard error."""
    total = sum(j.weight for j in group)
    mean = sum(j.weight * f(j) for j in group) / total
    var = sum((j.weight * (f(j) - mean)) ** 2 for j in group)
    return mean, math.sqrt(var) / total


def situation(play: CardPlay) -> list[str]:
    """The card play as Yosif would see it at the table."""
    obs = play.obs
    me = obs.seat

    def who(seat: int) -> str:
        return "you" if seat == me else f"seat {seat}"

    kind = f"fixed trump {TRUMPS[obs.fixed_trump]}" if obs.fixed_trump is not None else "open"
    trump = TRUMPS[obs.trump] if obs.trump is not None else "?"
    standing = []
    for seat in range(4):
        estimate = obs.estimates[seat]
        dash = " Dash" if seat in obs.dash_callers else ""
        caller = " Caller" if seat == obs.caller else ""
        standing.append(f"{who(seat)}{dash}{caller} {estimate} ({obs.tricks_won[seat]})")
    trick = ", ".join(f"{who(seat)} {card_text(card)}" for seat, card in obs.trick)
    return [
        f"Round {obs.round_no} ({kind}), trump {trump}, trick {len(obs.tricks) + 1}; "
        f"you are seat {me}.",
        "Scores before this round: " + ", ".join(f"{who(s)} {obs.scores[s]}" for s in range(4)),
        "Estimates (tricks won): " + ", ".join(standing),
        f"Your hand: {hand_text(obs.hand)}",
        f"This trick: {trick}" if trick else "You lead this trick.",
    ]


def format_evaluation(
    target: str,
    judged: Sequence[Judged],
    population: int,
    deals: int,
    card_play: str,
    t: Thresholds,
    notes: Sequence[str],
) -> str:
    lines = [
        f"Card-play evaluation: {target}, {len(judged)} of its {population} card plays "
        "with a real choice",
        f"Judge: each card played on {deals} re-dealt hands, the rest of the round by "
        f"{card_play}; it sees only the player's own view.",
        *notes,
        "",
        "Points lost per card (0 means the best card), weighted to how often each situation "
        "comes up",
        f"  {'Situation':<14} {'Cards':>5}  {'Mean lost':>11}  {'Best':>5}  "
        f"{'Inaccuracy ?':>12}  {'Blunder ??':>10}",
    ]
    rows: list[tuple[str, list[Judged]]] = [("all", list(judged))]
    rows += [(stage, [j for j in judged if j.play.stage == stage]) for stage in STAGES]
    rows += [(need, [j for j in judged if j.play.need == need]) for need in NEEDS]
    for name, group in rows:
        if not group:
            continue
        mean, se = weighted(group, lambda j: j.lost)
        share = {
            label: weighted(group, lambda j, label=label: float(label_of(j, t) == label))[0]
            for label in LABELS
        }
        lines.append(
            f"  {name:<14} {len(group):>5}  {mean:>5.2f} ± {se:<4.2f}  "
            f"{100 * share['best']:>4.0f}%  {100 * share['inaccuracy']:>11.0f}%  "
            f"{100 * share['blunder']:>9.0f}%"
        )
    counts = {label: sum(label_of(j, t) == label for j in judged) for label in LABELS}
    lines += [
        "",
        f"Labels of the {len(judged)} cards judged (Inaccuracy {t.inaccuracy:g}+ points lost, "
        f"Blunder {t.blunder:g}+, Brilliant Sabotage cuts the leader {t.sabotage:g}+)",
        f"  Best                    {counts['best']:>5}",
        f"    Brilliant Sabotage !! {sum(brilliant(j, t) for j in judged):>5}",
        f"  {f'Close, under {t.inaccuracy:g}':<24}{counts['close']:>5}",
        f"  Inaccuracy ?            {counts['inaccuracy']:>5}",
        f"  Blunder ??              {counts['blunder']:>5}",
    ]
    worst = [j for j in sorted(judged, key=lambda j: -j.lost)[:WORST] if j.lost > 0]
    if worst:
        lines += ["", f"The {len(worst)} costliest cards"]
    for n, j in enumerate(worst, 1):
        p = j.play
        chosen, best = j.values[p.chosen], j.values[j.best]
        clear = j.lost > 2 * j.lost_se
        lines += ["", f"{n}. {p.stage}, {p.need}: lost {j.lost:.1f} ± {j.lost_se:.1f} points"]
        lines += [f"   {line}" for line in situation(p)]
        lines.append(
            f"   Played {card_text(p.chosen)} ({chosen[0]:+.1f}); best was "
            f"{card_text(j.best)} ({best[0]:+.1f}); "
            + ("a clear difference." if clear else "within the noise.")
        )
    return "\n".join(lines)


def to_json(target: str, judged: Sequence[Judged], t: Thresholds) -> dict[str, Any]:
    return {
        "bot": target,
        "thresholds": {"inaccuracy": t.inaccuracy, "blunder": t.blunder, "sabotage": t.sabotage},
        "cards": [
            {
                "seat": j.play.seat,
                "game": None
                if j.play.game is None
                else {
                    "deal_seed": j.play.game.deal_seed,
                    "set_index": j.play.game.set_index,
                    "rotation": j.play.game.rotation,
                },
                "step": j.play.step,
                "round": j.play.obs.round_no,
                "trick": len(j.play.obs.tricks) + 1,
                "stage": j.play.stage,
                "need": j.play.need,
                "weight": j.weight,
                "chosen": card_text(j.play.chosen),
                "best": card_text(j.best),
                "lost": j.lost,
                "lost_se": j.lost_se,
                "label": label_of(j, t),
                "brilliant_sabotage": brilliant(j, t),
                "values": {card_text(c): list(v) for c, v in j.values.items()},
                "leader": j.leader,
                "leader_scores": {card_text(c): s for c, s in j.leader_scores.items()},
            }
            for j in judged
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.eval.cards",
        description="Points each of a bot's card plays cost against the best card.",
    )
    add_source_arguments(parser)
    parser.add_argument("--cards", type=int, default=DEFAULT_CARDS, help="cards to judge (300)")
    parser.add_argument("--deals", type=int, default=DEFAULT_DEALS, help="deals per card (32)")
    parser.add_argument("--card-play", default=DEFAULT_CARD_PLAY, help="card player in play-outs")
    parser.add_argument(
        "--inaccuracy", type=float, default=DEFAULT_THRESHOLDS.inaccuracy, help="(2)"
    )
    parser.add_argument("--blunder", type=float, default=DEFAULT_THRESHOLDS.blunder, help="(5)")
    parser.add_argument("--sabotage", type=float, default=DEFAULT_THRESHOLDS.sabotage, help="(5)")
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--out", type=Path, help="write report.txt and evaluation.json here")
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    args = parser.parse_args(argv)
    if min(args.games, args.cards, args.deals, args.workers) < 1 or args.seed < 0:
        parser.error("--games, --cards, --deals and --workers must be at least 1")
    thresholds = Thresholds(args.inaccuracy, args.blunder, args.sabotage)
    games = games_from(args, parser)
    notes = list(games.notes)
    plays: list[CardPlay] = []
    skipped = 0
    for record in games.records:
        try:
            plays += plays_of(record, games.rules, args.bot)
        except (IllegalActionError, ValueError):
            skipped += 1
    if skipped:
        notes.append(f"Skipped {skipped} saved games that no longer replay.")
    if not plays:
        parser.error(f"no card plays with a choice by {args.bot} in these games")
    sample = sample_plays(plays, args.cards, args.seed)
    judged = evaluate(
        sample,
        args.deals,
        args.card_play,
        args.seed,
        games.rules,
        args.workers,
        print_progress if args.progress else None,
    )
    text = format_evaluation(
        args.bot, judged, len(plays), args.deals, args.card_play, thresholds, notes
    )
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.txt").write_text(text + "\n")
        data = to_json(args.bot, judged, thresholds)
        (args.out / "evaluation.json").write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
