"""Measuring make chances by play-outs: calls assigned on random deals, not chosen (tuning).

Saved games only hold calls players chose, and they choose calls they can make, so they overstate
how makeable an unusual call is. Here each random deal is played several times with one seat (the
subject) committed to a different call each time, everything else equal, so how often a call is
made is measured for calls nobody would choose too.

Three scenarios per deal, round 1:

- ``caller``: seat 0 bids k in its best trump and the others pass, for k from 2 below to 2 above
  its natural trick count (the saved-games line in ``calibration``), 4 to 13;
- ``estimator``: seat 0 holds a contract near its natural count and seat 1, the first to
  estimate, estimates k, from 2 below to 2 above its own natural count, within 0 to the contract;
- ``dash``: seat 0 Dash Calls and seats 1 to 3 bid as HeuristicBot.

The other seats' later Dash Calls, bids and estimates are HeuristicBot's, and every seat plays its
cards with ``--card-play`` (PIMC with 8 deals by default), seeded the same for every call on a
deal. Each trial keeps the subject's counting numbers (as the yosif bot computes them) and the
tricks it took.

Example: ``uv run python -m estimation.eval.measure --hands 2000 --workers 5 --progress``.
"""

import argparse
import dataclasses
import gzip
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from estimation.arena.games import code_version
from estimation.bots import HeuristicBot, make_bot
from estimation.bots.yosif import YosifConfig
from estimation.bots.yosif.calibration import Calibration, load_calibration
from estimation.bots.yosif.counting import NO_READS, read_bids
from estimation.bots.yosif.odds import Outlook
from estimation.bots.yosif.policy import TRUMPS, Planner
from estimation.engine import (
    DASH_CALL,
    NO_DASH_CALL,
    PASS,
    SANS,
    Phase,
    Round,
    RulesConfig,
    bid_action,
    estimate_action,
)
from estimation.engine.cards import deal

from .work import print_progress, run_tasks

SCENARIOS = ("caller", "estimator", "dash")
REACH = 2  # calls tried either side of the natural count
DEFAULT_CARD_PLAY = "pimc:samples=8"
DEFAULT_OUT = Path("runs/calibration/playouts.jsonl.gz")


@dataclass(frozen=True, slots=True)
class Trial:
    scenario: str
    deal_seed: int
    call: int  # 0 for a Dash Call
    trump: int
    em: float  # the subject's counting: expected tricks trying to win
    el: float  # trying to lose
    trumps: int  # trumps the subject holds (0 in Sans)
    sans: bool
    zero: float  # for a Dash Call: the counting's pressed chance of no trick
    tricks: int
    made: bool


def _means(o: Outlook) -> tuple[float, float]:
    return sum(k * p for k, p in enumerate(o.most)), sum(k * p for k, p in enumerate(o.least))


def _natural(fit: Calibration | None, o: Outlook, hand: int, trump: int) -> float:
    em, el = _means(o)
    if fit is None:
        return em
    held = 0 if trump == SANS else ((hand >> (13 * trump)) & 0x1FFF).bit_count()
    return fit.natural(em=em, el=el, trumps=held, sans=trump == SANS, fixed=False, last_over=0)


def _held(hand: int, trump: int) -> int:
    return 0 if trump == SANS else ((hand >> (13 * trump)) & 0x1FFF).bit_count()


def _finish(rnd: Round, seed: int, card_play: str, rules: RulesConfig) -> Round:
    """Play the round out: HeuristicBot's Dash Calls, bids and estimates, ``card_play`` cards."""
    heuristic = HeuristicBot()
    players = [make_bot(card_play, seed * 4 + seat, rules) for seat in range(4)]
    while rnd.phase is not Phase.DONE:
        seat = rnd.current_player
        if seat is None:
            raise RuntimeError("a round that is not done always has a player to act")
        bot = players[seat] if rnd.phase is Phase.PLAY else heuristic
        rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
    return rnd


def _calls(natural: float, low: int, high: int) -> list[int]:
    centre = round(natural)
    calls = [k for k in range(centre - REACH, centre + REACH + 1) if low <= k <= high]
    return calls or [max(low, min(high, centre))]


def trials_for_deal(
    seed: int,
    card_play: str = DEFAULT_CARD_PLAY,
    rules: RulesConfig | None = None,
    fit: Calibration | None = None,
) -> list[Trial]:
    """Every scenario's trials on the round-1 deal of ``seed``."""
    rules = rules or RulesConfig()
    fit = fit if fit is not None else load_calibration()
    planner = Planner(YosifConfig(calibrated=False))
    hands = deal(seed, 1)
    trials: list[Trial] = []

    # caller: seat 0 bids k in its best trump, the others pass
    outlooks = {t: planner.counting_outlook(hands[0], t, NO_READS) for t in TRUMPS}
    trump = max(TRUMPS, key=lambda t: _natural(fit, outlooks[t], hands[0], t))
    em, el = _means(outlooks[trump])
    for k in _calls(_natural(fit, outlooks[trump], hands[0], trump), 4, 13):
        rnd = Round.new(1, hands, config=rules)
        for action in (NO_DASH_CALL,) * 4 + (bid_action(k, trump), PASS, PASS, PASS, PASS):
            rnd.step(action)
        tricks = _finish(rnd, seed, card_play, rules).result().seats[0].tricks
        trials.append(
            Trial(
                "caller",
                seed,
                k,
                trump,
                em,
                el,
                _held(hands[0], trump),
                trump == SANS,
                0.0,
                tricks,
                tricks == k,
            )
        )

    # estimator: seat 0 holds a contract near its natural count, seat 1 estimates k
    contract = max(4, min(9, round(_natural(fit, outlooks[trump], hands[0], trump))))
    opening = (NO_DASH_CALL,) * 4 + (bid_action(contract, trump), PASS, PASS, PASS, PASS)
    rnd = Round.new(1, hands, config=rules)
    for action in opening:
        rnd.step(action)
    obs = rnd.observation(1)
    o = planner.counting_outlook(hands[1], trump, read_bids(obs))
    em1, el1 = _means(o)
    for k in _calls(_natural(fit, o, hands[1], trump), 0, contract):
        rnd = Round.new(1, hands, config=rules)
        for action in (*opening, estimate_action(k)):
            rnd.step(action)
        tricks = _finish(rnd, seed, card_play, rules).result().seats[1].tricks
        trials.append(
            Trial(
                "estimator",
                seed,
                k,
                trump,
                em1,
                el1,
                _held(hands[1], trump),
                trump == SANS,
                0.0,
                tricks,
                tricks == k,
            )
        )

    # dash: seat 0 Dash Calls, the others bid as HeuristicBot
    rnd = Round.new(1, hands, config=rules)
    rnd.step(DASH_CALL)
    for _ in range(3):
        rnd.step(NO_DASH_CALL)
    result = _finish(rnd, seed, card_play, rules).result()
    if not result.skipped:
        zero = planner.dash_zero_chance(hands[0], NO_READS)
        tricks = result.seats[0].tricks
        trials.append(Trial("dash", seed, 0, -1, 0.0, 0.0, 0, False, zero, tricks, tricks == 0))
    return trials


def _deal_task(task: tuple[int, str, RulesConfig]) -> list[Trial]:
    seed, card_play, rules = task
    return trials_for_deal(seed, card_play, rules)


def write_trials(path: Path, header: dict[str, Any], trials: Sequence[Trial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"header": header}) + "\n")
        for trial in trials:
            out.write(json.dumps(dataclasses.asdict(trial)) + "\n")


def read_trials(path: Path) -> tuple[dict[str, Any], list[Trial]]:
    header: dict[str, Any] = {}
    trials: list[Trial] = []
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            data = json.loads(line)
            if "header" in data:
                header = dict(data["header"])
            else:
                trials.append(Trial(**data))
    return header, trials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.eval.measure",
        description="Measure make chances by playing assigned calls out on random deals.",
    )
    parser.add_argument("--hands", type=int, required=True, help="random deals to play")
    parser.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    parser.add_argument("--card-play", default=DEFAULT_CARD_PLAY, help="every seat's card play")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the trials")
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    args = parser.parse_args(argv)
    if args.hands < 1 or args.workers < 1 or args.seed < 0:
        parser.error("--hands and --workers must be at least 1, --seed 0 or more")
    rules = RulesConfig()
    tasks = [(args.seed + i, args.card_play, rules) for i in range(args.hands)]
    progress = print_progress if args.progress else None
    per_deal = run_tasks(_deal_task, tasks, args.workers, "deals", progress)
    trials = [trial for deal_trials in per_deal for trial in deal_trials]
    header = {
        "hands": args.hands,
        "seed": args.seed,
        "card_play": args.card_play,
        "code_version": code_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_trials(args.out, header, trials)
    counts = {s: sum(t.scenario == s for t in trials) for s in SCENARIOS}
    made = {s: sum(t.made for t in trials if t.scenario == s) for s in SCENARIOS}
    print(
        f"{len(trials):,} trials on {args.hands:,} deals: "
        + ", ".join(
            f"{s} {counts[s]:,} ({made[s] / max(1, counts[s]):.0%} made)" for s in SCENARIOS
        )
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
