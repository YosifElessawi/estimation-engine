"""How a bot bids: Dash Calls, contract sizes, super calls and how often each is made.

Tables of four copies of one bot play whole games. Shares of open rounds (1-13) count Dash Calls
per player per round and contracts per played round; "made" means exactly the estimate.

Example: ``uv run python -m estimation.arena.bidstats --bot yosif --games 300 --workers 5``.
"""

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from estimation.bots import make_bot
from estimation.engine import Bola, RoundResult

BIG = "8+"


@dataclass(frozen=True, slots=True)
class SeatRow:
    estimate: int
    tricks: int
    won: bool
    dash: bool
    caller: bool
    with_: bool
    super_: bool


@dataclass(frozen=True, slots=True)
class RoundRow:
    fixed: bool
    skipped: bool
    trump_changed: bool
    seats: tuple[SeatRow, ...]


@dataclass(frozen=True, slots=True)
class BidStats:
    open_rounds: int
    skipped_share: float
    dash_share: float  # per player per open round
    dash_made: float
    with_round_share: float  # played open rounds with at least one With player
    contract_share: dict[str, float]  # per played open round, by the Caller's bid
    contract_made: dict[str, float]
    super_share: float  # played open rounds with a contract of 8 or more
    super_made: float
    fixed_rounds: int
    fixed_super_share: float  # fixed-trump rounds whose Caller was scored as a super call
    trump_changes: int  # fixed-trump rounds whose trump a super call changed (FIX-5)
    caller_made: float


def _row(result: RoundResult) -> RoundRow:
    return RoundRow(
        fixed=result.fixed_trump is not None,
        skipped=result.skipped,
        trump_changed=result.trump_changed,
        seats=tuple(
            SeatRow(s.estimate, s.tricks, s.won, s.is_dash, s.is_caller, s.is_with, s.is_super)
            for s in result.seats
        ),
    )


def _play(task: tuple[str, int]) -> list[RoundRow]:
    spec, seed = task
    bots = [make_bot(spec, seed * 4 + seat) for seat in range(4)]
    bola = Bola(seed)
    while not bola.is_over:
        seat = bola.current_player
        if seat is None:
            raise RuntimeError("a game that is not over always has a player to act")
        bola.step(bots[seat].act(bola.observation(seat), bola.legal_actions()))
    return [_row(result) for result in bola.history]


def collect(spec: str, games: int, seed: int, workers: int) -> list[RoundRow]:
    """Every round of ``games`` games of four ``spec`` bots, dealt from ``seed``, ``seed + 1``..."""
    tasks = [(spec, seed + i) for i in range(games)]
    if workers == 1:
        return [row for task in tasks for row in _play(task)]
    with Pool(workers) as pool:
        return [row for rows in pool.map(_play, tasks) for row in rows]


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def summarize(rows: Sequence[RoundRow]) -> BidStats:
    open_rows = [r for r in rows if not r.fixed]
    played = [r for r in open_rows if not r.skipped]
    seats = [s for r in open_rows for s in r.seats]
    dashes = [s for s in seats if s.dash]
    callers = [next(s for s in r.seats if s.caller) for r in played]
    sizes: dict[str, list[SeatRow]] = {}
    for c in callers:
        sizes.setdefault(BIG if c.estimate >= 8 else str(c.estimate), []).append(c)
    big = sizes.get(BIG, [])
    fixed = [r for r in rows if r.fixed and not r.skipped]
    return BidStats(
        open_rounds=len(open_rows),
        skipped_share=_share(len(open_rows) - len(played), len(open_rows)),
        dash_share=_share(len(dashes), len(seats)),
        dash_made=_share(sum(s.won for s in dashes), len(dashes)),
        with_round_share=_share(sum(any(s.with_ for s in r.seats) for r in played), len(played)),
        contract_share={k: _share(len(v), len(callers)) for k, v in sorted(sizes.items())},
        contract_made={k: _share(sum(c.won for c in v), len(v)) for k, v in sorted(sizes.items())},
        super_share=_share(len(big), len(played)),
        super_made=_share(sum(c.won for c in big), len(big)),
        fixed_rounds=len(fixed),
        fixed_super_share=_share(
            sum(any(s.caller and s.super_ for s in r.seats) for r in fixed), len(fixed)
        ),
        trump_changes=sum(r.trump_changed for r in fixed),
        caller_made=_share(sum(c.won for c in callers), len(callers)),
    )


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def format_stats(spec: str, stats: BidStats) -> str:
    sizes = "  ".join(
        f"{k}: {_pct(share)} made {_pct(stats.contract_made[k])}"
        for k, share in stats.contract_share.items()
    )
    return "\n".join(
        [
            f"Bidding stats: 4 x {spec}, {stats.open_rounds} open rounds",
            f"  Skipped rounds          {_pct(stats.skipped_share)}",
            f"  Dash Calls per player   {_pct(stats.dash_share)} (made {_pct(stats.dash_made)})",
            f"  Rounds with a With      {_pct(stats.with_round_share)}",
            f"  Contracts               {sizes}",
            f"  Super calls (8+)        {_pct(stats.super_share)} (made {_pct(stats.super_made)})",
            f"  Fixed rounds            {stats.fixed_rounds}, super-call Caller "
            f"{_pct(stats.fixed_super_share)}, trump changed {stats.trump_changes}",
            f"  Caller makes exactly    {_pct(stats.caller_made)}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena.bidstats",
        description="How a bot bids, from games of four copies of it.",
    )
    parser.add_argument("--bot", required=True, help="the bot spec, for example yosif")
    parser.add_argument("--games", type=int, required=True, help="games to play")
    parser.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--out", type=Path, help="also write the stats as JSON to this file")
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 1 or args.seed < 0:
        parser.error("--games and --workers must be at least 1, --seed 0 or more")
    try:
        make_bot(args.bot, 0)
    except ValueError as error:
        parser.error(str(error))
    stats = summarize(collect(args.bot, args.games, args.seed, args.workers))
    print(format_stats(args.bot, stats))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dataclasses.asdict(stats), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
