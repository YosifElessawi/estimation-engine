"""Duplicate arena CLI (SPEC §6 M2).

Example: ``uv run python -m estimation.arena --lineup heuristic,random,heuristic,random
--sets 1000 --seed 0``. Bots are ``name`` or ``name:key=value:...`` (see estimation.bots).
"""

import argparse
import sys
from pathlib import Path

from estimation.bots import make_bot

from .duplicate import run_arena
from .ratings import DEFAULT_PRIOR, PairRecord
from .report import build_report, format_report, format_run, load_pairs, write_json
from .status import StatusWriter, status_path


def _progress(done: int, total: int, elapsed: float) -> None:
    left = elapsed / done * (total - done)
    print(
        f"[arena] {done:,}/{total:,} sets, {elapsed / 60:.1f} min elapsed, "
        f"about {left / 60:.1f} min left",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m estimation.arena",
        description="Rate bots by duplicate play: every bot plays every hand.",
    )
    parser.add_argument(
        "--lineup",
        required=True,
        help="4 bots, comma-separated, one per lineup slot (repeats allowed), "
        "for example heuristic,random,heuristic,random",
    )
    parser.add_argument("--sets", type=int, required=True, help="duplicate sets to play")
    parser.add_argument("--seed", type=int, default=0, help="run seed (default 0)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=250.0,
        help="card decisions slower than this are counted in the run section (default 250)",
    )
    parser.add_argument("--out", type=Path, help="also write the report as JSON to this file")
    parser.add_argument(
        "--elo-from",
        type=Path,
        action="append",
        default=[],
        help="include the set results of an earlier JSON report in the Elo fit (repeatable)",
    )
    parser.add_argument(
        "--elo-prior",
        type=float,
        default=DEFAULT_PRIOR,
        help="virtual draws added to each compared pair in the Elo fit (default 1)",
    )
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    parser.add_argument(
        "--status",
        type=Path,
        help="keep the live status here for python -m estimation.arena.watch "
        "(default: next to --out, as <name>.status.json)",
    )
    args = parser.parse_args(argv)

    lineup = tuple(name.strip() for name in args.lineup.split(","))
    if len(lineup) != 4 or not all(lineup):
        parser.error(f"--lineup needs 4 bots, one per slot; got {args.lineup!r}")
    for name in dict.fromkeys(lineup):
        try:
            make_bot(name, 0)
        except ValueError as error:
            parser.error(str(error))
    if args.sets < 1:
        parser.error("--sets must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be 0 or more")
    if args.elo_prior < 0:
        parser.error("--elo-prior must be 0 or more")
    earlier: list[PairRecord] = []
    for path in args.elo_from:
        try:
            earlier += load_pairs(path)
        except (OSError, ValueError) as error:
            parser.error(f"--elo-from {path}: {error}")

    status = None
    status_file = args.status or (status_path(args.out) if args.out else None)
    if status_file is not None:
        status = StatusWriter(
            status_file,
            "arena",
            [(", ".join(lineup), lineup)],
            args.sets,
            args.seed,
            args.workers,
            ["python -m estimation.arena", *(argv if argv is not None else sys.argv[1:])],
        )
        status.start(0)
    try:
        run = run_arena(
            lineup,
            args.sets,
            args.seed,
            workers=args.workers,
            budget=args.budget_ms / 1000,
            progress=_progress if args.progress else None,
            on_set=status.set_done if status is not None else None,
        )
    except BaseException as exc:  # Ctrl-C included: the watch screen shows why the run stopped
        if status is not None:
            status.finish(error=f"stopped: {exc!r}")
        raise
    report = build_report(run, earlier, [str(path) for path in args.elo_from], args.elo_prior)
    if status is not None:
        status.done(0)
        status.finish(ratings=dict(report.elo_ratings) or None)
    print(format_report(report))
    print()
    print(format_run(run))
    if args.out:
        write_json(args.out, report, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
