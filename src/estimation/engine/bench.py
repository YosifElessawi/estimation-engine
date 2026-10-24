"""Engine benchmark (SPEC §5.5): complete random-play rounds per second on one core.

A random-play round is the deal, every decision taken uniformly at random from the legal
actions, and scoring, all through the public Round API. Round numbers cycle 1-18, so the mix
is a Bola's without replays: 13 open rounds to 5 fixed-trump rounds. The rate inside full
random Bolas (where fixed-trump replays dominate) is printed too.

Run: ``uv run python -m estimation.engine.bench``. Exits 1 if the round rate is below target.
"""

import argparse
import random
import statistics
import time

from .cards import deal
from .observation import Phase
from .random_play import play_random_bola
from .round import FIXED_ORDER, Round

TARGET_ROUNDS_PER_SECOND = 5_000


def bench_rounds(rounds: int, seed: int) -> float:
    """Random-play rounds per second over ``rounds`` rounds."""
    choice = random.Random(seed).choice
    done = Phase.DONE
    start = time.perf_counter()
    for i in range(rounds):
        round_no = i % 18 + 1
        fixed_trump = None if round_no <= 13 else FIXED_ORDER[round_no - 14]
        rnd = Round.new(round_no, deal(seed + i // 18, round_no), fixed_trump)
        while rnd.phase is not done:
            rnd.step(choice(rnd.legal_actions()))
        rnd.result()
    return rounds / (time.perf_counter() - start)


def bench_bolas(bolas: int, seed: int) -> float:
    """Rounds per second inside ``bolas`` complete random Bolas."""
    rounds = 0
    start = time.perf_counter()
    for i in range(bolas):
        rounds += len(play_random_bola(seed + i, seed + i).history)
    return rounds / (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--rounds", type=int, default=20_000, help="rounds per timed run")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs; the median is reported")
    parser.add_argument("--bolas", type=int, default=100, help="random Bolas for the game rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=TARGET_ROUNDS_PER_SECOND)
    args = parser.parse_args(argv)

    rate = statistics.median(bench_rounds(args.rounds, args.seed) for _ in range(args.repeat))
    game_rate = bench_bolas(args.bolas, args.seed)
    print(
        f"random-play rounds: {rate:,.0f} rounds/s "
        f"(median of {args.repeat} x {args.rounds:,} rounds; target {args.target:,.0f})"
    )
    print(f"inside {args.bolas} random Bolas: {game_rate:,.0f} rounds/s")
    return 0 if rate >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
