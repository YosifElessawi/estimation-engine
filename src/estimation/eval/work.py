"""Running evaluation work on several processes, with a progress line (shared by the tools)."""

import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

Progress = Callable[[str, int, int, float], None]  # what, done, in all, seconds elapsed


def print_progress(what: str, done: int, total: int, elapsed: float) -> None:
    """A progress line on stderr: done so far, time spent and time left."""
    left = elapsed / done * (total - done)
    print(
        f"[eval] {what} {done}/{total}, {elapsed / 60:.1f} min so far, "
        f"about {left / 60:.1f} min left",
        file=sys.stderr,
        flush=True,
    )


def run_tasks[T, R](
    work: Callable[[T], R], tasks: Sequence[T], workers: int, what: str, progress: Progress | None
) -> list[R]:
    """``work`` over ``tasks`` in order, on ``workers`` processes, reporting each one done."""
    start = time.perf_counter()
    results: list[R] = []

    def collect(result: R) -> None:
        results.append(result)
        if progress is not None:
            progress(what, len(results), len(tasks), time.perf_counter() - start)

    if workers == 1:
        for task in tasks:
            collect(work(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(work, tasks):
                collect(result)
    return results
