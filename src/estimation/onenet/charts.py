"""Step 5's chart page: TensorBoard kept to one page of plain charts.

Self-play writes its numbers here as it goes. Each chart has a plain title and, where there is
one, a flat line for the mark to beat. Serve and open it on the home server:

    ~/jax-env/bin/tensorboard --logdir ~/estimation-engine/runs/onenet --host 100.103.16.25
    open http://100.103.16.25:6006/#custom_scalars   (any device on the tailnet)

Only the CUSTOM SCALARS page is meant to be read; it holds every chart below.
"""

from pathlib import Path
from typing import Any

# (section, chart title, [(tag, key in the loop's numbers, or a constant mark)])
Line = tuple[str, str | float]
CHARTS: tuple[tuple[str, str, tuple[Line, ...]], ...] = (
    (
        "1 Strength",
        "King rate against the copy (25% is even)",
        (("strength/king_vs_copy", "king_vs_copy"), ("marks/even_king", 0.25)),
    ),
    (
        "1 Strength",
        "Place against the copy (King +1 to Kooz -1, 0 is even)",
        (("strength/place_vs_copy", "place_vs_copy"), ("marks/even_place", 0.0)),
    ),
    (
        "2 Head start",
        "Distance from the copy's moves (0 is the copy)",
        (("headstart/distance", "distance_from_copy"),),
    ),
    (
        "2 Head start",
        "Pull back toward the copy (fades to 0)",
        (("headstart/pull", "anchor_weight"),),
    ),
    (
        "3 Practice questions",
        "Hidden cards: holder named (blind 35%, GuessNet 47%)",
        (
            ("practice/holders", "holders_named"),
            ("marks/blind_holders", 0.35),
            ("marks/guessnet_holders", 0.47),
        ),
    ),
    (
        "3 Practice questions",
        "Judge: error on final place (blind about 0.52)",
        (("practice/judge_error", "final_mse"), ("marks/blind_judge", 0.52)),
    ),
    (
        "4 Play",
        "Callers who make their call",
        (("play/caller_made", "caller_made"),),
    ),
    (
        "5 Progress and speed",
        "Loops done (the line is the run's total)",
        (("progress/loops", "loop"), ("marks/loops_total", "loops_total")),
    ),
    (
        "5 Progress and speed",
        "Bolas played so far",
        (("progress/bolas", "bolas_total"),),
    ),
    (
        "5 Progress and speed",
        "Minutes left",
        (("speed/minutes_left", "minutes_left"),),
    ),
    (
        "5 Progress and speed",
        "Decisions per second (this loop)",
        (("speed/decisions_per_s", "decisions_per_s_now"),),
    ),
    (
        "5 Progress and speed",
        "Bolas finished per second (this loop)",
        (("speed/bolas_per_s", "bolas_per_s"),),
    ),
)


def layout() -> dict[str, dict[str, list[Any]]]:
    """The CUSTOM SCALARS page: one chart per entry of ``CHARTS``, its lines together."""
    page: dict[str, dict[str, list[Any]]] = {}
    for section, title, lines in CHARTS:
        page.setdefault(section, {})[title] = ["Multiline", [tag for tag, _ in lines]]
    return page


def values_for(numbers: dict[str, float]) -> dict[str, float]:
    """The chart values of one loop: each line whose number this loop has, and its marks."""
    out: dict[str, float] = {}
    for _, _, lines in CHARTS:
        if not any(isinstance(key, str) and key in numbers for _, key in lines):
            continue
        for tag, key in lines:
            if isinstance(key, str):
                if key in numbers:
                    out[tag] = float(numbers[key])
            else:
                out[tag] = key
    return out


class Charts:
    """Writes each loop's numbers to the chart page in ``folder`` (TensorBoard event files)."""

    def __init__(self, folder: Path) -> None:
        from tensorboardX import SummaryWriter  # only where charts are written

        self._writer = SummaryWriter(str(folder))
        self._writer.add_custom_scalars(layout())

    def write(self, loop: int, numbers: dict[str, float]) -> None:
        for tag, value in values_for(numbers).items():
            self._writer.add_scalar(tag, value, loop)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
