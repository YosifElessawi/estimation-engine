"""The bid lessons' layout (step 6 of docs/rl/hybrid-iteration-plan.md), shared by their recorder
(``bid_lessons``, which needs the engine) and their learner (``bid_learn``, on the GPU). Needs
only numpy.

One row per taught bidding decision: a Dash Call, bid, With, Pass, estimate or super call with a
choice, searched by the bid teacher (``bid_teacher``). Each row has ``K`` option slots: slot 0 is
the player's favourite, the move it made; the others are the alternatives the teacher tried, by
the player's chances; unused slots hold -1 in ``options`` and 0 elsewhere.

==============  ===============  ==========================================================
field           shape, type      what it holds
==============  ===============  ==========================================================
``game``        [L] int32        the recorded Bola's game number
``step``        [L] int32        the decision's move number in that Bola
``seat``        [L] int8         the deciding seat
``phase``       [L] int8         0 a Dash Call, 1 the auction, 2 an estimate (or super call)
``x``           [L, 747] f16     the player's view (estimation.jaxenv.encode), ending in its
                                 legal moves
``tricks``      [L] f32          the tricks the player expected to take this round
``options``     [L, K] int8      each slot's action id
``chance``      [L, K] f32       the player's move chance of the option, among every legal one
``rank``        [L, K] int16     the option's place among all legal options by that chance
``value``       [L, K] f32       its mean value, in round points, over the deals it played
``deals``       [L, K] int16     how many deals it played before it was dropped
``margin``      [L, K] f32       its mean lead over the favourite on those deals: flattered
                                 for an option kept on, since it was kept on the same deals
``error``       [L, K] f32       that lead's standard error
``opening``     [L, K] f32       its mean lead on the first stage's deals alone, which every
                                 option played: a fair measure of any option, but a short one
``finalist``    [L] int8         the slot of the alternative that lasted longest
``exam``        [L] f32          the finalist's mean lead over the favourite on its exam: the
                                 deals it played once it was the only alternative left, none
                                 of which decided what was kept. The fair measure of it
``exam_error``  [L] f32          that lead's standard error
``exam_deals``  [L] int16        the exam's deals: 0 when every alternative was dropped as
                                 hopeless before one was left alone
``pick``        [L] int8         the slot the teacher picks: the finalist if it sat its exam
                                 to the last deal and led it, else the favourite (slot 0)
==============  ===============  ==========================================================

A lesson whose finalist was dropped as hopeless has a short exam or none: its ``pick`` is the
favourite, and its numbers say only that no alternative came close.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

VIEW = 747  # estimation.jaxenv.encode.SIZE
DTYPES: dict[str, Any] = {
    "game": np.int32,
    "step": np.int32,
    "seat": np.int8,
    "phase": np.int8,
    "x": np.float16,
    "tricks": np.float32,
    "options": np.int8,
    "chance": np.float32,
    "rank": np.int16,
    "value": np.float32,
    "deals": np.int16,
    "margin": np.float32,
    "error": np.float32,
    "opening": np.float32,
    "finalist": np.int8,
    "exam": np.float32,
    "exam_error": np.float32,
    "exam_deals": np.int16,
    "pick": np.int8,
}
PER_OPTION = ("options", "chance", "rank", "value", "deals", "margin", "error", "opening")
CAREFUL = 1.0  # standard errors an exam lead needs for the teacher's careful pick
Lessons = dict[str, npt.NDArray[Any]]


def lesson_path(out: Path, first: int) -> Path:
    """The lessons of the recording part whose first Bola is ``first``."""
    return out / "parts" / f"lessons-{first:05d}.npz"


def lesson_paths(out: Path) -> list[Path]:
    return sorted(p for p in (out / "parts").glob("lessons-*.npz") if ".tmp" not in p.name)


def empty(slots: int) -> Lessons:
    """No lessons, with ``slots`` option slots."""
    shapes = {name: (0, slots) if name in PER_OPTION else (0,) for name in DTYPES}
    shapes["x"] = (0, VIEW)
    return {name: np.zeros(shapes[name], DTYPES[name]) for name in DTYPES}


def join(parts: list[Lessons]) -> Lessons:
    """Several sets of lessons as one, in order."""
    if not parts:
        raise ValueError("no lessons to join")
    return {name: np.concatenate([part[name] for part in parts]) for name in DTYPES}


def save_lessons(path: Path, lessons: Lessons) -> None:
    """Write a part so that a reader never sees it half written."""
    missing = set(DTYPES) - set(lessons)
    if missing:
        raise ValueError(f"lessons need every field; missing {sorted(missing)}")
    tmp = path.with_name(path.stem + ".tmp.npz")
    with tmp.open("wb") as file:
        np.savez_compressed(
            file, allow_pickle=False, **{k: lessons[k].astype(DTYPES[k]) for k in DTYPES}
        )
    tmp.replace(path)


def load_lessons(path: Path) -> Lessons:
    with np.load(path) as data:
        return {name: data[name] for name in DTYPES}


def examined(lessons: Lessons) -> npt.NDArray[np.bool_]:
    """The lessons whose finalist sat its whole exam, to the search's last deal."""
    rows = np.arange(len(lessons["pick"]))
    total = lessons["deals"][:, 0].max() if len(rows) else 0
    sat = lessons["deals"][rows, lessons["finalist"]]
    return (sat == total) & (lessons["exam_deals"] > 0) & (lessons["finalist"] > 0)


def careful_pick(lessons: Lessons) -> npt.NDArray[np.int64]:
    """Each lesson's careful pick, as an action id: the finalist where it sat its whole exam
    and led it by ``CAREFUL`` standard errors, else the player's own move."""
    rows = np.arange(len(lessons["pick"]))
    sure = examined(lessons) & (lessons["exam"] >= CAREFUL * lessons["exam_error"])
    sure &= lessons["exam"] > 0
    slot = np.where(sure, lessons["finalist"], 0)
    return lessons["options"][rows, slot].astype(np.int64)


def load_all(out: Path) -> Lessons:
    """Every saved part of the lessons in ``out``, in order, as one set."""
    paths = lesson_paths(out)
    if not paths:
        raise FileNotFoundError(f"no lessons in {out}")
    return join([load_lessons(path) for path in paths])
