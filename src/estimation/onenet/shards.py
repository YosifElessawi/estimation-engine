"""The training set's layout, shared by its builder (``data``, on the Mac) and the copy
(``copy_hybrid``, on the GPU). Needs only numpy.

One row per decision with a choice (more than one legal move). Seats are relative to the player
deciding: 0 is the player, 1 the next seat, 2 the seat opposite, 3 the last.

====================  ===============  ===================================================
field                 shape, type      what it holds
====================  ===============  ===================================================
``bola``              [N] int32        the recorded Bola's game number
``seat``              [N] int8         the player's seat
``kind``              [N] int8         ``KIND_BID``: BiddingNet's values; ``KIND_CARD``: the
                                       teacher's values; ``KIND_PLAYED``: no values, only
                                       Hybrid v4's own move
``action``            [N] uint8        the move played
``x``                 [N, 747] f16     Essam's raw view (estimation.jaxenv.encode_np): the
                                       player's only input; its last 120 are the legal moves
``values``            [N, 120] f32     the kept value of each legal move, in points (0 for
                                       the others and for ``KIND_PLAYED``)
``holder``            [N, 52] int8     who holds each card the player cannot see: 0-2 for
                                       relative seats 1-3, -1 for the player's own cards and
                                       cards already played
``tricks``            [N, 4] int8      each player's tricks this round (-1: a skipped round)
``made``              [N, 4] int8      1 if each player made their call this round, else 0
                                       (-1: a skipped round)
``round_score``       [N, 4] int32     each player's score for this round
``final``             [N, 4] int32     each player's final score in the Bola
``card_net``          [N] int8         Hybrid v4's card net's own top move at ``KIND_CARD``
                                       rows (the copy is compared with it), else -1
====================  ===============  ===================================================
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

KIND_BID, KIND_CARD, KIND_PLAYED = 0, 1, 2  # the first two as in estimation.rl.onenet_record
NUM_ACTIONS = 120
VIEW = 747  # estimation.jaxenv.encode.SIZE
DTYPES: dict[str, Any] = {
    "bola": np.int32,
    "seat": np.int8,
    "kind": np.int8,
    "action": np.uint8,
    "x": np.float16,
    "values": np.float32,
    "holder": np.int8,
    "tricks": np.int8,
    "made": np.int8,
    "round_score": np.int32,
    "final": np.int32,
    "card_net": np.int8,
}
Shard = dict[str, npt.NDArray[Any]]


def shard_path(out: Path, first: int) -> Path:
    """The shard built from the recording part whose first Bola is ``first``."""
    return out / f"shard-{first:05d}.npz"


def save_shard(path: Path, shard: Shard) -> None:
    """Write a shard so that a reader never sees it half written."""
    missing = set(DTYPES) - set(shard)
    if missing:
        raise ValueError(f"a shard needs every field; missing {sorted(missing)}")
    tmp = path.with_name(path.stem + ".tmp.npz")
    with tmp.open("wb") as file:
        np.savez_compressed(
            file, allow_pickle=False, **{k: shard[k].astype(DTYPES[k]) for k in DTYPES}
        )
    tmp.replace(path)


def load_shard(path: Path) -> Shard:
    with np.load(path) as data:
        return {name: data[name] for name in DTYPES}


def shard_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("shard-*.npz") if not p.name.endswith(".tmp.npz"))


def load_data(folder: Path) -> Shard:
    """Every shard in ``folder``, in order, as one set. Filled in place, so loading takes about
    the set's own size in memory (about 4.5 GB for 2,000 Bolas), not twice it."""
    paths = shard_paths(folder)
    if not paths:
        raise FileNotFoundError(f"no shards in {folder}")
    counts: list[int] = []
    for path in paths:
        with np.load(path) as data:
            counts.append(len(data["kind"]))  # reads only that array
    first = load_shard(paths[0])
    total = sum(counts)
    out = {name: np.empty((total, *first[name].shape[1:]), DTYPES[name]) for name in DTYPES}
    start = 0
    for path, count in zip(paths, counts, strict=True):
        shard = first if path == paths[0] else load_shard(path)
        for name in DTYPES:
            out[name][start : start + count] = shard[name]
        start += count
    return out
