"""Build step 5's training set from the recording of Hybrid v4 (docs/rl/hybrid-iteration-plan.md).

    uv run python -m estimation.onenet.data --record runs/onenet/record --out runs/onenet/data

Replays every recorded Bola from its seed and keeps every decision with a choice, as the one
net sees it, with everything it learns from (the fields are in ``estimation.onenet.shards``).
One shard per recording part; a stopped build resumes where it stopped. Runs on the Mac: it
needs the engine, and PyTorch for Hybrid v4's card net, whose own top card is kept at every
valued card decision so the copy can be compared with it.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from estimation.bots import HybridBot, make_bot
from estimation.engine import Bola, Observation
from estimation.engine.config import DEFAULT_CONFIG
from estimation.engine.scoring import RoundResult
from estimation.jaxenv.encode_np import encode as raw_view
from estimation.rl.encoder import encode as rl_view
from estimation.rl.onenet_record import Arrays, RecordConfig, load_part, part_path, part_starts

from .shards import (
    DTYPES,
    KIND_BID,
    KIND_CARD,
    KIND_PLAYED,
    NUM_ACTIONS,
    Shard,
    save_shard,
    shard_path,
    shard_paths,
)

PLAY_BASE = 68
_BITS = np.arange(52, dtype=np.uint64)


def card_bits(hand: int) -> npt.NDArray[np.bool_]:
    """[52] bool: the cards of a 52-bit hand."""
    return ((np.uint64(hand) >> _BITS) & np.uint64(1)).astype(bool)


def hands_now(bola: Bola, obs: Observation) -> list[int]:
    """Each seat's cards right now: the round's deal less the cards each seat has played."""
    hands = list(bola.round.hands)
    for trick in obs.tricks:
        for i, card in enumerate(trick.cards):
            hands[(trick.leader + i) % 4] &= ~(1 << card)
    for seat, card in obs.trick:
        hands[seat] &= ~(1 << card)
    return hands


def holders(hands: Sequence[int], seat: int) -> npt.NDArray[np.int8]:
    """[52]: for each card another seat holds, which one (0-2 for relative seats 1-3), else -1."""
    out = np.full(52, -1, np.int8)
    for k in (1, 2, 3):
        out[card_bits(hands[(seat + k) % 4])] = k - 1
    return out


def relative(values: Sequence[int], seat: int) -> list[int]:
    """Per-seat ``values`` from ``seat``'s side: the seat itself first."""
    return [values[(seat + k) % 4] for k in range(4)]


@dataclass(slots=True)
class _Rows:
    """A shard's rows while it is built, and its card decisions still to be shown to the card
    net (their row, legal moves and RL view)."""

    columns: dict[str, list[Any]] = field(default_factory=lambda: {name: [] for name in DTYPES})
    card_rows: list[int] = field(default_factory=lambda: list[int]())
    card_legal: list[list[int]] = field(default_factory=lambda: list[list[int]]())
    card_views: list[npt.NDArray[np.float32]] = field(
        default_factory=lambda: list[npt.NDArray[np.float32]]()
    )

    def __len__(self) -> int:
        return len(self.columns["bola"])

    def label_round(self, rows: list[int], result: RoundResult) -> None:
        """Fill the round's answers into the rows decided during it."""
        tricks = [-1] * 4 if result.skipped else [s.tricks for s in result.seats]
        made = [-1] * 4 if result.skipped else [int(s.won) for s in result.seats]
        for row in rows:
            seat = self.columns["seat"][row]
            self.columns["tricks"][row] = relative(tricks, seat)
            self.columns["made"][row] = relative(made, seat)
            self.columns["round_score"][row] = relative(result.scores, seat)

    def label_bola(self, rows: range, scores: Sequence[int]) -> None:
        for row in rows:
            self.columns["final"][row] = relative(scores, self.columns["seat"][row])


def _replay_bola(part: Arrays, row: int, kept: dict[int, int], rows: _Rows) -> None:
    """Replay Bola ``row`` of ``part`` and add a row for each of its decisions with a choice."""
    bola = Bola(int(part["seed"][row]), DEFAULT_CONFIG)
    game = int(part["game"][row])
    actions = part["actions"].tolist()
    first = len(rows)
    in_round: list[int] = []
    columns = rows.columns
    for index in range(int(part["log_start"][row]), int(part["log_start"][row + 1])):
        seat = bola.current_player
        if seat is None:
            raise ValueError(f"Bola {game} has no player to act at step {index}")
        legal = bola.legal_actions()
        action = actions[index]
        if len(legal) > 1:
            obs = bola.observation(seat)
            values = np.zeros(NUM_ACTIONS, np.float32)
            k = kept.get(index)
            if k is None:
                kind = KIND_PLAYED
            else:
                kind = int(part["kind"][k])
                stored = part["values"][part["value_start"][k] : part["value_start"][k + 1]]
                values[legal] = stored
            if kind == KIND_CARD:
                rows.card_rows.append(len(rows))
                rows.card_legal.append(legal)
                rows.card_views.append(rl_view(obs, DEFAULT_CONFIG))
            in_round.append(len(rows))
            for name, value in (
                ("bola", game),
                ("seat", seat),
                ("kind", kind),
                ("action", action),
                ("x", raw_view(obs, legal)),
                ("values", values),
                ("holder", holders(hands_now(bola, obs), seat)),
                ("tricks", None),
                ("made", None),
                ("round_score", None),
                ("final", None),
                ("card_net", -1),
            ):
                columns[name].append(value)
        played = len(bola.history)
        bola.step(action)
        if len(bola.history) > played:  # the action ended a round
            rows.label_round(in_round, bola.history[-1])
            in_round = []
    if not bola.is_over or bola.scores != tuple(part["scores"][row].tolist()):
        raise ValueError(f"Bola {game} does not replay to its recorded scores")
    rows.label_bola(range(first, len(rows)), bola.scores)


def build_shard(part: Arrays, card_net: HybridBot | None) -> Shard:
    """The training rows of one recording part. ``card_net``: the Hybrid whose card net's own
    top card is kept at the valued card decisions (None leaves ``card_net`` at -1)."""
    kept = {int(step): k for k, step in enumerate(part["step"].tolist())}
    rows = _Rows()
    for row in range(len(part["seed"])):
        _replay_bola(part, row, kept, rows)
    if card_net is not None and rows.card_rows:
        values = card_net.policy.card_values(np.stack(rows.card_views))
        for row, legal, cards in zip(rows.card_rows, rows.card_legal, values, strict=True):
            rows.columns["card_net"][row] = max(legal, key=lambda a: cards[a - PLAY_BASE])
    return {name: np.array(rows.columns[name], DTYPES[name]) for name in DTYPES}


def record_config(record: Path) -> RecordConfig:
    return RecordConfig(**json.loads((record / "config.json").read_text()))


def build(record: Path, out: Path, card_net: bool = True) -> dict[str, Any]:
    """Build every missing shard from the recording in ``record``; returns the summary."""
    config = record_config(record)
    if card_net and config.player.split(":")[0] != "hybrid":
        raise ValueError(
            f"the recording's player {config.player!r} has no card net: use --no-card-net"
        )
    bot = make_bot(config.player, 0) if card_net else None
    if bot is not None and not isinstance(bot, HybridBot):
        raise ValueError(f"the recording's player {config.player!r} is not a Hybrid")
    out.mkdir(parents=True, exist_ok=True)
    starts = part_starts(config)
    started = time.time()
    for done, (first, _) in enumerate(starts, 1):
        target = shard_path(out, first)
        if target.exists():
            continue
        save_shard(target, build_shard(load_part(part_path(record, first)), bot))
        elapsed = time.time() - started
        print(f"[data] {done}/{len(starts)} shards, {elapsed / 60:.1f} min", flush=True)
    summary = summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize(out: Path) -> dict[str, Any]:
    rows = kinds = 0
    counts = {"bids": 0, "valued_cards": 0, "played_cards": 0}
    bolas: set[int] = set()
    for path in shard_paths(out):
        with np.load(path) as data:
            kind = data["kind"]
            bolas.update(np.unique(data["bola"]).tolist())
        rows += len(kind)
        counts["bids"] += int((kind == KIND_BID).sum())
        counts["valued_cards"] += int((kind == KIND_CARD).sum())
        counts["played_cards"] += int((kind == KIND_PLAYED).sum())
        kinds += 1
    return {"shards": kinds, "bolas": len(bolas), "rows": rows, **counts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m estimation.onenet.data", description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="the recording's folder")
    parser.add_argument("--out", type=Path, required=True, help="where the shards go")
    parser.add_argument("--no-card-net", action="store_true", help="skip the card net's picks")
    args = parser.parse_args(argv)
    summary = build(args.record, args.out, card_net=not args.no_card_net)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
