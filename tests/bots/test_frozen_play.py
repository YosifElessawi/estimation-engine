"""Frozen play: the measuring sticks and the old bot names play exactly as they always did.

HeuristicBot anchors the Elo ladder, and saved ladder pairs and games name ``pimc``,
``pimc:samples=N`` and ``yosif``, so a change to any of them would quietly move every rating
measured through them. Each test plays seeded games and compares a fingerprint of every action
and score with the one recorded before the Search bot got its own play-out rules (2026-10-27),
under the rules of that day (the With rules changed on 2026-10-30).

HeuristicBot and ``pimc`` are measuring sticks and stay frozen: if one of their tests fails,
undo the change. ``yosif`` is the Search bot's old name, so its tests also prove the merge left
the bot as it was rated; when the Search bot's rules change on purpose, record the new
fingerprints and re-rate it (``--fresh search``).
"""

import hashlib
from collections.abc import Sequence

from estimation.bots import make_bot
from estimation.engine import Bola

from .helpers import LATEST_CALL_ONLY


def fingerprint(lineup: Sequence[str], seeds: Sequence[int], rounds: int | None = None) -> str:
    """A hash of every action and the scores of seeded Bolas with ``lineup`` in seats 0-3,
    stopping each one after ``rounds`` rounds when given."""
    digest = hashlib.sha256()
    for seed in seeds:
        bots = [
            make_bot(spec, seed * 4 + seat, LATEST_CALL_ONLY) for seat, spec in enumerate(lineup)
        ]
        bola = Bola(seed, LATEST_CALL_ONLY)
        while not bola.is_over and (rounds is None or len(bola.history) < rounds):
            seat = bola.current_player
            assert seat is not None
            action = bots[seat].act(bola.observation(seat), bola.legal_actions())
            digest.update(action.to_bytes(1))
            bola.step(action)
        digest.update(repr(bola.scores).encode())
    return digest.hexdigest()[:16]


def test_heuristic_bot_plays_exactly_as_before():
    assert fingerprint(["heuristic"] * 4, range(12)) == "94e5d5d18a3a00fa"


def test_noisy_heuristic_rungs_play_exactly_as_before():
    lineup = ["heuristic:noise=0.75", "heuristic:noise=0.1", "heuristic:noise=0.01", "random"]
    assert fingerprint(lineup, range(6)) == "6f243c2e572b07af"


def test_pimc_measuring_sticks_play_exactly_as_before():
    lineup = ["pimc", "pimc:samples=4", "pimc:samples=1:play_noise=0.3", "heuristic"]
    assert fingerprint(lineup, [0]) == "024ed1537b476786"


def test_yosif_specs_play_exactly_as_before():
    lineup = ["yosif", "yosif:risk_mode=false", "yosif:calibrated=true", "pimc:samples=8"]
    assert fingerprint(lineup, [1], rounds=5) == "957b080ddf894ef7"


def test_yosif_plays_whole_bolas_exactly_as_before():
    assert fingerprint(["yosif", "pimc:samples=2", "yosif", "heuristic"], [2]) == "4cc139ad74b413de"


def test_search_plays_exactly_like_its_old_name_yosif():
    lineup = ["search", "search:risk_mode=false", "search:calibrated=true", "pimc:samples=8"]
    assert fingerprint(lineup, [1], rounds=5) == "957b080ddf894ef7"
