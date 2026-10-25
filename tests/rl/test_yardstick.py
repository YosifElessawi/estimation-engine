"""The yardstick: a fixed duplicate match that rates the nets the same way every time."""

from pathlib import Path

import pytest

from estimation.arena.duplicate import SetResult
from estimation.rl.config import config_from_dict
from estimation.rl.yardstick import PIMC_MATCH, REFERENCE_MATCH, Match, matches, summarize

MATCH = Match("pimc", "rl:checkpoint=a.pt", "pimc:samples=8")


def a_set(index: int, a: int, b: int) -> SetResult:
    """A set where the bot (slots 0 and 2) scores ``a`` and the opponent ``b`` in every game.
    In rotation k seat i plays slot (i + k) % 4, so the bot sits at seats k % 2 and k % 2 + 2."""
    scores = tuple((a, b, a, b) if k % 2 == 0 else (b, a, b, a) for k in range(4))
    return SetResult(index, index, scores, (18, 18, 18, 18))


def test_sets_won_rsa_and_first_places_come_from_the_bots_seats():
    summary = summarize([a_set(0, 30, 0), a_set(1, 0, 0), a_set(2, 0, 30)], MATCH)
    assert summary["sets"] == 3
    assert summary["opponent"] == "pimc:samples=8"
    assert summary["won"]["mean"] == pytest.approx(0.5)  # won, tied (half), lost
    assert summary["rsa"]["mean"] == pytest.approx(0.0)
    assert summary["won"]["low"] < 0.5 < summary["won"]["high"]
    # Two bot seats tie on top in a won set: half a first place each.
    assert summary["first"]["mean"] == pytest.approx((0.5 + 0.25 + 0.0) / 3)
    won = summarize([a_set(0, 30, 0)] * 2, MATCH)
    assert won["won"]["mean"] == 1.0
    assert won["rsa"]["mean"] == pytest.approx(30 - 30 / 3)


def test_the_reference_match_is_played_only_when_configured():
    config = config_from_dict({}).selfplay
    found = matches(config, Path("runs/x/iter-0025.pt"))
    assert [m.name for m in found] == [PIMC_MATCH]
    assert found[0].lineup == (
        "rl:checkpoint=runs/x/iter-0025.pt",
        "pimc:samples=8",
        "rl:checkpoint=runs/x/iter-0025.pt",
        "pimc:samples=8",
    )
    config = config_from_dict({"selfplay": {"yardstick_reference": "ref.pt"}}).selfplay
    found = matches(config, Path("now.pt"))
    assert [m.name for m in found] == [PIMC_MATCH, REFERENCE_MATCH]
    assert found[1].lineup == ("rl:checkpoint=now.pt", "rl:checkpoint=ref.pt") * 2
