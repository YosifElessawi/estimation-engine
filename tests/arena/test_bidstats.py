"""Bidding stats from saved games: per bot, per role and per table (design 5.2)."""

from pathlib import Path

import pytest

from estimation.arena.bidstats import main, play_games, stats_from_games, stats_from_rounds
from estimation.arena.games import GamesHeader, write_games
from estimation.engine import RoundOutcome, RulesConfig, score_round

RULES = RulesConfig()
SEATS = ("a", "b", "a", "b")
G1 = score_round(  # Over (14): Caller seat 0 makes 5, seat 2 takes 3 of its 4 (SPEC G1)
    RoundOutcome(estimates=(5, 3, 4, 2), tricks=(5, 3, 3, 2), caller=0, last_estimator=3, trump=3)
)
G2 = score_round(  # Under (9): seat 2 With makes 4, last estimator seat 0 takes 3 for 0 (SPEC G2)
    RoundOutcome(
        estimates=(0, 4, 4, 1),
        tricks=(3, 4, 4, 2),
        caller=1,
        with_players=(2,),
        last_estimator=0,
        risks=2,
        trump=2,
    )
)
SKIPPED = score_round(RoundOutcome(skipped=True))


def test_stats_count_roles_tables_and_the_last_estimator():
    bots, tables = stats_from_rounds([(SEATS, G1), (SEATS, G2), (SEATS, SKIPPED)])
    a = bots["a"]
    assert a.made_by_role == {"Caller": 1.0, "With": 1.0, "plain": 0.0, "last": 0.0}
    assert a.contract_share == {"5": 1.0}
    assert a.contract_made == {"5": 1.0}
    assert a.over_tricks == pytest.approx(1 / 4)
    assert a.under_tricks == pytest.approx(1 / 4)
    assert (a.last_over, a.last_under, a.last_risk) == (0.0, 1.0, 1.0)
    assert a.dash_share == 0.0
    assert a.with_share == pytest.approx(1 / 4)
    (table,) = tables.values()
    assert (table.rounds, table.skipped) == (3, pytest.approx(1 / 3))
    assert (table.over, table.under) == (0.5, 0.5)
    assert table.mean_total == pytest.approx(11.5)


def header(version: str) -> GamesHeader:
    return GamesHeader(version, {}, ("heuristic",) * 4, 1, 0, "2026-10-26T00:00:00+00:00")


def test_mixed_code_versions_are_refused_unless_allowed():
    records = play_games("heuristic", 1, 0, 1)
    with pytest.raises(ValueError, match="code versions"):
        stats_from_games([(header("one"), records), (header("two"), records)], RULES)
    stats_from_games([(header("one"), records), (header("two"), records)], RULES, allow_mixed=True)


def test_saved_games_give_the_same_stats_as_playing(tmp_path: Path):
    records = play_games("heuristic", 1, 0, 1)
    played, _, replayed = stats_from_games([(None, records)], RULES)
    path = tmp_path / "games.jsonl.gz"
    write_games(path, header("x"), records)
    assert main(["--from", str(path), "--out", str(tmp_path / "s.json")]) == 0
    from_file, _, stats = stats_from_games([(header("x"), records)], RULES)
    assert from_file == played
    assert replayed.kept == stats.kept == 1
    assert (tmp_path / "s.json").exists()
