"""The JAX rules must match the engine move for move (a quick slice of jaxenv.check)."""

from estimation.jaxenv.check import SCRIPTS, check_games, check_rounds


def test_rounds_match_the_engine():
    assert check_rounds(300, seed=11, batch=150) > 10_000


def test_games_match_the_engine():
    assert check_games(6, seed=12) > 3_000


def test_scripted_auctions_match_the_engine():
    assert check_rounds(len(SCRIPTS) * 3, seed=13, batch=1_000, scripts=SCRIPTS) > 1_000


def test_gpu_and_numpy_views_match():
    assert check_rounds(60, seed=14, views=True) > 2_000
