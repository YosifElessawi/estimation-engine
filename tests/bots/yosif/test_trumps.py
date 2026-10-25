"""The trump-split model (design 5.1): Yosif's trump counts as worked examples."""

from estimation.bots.yosif.trumps import split_tricks, trump_odds

RANKS = "23456789TJQKA"


def pattern(ranks: str) -> int:
    """A 13-bit rank pattern from rank letters, for example ``"AK543"``."""
    return sum(1 << RANKS.index(rank) for rank in ranks)


def mean(odds: tuple[float, ...]) -> float:
    return sum(k * p for k, p in enumerate(odds))


def test_split_tricks_ak_plus_three_small_even_split_is_4():
    # The A and K draw 8 trumps, 5 are left and I hold 3, so I lose 1.
    assert split_tricks(pattern("AK543"), [pattern("Q96"), pattern("J82"), pattern("T7")]) == 4


def test_split_tricks_akq_plus_three_small_takes_all_6_on_a_3_2_2_split():
    assert split_tricks(pattern("AKQ543"), [pattern("JT9"), pattern("87"), pattern("62")]) == 6


def test_split_tricks_akq_plus_three_small_loses_1_to_a_guarded_jack():
    assert split_tricks(pattern("AKQ543"), [pattern("J982"), pattern("T7"), pattern("6")]) == 5


def test_split_tricks_akqj97_takes_6():
    assert split_tricks(pattern("AKQJ97"), [pattern("T65"), pattern("84"), pattern("32")]) == 6


def test_trump_odds_akqj97_is_almost_always_6():
    odds = trump_odds(pattern("AKQJ97"))
    assert odds[6] >= 0.85
    assert abs(sum(odds) - 1) < 1e-9


def test_trump_odds_akq543_mostly_6_sometimes_5():
    odds = trump_odds(pattern("AKQ543"))
    assert odds[6] == max(odds)
    assert odds[5] > 0.2


def test_trump_odds_a_long_opponent_lowers_the_count():
    assert mean(trump_odds(pattern("AK5432"), 1)) < mean(trump_odds(pattern("AK5432"))) - 0.3


def test_trump_odds_is_deterministic():
    assert trump_odds(pattern("AK543")) == trump_odds.__wrapped__(pattern("AK543"), 0)


def test_trump_odds_no_trumps_and_all_trumps():
    assert trump_odds(0) == (1.0,)
    assert trump_odds((1 << 13) - 1) == (0.0,) * 13 + (1.0,)
