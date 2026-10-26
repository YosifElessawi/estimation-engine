"""TBL-7: right after the deal, every seat missing a suit is announced void to everyone."""

from estimation.engine import SANS, Bola, Phase, Round, mask_of
from estimation.engine.cards import card_id

from .helpers import NO_DASH_ROUND, run


def suit_cards(suit: int, ranks: range) -> list[int]:
    return [card_id(rank, suit) for rank in ranks]


# Seat 2 has no Clubs; seats 0, 1 and 3 hold all four suits.
CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
HANDS = (
    mask_of(suit_cards(CLUBS, range(4)) + suit_cards(DIAMONDS, range(5, 8))
            + suit_cards(HEARTS, range(4, 7)) + suit_cards(SPADES, range(4, 7))),
    mask_of(suit_cards(CLUBS, range(4, 9)) + suit_cards(DIAMONDS, range(8, 11))
            + suit_cards(HEARTS, range(7, 10)) + suit_cards(SPADES, range(7, 9))),
    mask_of(suit_cards(DIAMONDS, range(5)) + suit_cards(HEARTS, range(4))
            + suit_cards(SPADES, range(4))),
    mask_of(suit_cards(CLUBS, range(9, 13)) + suit_cards(DIAMONDS, range(11, 13))
            + suit_cards(HEARTS, range(10, 13)) + suit_cards(SPADES, range(9, 13))),
)  # fmt: skip


def test_TBL_7_a_seat_missing_a_suit_is_announced_void_before_the_first_decision():
    rnd = Round.new(1, HANDS)
    assert rnd.phase is Phase.DASH  # nobody has decided anything yet
    for seat in range(4):
        assert rnd.observation(seat).void_announced == (2,)


def test_TBL_7_the_announcement_lasts_the_whole_round():
    rnd = run(Round.new(1, HANDS), *NO_DASH_ROUND)
    while rnd.phase is not Phase.DONE:
        rnd.step(rnd.legal_actions()[0])
        assert rnd.observation(0).void_announced == (2,)


def test_TBL_7_fixed_trump_rounds_announce_too():
    rnd = Round.new(14, HANDS, SANS)
    assert rnd.phase is Phase.ESTIMATION
    assert rnd.observation(1).void_announced == (2,)


def test_TBL_7_announcement_says_only_who_not_which_suit_or_how_many():
    two_voids = mask_of(suit_cards(SPADES, range(13)))  # seat 0 misses three suits
    rest = [card for card in range(52) if card // 13 != SPADES]
    hands = (two_voids, mask_of(rest[0:13]), mask_of(rest[13:26]), mask_of(rest[26:39]))
    obs = Round.new(1, hands).observation(3)
    assert 0 in obs.void_announced
    assert all(isinstance(seat, int) for seat in obs.void_announced)


def test_TBL_7_matches_the_dealt_hands_in_whole_games():
    for seed in range(3):
        bola = Bola(seed)
        while not bola.is_over:
            rnd = bola.round
            missing = tuple(
                seat
                for seat in range(4)
                if any(not (rnd.hands[seat] >> (13 * suit)) & 0x1FFF for suit in range(4))
            )
            assert bola.observation(0).void_announced == missing
            bola.step(bola.legal_actions()[0])
