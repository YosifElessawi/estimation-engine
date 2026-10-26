"""Re-dealing the unseen cards and playing each choice out (design 5.4)."""

import random

from estimation.engine import PASS, SPADES, Phase, Round, RulesConfig, bid_action
from estimation.engine.cards import FULL_DECK, parse_hand
from estimation.eval.decisions import decision_at, decisions_of
from estimation.eval.playout import redeal, value_table
from tests.bots.helpers import round_with
from tests.eval.test_decisions import recorded_game

RULES = RulesConfig()
EXAMPLE = "AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D"


def test_redeal_keeps_the_deciders_hand_and_the_void_announcements():
    record = recorded_game()
    d = decisions_of(record, RULES, "heuristic")[0]
    rng = random.Random(0)
    for _ in range(20):
        hands = redeal(d, rng)
        assert hands[d.seat] == d.obs.hand
        assert hands[0] | hands[1] | hands[2] | hands[3] == FULL_DECK
        assert sum(h.bit_count() for h in hands) == 52
        rnd = Round.new(d.round_no, hands, d.fixed_trump, d.carry, d.scores, RULES)
        assert rnd.observation(d.seat).void_announced == d.obs.void_announced


def test_fixed_trump_decisions_play_out_legally():
    record = recorded_game()
    fixed = [d for d in decisions_of(record, RULES, "heuristic") if d.fixed_trump is not None]
    assert fixed
    table = value_table(fixed[0], [fixed[0].chosen], record.seats, 2, "heuristic", 0, RULES)
    assert set(table) == {fixed[0].chosen}


def natural_round(*actions: int) -> Round:
    """Round 1 with the example hand in seat 0 and the other 39 cards dealt round-robin, so no
    other seat is void (a sorted deal would announce all three as void)."""
    mine = parse_hand(EXAMPLE)
    rest = [card for card in range(52) if not mine >> card & 1]
    hands = [mine, 0, 0, 0]
    for i, card in enumerate(rest):
        hands[1 + i % 3] |= 1 << card
    rnd = Round.new(1, (hands[0], hands[1], hands[2], hands[3]))
    for action in actions:
        rnd.step(action)
    return rnd


def test_example_final_turn_raising_to_9_loses_points():
    # The 7 Spades against 8 Spades verdict depends on how hard the card player forces extra
    # tricks (measured 2026-10-26: a tie with PIMC-4 cards, 8 ahead by about 13 with full PIMC),
    # so this checks what holds under any card player: 9 Spades needs both queens and loses.
    rnd = natural_round(*[0] * 4, bid_action(7, SPADES), PASS, PASS, PASS)
    assert rnd.phase is Phase.AUCTION
    assert rnd.observation(0).void_announced == (0,)  # only its own club void
    d = decision_at(rnd.observation(0), rnd.legal_actions(), PASS)
    seats = ("yosif", "pimc", "pimc", "pimc")
    choices = [PASS, bid_action(8, SPADES), bid_action(9, SPADES)]
    table = value_table(d, choices, seats, 48, "pimc:samples=4", 0, RULES)
    assert table[bid_action(9, SPADES)][0] < table[PASS][0]
    assert table[bid_action(9, SPADES)][0] < table[bid_action(8, SPADES)][0]


def test_value_table_is_deterministic_for_a_seed():
    rnd = round_with(0, EXAMPLE, *[0] * 4, bid_action(7, SPADES), PASS, PASS, PASS)
    d = decision_at(rnd.observation(0), rnd.legal_actions(), PASS)
    seats = ("heuristic",) * 4
    first = value_table(d, [PASS], seats, 3, "heuristic", 5, RULES)
    assert first == value_table(d, [PASS], seats, 3, "heuristic", 5, RULES)
