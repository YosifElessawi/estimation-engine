"""The state encoder (SPEC §6 M4): what each block of features holds."""

import dataclasses

import numpy as np
import pytest

from estimation.bots import Bot, HeuristicBot, RandomBot
from estimation.engine import (
    CLUBS,
    DASH_CALL,
    DIAMONDS,
    HEARTS,
    NO_DASH_CALL,
    PASS,
    SPADES,
    WITH,
    Bola,
    Observation,
    Phase,
    Round,
    RoundResult,
    RulesConfig,
)
from estimation.engine.cards import cards_of
from estimation.rl.encoder import (
    AUCTION,
    CONTEXT,
    HAND,
    NUM_FEATURES,
    OFFSETS,
    PLAYER_FEATURES,
    PLAYERS,
    SCORES,
    SHAPE,
    TRICK,
    VOIDS,
    encode,
    stakes,
)
from tests.engine.helpers import NO_DASH_ROUND, bid, build_play, est, new_round, run

NO, DASH = NO_DASH_CALL, DASH_CALL
PLAYED = OFFSETS["played"]
TABLE = OFFSETS["table"]
UNSEEN = OFFSETS["unseen"]


def bits(features: np.ndarray, offset: int) -> list[int]:
    """The card ids set in the 52-card block at ``offset``."""
    return [card for card in range(52) if features[offset + card]]


def player(features: np.ndarray, relative: int, field: int) -> float:
    return float(features[PLAYERS + PLAYER_FEATURES * relative + field])


def g1_round() -> tuple[Round, list[int]]:
    """SPEC G1: Caller seat 0 at 5 Spades; seats 1, 2 and 3 estimate 3, 4 and 2. Returns the
    round ready for play and a legal card sequence for it."""
    hands, cards = build_play(SPADES, leader=0, targets=(5, 3, 3, 2))
    rnd = Round.new(1, hands)
    run(rnd, *NO_DASH_ROUND, bid(5, SPADES), PASS, PASS, PASS, PASS, est(3), est(4), est(2))
    return rnd, cards


def test_about_420_features_in_the_spec_blocks():
    assert sum((52, 156, 52, 52, 12, 14, 23, 13, 56, 13, 9)) == NUM_FEATURES
    assert 400 <= NUM_FEATURES <= 460
    assert list(OFFSETS) == [
        "hand",
        "played",
        "table",
        "unseen",
        "voids",
        "trick",
        "context",
        "auction",
        "players",
        "scores",
        "shape",
    ]


def test_card_blocks_hold_hand_plays_table_and_unseen_cards_seat_relative():
    rnd = new_round(1)
    run(rnd, *NO_DASH_ROUND, bid(4, SPADES), PASS, PASS, PASS, PASS, est(3), est(2), est(2))
    # Seat 0 (Caller) leads; play three tricks and one card of the fourth.
    for _ in range(13):
        rnd.step(rnd.legal_actions()[0])
    obs = rnd.observation(rnd.current_player or 0)
    features = encode(obs)
    me = obs.seat
    assert bits(features, HAND) == cards_of(obs.hand)
    played_by: list[set[int]] = [set(), set(), set(), set()]
    for record in obs.tricks:
        for i, card in enumerate(record.cards):
            played_by[(record.leader + i) % 4].add(card)
    for seat, card in obs.trick:
        played_by[seat].add(card)
    for r in range(1, 4):
        assert set(bits(features, PLAYED + 52 * (r - 1))) == played_by[(me + r) % 4]
    assert bits(features, TABLE) == sorted(card for _, card in obs.trick)
    everything_seen = set(cards_of(obs.hand)).union(*played_by)
    assert set(bits(features, UNSEEN)) == set(range(52)) - everything_seen
    assert len(bits(features, UNSEEN)) == 52 - len(everything_seen)


def test_voids_come_from_failing_to_follow_suit():
    # In G1's scripted play, the last trick is led with the 2 of Spades and the others
    # discard the 2s of other suits: each of them shows out of Spades.
    rnd, cards = g1_round()
    for card in cards[:50]:
        rnd.step(68 + card)
    obs = rnd.observation(rnd.current_player or 0)
    features = encode(obs)
    me = obs.seat
    leader = obs.trick[0][0]
    shown_out = [seat for seat, card in obs.trick if card // 13 != obs.trick[0][1] // 13]
    assert shown_out
    for r in range(1, 4):
        seat = (me + r) % 4
        expected = 1.0 if seat in shown_out else 0.0
        assert features[VOIDS + 4 * (r - 1) + SPADES] == expected, (seat, leader)


def test_the_trick_block_names_the_led_suit_position_and_current_winner():
    rnd = new_round(1)
    run(rnd, *NO_DASH_ROUND, bid(4, SPADES), PASS, PASS, PASS, PASS, est(3), est(2), est(2))
    rnd.step(rnd.legal_actions()[0])
    rnd.step(rnd.legal_actions()[-1])
    obs = rnd.observation(rnd.current_player or 0)
    features = encode(obs)
    (s0, c0), (s1, c1) = obs.trick
    led = c0 // 13
    assert features[TRICK + led] == 1.0
    assert features[TRICK + 4 + 2] == 1.0  # third to play
    if c1 // 13 == led:
        winner, best = (s1, c1) if c1 > c0 else (s0, c0)
    else:
        winner, best = (s1, c1) if c1 // 13 == SPADES else (s0, c0)
    assert features[TRICK + 8 + (winner - obs.seat) % 4] == 1.0
    assert features[TRICK + 12] == float(best // 13 == SPADES)
    assert features[TRICK + 13] == pytest.approx((best % 13) / 12)


def test_stakes_follow_the_score_table_for_G1():
    rnd, _ = g1_round()
    obs = rnd.observation(1)
    # Caller: 10 + 5 and the Caller bonus, or a one-trick miss and the bonus lost.
    assert stakes(obs, 0, RulesConfig()) == (25, -11)
    assert stakes(obs, 1, RulesConfig()) == (13, -1)
    assert stakes(obs, 3, RulesConfig()) == (12, -1)
    # Stakes are computed from the rules, so a rule change shows up in the inputs.
    assert stakes(obs, 0, RulesConfig(win_base=20, caller_with_bonus=5)) == (30, -6)
    features = encode(obs)
    assert player(features, 3, 4) == pytest.approx(0.25)  # seat 0 is three seats after seat 1
    assert player(features, 3, 5) == pytest.approx(-0.11)
    assert player(features, 3, 8) == 1.0  # seat 0 is the Caller
    assert player(features, 2, 11) == 1.0  # seat 3 is the last estimator


def test_super_call_flag_and_the_contract_of_8_or_more_are_separate_inputs():
    # L7: a fixed-trump round (Hearts) where seat 0 super-calls 8 Spades.
    rnd = run(new_round(16, HEARTS), est(2), bid(8, SPADES), PASS, PASS, PASS, PASS)
    features = encode(rnd.observation(1))
    assert features[CONTEXT + 10] == 1.0  # super call made (FIX-4)
    assert features[CONTEXT + 11] == 1.0  # contract of 8 or more (SCR-6)
    assert features[CONTEXT + 4] == 1.0  # fixed-trump round
    assert features[CONTEXT + 5 + HEARTS] == 1.0
    assert features[CONTEXT + 12 + SPADES] == 1.0  # the trump is now Spades
    # An open round bid to 8 is a contract of 8 or more without any super call.
    open_round = run(new_round(1), *NO_DASH_ROUND, bid(8, HEARTS), PASS, PASS, PASS, PASS)
    features = encode(open_round.observation(1))
    assert (features[CONTEXT + 10], features[CONTEXT + 11]) == (0.0, 1.0)
    # And a 5 Hearts contract is neither.
    small = run(new_round(1), *NO_DASH_ROUND, bid(5, HEARTS), PASS, PASS, PASS, PASS)
    features = encode(small.observation(1))
    assert (features[CONTEXT + 10], features[CONTEXT + 11]) == (0.0, 0.0)


def test_the_contract_threshold_follows_the_rules():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(8, HEARTS), PASS, PASS, PASS, PASS)
    features = encode(rnd.observation(1), RulesConfig(super_call_min=9))
    assert features[CONTEXT + 11] == 0.0


def test_auction_block_shows_the_top_bid_and_the_final_turn():
    rnd = run(new_round(1), *NO_DASH_ROUND, bid(5, HEARTS), WITH, PASS, PASS)
    obs = rnd.observation(0)  # seat 1's With took it out, so seat 0 has its final turn
    assert obs.phase is Phase.AUCTION
    assert obs.current_player == 0
    features = encode(obs)
    assert features[AUCTION] == 1.0
    assert features[AUCTION + 1] == pytest.approx(1 / 9)  # 5 tricks
    assert features[AUCTION + 2 + HEARTS] == 1.0
    assert features[AUCTION + 7] == 1.0  # the top bidder is the acting seat itself
    assert features[AUCTION + 11] == 1.0  # its final turn
    assert player(features, 1, 9) == 1.0  # seat 1 is With...
    assert player(features, 1, 13) == pytest.approx(5 / 13)  # ...at 5


def test_scoreboard_is_seat_relative_with_standings_weights():
    bola = Bola(3)
    bot = HeuristicBot()
    while bola.round_no < 3:
        seat = bola.current_player or 0
        bola.step(bot.act(bola.observation(seat), bola.legal_actions()))
    obs = bola.observation(2)
    features = encode(obs)
    scores = obs.scores
    for r in range(4):
        assert features[SCORES + r] == pytest.approx(scores[(2 + r) % 4] / 200)
    assert features[SCORES + 4] == pytest.approx((max(scores) - scores[2]) / 100)
    for r in range(1, 4):
        other = scores[(2 + r) % 4]
        expected = 1.0 if other >= scores[2] else 0.5
        assert features[SCORES + 4 + r] == expected
    assert features[SCORES + 9] == pytest.approx(11 / 13)  # rounds 3 to 13 still open
    assert features[SCORES + 10] == 1.0  # all five fixed trumps to come
    assert 0.0 < features[SCORES + 12] < 1.0


def test_hand_shape_is_from_the_whole_hand_dealt():
    rnd = new_round(1)
    features = encode(rnd.observation(0))
    hand = rnd.observation(0).hand
    for suit in range(4):
        length = len([c for c in cards_of(hand) if c // 13 == suit])
        assert features[SHAPE + suit] == pytest.approx(length / 13)


def round_ends(bot: Bot, seeds: range, rules: RulesConfig) -> list[tuple[Observation, RoundResult]]:
    """For every round played, the last card decision's Observation and the round's result."""
    found: list[tuple[Observation, RoundResult]] = []
    for seed in seeds:
        bola = Bola(seed, rules)
        while not bola.is_over:
            seat = bola.current_player or 0
            obs = bola.observation(seat)
            last_card = obs.phase is Phase.PLAY and len(obs.tricks) == 12 and len(obs.trick) == 3
            bola.step(bot.act(obs, bola.legal_actions()))
            if last_card:
                found.append((obs, bola.history[-1]))
    return found


@pytest.mark.parametrize(
    "rules",
    [
        RulesConfig(),
        RulesConfig(dash_replaces_base=False),
        RulesConfig(super_call_scoring=False),
        RulesConfig(calls_count_risk_and_dash=True),
        RulesConfig(all_lose_doubles_next=False, carry_stacks=False),
        RulesConfig(win_base=15, caller_with_bonus=5, risk_points=20, dash_over=20, dash_under=40),
    ],
    ids=["default", "dash-adds-base", "no-super-scoring", "calls-count", "all-lose", "points"],
)
def test_stakes_match_the_engines_scores(rules: RulesConfig):
    checked = {"win": 0, "lose": 0}
    ends = round_ends(RandomBot(7), range(6), rules) + round_ends(HeuristicBot(), range(2), rules)
    for obs, result in ends:
        if result.multiplier == 0:
            continue  # SCR-9 zeroed the round
        for seat in range(4):
            row = result.seats[seat]
            win, lose = stakes(obs, seat, rules)
            without_only = row.score - row.only_points * row.multiplier
            if row.won:
                assert without_only == win, (seat, row)
                checked["win"] += 1
                continue
            flat_loss = (row.is_super and rules.super_call_scoring) or (
                row.is_dash and rules.dash_replaces_base
            )
            if flat_loss or abs(row.estimate - row.tricks) == 1:
                assert without_only == lose, (seat, row)
                checked["lose"] += 1
    assert checked["win"] > 50
    assert checked["lose"] > 50


def test_every_feature_stays_small_over_whole_games():
    bot = RandomBot(1)
    worst = 0.0
    for seed in range(3):
        bola = Bola(seed)
        while not bola.is_over:
            seat = bola.current_player or 0
            obs = bola.observation(seat)
            features = encode(obs)
            assert features.dtype == np.float32
            assert features.shape == (NUM_FEATURES,)
            worst = max(worst, float(np.abs(features).max()))
            bola.step(bot.act(obs, bola.legal_actions()))
    assert worst <= 5.0


def test_encoding_is_a_pure_function_of_the_observation():
    rnd, _ = g1_round()
    obs = rnd.observation(2)
    assert np.array_equal(encode(obs), encode(dataclasses.replace(obs)))


def g3_round(rules: RulesConfig) -> Round:
    """SPEC G3, ready for play: seat 2 Dash Calls, Caller seat 3 at 6 Clubs, seats 0 and 1
    estimate 4 each (total 14, Over)."""
    hands, _ = build_play(CLUBS, leader=3, targets=(4, 3, 0, 6))
    rnd = run(Round.new(1, hands, config=rules), NO, NO, DASH, NO)
    return run(rnd, bid(5, DIAMONDS), PASS, bid(6, CLUBS), PASS, PASS, est(4), est(4))


@pytest.mark.parametrize(
    ("rules", "dash_win"), [(RulesConfig(), 25), (RulesConfig(dash_replaces_base=False), 35)]
)
def test_stakes_of_a_dash_call_in_an_over_round_follow_G3(rules: RulesConfig, dash_win: int):
    obs = g3_round(rules).observation(0)
    assert obs.over
    win, lose = stakes(obs, 2, rules)
    assert win == dash_win
    assert lose == (-25 if rules.dash_replaces_base else -26)
    assert stakes(obs, 3, rules) == (26, -11)  # Caller at 6: 10 + 6 + 10, or -1 - 10


@pytest.mark.parametrize(
    ("rules", "caller"),
    [(RulesConfig(), (74, -42)), (RulesConfig(super_call_scoring=False), (28, -11))],
)
def test_stakes_of_a_super_call_follow_G5(rules: RulesConfig, caller: tuple[int, int]):
    hands, _ = build_play(SPADES, leader=0, targets=(8, 2, 1, 2))
    rnd = run(Round.new(1, hands, config=rules), *NO_DASH_ROUND)
    run(rnd, bid(8, SPADES), PASS, PASS, PASS, PASS, est(2), est(1), est(3))
    # G5: 64 for the super call (or 10 + 8 without SCR-6) and 10 for the Caller.
    assert stakes(rnd.observation(1), 0, rules) == caller


def test_context_and_player_blocks_once_estimation_is_complete():
    obs = g3_round(RulesConfig()).observation(3)
    features = encode(obs)
    assert features[CONTEXT + 3] == 1.0  # play phase
    assert features[CONTEXT + 17] == pytest.approx(1 / 13)  # total 14: Over by 1
    assert features[CONTEXT + 18] == 1.0  # estimation complete
    assert features[CONTEXT + 19] == 0.0  # no carry
    assert features[CONTEXT + 20] == 0.0  # one Caller, no With: x1
    assert features[CONTEXT + 21] == 0.0  # no Risk
    assert features[CONTEXT + 22] == 0.0  # no trick played yet
    # Seat 3 itself (relative 0) is the Caller at 6; seat 2 (relative 3) Dash Called.
    assert player(features, 0, 0) == pytest.approx(6 / 13)
    assert player(features, 0, 1) == 1.0
    assert player(features, 0, 3) == pytest.approx(6 / 13)
    assert (player(features, 0, 6), player(features, 0, 7)) == (1.0, 0.0)
    assert player(features, 0, 8) == 1.0
    assert player(features, 3, 10) == 1.0
    assert player(features, 3, 0) == 0.0
    assert player(features, 3, 1) == 1.0  # a Dash caller's estimate of 0 is known
    assert player(features, 2, 11) == 1.0  # seat 1 was the last estimator


def test_tricks_and_alive_or_broken_follow_play():
    rnd, cards = g1_round()
    for card in cards[:40]:
        rnd.step(68 + card)
    obs = rnd.observation(rnd.current_player or 0)
    features = encode(obs)
    left = 13 - len(obs.tricks)
    assert features[CONTEXT + 22] == pytest.approx(len(obs.tricks) / 13)
    for r in range(4):
        seat = (obs.seat + r) % 4
        estimate, won = obs.estimates[seat], obs.tricks_won[seat]
        assert estimate is not None
        alive = won <= estimate <= won + left
        assert player(features, r, 2) == pytest.approx(won / 13)
        assert player(features, r, 3) == pytest.approx((estimate - won) / 13)
        assert (player(features, r, 6), player(features, r, 7)) == (float(alive), float(not alive))


def test_carry_and_the_three_call_double_set_the_multiplier():
    # G6's auction: Caller seat 0 at 4 Clubs, seat 1 With, seat 2 estimates 4 and is With too,
    # so 3 Callers and Withs double the round; with a carry of 2 it plays x4.
    hands, _ = build_play(CLUBS, leader=0, targets=(5, 3, 2, 3))
    rnd = run(Round.new(1, hands, carry=2), *NO_DASH_ROUND)
    run(rnd, bid(4, CLUBS), WITH, PASS, PASS, PASS, est(4), est(0))
    obs = rnd.observation(0)
    features = encode(obs)
    assert features[CONTEXT + 19] == pytest.approx(1 / 3)  # log2(2) / 3
    assert features[CONTEXT + 20] == pytest.approx(2 / 3)  # log2(4) / 3
    # Every stake is scaled by the x4: the Caller wins 10 + 4 + 10 = 24, times 4.
    assert stakes(obs, 0, RulesConfig()) == (96, -44)
