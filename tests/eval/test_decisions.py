"""Finding a bot's bidding decisions in a saved game, and the choices to try (design 5.4)."""

from estimation.arena.games import GameRecord
from estimation.arena.match import play_bola
from estimation.bots import make_bot
from estimation.engine import PASS, SPADES, WITH, RulesConfig, bid_action, estimate_action
from estimation.eval.decisions import KINDS, candidates, decision_at, decisions_of
from tests.bots.helpers import round_with

RULES = RulesConfig()


def recorded_game(spec: str = "heuristic", seed: int = 3) -> GameRecord:
    bots = [make_bot(spec, seed * 4 + seat) for seat in range(4)]
    played = play_bola(seed, bots, keep_actions=True)
    seats = (spec, spec, spec, spec)
    return GameRecord(0, 0, seed, seats, (0, 0, 0, 0), played.scores, played.rounds, played.actions)


def test_decisions_skip_forced_moves():
    decisions = decisions_of(recorded_game(), RULES, "heuristic")
    assert decisions
    assert all(len(d.legal) > 1 for d in decisions)
    assert all(d.chosen in d.legal for d in decisions)


def test_kinds_cover_open_final_turn_and_estimate():
    kinds = {d.kind for d in decisions_of(recorded_game(), RULES, "heuristic")}
    assert {"dash", "open", "final turn", "estimate", "last estimate"} <= kinds
    assert kinds <= set(KINDS)


def test_only_the_target_bots_decisions_are_taken():
    record = recorded_game()
    mixed = GameRecord(
        0,
        0,
        record.deal_seed,
        ("heuristic", "x", "x", "x"),
        (0, 0, 0, 0),
        record.scores,
        record.rounds,
        record.actions,
    )
    assert {d.seat for d in decisions_of(mixed, RULES, "heuristic")} == {0}


def test_candidates_are_legal_and_start_with_the_choice():
    rnd = round_with(
        0,
        "AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D",
        *[0] * 4,
        bid_action(7, SPADES),
        PASS,
        PASS,
        PASS,
    )
    d = decision_at(rnd.observation(0), rnd.legal_actions(), PASS)
    assert d.kind == "final turn"
    options = candidates(d, bid_action(7, SPADES))
    assert options[0] == PASS
    assert bid_action(8, SPADES) in options
    assert len(options) <= 8
    assert set(options) <= set(d.legal)
    assert WITH not in options


def test_estimate_candidates_reach_two_either_side():
    rnd = round_with(
        1,
        "AS KS QS JS 9S 7S QH TH 3H AD QD 7D 4D",
        *[0] * 4,
        bid_action(5, 0),
        PASS,
        PASS,
        PASS,
        PASS,
    )
    d = decision_at(rnd.observation(1), rnd.legal_actions(), estimate_action(3))
    assert d.kind == "estimate"
    options = candidates(d, estimate_action(3))
    assert {estimate_action(k) for k in (1, 2, 3, 4, 5)} <= set(options)


def test_a_passed_opening_still_tries_opening_bids():
    rnd = round_with(0, "2C 3C 4C 5C 2D 3D 4D 5D 2H 3H 4H 5H 6S", *[0] * 4)
    d = decision_at(rnd.observation(0), rnd.legal_actions(), PASS)
    assert d.kind == "open"
    options = candidates(d, PASS)
    assert options[0] == PASS
    assert {bid_action(4, trump) for trump in range(5)} <= set(options)
