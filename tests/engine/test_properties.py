"""Property tests over thousands of random Bolas (SPEC §6, M1 Done when 4 and 5).

2,000 random Bolas under the default config, plus 200 with each bool flag flipped. Every game
must end without exceptions, and each played round must use all 52 cards once, never total 13
in estimates, have a loser, and score components x multiplier = score. Game scores must be the
sum of round scores and the fixed trumps must follow FIX-1 and FIX-5. Under the default config
the run also checks that no observation reveals another seat's cards or a pending Dash choice,
and that replaying a recorded action log reproduces the same round and the same game.
"""

import dataclasses
import random

import pytest

from estimation.engine.actions import DASH_CALL, NO_DASH_CALL
from estimation.engine.bola import Bola
from estimation.engine.cards import FULL_DECK, SUIT_MASKS
from estimation.engine.config import RulesConfig
from estimation.engine.observation import Observation, Phase
from estimation.engine.round import FIXED_ORDER, Round

DEFAULT = RulesConfig()
DEFAULT_BOLAS = 2000
FLAG_BOLAS = 200
CHUNK = 100
BOOL_FLAGS = [f.name for f in dataclasses.fields(RulesConfig) if f.type is bool]
OTHER_VARIANTS = [
    ("max_fixed_replays", 0),
    ("max_fixed_replays", 10_000),
    ("super_call_min", 11),
    ("calls_to_double", 2),
]
MAX_STEPS = 50_000
CARD_CHECK_RATE = 0.015
ROUND_REPLAY_RATE = 0.1
BOLA_REPLAY_EVERY = 10


def test_flags_under_test_are_the_seven_rules_config_switches():
    assert BOOL_FLAGS == [
        "only_bonus_for_dash_calls",
        "only_bonus_for_super_calls",
        "dash_replaces_base",
        "super_call_scoring",
        "calls_count_risk_and_dash",
        "all_lose_doubles_next",
        "carry_stacks",
    ]


@pytest.mark.parametrize("chunk", range(DEFAULT_BOLAS // CHUNK))
def test_random_bolas_under_the_default_config(chunk: int):
    for seed in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        bola, actions = run_bola(seed, DEFAULT, deep=True)
        if seed % BOLA_REPLAY_EVERY == 0:
            check_bola_replay(bola, actions)


@pytest.mark.parametrize("chunk", range(FLAG_BOLAS // CHUNK))
@pytest.mark.parametrize("flag", BOOL_FLAGS)
def test_random_bolas_with_each_flag_flipped(flag: str, chunk: int):
    config = dataclasses.replace(DEFAULT, **{flag: not getattr(DEFAULT, flag)})
    start = 100_000 + 1_000 * BOOL_FLAGS.index(flag) + chunk * CHUNK
    for seed in range(start, start + CHUNK):
        run_bola(seed, config, deep=False)


@pytest.mark.parametrize(("field", "value"), OTHER_VARIANTS)
def test_random_bolas_with_other_rule_values(field: str, value: int):
    config = dataclasses.replace(DEFAULT, **{field: value})
    for seed in range(200_000, 200_000 + CHUNK):
        run_bola(seed, config, deep=False)


def run_bola(seed: int, config: RulesConfig, *, deep: bool) -> tuple[Bola, list[int]]:
    """Play one random Bola, checking invariants at every step and after every round."""
    bola = Bola(seed, config)
    policy = random.Random(seed * 7919 + 1)
    sampler = random.Random(seed * 104_729 + 3)
    actions: list[int] = []
    for _ in range(MAX_STEPS):
        if bola.is_over:
            break
        legal = bola.legal_actions()
        assert legal, f"seed {seed}: no legal action before the game ended"
        rnd = bola.round
        if deep:
            if rnd.phase is Phase.DASH and len(rnd.log) == 3:
                check_dash_privacy(rnd)
            if sampler.random() < CARD_CHECK_RATE:
                check_card_privacy(rnd, sampler)
        action = policy.choice(legal)
        bola.step(action)
        actions.append(action)
        if bola.round is not rnd or bola.is_over:
            check_round(rnd)
            if deep and sampler.random() < ROUND_REPLAY_RATE:
                check_round_replay(rnd)
    assert bola.is_over, f"seed {seed}: the game did not end in {MAX_STEPS} steps"
    check_game(bola)
    return bola, actions


def check_round(rnd: Round) -> None:
    result = rnd.result()
    assert result.scores == tuple(seat.score for seat in result.seats)
    for seat in result.seats:
        assert seat.points * seat.multiplier == seat.score
    if result.skipped:
        assert rnd.fixed_trump is None
        assert result.scores == (0, 0, 0, 0)
        return
    tricks = rnd.observation(0).tricks
    assert len(tricks) == 13
    seen = 0
    for trick in tricks:
        for i, card in enumerate(trick.cards):
            assert not seen >> card & 1, "a card was played twice"
            assert rnd.hands[(trick.leader + i) % 4] >> card & 1, "a card came from the wrong hand"
            seen |= 1 << card
    assert seen == FULL_DECK
    assert result.total == sum(seat.estimate for seat in result.seats)
    assert result.total != 13
    assert any(not seat.won for seat in result.seats)
    assert sum(seat.tricks for seat in result.seats) == 13


def check_game(bola: Bola) -> None:
    history = bola.history
    assert [r.round_no for r in history] == list(range(1, len(history) + 1))
    for seat in range(4):
        assert bola.scores[seat] == sum(r.scores[seat] for r in history)
    carry = 1
    for result in history:
        assert result.carry == carry
        carry = result.next_carry
    assert all(r.fixed_trump is None for r in history[:13])
    cap = bola.config.max_fixed_replays
    position = replays = 0
    for result in history[13:]:
        assert position < len(FIXED_ORDER), "a fixed-trump round after all five were played"
        assert result.fixed_trump == FIXED_ORDER[position]
        if result.trump_changed and replays < cap:
            replays += 1  # the same fixed trump comes next
        else:
            assert not result.trump_changed or replays == cap
            position += 1
    assert position == len(FIXED_ORDER)
    assert replays == bola.replays


def check_dash_privacy(rnd: Round) -> None:
    """With three Dash choices made, no seat's view depends on another seat's choice."""
    made = [action for _, action in rnd.log]
    for seat in range(4):
        obs = rnd.observation(seat)
        assert all(entry_seat == seat for entry_seat, _ in obs.log)
        assert set(obs.dash_callers) <= {seat}
        assert obs.estimates == (None, None, None, None)
        mine = (seat - rnd.first) % 4
        for k in range(len(made)):
            if k == mine:
                continue
            flipped = made.copy()
            flipped[k] = DASH_CALL if made[k] == NO_DASH_CALL else NO_DASH_CALL
            assert replay(rnd, rnd.hands, flipped).observation(seat) == obs


def check_card_privacy(rnd: Round, sampler: random.Random) -> None:
    """A seat sees its own remaining hand, and swapping unplayed cards between two other seats
    (same suit, so every past play stays legal) leaves its view unchanged."""
    seat = sampler.randrange(4)
    obs = rnd.observation(seat)
    played = played_by_seat(obs)
    assert obs.hand == rnd.hands[seat] & ~played[seat]
    a, b = sampler.sample([s for s in range(4) if s != seat], 2)
    left_a = rnd.hands[a] & ~played[a]
    left_b = rnd.hands[b] & ~played[b]
    for suit_mask in sampler.sample(SUIT_MASKS, 4):
        mine, theirs = left_a & suit_mask, left_b & suit_mask
        if mine and theirs:
            swap = (mine & -mine) | (theirs & -theirs)
            hands = list(rnd.hands)
            hands[a] ^= swap
            hands[b] ^= swap
            other = replay(rnd, (hands[0], hands[1], hands[2], hands[3]), list_actions(rnd))
            assert other.observation(seat) == obs
            return


def check_round_replay(rnd: Round) -> None:
    """SPEC §5.4: a replayed action log gives the same phase, player, views and result."""
    other = replay(rnd, rnd.hands, list_actions(rnd))
    assert other.phase is rnd.phase
    assert other.current_player == rnd.current_player
    assert all(other.observation(s) == rnd.observation(s) for s in range(4))
    assert other.result() == rnd.result()


def check_bola_replay(bola: Bola, actions: list[int]) -> None:
    """Two Bolas with the same seed and actions have identical scores and history."""
    other = Bola(bola.seed, bola.config)
    for action in actions:
        other.step(action)
    assert other.is_over
    assert other.scores == bola.scores
    assert other.history == bola.history


def replay(rnd: Round, hands: tuple[int, int, int, int], actions: list[int]) -> Round:
    return Round.replay(
        rnd.round_no,
        hands,
        rnd.fixed_trump,
        rnd.carry,
        rnd.scores,
        rnd.config,
        actions,
        fixed_left=rnd.fixed_left,
        replays=rnd.replays,
    )


def list_actions(rnd: Round) -> list[int]:
    return [action for _, action in rnd.log]


def played_by_seat(obs: Observation) -> list[int]:
    played = [0, 0, 0, 0]
    for trick in obs.tricks:
        for i, card in enumerate(trick.cards):
            played[(trick.leader + i) % 4] |= 1 << card
    for seat, card in obs.trick:
        played[seat] |= 1 << card
    return played
