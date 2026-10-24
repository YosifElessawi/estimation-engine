"""Rollouts (SPEC §6 M3): playing a round out with HeuristicBot's card play for every seat must
match four HeuristicBots playing it in the engine."""

import random

from estimation.bots.heuristic import HeuristicBot
from estimation.bots.rollout import finish_round, trick_winner_offset
from estimation.engine import Phase, Round, deal, trick_winner


def test_the_trick_winner_matches_the_engine():
    rng = random.Random(1)
    for _ in range(5000):
        cards = rng.sample(range(52), 4)
        trump = rng.randint(0, 4)
        leader = rng.randint(0, 3)
        trick = [((leader + i) % 4, card) for i, card in enumerate(cards)]
        expected = trick_winner(trick, trump)
        offset = trick_winner_offset(cards[0], cards[1], cards[2], cards[3], trump)
        assert (leader + offset) % 4 == expected


def test_finishing_a_round_plays_like_heuristic_bots_in_the_engine():
    bot = HeuristicBot()
    rng = random.Random(2)
    compared = 0
    for seed in range(150):
        round_no = 1 + seed % 18
        fixed = None if round_no <= 13 else (4, 3, 2, 1, 0)[round_no - 14]
        rnd = Round.new(round_no, deal(seed, round_no), fixed)
        while rnd.phase is not Phase.PLAY and rnd.phase is not Phase.DONE:
            seat = rnd.current_player
            assert seat is not None
            rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
        if rnd.phase is Phase.DONE:
            continue
        for _ in range(rng.randint(0, 40)):  # random plays first, so positions vary
            rnd.step(rng.choice(rnd.legal_actions()))
        views = [rnd.observation(seat) for seat in range(4)]
        obs = views[0]
        estimates = [e if e is not None else 0 for e in obs.estimates]
        assert obs.trump is not None
        assert obs.leader is not None
        tricks = finish_round(
            [view.hand for view in views],
            [card for _, card in obs.trick],
            obs.leader,
            obs.trump,
            estimates,
            list(obs.tricks_won),
        )
        while rnd.phase is not Phase.DONE:
            seat = rnd.current_player
            assert seat is not None
            rnd.step(bot.act(rnd.observation(seat), rnd.legal_actions()))
        assert tuple(tricks) == tuple(seat.tricks for seat in rnd.result().seats)
        compared += 1
    assert compared > 120


def test_a_complete_trick_is_scored_before_play_goes_on():
    # Seat 0 led 2C, seats 1-3 followed: the trick goes to seat 2's AC, which leads next.
    hands = [1 << 13, 1 << 14, 1 << 15, 1 << 16]  # one Diamond each: 2D 3D 4D 5D
    tricks = finish_round(hands, [0, 1, 12, 2], 0, 4, [0, 0, 2, 0], [0, 0, 0, 0])
    assert tricks == [0, 0, 1, 1]  # seat 2 wins the Clubs, seat 3's 5D wins the Diamonds
