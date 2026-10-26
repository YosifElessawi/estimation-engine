"""Training games report each seat's round reward (SPEC §6 M4)."""

import pytest

from estimation.bots import HeuristicBot, RandomBot
from estimation.engine import Bola, RulesConfig
from estimation.rl.games import RewardedBola
from estimation.rl.standings import RewardConfig, round_reward, rounds_remaining


def rounds_left(bola: Bola) -> int:
    obs = bola.observation(0)
    return rounds_remaining(obs.round_no, len(obs.fixed_left), obs.fixed_trump is not None)


@pytest.mark.parametrize("bot", [HeuristicBot(), RandomBot(5)], ids=["heuristic", "random"])
def test_each_round_reports_the_reward_of_every_seat_divided_by_its_carry(bot: HeuristicBot):
    config = RewardConfig()
    game = RewardedBola(12, RulesConfig(), config)
    shadow = Bola(12)
    reported = 0
    fixed_rounds = skipped = carried = 0
    while not game.bola.is_over:
        before = shadow.scores
        fixed = shadow.round.fixed_trump is not None
        left_before = rounds_left(shadow)
        seat = game.bola.current_player or 0
        action = bot.act(game.bola.observation(seat), game.bola.legal_actions())
        rewards = game.step(action)
        played = len(shadow.history)
        shadow.step(action)
        if len(shadow.history) == played:
            assert rewards is None
            continue
        result = shadow.history[-1]
        left_after = 0 if shadow.is_over else rounds_left(shadow)
        expected = tuple(
            round_reward(before, result.scores, s, fixed, left_before, left_after, config)
            / result.carry
            for s in range(4)
        )
        assert rewards == pytest.approx(expected)
        reported += 1
        fixed_rounds += fixed
        skipped += result.skipped
        carried += result.carry > 1
        if result.skipped:
            assert rewards == (0.0, 0.0, 0.0, 0.0)
    assert reported == len(shadow.history)
    assert fixed_rounds >= 5
    if isinstance(bot, RandomBot):
        assert skipped > 0
        assert carried > 0
