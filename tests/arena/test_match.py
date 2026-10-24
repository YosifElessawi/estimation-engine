"""Playing one Bola with four bots (SPEC §6 M2)."""

import pytest

from estimation.arena.match import BotError, play_bola
from estimation.arena.timing import Timing
from estimation.bots import HeuristicBot, RandomBot
from estimation.engine import Bola, Observation, Phase


class SeatRecorder:
    """Plays like HeuristicBot and remembers which seats' observations it was shown."""

    def __init__(self) -> None:
        self.seats: set[int] = set()
        self.inner = HeuristicBot()

    def act(self, obs: Observation, legal: list[int]) -> int:
        self.seats.add(obs.seat)
        return self.inner.act(obs, legal)


class Cheater:
    def act(self, obs: Observation, legal: list[int]) -> int:
        return 999


def test_play_bola_plays_the_whole_game():
    bots = [HeuristicBot(), RandomBot(1), HeuristicBot(), RandomBot(2)]
    record = play_bola(3, bots)
    bola = Bola(3)
    again = [HeuristicBot(), RandomBot(1), HeuristicBot(), RandomBot(2)]
    while not bola.is_over:
        seat = bola.current_player
        assert seat is not None
        bola.step(again[seat].act(bola.observation(seat), bola.legal_actions()))
    assert record.scores == bola.scores
    assert record.rounds == len(bola.history)


def test_each_bot_is_shown_only_its_own_seat():
    bots = [SeatRecorder() for _ in range(4)]
    play_bola(0, bots)
    assert [bot.seats for bot in bots] == [{0}, {1}, {2}, {3}]


def test_an_illegal_action_stops_the_game_naming_the_seat():
    with pytest.raises(BotError, match="seat 0"):
        play_bola(0, [Cheater(), HeuristicBot(), HeuristicBot(), HeuristicBot()])


def test_only_card_decisions_are_timed():
    timers = [Timing() for _ in range(4)]
    bots = [RandomBot(seat) for seat in range(4)]
    record = play_bola(5, bots, timers=timers)
    bola = Bola(5)
    again = [RandomBot(seat) for seat in range(4)]
    plays = [0, 0, 0, 0]
    while not bola.is_over:
        seat = bola.current_player
        assert seat is not None
        if bola.phase is Phase.PLAY:
            plays[seat] += 1
        bola.step(again[seat].act(bola.observation(seat), bola.legal_actions()))
    assert record.scores == bola.scores
    assert [timer.count for timer in timers] == plays
