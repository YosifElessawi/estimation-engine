"""RLBot (SPEC §6 M4): a trained checkpoint in the arena through the bot registry."""

from pathlib import Path

import pytest

from estimation.arena.duplicate import run_arena
from estimation.arena.match import play_bola
from estimation.bots import HeuristicBot, make_bot
from estimation.bots.rl import RLBot
from estimation.engine import (
    DASH_CALL,
    PASS,
    Bola,
    Phase,
    RulesConfig,
    play_action,
)
from estimation.engine.actions import PLAY_BASE
from tests.rl.helpers import make_checkpoint


def test_the_registry_builds_an_rl_bot_from_a_checkpoint_path(tmp_path: Path):
    path = make_checkpoint(tmp_path / "net.pt")
    rules = RulesConfig(super_call_min=9)
    bot = make_bot(f"rl:checkpoint={path}", 4, rules)
    assert isinstance(bot, RLBot)
    assert bot.config.checkpoint == str(path)
    assert bot.config.rules == rules
    assert bot.config.heuristic.super_call_min == 9
    assert not bot.config.heuristic_bidding
    with_heuristic = make_bot(f"rl:checkpoint={path}:heuristic_bidding=true", 0)
    assert isinstance(with_heuristic, RLBot)
    assert with_heuristic.config.heuristic_bidding


def test_the_rl_bot_needs_a_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_bot("rl", 0)
    with pytest.raises(FileNotFoundError):
        make_bot(f"rl:checkpoint={tmp_path / 'missing.pt'}", 0)


def test_a_card_play_only_checkpoint_needs_heuristic_bidding(tmp_path: Path):
    path = make_checkpoint(tmp_path / "card.pt", bidding=False)
    with pytest.raises(ValueError, match="heuristic_bidding=true"):
        make_bot(f"rl:checkpoint={path}", 0)
    assert isinstance(make_bot(f"rl:checkpoint={path}:heuristic_bidding=true", 0), RLBot)


def test_card_play_takes_the_legal_card_with_the_highest_advantage(tmp_path: Path):
    favourite = 51  # the Ace of Spades
    bot = make_bot(
        f"rl:checkpoint={make_checkpoint(tmp_path / 'a.pt', favourite_card=favourite)}", 0
    )
    bola = Bola(11)
    heuristic = HeuristicBot()
    played_favourite = checked = 0
    while not bola.is_over and checked < 400:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        if obs.phase is Phase.PLAY:
            choice = bot.act(obs, legal)
            cards = [action - PLAY_BASE for action in legal]
            # The rigged advantages: the favourite first, then lower card ids.
            expected = favourite if favourite in cards else min(cards)
            assert choice == play_action(expected)
            played_favourite += expected == favourite and len(legal) > 1
            checked += 1
        bola.step(heuristic.act(obs, legal))
    assert checked == 400
    assert played_favourite > 0


def test_bidding_takes_the_legal_action_with_the_highest_value(tmp_path: Path):
    path = make_checkpoint(tmp_path / "b.pt", favourite_action=DASH_CALL)
    bot = make_bot(f"rl:checkpoint={path}", 0)
    bola = Bola(5)
    obs = bola.observation(0)
    assert obs.phase is Phase.DASH
    assert bot.act(obs, bola.legal_actions()) == DASH_CALL
    passer = make_bot(
        f"rl:checkpoint={make_checkpoint(tmp_path / 'p.pt', favourite_action=PASS)}", 0
    )
    for _ in range(4):
        bola.step(0)  # no Dash Calls
    obs = bola.observation(bola.current_player or 0)
    assert obs.phase is Phase.AUCTION
    assert passer.act(obs, bola.legal_actions()) == PASS


def test_heuristic_bidding_copies_heuristicbot_off_the_cards(tmp_path: Path):
    path = make_checkpoint(tmp_path / "c.pt", bidding=False)
    bot = make_bot(f"rl:checkpoint={path}:heuristic_bidding=true", 0)
    heuristic = HeuristicBot()
    bola = Bola(2)
    compared = 0
    while not bola.is_over and bola.round_no <= 3:
        seat = bola.current_player or 0
        obs = bola.observation(seat)
        legal = bola.legal_actions()
        if obs.phase is not Phase.PLAY:
            assert bot.act(obs, legal) == heuristic.act(obs, legal)
            compared += 1
        bola.step(bot.act(obs, legal))
    assert compared > 10


def test_four_rl_bots_play_a_whole_bola_legally_and_the_seed_does_not_matter(tmp_path: Path):
    path = make_checkpoint(tmp_path / "d.pt", seed=8)
    first = play_bola(3, [make_bot(f"rl:checkpoint={path}", seat) for seat in range(4)])
    again = play_bola(3, [make_bot(f"rl:checkpoint={path}", 100 + seat) for seat in range(4)])
    assert first == again
    assert first.rounds >= 18


def test_the_arena_plays_rl_bots_the_same_on_one_or_two_workers(tmp_path: Path):
    path = make_checkpoint(tmp_path / "e.pt", seed=9)
    lineup = (f"rl:checkpoint={path}", "heuristic", "heuristic", "heuristic")
    one = run_arena(lineup, 2, seed=1, workers=1)
    two = run_arena(lineup, 2, seed=1, workers=2)
    assert one.results == two.results
    assert one.timings[lineup[0]].count > 0
