"""The Why panel: what an RL bot's nets predicted for each legal option of one decision.

For a card, CardPlayNet's critic values the position and its advantage head says how much better
or worse each card is than that; the predicted reward of a card is the two added. For any other
decision (Dash Call, bid, With, Pass, estimate, super call), BiddingNet predicts each action's
reward directly. Every value is converted from the nets' ``REWARD_SCALE`` units to points.

The reward is the M4 training reward: the bot's round score change minus a weighted mean of the
opponents' (SPEC §6 M4), at a carry of x1. The bot plays the legal option with the highest value,
so the first option is the one it chose unless it plays with noise.

Only spectators get this panel: it is computed from the bot's own Observation, so it reveals that
seat's hand.
"""

from typing import TypedDict, cast

import numpy as np
import torch

from estimation.bots import Bot, NoisyBot
from estimation.bots.rl import RLBot, policy_at
from estimation.engine import Observation, Phase
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode
from estimation.rl.networks import REWARD_SCALE

from .views import ActionView, describe_action


class OptionView(TypedDict):
    """One legal option: its predicted reward in points and, for a card, its advantage."""

    action: ActionView
    points: float
    advantage: float | None
    chosen: bool


class WhyView(TypedDict):
    """``net`` is CardPlayNet, BiddingNet, or HeuristicBot when a warm-start checkpoint (no
    BiddingNet) left the call to HeuristicBot. ``baseline`` is the critic's value of the
    position, for cards. ``options`` are sorted best first, ties in legal-action order, the
    order the bot breaks them in."""

    net: str
    baseline: float | None
    options: list[OptionView]
    note: str | None


def rl_bot(bot: Bot) -> RLBot | None:
    """The RL bot behind ``bot`` (unwrapping noise), if it is one."""
    inner = bot.inner if isinstance(bot, NoisyBot) else bot
    return inner if isinstance(inner, RLBot) else None


def explain(bot: Bot, obs: Observation, legal: list[int], chosen: int) -> WhyView | None:
    """What ``bot``'s nets predicted for each legal action at ``obs``, if it is an RL bot."""
    rl = rl_bot(bot)
    if rl is None:
        return None
    policy = policy_at(rl.config.checkpoint)
    features = torch.as_tensor(encode(obs, rl.config.rules)[np.newaxis])
    if obs.phase is Phase.PLAY:
        with torch.inference_mode():
            value, advantage = cast(tuple[torch.Tensor, torch.Tensor], policy.card(features))
        baseline = float(value[0]) * REWARD_SCALE
        advantages = advantage[0].numpy()
        ranked = sorted(legal, key=lambda action: -advantages[action - PLAY_BASE])
        options = [
            OptionView(
                action=describe_action(obs, action),
                points=round(baseline + float(advantages[action - PLAY_BASE]) * REWARD_SCALE, 2),
                advantage=round(float(advantages[action - PLAY_BASE]) * REWARD_SCALE, 2),
                chosen=action == chosen,
            )
            for action in ranked
        ]
        return WhyView(net="CardPlayNet", baseline=round(baseline, 2), options=options, note=None)
    if rl.config.heuristic_bidding:
        return WhyView(
            net="HeuristicBot",
            baseline=None,
            options=[],
            note="This checkpoint has no BiddingNet, so HeuristicBot made this call.",
        )
    values = policy.bid_values(features.numpy())[0]
    ranked = sorted(legal, key=lambda action: -values[action])
    options = [
        OptionView(
            action=describe_action(obs, action),
            points=round(float(values[action]) * REWARD_SCALE, 2),
            advantage=None,
            chosen=action == chosen,
        )
        for action in ranked
    ]
    return WhyView(net="BiddingNet", baseline=None, options=options, note=None)
