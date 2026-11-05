"""BiddingNet and CardPlayNet (SPEC §6 M4): MLPs of about 540K parameters each.

- **BiddingNet** predicts the expected round reward of each of the 68 non-card actions (No Dash
  Call, Dash Call, the 50 bids, With, Pass and the 14 estimates; super calls use the bid ids).
- **CardPlayNet** has a critic, the expected reward of the position (deal luck), and an
  advantage head, how much better or worse each of the 52 cards is than that (skill). It plays
  the legal card with the highest advantage.

Each net is a 3-layer trunk over the encoded state and an output layer with one value per
action, plus a small action head shared by all actions. The action head sees a few facts about
each action, derived inside the net from the state features: for a card, its suit and rank,
whether it is trump or of the led suit, whether it would win the trick now, whether it is the top
card left in its suit, and suit lengths and voids; for a bid or estimate, its trick count and
trump and the hand's expected tricks in that trump. Sharing that head lets what one card or bid
teaches carry over to the others (with one independent output per card, imitating PIMC
generalised badly to new games).

A third, smaller job has a net of its own: **GuessNet** guesses who holds each unseen card. For
each of the 52 cards it gives a logit for each of the other three seats, counted from the acting
seat (the next seat counterclockwise first). It is trained on the real hands, which training
games know, from the same encoded Observation; no action value reads it, so it cannot change how
the bot plays, and the Hybrid can use it to imagine deals. It is a separate net because the play
nets keep little of that information: a guess head on RL v2's trunk could not beat a blind guess.

Rewards are predicted in units of ``REWARD_SCALE`` points.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from estimation.engine.actions import (
    DASH_CALL,
    ESTIMATE_BASE,
    FIRST_BID,
    LAST_BID,
    NO_DASH_CALL,
    PASS,
    PLAY_BASE,
    WITH,
)

from .encoder import (
    AUCTION,
    CONTEXT,
    ENCODER_VERSION,
    HAND,
    NUM_FEATURES,
    OFFSETS,
    SHAPE,
    TRICK,
    VOIDS,
)

BID_ACTIONS = PLAY_BASE  # action ids 0-67: everything but card play
CARDS = 52
HIDDEN = 384
HEAD_HIDDEN = 128
REWARD_SCALE = 50.0
CHECKPOINT_FORMAT = 2
UNSEEN = OFFSETS["unseen"]
CARD_FEATURES = 15
BID_FEATURES = 17
GUESS_SEATS = 3  # the three other seats, counted from the acting seat


def _trunk(hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(NUM_FEATURES, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class ActionHead(nn.Module):
    """One value per action: ``w_a . h + b_a + v . relu(P h + Q phi(a) + c)``, where ``h`` is the
    trunk's output and ``phi(a)`` the action's features."""

    def __init__(self, hidden: int, actions: int, features: int) -> None:
        super().__init__()
        self.per_action = nn.Linear(hidden, actions)
        self.state = nn.Linear(hidden, HEAD_HIDDEN)
        self.action = nn.Linear(features, HEAD_HIDDEN, bias=False)
        self.out = nn.Linear(HEAD_HIDDEN, 1)

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        joint = torch.relu(self.state(h)[:, None, :] + self.action(phi))
        return self.per_action(h) + self.out(joint)[:, :, 0]


def _card_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per card id: rank, suit one-hot (52 x 4), and the same-suit higher and lower matrices
    (``higher[c2, c] = 1`` when ``c2`` is a higher card of ``c``'s suit)."""
    cards = torch.arange(CARDS)
    rank = (cards % 13).float()
    suit = nn.functional.one_hot(cards // 13, 4).float()
    same = suit @ suit.T
    higher = same * (rank[:, None] > rank[None, :]).float()
    return rank, suit, higher, higher.T.contiguous()


def card_features(
    x: torch.Tensor,
    rank: torch.Tensor,
    suit: torch.Tensor,
    higher: torch.Tensor,
    lower: torch.Tensor,
) -> torch.Tensor:
    """``(batch, 52, CARD_FEATURES)`` facts about each card in each encoded position."""
    batch = x.shape[0]
    hand = x[:, HAND : HAND + CARDS]
    unseen = x[:, UNSEEN : UNSEEN + CARDS]
    is_trump = x[:, CONTEXT + 12 : CONTEXT + 16] @ suit.T  # Sans makes no suit trump
    is_led = x[:, TRICK : TRICK + 4] @ suit.T
    leading = x[:, TRICK + 4 : TRICK + 5]
    winner_trump = x[:, TRICK + 12 : TRICK + 13]
    winner_rank = x[:, TRICK + 13 : TRICK + 14] * 12
    above = (rank[None, :] > winner_rank + 0.5).float()
    beats = is_trump * (1 - winner_trump + winner_trump * above)
    beats = beats + is_led * (1 - winner_trump) * above
    wins_now = (1 - leading) * beats.clamp(max=1.0)
    higher_unseen = unseen @ higher
    voids = x[:, VOIDS : VOIDS + 12].reshape(batch, 3, 4).sum(1) / 3
    columns = [
        is_trump,
        is_led,
        (rank / 12).expand(batch, CARDS),
        wins_now,
        (higher_unseen < 0.5).float(),  # the top card left in its suit
        higher_unseen / 13,
        (unseen @ suit) @ suit.T / 13,  # cards of its suit in other hands
        (hand @ suit) @ suit.T / 13,  # own length in its suit
        hand @ higher / 13,
        hand @ lower / 13,
        voids @ suit.T,
    ]
    return torch.cat([torch.stack(columns, dim=2), suit[None].expand(batch, CARDS, 4)], dim=2)


def _bid_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per non-card action: type one-hot (6), trick count / 13, trump one-hot (5, bids only),
    and a mask of the estimate actions."""
    kind = torch.zeros(BID_ACTIONS, 6)
    tricks = torch.zeros(BID_ACTIONS)
    trump = torch.zeros(BID_ACTIONS, 5)
    kind[NO_DASH_CALL, 0] = 1
    kind[DASH_CALL, 1] = 1
    for action in range(FIRST_BID, LAST_BID + 1):
        n, t = divmod(action - FIRST_BID, 5)
        kind[action, 2] = 1
        tricks[action] = (n + 4) / 13
        trump[action, t] = 1
    kind[WITH, 3] = 1
    kind[PASS, 4] = 1
    estimate = torch.zeros(BID_ACTIONS)
    for k in range(14):
        kind[ESTIMATE_BASE + k, 5] = 1
        tricks[ESTIMATE_BASE + k] = k / 13
        estimate[ESTIMATE_BASE + k] = 1
    return kind, tricks, trump, estimate


def bid_features(
    x: torch.Tensor,
    kind: torch.Tensor,
    tricks: torch.Tensor,
    trump: torch.Tensor,
    estimate: torch.Tensor,
) -> torch.Tensor:
    """``(batch, 68, BID_FEATURES)`` facts about each non-card action in each position: its
    type, trick count and trump (With joins the top bid; an estimate is in the round's trump),
    the hand's expected tricks in that trump and the gap to the trick count, whether it is 8 or
    more, and for an estimate the Over/Under delta it leaves and whether it equals the
    contract.

    "8 or more" is a fixed property of the action, the default super call size: the net has no
    RulesConfig. Under other rules the state's contract-of-8+ flag and every seat's stakes
    still follow ``super_call_min``, but the nets would need retraining to value such bids."""
    batch = x.shape[0]
    top_exists = x[:, AUCTION : AUCTION + 1]
    top_tricks = (x[:, AUCTION + 1 : AUCTION + 2] * 9 + 4) / 13 * top_exists
    with_mask = torch.zeros(BID_ACTIONS, device=x.device)
    with_mask[WITH] = 1
    n = tricks[None, :] + with_mask[None, :] * top_tricks
    round_trump = x[:, CONTEXT + 12 : CONTEXT + 17]
    top_trump = x[:, AUCTION + 2 : AUCTION + 7]
    action_trump = (
        trump[None]
        + with_mask[None, :, None] * top_trump[:, None, :]
        + estimate[None, :, None] * round_trump[:, None, :]
    )
    expected = (action_trump * x[:, None, SHAPE + 4 : SHAPE + 9]).sum(2)
    delta = estimate[None, :] * (x[:, CONTEXT + 17 : CONTEXT + 18] + tricks[None, :])
    equals = estimate[None, :] * top_exists * ((tricks[None, :] - top_tricks).abs() < 0.01)
    columns = [
        n,
        expected,
        expected - n,
        (n * 13 > 7.5).float(),
        delta,
        equals.float(),
    ]
    return torch.cat(
        [kind[None].expand(batch, -1, -1), action_trump, torch.stack(columns, dim=2)], dim=2
    )


class BiddingNet(nn.Module):
    """Expected reward of each non-card action, in ``REWARD_SCALE`` units."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = _trunk(hidden)
        self.head = ActionHead(hidden, BID_ACTIONS, BID_FEATURES)
        kind, tricks, trump, estimate = _bid_tables()
        self.register_buffer("kind", kind, persistent=False)
        self.register_buffer("tricks", tricks, persistent=False)
        self.register_buffer("trump", trump, persistent=False)
        self.register_buffer("estimate", estimate, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor, self.trunk(x))
        phi = bid_features(
            x,
            cast(torch.Tensor, self.kind),
            cast(torch.Tensor, self.tricks),
            cast(torch.Tensor, self.trump),
            cast(torch.Tensor, self.estimate),
        )
        return cast(torch.Tensor, self.head(h, phi))


class CardPlayNet(nn.Module):
    """The critic ``V(s)`` and the advantage ``A(s, card)`` of each card, in ``REWARD_SCALE``
    units: ``Q(s, card) = V(s) + A(s, card)``."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = _trunk(hidden)
        self.critic = nn.Linear(hidden, 1)
        self.head = ActionHead(hidden, CARDS, CARD_FEATURES)
        rank, suit, higher, lower = _card_tables()
        self.register_buffer("rank", rank, persistent=False)
        self.register_buffer("suit", suit, persistent=False)
        self.register_buffer("higher", higher, persistent=False)
        self.register_buffer("lower", lower, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = cast(torch.Tensor, self.trunk(x))
        phi = card_features(
            x,
            cast(torch.Tensor, self.rank),
            cast(torch.Tensor, self.suit),
            cast(torch.Tensor, self.higher),
            cast(torch.Tensor, self.lower),
        )
        value = cast(torch.Tensor, self.critic(h))[:, 0]
        return value, cast(torch.Tensor, self.head(h, phi))


class GuessNet(nn.Module):
    """Logits ``(batch, 52, 3)`` of which other seat holds each card."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = _trunk(hidden)
        self.out = nn.Linear(hidden, CARDS * GUESS_SEATS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor, self.trunk(x))
        return cast(torch.Tensor, self.out(h)).view(-1, CARDS, GUESS_SEATS)


def parameter_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


@dataclass(slots=True)
class Policy:
    """A saved bot: CardPlayNet, BiddingNet unless it was never trained, GuessNet if one was,
    and the checkpoint's metadata. Runs on the CPU in inference mode."""

    card: CardPlayNet
    bidding: BiddingNet | None
    meta: dict[str, Any]
    guess: GuessNet | None = None

    def card_values(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """The advantage of each of the 52 cards for a batch of encoded positions."""
        with torch.inference_mode():
            _, advantage = self.card(torch.as_tensor(features))
        return advantage.numpy()

    def bid_values(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """The value of each of the 68 non-card actions for a batch of encoded positions."""
        if self.bidding is None:
            raise ValueError("this checkpoint has no trained BiddingNet")
        with torch.inference_mode():
            values = self.bidding(torch.as_tensor(features))
        return values.numpy()

    def card_guesses(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """GuessNet's guess, ``(batch, 52, 3)``: for each card, the chance that each other seat
        holds it, counted from the acting seat. Meaningful for the unseen cards only."""
        if self.guess is None:
            raise ValueError("this checkpoint has no trained GuessNet")
        with torch.inference_mode():
            logits = self.guess(torch.as_tensor(features))
        return torch.softmax(logits, dim=2).numpy()


def state_dict_of(net: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of ``net``'s weights."""
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def save_checkpoint(
    path: Path,
    card: CardPlayNet,
    bidding: BiddingNet | None,
    meta: Mapping[str, Any],
    guess: GuessNet | None = None,
) -> None:
    """Save a bot: its nets' weights, the encoder it reads and JSON-like ``meta``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "encoder": ENCODER_VERSION,
        "num_features": NUM_FEATURES,
        "hidden": card.hidden,
        "card": state_dict_of(card),
        "bidding": None if bidding is None else state_dict_of(bidding),
        "guess": None if guess is None else state_dict_of(guess),
        "meta": dict(meta),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


FIRST_LAYER = "trunk.0.weight"  # (hidden, features): the only weights that see the inputs
UPGRADABLE = {1: 452, 2: 456}  # older encoder versions whose features begin this version's


def widen(state: Mapping[str, torch.Tensor], features: int) -> dict[str, torch.Tensor]:
    """``state``, saved for ``features`` inputs, with zero weights for the inputs appended
    since: the net gives exactly the same values as before, whatever the new inputs hold, and
    training learns their weights from there."""
    weight = state[FIRST_LAYER]
    extra = torch.zeros(weight.shape[0], NUM_FEATURES - features, dtype=weight.dtype)
    return {**state, FIRST_LAYER: torch.cat([weight, extra], dim=1)}


def load_policy(path: Path) -> Policy:
    """Load a checkpoint saved by ``save_checkpoint`` for play on the CPU. A checkpoint of an
    older encoder whose features this one only appends to (``UPGRADABLE``) is widened by
    ``widen``, and its meta records ``upgraded_from_encoder``."""
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an M4 checkpoint (format {payload.get('format')})")
    encoder, features = int(payload["encoder"]), int(payload["num_features"])
    card_state = cast(dict[str, torch.Tensor], payload["card"])
    bidding_state = cast(dict[str, torch.Tensor] | None, payload["bidding"])
    guess_state = cast(dict[str, torch.Tensor] | None, payload.get("guess"))
    meta = dict(payload["meta"])
    if (encoder, features) != (ENCODER_VERSION, NUM_FEATURES):
        if UPGRADABLE.get(encoder) != features:
            raise ValueError(
                f"{path} was trained on encoder version {encoder} with {features} features; "
                f"this code has version {ENCODER_VERSION} with {NUM_FEATURES}"
            )
        card_state = widen(card_state, features)
        if bidding_state is not None:
            bidding_state = widen(bidding_state, features)
        if guess_state is not None:
            guess_state = widen(guess_state, features)
        meta["upgraded_from_encoder"] = encoder
    hidden = int(payload["hidden"])
    card = CardPlayNet(hidden)
    card.load_state_dict(card_state)
    card.eval()
    bidding = None
    if bidding_state is not None:
        bidding = BiddingNet(hidden)
        bidding.load_state_dict(bidding_state)
        bidding.eval()
    guess = None
    if guess_state is not None:
        guess = GuessNet(int(guess_state["out.weight"].shape[1]))
        guess.load_state_dict(guess_state)
        guess.eval()
    return Policy(card, bidding, meta, guess)
