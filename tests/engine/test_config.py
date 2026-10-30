"""RulesConfig defaults mirror SPEC §2-§3 (SPEC §5.3)."""

import dataclasses

import pytest

from estimation.engine.config import RulesConfig


def test_rules_config_defaults_match_the_spec():
    assert dataclasses.asdict(RulesConfig()) == {
        "win_base": 10,
        "caller_with_bonus": 10,
        "risk_points": 10,
        "only_bonus": 10,
        "only_bonus_for_dash_calls": True,
        "only_winner_for_super_calls": True,
        "only_loser_for_super_calls": True,
        "dash_over": 25,
        "dash_under": 33,
        "dash_replaces_base": True,
        "super_call_scoring": True,
        "super_call_min": 8,
        "super_caller_may_bid_fixed": True,
        "calls_to_double": 3,
        "calls_count_risk_and_dash": False,
        "all_lose_doubles_next": True,
        "carry_stacks": True,
        "max_fixed_replays": 20,
    }


def test_rules_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RulesConfig().win_base = 5  # type: ignore[misc]  # asserting the dataclass is frozen


@pytest.mark.parametrize(
    "kwargs",
    [
        {"super_call_min": 3},
        {"super_call_min": 14},
        {"calls_to_double": 0},
        {"max_fixed_replays": -1},
    ],
)
def test_rules_config_rejects_values_the_engine_cannot_play(kwargs: dict[str, int]):
    with pytest.raises(ValueError, match="must"):
        RulesConfig(**kwargs)  # type: ignore[arg-type]  # values are ints for int fields
