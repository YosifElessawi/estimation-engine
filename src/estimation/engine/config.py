"""Rule switches and point values (SPEC §5.3). Defaults equal the rules in SPEC §2-§3."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RulesConfig:
    win_base: int = 10  # SCR-1: a win scores win_base + B
    caller_with_bonus: int = 10  # SCR-3
    risk_points: int = 10  # SCR-4, per Risk
    only_bonus: int = 10  # SCR-5
    only_bonus_for_dash_calls: bool = True  # SCR-5
    only_bonus_for_super_calls: bool = True  # SCR-5
    dash_over: int = 25  # SCR-2
    dash_under: int = 33  # SCR-2
    dash_replaces_base: bool = True  # SCR-2
    super_call_scoring: bool = True  # SCR-6
    super_call_min: int = 8  # SCR-6 and FIX-4
    calls_to_double: int = 3  # SCR-8
    calls_count_risk_and_dash: bool = False  # SCR-8
    all_lose_doubles_next: bool = True  # SCR-9
    carry_stacks: bool = True  # GAM-4
    max_fixed_replays: int = 20  # FIX-5 safety cap

    def __post_init__(self) -> None:
        if not 4 <= self.super_call_min <= 13:
            raise ValueError("super_call_min must be a bid size, 4 to 13")
        if self.calls_to_double < 1:
            raise ValueError("calls_to_double must be at least 1")
        if self.max_fixed_replays < 0:
            raise ValueError("max_fixed_replays must be 0 or more")


DEFAULT_CONFIG = RulesConfig()
