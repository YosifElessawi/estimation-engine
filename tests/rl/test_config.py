"""Training configs (SPEC §6 M4): one TOML file plus a seed defines a run."""

from pathlib import Path

import pytest

from estimation.rl.config import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
    derive_seed,
    load_config,
)


def test_an_empty_file_gives_the_defaults(tmp_path: Path):
    path = tmp_path / "empty.toml"
    path.write_text("")
    assert load_config(path) == TrainConfig()


def test_a_file_sets_values_in_any_section_and_the_seed_can_be_overridden(tmp_path: Path):
    path = tmp_path / "run.toml"
    path.write_text(
        "seed = 4\nworkers = 2\n"
        "[reward]\nahead_weight = 1\nbehind_weight = 0.25\n"
        "[opponents]\npimc_samples = [4, 8]\n"
        "[rules]\ndash_replaces_base = false\n"
        "[selfplay]\niterations = 3\n"
    )
    config = load_config(path)
    assert config.seed == 4
    assert config.workers == 2
    assert config.reward.ahead_weight == 1.0
    assert isinstance(config.reward.ahead_weight, float)
    assert config.reward.behind_weight == 0.25
    assert config.opponents.pimc_samples == (4, 8)
    assert config.rules.dash_replaces_base is False
    assert config.selfplay.iterations == 3
    assert load_config(path, seed=9).seed == 9


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"nonsense": 1}, "unknown config key"),
        ({"reward": {"ahead": 1.0}}, "unknown key 'ahead'"),
        ({"reward": 3}, "must be a table"),
        ({"selfplay": {"iterations": 2.5}}, "must be an integer"),
        ({"selfplay": {"iterations": True}}, "must be an integer"),
        ({"rules": {"carry_stacks": 1}}, "must be true or false"),
        ({"opponents": {"pimc_samples": []}}, "non-empty list"),
        ({"opponents": {"self_play": 0.5}}, "add up to 1"),
        ({"selfplay": {"iterations": 0}}, "at least 1"),
        ({"rules": {"super_call_min": 3}}, "super_call_min"),
        ({"selfplay": {"lr_end_factor": -0.1}}, "lr_end_factor"),
        ({"selfplay": {"yardstick_sets": 0}}, "yardstick_sets"),
        ({"selfplay": {"overlap": 1}}, "must be true or false"),
    ],
)
def test_bad_configs_are_refused(data: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        config_from_dict(data)


def test_a_config_round_trips_through_plain_data():
    config = config_from_dict({"seed": 2, "selfplay": {"games": 32}})
    data = config_to_dict(config)
    assert data["seed"] == 2
    assert data["selfplay"]["games"] == 32
    assert data["rules"]["super_call_min"] == 8


def test_derived_seeds_depend_on_every_key():
    seeds = {derive_seed(0, 1, 2), derive_seed(0, 2, 1), derive_seed(1, 1, 2), derive_seed(0, 1)}
    assert len(seeds) == 4
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)


def test_the_committed_m4_config_holds_the_decided_design():
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / "m4.toml")
    assert config.seed == 0
    # Yosif's M4 decisions (SPEC §6 M4, 2026-10-25).
    assert (config.reward.ahead_weight, config.reward.behind_weight) == (1.0, 0.5)
    assert config.reward.win_blend > 0
    mix = config.opponents
    assert (mix.self_play, mix.heuristic, mix.pimc) == (0.70, 0.15, 0.15)
    assert set(mix.pimc_samples) == {4, 5, 6, 7, 8}
    assert config.record.pimc_samples == 30  # the M3 PIMC bot is the warm start's teacher
    # The design changes recorded in the M4 report.
    assert config.reward.normalize_weights
    assert config.selfplay.card_anchor_weight > 0
    assert config.selfplay.bid_temperature_start > 0
    assert config.nets.hidden == 384


def test_the_staged_config_keeps_the_m4_design_and_changes_only_the_schedules_and_speed():
    configs = Path(__file__).resolve().parents[2] / "configs"
    m4 = config_to_dict(load_config(configs / "m4.toml"))
    staged = config_to_dict(load_config(configs / "m4-staged.toml"))
    for section in ("rules", "reward", "record", "imitation", "opponents", "nets"):
        assert staged[section] == m4[section], section
    allowed = {
        "iterations",
        "checkpoint_every",
        "card_anchor_end_factor",
        "lr_end_factor",
        "overlap",
        "balance",
        "chunk",
        "yardstick_every",
        "yardstick_reference",
    }
    changed = {key for key, value in staged["selfplay"].items() if value != m4["selfplay"][key]}
    assert changed <= allowed
    assert (staged["seed"], staged["workers"]) == (m4["seed"], m4["workers"])
    assert staged["selfplay"]["card_anchor_end_factor"] == 0.1
    assert staged["selfplay"]["lr_end_factor"] == 0.1
