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
