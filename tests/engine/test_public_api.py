"""The SPEC §5.4 names are importable from ``estimation.engine``."""

import estimation.engine as engine


def test_spec_api_names_are_exported():
    for name in (
        "deal",
        "score_round",
        "next_carry",
        "legal_plays",
        "trick_winner",
        "Round",
        "Bola",
        "Observation",
        "RoundOutcome",
        "RoundResult",
        "RulesConfig",
        "IllegalActionError",
        "Phase",
    ):
        assert hasattr(engine, name), name
        assert name in engine.__all__


def test_phases_are_the_spec_five():
    assert [p.name for p in engine.Phase] == ["DASH", "AUCTION", "ESTIMATION", "PLAY", "DONE"]
