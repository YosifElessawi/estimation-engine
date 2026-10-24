from hypothesis import settings

# Deterministic property tests: every run explores the same examples (SPEC §8, reproducibility).
settings.register_profile("estimation", derandomize=True, database=None, deadline=None)
settings.load_profile("estimation")
