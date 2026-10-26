# Estimation Engine

A bot for Estimation, the 4-player trick-taking card game, played by the rules of the Pocket Estimation app. It is built one milestone at a time: rules engine, baseline bots and arena, search bot, self-play RL, hybrid, web table.

**SPEC.md is the source of truth.** The rules, engine contract, milestones and conventions all live there. This file only says how to work with it.

## Executing a milestone

When asked to execute a milestone (for example "Execute Milestone 1 from SPEC.md"):

1. Branch from `main`: `git checkout -b m<N>-<short-name>`, for example `m1-rules-engine`.
2. Read SPEC.md: §2-§3 (rules and scoring), §5 (engine contract), the milestone in §6, and §8 (conventions). For M1, also scaffold the project as described in §8 if `pyproject.toml` does not exist yet.
3. Build only that milestone. Do not start work from a later one.
4. Test as you go. Every rule ID (`AUC-3`, `SCR-6`, ...) and every worked example (`G1`-`G10`, `L1`-`L11`, `T1`-`T4`) gets a test whose name contains the ID, for example `test_EST_5_last_estimator_cannot_make_13`. Worked examples are acceptance tests: match their numbers exactly.
5. Run the full gate (below) until it is clean.
6. Run the `spec-checker` subagent on the work and fix everything it reports.
7. Stop and report: the milestone's **Done when** list with pass or fail for each item, the benchmark number (M1), and anything blocked. Do not start the next milestone unprompted.

## Hard rules

- **Implement the rules exactly.** Do not add, drop or "fix" a rule, and never fill a gap with a guess. If a rule is missing, ambiguous or contradicts another, stop and ask Yosif. He confirms rules from playing the app.
- **Spec changes need Yosif.** SPEC.md changes only when Yosif confirms a rule or answers a question. When that happens, update the rule text, its source tag (`[Page]`, `[Play]`, `[Default]`), §4 if a question stays open, the `RulesConfig` default in §5.3, and every worked example it affects.
- **Config mirrors the spec.** `RulesConfig` defaults (§5.3) must equal the rules in §2-§3, and every bool flag is tested both on and off.
- **The action space is frozen.** The 120 action ids in §5.2 never change; RL milestones depend on them.
- **Bots see only their own `Observation`.** Nothing may leak another seat's cards.
- **Ask Yosif before any long or heavy run.** Never start an arena run, an Elo ladder or any other rating, bidding statistics, training, data recording, a long benchmark or any job on several worker processes without asking him first: say what it is, roughly how long it takes and how many workers it uses, and wait for his yes. Check `ps` for other sessions' jobs first; the Mac runs one heavy job at a time. The test gate (pre-commit, pytest on 2 workers) and short runs inside tests need no asking. Watch a rating run with `uv run python -m estimation.arena.watch <its folder>` and a training run with `uv run python -m estimation.rl.watch <run>`.

## Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pre-commit run --all-files
uv run python -m estimation.engine.bench
```

The gate is pre-commit, pytest and pyright all passing.

## Conventions

- Python 3.12 with uv, `src/estimation/` layout, `tests/` mirroring it. Full layout in SPEC.md §8.
- Conventional commits (`feat(engine): add auction`), one branch per milestone, never commit directly to `main`. Yosif merges.
- Every random choice takes an explicit seed, so any game can be replayed.
- The dev Mac has 18 GB of RAM: run tests with at most 2 workers, and one heavy job (training, a big arena run) at a time.
