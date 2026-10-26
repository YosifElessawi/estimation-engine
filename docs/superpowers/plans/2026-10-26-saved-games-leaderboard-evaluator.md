# Saved Games, Leaderboard and Bidding Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Save every arena and ladder game, and compute stats from saved games.
- Give the leaderboard plain names in three tables with Elo ranges.
- Add a bidding evaluator that measures the points each bidding decision cost.

**Architecture:**
- `estimation.arena.games` records, writes, reads and replays games.
- The arena and the ladder save through it.
- `bidstats` and the new `estimation.eval` package read from it.
- `estimation.arena.roster` names every bot for the ladder report, the watch view and the web table.
- The evaluator replays a saved game to a decision, deals the hidden cards again, and plays the round out once per candidate choice.

**Tech Stack:** Python 3.12, uv, pytest (`-n 2`), pyright, ruff, gzip JSON lines.

**Spec:** `docs/superpowers/specs/2026-10-26-saved-games-leaderboard-evaluator-design.md`

## Global Constraints

- **Recording never changes a result.** Same seeds give the same scores and ratings, with or without saving.
- **No heavy run starts during this plan.** The arena, ladder, bidstats and evaluator are run only in tests on tiny inputs; real runs need Yosif's yes.
- **Code under `estimation.bots` never imports** `estimation.arena` or `estimation.eval`. The static guard in `tests/bots/test_isolation.py` enforces it.
- **Tests run on at most 2 workers.** The gate is `uv run pre-commit run --all-files`.
- **Conventional commits** ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No em-dashes.
- **Every value marked `[Claude]` in the spec** becomes a named constant or CLI default, with a comment.

## Review Focus

1. **A saved game that no longer replays** (illegal under today's rules): the reader skips it and counts it, and never crashes a report. Tested in Task 1.
2. **Web games with a human seat:** `--bot human` scores the human, and the yosif bot stands in for the human in play-outs. Tested in Task 7.
3. **A decision with one legal action** is never sampled. Tested in Task 6.
4. **A pair report from older code is never reused**, even with the same lineup, sets and seed. Tested in Task 2.
5. **A decision in a fixed-trump round** (estimate or super call) re-deals and plays out legally. Tested in Task 6.

---

### Task 1: Game records

**Files:**
- Create: `src/estimation/arena/games.py`
- Modify: `src/estimation/arena/match.py` (`BolaRecord.actions`, `play_bola(..., keep_actions=False)`), `src/estimation/arena/duplicate.py` (`SetResult.games`, `play_set` and `run_arena` with `keep_games=False`), `src/estimation/arena/__main__.py` (`--save-games PATH`)
- Test: `tests/arena/test_games.py`

**Interfaces (Produces):**
- `GameRecord`: frozen dataclass.

  | Field | Type |
  | :--- | :--- |
  | `set_index` | int |
  | `rotation` | int |
  | `deal_seed` | int |
  | `seats` | tuple[str, str, str, str] |
  | `bot_seeds` | tuple[int, int, int, int] |
  | `scores` | tuple[int, int, int, int] |
  | `rounds` | int |
  | `actions` | tuple[int, ...] |

- `GamesHeader`: frozen dataclass with `code_version: str`, `rules: dict[str, Any]`, `lineup: tuple[str, ...]`, `sets: int`, `seed: int`, `created: str`.
- `code_version() -> str`: `git rev-parse --short HEAD`, plus `+dirty` when `git status --porcelain` lists tracked changes. Returns `"unknown"` outside git.
- `write_games(path: Path, header: GamesHeader, records: Iterable[GameRecord]) -> None`: gzip JSON lines, header first.
- `read_games(path: Path) -> tuple[GamesHeader | None, list[GameRecord]]`. It also reads the web table's `games.jsonl` (no header, `seats[i]["spec"]`, `set_index=-1`, `rotation=0`, `bot_seeds` all 0: replay uses the saved actions, never the bots). A web game's header is None.
- `replay(record: GameRecord, rules: RulesConfig) -> Bola | None`: steps every action through `Bola(record.deal_seed, rules)`. Returns None if an action is illegal or the final scores differ.
- `ReplayStats(kept: int, skipped: int)`, and `replay_all(records, rules) -> tuple[list[tuple[GameRecord, Bola]], ReplayStats]`.

- [ ] **Step 1: Write the failing tests**
  - `test_a_saved_game_replays_to_its_scores`: play 1 heuristic game with `keep_actions=True`, build a record, write, read and replay; the replayed `bola.scores == record.scores`.
  - `test_saving_does_not_change_results`: `run_arena(("heuristic",) * 4, 2, 0)` and the same with `keep_games=True` give equal `results` scores.
  - `test_an_unreplayable_game_is_skipped`: a record with one action changed to an illegal one gives `ReplayStats(kept=0, skipped=1)`.
  - `test_web_games_are_read`: a one-line `games.jsonl` in the web format reads to one record with a None header.
- [ ] **Step 2: Run them and check they fail** (`uv run pytest tests/arena/test_games.py -q`; ImportError).
- [ ] **Step 3: Implement.** `play_bola` appends each stepped action when `keep_actions`; `play_set` builds a `GameRecord` per rotation. Only `run_arena(keep_games=True)` returns games.
- [ ] **Step 4: Run them and check they pass**, plus `uv run pytest tests/arena -q -n 2`.
- [ ] **Step 5: Commit** `feat(arena): save every game a run plays`.

### Task 2: The ladder saves games, reuses only matching code, and has the v2 preset

**Files:**
- Modify: `src/estimation/arena/ladder.py`, `src/estimation/arena/report.py` (`code_version` in the report JSON)
- Test: `tests/arena/test_ladder.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `play_pair(...)` writes `<dir>/games/<pair>.jsonl.gz`.
  - `_saved(...)` also requires `report["code_version"] == code_version()` and an existing games file.
  - `PRESETS: dict[str, Preset]` with `Preset(bots: tuple[str, ...], extras: tuple[tuple[str, str], ...], dir: Path, sets: int, seed: int)`.
  - `PRESETS["v2"]`:
    - bots: the 25 bots of `runs/ladder-yosif/ladder.json` in order, plus `yosif`;
    - extras: the 8 existing non-neighbour pairs, plus `("pimc", "yosif")` and `("heuristic", "yosif")`;
    - `dir=Path("runs/ladder-v2")`, `sets=200`, `seed=0`.
  - CLI `--preset NAME`: the other flags override it.

- [ ] **Step 1: Write the failing tests**
  - `test_a_pair_from_other_code_is_replayed`: a saved pair report with a different `code_version` is not reused. Monkeypatch `code_version` and use a 1-set heuristic pair.
  - `test_a_ladder_pair_saves_its_games`: after `play_pair(..., sets=1)`, the games file holds 4 records that replay.
  - `test_the_v2_preset_lists_26_bots_and_10_extras`: every extra names two bots in the list.
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and check they pass**, plus `uv run pytest tests/arena -q -n 2`.
- [ ] **Step 5: Commit** `feat(arena): save ladder games and add the v2 ladder preset`.

### Task 3: Roster and the three-table leaderboard with Elo ranges

**Files:**
- Create: `src/estimation/arena/roster.py`
- Modify: `src/estimation/arena/ladder.py` (`format_ladder`, the `ladder.json` fields), `src/estimation/arena/ratings.py` (`bootstrap_ranges`)
- Test: `tests/arena/test_roster.py`, `tests/arena/test_ratings.py`

**Interfaces (Produces):**
- `RosterEntry(spec: str, name: str, groups: tuple[str, ...], description: str)`.
- `GROUPS = ("Leaderboard", "RL checkpoints", "Measuring rungs")`.
- `entry_for(spec: str) -> RosterEntry`: unknown specs give `RosterEntry(spec, spec, ("Measuring rungs",), "")`.
- Groups follow the spec exactly:
  - Leaderboard: random, heuristic, pimc, the two RL finals, yosif.
  - RL checkpoints: `warm-start.pt:heuristic_bidding=true`, iterations 10, 20, 60 and 100, M4 final, staged final.
  - Measuring rungs: everything else.
- Names:
  - `heuristic:noise=0.1` is "HeuristicBot, 10% random moves".
  - `pimc:samples=8` is "PIMC, 8 deals".
  - `pimc:samples=1:play_noise=0.3` is "PIMC, 1 deal, 30% random cards".
  - `rl:checkpoint=runs/m4/checkpoints/iter-0020.pt` is "RL iteration 20".
  - The warm start is "RL warm start (HeuristicBot bidding)".
  - `rl:checkpoint=runs/m4/checkpoints/final.pt` is "RL M4 final".
  - `rl:checkpoint=runs/m4-staged/checkpoints/staged-final.pt` is "RL staged final".
  - `yosif` is "Yosif bot".
  - random, heuristic and pimc are "RandomBot", "HeuristicBot" and "PIMC".
- `bootstrap_ranges(results_by_pair: dict[tuple[str, str], tuple[list[SetResult], list[str]]], resamples: int = 200, seed: int = 0) -> dict[str, tuple[float, float]]`:
  - resample each pair's sets with replacement;
  - refit `bradley_terry` with the same prior;
  - take the 2.5% and 97.5% points.
- `format_ladder` prints the three groups: name, Elo, range, and record against its neighbour. The pair table moves below them.
- `ladder.json` gains, per bot, `name`, `groups` and `range`.

- [ ] **Step 1: Write the failing tests**
  - `test_every_v2_bot_has_a_plain_name_and_group`.
  - `test_unknown_specs_fall_back_to_the_spec`.
  - `test_the_rl_finals_sit_in_two_tables`.
  - `test_bootstrap_range_contains_the_fit_and_narrows_with_more_sets`: synthetic `SetResult`s for a 3-bot chain.
  - `test_format_ladder_prints_three_tables_with_names`.
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.** The ranges need set-level results, so `play_pair` keeps each pair's `SetResult`s; reused pairs rebuild them from their saved report.
- [ ] **Step 4: Run them and check they pass.**
- [ ] **Step 5: Commit** `feat(arena): show the ladder as three named tables with Elo ranges`.

### Task 4: The watch view and the web table use roster names

**Files:**
- Modify: `src/estimation/arena/watch.py` (`short()` returns `entry_for(spec).name`), `server/catalog.py` (`label_of` uses the roster name when the spec is in it; `LADDER_FILE` prefers `runs/ladder-v2/ladder.json` when it exists)
- Test: `tests/arena/test_watch.py` (create if missing), `tests/server/test_catalog.py`

- [ ] **Step 1: Write the failing tests**
  - The watch view shows "RL M4 final" for the final checkpoint's spec.
  - The catalog's `label_of("yosif") == "Yosif bot"`.
  - The catalog loads `runs/ladder-v2/ladder.json` when present under the root (tmp_path).
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and check they pass**, plus `uv run pytest tests/server tests/arena -q -n 2`.
- [ ] **Step 5: Commit** `feat(table): name bots by the roster in the watch view and the web table`.

### Task 5: Bidding stats from saved games, per bot, per role and per table

**Files:**
- Modify: `src/estimation/arena/bidstats.py` (rewrite around replayed games)
- Test: `tests/arena/test_bidstats.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `SeatStats`: frozen dataclass, per bot spec:
    - Dash: rate and made;
    - With: rate and made;
    - `contract_share` and `contract_made` (keys "4" to "7" and "8+");
    - super calls: rate and made;
    - `made_by_role` (Caller, With, plain, last);
    - `over_tricks` and `under_tricks` shares;
    - the last estimator's `last_over`, `last_under` and `last_risk` shares;
    - `rounds`.
  - `TableStats`: per lineup, with `over`, `under`, `skipped`, `mean_total` and `rounds`.
  - `stats_from_games(records, rules, *, allow_mixed: bool = False, headers=...) -> tuple[dict[str, SeatStats], dict[tuple[str, ...], TableStats], ReplayStats]`. It raises ValueError on mixed code versions unless `allow_mixed`.
  - CLI: `--from PATH` (a run dir: every `games/*.jsonl.gz`; or one file) and `--bot SPEC` filter. The play mode (`--bot --games`) now records games in memory and goes through the same summarizer.
- [ ] **Step 1: Write the failing tests**
  - Hand-built records: one Over round, one Under round and one skipped round, with exact shares.
  - Mixed versions raise unless `allow_mixed`.
  - The play mode and the `--from` mode give equal stats for the same games (play 1 game, save, read).
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.** Roles come from `RoundResult.seats`. The last estimator's direction comes from `RoundResult.over` and its seat. "Took more tricks than estimated" is `tricks > estimate`.
- [ ] **Step 4: Run them and check they pass.**
- [ ] **Step 5: Commit** `feat(arena): compute bidding stats from saved games`.

### Task 6: Evaluator core: decisions, candidates, re-deal, play-out, value

**Files:**
- Create: `src/estimation/eval/__init__.py`, `src/estimation/eval/decisions.py`, `src/estimation/eval/playout.py`
- Modify: `src/estimation/bots/yosif/bot.py` (`YosifBot.predicted_make(obs, action) -> float | None`, as the spec defines it)
- Test: `tests/eval/__init__.py`, `tests/eval/test_decisions.py`, `tests/eval/test_playout.py`, `tests/bots/yosif/test_bot.py`

**Interfaces (Produces):**
- `Decision`: frozen dataclass.

  | Field | Meaning |
  | :--- | :--- |
  | `game` | `GameRecord` |
  | `step` | index into `actions` |
  | `seat` | int |
  | `kind` | one of `KINDS` |
  | `obs` | `Observation` |
  | `legal` | tuple[int, ...] |
  | `chosen` | int |
  | `round_no` | int |
  | `fixed_trump` | int or None |
  | `carry` | int |
  | `scores` | tuple of 4 |
  | `round_actions` | the round's actions before `step` |

- `KINDS = ("dash", "open", "bid", "final turn", "answer", "with or pass", "estimate", "last estimate", "super call")`.
- `decisions_of(record, rules, target: str) -> list[Decision]`: every non-card decision by a seat whose spec equals `target` (or `"human"`) with more than one legal action.
- `candidates(d: Decision, heuristic_choice: int) -> list[int]`: at most 8, legal, the chosen action first, built by the spec's rule.
- `redeal(d: Decision, rng: random.Random) -> tuple[int, int, int, int]`:
  - the decider keeps its hand;
  - the other 39 cards go to the other seats in 13s;
  - the deal is rejected and redrawn until every seat's void announcement matches `d.obs.void_announced`;
  - every round's first decision is before card play, so no card is played yet.
- `playout(d, hands, action, seats_specs, card_play: str, seed: int, rules) -> float`:
  - `Round.replay(d.round_no, hands, d.fixed_trump, d.carry, d.scores, rules, d.round_actions)`, then step `action`;
  - then each seat acts: its own spec's bot for non-card phases (a human seat uses `yosif`), the `card_play` bot for cards;
  - returns the decider's round score minus `standings_weights(d.scores, d.seat)` times the others.
- `value_table(d, candidates, deals: int, card_play: str, seed: int, rules) -> dict[int, tuple[float, float]]`: per candidate, the mean and standard error over the same deals and seeds.
- `YosifBot.predicted_make`:
  - estimates and Dash go through `Planner`;
  - a final turn's keep or raise uses the contract's `make_chance`;
  - With uses the top bid's number;
  - anything else returns None.

- [ ] **Step 1: Write the failing tests**
  - `test_decisions_skip_forced_moves`.
  - `test_kinds_cover_open_final_turn_and_estimate` (on one recorded heuristic game).
  - `test_redeal_keeps_the_deciders_hand_and_the_void_announcements`.
  - `test_fixed_trump_decisions_play_out_legally` (Review Focus 5).
  - `test_example_final_turn_keeping_7_loses_to_raising_to_8`: build the example-hand round with others passing and the bot opening 4♠. At the final turn, with 12 deals and `pimc:samples=2` card play, the mean value of `bid_action(8, SPADES)` beats `PASS` at 7♠. Mark it slow if it takes over 30 s.
  - `test_value_table_is_deterministic_for_a_seed`.
  - `test_yosif_predicts_its_estimate_and_returns_none_for_an_opening`.
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.** Make sure nothing in `estimation.bots` imports `estimation.eval`.
- [ ] **Step 4: Run them and check they pass**, plus `uv run pytest tests/eval tests/bots -q -n 2`.
- [ ] **Step 5: Commit** `feat(eval): price each bidding decision by playing every candidate out`.

### Task 7: Evaluator runs, reports and the CLI

**Files:**
- Create: `src/estimation/eval/bidding.py` (`__main__` entry: `python -m estimation.eval.bidding`), `src/estimation/eval/report.py`
- Test: `tests/eval/test_bidding.py`

**Interfaces:**
- Consumes: Tasks 1, 5 and 6.
- Produces:
  - `sample_decisions(decisions, n: int, seed: int) -> list[Decision]`: round-robin over kinds.
  - `evaluate(decisions, deals=12, card_play="pimc:samples=4", seed=0, workers=1, rules) -> list[Evaluated]`, where `Evaluated` holds the decision, candidate values, best, chosen and `lost = best mean - chosen mean`.
  - `calibration(records, rules, target) -> list[CalibrationBin]` over every call in the games that has a `predicted_make`. Each bin has `low`, `high`, `count`, `predicted` and `actual`.
  - `format_evaluation(...) -> str`:
    - a per-kind table: count, mean points lost, how often it chose best;
    - the 20 worst decisions as text, reusing the quiz layout (hand by suit, situation, scores, chosen against best, points lost);
    - calibration;
    - a warning line when the source games' code version differs from today's.
  - Writes `report.txt` and `evaluation.json` under `--out`.
  - CLI: `--bot SPEC`; one of `--from PATH`, `--lineup a,b,c,d --games N`, or `--web [PATH]`; then `--decisions 300 --deals 12 --card-play pimc:samples=4 --seed 0 --workers 1 --out DIR`.
- [ ] **Step 1: Write the failing tests**
  - `test_sampling_round_robins_over_kinds`.
  - `test_a_bot_that_always_picks_the_best_loses_nothing`: a fake value table.
  - `test_cli_runs_on_one_tiny_game`: `--lineup heuristic x4 --games 1 --decisions 3 --deals 2 --card-play heuristic --out tmp`, then report.txt has the per-kind table.
  - `test_web_mode_scores_the_human_seat` (Review Focus 2): a one-game `games.jsonl` with a human seat, `--bot human`.
- [ ] **Step 2: Run them and check they fail.**
- [ ] **Step 3: Implement.** Workers use `ProcessPoolExecutor` over decisions, like `run_arena`.
- [ ] **Step 4: Run them and check they pass.**
- [ ] **Step 5: Run the gate** with `uv run pre-commit run --all-files`; it must be clean.
- [ ] **Step 6: Commit** `feat(eval): report bidding points lost per decision`.

### Task 8: Final review

- [ ] Run the full gate, then a fresh code review of the branch.
- [ ] Fix Critical and Important findings with tests first.
- [ ] Ledger the deferred minors.
- [ ] Report to Yosif: what each command does, and the exact runs proposed with time and workers. Start no heavy run.
