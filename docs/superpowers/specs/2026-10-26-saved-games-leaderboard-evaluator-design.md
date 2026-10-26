# Saved Games, a Clean Leaderboard and the Bidding Evaluator: Design

Status: draft for Yosif's approval, 2026-10-26. Branch `bidding-eval`, on top of `yosif-bot`.

Tags: `[Yosif]` is something Yosif asked for; `[Claude]` is Claude's choice or guess.

## 1. Why

- **Heavy runs throw their games away.** The arena and the ladder keep only final scores, so every new question means hours of new games. Yosif wants the games kept, so stats and evaluations come from runs already done `[Yosif]`.
- **The leaderboard is hard to read.** It has 26 rows with code names like `rl:checkpoint=runs/m4/...`. Most rows are measuring rungs, not bots anyone plays `[Yosif]`.
- **Bidding needs a per-decision score.** How often a bot makes its number is not the goal. A planned miss that sinks the leader is a good play. We need the points each bidding decision cost against the best choice `[Yosif]`.
- **Yosif's games are almost all Under.** The bots seem to share this: 14 of 18 rounds were Under in a 4-HeuristicBot test game. The next run should say how common it is, per bot.

## 2. Goals

1. **Every arena and ladder game is saved:** the deal seed, the bot in each seat, each bot's seed, every move, the final scores, the code version and the rules. Replaying a saved game reproduces its scores exactly.
2. **Stats come from saved games, with no new games.** Per bot, from any seat it sat in, and per table.
3. **A clean leaderboard.** Every bot has a plain name, a one-line description and a group: Leaderboard, RL checkpoints, or Measuring rungs. The ladder report prints three tables with each bot's Elo and a 95% range. The watch view and the web table use the same names.
4. **A bidding evaluator** for any bot. For each sampled Dash, bid, With, Pass, estimate or super call, it measures the points lost against the best choice. It reports them by decision type, writes out the worst decisions, and checks predicted against actual make chances for bots that predict them.
5. **One short command runs the full v2 ladder:** all 26 bots, every needed extra pair, games saved.

## 3. Non-goals

- Browsing ladder games in the web table. The saved format matches the web table's own, so this can come later.
- Retraining the RL bots.
- Changing any bot's play, apart from the yosif bot exposing its predicted make chance (4.4).
- Changing how Elo is fitted.

## 4. Anti-goals and constraints

- **Recording never changes a result.** Same seeds give the same scores and ratings, with or without saving.
- **Stats never mix versions.** One stats result never mixes games from different code versions or rules, unless explicitly told to.
- **No heavy run starts without Yosif's yes** (project CLAUDE.md).
- **Bots still see only their own Observation.** The evaluator and the arena are tools and may see every hand, as the arena already does. Nothing under `estimation.bots` may import them.
- **Machine limits:** 18 GB, arena runs on 5 workers, tests on 2, one heavy job at a time.
- **Storage:** a full ladder's games are about 1.4 KB per game compressed, so about 30-40 MB for the v2 ladder's 27,200 games. That is acceptable `[Claude]`.
- **Conventions (SPEC §8):** Python 3.12 with uv, the existing engine API, and the 120 action ids unchanged.

## 5. Design

### 5.1 Saved games

- **Arena:** gains `--save-games <file>`.
- **Ladder:** always saves, one file per pair: `<ladder dir>/games/<pair>.jsonl.gz`.
- **Reused pairs:** a pair is reused only when lineup, sets, seed and code version all match and its games file exists `[Claude]`. Reuse is for resuming an interrupted run; the v2 ladder starts in a fresh folder, so the first run replays everything.
- **Each file:**
  - a header line with the code version (git commit, marked dirty if the tree had changes), the rules, the lineup, sets, seed and date;
  - then one line per game: set index, rotation, deal seed, the 4 seat specs, the 4 bot seeds, the final scores, the round count and the action list.
- **Recording:** `play_bola` records the actions it steps.
- **Replay:** a reader replays any record through `Bola(seed)` and checks the replay reaches the saved scores. A record that no longer replays, for example after a rules change, is skipped and counted in the report `[Claude]`.
- **Versions:** web games carry no code version, so they count as "unknown" and are allowed with a warning.
- **Format:** the fields match the web table's saved games (seed, seats, scores, rounds, actions), so a later web feature can open ladder games.

### 5.2 Stats from saved games

`python -m estimation.arena.bidstats --from <run dir or file> [--bot SPEC]` replays saved games through the engine (no bots run). Today's play mode stays.

Per bot, over every seat it sat in:

- **Dash Calls:** rate and made.
- **With:** rate and made.
- **Contract sizes and super calls:** each with its made rate.
- **Made rate by role:** Caller, With, ordinary estimator, last estimator.
- **Taking more or fewer tricks than estimated.**
- **The last estimator:** how often it goes Over, how often it goes Under, and its Risks.

Per lineup (table):

- Over, Under and skipped shares.
- The mean estimate total.

### 5.3 Leaderboard and roster

- **Roster:** `estimation.arena.roster` maps every ladder spec to a plain name, a group and one line of description. Examples:

  | Spec | Name |
  | :--- | :--- |
  | `pimc` | PIMC |
  | `rl:checkpoint=runs/m4/checkpoints/final.pt` | RL M4 final |
  | `heuristic:noise=0.1` | HeuristicBot, 10% random moves |
  | `yosif` | Yosif bot |

  Unknown specs keep their spec as the name.
- **Groups `[Yosif]`, as shown to Yosif:**
  - Leaderboard: RandomBot, HeuristicBot, PIMC, RL M4 final, RL staged final, Yosif bot.
  - RL checkpoints: warm start, iterations 10, 20, 60 and 100, M4 final, staged final.
  - Measuring rungs: the noisy HeuristicBots, and PIMC with 1 (plus 30% random cards), 1, 2, 4, 8 and 16 deals.
  - A bot may sit in two tables; the two RL finals do.
- **Ladder preset:** `--preset v2` lists the full 26-bot ladder in calibration order, plus every extra pair: the current 8, plus `pimc,yosif` and `heuristic,yosif` `[Yosif]`. It defaults to `--dir runs/ladder-v2 --sets 200 --seed 0`. The web table's catalog reads `runs/ladder-v2/ladder.json` when it exists, and the old file otherwise.
- **Report:** three tables, Leaderboard, RL checkpoints and Measuring rungs, each sorted by Elo.
- **Elo range:** each Elo gets a 95% range by bootstrap: resample each pair's sets with replacement, refit, 200 times, take the 2.5% and 97.5% points `[Claude]`.
- **Sharing:** `ladder.json` carries names, groups and ranges. The web table's bot catalog and the watch view take names from the roster.

### 5.4 Bidding evaluator

`python -m estimation.eval.bidding --bot SPEC (--from <run dir> | --lineup a,b,c,d --games N | --web) --decisions 300 --workers 5 --out <dir>`

- **Web games:** `--bot human` scores Yosif's own decisions. In play-outs, any human seat's later decisions are made by the yosif bot, the closest stand-in for his style `[Claude]`. "The bot's choice" is always what the saved game recorded; when the code version differs from today's, the report says so.

- **Decisions:** the target bot's non-card decisions where it had more than one legal choice, sampled evenly across decision types with a fixed seed.
  - Types: Dash, opening bid, auction bid, final turn, answer to a raise, With or Pass, estimate, last estimate, super call.
- **Choices tried (at most 8)** `[Claude]`:
  - the bot's choice;
  - HeuristicBot's choice;
  - Pass and With when legal;
  - for a bid, the same trump one trick lower and one higher;
  - for an estimate, the numbers 1 and 2 either side;
  - for a Dash decision, both options.
- **Hidden cards:** dealt again 12 times `[Claude]`, consistent with what the deciding player saw: its own hand, the hand sizes and the void announcements (TBL-7). Bids do not weight the deals in version 1; that is a known simplification.
- **Play-out:**
  - The real game's actions up to the decision are replayed on each deal, then the choice is applied.
  - Each seat keeps its own bot for later bidding.
  - Every seat plays its cards with one card player, `pimc:samples=4` by default `[Claude]`. That is fast, and it breaks other players' contracts the way strong players do.
  - Every choice uses the same deals and seeds, so differences come from the choice, not luck.
- **Value of a choice:** the decider's round score minus the other three, with players ahead in the game weighted double. These are the same weights as PIMC's leader option, with the round's multiplier included. Effects on later rounds, such as a skipped round's x2 carry, are left out.
- **Points lost:** the best choice's mean value minus the chosen one's (0 when the bot chose best), with a standard error.
- **Report:**
  - per decision type: count, mean points lost, and how often the bot chose best;
  - the 20 worst decisions written out like quiz spots (hand, situation, scores, chosen versus best, points lost);
  - calibration, for bots that implement `predicted_make(obs, action) -> float | None`. Only the yosif bot will; this branch adds it.
    - Its meaning: the chance of taking exactly the number the action commits to. Keeping or raising on the final turn means the contract; With means the top bid's number; an estimate means that number; a Dash Call means 0 tricks.
    - Openings, other auction bids and Pass return None.
    - Calibration uses every such call in the source games, not just the sampled decisions. It groups the predictions in tenths and compares them with how often those calls were made.
- **Package:** a new package, `estimation.eval`, outside `estimation.bots`.
- **Cost:** about 300 decisions × 8 choices × 12 deals of play-outs, roughly 10-20 minutes on 5 workers.

## 6. Testing

- **Saved games:** a saved game replays to its saved scores, and the same seed gives the same ratings with and without saving.
- **Stats:** checked on hand-built records, with exact shares; mixed versions are refused.
- **Roster:** every v2 spec has a name and group; unknown specs fall back to the spec.
- **Elo range:** contains the fitted Elo, and is narrower with more sets.
- **Evaluator:**
  - On the example hand's final turn (7♠ against 8♠), keeping 7 loses points to raising to 8.
  - A bot that always plays the best candidate scores 0.
  - Results are the same for the same seed.
  - The worst-decision text is readable.
- **Gate:** pre-commit, pytest (2 workers) and pyright.

## 7. Left to the builder

- File and function layout.
- Compression details.
- How decision types are detected from the Observation.
- How the evaluator's work is split across workers.
