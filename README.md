# Estimation Engine

A bot for **Estimation**, the 4-player trick-taking card game, played by the rules of the
Pocket Estimation app. It is built one milestone at a time: an exact rules engine, baseline bots
and a rating arena, a search bot, a self-play RL bot, a hybrid of the two, and a web table to
play and study games.

**[SPEC.md](SPEC.md) is the source of truth:** the rules (§1 is the whole game in one minute),
the engine contract, the milestones and their results, and the open rule questions.

## Where it stands

Milestones 1-5 are built, and the web table (M6) has its first slice. Players meet three bots:

| Bot | How it plays | Elo (ladder v2) |
|---|---|---|
| **Hybrid** (M5), shown to friends as "B-Tig" | RL v2's bidding; for each card it imagines 16 deals that fit the bidding and plays them out with RL v2's card net | about 5,445 |
| **RL bot** (M4, the RL v2 checkpoint) | Neural nets trained by self-play | about 5,125 |
| **Search bot** (M3) | Yosif's bidding rules; PIMC card search over 30 imagined deals | about 4,885 |

RandomBot is fixed at 400 Elo; HeuristicBot (about 3,600) and PIMC (about 4,910) stay as frozen
measuring sticks. The ratings predate the rule changes of 2026-10-30. The Hybrid is the main
project now; the next steps are in [docs/rl/hybrid-iteration-plan.md](docs/rl/hybrid-iteration-plan.md)
and [docs/rl/next-steps.md](docs/rl/next-steps.md).

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/); the web table also needs Node 24.

```bash
uv sync
```

The trained nets, ladder results and saved games live in `runs/`, which is not in git. Get the
`runs` zip from Yosif and unzip it in the project folder, so the nets sit at
`runs/m4-rules-v2/checkpoints/final.pt` and so on. Without it, the engine, the rule-based bots
and the tests work, but the RL bot and the Hybrid cannot load.

## Commands

| What | Command |
|---|---|
| Tests and checks (the gate) | `uv run pre-commit run --all-files` |
| Engine speed | `uv run python -m estimation.engine.bench` |
| Pit four bots against each other | `uv run python -m estimation.arena --lineup hybrid,search,hybrid,search --sets 10 --seed 0` |
| Elo ladder | `uv run python -m estimation.arena.ladder --preset v2 --workers 5` |
| Watch a ladder or arena run | `uv run python -m estimation.arena.watch <run folder>` |
| RL training | `uv run python -m estimation.rl.train run --config configs/m4-rules-v2.toml --out runs/<name>` |
| Watch a training run | `uv run python -m estimation.rl.watch runs/<name>` |
| Bidding evaluator | `uv run python -m estimation.eval.bidding --help` |
| Card-play evaluator | `uv run python -m estimation.eval.cards --help` |

Bots are named by registry spec, `name` or `name:key=value:...`: `random`, `heuristic`, `pimc`,
`search`, `rl:checkpoint=<path>`, `hybrid`. Ladder runs, training and the evaluators are heavy:
they use several CPU cores for minutes to hours (the dev Mac has 18 GB of RAM and runs one heavy
job at a time).

## Web table

- **Full table** (watch, play, review, leaderboard): see [web/README.md](web/README.md).
- **Public B-Tig table** (a friend types a name and plays three Hybrids; every game is recorded):
  see [docs/public-table.md](docs/public-table.md).

## Layout

```
src/estimation/
  engine/   rules engine: cards, auction, estimation, play, scoring, a full game (Bola)
  bots/     random, heuristic, pimc, search, rl, hybrid, and the registry that names them
  arena/    duplicate arena, Elo ladder, roster of bot names, live status
  rl/       state encoder, nets, self-play training, recording, yardstick
  eval/     bidding and card-play evaluators
server/     FastAPI backend for the web table (full and public editions)
web/        Next.js frontend
tests/      mirrors src/; every rule ID and worked example in SPEC has a test
configs/    RL training configs
docs/       plans and notes
```

## Working on it

Read [SPEC.md](SPEC.md) and [CLAUDE.md](CLAUDE.md) first. Rules change only when Yosif confirms
them from playing the app. Every random choice takes an explicit seed, so any game can be
replayed. Conventional commits (`feat(engine): ...`), one branch per change, no direct commits to
`main`.
