# Estimation web table

A local table for watching the bots play a whole Bola step by step, or playing one yourself
against three of them (SPEC §6 M6, first slice). The page is a Next.js app; the game runs in the
FastAPI server in `../server`, which `npm run dev` starts for you.

## Start it

Needs Node 24 (nvm) and `uv`. Once, in `web/`:

```bash
npm install
```

Then, in `web/`:

```bash
portless run
```

That prints the address, for example `https://estimation.localhost:1355` (inside a git worktree
portless puts the branch in front: `https://m6-web-table.estimation.localhost:1355`). If the
portless proxy is not running yet, start it once with `portless proxy start --port 1355 --https`.
`Ctrl+C` stops both the page and the game server.

Without portless, `npm run dev` serves it at `http://localhost:3000`.

## Bots and ratings

The seat menus offer the bots players meet: the Search bot, the RL bot (its best checkpoint) and
the Hybrid bot. The start page's leaderboard shows those on top, then, on request, the RL
checkpoints and the measuring sticks (RandomBot, HeuristicBot, PIMC and their weakened
versions). Bots are named and described by `estimation.arena.roster` and labelled with their
ladder Elo. The ratings come from the newest
ladder report under the project's `runs/` folder (`runs/ladder-v2/ladder.json`, else
`runs/ladder-200/ladder.json`) and the checkpoints from `runs/m4/checkpoints/index.json`. `runs/` is
git-ignored and lives in the main checkout, so a table started from a worktree reads the main
checkout's. Without those files the menus show the default ladder's bots, unrated. Set
`ESTIMATION_ROOT` to read another checkout.

## Modes

- **Play:** you take one seat against three bots. The Dash call, your bid and your estimate pop
  up just above your hand; bids show as bubbles at each player and cards land in front of whoever
  played them. You only ever receive your own hand and your own legal moves; the seed (which
  fixes the deal) stays hidden until the game ends. Once it is over, **Review with every hand**
  replays it in watch mode.
- **Watch:** four bots, every hand shown. Step one decision at a time (arrow keys), jump by
  trick (Shift + arrows), to a round's result, to the next deal or to the end, or press Space to
  let it play. RL bots get a **Why** panel: what their nets predicted for each legal option.

**Scores**, the round's **Log** and **Why** open as a drawer from the top bar. The table fits a
laptop or a phone held upright.

## Past games

Every game that reaches its end, played or watched, is saved to `runs/web/games.jsonl` under the
project root (git-ignored): its seats, seed, final scores and every move. A game still in progress
is never written, so a hidden seed stays hidden. The start page lists them under **Past games**:
copy a seed, **Same deals** fills in the same seats and seed for a new game, and **Review**
replays the game with every hand shown, even after the server restarts.

## Checks

```bash
npm run typecheck
```

The server's tests (`tests/server/`) run with the rest of the Python suite.
