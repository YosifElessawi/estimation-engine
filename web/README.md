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

The seat menus list the Elo ladder bots and the M4 checkpoints, labelled with their ladder Elo.
They come from two git-ignored files under the project root: `runs/ladder-200/ladder.json` and
`runs/m4/checkpoints/index.json` (plus the `.pt` checkpoints next to it). Without them the menus
show the default ladder's bots, unrated. Set `ESTIMATION_ROOT` to read them from another
checkout.

## Modes

- **Watch:** four bots, every hand shown. Step one decision at a time (arrow keys), jump by
  trick (Shift + arrows), to a round's result, to the next deal or to the end, or press Space to
  let it play. RL bots get a **Why** panel: what their nets predicted for each legal option.
- **Play:** you take one seat against three bots. You only ever receive your own hand and your
  own legal moves; the seed (which fixes the deal) stays hidden until the game ends. Once it is
  over, **Review with every hand** replays it in watch mode.

## Checks

```bash
npm run typecheck
```

The server's tests (`tests/server/`) run with the rest of the Python suite.
