# The public B-Tig table

A friend opens the link, types a name and plays Estimation against three copies of the Hybrid bot, shown as "B-Tig". Nothing else of the full table is there: no lobby, ratings, watching, reviews, past games or Why panel, and no stepping back past the last trick. Every game is saved move by move with the friend's name, for the bidding and card-play evaluators. The plan and the reasons behind it are in [public-table-plan.md](public-table-plan.md).

## Run it

From a checkout of main (a worktree is fine; the games and the bot's checkpoint live in the main checkout's `runs/`):

```bash
cd web && npm ci && npm run build:public && cd ..
uv run python -m server.public serve --port 8100
```

`npm run build:public` builds the page into `web/out` and fails if anything of the full table slipped in. The server serves that page and its own API on `127.0.0.1:8100` only.

To share it, put it on the internet with Tailscale Funnel (turn Funnel on once in the Tailscale admin page):

```bash
tailscale funnel --bg 8100
```

The link is `https://<this Mac's Tailscale name>.ts.net`. `tailscale funnel reset` takes it off the internet again.

## Update it

Merge to main, then in the table's checkout: pull main, rebuild the page and restart the server. Games in progress are not lost: each move is saved as it is made, and a friend who reloads carries on where they were.

## Get the games

```bash
uv run python -m server.public export
```

It reads `runs/public/games.db` and writes, under `runs/public/`: `games.jsonl` (every finished game), `players/<name>.jsonl` (one file per friend) and `unfinished.jsonl` (games given up or still going; the evaluators read finished games only). Then, from the main checkout:

```bash
uv run python -m estimation.eval.bidding --bot human --web runs/public/players/<name>.jsonl
```

and the same with `estimation.eval.cards`.

## Limits

- 2 unfinished games per browser: a third leaves the oldest (it stays saved).
- 30 new games per browser per day, and 60 new games an hour from everyone together.
- 20 games in play at once. A game untouched for 10 minutes leaves memory to make room and comes back when its player returns.
- 60 requests per 10 seconds per browser or game.

Names are 1-20 letters (any script), digits, spaces and `. - ' _`. Only a hash of each game's token is stored, and no IP address.
