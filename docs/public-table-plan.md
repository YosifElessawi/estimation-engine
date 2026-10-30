# B-Tig public table: plan

Status (2026-10-30): approved and built on branch `claude/btig-public-table-a83358`, not yet on the internet. How to run, update and export: [public-table.md](public-table.md). The rule change in section 7 is its own session and PR.

## 1. What friends get

- They open one link, type their name and play. They sit in one seat and three copies of the Hybrid sit in the others, each shown as **B-Tig**. The rules and the table are the same as today. When a game ends they can start another, as many as they like.
- Left out: the lobby, other bots, the leaderboard, Elo, watch mode, review mode, past games, the Why panel, seeds and every internal name.
- Rewind: friends see the current trick and can peek at the last finished trick, nothing earlier. The server enforces this by never sending earlier tricks again, so it is not just hidden in the page.
- Assumption: all three bots read "B-Tig", told apart by seat position and avatar colour as today.

## 2. Hosting

Measured on the Mac (one process, three Hybrids, 3 rounds): 16 ms per card decision on average (95% under 50 ms), under 1 ms per bid, 283 MB of memory in total, because all games share one copy of the nets. A whole game is about 700 bot card decisions: about 11 seconds of computing spread over a 20-30 minute game. A cloud CPU is 2-3 times slower, which still means well under a tenth of a second per bot move.

| Option | Cost | Speed and limits | What Yosif does |
| :--- | :--- | :--- | :--- |
| **A. Hetzner cloud server (pick)** | CX23 (2 vCPU, 4 GB): EUR 6.49 a month plus a small IPv4 fee, about EUR 7 all in. Billed by the hour up to that monthly cap, 20 TB of traffic included, deleting the server stops the bill. | Always on. About 20 games at once with no noticeable delay. | Put the separate low-limit card on the Hetzner account (the `estimation` project and `hcloud` context already exist). Turn on Funnel in the Tailscale admin page (one switch). |
| **B. Your Mac, through Tailscale Funnel** | Free | Down whenever the Mac sleeps, the lid is closed or it is offline. Slower during training or rating runs, and it shares the 18 GB. Public traffic reaches your own laptop. | Turn on Funnel. Keep the Mac awake when friends play. |
| C. Usage-billed hosts (Fly.io, Railway, Render) | Varies | Not recommended: billed by usage, which is the surprise bill you want to avoid, and their free tiers sleep and lack the memory PyTorch needs. | - |

**Why A:** friends can play at any hour, the table can never touch your Mac or its heavy jobs, and the bill is fixed and on a separate card.

**Chosen (Yosif, 2026-10-30): B, the Mac, for now.** The public table runs from its own worktree that follows main (like the live table, under a different portless name, not `estimation`), listening only on the Mac itself. Funnel forwards that one port to the internet and nothing else on the Mac is reachable. The link goes down while the Mac sleeps. Moving to A later only means copying the database file and changing the link.

**The link:** Tailscale Funnel gives a free, stable https link such as `https://btig.<your-tailnet>.ts.net`, with no domain to buy and no open ports on the server. A custom domain (about EUR 10 a year) can come later if you want a nicer link.

**PyTorch on the server:** the server installs the CPU-only PyTorch build (about 200 MB instead of several GB of GPU libraries). Your Mac setup does not change. The Hybrid's checkpoint (git-ignored) is copied to the server once and on every deploy that changes it.

## 3. Public safety

- **Only the friend's own seat:** as today, every frame is built from the friend's own Observation. The seed is never sent and never accepted, so nobody can pick a deal they already know.
- **Only public routes:** the public server has exactly three things: start a game (with a name), read your game's frames, make a move. No bots list, history, review, watch, Why or debug routes, and the automatic API docs pages are off.
- **Games cannot be hijacked:** each game gets a secret token that only the friend's browser holds. Moves without it are refused.
- **Names:** trimmed, 1-20 characters, letters in any language (Arabic included), digits, spaces and `. - ' _`. Shown as plain text only.
- **Limits:** 2 unfinished games per browser (a third leaves the oldest, which stays saved), 30 new games per browser per day, 60 new games an hour from everyone together, 20 games in play at once, and a request rate limit. Past that, a friendly "The table is full, try again in a few minutes". IP addresses are not used or stored.
- **Many at once:** one server process. Bot moves queue behind each other at 16-50 ms each, so 20 games at once still feel instant. The cap protects memory (about 10 MB per game in play). A game untouched for 10 minutes leaves memory to make room and comes back from its saved moves when its player returns.
- **Server hygiene:** the table listens on the Mac itself only; Funnel forwards that one port. Every page response tells browsers to check for updates, so a deploy reaches everyone at once.

## 4. Recording

- **Where:** one SQLite database file on the server (`/var/lib/btig/games.db`). On option B: `runs/public/games.db` in the main checkout.
- **What:** every game, written move by move: the friend's name, a random visitor id (so one friend's games stay together even if they type their name differently), seats, seed, every action, start and end time, finished or not, and final scores.
- **Nothing lost on a restart:** each move is saved the moment it is made. A game in progress resumes after a restart or an update: the friend reloads the page and carries on.
- **Download:** the database is already on your Mac. One command, `uv run python -m server.public export`, writes `runs/public/games.jsonl` (every finished game) and `runs/public/players/<name>.jsonl` (one file per friend), in the web-games format the evaluators already read: `python -m estimation.eval.bidding --bot human --web runs/public/players/<name>.jsonl`, and the same for `estimation.eval.cards`. Unfinished games go to `runs/public/unfinished.jsonl`; the evaluators replay finished games only.
- **Backups:** every pull is a full copy on your Mac. Hetzner's automatic backups add 20% (about EUR 1.30 a month) if you want them.

## 5. Staying in sync with main

**Pick: one codebase with a public edition, not a second app.**

- **Server:** a separate entry point, `python -m server.public`, that mounts only the public routes and reuses the same engine, bots and game code. The full app's routes do not exist on the public server at all.
- **Page:** the same Next.js app built with a "public" switch that renders only the name screen and the table. The public build is plain static files, so the server runs no Node. The table, cards and bidding box are shared, so a gameplay fix on main reaches friends on the next deploy.
- **A gate test guards the split:** the public page build must contain no leaderboard, Elo, Why, history or internal bot names, and the public server's route list must equal the allowed set.
- **Updating:** you merge to main as now, then run one command (`scripts/deploy-public`) or ask me. It builds from main, copies it to the server and restarts the table. Games in progress survive the restart. There is no auto-deploy, so a half-finished main never reaches friends.
- Not picked: a separate small frontend. More isolated, but every table fix would have to be made twice.

## 6. Bidding box and phones

These changes are shared, so your own table gets them too.

- **Standing bid inside the box,** at the top: the trump symbol large and in colour, the number, and who holds it, for example "Top bid 6 ♥, B-Tig on your right". Below it, what you hold, for example "You hold With 5 NT".
- **The confirm button names the exact bid** ("Bid 6 ♠") instead of "Pick tricks and trump". With and Pass say what they do: "With 6 ♥ (join B-Tig)", "Pass (you stay at 5 NT)".
- **During the auction** the table centre shows the standing bid larger, next to the holder's seat, and the trump stays visible in the top bar for the whole round.
- **Estimate box:** the hint also says why high numbers are out, for example "7 and up are above the Caller's 6".
- **Phones:** large touch targets, tricks in two rows of five, the decision box docked at the bottom above the hand with nothing hidden behind it. Every screen is checked at phone size (375 x 812) and on a small Android size: name screen, Dash Call, auction, estimate, card play, last-trick peek, round result and game end. Anything that breaks gets fixed.

## 7. Auction finding (item 5)

**The game:** live table game `ad15a4bc0a64` (you in seat 0, three Hybrids), still in progress and held only in the server's memory. It is saved to disk when it finishes, and lost if the table restarts first. Round 10, first seat 1, nobody Dash Called.

**What happened:** seats 1 and 2 passed. Seat 3 bid 5 NT. You said With 5 NT. Seat 3, on its final turn, raised to 6 ♥. You answered by outbidding with 6 NT. Seat 3 passed, and you kept 6 NT on your final turn. Result: you are Caller at 6 NT and **nobody is With**. Seat 3 estimated freely (3, as last estimator), seats 1 and 2 estimated 2 and 1, total 12, Under. You took 5 and scored -11. Replaying the same auction in the engine gives the same result.

**What SPEC says:** AUC-2 defines a player's held call as their latest call: "the last bid they made, or the bid they said With to". AUC-4 makes With only the players whose held call is in the winning trump, and says "A player whose held call is in another trump is not With". Seat 3's latest call was 6 ♥, so by SPEC it is not With. L11 covers a player outbid in another trump whose own trump wins, but no rule or example covers a player who bid the winning trump earlier and then switched to another trump.

**Verdict:** not a code bug. The engine does exactly what SPEC says. It is a rule question: SPEC's "latest call only" was never checked against the app for this case, and your play says earlier calls count.

**Question for Yosif:** in the app, if a player bids in one trump (5 NT), later bids another trump (6 ♥), and the auction ends in the first trump (NT, won by someone else), is that player With? If yes, at which number: exactly 5, their last NT bid, or do they estimate freely with 5 as a minimum?

**Proposed answer:** yes, With at exactly 5. As a rule: a player is With if any of their calls in this auction (a bid or a With) is in the winning trump, held at the last such call's number. Later calls in other trumps do not cancel it.

**Yosif's answer (2026-10-30), confirmed:** yes. When the auction ends, any player other than the Caller who made a call (a bid or a With) in the winning trump is With, even if their latest call was in another trump. Their minimum is their last call in that trump, and at their normal estimating turn they choose any number from that minimum up to the Caller's number. In round 10 the Hybrid picks 5 or 6. The existing held cases work the same way: in L1, seat 0 (held at 4 Hearts, Caller at 6) picks 4, 5 or 6 instead of being fixed at 4. A With player who already answered a raise with "stay" keeps that number.

**Next:** a separate small PR: AUC-2 and AUC-4 text, this round as a new worked example (L12), the RulesConfig switch, the "nobody left to estimate" check (AUC-6) and the auction view of who is With, all tested. One cost: the Hybrid and RL bots were trained and rated under the current rule, so their bidding has never seen this case. They stay legal, but may misjudge these spots until a retrain, which would be a separate heavy run that needs your yes.

**Your estimate screenshot** is round 12 of the same game and is correct: seat 1 Dash Called (0), seat 3 is Caller at 6 NT and seat 2 is With at 6, so the others total 12 and 1 would make 13 (EST-5). 7 and up are out because nobody may estimate above the Caller's 6 (EST-3). The new estimate hint in section 6 explains this on screen.

## 8. Order and size

1. Server public mode: public routes, B-Tig, name check, limits, last-trick-only frames, move-by-move recording, resume after restart, the pull command. With tests.
2. Bidding box and standing bid (shared with your table).
3. Public page: name screen, table only, last-trick peek, phone pass.
4. Deploy, once you pick hosting and the card is on the account.
5. End-to-end check (section 9).

About one working day in this session, landing as one PR to main. The rule change in section 7, if you confirm it, is a separate small PR and does not block the table.

## 9. How it gets verified

- **Desktop:** play a full 18-round game on the real public link (scripted clicks for speed, key moments by hand). Confirm only "B-Tig" shows, no Elo, no stepping back past the last trick, and every full-app route answers "not found".
- **Phone size:** play several rounds by hand at 375 x 812 in the built-in browser.
- **Restart:** restart the server in the middle of a game. The game resumes with no moves lost.
- **Recording:** run the pull command, check the files, and run the bidding evaluator on that game (one short run).
- **Limits:** a third game at once is refused, and bad names are refused.
- **Gate:** pre-commit, pytest on 2 workers and pyright clean.

## 10. Decisions for Yosif

1. **Hosting:** decided, B: free on the Mac for now (section 2). Recording then lives in `runs/public/games.db` in the main checkout, and the download step becomes a local export command.
2. **Auction rule:** answered and confirmed (section 7). Built in its own session and PR.
3. **Defaults:** OK with "B-Tig" on all three bots, the free `ts.net` link, and the limits in section 3 (2 games at once, 30 a day per visitor, 20 in total)?
