# Hybrid iteration plan

Agreed with Yosif on 2026-10-30, updated with his tweaks on 2026-11-03. Each heavy run still
needs his yes (CLAUDE.md).

## Direction

The Hybrid (M5) is the main project. It is built from the RL nets and the search, so improving
either part means improving the Hybrid, and every change is judged by one question: is the Hybrid
stronger? The RL bot (RL v2) and the Search bot are frozen as yardsticks: RL v2 for "did the
learning improve", the Search bot for "did we beat search without learning". Neither is tuned on
its own any more. M6 (coach mode, review labels) waits.

Starting point (ladder v2, rated before the 2026-10-30 rule changes): Hybrid 5,446, RL v2 5,123,
PIMC 4,909, Search bot 4,887. The Hybrid beat RL v2 168-32, the Search bot 196-4 and PIMC 194-6
(sets of 200), at about 16 ms per card decision.

Two rule changes landed after RL v2 was trained, so RL v2 and the Hybrid built on it have never
played them:

- **Super-call scoring (SCR-3, SCR-5, SCR-6):** a call of 8 or more scores only its square, with
  no +-10 Caller/With bonus, and its only-winner and only-loser bonuses have separate switches.
- **With from any call (AUC-4, EST-2):** any call in the winning trump makes a player With, from
  their last call in that trump; a With player not at the Caller's number (and who did not stay
  at it on a raise) chooses an estimate first, from that number up to the Caller's.

## How every step is run and judged

- Everything runs on the Mac, one heavy job at a time, each after Yosif's yes (what runs, how
  long, workers, RAM). Heavy runs slow the public B-Tig table, so each ask says so. Step 5's copy
  and self-play run on the home server's GPU instead, under the same rule.
- Each step is built with tests and the full gate (pre-commit, pytest on 2 workers, pyright),
  then measured against the previous Hybrid and the frozen yardsticks (RL v2, Search bot) with
  head-to-head duplicate matches, the ladder and the bidding and card-play evaluators.
- A change is kept only if it measures better. If it does not, the step stops with a report and
  a proposal of what to try instead.
- Bots read only their own Observation. The hidden-card guesses below are trained on the real
  hands, which training games know, but a playing bot only ever sees its own guesses.
- New checkpoints get clear names in the roster (RL v3 for step 1's nets, Hybrid v2, v3 and so on
  for each kept Hybrid). The Hybrid on the live and B-Tig tables switches to a new version only
  after Yosif approves it.
- Each step ends with a short result in this file and its own PR.

## Steps

| Step | What it does | Compute | How we know it worked |
|---|---|---|---|
| 1. Hidden-card guesses, current rules | Retrain the nets under the current rules, and teach them to guess who holds each unseen card. | About 6.5 h: a fresh PIMC recording (about 30 min), then about 1,500 self-play loops from RL v2 (about 6 h). | Guess accuracy climbs; RL v3 vs RL v2 and the Hybrid on RL v3 vs today's Hybrid, 200 sets each. |
| 2. Copy a stronger search (expert iteration) | The card net learns the values of a deeper Hybrid search, then BiddingNet relearns to bid for the new card play. Repeat. | Per round about 4-5 h: recording 2-3 h, copying about 1 h, bidding refit about 1 h. Plan 3 rounds. | New Hybrid vs previous Hybrid each round, 200 sets; card-play evaluator. |
| 3. Hybrid uses the guesses | Imagine deals where step 1's guesses say the cards are, plus a settings check. | No training; about 2-3 h of measuring. | Card-play evaluator and new vs previous Hybrid. |
| 4. Search the bidding | For each bid, try the net's best few options on imagined deals played out by the nets. | No training; about 2 h of measuring. | Bidding evaluator and new vs previous Hybrid. |
| 5. One net for every decision | One player net copies Hybrid v4, then learns final place by GPU self-play (Essam's design). | Recording on the Mac (about 5 h per 2,000 Bolas); copy and self-play on the home server's GPU, sized by a timing test. | The player inside the search vs Hybrid v4: weighted place, King only and points per game. |

Steps 1-4 took about 25-30 hours of compute over several days.

### Step 1: hidden-card guesses, under the current rules

- **Current rules everywhere.** The engine, the rewards (round scores from the engine) and the
  `points_if_win` / `points_if_lose` inputs already follow the super-call scoring fix. The With
  change needs one new input: the nets today see each player's With call only for the top bid's
  trump, so a bidder cannot tell who would become With if the auction ended in another trump.
  Encoder v3 appends each player's last call in each of the five trumps this round (20 inputs).
  A With player who still has to choose is already visible (With, no estimate yet). Old
  checkpoints load with zero weights for the new inputs, as v1 did for v2.
- **The guess head.** Both nets get one more output: for each unseen card, the chance that each
  of the other three players holds it. Training games know the real hands, so the head learns
  from them alongside the current losses; the inputs stay the seat's own Observation. RL v2's
  training is kept: the run continues from its nets.
- **Fresh recording.** RL v2's recorded PIMC games predate both rule changes and the new inputs,
  so the card anchor is recorded again: 1,000 PIMC-30 games, about 30 min on 5 workers.
- **Self-play.** About 1,500 loops from RL v2's nets, at a gentler learning rate than RL v2's run
  (its restart at the full rate cost about 1,500 loops of dip), with the yardstick against PIMC-8
  and RL v2 every 50 loops.
- **Measured by:** guess accuracy over the run against a blind guess by hand sizes; RL v3 vs
  RL v2 (200 sets); the Hybrid on RL v3's nets vs today's Hybrid (200 sets).

### Step 2: copy a stronger search (expert iteration)

- **(a) A stronger teacher, and every card's value.** Games are played by the current Hybrid.
  At the decisions the teacher labels, a deeper Hybrid search than the playing one runs: more
  deals (64 against 16) and every legal card (not only the net's top 3). The card net then
  learns each searched card's value, not only which card the search chose, as the warm start
  learned PIMC's values (Essam's net copying PIMC-100's card values had the best card in its top
  3 almost always). The guess head learns the real hidden hands of the same positions. The share
  of decisions the teacher labels is set by a timing test, to keep a round's recording at 2-3 h.
- **(b) Bidding refit after every round.** A short self-play phase (about 1 h) in which only
  BiddingNet learns, playing with the new card net, so bidding fits the card play it bids for.
  Essam's swap test showed bidding and card play work only when trained to fit each other.
- **Measured by:** the copy's held-out regret under the teacher's values (top 1, top 2, top 3),
  then the new Hybrid vs the previous one, 200 sets.

### Step 3: the Hybrid uses the guesses

- Imagined deals place each hidden card by step 1's guesses (still fitting hand sizes, voids and
  the void announcements), instead of dealing at random and keeping the deals that best fit the
  bidding. Both orders are measured, and the two can be combined.
- **Settings check:** top 2 vs top 3 cards searched, and 16 vs 32 deals. A sharper net may need
  fewer cards searched (Essam's top 2 beat top 3 once his net was trained by self-play).
- **Measured by:** card-play evaluator (points lost per card) and new vs previous Hybrid.

### Step 4: search the bidding

For each bid, try the net's best few options: imagine deals, finish the auction and estimates
with BiddingNet, play the round out with CardPlayNet, and pick the best-scoring option. The
play-outs already use the bot's own card net, so the searched bids fit its card play. Bids get
slower, maybe 0.1-0.3 s each, still fine for the table. Measured by the bidding evaluator
(points lost per decision) and new vs previous Hybrid.

Why this order: steps 2 and 3 both use the step-1 head. Doing 2 before 4 improves card play
first; once 4 exists, one more round of step 2 also teaches BiddingNet to bid like the searched
bidding. Step 4's code can be written while step 1 trains; only its measuring waits for the Mac.

### Step 5: one net for every decision (Yosif, 2026-11-06)

Replaces the "Play for King" step and the earlier one-net candidate (both agreed on 2026-11-03 and
2026-11-05). Essam's design plus our head start: Essam's single net (estimation-arena), trained
from zero by GPU self-play, tied the original Hybrid once our Hybrid search ran on top of it
(`bitter-search` vs Hybrid v1, 150 sets: -3.3 points a game, -9.8 to +3.2, first place 25.8%).
This step starts the same kind of net from Hybrid v4 instead of from zero.

**The nets**

- **Player:** one net for every decision: Dash Calls, bids, With, estimates, super calls and
  cards. Its input is the raw table view only: own hand, cards played, calls, estimates, tricks
  won, scores and the legal moves. No hand-made hints (no HeuristicBot trick counts, no "wins the
  trick now", no "top card left"). Only the player is used at the table.
- **Judge, training only:** the same view plus the three hidden hands, so the learning signal
  separates deal luck from skill. The player never sees hidden cards; the judge is thrown away
  after training.
- **Reward:** final place in the Bola: King +1, Sub king +0.2, Sub kooz -0.2, Kooz -1, ties
  shared (Essam's per-round reward stalled). Playing for final place is built into this net and
  its search, so the separate Play for King step is dropped.
- **Practice questions:** extra outputs of the player, learned from the real game at light
  weight next to the main goal, all seven from the start:
  1. who holds each hidden card;
  2. my tricks this round;
  3. will each player make their call;
  4. each other player's tricks;
  5. who is out of which suit;
  6. my chance of finishing King from here;
  7. each player's round score.

  Each gets its own held-out score against a blind guess (today GuessNet names 47% of holders
  against 35% blind). One comparison run trains without the questions, and a question whose
  score never moves is dropped. Questions 4 and 5 overlap 1 and 3, so they go first if the net
  runs short of room.
- **At the table:** the player inside the Hybrid search. Imagined deals are placed by its own
  hidden-card guesses (question 1, kept by the bidding as `deals=both` is today), played out by
  the player for every seat, and each card is scored by its chance of finishing well (question 6
  or a judge-style value) instead of round points.

**The steps**

| # | What | Where | Time |
|---|---|---|---|
| 1 | **Record the teacher:** Hybrid v4 plays a few thousand Bolas, saving every decision, the deeper search's value of every card, BiddingNet's value of every bid and the real hands (like step 2) | Mac, 5 workers | about 5 h per 2,000 Bolas at step 2's pace; a short timing test first |
| 2 | **Copy:** the player learns to play like Hybrid v4 from those recordings, so it starts near our best level; the judge and the practice questions learn from the same games | home server GPU | sized by the timing test |
| 3 | **Self-play** on Essam's JAX rules, against itself and past versions; every update learns from final place, the judge and the practice questions | home server GPU | sized by the timing test |
| 4 | **Check-ups:** the player inside the search plays Hybrid v4 at regular points, so it does not drift into habits that only work against itself | Mac | a short match each |
| 5 | **Final judgement:** the player inside the search vs Hybrid v4 on all three agreed measures: weighted place (King +1 to Kooz -1), King only (first-place share) and score difference (points per game). Keep whichever wins | Mac | 200 sets |

**Machines.** The Mac records, searches and plays every match (its CPU is the faster one). The
home server (GTX 1060 6 GB, a weak 4-core CPU) runs the copy and the self-play. Runs move between
them with rsync over Tailscale. Each long run on either machine still waits for Yosif's yes.

**Timing test** (2026-11-06, the 1060, Essam's loop unchanged, batches of 4,096 Bolas x 128
decisions, peak GPU memory about 2 GB of 6):

| Player net | Self-play | 1,000 loops | Copy, one pass over 1 M decisions |
|---|---|---|---|
| 384 x 3 layers (Essam's size, about as fast as today's card net in the search) | 116,000 decisions a second | about 75 min | 3 s |
| 512 x 4 layers (twice the room) | 70,000 decisions a second | about 2.1 h | 5 s |

Essam's whole from-zero run was 5,000 loops: about 6.5 h here at his size. The copy is minutes,
not hours, so both sizes are copied and the bigger one is kept only if it copies Hybrid v4
clearly better. The practice questions and the regular check-ups add a little to each loop.

**Credit.** The one-net design, the final-place reward, the judge that sees the hidden hands and
the JAX rules (`jaxenv`, checked move for move against our engine over about 7.6 M steps) are
Essam Gouda's, from estimation-arena (MIT licence).

**How it is built**

- **Raw view:** Essam's raw view as is (747 inputs). It exists on the GPU (`jaxenv.encode`) and on
  the CPU from an engine Observation (`jaxenv.encode_np`), and his check requires the two to be
  equal at every step, so what the player learns on the GPU is what it sees on the Mac.
- **Recorder (Mac):** step 2's recorder with Hybrid v4 playing. The teacher is Hybrid v4's search
  made deeper, as in step 2 (64 deals instead of 32, every legal card instead of the top 2). It
  saves every seat's decisions, BiddingNet's values at bids, the real hands and each Bola's final
  places, which also label the practice questions.
- **Copy (GPU, JAX):** the player's move output learns the teacher's values turned into chances;
  measured as in step 2 by held-out regret (points the top 1, 2 and 3 moves lose by the teacher's
  values). The judge learns each Bola's final place from the view and the real hands; each
  practice question learns from the recorded games, with its held-out score.
- **Self-play (GPU, JAX):** Essam's loop (PPO, his judge, credit by TD(lambda), half the games
  against past versions) started from the copy, with the seven practice outputs labelled inside
  the games. Three guards keep the head start (Yosif, 2026-11-06):
  - gentle exploration and a fading pull back toward the copy, like the card anchor: step 1
    showed that a trained net pushed to explore at the from-scratch rate loses most of its
    strength within 50 loops;
  - the frozen copy stays among the opponents next to the past versions, so the net keeps
    having to beat a Hybrid v4-style player, not only itself;
  - the kept net is the one with the best check-up against Hybrid v4, not the last loop.

  First run: 5,000 loops (Yosif, 2026-11-07; about 7.5 h on the 1060), a check against the copy
  every 100 loops and a check-up against Hybrid v4 every 500, continued while the check-ups
  improve.
- **At the table (Mac):** the player in numpy (no JAX needed) inside the Hybrid search. Essam's
  `bitter-search` already runs his net inside our search; ours adds guessed deals and King scoring.
- **Arena:** our duplicate arena reports weighted place too, so one match shows all three
  measures.

**Where the code lives:** in this repo (Yosif, 2026-11-07). Essam's `jaxenv` is copied into
`src/estimation/jaxenv` with credit and its MIT licence, kept as it is in estimation-arena so later
fixes there can be diffed in, and its move-by-move check against our engine runs in our tests.
Step 5's own code sits next to it and only imports it.

## Parked net changes

- **Suit symmetry built into the net:** an efficiency gain (the bot already learns symmetry from
  shuffled copies). Worth more once step 2 makes games expensive; only if learning stalls.
- **Bigger nets:** only if step 2 shows the net cannot copy the Hybrid well (a sign of too little
  capacity). Costs speed in every Hybrid play-out.
- **Transformer over cards:** last, if ever. The Hybrid calls the card net thousands of times per
  decision, so a net several times slower makes the Hybrid several times slower.

## Decisions

1. Step 2's recordings run on the Mac (Yosif, 2026-11-03).
2. One session runs the whole plan, stopping for Yosif's yes before each heavy run (Yosif,
   2026-11-03).
3. Step 5 is Essam's one-net design started from Hybrid v4, and it replaces the Play for King
   step: final place is its reward (Yosif, 2026-11-06).
4. Step 5's JAX code lives in this repo: Essam's `jaxenv` copied in with credit and its MIT
   licence (Yosif, 2026-11-07).

## Results

Each step adds its result here.

### Step 1 (done 2026-11-04): RL v3 and Hybrid v2 kept

**The run** (`configs/rl-v3.toml`, git-ignored `runs/rl-v3/`): a fresh 1,000-game PIMC-30 recording
under the current rules (20 min), then 1,500 self-play loops from RL v2 (6.9 h, about 17 s a
loop on 5 workers). Two restarts on the way, both fixed in this PR:

- A guess output on the play nets never learned (its loss stayed at ln 3 for 30 loops). A probe
  showed RL v2's trunk keeps almost nothing about who holds which card (a head on it scored 34%,
  a blind guess 35%), so the guesses moved to GuessNet, a separate net.
- Restarting bidding exploration at the from-scratch settings (temperature 0.2, 5% random) sank
  the nets from RL v2's 92% of sets against PIMC-8 to 45% within 50 loops. Continuing at RL v2's
  last settings (0.05, 1%) and 3e-5 falling to 1e-5 kept them at 90-98% throughout.

**Guesses:** GuessNet's top guess names the holder of 47% of unseen cards (a blind guess among
the players not shown out: 35%), climbing fast in the first 100 loops.

**Matches** (200 duplicate sets each, seed 2026, current rules, 5 workers):

| Match | Sets | First place (fair 25%) | Points per game (95% CI) |
|---|---|---|---|
| RL v3 vs RL v2 | 129-71 | 29.0% vs 21.0% | +18.2 (+11.5 to +25.0) |
| Hybrid v2 vs Hybrid | 110-90 | 26.6% vs 23.4% | +5.4 (+0.6 to +10.3) |
| Hybrid v2 vs Search bot | 194-6 | 41.2% vs 8.8% | +69.5 (+64.4 to +74.7) |

**Evaluators** (on the Hybrid v2 vs Hybrid games, 300 decisions or cards each): points lost per
bidding decision 4.3 (Hybrid v2) vs 4.5 (Hybrid); per card 2.38 +- 0.40 vs 2.21 +- 0.27, best
card 53% vs 48%. Both within noise: a 5-point edge per game spreads over hundreds of decisions.

**Reading:** better nets helped the bare RL bot a lot and the Hybrid a little, since the search
already covers some of the nets' mistakes. Hybrid v2 moves as fast as the Hybrid (about 14 ms a
card). Both changes are kept; the tables switch to Hybrid v2 only on Yosif's approval.

### Step 2 (done 2026-11-04): Hybrid v3 kept after one round; round 2 added nothing

**The rounds** (`configs/expert.toml`, git-ignored `runs/expert/`, about 7.5 h a round on 5
workers): Hybrid v2 played 2,000 Bolas while a deeper search (64 deals, every legal card) valued
every card at half its card decisions, about 685,000 labelled decisions a round (recording took
about 5 h, not the 2.3 h a one-game timing suggested). The card net copied those values (30 s on
the GPU), GuessNet the recorded hands, and BiddingNet was refit by 250 self-play loops (2.2 h, 32 s
a loop) against the new opponent mix: 80% self-play, 6% tables of two light Hybrids, 14% mixed
tables (recent versions, Search bot, RL v2, a light Hybrid).

| Round | Copy: points the net's top card loses (top 3) | Guesses right | Match vs previous Hybrid (200 sets) |
|---|---|---|---|
| 1 | 1.98 to 1.70 (0.43 to 0.31) | 49% to 50% | won 144-56, first place 30.0% vs 20.0%, +19.0 (+14.9 to +23.1): kept |
| 2 | 1.63 to 1.57 (0.29 to 0.26) | 48% to 51% | 100-100, first place 25.2% vs 24.8%, -0.2 (-4.4 to +4.0): stopped |

**Hybrid v3** (round 1, `hybrid:checkpoint=runs/expert/round-1/hybrid.pt`) is kept: the biggest
gain so far, at the same speed (about 15 ms a card). Round 2's copy had little left to learn, and
its match was a tie, so step 2 stopped there, as the rule says.

**Card-play evaluator** (round 1's match games, 300 cards, PIMC-8 judge): Hybrid v3 loses 2.07 +-
0.31 points a card, Hybrid v2 1.68 +- 0.29. Within the noise, and not better: the +19 may come
mostly from the bidding refit against the stronger opponents, or from card play a PIMC-8 judge
cannot see. A swap match (v3's bidding with v2's card net, and the reverse) would tell which.

**Swap match** (2026-11-05, 200 sets each against Hybrid v2, seed 2030): v3's bidding with v2's
card net won 110-90, +7.1 (+2.3 to +11.8), first place 26.6% vs 23.4%; v2's bidding with v3's
card net won 135-65, +13.4 (+9.1 to +17.7), first place 28.8% vs 21.2%. Both parts earned their
place and add up to about Hybrid v3's +19: the copied card net about two thirds, the bidding refit
against stronger opponents about one third. The card-play evaluator missed the card net's gain:
its judge plays the rest of the round as PIMC-8, now weaker than the bots it judges, so its card
verdicts are not reliable at this level.

**On the tables** since 2026-11-05 (PR #6): Hybrid v3 is the Hybrid on the full table and in every
B-Tig seat; the M5 Hybrid stays in the roster as Hybrid v1.

### Step 3 (done 2026-11-06): Hybrid v4, the same nets with a better search

No training: Hybrid v3's nets with new search settings, each played against plain Hybrid v3 over
200 duplicate sets (seed 2031), then the best three together against the best single one (seed
2032).

| Setting (against Hybrid v3) | Sets | First place | Points per game | Time per card |
|---|---|---|---|---|
| Deals placed by GuessNet's guesses (`deals=guesses`) | 134-66 | 27.4% vs 22.6% | +11.2 (+7.2 to +15.1) | 12 ms |
| Guessed deals, kept by the bidding (`deals=both`) | 154-46 | 27.9% vs 22.1% | +19.6 (+15.5 to +23.6) | 14 ms |
| Top 2 cards searched instead of 3 (`top_k=2`) | 123-77 | 26.2% vs 23.8% | +8.6 (+4.6 to +12.6) | 12 ms |
| 32 deals instead of 16 (`samples=32`) | 140-60 | 29.2% vs 20.8% | +14.7 (+10.7 to +18.6) | 23 ms |
| All three of `deals=both`, `top_k=2`, `samples=32`, against `deals=both` alone | 143-57 | 28.3% vs 21.7% | +14.2 (+10.4 to +18.0) | 24 ms |

**Hybrid v4** (`hybrid:checkpoint=runs/expert/round-1/hybrid.pt:deals=both:top_k=2:samples=32`) is
kept: the guesses make the imagined deals more realistic, the sharper card net lets the search
skip its third card, and the saved time pays for twice the deals. About 24 ms a card (p99 about
90 ms), against 14 ms for Hybrid v3: still quick at the table. Together the step is worth about
+30 points a game over Hybrid v3 (the two matches chained, not one match).

### Step 4 (done 2026-11-06): searching the bids made the Hybrid worse, not kept

Hybrid v4 with bid search on (`bid_search=true`, BiddingNet's top 3 options, 16 imagined deals
each) against plain Hybrid v4, 200 duplicate sets (seed 2033), then both through the bidding
evaluator (300 decisions, seed 2033).

| | Sets | First place | Points per game | Bidding evaluator, points lost a decision |
|---|---|---|---|---|
| Hybrid v4 + bid search | 65-135 | 21.9% vs 28.1% | -12.8 (-17.1 to -8.5) | 6.2 |
| Hybrid v4 (bids by BiddingNet) | | | | 5.3 |

Both measures agree: the searched bids lose points. The search picks the option with the best
mean over a few noisy play-outs, so it tends to pick the option that got lucky; in its imagined
rounds every other seat bids greedily by BiddingNet, which is not how real opponents bid; and it
trusts its own play-outs on openings and super calls, where it did worst (open: best chosen 0% vs
12%, super call 15.8 vs 10.4 points lost). Hybrid v4 stays the best Hybrid. The bid search stays
in the code, off by default (`bid_search=false`), for a later try with more deals or a learned
value for the rest of the auction; no roster name.

### Step 5 (under way): one net for every decision

**Recording (done 2026-11-07)** (`configs/onenet-record.toml`, git-ignored `runs/onenet/record/`):
Hybrid v4 played 2,000 Bolas in every seat, 6.0 h on 5 workers (the 10-Bola timing test suggested
7.9 h). At half its card decisions the teacher (Hybrid v4's search with 64 deals and every legal
card) valued every legal card and played its best; at every bid, BiddingNet's values were kept.

| Saved | Count |
|---|---|
| Bolas | 2,000, in 80 parts of 25 (48 MB) |
| Moves | 2,497,727 |
| Bids with BiddingNet's values | 407,899 (24 options on average) |
| Cards with the teacher's values | 702,968 (5.2 legal cards on average) |

Each Bola is kept as its seed and its moves, so a replay gives the raw view, the real hands and
the final places for the copy and the practice questions. Checked after the run: every Bola
replays to its recorded final scores, every kept value is a real number, and on 100 Bolas every
kept decision has one value per legal move and the teacher played its best card. First place is
spread evenly over the four seats (479 to 509 Bolas each), as it should be with Hybrid v4 in all
of them.

**The JAX rules in this repo (2026-11-07):** Essam's `jaxenv` copied into `src/estimation/jaxenv`.
Its check against our engine passes on today's rules: every legal move, turn, round score and
final score identical over about 31,000 steps: 390 rounds of random, rule-based and scripted play
(60 of them also requiring the GPU view to equal the numpy view at every step) and 6 whole Bolas.
It takes 12 s on the Mac and is part of the test gate.

**Copy (done 2026-11-07)** (`configs/onenet-copy.toml`, git-ignored `runs/onenet/data/` and
`runs/onenet/copy/`): the recording became 2,072,395 decisions with a choice (2.9 min on the Mac);
1,900 Bolas trained the player and the judge for 12 passes (1.5 min for 384 x 3, 2.5 min for
512 x 4 on the 1060), and the other 100 Bolas scored every pass.

| Held-out Bolas | 384 x 3 | 512 x 4 | Hybrid v4's nets |
|---|---|---|---|
| Points the top card loses, by the teacher's values | 2.71 | 2.66 | 1.78 (card net) |
| Points the best of the top 3 cards loses | 0.50 | 0.50 | |
| Best card picked | 50% | 50% | 54% |
| Points the top bid loses, by BiddingNet's values | 0.10 | 0.08 | 0 (BiddingNet) |
| Same bid as BiddingNet's best | 88% | 89% | |

Practice questions (384 x 3), against a blind guess: holders named 44% against 35% (GuessNet 47%);
my tricks off by 0.61 against 1.73; the others' tricks off by 0.71 against 1.71; making the call
61% against 58%; out of a suit 90% against 85% (only what the table showed); King chance Brier
score 0.142 against 0.186; round score error 0.185 against 0.200. The judge scored 0.369 after the
first pass (blind 0.519), then worse every pass, 0.535 at the end: it learns the training Bolas'
final places by heart (training error 0.04).

**Reading:** the bids copy almost exactly; the cards do not. The player's top card loses about 0.9
points a card more than Hybrid v4's card net at either size, so 384 x 3 is kept (512 x 4 is not
clearly better). The judge has to be kept at its best pass, not the player's.

**Copy variants (2026-11-07, Yosif: improve the copy first)**, 384 x 3, 12 passes each (2 min each
on the 1060), scored at the last pass:

| Variant | Points the top card loses | Best of the top 3 | Points the top bid loses |
|---|---|---|---|
| First copy (the kept values only, temperature 2) | 2.71 | 0.50 | 0.10 |
| Also Hybrid v4's own card moves (the 961,528 decisions without values) | 2.48 | 0.36 | 0.12 |
| Sharper targets (temperature 0.5) | 2.88 | 0.48 | 0.09 |
| Both | 2.54 | 0.37 | 0.11 |
| Own moves, temperature 1 | 2.47 | 0.35 | 0.11 |

Learning Hybrid v4's own card moves gains about a quarter of a point a card; sharper targets lose
some. Temperatures 1 and 2 tie, so 2 stays. Every variant was still improving at pass 12. From
now on the copy learns the own moves, runs 20 passes, and keeps the pass whose top moves lose the
fewest points: keeping by the move loss picked an early pass, since the own moves pull that loss
up while the points lost keep falling.

**Final copy (2026-11-07)** (`runs/onenet/copy-final/384x3`, 2.5 min): the own moves on, 20 passes,
pass 18 kept. Its top card loses 2.41 points (the best of its top 3: 0.35) and is the best card
51% of the time, against Hybrid v4's card net's 1.78 and 54%; its top bid loses 0.11 points and
is BiddingNet's best 88% of the time. The judge is the first pass's (0.37 against 0.52 blind).
This player and judge are where self-play starts.

**Self-play, built (2026-11-07)** (`configs/onenet-selfplay.toml`, `estimation.onenet.selfplay`):
Essam's loop started from the final copy, with every decision's practice answers read from its
own game. The three guards as set: a learning rate of 1e-4 (a third of Essam's) and exploration
of 0.003 falling to 0.001 (Essam's from-scratch run: 0.02 to 0.01), with a pull toward the copy's
moves that fades out by loop 1,000; the frozen copy as the opponent in a quarter of the loops;
and a player kept every 100 loops for the check-ups. Every 100 loops the player meets the copy
(512 games at once, greedy), and the numbers go to a one-page chart (TensorBoard's CUSTOM SCALARS
page on the home server, `estimation.onenet.charts`).

The copy's move chances are soft (learned at a temperature of 2 points): moves sampled from them
lose about a point a card more than its best move, and bids ten times more, and in a short trial
callers made only 27% of their calls. Self-play therefore starts from them sharpened 4 times
(sampled moves then lose about 0.15 points more than the best; callers made 41%). On the 1060 a
loop takes 5.4 s, so the first run of 5,000 loops takes about 7.5 h.

See also `docs/rl/next-steps.md` (the RL v2 run's own notes).
