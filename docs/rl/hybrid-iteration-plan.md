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
  long, workers, RAM). Heavy runs slow the public B-Tig table, so each ask says so.
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

Total: about 25-30 hours of compute over several days.

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

### Play for King (after step 4, Yosif's choice 2026-11-05)

Today's reward plays each open round for points against the table (opponents ahead weigh double)
and blends in the chance of finishing first only in rounds 14-18, from a simple formula
(`win_blend`, a 20-point spread per round). Winning a Bola needs a risk attitude from the score
situation: gamble when far behind, play safe when far ahead, which costs some average score.

- **A learned King chance:** a net output that predicts the chance of finishing first (King) from
  here, trained on how real Bolas ended, replacing the formula.
- **The reward in every round** becomes the change in that King chance, instead of the
  fixed-trump-only blend.
- **The Hybrid's search** scores its imagined rounds by the King chance too, not only by points.
- **Judged by Bolas won:** Win Elo and first-place share, with points per game only as a side
  note (a bot playing for King may score less on average).

Nothing runs for it yet.

### Step 5 (candidate, after step 4): one net for the whole game

Added 2026-11-03 at Yosif's question (a friend's pointer to Sutton's "The Bitter Lesson"): three
nets (bidding, card play, guesses) are there for history and speed, not because they are known to
be better. One net with three outputs, trained on every part of the game together, may learn
shared understanding (where the cards are helps bidding and card play alike).

- **Copy:** the single net learns from the best Hybrid's games, as step 2 does (its bids, its
  search's value of every card, the real hidden hands), so it starts near the Hybrid's level
  instead of from zero.
- **Self-play:** then it improves as RL v2 did, against itself, recent versions, PIMC and the
  Search bot.
- **Judged as a Hybrid:** the Hybrid rebuilt on it plays the best three-net Hybrid; strength and
  time per move both count. Keep whichever wins.

Runs only on Yosif's yes; its compute is sized after step 4.

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

See also `docs/rl/next-steps.md` (the RL v2 run's own notes).
