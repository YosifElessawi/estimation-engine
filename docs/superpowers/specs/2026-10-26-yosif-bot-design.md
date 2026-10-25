# Yosif's Rule Bot: Design

Status: draft for Yosif's approval, 2026-10-26.

Every strategy below is tagged with where it came from:

- `[Yosif]`: Yosif said it, from how he plays the Pocket Estimation app.
- `[Claude]`: Claude's guess or method, to be checked by the quiz (Testing) or the arena.

## 1. Why

All of our bots except the RL ones bid with HeuristicBot's rules: HeuristicBot itself, PIMC and the RL warm start. Those rules misjudge strong hands. Measured on 2026-10-25:

- **Example hand:** ♠ A K Q J 9 7, ♥ Q 10 3, ♦ A Q 7 4, ♣ void. HeuristicBot scores it at 7.8 tricks and bids 7♠. Yosif says a strong player calls 8♠. A simulation agreed: on the same 500 random deals, played by PIMC in all four seats, 7♠ was made 40% of the time and averaged +4 points, 8♠ 48% and +17, 9♠ 31% and -3.
- **The cause:** HeuristicBot rounds the hand score but holds anything below 8.5 at 7, and 8.5 rounds to 9, so it never bids exactly 8. Hands it holds at 7 that are worth 7.5 to 8.5 are made 26% of the time (47% with PIMC card play). Most of the failures are taking 8: the hand is too strong to stop winning tricks.
- **The other bidders:** the RL BiddingNet never super-calls (0 in 9,000 rounds). Search bidding super-calls in 41% of rounds and makes 12% of them.

Yosif plays at a strong level and can state his strategy. This bot puts his bidding and estimating style into rules, plays cards with full PIMC, and is then measured against today's bots.

## 2. Goals

1. The bot makes Dash Calls, bids, says With, estimates and super-calls by the strategy in section 5.
2. **Example hand:** opening in round 1, with all scores at 0, it ends the auction at 8♠ (not 7, not 9).
3. **Arena:** against plain PIMC (same card play, today's bidding), its mean score advantage per set is above 0 with the 95% confidence interval above 0. Report the first-place share too, then rate it on the Elo ladder. "The bot" here means every feature on, risk mode (5.3) included. Risk mode is judged separately, on vs off, by first-place share, because it trades mean score for chances of winning `[Claude]`.
4. **Quiz:** Yosif agrees with its call on most quiz hands (Testing). The agreement target is set after the first quiz; Claude's starting guess is 70%.

## 3. Non-goals

- **No new card-play rules.** PIMC with 30 deals, the default and strongest version, plays the cards `[Yosif]`. The one change is the opt-in leader targeting in 5.9.
- **No bluffing.** Yosif rarely bluffs now, because strong players read it `[Yosif]`.
- **No learning.** This is a rule bot, not RL.
- **No web table work.** It can be listed in the web table once the branches are merged.

## 4. Anti-goals and constraints

- **HeuristicBot, PIMC and the RL bots keep their defaults.** HeuristicBot is the measuring stick the whole Elo ladder is calibrated on, so changing it would move every rating. All new behaviour lives in the new bot, or behind options that are off by default.
- **No randomness in its decisions** `[Yosif]` ("remove the random risk"). The only randomness is PIMC's seeded deal sampling.
- **It never sees another seat's cards.** It reads only its own `Observation`.
- **Yosif's rules are not tuned away silently.** If one of his rules loses in the arena, report it to him with the numbers; don't quietly change it to Claude's version.
- **No rule changes.** Strategy is bot design, not game rules, so SPEC §2-§3 stay as they are.
- **Conventions (SPEC §8):** Python 3.12 with uv, the existing engine API, and the 120 action ids unchanged. Every random choice takes a seed. The gate is pre-commit, pytest and pyright.
- **Machine:** the dev Mac has 18 GB. Arena runs use 5 workers, tests at most 2, one heavy job at a time.
- **Speed:** a Dash Call, bid or estimate should take under 10 ms on average and at most 50 ms `[Claude]`. For scale, a PIMC card decision averages 4.5 ms (SPEC M3).

## 5. The strategy

### 5.1 Counting the hand

The counting gives, for each possible trump, the chance of ending the round with exactly k tricks, for every k. A single number is not enough: the choice in 5.2 needs the whole spread.

**Trumps** `[Yosif]`. Count by how the missing trumps are split among the three other players. When you play your top trumps, every player who still has trumps has to follow, so each round removes up to four trumps. These are acceptance tests:

| Trumps held | Tricks | Why |
| :--- | :--- | :--- |
| A K + 3 small | 4 | The A and K draw 8 trumps. 5 are left and you hold 3, so you lose 1 and take the other 2. |
| A K Q + 3 small | 6, or 5 | Usually all 6. You lose 1 when someone holds the J or the 10 with 3 more trumps beside it. |
| A K Q J 9 7 | 6 | |

- **Voids and singletons** add no separate tricks. They are why low trumps win, so they are already in the trump count `[Yosif]`.
- **Other players' bids change the split.** A player who bid the same suit probably holds length or honours there `[Yosif]`. How much a bid shifts the split is Claude's guess. In version 1, calibrate those reads on bot games, since there is no data on human bidders; the quiz checks them against Yosif's reads `[Claude]`.
- **Method** `[Claude]`: the chance of each split of the missing trumps, counting tricks as the trumps are drawn from the top. The builder picks exact counting or sampling.

**Side suits.**

- An ace is a sure trick `[Yosif]`. *Claude's note:* that holds until the suit is trumped; the builder may lower it for long suits.
- A K or a Q is worth more the more cards it has beside it `[Yosif]`.
- Starting values, to be calibrated by the quiz `[Claude]`, as the chance of making a trick:

| Holding | Chance |
| :--- | :--- |
| K alone | 0.30 |
| K + 1 | 0.50 |
| K + 2 or more | 0.75 |
| Q alone, or Q + 1 | 0.20 |
| Q + 2 or more | 0.45 |

**Position** `[Yosif]`. A K or Q that plays after a player whose bid suggests he holds the next higher card is worth more. Example: the player before you bid 4♦ and you hold ♦ A Q, so he likely has the K and your Q can win. If that player plays after you, the card is worth less. The starting adjustment is ±0.25 `[Claude]`.

**Over and Under** `[Yosif]`. Over and Under are only known during estimation; a Caller bids without knowing (5.2).

- **Over:** everyone fights for tricks, so only strong cards count and weak honours lose value.
- **Under:** everyone ducks, so a bare K is a very likely trick.
- Strong players also calculate a loss on purpose (5.7).
- Starting values `[Claude]`: in an Over round, K and Q chances × 0.75. In an Under round, a bare K counts 0.75.

**Control** `[Yosif]`. Low cards, the 2s, 3s and 4s, let you get out of tricks, mainly in Under rounds and also in Over ones.

- For each high card, check whether it can be dodged: a low card in the same suit to play instead, or a short suit that lets you throw it away.
- The result is the floor: the number of tricks you will be forced to take whatever you do.
- A hand with no control, like the example hand, has a high floor. That is why bidding 7 on it fails.

### 5.2 Choosing a number

`[Yosif]` Pick the number with the best expected score. Taking too many tricks is as bad as taking too few, so he rarely bids under what the hand can take.

- **Method** `[Claude]`: from 5.1, take the chance of ending with exactly k tricks for each k, then score each k by SPEC §3. That includes the +10 for being Caller or With, the super call (k × k if made, minus half of that if missed), the Risk points and the only-winner and only-loser points. Pick the best.
- **The safety margin comes out of this by itself.** 8♠ on the example hand needs only one of the two queens, so it is made often.
- **Underbidding** is fine when your control lets you hit the lower number: the 2s, 3s and 4s cover your high cards `[Yosif]`.
- **A Caller bids before Over or Under is known.** Use the neutral values `[Claude]`.
- **The other players' estimates and results** drive the only-winner and only-loser points, the x2 when 3 or more players are Caller or With, and Over or Under. When choosing a bid, the builder decides how to predict them; sampling the other hands, as PIMC does, is the suggested start `[Claude]`.

### 5.3 Risk from the scoresheet

`[Yosif]` Normally, maximise the expected round score. When you need a bigger jump to win the game, prefer the higher-paying call that can still be made, for example 9 over 8. No randomness.

- **When it switches on** `[Claude]`: when the gap to the leader is bigger than the rounds left times X, where X is about what a good normal round scores. The builder sets X from arena data.
- **When leading:** Yosif hasn't said whether he plays safer. Use the normal rule until he says.

### 5.4 Auction

- Start low to gather information, then raise to your number on your final turn `[Yosif]`. Today's bot already does this.
- Read every bid, pass and Dash Call to predict the other hands, and feed that into 5.1 `[Yosif]`.
- With and passing: Yosif hasn't stated rules yet. Keep today's: say With when your hand fits the top bid's number in its trump `[Claude]`.

### 5.5 Dash Call

`[Yosif]` Dash Call whenever it scores better than the best bid or estimate the hand would otherwise make.

- **The hand doesn't have to be empty.** It's possible when your low cards cover your high cards. Example: A K 4 3 2 plus low cards elsewhere is a Dash candidate, planning to throw away the A and K when you can't follow suit.
- **Payoff:** a Dash Call is worth 25 in an Over round and 33 in an Under round. Earlier Dash Calls are visible.
- **Method** `[Claude]`: the chance of taking 0 tricks, from the control model in 5.1.
- **Opponents will try to break it.** Strong players drag others down when they can `[Yosif]`, and PIMC does it by design. So the 0-trick chance must assume opponents who try to break the Dash Call, not opponents who ignore it `[Claude]`. This is the mistake that sank search bidding: it Dash Called in about 40% of rounds, and against PIMC those calls failed 72% of the time (SPEC M3). The builder may tune this discount.
- **Whether A K 4 3 2 should actually Dash Call** depends on the rest of the hand, so it is a quiz hand, not a unit test.

### 5.6 Estimating when you're not the Caller

`[Yosif]` Read the earlier bids and estimates to predict the other hands, then estimate straight. Go under sometimes, when your 2s, 3s and 4s cover your high cards. Use the same counting and choice as 5.1 and 5.2, within the legal range (capped at the Caller's number, and the With range).

### 5.7 Last estimator

`[Yosif]` You decide whether the round ends Over or Under, so weigh both and take a Risk when it pays.

**When the sum rule bans your exact number**, for example your hand is exactly 1 and 1 is banned:

- Pick a number you will probably miss, then play so another player misses too. That avoids the -10 for being the only loser.
- The card-play part is PIMC's job. Its score already rewards making other players fail.
- The estimate choice counts the only-loser points in 5.2.

### 5.8 Fixed-trump rounds

`[Yosif]` Same rules.

- Estimate with the same counting.
- Super-call (FIX-4) when another trump's best number is worth more than the fixed trump's.
- *Claude's addition:* the highest estimate becomes Caller and ties become With (FIX-3), so the expected score must include those +10 and -10 points.

### 5.9 Card play

- **PIMC with 30 deals** `[Yosif]`.
- **Leader targeting** `[Yosif]`: when you'll fail anyway, drag down whoever you can, the leader first.
  - Add an opt-in PIMC option that weights other players by the standings: players ahead of you count more than players behind. Starting weights `[Claude]`: 1.0 for players ahead, 0.5 for players behind, normalised. This matches M4's reward.
  - Off by default; this bot turns it on.
- *Claude's note:* PIMC guesses the hidden hands by assuming each player's estimate fits HeuristicBot's hand scores. Keep that in version 1.

## 6. Testing

- **Unit tests** for every worked example above: the three trump counts, the example hand at 8♠, and the banned-number case for the last estimator. Plus the usual legality checks.
- **Arena** (duplicate deals, as in M2):
  - The new bot against plain PIMC, and against HeuristicBot.
  - Report each one's mean score advantage per set with its 95% confidence interval, and the first-place share.
  - Report the same bidding stats as the 2026-10-25 table: super call rate and make rate, Dash rate and make rate, the Caller's make rate by bid size. Those numbers came from scratch scripts outside the repo, so the builder adds this report to the arena tooling. The definitions:
    - Tables of 4 copies of the same bot, full games.
    - Open rounds (1-13) only.
    - Bid sizes and super calls are counted per played round, and Dash Calls per player per round.
    - "Made" means exactly the estimate.
- **Quiz** `[Yosif asked for it]`:
  - 20 to 30 real hands from bot games, picked where the new bot's call differs from today's or is borderline.
  - Each shown in chat as text: the hand, the seat, earlier bids and estimates, and the scores.
  - Yosif gives his call, and every disagreement becomes a rule fix or a question.
  - Run it once before tuning and once after.
- **Hand simulation:** the 2026-10-25 method, the same deals played three ways at different bids, for spot checks of single hands.

**Order and when to stop** `[Claude]`:

1. Build it, with unit tests, until the gate passes.
2. Run the arena.
3. Run the first quiz with Yosif. Building doesn't wait for his answers; tuning does.
4. Tune the `[Claude]` values only.
5. Run the arena and a second quiz.
6. Report.

If the bot still loses to plain PIMC while following every `[Yosif]` rule, stop and report the numbers to Yosif. Don't change his rules to win.

## 7. Where it lives

All `[Claude]`:

- A branch cut from `main`. Main has HeuristicBot and PIMC, and the bot needs nothing from the M4 or M6 branches.
- New modules in `src/estimation/bots/`, registry name `yosif`.
- PIMC gains the leader-weight option, off by default.
- SPEC milestones stay as they are. SPEC gets at most a pointer to this document, if Yosif wants one.

## 8. Left to the builder

- The exact methods: the trump-split count, the chance of each trick count, and the control floor.
- Code structure.
- Starting values marked `[Claude]`, then tuning by the quiz and the arena, within the rule in section 4 about not tuning Yosif's rules away.

## 9. Still open for Yosif (none blocks version 1)

- When leading the game, does he play safer?
- When does he say With, and when does he pass a good hand?
- As Caller, does he try to predict Over or Under before bidding?
