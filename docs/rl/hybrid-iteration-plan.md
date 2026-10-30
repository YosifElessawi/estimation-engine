# Hybrid iteration plan

Agreed with Yosif on 2026-10-30. Nothing here has started; each heavy run still needs his yes
(CLAUDE.md).

## Direction

The Hybrid (M5) is the main project. It is built from the RL nets and the search, so improving
either part means improving the Hybrid, and every change is judged by one question: is the Hybrid
stronger? The RL bot (RL v2) and the Search bot are frozen as yardsticks: RL v2 for "did the
learning improve", the Search bot for "did we beat search without learning". Neither is tuned on
its own any more. M6 (coach mode, review labels) waits.

Starting point (ladder v2, rated before the 2026-10-30 super-call scoring fix): Hybrid 5,446,
RL v2 5,123, PIMC 4,909, Search bot 4,887. The Hybrid beat RL v2 168-32, the Search bot 196-4 and
PIMC 194-6 (sets of 200), at about 16 ms per card decision.

## Steps

| Step | What it does | Code | Compute | How we know it worked |
|---|---|---|---|---|
| 1. Hidden-card head + fixed scoring | The nets also learn to guess who holds each unseen card; training uses the corrected super-call scoring. | Small to medium: one new output on the nets, its training signal (real hands are known in self-play), old checkpoints still load. | About 5-6 h on the Mac: roughly 1,500 self-play loops continuing from RL v2. Reuses RL v2's recorded PIMC games. | Guess accuracy climbs; head-to-head vs RL v2 (~15 min). |
| 2. Copy the Hybrid (expert iteration) | Record Hybrid games, train the card net to pick what the Hybrid picked (plus the real hidden hands for the step-1 head), repeat with the improved Hybrid. | Medium to large: Hybrid game recorder, a copy-the-search training step, the loop, the cloud script with self-delete. | Per round: about 2,000 Hybrid games (about 2 h on the Mac, or about 1 h on a cloud machine for roughly $0.25-0.50), then about 1 h of learning on the Mac. Plan 3 rounds: about 9-12 h over a few days. | New Hybrid vs old Hybrid, about 45 min per check. |
| 3. Hybrid uses the guesses | When imagining deals, place hidden cards where the step-1 head thinks they are, instead of dealing at random and reweighting by the bidding. | Small to medium: only the Hybrid's deal sampling changes. | No training; about 1-2 h of measuring. | Card-play evaluator (points lost per card) and new vs old Hybrid. |
| 4. Search the bidding | For each bid, try the net's best few options: imagine deals, finish the auction and estimates with BiddingNet, play the round out with CardPlayNet, pick the best scoring option. | Large: the rest of the auction and the estimates must be simulated. | No training; about 2 h of measuring. Bids get slower, maybe 0.1-0.3 s each, still fine for the table. | Bidding evaluator (points lost per decision) and new vs old Hybrid. |

Total: about 20-25 hours of compute over several days.

Why this order: steps 2 and 3 both use the step-1 head. Doing 2 before 4 improves card play
first; once 4 exists, one more round of step 2 also teaches BiddingNet to bid like the searched
bidding. Step 4's code can be written while step 1 trains (code and tests are light); only its
measuring waits for the Mac.

## Parked net changes

- **Suit symmetry built into the net:** an efficiency gain (the bot already learns symmetry from
  shuffled copies). Worth more once step 2 makes games expensive; only if learning stalls.
- **Bigger nets:** only if step 2 shows the net cannot copy the Hybrid well (a sign of too little
  capacity). Costs speed in every Hybrid play-out.
- **Transformer over cards:** last, if ever. The Hybrid calls the card net thousands of times per
  decision, so a net several times slower makes the Hybrid several times slower.

## Open decisions

1. Where step 2's recordings run: the Mac (free, about 2 h per round, Mac busy) or the Hetzner
   cloud (about 1 h per round, a few dollars in total; Yosif first moves billing to a separate
   low-limit card, and servers self-delete after a time limit).
2. One session per step, each stopping for Yosif's yes before any heavy run.

See also `docs/rl/next-steps.md` (the RL v2 run's own notes).
