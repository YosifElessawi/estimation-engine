# RL v2: results and next steps

RL v2 is the self-play bot from the 12-hour run on 2026-10-27/28 under the current rules (the With
auction, void announcements, encoder v2). It is the strongest bot so far.

## The run

- **Config:** `configs/m4-rules-v2.toml`, seed 0. Output in the git-ignored `runs/m4-rules-v2/`.
- **Start:** the staged final's two nets (bidding and card play), so no imitation warm start.
- **Card anchor:** a fresh recording of 1,000 PIMC-30 games under the new rules (754,185 card
  decisions). It took 26 minutes on 5 workers, not the 16 expected, because the Mac was busy and
  on a low battery at first.
- **Self-play:** 3,000 loops of 240 training games plus 32 test games against 3 HeuristicBots:
  816,000 games in 12.3 hours, about 14 s per loop (20 games per second). Loops ran at 23 s until
  the Mac was plugged in.
- **Schedules:** both learning rates 1e-4 falling to 1e-5; card anchor 0.3 falling to 0.03; bid
  exploration and temperature as in the staged run; 10 loops of bidding-only training first.
- **Opponents:** 70% itself and recent snapshots, 15% HeuristicBot and noisy versions, 15% PIMC
  with 4-8 deals.
- **Yardstick:** every 50 loops, 40 duplicate sets against PIMC-8 and 40 against the staged final.

## Results

**Elo ladder** (v2 ladder, 200 sets per pair, only RL v2's pairs played):

| Bot | Elo | 95% range |
|---|---|---|
| RL v2 | 5,115 | 4,883 to 5,515 |
| PIMC | 4,910 | 4,688 to 5,329 |
| Yosif bot | 4,891 | 4,664 to 5,281 |
| RL staged final | 4,780 | 4,560 to 5,191 |

**Head to head** (200 sets each, RL v2 in two seats; a fair first-place share is 50%):

| Against | Sets won | First place | Points per game (95% CI) |
|---|---|---|---|
| Yosif bot | 81% | 65% | +32.9 (+27.3 to +38.4) |
| PIMC | 77% | 71% | +31.5 (+26.3 to +36.8) |
| RL staged final | 84% | 77% | +54.0 (+45.8 to +62.1) |

**Yardstick trend** (mean sets won over each block's 10 checks):

| Loops | vs PIMC-8 | vs staged final |
|---|---|---|
| 1-500 | 35% | 43% |
| 501-1,000 | 38% | 42% |
| 1,001-1,500 | 59% | 53% |
| 1,501-2,000 | 70% | 52% |
| 2,001-2,500 | 81% | 68% |
| 2,501-3,000 | 87% | 76% |

The staged final itself wins 72% of the PIMC-8 sets. In training games against HeuristicBots,
RL v2's first-place share rose from 17% to 63%, and skipped rounds in self-play fell from 3.6% to
0.4%. Its greedy test games against 3 HeuristicBots stayed at about 83% first places all run.

**Shape of the run:** the first ~1,500 loops were spent relearning the auction under the new rules
(the bot fell to about 30% against PIMC-8). It drew level around loop 2,000 and pulled clearly
ahead from about loop 2,300, once the learning rate had fallen to roughly half. It was still
improving at loop 3,000.

## Improvements and open questions

Costs are rough: a "run" means Mac time on 5 workers, "code" means a session of work.

### Training

1. **Continue RL v2 at a low learning rate.** It was still improving at the end, and most of the
   gain came in the low-rate tail. Continue from RL v2 for 2,000-3,000 more loops at 3e-5 falling
   to 1e-5, with the same anchor tail. Cost: 8-12 h run, config only.
2. **Gentler restart when continuing a net.** Restarting at 1e-4 cost about 1,500 loops of dip
   before any gain. A lower start (3e-5 to 5e-5) or a short warm-up should keep more of the
   starting net. Cost: config only; best judged from the first 500 loops of the next run.
3. **Card anchor.** 0.3 falling to 0.03 worked, but it was never compared with lower values or
   none. A trained net may no longer need PIMC's card play as a guide. Cost: two short runs of
   about 500 loops each, compared on the yardstick.
4. **Train against the Search bot, not plain PIMC.** The bot to beat now bids by Yosif's rules.
   Use it for the 15% search opponents and for the anchor recording, so RL v2 meets that bidding
   style in training. Cost: small code change (choose the opponent and recording bot by name),
   plus a check that the Search bot's bidding is fast enough for training games.
5. **Opponent mix.** RL v2 now beats HeuristicBots easily, so their 15% of games teaches little.
   Shift that share to the Search bot or to stronger past checkpoints. Cost: config, once item 4
   exists.
6. **Recording speed.** A fresh recording is needed only when the rules or encoder change; reuse
   `runs/m4-rules-v2/warmstart/pimc.npz` otherwise. Run long jobs plugged in: on battery the Mac
   ran loops about 60% slower. Cost: none.

### Measurement

7. **Yardstick noise.** 40 sets swing about ±15 points (85% one check, 62% the next). Use 100
   sets every 100 loops instead, and read the first-place share alongside sets won. Cost: about
   the same total time, config only.
8. **Points are useless in RL against RL matches.** When nets play only each other they sometimes
   all pass round after round, and the carry doubles each time (one test game reached x16,384). It
   happens with greedy bidding, rarely in training. Sets won and first places are unaffected, but
   any points average from such matches is noise. Open question: should training discourage these
   pass chains, or is passing at a high carry correct play? Cost: an investigation first.
9. **Test games against 3 HeuristicBots are saturated** at about 83% first places, so they no
   longer show progress. Replace them with greedy games against PIMC or the Search bot. Cost:
   small code change; slower per loop.
10. **No PIMC-30 or Search bot check during training.** Only the ladder measures those. A cheap
    check every 500 loops against the Search bot would show the trend that matters. Cost: small
    code change, a few minutes per check.

### The net

11. **Teach it to guess hidden cards** (first pick). Add an output that predicts who holds each
    unseen card, trained alongside the current losses. Strong card bots do this; it sharpens card
    play and is the piece search needs later. It keeps all of RL v2's training. Cost: code plus a
    continued run.
12. **Built-in suit symmetry.** Hearts and spades behave alike unless one is trump; the net learns
    that today from shuffled copies. A symmetric design learns faster from the same games. It
    needs a new net that first copies RL v2 (hours), then self-play. Cost: code plus a long run.
13. **A small transformer over cards.** It reasons about how cards relate (my Queen is top now
    that the King fell), but it is several times slower per decision, so far fewer games per hour.
    Cost: large code change and long runs.
14. **A bigger net.** It can be widened in place and keep its training. Only worth it once there
    is a sign the net is short of capacity. Cost: small code change plus a continued run.

### Beyond self-play

15. **Search on top of the net (M5).** At play time, imagine a few deals as PIMC does, but play
    them out with the nets and weight the deals by the hidden-card guesses (item 11). This is the
    biggest expected gain and the next milestone in SPEC.md.
