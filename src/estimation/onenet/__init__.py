"""Step 5 of docs/rl/hybrid-iteration-plan.md: one net for every decision, started from Hybrid v4.

Essam Gouda's design, on his JAX rules (estimation.jaxenv), which this package only imports:

- ``data`` (Mac) turns the recording of Hybrid v4 (estimation.rl.onenet_record) into the
  training set, laid out in ``shards``;
- ``nets`` holds the player (one net for every decision, with the practice questions'
  outputs) and the judge (training only);
- ``copy_hybrid`` (home server GPU) copies Hybrid v4 into the player and trains the judge.

Step 6, learning bids from a slow bid search:

- ``bid_teacher`` (GPU) searches one bidding decision slowly, many decisions at once;
- ``bid_lessons`` replays recorded games and keeps the teacher's verdicts at their bidding
  decisions, laid out in ``lessons``;
- ``bid_leads`` (GPU) reads from the lessons what a view says of each bid's lead, with the
  suit shuffles of ``suits``;
- ``bid_learn`` (GPU) teaches the player those read leads and holds the rest of it still;
- ``bid_check`` (Mac) plays other bids for real, on the recorded deals with the table player
  in every seat, to see what the teacher's leads are worth outside the rounds it imagines.

Step 7, the last tricks played exactly:

- ``endgame`` plays every line of a round's last tricks on imagined deals, each seat for itself;
- ``endgame_check`` (Mac) holds the cards the table player played there against that answer,
  and plays the cards it would change for real.

The training code the GPU side imports (``copy_hybrid``, ``selfplay``, ``bid_leads``,
``bid_learn``) needs only numpy, JAX and optax, never PyTorch; the recorders and the check
replay games with the engine and the bots.
"""
