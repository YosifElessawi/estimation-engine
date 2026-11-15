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
- ``bid_learn`` (GPU) teaches the player those verdicts and holds the rest of it still.

The training code the GPU side imports (``copy_hybrid``, ``selfplay``, ``bid_learn``) needs only
numpy, JAX and optax, never PyTorch; the recorders replay games with the engine and the bots.
"""
