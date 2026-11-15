"""Step 5 of docs/rl/hybrid-iteration-plan.md: one net for every decision, started from Hybrid v4.

Essam Gouda's design, on his JAX rules (estimation.jaxenv), which this package only imports:

- ``data`` (Mac) turns the recording of Hybrid v4 (estimation.rl.onenet_record) into the
  training set, laid out in ``shards``;
- ``nets`` holds the player (one net for every decision, with the practice questions'
  outputs) and the judge (training only);
- ``copy_hybrid`` (home server GPU) copies Hybrid v4 into the player and trains the judge.

Everything the GPU side imports needs only numpy, JAX and optax, never PyTorch.
"""
