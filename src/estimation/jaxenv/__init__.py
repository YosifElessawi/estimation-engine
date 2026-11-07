"""Estimation's rules in JAX: whole batches of games stepped together on the GPU.

``rules`` is one round as pure array functions (``init_round``, ``legal_mask``, ``step``) and
``game`` chains rounds into a Bola. Both mirror Yosif Elessawi's engine rule for rule; the
tests in tests/jaxenv/test_check.py replay thousands of games in both and require identical legal
moves, turns and scores at every step. Only the engine's default rules are supported.

This package is Essam Gouda's work, copied from estimation-arena (MIT licence, see LICENSE in
this folder) at commit e43b15b for step 5 of docs/rl/hybrid-iteration-plan.md. The files are
kept as they are there, so later fixes can be brought over by diff. Changes in this copy:

- imports and commands point at ``estimation.jaxenv`` instead of ``jaxenv``;
- ``selfplay`` defines the final-place rewards itself (estimation-arena takes them from
  ``arena.positions``) and drops the RL v2 benchmark option, which needs ``jaxenv.rlv2``;
- not copied: ``rlv2``, ``check_rlv2``, ``stats``, ``bench``, ``lab_capture`` and
  ``table_labels``, which step 5 does not use.
"""
