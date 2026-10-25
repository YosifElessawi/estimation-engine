"""The web table's backend (SPEC §6 M6): a FastAPI app wrapping ``Bola`` and the bots.

Run it with ``uv run python -m server`` (it listens on ``$PORT``, as portless sets it). The
Next.js app in ``web/`` starts it for you and proxies ``/api`` to it.
"""
