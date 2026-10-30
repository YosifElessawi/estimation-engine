"""The public table: a friend types their name and plays against three copies of one bot, shown
as "B-Tig". Only the table: no lobby, ratings, watching, reviews or Why panel, and no step back
past the last trick. Every game is saved move by move (``records``) for the evaluators, and
``export`` turns the saved games into the web-games files they read.

Run it with ``uv run python -m server.public serve``; see ``__main__``.
"""
