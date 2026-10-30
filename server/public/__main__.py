"""Run the public table, or export its games.

- ``uv run python -m server.public serve [--host HOST] [--port PORT]``: the API and the page
  (the public build in ``web/out``, made by ``npm run build:public`` in web/). The port defaults
  to ``$PORT``, else 8100; it listens on this machine only unless ``--host`` says otherwise.
- ``uv run python -m server.public export``: write the saved games as web-games files under
  ``runs/public/`` (see ``export``).

Games are saved in ``runs/public/games.db`` under the checkout whose ``runs/`` holds the
checkpoints (the main checkout when run from a worktree), or ``--db``.
"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from ..catalog import ROOT, Catalog, runs_root
from .app import BOT_SPEC, PublicTable, create_public_app
from .export import export
from .records import DB_FILE, Records

STATIC = ROOT / "web" / "out"
OUT = Path("runs/public")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server.public", description=__doc__)
    parser.add_argument("--db", type=Path, help=f"the games database (default <root>/{DB_FILE})")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the public table")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8100")))
    serve.add_argument("--static", type=Path, default=STATIC, help="the public page build")
    serve.add_argument("--bot", default=BOT_SPEC, help="the bot behind every B-Tig seat")
    out = commands.add_parser("export", help="write the saved games for the evaluators")
    out.add_argument("--out", type=Path, help=f"where to write (default <root>/{OUT})")
    args = parser.parse_args(argv)

    root = runs_root()
    records = Records(args.db if args.db is not None else root / DB_FILE)
    if args.command == "export":
        folder: Path = args.out if args.out is not None else root / OUT
        done = export(records, folder)
        print(f"{done.finished} finished and {done.unfinished} unfinished games, in {folder}")
        for file, games in sorted(done.players.items()):
            print(f"  players/{file}: {games} games")
        return

    logging.basicConfig(level=logging.INFO)
    static: Path = args.static
    if not (static / "index.html").is_file():
        parser.error(f"no page build in {static}: run `npm run build:public` in web/ first")
    table = PublicTable(Catalog.from_env(), records, bot=args.bot)
    table.catalog.make(args.bot, 0, table.rules)  # fail now, not on the first game
    # No access log: it would write every visitor's IP address down.
    uvicorn.run(create_public_app(table, static), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
