"""Run the web table's API: ``uv run python -m server [--host HOST] [--port PORT]``.

The port defaults to ``$PORT`` (portless sets it), else 8000. The API serves the catalog under
``$ESTIMATION_ROOT``, by default this checkout.
"""

import argparse
import os

import uvicorn

from .app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
