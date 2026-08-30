"""Command-line entry point for locked scientific evaluation."""

from __future__ import annotations

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="RateMem-DiT locked scientific evaluation.",
)


@app.callback()
def root() -> None:
    """Validate, freeze, replay, and publish scientific artifacts."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
