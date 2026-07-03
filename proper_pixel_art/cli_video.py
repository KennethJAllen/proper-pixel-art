"""Deprecated ``ppa-video`` entry point; kept as a thin alias for ``ppa``."""

import sys

from proper_pixel_art import cli


def main() -> None:
    print(
        "note: 'ppa-video' is deprecated; use 'ppa' (video inputs are auto-detected).",
        file=sys.stderr,
    )
    cli.main()


if __name__ == "__main__":
    main()
