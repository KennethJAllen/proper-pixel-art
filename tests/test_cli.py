"""Smoke tests for the ``ppa`` command line entry point.

Runs ``main()`` end to end on a small asset and checks output path handling
and argument errors. Visual quality is validated separately by eye -- see
CONTRIBUTING.md -> Visual validation.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

from proper_pixel_art import cli


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa", *argv])
    cli.main()


def test_main_writes_output_file(
    monkeypatch: pytest.MonkeyPatch, assets: Path, tmp_path: Path
) -> None:
    """Pixelating an asset via the CLI produces a valid, non-empty PNG."""
    input_path = assets / "anchor" / "anchor.png"
    out_path = tmp_path / "result.png"

    run_cli(monkeypatch, str(input_path), "-o", str(out_path), "-c", "8")

    assert out_path.is_file()
    with Image.open(out_path) as result:
        assert result.width > 0 and result.height > 0


def test_main_output_to_directory(
    monkeypatch: pytest.MonkeyPatch, assets: Path, tmp_path: Path
) -> None:
    """A directory output path gets the default '<stem>_<W>x<H>.png' name."""
    input_path = assets / "anchor" / "anchor.png"

    run_cli(monkeypatch, str(input_path), "-o", str(tmp_path), "-c", "8")

    assert (tmp_path / "anchor_32x32.png").is_file()


def test_main_requires_input_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run_cli(monkeypatch)


def test_resolve_output_path_directory() -> None:
    resolved = cli.resolve_output_path(Path("out"), Path("input/sprite.png"))
    assert resolved == Path("out") / "sprite_pixelated.png"


def test_resolve_output_path_file_passthrough() -> None:
    resolved = cli.resolve_output_path(Path("out/result.png"), Path("sprite.png"))
    assert resolved == Path("out/result.png")
