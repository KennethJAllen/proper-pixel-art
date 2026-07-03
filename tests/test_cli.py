"""Smoke tests for the ``ppa`` and ``ppa-video`` command line entry points.

Runs ``main()`` end to end on a small asset and checks output path handling
and argument errors. Visual quality is validated separately by eye -- see
CONTRIBUTING.md -> Visual validation.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

from proper_pixel_art import cli, cli_video
from tests.test_video import LOGICAL_SIZE, _make_noisy_arrays, _save_gif


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


def test_main_dispatches_video_input(
    monkeypatch: pytest.MonkeyPatch, gif_path: Path, tmp_path: Path
) -> None:
    """`ppa` routes video/GIF inputs through the video pipeline by extension."""
    out_path = tmp_path / "result.gif"

    run_cli(monkeypatch, str(gif_path), "-o", str(out_path), "-c", "8")

    with Image.open(out_path) as result:
        assert result.n_frames == 3
        assert result.size == (LOGICAL_SIZE, LOGICAL_SIZE)


def run_video_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa-video", *argv])
    cli_video.main()


@pytest.fixture(name="gif_path")
def fixture_gif_path(tmp_path: Path) -> Path:
    """A small programmatically generated pixel-art GIF (no committed assets)."""
    input_path = tmp_path / "anim.gif"
    _save_gif(_make_noisy_arrays(3), input_path, [50] * 3)
    return input_path


def test_video_parse_args_positional_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa-video", "in.gif"])
    args = cli_video.parse_args()
    assert args.input_path == Path("in.gif")
    # Unset flags stay None ("not provided"), consistent with the image CLI
    assert args.config is None
    assert args.intermediate_dir is None
    assert args.transparent_background is None


def test_video_parse_args_input_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa-video", "-i", "in.gif"])
    args = cli_video.parse_args()
    assert args.input_path == Path("in.gif")


def test_video_parse_args_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa-video", "in.gif", "--config", "cfg.yaml"])
    args = cli_video.parse_args()
    assert args.config == Path("cfg.yaml")


def test_video_main_requires_input_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run_video_cli(monkeypatch)


def test_video_main_with_config_yaml(
    monkeypatch: pytest.MonkeyPatch, gif_path: Path, tmp_path: Path
) -> None:
    """--config values are applied end to end (scale_result is observable),
    and --intermediate-dir produces debug images."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("num_colors: 8\nscale_result: 2\n")
    out_path = tmp_path / "result.gif"
    intermediate_dir = tmp_path / "intermediate"

    run_video_cli(
        monkeypatch,
        str(gif_path),
        "-o",
        str(out_path),
        "--config",
        str(config_path),
        "--intermediate-dir",
        str(intermediate_dir),
    )

    with Image.open(out_path) as result:
        assert result.size == (LOGICAL_SIZE * 2, LOGICAL_SIZE * 2)
    assert (intermediate_dir / "aggregated_edges.png").is_file()
