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


def test_main_intermediate_dir_uses_input_subdir(
    monkeypatch: pytest.MonkeyPatch, assets: Path, tmp_path: Path
) -> None:
    """Image debug images land in a per-input subdirectory named after the stem."""
    input_path = assets / "anchor" / "anchor.png"
    intermediate_dir = tmp_path / "intermediate"

    run_cli(
        monkeypatch,
        str(input_path),
        "-o",
        str(tmp_path / "out.png"),
        "-c",
        "8",
        "--intermediate-dir",
        str(intermediate_dir),
    )

    assert (intermediate_dir / "anchor" / "mesh.png").is_file()


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


@pytest.fixture(name="gif_path")
def fixture_gif_path(tmp_path: Path) -> Path:
    """A small programmatically generated pixel-art GIF (no committed assets)."""
    input_path = tmp_path / "anim.gif"
    _save_gif(_make_noisy_arrays(3), input_path, [50] * 3)
    return input_path


def test_parse_args_positional_input() -> None:
    args = cli.parse_args(["in.gif"])
    assert args.input_path == Path("in.gif")
    # Unset flags stay None ("not provided") so they fall back to --config
    assert args.config is None
    assert args.intermediate_dir is None
    assert args.transparent_background is None
    assert args.color_method is None
    assert args.colors is None
    assert args.output_format is None
    assert args.sample_frames is None


def test_parse_args_config_flag() -> None:
    args = cli.parse_args(["in.gif", "--config", "cfg.yaml"])
    assert args.config == Path("cfg.yaml")


def test_parse_args_video_flags() -> None:
    """-f/--format and -n/--sample-frames parse on the main ppa command."""
    args = cli.parse_args(["in.gif", "-f", "mp4", "-n", "4"])
    assert args.output_format == "mp4"
    assert args.sample_frames == 4


def test_parse_args_no_transparent() -> None:
    """BooleanOptionalAction: --no-transparent overrides a config's true."""
    assert cli.parse_args(["in.png", "-t"]).transparent_background is True
    assert (
        cli.parse_args(["in.png", "--no-transparent"]).transparent_background is False
    )


def test_colors_flag_implies_palette_method() -> None:
    args = cli.parse_args(["in.png", "-c", "8"])
    cfg = cli.config_from_args(args)
    assert cfg.colors.method == "palette"
    assert cfg.colors.palette.num_colors == 8


def test_colors_flag_conflicts_with_dominant_method() -> None:
    args = cli.parse_args(["in.png", "-c", "8", "--color-method", "dominant"])
    with pytest.raises(ValueError, match="palette method"):
        cli.config_from_args(args)


def test_video_flags_land_in_video_config() -> None:
    args = cli.parse_args(["in.gif", "-f", "mp4", "-n", "4"])
    cfg = cli.config_from_args(args)
    assert cfg.video.output_format == "mp4"
    assert cfg.video.num_sample_frames == 4


def test_web_subcommand_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ppa web` launches the Gradio UI with the parsed host/port."""
    launched = {}

    class FakeDemo:
        def launch(self, server_name=None, server_port=None):
            launched["host"] = server_name
            launched["port"] = server_port

    import proper_pixel_art.web as web

    monkeypatch.setattr(web, "create_demo", lambda: FakeDemo())
    cli.main(["web", "--host", "0.0.0.0", "--port", "7861"])
    assert launched == {"host": "0.0.0.0", "port": 7861}


def test_video_main_requires_input_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "-f", "gif")


def test_video_main_with_config_yaml(
    monkeypatch: pytest.MonkeyPatch, gif_path: Path, tmp_path: Path
) -> None:
    """--config values are applied end to end (scale_result is observable),
    and --intermediate-dir produces debug images."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "colors:\n  method: palette\n  palette:\n    num_colors: 8\nscale_result: 2\n"
    )
    out_path = tmp_path / "result.gif"
    intermediate_dir = tmp_path / "intermediate"

    run_cli(
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
    # Debug images land in a per-input subdirectory named after the input stem.
    assert (intermediate_dir / gif_path.stem / "mesh.png").is_file()
