"""Smoke tests for the ``ppa`` and ``ppa-video`` command line entry points.

Runs ``main()`` end to end on a small asset and checks output path handling
and argument errors. Visual quality is validated separately by eye -- see
CONTRIBUTING.md -> Visual validation.
"""

import argparse
import os
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


def run_video_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa-video", *argv])
    cli_video.main()


@pytest.fixture(name="gif_path")
def fixture_gif_path(tmp_path: Path) -> Path:
    """A small programmatically generated pixel-art GIF (no committed assets)."""
    input_path = tmp_path / "anim.gif"
    _save_gif(_make_noisy_arrays(3), input_path, [50] * 3)
    return input_path


def test_parse_args_positional_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa", "in.gif"])
    args = cli.parse_args()
    assert args.input_path == Path("in.gif")
    # Unset flags stay None ("not provided") so they fall back to --config
    assert args.config is None
    assert args.intermediate_dir is None
    assert args.transparent_background is None
    assert args.output_format is None
    assert args.sample_frames is None
    assert args.out_path is None
    assert args.diagnose is False


def test_parse_args_input_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa", "-i", "in.gif"])
    args = cli.parse_args()
    assert args.input_path == Path("in.gif")


def test_parse_args_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa", "in.gif", "--config", "cfg.yaml"])
    args = cli.parse_args()
    assert args.config == Path("cfg.yaml")


def test_parse_args_video_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """-f/--format and -n/--sample-frames parse on the main ppa command."""
    monkeypatch.setattr(sys, "argv", ["ppa", "in.gif", "-f", "mp4", "-n", "4"])
    args = cli.parse_args()
    assert args.output_format == "mp4"
    assert args.sample_frames == 4


def test_parse_args_diagnose_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ppa", "in.png", "--diagnose"])
    assert cli.parse_args().diagnose is True


def test_add_pixelation_args_excludes_diagnose() -> None:
    """--diagnose belongs to `ppa` only, not the shared pixelation helper that
    scripts/ppa_gen.py reuses."""
    parser = cli.add_pixelation_args(argparse.ArgumentParser())

    assert "--diagnose" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--diagnose"])


def test_main_diagnose_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """--diagnose reports on the image instead of writing an output file."""
    input_path = assets / "anchor" / "anchor.png"

    run_cli(monkeypatch, str(input_path), "--diagnose")

    stdout = capsys.readouterr().out
    assert "Image diagnosis" in stdout
    assert "Size: 500x500" in stdout
    assert "Noise level: low" in stdout


def test_main_diagnose_rejects_video(
    monkeypatch: pytest.MonkeyPatch, gif_path: Path
) -> None:
    with pytest.raises(SystemExit, match="still images"):
        run_cli(monkeypatch, str(gif_path), "--diagnose")


def test_main_diagnose_warns_on_ignored_flags(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Pixelation flags are inert under --diagnose, so they get called out."""
    input_path = assets / "anchor" / "anchor.png"

    run_cli(monkeypatch, str(input_path), "--diagnose", "-c", "8")
    stderr = capsys.readouterr().err
    assert "--diagnose ignores" in stderr
    assert "-c/--colors" in stderr

    run_cli(monkeypatch, str(input_path), "--diagnose")
    assert capsys.readouterr().err == ""


def test_main_diagnose_warns_on_ignored_output_flag(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """-o is inert under --diagnose too: the report only goes to stdout."""
    input_path = assets / "anchor" / "anchor.png"
    out_path = tmp_path / "report.txt"

    run_cli(monkeypatch, str(input_path), "--diagnose", "-o", str(out_path))

    assert "-o/--output" in capsys.readouterr().err
    assert not out_path.exists()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-c", "8"], "-c/--colors"),
        (["-s", "2"], "-s/--scale-result"),
        (["-t"], "-t/--transparent"),
        (["-w", "4"], "-w/--pixel-width"),
        (["-u", "2"], "-u/--initial-upscale"),
        (["-f", "gif"], "-f/--format"),
        (["-n", "4"], "-n/--sample-frames"),
        (["--config", "cfg.yaml"], "--config"),
        (["--intermediate-dir", "steps"], "--intermediate-dir"),
    ],
)
def test_main_diagnose_warns_on_every_ignored_flag(
    monkeypatch: pytest.MonkeyPatch,
    assets: Path,
    capsys: pytest.CaptureFixture,
    argv: list[str],
    expected: str,
) -> None:
    """Every flag --diagnose does not act on is named in the warning."""
    input_path = assets / "anchor" / "anchor.png"

    run_cli(monkeypatch, str(input_path), "--diagnose", *argv)

    assert expected in capsys.readouterr().err


def test_collect_ignored_diagnose_flags_ignores_declared_defaults() -> None:
    """Detection is by what argv supplies, not by comparing against defaults.

    Guards the flags declared the ordinary argparse way -- ``store_true``
    with ``default=False``, or any concrete default -- which a
    value-versus-default comparison would silently drop from the warning.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--flag", action="store_true", default=False)
    parser.add_argument("--valued", type=int, default=8)

    assert cli.collect_ignored_diagnose_flags(parser, ["--diagnose"]) == []
    assert cli.collect_ignored_diagnose_flags(parser, ["--diagnose", "--flag"]) == [
        "--flag"
    ]
    # Supplying exactly the default value still counts as supplying it.
    assert cli.collect_ignored_diagnose_flags(
        parser, ["--diagnose", "--valued", "8"]
    ) == ["--valued"]

    # The parser is left as it was found.
    assert parser.get_default("valued") == 8
    assert parser.get_default("flag") is False


def test_main_reports_missing_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A path that does not exist is reported as missing."""
    with pytest.raises(SystemExit, match="does not exist"):
        run_cli(monkeypatch, str(tmp_path / "missing.png"))


@pytest.mark.skipif(
    not hasattr(os, "geteuid"), reason="requires POSIX permission semantics"
)
def test_main_reports_unreadable_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permission-blocked input is reported as unreadable, not missing."""
    if os.geteuid() == 0:
        pytest.skip("permissions are not enforced for root")

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    input_path = blocked_dir / "sprite.png"
    input_path.touch()
    blocked_dir.chmod(0o000)

    try:
        # chmod is advisory on some mounts (WSL DrvFs, CIFS) and is bypassed
        # by CAP_DAC_OVERRIDE, so confirm the block took before asserting.
        try:
            input_path.stat()
        except OSError:
            pass
        else:
            pytest.skip("filesystem does not enforce directory permissions")

        with pytest.raises(SystemExit, match="Cannot access input file"):
            run_cli(monkeypatch, str(input_path))
    finally:
        blocked_dir.chmod(0o755)


def test_main_reports_non_image_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A readable file PIL cannot decode is a CLI error, not a traceback."""
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not an image")

    for extra in ([], ["--diagnose"]):
        with pytest.raises(SystemExit, match="not a supported image"):
            run_cli(monkeypatch, str(input_path), *extra)


def test_ppa_video_alias_deprecated(
    monkeypatch: pytest.MonkeyPatch,
    gif_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """The deprecated ppa-video alias still works end to end and notes the
    deprecation on stderr."""
    out_path = tmp_path / "result.gif"

    run_video_cli(monkeypatch, str(gif_path), "-o", str(out_path), "-c", "8")

    assert "deprecated" in capsys.readouterr().err
    assert out_path.is_file()
    with Image.open(out_path) as result:
        assert result.n_frames == 3


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
    # Debug images land in a per-input subdirectory named after the input stem.
    assert (intermediate_dir / gif_path.stem / "mesh.png").is_file()
