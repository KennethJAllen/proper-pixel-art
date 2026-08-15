"""Command line interface."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate, utils
from proper_pixel_art.config import PixelateConfig
from proper_pixel_art.diagnostics import diagnose_image, format_diagnostics

# Inputs with these suffixes are dispatched to the video pipeline; everything
# else is treated as a still image. GIFs always take the video path, which
# preserves animation and handles single-frame GIFs correctly.
VIDEO_SUFFIXES = frozenset({".mp4", ".gif", ".webm", ".mov", ".avi", ".mkv", ".m4v"})


def add_pixelation_args(
    parser: argparse.ArgumentParser,
    group_name: str = "Pixelation options",
) -> argparse.ArgumentParser:
    """Add common pixelation arguments to an argument parser.

    Every flag defaults to ``None`` meaning "not provided". Unset flags fall
    back to the config or built-in defaults inside ``pixelate``.
    """
    pixel_group = parser.add_argument_group(group_name)

    pixel_group.add_argument(
        "-c",
        "--colors",
        dest="num_colors",
        type=int,
        default=None,
        help=(
            "Number of colors to quantize the image to (1-256). "
            "Use 0 to skip quantization and preserve all colors."
        ),
    )

    pixel_group.add_argument(
        "-s",
        "--scale-result",
        dest="scale_result",
        type=int,
        default=None,
        help="Width of the 'pixels' in the output image (1 = no scaling).",
    )

    pixel_group.add_argument(
        "-t",
        "--transparent",
        dest="transparent_background",
        action="store_true",
        default=None,
        help="Produce a transparent background in the output if set.",
    )

    pixel_group.add_argument(
        "-w",
        "--pixel-width",
        dest="pixel_width",
        type=int,
        default=None,
        help=(
            "Width of the pixels in the input image. "
            "Use 0 or omit it to determine the width automatically."
        ),
    )

    pixel_group.add_argument(
        "-u",
        "--initial-upscale",
        dest="initial_upscale_factor",
        type=int,
        default=None,
        help=(
            "Initial image upscale factor in the mesh detection algorithm. "
            "If the detected spacing is too large, it may be useful to "
            "increase this value."
        ),
    )

    return parser


# Each destination added by add_pixelation_args matches a
# pixelate/pixelate_video keyword argument.
PIXELATION_FIELDS = (
    "num_colors",
    "scale_result",
    "transparent_background",
    "pixel_width",
    "initial_upscale_factor",
)


def add_video_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add video/GIF-only arguments."""
    video_group = parser.add_argument_group("Video/GIF options")

    video_group.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["mp4", "gif"],
        default=None,
        help=(
            "Output format. Defaults to the output extension, then the input "
            "extension. Applies to video/GIF inputs only; ignored for images."
        ),
    )

    video_group.add_argument(
        "-n",
        "--sample-frames",
        dest="sample_frames",
        type=int,
        default=8,
        help=(
            "Number of frames to sample for mesh detection (default: 8). "
            "Applies to video/GIF inputs only; ignored for images."
        ),
    )

    return parser


def add_config_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add shared config and intermediate-output arguments."""
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=None,
        help=(
            "Path to a YAML config file of pixelation parameters. "
            "Explicit flags override values from the config file."
        ),
    )

    parser.add_argument(
        "--intermediate-dir",
        dest="intermediate_dir",
        type=Path,
        default=None,
        help=(
            "Directory to save images visualizing intermediate algorithm "
            "steps. Created if needed."
        ),
    )

    return parser


def collect_pixelation_overrides(
    args: argparse.Namespace,
) -> dict:
    """Collect explicit pixelation flag values from parsed arguments.

    Values are ``None`` when a flag was not provided, allowing downstream
    functions to use config or built-in defaults.
    """
    return {field: getattr(args, field) for field in PIXELATION_FIELDS}


def resolve_input_path(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve the positional input path against the ``-i`` flag."""
    if args.input_path is None and args.input_path_flag is None:
        parser.error(
            "You must provide an input path (positionally or with -i/--input)."
        )

    args.input_path = (
        args.input_path if args.input_path is not None else args.input_path_flag
    )

    return args


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate true-resolution pixel art from a source image, "
            "video, or GIF. Video and GIF inputs are automatically "
            "dispatched to the video pipeline."
        )
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version('proper-pixel-art')}",
    )

    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Path to the source image, video, or GIF.",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        metavar="INPUT_PATH",
        type=Path,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        type=Path,
        default=Path("."),
        help=(
            "Path where the pixelated image will be saved. "
            "Can be either a directory or a file path."
        ),
    )

    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Analyze the input image and print a diagnostic report "
            "instead of pixelating it."
        ),
    )

    add_config_args(parser)
    add_pixelation_args(parser)
    add_video_args(parser)

    args = parser.parse_args()

    return resolve_input_path(parser, args)


def resolve_output_path(
    out_path: Path,
    input_path: Path,
    suffix: str = "_pixelated",
) -> Path:
    """Resolve the final output path."""
    return utils.build_output_path(
        out_path,
        input_path,
        suffix,
        ext="png",
    )


# Flags that --diagnose does not act on, mapped to their user-facing names.
# Only flags defaulting to None are listed: for those, a non-None value means
# the user passed them explicitly. -o/--output and -n/--sample-frames have
# non-None defaults, so an explicit value is indistinguishable from the
# default and they are left out.
DIAGNOSE_IGNORED_FLAGS = (
    ("num_colors", "-c/--colors"),
    ("scale_result", "-s/--scale-result"),
    ("transparent_background", "-t/--transparent"),
    ("pixel_width", "-w/--pixel-width"),
    ("initial_upscale_factor", "-u/--initial-upscale"),
    ("config", "--config"),
    ("intermediate_dir", "--intermediate-dir"),
    ("output_format", "-f/--format"),
)


def warn_ignored_diagnose_flags(args: argparse.Namespace) -> None:
    """Warn about explicitly passed flags that ``--diagnose`` ignores."""
    names = [
        name
        for dest, name in DIAGNOSE_IGNORED_FLAGS
        if getattr(args, dest, None) is not None
    ]

    if names:
        print(f"Warning: --diagnose ignores: {', '.join(names)}", file=sys.stderr)


def run_diagnostics(input_path: Path) -> None:
    """Analyze a still image and print the diagnostic report."""
    with Image.open(input_path) as image:
        diagnostics = diagnose_image(image)

    print(format_diagnostics(diagnostics))


def main() -> None:
    """Run the command-line application."""
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")

    is_video = input_path.suffix.lower() in VIDEO_SUFFIXES

    if args.diagnose:
        if is_video:
            raise SystemExit(
                "--diagnose currently supports still images only. "
                "Video and GIF diagnostics are not supported yet."
            )

        warn_ignored_diagnose_flags(args)

        run_diagnostics(input_path)
        return

    config = PixelateConfig.from_yaml(args.config) if args.config else None

    # None values fall back to config or built-in defaults inside pixelate.
    overrides = collect_pixelation_overrides(args)

    # Debug images go into a per-input subdirectory named after the input
    # stem, for example: foo/wisp/.
    intermediate_dir = args.intermediate_dir

    if intermediate_dir is not None:
        intermediate_dir = intermediate_dir / input_path.stem
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    if is_video:
        # Deferred import so image runs do not pay the video import cost.
        from proper_pixel_art import video

        video.pixelate_video(
            input_path=input_path,
            output_path=Path(args.out_path),
            output_format=args.output_format,
            num_sample_frames=args.sample_frames,
            intermediate_dir=intermediate_dir,
            config=config,
            **overrides,
        )
        return

    with Image.open(input_path) as image:
        pixelated = pixelate(
            image,
            config=config,
            intermediate_dir=intermediate_dir,
            **overrides,
        )

    width, height = pixelated.size

    out_path = resolve_output_path(
        Path(args.out_path),
        input_path,
        f"_{width}x{height}",
    )

    out_path.parent.mkdir(exist_ok=True, parents=True)
    pixelated.save(out_path)


if __name__ == "__main__":
    main()
