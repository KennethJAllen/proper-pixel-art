"""Command line interface"""

import argparse
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate, utils
from proper_pixel_art.config import PixelateConfig

# Inputs with these suffixes are dispatched to the video pipeline; everything
# else is treated as a still image. GIFs always take the video path, which
# preserves animation (and handles single-frame GIFs fine).
VIDEO_SUFFIXES = frozenset({".mp4", ".gif", ".webm", ".mov", ".avi", ".mkv", ".m4v"})


def add_pixelation_args(
    parser: argparse.ArgumentParser,
    group_name: str = "Pixelation options",
) -> argparse.ArgumentParser:
    """Add common pixelation arguments to an argument parser.

    Every flag defaults to ``None`` (meaning "not provided"), so unset flags fall
    back to the config / built-in defaults inside ``pixelate``.

    Args:
        parser: The argument parser to add arguments to
        group_name: Name of the argument group (default: "Pixelation options")

    Returns:
        The parser with pixelation arguments added
    """
    pixel_group = parser.add_argument_group(group_name)
    pixel_group.add_argument(
        "-c",
        "--colors",
        dest="num_colors",
        type=int,
        default=None,
        help="Number of colors to quantize the image to (1-256). Use 0 to skip quantization and preserve all colors.",
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
        help="Width of the pixels in the input image. Use 0 (or omit) to determine it automatically.",
    )
    pixel_group.add_argument(
        "-u",
        "--initial-upscale",
        dest="initial_upscale_factor",
        type=int,
        default=None,
        help=(
            "Initial image upscale factor in mesh detection algorithm. "
            "If the detected spacing is too large, "
            "it may be useful to increase this value."
        ),
    )
    return parser


# Each dest added by add_pixelation_args matches a pixelate/pixelate_video kwarg.
PIXELATION_FIELDS = (
    "num_colors",
    "scale_result",
    "transparent_background",
    "pixel_width",
    "initial_upscale_factor",
)


def add_video_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the video/GIF-only arguments (ignored for image inputs)."""
    video_group = parser.add_argument_group("Video/GIF options")
    video_group.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["mp4", "gif"],
        default=None,
        help=(
            "Output format (default: gif; a .mp4 output path also selects mp4). "
            "Applies to video/GIF inputs only; ignored for images."
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


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--config`` / ``--intermediate-dir`` arguments shared by both CLIs."""
    parser.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=None,
        help="Path to a YAML config file of pixelation parameters. Any flags passed explicitly override values in the file.",
    )
    parser.add_argument(
        "--intermediate-dir",
        dest="intermediate_dir",
        type=Path,
        default=None,
        help="Directory to save images visualizing intermediate algorithm steps (created if needed).",
    )
    return parser


def collect_pixelation_overrides(args: argparse.Namespace) -> dict:
    """Collect the explicit pixelation flag values from parsed args.

    Values are ``None`` when the flag wasn't provided, so they fall back to
    the config / built-in defaults downstream.
    """
    return {field: getattr(args, field) for field in PIXELATION_FIELDS}


def resolve_input_path(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Resolve the positional input path against the ``-i``/``--input`` flag,
    erroring out when neither is given."""
    if args.input_path is None and args.input_path_flag is None:
        parser.error("You must provide an input path (positional or with -i).")
    args.input_path = (
        args.input_path if args.input_path is not None else args.input_path_flag
    )
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate true-resolution pixel art from a source image, video, or GIF. "
            "Video and GIF inputs are auto-detected and dispatched to the video pipeline."
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
        help="Path where the pixelated image will be saved. Can be either a directory or a file path.",
    )
    add_config_args(parser)

    # Flags default to None so unset ones fall back to --config; see pixelate().
    add_pixelation_args(parser)
    add_video_args(parser)

    args = parser.parse_args()

    # Either take the input as the first argument or use the -i flag
    return resolve_input_path(parser, args)


def resolve_output_path(
    out_path: Path, input_path: Path, suffix: str = "_pixelated"
) -> Path:
    """
    If out_path is a directory, make it a file path with filename
    ``(input stem){suffix}.png`` (main passes the output size as the suffix,
    e.g. ``sprite_128x128.png``). An out_path that already names a file is
    returned unchanged.
    """
    return utils.build_output_path(out_path, input_path, suffix, ext="png")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    config = PixelateConfig.from_yaml(args.config) if args.config else None

    # None values fall back to config / built-in defaults inside pixelate.
    overrides = collect_pixelation_overrides(args)

    # Debug images go in a per-input subdirectory named after the input stem
    # (e.g. --intermediate-dir foo + wisp.mp4 -> foo/wisp/), mirroring the
    # stem-based output-path convention and keeping multiple runs from
    # clobbering each other in a shared directory.
    intermediate_dir = args.intermediate_dir
    if intermediate_dir is not None:
        intermediate_dir = intermediate_dir / input_path.stem
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() in VIDEO_SUFFIXES:
        # Deferred import so image runs don't pay the cv2 import cost.
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

    img = Image.open(input_path)
    pixelated = pixelate(
        img, config=config, intermediate_dir=intermediate_dir, **overrides
    )

    width, height = pixelated.size

    out_path = resolve_output_path(
        Path(args.out_path), input_path, f"_{width}x{height}"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)

    pixelated.save(out_path)


if __name__ == "__main__":
    main()
