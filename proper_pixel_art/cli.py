"""Command line interface"""

import argparse
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate
from proper_pixel_art.config import PixelateConfig


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a true-resolution pixel-art image from a source image."
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version('proper-pixel-art')}",
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Path to the source input file."
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

    # Flags default to None so unset ones fall back to --config; see pixelate().
    add_pixelation_args(parser)

    args = parser.parse_args()

    # Either take the input as the first argument or use the -i flag
    if args.input_path is None and args.input_path_flag is None:
        parser.error("You must provide an input path (positional or with -i).")
    args.input_path = (
        args.input_path if args.input_path is not None else args.input_path_flag
    )

    return args


def resolve_output_path(
    out_path: Path, input_path: Path, suffix: str = "_pixelated"
) -> Path:
    """
    If outpath is a directory, make it a file path
    with filename e.g. (input stem)_pixelated.png
    """
    if out_path.suffix:
        return out_path
    filename = f"{input_path.stem}{suffix}.png"
    return out_path / filename


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    out_path = resolve_output_path(Path(args.out_path), input_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    config = PixelateConfig.from_yaml(args.config) if args.config else None

    # Each arg's dest matches a pixelate kwarg; None values fall back to config.
    pixelate_fields = (
        "num_colors",
        "scale_result",
        "transparent_background",
        "pixel_width",
        "initial_upscale_factor",
    )
    overrides = {field: getattr(args, field) for field in pixelate_fields}

    if args.intermediate_dir is not None:
        args.intermediate_dir.mkdir(exist_ok=True, parents=True)

    img = Image.open(input_path)
    pixelated = pixelate(
        img, config=config, intermediate_dir=args.intermediate_dir, **overrides
    )

    pixelated.save(out_path)


if __name__ == "__main__":
    main()
