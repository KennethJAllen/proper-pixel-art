"""Command line interface"""

import argparse
from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a true-resolution pixel-art image from a source image."
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Path to the source input file."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        type=Path,
        help="Path to the source input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        type=Path,
        default=".",
        help="Path where the pixelated image will be saved. Can be either a directory or a file path.",
    )
    parser.add_argument(
        "-c",
        "--colors",
        dest="num_colors",
        type=int,
        default=None,
        help="Number of colors to quantize the image to (1-256). Omit to skip quantization and preserve all colors.",
    )
    parser.add_argument(
        "-s",
        "--scale-result",
        dest="scale_result",
        type=int,
        default=1,
        help="Width of the 'pixels' in the output image (default: 1).",
    )
    parser.add_argument(
        "-t",
        "--transparent",
        dest="transparent",
        action="store_true",
        default=False,
        help="Produce a transparent background in the output if set.",
    )
    parser.add_argument(
        "-w",
        "--pixel-width",
        dest="pixel_width",
        type=int,
        default=None,
        help="Width of the pixels in the input image. If not set, it will be determined automatically.",
    )
    parser.add_argument(
        "-u",
        "--initial-upscale",
        dest="initial_upscale",
        type=int,
        default=2,
        help=(
            "Initial image upscale factor in mesh detection algorithm. "
            "If the detected spacing is too large, "
            "it may be useful to increase this value."
        ),
    )
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

    img = Image.open(input_path)
    pixelated = pixelate.pixelate(
        img,
        num_colors=args.num_colors,
        scale_result=args.scale_result,
        transparent_background=args.transparent,
        pixel_width=args.pixel_width,
        initial_upscale_factor=args.initial_upscale,
    )

    pixelated.save(out_path)


if __name__ == "__main__":
    main()
