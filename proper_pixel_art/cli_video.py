"""Command line interface for video/GIF pixelation."""

import argparse
from importlib.metadata import version
from pathlib import Path

from proper_pixel_art import video
from proper_pixel_art.cli import add_pixelation_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pixelate a video or GIF into true-resolution pixel art."
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version('proper-pixel-art')}",
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Path to the source video or GIF."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        type=Path,
        help="Path to the source video or GIF.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        type=Path,
        default=Path("."),
        help="Output path. Can be a directory or file path.",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["mp4", "gif"],
        default=None,
        help="Output format (default: inferred from output extension, then input extension).",
    )
    parser.add_argument(
        "-n",
        "--sample-frames",
        dest="sample_frames",
        type=int,
        default=8,
        help="Number of frames to sample for mesh detection (default: 8).",
    )

    add_pixelation_args(parser)

    args = parser.parse_args()

    if args.input_path is None and args.input_path_flag is None:
        parser.error("You must provide an input path (positional or with -i).")
    args.input_path = (
        args.input_path if args.input_path is not None else args.input_path_flag
    )

    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    video.pixelate_video(
        input_path=input_path,
        output_path=Path(args.out_path),
        num_colors=args.num_colors,
        scale_result=args.scale_result,
        transparent_background=bool(args.transparent_background),
        pixel_width=args.pixel_width,
        initial_upscale_factor=args.initial_upscale_factor,
        output_format=args.output_format,
        num_sample_frames=args.sample_frames,
    )


if __name__ == "__main__":
    main()
