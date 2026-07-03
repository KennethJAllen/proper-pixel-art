"""Command line interface for video/GIF pixelation."""

import argparse
from importlib.metadata import version
from pathlib import Path

from proper_pixel_art import video
from proper_pixel_art.cli import (
    add_config_args,
    add_pixelation_args,
    collect_pixelation_overrides,
    resolve_input_path,
)
from proper_pixel_art.config import PixelateConfig


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
    add_config_args(parser)

    # Flags default to None so unset ones fall back to --config; see pixelate_video().
    add_pixelation_args(parser)

    args = parser.parse_args()

    # Either take the input as the first argument or use the -i flag
    return resolve_input_path(parser, args)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()

    config = PixelateConfig.from_yaml(args.config) if args.config else None

    # None values fall back to config / built-in defaults inside pixelate_video.
    overrides = collect_pixelation_overrides(args)

    if args.intermediate_dir is not None:
        args.intermediate_dir.mkdir(exist_ok=True, parents=True)

    video.pixelate_video(
        input_path=input_path,
        output_path=Path(args.out_path),
        output_format=args.output_format,
        num_sample_frames=args.sample_frames,
        intermediate_dir=args.intermediate_dir,
        config=config,
        **overrides,
    )


if __name__ == "__main__":
    main()
