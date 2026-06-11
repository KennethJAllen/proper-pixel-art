"""Video and GIF pixelation support.

Pixelates animations in two passes for temporal consistency and speed:

1. Sample a few evenly-spaced frames and make every global decision once:
   the pixel mesh (from aggregated edge maps), the color palette, and the
   background color for transparency.
2. Stream all frames through a :class:`FramePipeline` that applies those
   decisions with vectorized per-frame work (no Python per-cell loops, no
   per-frame resizes).
"""

from collections import Counter
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils, video_io
from proper_pixel_art.config import ColorConfig, MeshConfig, PixelateConfig
from proper_pixel_art.pixelate import (
    build_cell_map,
    downsample_binned,
    downsample_quantized,
)
from proper_pixel_art.utils import Mesh

# Keep a grid edge if it appears in at least this fraction of sampled frames.
# In clean pixel-art video every content edge lies on the grid, so a low
# threshold strengthens the Hough evidence; requiring more than one frame
# still suppresses single-frame compression noise. A strict majority vote
# would erase grid segments only visible where moving content happens to
# create contrast.
DEFAULT_MIN_VOTE_FRACTION = 0.25

# Pixel budget for the all-frames mosaic used to build the shared GIF palette.
_MAX_PALETTE_MOSAIC_PIXELS = 2_000_000


def aggregate_edge_maps(
    frames: list[Image.Image],
    upscale_factor: int = 2,
    mesh_config: MeshConfig | None = None,
    min_vote_fraction: float = DEFAULT_MIN_VOTE_FRACTION,
) -> np.ndarray:
    """
    Compute per-frame edge maps and keep edges present in at least
    ``min_vote_fraction`` of the frames.

    Args:
        frames: Sampled RGBA frames (all the same size)
        upscale_factor: Factor to upscale frames before edge detection
        mesh_config: Tunable mesh-detection parameters (Canny, closing, ...)
        min_vote_fraction: Minimum fraction of frames an edge pixel must
            appear in to be kept

    Returns:
        Aggregated binary edge map (uint8, values 0 or 255)
    """
    if not frames:
        raise ValueError("Cannot aggregate edge maps without frames")
    mesh_config = mesh_config or MeshConfig()

    accumulator = None
    for frame in frames:
        scaled_frame = utils.scale_img(frame.convert("RGBA"), upscale_factor)
        edge_map = mesh.compute_edge_map(scaled_frame, mesh_config=mesh_config)
        if accumulator is None:
            accumulator = np.zeros(edge_map.shape, dtype=np.int32)
        accumulator += edge_map > 0

    min_votes = max(1, ceil(min_vote_fraction * len(frames)))
    return ((accumulator >= min_votes) * 255).astype(np.uint8)


def compute_video_mesh(
    frames: list[Image.Image],
    upscale_factor: int = 2,
    pixel_width: int | None = None,
    output_dir: Path | None = None,
    mesh_config: MeshConfig | None = None,
    min_vote_fraction: float = DEFAULT_MIN_VOTE_FRACTION,
) -> tuple[Mesh, int]:
    """
    Compute a master mesh for a video from sampled frames.
    Falls back to upscale_factor=1 if the upscaled mesh is trivial.

    Args:
        frames: Sampled RGBA frames (all the same size)
        upscale_factor: Initial upscale factor for edge detection
        pixel_width: If set, skip automatic pixel width detection
        output_dir: If set, save debug images
        mesh_config: Tunable mesh-detection parameters
        min_vote_fraction: See :func:`aggregate_edge_maps`

    Returns:
        Tuple of (mesh, upscale_factor used)
    """
    aggregated = aggregate_edge_maps(
        frames,
        upscale_factor=upscale_factor,
        mesh_config=mesh_config,
        min_vote_fraction=min_vote_fraction,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(aggregated, mode="L").save(output_dir / "aggregated_edges.png")

    mesh_lines = mesh.compute_mesh_from_edges(
        aggregated, pixel_width=pixel_width, output_dir=output_dir
    )
    if not mesh.is_trivial_mesh(mesh_lines):
        return mesh_lines, upscale_factor

    # Fallback: try without upscaling
    aggregated_fallback = aggregate_edge_maps(
        frames,
        upscale_factor=1,
        mesh_config=mesh_config,
        min_vote_fraction=min_vote_fraction,
    )
    fallback_mesh = mesh.compute_mesh_from_edges(
        aggregated_fallback, pixel_width=pixel_width, output_dir=output_dir
    )
    return fallback_mesh, 1


def build_global_palette(
    frames: list[Image.Image],
    num_colors: int,
    color_config: ColorConfig | None = None,
) -> tuple[Image.Image, str]:
    """
    Build one shared palette from sampled frames so every frame quantizes to
    the same colors (no palette flicker between frames).

    Returns:
        Tuple of (P-mode palette image, background hex used by clamp_alpha).
        The background hex must be reused when clamping each frame so
        transparent regions map to the same palette entry everywhere.
    """
    color_config = color_config or ColorConfig()
    mosaic_array = np.concatenate(
        [np.asarray(frame.convert("RGBA")) for frame in frames], axis=0
    )
    mosaic = Image.fromarray(mosaic_array, mode="RGBA")

    common = colors.top_opaque_colors(
        mosaic,
        color_config.alpha_threshold,
        limit=color_config.top_colors_limit,
        thumbnail_size=color_config.thumbnail_size,
    )
    background = colors.pick_background(
        common, candidates=color_config.background_candidates
    )
    background_hex = "#{:02x}{:02x}{:02x}".format(*background)

    clamped = colors.clamp_alpha(
        mosaic,
        alpha_threshold=color_config.alpha_threshold,
        mode="RGB",
        background_hex=background_hex,
    )
    palette_img = clamped.quantize(
        colors=num_colors, method=color_config.quantize, dither=Image.Dither.NONE
    )
    return palette_img, background_hex


class FramePipeline:
    """Applies precomputed global decisions (mesh, palette, background) to
    individual frames with vectorized per-frame work.

    The mesh was detected on frames upscaled by ``upscale_factor``; instead of
    resizing every frame, the constructor precomputes nearest-neighbor gather
    indices so per-frame data computed at the original resolution is expanded
    into the upscaled cell grid — identical cell statistics, no per-frame
    resize.
    """

    def __init__(
        self,
        mesh_lines: Mesh,
        upscale_factor: int,
        frame_size: tuple[int, int],
        num_colors: int | None,
        sample_frames: list[Image.Image],
        transparent_background: bool = False,
        scale_result: int | None = None,
        color_config: ColorConfig | None = None,
    ):
        width, height = frame_size
        factor = upscale_factor
        self.color_config = color_config or ColorConfig()
        self.transparent_background = transparent_background
        self.scale_result = scale_result
        self.skip_quantization = not num_colors  # 0 / None -> skip

        self.cell_map = build_cell_map(mesh_lines, (height * factor, width * factor))
        self._row_idx = np.arange(height * factor) // factor
        self._col_idx = np.arange(width * factor) // factor

        if self.skip_quantization:
            self.palette_img = None
            self.palette_rgb = None
            self.background_hex = None
        else:
            self.palette_img, self.background_hex = build_global_palette(
                sample_frames, num_colors, color_config=self.color_config
            )
            self.palette_rgb = np.array(
                self.palette_img.getpalette(), dtype=np.uint8
            ).reshape(-1, 3)

        self.background_color: colors.RGB | None = None
        if transparent_background:
            self.background_color = self._pick_transparency_color(sample_frames)

    def _upscaled(self, array: np.ndarray) -> np.ndarray:
        """Nearest-neighbor expand an original-resolution array to the
        upscaled grid the mesh was computed on."""
        return array[self._row_idx][:, self._col_idx]

    def _downsample_frame(self, frame: Image.Image) -> Image.Image:
        """Collapse one frame to true-resolution RGBA (no post-processing)."""
        frame_rgba = frame.convert("RGBA")

        if self.skip_quantization:
            rgba = self._upscaled(np.asarray(frame_rgba))
            out = downsample_binned(rgba, self.cell_map, self.color_config)
        else:
            clamped = colors.clamp_alpha(
                frame_rgba,
                alpha_threshold=self.color_config.alpha_threshold,
                mode="RGB",
                background_hex=self.background_hex,
            )
            quantized = clamped.quantize(
                palette=self.palette_img, dither=Image.Dither.NONE
            )
            palette_idx = self._upscaled(np.asarray(quantized))
            alpha = self._upscaled(np.asarray(frame_rgba)[..., 3])
            out = downsample_quantized(
                palette_idx, alpha, self.cell_map, self.palette_rgb, self.color_config
            )

        return Image.fromarray(out, mode="RGBA")

    def _pick_transparency_color(self, sample_frames: list[Image.Image]) -> colors.RGB:
        """Most common boundary color across the downsampled sample frames,
        decided once so every frame clears the same background color."""
        boundary_counts: Counter = Counter()
        for frame in sample_frames:
            small = np.asarray(self._downsample_frame(frame).convert("RGB"))
            boundary = np.concatenate(
                [small[0], small[-1], small[1:-1, 0], small[1:-1, -1]]
            )
            boundary_counts.update(map(tuple, boundary))
        return boundary_counts.most_common(1)[0][0]

    def process(self, frame: Image.Image) -> Image.Image:
        """Pixelate a single frame using the precomputed global decisions."""
        result = self._downsample_frame(frame)

        if self.background_color is not None:
            result = colors.apply_background_transparency(result, self.background_color)
        if self.scale_result and self.scale_result > 1:
            result = utils.scale_img(result, int(self.scale_result))
        return result


def pixelate_frame(
    frame: Image.Image,
    mesh_lines: Mesh,
    upscale_factor: int,
    num_colors: int | None = None,
    transparent_background: bool = False,
    scale_result: int | None = None,
) -> Image.Image:
    """
    Pixelate a single frame using an externally-provided mesh.

    Convenience wrapper building a one-off :class:`FramePipeline`; when
    processing many frames, build the pipeline once instead.
    """
    frame_rgba = frame.convert("RGBA")
    pipeline = FramePipeline(
        mesh_lines,
        upscale_factor,
        frame_rgba.size,
        num_colors,
        sample_frames=[frame_rgba],
        transparent_background=transparent_background,
        scale_result=scale_result,
    )
    return pipeline.process(frame_rgba)


def _write_mp4(
    frames_iter, output_path: Path, fps: float, frame_size: tuple[int, int]
) -> None:
    """Write frames to MP4, streaming one frame at a time.

    Tries the H.264 fourcc first and falls back to MPEG-4 Part 2, which is
    always available in opencv-python builds.
    """
    writer = None
    chosen_codec = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if writer.isOpened():
            chosen_codec = codec
            break
        writer.release()
        writer = None
    if writer is None:
        raise ValueError(f"Could not open a video writer for {output_path}")

    print(f"Writing MP4 with codec {chosen_codec}")
    try:
        for frame in frames_iter:
            rgb = np.array(frame.convert("RGB"))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_gif(
    frames: list[Image.Image],
    output_path: Path,
    durations_ms: list[int],
    color_config: ColorConfig | None = None,
) -> None:
    """Write frames to GIF with a single shared palette and per-frame durations.

    GIF transparency is binary: pixels with alpha below the configured
    threshold map to one reserved transparent palette index.
    """
    color_config = color_config or ColorConfig()
    alpha_arrays = [np.asarray(f.convert("RGBA"))[..., 3] for f in frames]
    has_transparency = any(
        (alpha < color_config.alpha_threshold).any() for alpha in alpha_arrays
    )

    # One palette for the whole animation: quantize a mosaic of all frames.
    # Reserve index 255 for transparency when needed.
    max_colors = 255 if has_transparency else 256
    rgb_frames = [f.convert("RGB") for f in frames]
    # Frames are nearest-neighbor upscaled true-resolution images, so every
    # color is duplicated scale_result**2 times; stride-subsampling large
    # mosaics bounds memory without meaningfully changing the palette.
    total_pixels = sum(f.width * f.height for f in rgb_frames)
    stride = max(1, round((total_pixels / _MAX_PALETTE_MOSAIC_PIXELS) ** 0.5))
    mosaic_array = np.concatenate(
        [np.asarray(f)[::stride, ::stride] for f in rgb_frames], axis=0
    )
    global_palette = Image.fromarray(mosaic_array, mode="RGB").quantize(
        colors=max_colors, method=color_config.quantize, dither=Image.Dither.NONE
    )
    palette_data = global_palette.getpalette()
    palette_data = palette_data + [0] * (768 - len(palette_data))

    p_frames = []
    for rgb, alpha in zip(rgb_frames, alpha_arrays, strict=True):
        p_frame = rgb.quantize(palette=global_palette, dither=Image.Dither.NONE)
        if has_transparency:
            indices = np.asarray(p_frame).copy()
            indices[alpha < color_config.alpha_threshold] = 255
            p_frame = Image.fromarray(indices, mode="P")
        p_frame.putpalette(palette_data)
        p_frames.append(p_frame)

    save_kwargs = {}
    if has_transparency:
        save_kwargs.update(transparency=255, disposal=2)
    p_frames[0].save(
        output_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        # Passing the shared palette makes Pillow write one global color
        # table instead of a local table on every delta frame.
        palette=bytes(palette_data),
        **save_kwargs,
    )


def pixelate_video(
    input_path: Path,
    output_path: Path,
    num_colors: int | None = None,
    scale_result: int | None = None,
    transparent_background: bool = False,
    pixel_width: int | None = None,
    initial_upscale_factor: int | None = None,
    output_format: str | None = None,
    num_sample_frames: int = 8,
) -> Path:
    """
    Pixelate a video or GIF file.

    Samples a few frames to compute a master mesh and a global color palette,
    then applies both to every frame for a temporally consistent result.

    Args:
        input_path: Path to input video/GIF
        output_path: Path for output file (directory or file)
        num_colors: Number of colors for quantization (0/None to skip)
        scale_result: Upscale result by this factor
        transparent_background: Make background transparent
        pixel_width: Override automatic pixel width detection (0/None = auto)
        initial_upscale_factor: Upscale factor for mesh detection
        output_format: Output format ("mp4" or "gif"). Inferred if None.
        num_sample_frames: Frames to sample for mesh/palette detection

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    initial_upscale_factor = (
        initial_upscale_factor or PixelateConfig().initial_upscale_factor
    )
    pixel_width = pixel_width or None  # 0 / None -> auto-detect

    # Resolve output format
    if output_format is None:
        if output_path.suffix:
            output_format = output_path.suffix.lstrip(".")
        else:
            # Inferring from the input is best-effort: any non-GIF input
            # (webm, mov, ...) simply produces an mp4.
            inferred = input_path.suffix.lstrip(".").lower()
            output_format = inferred if inferred in ("mp4", "gif") else "mp4"
    output_format = output_format.lower()
    if output_format not in ("mp4", "gif"):
        raise ValueError(
            f"Unsupported output format {output_format!r}: expected 'mp4' or "
            "'gif'. Use a .mp4/.gif output path or pass output_format."
        )

    # Resolve output path
    if not output_path.suffix:
        output_path = output_path / f"{input_path.stem}_pixelated.{output_format}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if transparent_background and output_format == "gif":
        print(
            "Warning: GIF only supports binary transparency. "
            "Results may not look as expected with --transparent."
        )

    info = video_io.probe(input_path)

    # Pass 1: sample frames, then make all global decisions from them
    sample_frames = video_io.read_sample_frames(input_path, num_sample_frames, info)
    mesh_lines, upscale_factor = compute_video_mesh(
        sample_frames,
        upscale_factor=initial_upscale_factor,
        pixel_width=pixel_width,
    )
    pipeline = FramePipeline(
        mesh_lines,
        upscale_factor,
        info.size,
        num_colors,
        sample_frames,
        transparent_background=transparent_background,
        scale_result=scale_result,
    )

    # Pass 2: stream every frame through the pipeline
    total = info.n_frames if info.n_frames > 0 else "?"
    durations: list[int] = []

    def processed_frames():
        for index, (frame, duration) in enumerate(video_io.iter_frames(input_path)):
            durations.append(duration)
            print(f"\rProcessing frame {index + 1}/{total}", end="", flush=True)
            yield pipeline.process(frame)
        print()

    if output_format == "gif":
        frames = list(processed_frames())
        if not frames:
            raise ValueError(f"Could not read any frames from {input_path}")
        _write_gif(frames, output_path, durations)
    else:
        frames_iter = processed_frames()
        try:
            first = next(frames_iter)
        except StopIteration:
            raise ValueError(f"Could not read any frames from {input_path}") from None

        def with_first():
            yield first
            yield from frames_iter

        _write_mp4(with_first(), output_path, info.fps, (first.width, first.height))
        if durations and max(durations) > 1.1 * min(durations):
            print(
                "Warning: input has variable frame durations; "
                "MP4 output uses a constant frame rate."
            )

    print(f"Saved pixelated video to {output_path}")
    return output_path
