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
from dataclasses import replace
from fractions import Fraction
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from proper_pixel_art import colors, mesh, utils, video_io
from proper_pixel_art.config import (
    ColorConfig,
    MeshConfig,
    PixelateConfig,
    with_num_colors,
)
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


def _edge_density_profiles(edge_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column/row edge-density sums of a binary edge map, used as the 1-D
    evidence profiles for line snapping. There is no single grayscale source
    for the aggregated map, but edge density peaks at the same grid boundaries
    a gradient profile would."""
    binary = (edge_map > 0).astype(np.float64)
    return binary.sum(axis=0), binary.sum(axis=1)


def _grey_stack(
    frames: list[Image.Image],
    upscale_factor: int,
    mesh_config: MeshConfig | None = None,
) -> np.ndarray:
    """(frames, H, W) grayscale stack over the sampled frames, replicating the
    exact per-frame transform the edge maps saw (upscale, then crop/clamp via
    ``mesh.compute_grey``) so it lives in edge-map coordinates. A stack rather
    than a mean: averaging frames with moving content washes out the spatial
    structure that width scoring depends on."""
    return np.stack(
        [
            mesh.compute_grey(
                utils.scale_img(frame.convert("RGBA"), upscale_factor),
                mesh_config=mesh_config,
            )
            for frame in frames
        ]
    )


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

    # The aggregated edge map is at the upscaled resolution (each frame is scaled
    # by upscale_factor before edge detection), so the debug overlay must match.
    representative = (
        utils.scale_img(frames[0].convert("RGBA"), upscale_factor)
        if output_dir is not None
        else None
    )
    mesh_lines = mesh.compute_mesh_from_edges(
        aggregated,
        pixel_width=pixel_width,
        output_dir=output_dir,
        original_img=representative,
        mesh_config=mesh_config,
        profiles=_edge_density_profiles(aggregated),
        score_image=_grey_stack(frames, upscale_factor, mesh_config),
    )
    if not mesh.is_trivial_mesh(mesh_lines):
        return mesh_lines, upscale_factor

    # Fallback: try without upscaling. This overwrites the debug images from the
    # attempt above, which is fine — the fallback mesh is the one we return.
    aggregated_fallback = aggregate_edge_maps(
        frames,
        upscale_factor=1,
        mesh_config=mesh_config,
        min_vote_fraction=min_vote_fraction,
    )
    fallback_mesh = mesh.compute_mesh_from_edges(
        aggregated_fallback,
        pixel_width=pixel_width,
        output_dir=output_dir,
        original_img=frames[0].convert("RGBA") if output_dir is not None else None,
        mesh_config=mesh_config,
        profiles=_edge_density_profiles(aggregated_fallback),
        # The fallback edge map is at the original scale, so the score image
        # must be recomputed at scale 1 rather than reused.
        score_image=_grey_stack(frames, 1, mesh_config),
    )
    return fallback_mesh, 1


def build_global_palette(
    frames: list[Image.Image],
    color_config: ColorConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[Image.Image, str]:
    """
    Build one shared palette from sampled frames so every frame quantizes to
    the same colors (no palette flicker between frames).

    Args:
        output_dir: If set, save the quantized sample mosaic as a debug image.

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
    palette = color_config.palette
    palette_img = clamped.quantize(
        colors=palette.num_colors,
        method=palette.quantize,
        dither=palette.dither_mode,
        kmeans=palette.kmeans,
    )
    if output_dir is not None:
        palette_img.save(output_dir / "quantized_original.png")
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
        sample_frames: list[Image.Image],
        transparent_background: bool = False,
        scale_result: int | None = None,
        color_config: ColorConfig | None = None,
        intermediate_dir: Path | None = None,
    ):
        width, height = frame_size
        factor = upscale_factor
        self.color_config = color_config or ColorConfig()
        self.transparent_background = transparent_background
        self.scale_result = scale_result
        self.skip_quantization = self.color_config.method == "dominant"

        self.cell_map = build_cell_map(mesh_lines, (height * factor, width * factor))
        self._row_idx = np.arange(height * factor) // factor
        self._col_idx = np.arange(width * factor) // factor

        if self.skip_quantization:
            self.palette_img = None
            self.palette_rgb = None
            self.background_hex = None
        else:
            self.palette_img, self.background_hex = build_global_palette(
                sample_frames,
                color_config=self.color_config,
                output_dir=intermediate_dir,
            )
            self.palette_rgb = np.array(
                self.palette_img.getpalette(), dtype=np.uint8
            ).reshape(-1, 3)

        # One merger fitted on the sample frames gives every frame the same
        # color mapping, so merging cannot flicker between frames.
        self.color_merger: colors.ColorMerger | None = None
        dominant = self.color_config.dominant
        if self.skip_quantization and dominant.merge_distance > 0:
            small_samples = [self._downsample_frame(f) for f in sample_frames]
            self.color_merger = colors.ColorMerger(
                dominant.merge_distance,
                linkage=dominant.merge_linkage,
                max_colors=dominant.max_linkage_colors,
            ).fit(small_samples)

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
                palette=self.palette_img,
                dither=self.color_config.palette.dither_mode,
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
            small_img = self._downsample_frame(frame)
            # Match process(): the boundary color is picked on merged colors
            # because transparency is applied after merging there too.
            if self.color_merger is not None:
                small_img = self.color_merger.apply(small_img)
            small = np.asarray(small_img.convert("RGB"))
            boundary = np.concatenate(
                [small[0], small[-1], small[1:-1, 0], small[1:-1, -1]]
            )
            boundary_counts.update(map(tuple, boundary))
        return boundary_counts.most_common(1)[0][0]

    def process(self, frame: Image.Image) -> Image.Image:
        """Pixelate a single frame using the precomputed global decisions."""
        result = self._downsample_frame(frame)

        if self.color_merger is not None:
            result = self.color_merger.apply(result)
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
    ``num_colors`` is the usual shorthand: >= 1 selects the palette method
    with that palette size, 0 / None selects the dominant method.
    """
    frame_rgba = frame.convert("RGBA")
    color_config = with_num_colors(PixelateConfig(), num_colors or 0).colors
    pipeline = FramePipeline(
        mesh_lines,
        upscale_factor,
        frame_rgba.size,
        sample_frames=[frame_rgba],
        transparent_background=transparent_background,
        scale_result=scale_result,
        color_config=color_config,
    )
    return pipeline.process(frame_rgba)


def _write_mp4(
    frames_iter, output_path: Path, fps: float, frame_size: tuple[int, int]
) -> None:
    """Write frames to MP4, streaming one frame at a time.

    Encodes with libx264 at crf=1 (near-lossless) in yuv420p, the one pixel
    format every player (including Safari/QuickTime) accepts. The 4:2:0
    chroma subsampling is invisible for pixel art once each logical pixel
    spans at least a 2x2 block, and crf=1 preserves the hard edges that
    default rate control smears.
    """
    import av  # lazy: only MP4 writing needs it

    width, height = frame_size
    # yuv420p requires even dimensions; pad right/bottom by replicating the
    # edge row/column (visually invisible for pixel art).
    out_w, out_h = width + width % 2, height + height % 2

    print("Writing MP4 (libx264, crf 1)")
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(65535)
        )
        stream.width, stream.height = out_w, out_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "1", "preset": "medium"}
        for frame in frames_iter:
            rgb = np.asarray(frame.convert("RGB"))
            if (out_w, out_h) != (width, height):
                rgb = np.pad(
                    rgb, ((0, out_h - height), (0, out_w - width), (0, 0)), mode="edge"
                )
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            # Convert chroma with an exact 2x2 box filter ("AREA"): the
            # default bicubic filter's wide taps bleed chroma several pixels
            # across the hard color edges pixel art is made of.
            video_frame = video_frame.reformat(format="yuv420p", interpolation="AREA")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)


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
        colors=max_colors,
        method=color_config.palette.quantize,
        dither=Image.Dither.NONE,
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
    transparent_background: bool | None = None,
    pixel_width: int | None = None,
    initial_upscale_factor: int | None = None,
    output_format: str | None = None,
    num_sample_frames: int = 8,
    intermediate_dir: Path | None = None,
    config: PixelateConfig | None = None,
) -> Path:
    """
    Pixelate a video or GIF file.

    Samples a few frames to compute a master mesh and a global color palette,
    then applies both to every frame for a temporally consistent result.

    Every pixelation parameter defaults to ``None``, meaning "not provided" —
    the value is taken from ``config`` (or the built-in defaults). Pass a
    concrete value to override the config.

    Args:
        input_path: Path to input video/GIF
        output_path: Path for output file (directory or file)
        num_colors: Shorthand for colors.method: >= 1 selects the palette
            method with that palette size, 0 selects the dominant method
            (original colors preserved)
        scale_result: Upscale result by this factor
        transparent_background: Make background transparent
        pixel_width: Override automatic pixel width detection (0 = auto)
        initial_upscale_factor: Upscale factor for mesh detection
        output_format: Output format ("mp4" or "gif"). Defaults to "gif"
            regardless of the input format; a ".mp4" output path also
            selects "mp4".
        num_sample_frames: Frames to sample for mesh/palette detection
        intermediate_dir: Directory to save images visualizing intermediate steps
        config: A PixelateConfig bundling every tunable parameter. Load one from
            YAML with PixelateConfig.from_yaml. Any of the explicit arguments
            above, when provided (not None), override the corresponding value
            in config.

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Resolution order: explicit argument > config > built-in defaults.
    cfg = config if config is not None else PixelateConfig()
    overrides = {
        name: value
        for name, value in (
            ("initial_upscale_factor", initial_upscale_factor),
            ("scale_result", scale_result),
            ("transparent_background", transparent_background),
            ("pixel_width", pixel_width),
        )
        if value is not None
    }
    if overrides:
        cfg = replace(cfg, **overrides)
    if num_colors is not None:
        cfg = with_num_colors(cfg, num_colors)

    # Resolve output format: explicit argument, then output extension,
    # otherwise GIF regardless of the input format.
    if output_format is None:
        output_format = output_path.suffix.lstrip(".") if output_path.suffix else "gif"
    output_format = output_format.lower()
    if output_format not in ("mp4", "gif"):
        raise ValueError(
            f"Unsupported output format {output_format!r}: expected 'mp4' or "
            "'gif'. Use a .mp4/.gif output path or pass output_format."
        )

    # A directory output gets its default '{stem}_{W}x{H}.{ext}' filename once
    # the first processed frame reveals the output size.
    output_is_dir = not output_path.suffix

    if cfg.transparent_background and output_format == "gif":
        print(
            "Warning: GIF only supports binary transparency. "
            "Results may not look as expected with --transparent."
        )

    info = video_io.probe(input_path)

    # Pass 1: sample frames, then make all global decisions from them
    sample_frames = video_io.read_sample_frames(input_path, num_sample_frames, info)
    mesh_lines, upscale_factor = compute_video_mesh(
        sample_frames,
        upscale_factor=cfg.initial_upscale_factor,
        pixel_width=cfg.pixel_width or None,  # 0 / None -> auto-detect
        output_dir=intermediate_dir,
        mesh_config=cfg.mesh,
    )
    pipeline = FramePipeline(
        mesh_lines,
        upscale_factor,
        info.size,
        sample_frames,
        transparent_background=cfg.transparent_background,
        scale_result=cfg.scale_result,
        color_config=cfg.colors,
        intermediate_dir=intermediate_dir,
    )

    # Pass 2: stream every frame through the pipeline. tqdm renders the CLI
    # progress bar; under ppa-web, gr.Progress(track_tqdm=True) hooks the same
    # loop to drive the browser progress bar.
    total = info.n_frames if info.n_frames > 0 else None
    durations: list[int] = []

    def processed_frames():
        for frame, duration in tqdm(
            video_io.iter_frames(input_path),
            total=total,
            desc="Processing frames",
            unit="frame",
        ):
            durations.append(duration)
            yield pipeline.process(frame)

    if output_format == "gif":
        frames = list(processed_frames())
        if not frames:
            raise ValueError(f"Could not read any frames from {input_path}")
        if output_is_dir:
            width, height = frames[0].size
            output_path = utils.build_output_path(
                output_path, input_path, f"_{width}x{height}", output_format
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_gif(frames, output_path, durations, color_config=cfg.colors)
    else:
        frames_iter = processed_frames()
        try:
            first = next(frames_iter)
        except StopIteration:
            raise ValueError(f"Could not read any frames from {input_path}") from None

        def with_first():
            yield first
            yield from frames_iter

        if output_is_dir:
            output_path = utils.build_output_path(
                output_path, input_path, f"_{first.width}x{first.height}", output_format
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_mp4(with_first(), output_path, info.fps, (first.width, first.height))
        if durations and max(durations) > 1.1 * min(durations):
            print(
                "Warning: input has variable frame durations; "
                "MP4 output uses a constant frame rate."
            )

    print(f"Saved pixelated video to {output_path}")
    return output_path
