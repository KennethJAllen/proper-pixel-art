"""Video and GIF pixelation support.

Uses aggregated edge maps from multiple frames to detect the pixel grid,
then applies that grid consistently to all frames.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.config import MeshConfig
from proper_pixel_art.pixelate import downsample
from proper_pixel_art.utils import Mesh


def _read_frame_as_pil(cap: cv2.VideoCapture) -> Image.Image | None:
    """Read a single frame from a VideoCapture and return as RGBA PIL Image."""
    ret, frame = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGBA")


def aggregate_edge_maps(
    video_path: Path,
    num_samples: int = 8,
    upscale_factor: int = 2,
    canny_thresholds: tuple[int] = (50, 200),
    closure_kernel_size: int = 8,
) -> tuple[np.ndarray, int]:
    """
    Sample K evenly-spaced frames, compute per-frame edge maps, and aggregate
    via majority vote to reinforce grid lines and dilute content edges.

    Args:
        video_path: Path to input video/GIF
        num_samples: Number of frames to sample for edge detection
        upscale_factor: Factor to upscale frames before edge detection
        canny_thresholds: Thresholds for Canny edge detection
        closure_kernel_size: Kernel size for morphological closing

    Returns:
        Tuple of (aggregated binary edge map, upscale_factor used)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(num_samples, total_frames)

    # Evenly-spaced frame indices
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    accumulator = None
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        frame = _read_frame_as_pil(cap)
        if frame is None:
            continue

        scaled_frame = utils.scale_img(frame, upscale_factor)
        edge_map = mesh.compute_edge_map(
            scaled_frame,
            mesh_config=MeshConfig(
                canny_thresholds=canny_thresholds,
                closure_kernel_size=closure_kernel_size,
            ),
        )

        if accumulator is None:
            accumulator = np.zeros_like(edge_map, dtype=np.float32)
        accumulator += (edge_map > 0).astype(np.float32)

    cap.release()

    if accumulator is None:
        raise ValueError(f"Could not read any frames from {video_path}")

    # Majority vote: pixel is edge if present in >50% of sampled frames
    threshold = num_samples / 2
    aggregated = ((accumulator > threshold) * 255).astype(np.uint8)
    return aggregated, upscale_factor


def compute_video_mesh(
    video_path: Path,
    num_samples: int = 8,
    upscale_factor: int = 2,
    pixel_width: int | None = None,
    output_dir: Path | None = None,
) -> tuple[Mesh, int]:
    """
    Compute a master mesh for a video using aggregated edge maps.
    Falls back to upscale_factor=1 if the upscaled mesh is trivial.

    Args:
        video_path: Path to input video/GIF
        num_samples: Number of frames to sample
        upscale_factor: Initial upscale factor for edge detection
        pixel_width: If set, skip automatic pixel width detection
        output_dir: If set, save debug images

    Returns:
        Tuple of (mesh, upscale_factor used)
    """
    aggregated, factor = aggregate_edge_maps(
        video_path, num_samples=num_samples, upscale_factor=upscale_factor
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        agg_img = Image.fromarray(aggregated, mode="L")
        agg_img.save(output_dir / "aggregated_edges.png")

    mesh_lines = mesh.compute_mesh_from_edges(
        aggregated, pixel_width=pixel_width, output_dir=output_dir
    )

    if not mesh._is_trivial_mesh(mesh_lines):
        return mesh_lines, factor

    # Fallback: try without upscaling
    aggregated_fallback, _ = aggregate_edge_maps(
        video_path, num_samples=num_samples, upscale_factor=1
    )
    fallback_mesh = mesh.compute_mesh_from_edges(
        aggregated_fallback, pixel_width=pixel_width, output_dir=output_dir
    )
    return fallback_mesh, 1


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

    Args:
        frame: RGBA PIL image
        mesh_lines: Pre-computed mesh (x_lines, y_lines)
        upscale_factor: Scale factor the mesh was computed at
        num_colors: Number of colors for quantization (None to skip)
        transparent_background: If True, make background transparent
        scale_result: If set, upscale the result

    Returns:
        Pixelated RGBA image
    """
    frame_rgba = frame.convert("RGBA")
    skip_quantization = num_colors is None

    if skip_quantization:
        processed_img = frame_rgba
    else:
        processed_img = colors.palette_img(frame_rgba, num_colors=num_colors)

    scaled_img = utils.scale_img(processed_img, upscale_factor)

    scaled_alpha_array = (
        None
        if skip_quantization
        else colors.extract_and_scale_alpha(frame_rgba, upscale_factor)
    )

    result = downsample(
        scaled_img,
        mesh_lines,
        skip_quantization=skip_quantization,
        original_alpha=scaled_alpha_array,
    )

    if transparent_background:
        result = colors.make_background_transparent(result)

    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result


def _get_video_fps(video_path: Path) -> float:
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24.0


def _write_mp4(frames_iter, output_path: Path, fps: float, frame_size: tuple[int, int]):
    """Write frames to MP4 using cv2.VideoWriter. Streams one frame at a time."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    for frame in frames_iter:
        rgb = np.array(frame.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def _write_gif(frames: list[Image.Image], output_path: Path, fps: float):
    """Write frames to GIF using PIL save_all."""
    duration_ms = int(1000 / fps)
    # Convert RGBA to P mode for GIF compatibility
    p_frames = []
    for f in frames:
        rgb = f.convert("RGB")
        p_frames.append(rgb.quantize(method=Image.Quantize.MEDIANCUT))
    p_frames[0].save(
        output_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def pixelate_video(
    input_path: Path,
    output_path: Path,
    num_colors: int | None = None,
    scale_result: int | None = None,
    transparent_background: bool = False,
    pixel_width: int | None = None,
    initial_upscale_factor: int = 2,
    output_format: str | None = None,
    num_sample_frames: int = 8,
) -> Path:
    """
    Pixelate a video or GIF file.

    Computes a master mesh from aggregated edge maps of sampled frames,
    then applies that mesh to every frame for consistent output.

    Args:
        input_path: Path to input video/GIF
        output_path: Path for output file (directory or file)
        num_colors: Number of colors for quantization (None to skip)
        scale_result: Upscale result by this factor
        transparent_background: Make background transparent
        pixel_width: Override automatic pixel width detection
        initial_upscale_factor: Upscale factor for mesh detection
        output_format: Output format ("mp4" or "gif"). Inferred if None.
        num_sample_frames: Frames to sample for mesh detection

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)

    # Resolve output format
    if output_format is None:
        if output_path.suffix:
            output_format = output_path.suffix.lstrip(".")
        else:
            output_format = input_path.suffix.lstrip(".")
    output_format = output_format.lower()
    if output_format not in ("mp4", "gif"):
        output_format = "mp4"

    # Resolve output path
    if not output_path.suffix:
        output_path = output_path / f"{input_path.stem}_pixelated.{output_format}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if transparent_background and output_format == "gif":
        print(
            "Warning: GIF only supports binary transparency. "
            "Results may not look as expected with --transparent."
        )

    # Compute master mesh
    mesh_lines, upscale_factor = compute_video_mesh(
        input_path,
        num_samples=num_sample_frames,
        upscale_factor=initial_upscale_factor,
        pixel_width=pixel_width,
    )

    fps = _get_video_fps(input_path)

    # Process all frames
    cap = cv2.VideoCapture(str(input_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_format == "gif":
        # GIF: collect all frames in memory
        frames = []
        for i in range(frame_count):
            frame = _read_frame_as_pil(cap)
            if frame is None:
                break
            result = pixelate_frame(
                frame,
                mesh_lines,
                upscale_factor,
                num_colors=num_colors,
                transparent_background=transparent_background,
                scale_result=scale_result,
            )
            frames.append(result)
            print(f"\rProcessing frame {i + 1}/{frame_count}", end="", flush=True)
        cap.release()
        print()

        if frames:
            _write_gif(frames, output_path, fps)
    else:
        # MP4: stream frames via generator for memory efficiency
        # First, pixelate one frame to get output dimensions
        frame = _read_frame_as_pil(cap)
        if frame is None:
            cap.release()
            raise ValueError(f"Could not read first frame from {input_path}")

        first_result = pixelate_frame(
            frame,
            mesh_lines,
            upscale_factor,
            num_colors=num_colors,
            transparent_background=transparent_background,
            scale_result=scale_result,
        )
        frame_size = (first_result.width, first_result.height)

        def frame_generator():
            yield first_result
            print(f"\rProcessing frame 1/{frame_count}", end="", flush=True)
            for i in range(1, frame_count):
                f = _read_frame_as_pil(cap)
                if f is None:
                    break
                result = pixelate_frame(
                    f,
                    mesh_lines,
                    upscale_factor,
                    num_colors=num_colors,
                    transparent_background=transparent_background,
                    scale_result=scale_result,
                )
                print(f"\rProcessing frame {i + 1}/{frame_count}", end="", flush=True)
                yield result
            print()

        _write_mp4(frame_generator(), output_path, fps, frame_size)
        cap.release()

    print(f"Saved pixelated video to {output_path}")
    return output_path
