"""Sequential frame I/O for videos and GIFs.

GIFs are decoded with Pillow so that frame disposal, partial (offset) frames,
per-frame durations, and transparency are handled correctly: every yielded
frame is a full logical-screen RGBA image. Other formats (mp4, ...) are decoded
with OpenCV using sequential reads only — random seeks via
``CAP_PROP_POS_FRAMES`` are slow and unreliable for some codecs.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

DEFAULT_FPS = 24.0
DEFAULT_GIF_DURATION_MS = 100


@dataclass
class VideoInfo:
    """Basic metadata about a video/GIF file."""

    n_frames: int  # exact for GIF; a codec estimate for other formats
    fps: float  # average rate for GIFs with variable frame durations
    size: tuple[int, int]  # (width, height)
    exact_count: bool


def _is_gif(path: Path) -> bool:
    if path.suffix.lower() == ".gif":
        return True
    try:
        with Image.open(path) as img:
            return img.format == "GIF"
    except OSError:
        return False


def _frame_duration_ms(frame: Image.Image) -> int:
    duration = frame.info.get("duration", DEFAULT_GIF_DURATION_MS)
    # Some GIFs declare a 0ms delay; browsers clamp those to a sane default.
    return int(duration) if duration else DEFAULT_GIF_DURATION_MS


def probe(path: Path) -> VideoInfo:
    """Read metadata for a video/GIF without keeping frames in memory."""
    path = Path(path)
    if _is_gif(path):
        durations = [d for _, d in _iter_gif_durations(path)]
        if not durations:
            raise ValueError(f"Could not read any frames from {path}")
        with Image.open(path) as img:
            size = img.size
        fps = 1000.0 / (sum(durations) / len(durations))
        return VideoInfo(n_frames=len(durations), fps=fps, size=size, exact_count=True)

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    return VideoInfo(
        n_frames=n_frames,
        fps=fps if fps > 0 else DEFAULT_FPS,
        size=size,
        exact_count=False,
    )


def _iter_gif_durations(path: Path) -> Iterator[tuple[int, int]]:
    """Yield (frame_index, duration_ms) without converting frames to RGBA."""
    with Image.open(path) as img:
        for index, frame in enumerate(ImageSequence.Iterator(img)):
            yield index, _frame_duration_ms(frame)


def _iter_gif_frames(path: Path) -> Iterator[tuple[Image.Image, int]]:
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            # Pillow composites disposal/offset semantics when seeking, so the
            # converted frame is always the full logical-screen size.
            yield frame.convert("RGBA"), _frame_duration_ms(frame)


def _iter_cv2_frames(path: Path) -> Iterator[tuple[Image.Image, int]]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_ms = int(round(1000.0 / (fps if fps > 0 else DEFAULT_FPS)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb).convert("RGBA"), duration_ms
    finally:
        cap.release()


def iter_frames(path: Path) -> Iterator[tuple[Image.Image, int]]:
    """Yield (full-size RGBA frame, duration_ms) for every frame, in order."""
    path = Path(path)
    if _is_gif(path):
        yield from _iter_gif_frames(path)
    else:
        yield from _iter_cv2_frames(path)


def read_sample_frames(
    path: Path, num_samples: int, info: VideoInfo | None = None
) -> list[Image.Image]:
    """
    Collect up to ``num_samples`` evenly-spaced RGBA frames in one sequential pass.

    The spacing is based on the probed frame count; if that count is a codec
    over-estimate, fewer frames are returned (whatever was actually decodable).
    If the count is unknown or an under-estimate, all frames are read and the
    sample is drawn evenly from the decoded frames.

    Args:
        path: Path to the video/GIF file
        num_samples: Number of frames to sample
        info: Already-probed metadata for ``path``; pass it to avoid a second
            probe pass (probing a GIF walks every frame for durations).
    """
    if info is None:
        info = probe(path)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    if info.n_frames > 0:
        num_samples = min(num_samples, info.n_frames)
        targets = set(np.linspace(0, info.n_frames - 1, num_samples, dtype=int))
        samples = [
            frame
            for index, (frame, _) in enumerate(iter_frames(path))
            if index in targets
        ]
        if samples:
            return samples

    # Unknown/bad frame count: decode everything and sample evenly after the fact.
    frames = [frame for frame, _ in iter_frames(path)]
    if not frames:
        raise ValueError(f"Could not read any frames from {path}")
    num_samples = min(num_samples, len(frames))
    indices = np.unique(np.linspace(0, len(frames) - 1, num_samples, dtype=int))
    return [frames[i] for i in indices]
