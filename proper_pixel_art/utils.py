"""Utility functions"""

from typing import Iterator
from pathlib import Path
from PIL import Image, ImageSequence, ImageDraw
import cv2

Lines = list[int]  # Lines are a list of pixel indices for an image
Mesh = tuple[
    Lines, Lines
]  # A mesh is a tuple of lists of x coordinates and y coordinates for lines


def crop_border(image: Image.Image, num_pixels: int = 1) -> Image.Image:
    """
    Crop the boder of an image by a few pixels.
    Sometimes when requesting an image from GPT-4o with a transparent background,
    the boarder pixels will not be transparent, so just remove them.
    """
    width, height = image.size
    box = (num_pixels, num_pixels, width - num_pixels, height - num_pixels)
    cropped = image.crop(box)
    return cropped


def overlay_grid_lines(
    image: Image.Image,
    mesh: Mesh,
    line_color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 1,
) -> Image.Image:
    """
    Overlay mesh which includes vertical (lines_x) and horizontal (lines_y) grid lines
    over image for visualization.
    """
    # Ensure we draw on an RGBA canvas
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    lines_x, lines_y = mesh

    w, h = canvas.size
    # Draw each vertical line
    for x in lines_x:
        draw.line([(x, 0), (x, h)], fill=(*line_color, 255), width=line_width)

    # Draw each horizontal line
    for y in lines_y:
        draw.line([(0, y), (w, y)], fill=(*line_color, 255), width=line_width)

    return canvas


def scale_img(img: Image.Image, scale: int) -> Image.Image:
    """Scales the image up via nearest neightbor by scale factor."""
    w, h = img.size
    w_new, h_new = int(w * scale), int(h * scale)
    new_size = w_new, h_new
    scaled_img = img.resize(new_size, resample=Image.Resampling.NEAREST)
    return scaled_img


def extract_frames_gif(path: str) -> Iterator[tuple[Image.Image, int]]:
    """
    Extract the frames from a .gif file.
    Yields a tuple of the frame image and duration.
    """
    im = Image.open(path)
    for frame in ImageSequence.Iterator(im):
        duration_ms = frame.info.get("duration", 10)
        yield frame.convert("RGB"), duration_ms


def extract_frames_mp4(path: str) -> Iterator[tuple[Image.Image, int]]:
    """
    Extracts the frames from a .mp4 file.
    Yields a tuple of the frame image and duration.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0  # fallback if fps metadata is missing
    duration_ms = int(round(1000 / fps))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(frame_rgb), duration_ms
    cap.release()


def extract_frames(path: Path) -> Iterator[tuple[Image.Image, int]]:
    """
    Extract frames from gif or video.
    Yields (PIL.Image, duration_ms).
    """
    if path.suffix.lower() == ".gif":
        yield from extract_frames_gif(path)
    else:
        yield from extract_frames_mp4(path)
