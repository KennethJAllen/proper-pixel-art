"""Utility functions"""

from pathlib import Path

from PIL import Image, ImageDraw

Lines = list[int]  # Lines are a list of pixel indices for an image
Mesh = tuple[
    Lines, Lines
]  # A mesh is a tuple of lists of x coordinates and y coordinates for lines


def build_output_path(
    out_path: Path, input_path: Path, stem_suffix: str = "", ext: str = "png"
) -> Path:
    """Resolve a directory output path to a default file name.

    A path with a suffix is treated as an explicit file path and returned
    unchanged; a directory gets ``{input stem}{stem_suffix}.{ext}``.
    """
    if out_path.suffix:
        return out_path
    return out_path / f"{input_path.stem}{stem_suffix}.{ext}"


def crop_border(image: Image.Image, num_pixels: int = 1) -> Image.Image:
    """
    Crop the border of an image by a few pixels.
    Sometimes when requesting an image from GPT-4o with a transparent background,
    the border pixels will not be transparent, so just remove them.
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
    """Scales the image up via nearest neighbor by scale factor."""
    w, h = img.size
    w_new, h_new = int(w * scale), int(h * scale)
    new_size = w_new, h_new
    scaled_img = img.resize(new_size, resample=Image.Resampling.NEAREST)
    return scaled_img
