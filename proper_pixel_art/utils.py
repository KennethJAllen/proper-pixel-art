"""Utility functions"""
from PIL import Image, ImageDraw

def crop_border(image : Image.Image, num_pixels: int=1) -> Image.Image:
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
        lines_x: list[int],
        lines_y: list[int],
        line_color: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 1
        ) -> Image.Image:
    """
    Overlay vertical (lines_x) and horizontal (lines_y) grid lines over image for visualization.
    """
    # Ensure we draw on an RGBA canvas
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

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
