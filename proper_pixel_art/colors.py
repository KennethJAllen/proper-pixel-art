"""
Contains logic for handling image
"""

from pathlib import Path
from collections import Counter
from PIL import Image, ImageColor
from PIL.Image import Quantize
import numpy as np

RGB = tuple[int, int, int]

def _rgb_dist(a: RGB, b: RGB) -> int:
    """Naive color distance"""
    dr, dg, db = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    return dr*dr + dg*dg + db*db

def _top_opaque_colors(img: Image.Image, alpha_threshold: int, limit: int = 8) -> list[RGB]:
    """Return the most common opaque colors (RGB) up to limit."""
    rgba = img.convert("RGBA").copy()
    rgba.thumbnail((160, 160)) # speed and de-noise tiny details
    counts = Counter()
    for r, g, b, a in rgba.getdata():
        if a >= alpha_threshold:
            counts[(r, g, b)] += 1
    return [c for c, _ in counts.most_common(limit)]

def _pick_background(common: list[RGB]) -> RGB:
    """Pick the candidate farthest from the common colors"""
    background_color_candidates: list[RGB] = [
        (255, 255, 255), # white
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (0, 0, 255),     # blue
        (255, 255, 0),   # yellow
        (255, 0, 255),   # magenta
        (0, 255, 255),   # cyan
        (255, 128, 0),   # orange
        (128, 0, 255),   # violet
        (0, 128, 255),   # sky
        (0, 255, 128),   # mint
        (255, 0, 128),   # pink
    ]

    common = list(common)
    if not common:
        return (255, 255, 255)
    best, best_score = background_color_candidates[0], -1
    for cand in background_color_candidates:
        score = min(_rgb_dist(cand, c) for c in common)
        if score > best_score:
            best, best_score = cand, score
    return best

def clamp_alpha(
    image: Image.Image,
    alpha_threshold: int = 128,
    mode: str = "RGB",
    background_hex: str | None = None
    ) -> Image.Image:
    """
    Replace pixels with alpha < threshold by a background color.
    If background_hex is None, choose a color far from the most common image colors.
    mode: 'RGB' or 'L'
    """
    if mode not in ("RGB", "L"):
        raise ValueError("mode must be 'RGB' or 'L'")

    if background_hex is None:
        common = _top_opaque_colors(image, alpha_threshold)
        bg_rgb = _pick_background(common)
    else:
        bg_rgb = ImageColor.getrgb(background_hex)

    base = image.convert(mode)
    alpha = image.getchannel("A")
    mask = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)

    background = Image.new("RGB", image.size, bg_rgb).convert(mode)

    return Image.composite(base, background, mask)

def get_cell_color(cell_pixels: np.ndarray) -> RGB:
    """
    cell_pixels: shape (height_cell, width_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in the cell_pixels block.
    """
    # flatten to tuple of pixel values
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    cell_color = Counter(flat).most_common(1)[0][0]
    return cell_color

def palette_img(
        image: Image.Image,
        num_colors: int = 16,
        quantize_method: int = Quantize.MAXCOVERAGE,
        output_dir: Path | None = None) -> Image.Image:
    """
    Discretizes the colors in the image img to at most num_colors.
    Saves the quantized image to output_dir if not None.
    Returns the color pallete of the image.

    The maximum coverage algorithm is used by default as the quantization method.
    Emperically this algorithm proivdes the best results overall, although
    for some examples num_colors needs to be chosen very large even when the
    image has a small number of actual colors. In these instances, Quantize.FASTOCTREE
    can work instead.

    If the colors of the result don't look right, try increasing num_colors.
    """
    image_rgb = clamp_alpha(image, mode='RGB')
    quantized_img = image_rgb.quantize(colors=num_colors, method=quantize_method, dither=Image.Dither.NONE)
    if output_dir is not None:
        quantized_img.save(output_dir / "quantized_original.png")
    return quantized_img

def most_common_boundary_color(image: Image.Image) -> RGB:
    """
    Return the exact RGB color that occurs most on the image boundary.
    """
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size

    # top and bottom rows
    top = list(image_rgb.crop((0, 0, w, 1)).getdata())
    bottom = list(image_rgb.crop((0, h - 1, w, h)).getdata())
    # left and right columns (excluding corners)
    left = [image_rgb.getpixel((0, y)) for y in range(1, h - 1)]
    right = [image_rgb.getpixel((w - 1, y)) for y in range(1, h - 1)]

    counts = Counter(top + bottom + left + right)
    mode_color = counts.most_common(1)[0][0]
    return mode_color # (R, G, B)

def make_background_transparent(image: Image.Image) -> Image.Image:
    """
    Make the background fully transparent by:
      1) take the most common color on the boundary
      2) setting alpha=0 for all pixels equal to that color
    """
    background_color = most_common_boundary_color(image)
    image_rgba = image.convert("RGBA")
    px = list(image_rgba.getdata())

    out = []
    for (r, g, b, a) in px:
        # If the color is the same as the background color, make it transparent
        if (r, g, b) == background_color:
            out.append((r, g, b, 0))
        else:
            out.append((r, g, b, a or 255))

    image_rgba.putdata(out)
    return image_rgba

def main():
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    paletted = palette_img(img)
    paletted.show()

if __name__ == "__main__":
    main()
