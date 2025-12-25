"""Handles image colors logic"""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor
from PIL.Image import Quantize

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# Alpha threshold for determining pixel opacity
# Pixels with alpha >= this value are considered opaque
ALPHA_THRESHOLD = 128


def _is_majority_transparent(opaque_count: int, total_count: int) -> bool:
    return opaque_count <= total_count / 2


def _rgb_dist(a: RGB, b: RGB) -> int:
    """Naive color distance"""
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dr**2 + dg**2 + db**2


def _top_opaque_colors(
    img: Image.Image, alpha_threshold: int, limit: int = 8
) -> list[RGB]:
    """Return the most common opaque colors (RGB) up to limit."""
    rgba = img.convert("RGBA").copy()
    rgba.thumbnail((160, 160))  # speed and de-noise tiny details
    counts = Counter()
    for r, g, b, a in rgba.getdata():
        if a >= alpha_threshold:
            counts[(r, g, b)] += 1
    return [c for c, _ in counts.most_common(limit)]


def _pick_background(colors: list[RGB]) -> RGB:
    """
    Pick the candidate farthest from the common colors.
    Used for choosing the color of pixels with alpha to avoid clashing
    with the actual colors in the image
    """
    background_color_candidates: list[RGB] = [
        (0, 255, 255),  # cyan
        (255, 255, 255),  # white
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (255, 128, 0),  # orange
        (128, 0, 255),  # violet
        (0, 128, 255),  # sky
        (0, 255, 128),  # mint
        (255, 0, 128),  # pink
    ]

    if not colors:
        return (255, 255, 255)
    best, best_score = background_color_candidates[0], -1
    for color_candidate in background_color_candidates:
        score = min(_rgb_dist(color_candidate, c) for c in colors)
        if score > best_score:
            best, best_score = color_candidate, score
    return best


def clamp_alpha(
    image: Image.Image,
    alpha_threshold: int = ALPHA_THRESHOLD,
    mode: str = "RGB",
    background_hex: str | None = None,
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


def extract_and_scale_alpha(image: Image.Image, scale_factor: int = 1) -> np.ndarray:
    """
    Extract alpha channel from an RGBA image and scale it.

    Args:
        image: RGBA PIL Image
        scale_factor: Integer scale factor for resizing (default 1 = no scaling)

    Returns:
        Numpy array of alpha channel values (uint8), scaled if scale_factor != 1
    """
    # Extract alpha channel from RGBA image
    alpha_channel = np.array(image.convert("RGBA"))[:, :, 3]

    # Scale alpha channel if needed
    if scale_factor != 1:
        alpha_img = Image.fromarray(alpha_channel, mode="L")
        new_size = (alpha_img.width * scale_factor, alpha_img.height * scale_factor)
        scaled_alpha_img = alpha_img.resize(new_size, Image.Resampling.NEAREST)
        return np.array(scaled_alpha_img)
    else:
        return alpha_channel


def get_opaque_cell_color(cell_pixels: np.ndarray) -> RGBA:
    """
    cell_pixels: shape (height_cell, width_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in the cell_pixels block, with 255 in fourth entry for opaque alpha.
    """
    # flatten to tuple of pixel values
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    cell_color = Counter(flat).most_common(1)[0][0]
    return (*cell_color, 255)


def get_cell_color_with_alpha(cell_pixels: np.ndarray, cell_alpha: np.ndarray) -> RGBA:
    """
    Select representative color for a quantized cell, considering transparency.

    Decision logic:
    - If >=50% of pixels are transparent (alpha < ALPHA_THRESHOLD), return (0,0,0,0)
    - Otherwise, return the most common RGB color with full opacity (R,G,B,255)

    Args:
        cell_pixels: shape (height, width, 3), dtype=uint8 (RGB from quantized image)
        cell_alpha: shape (height, width), dtype=uint8 (alpha channel from original)

    Returns:
        RGBA tuple - either (0,0,0,0) for transparent or (R,G,B,255) for opaque
    """
    total_pixels = cell_alpha.size
    opaque_count = np.sum(cell_alpha >= ALPHA_THRESHOLD)

    # If majority is transparent (>= 50%), return fully transparent
    if _is_majority_transparent(opaque_count, total_pixels):
        return (0, 0, 0, 0)

    # Otherwise return most common color with full opacity
    cell_color = get_opaque_cell_color(cell_pixels)
    return cell_color


def _dominant_rgb_by_binning(rgb_pixels: np.ndarray) -> RGB:
    """
    Find the dominant color using offset binning algorithm.

    Algorithm (offset binning):
    1. Quantize RGB space into bins using two offset grids
       - Grid 1: boundaries at 0, 52, 104, 156, 208
       - Grid 2: boundaries at 26, 78, 130, 182, 234 (offset by half bin size)
    2. For each grid, find the bin with the most pixels
    3. Use the grid that produces the larger dominant cluster
    4. Return the median color of pixels in that bin

    This avoids bin boundary artifacts: colors straddling one grid's boundary
    will be grouped together in the other grid.

    For small pixel counts (<=3), the binning algorithm is skipped and
    either the single pixel value or median is returned directly.

    Args:
        rgb_pixels: shape (N, 3), dtype=uint8 - flat array of RGB values

    Returns:
        RGB tuple of the representative color
    """
    # Edge cases for small pixel counts - binning doesn't make sense
    if len(rgb_pixels) == 1:
        r, g, b = rgb_pixels[0]
        return (int(r), int(g), int(b))
    if len(rgb_pixels) <= 3:
        median = np.median(rgb_pixels, axis=0).astype(np.uint8)
        return (int(median[0]), int(median[1]), int(median[2]))

    bin_size = 52

    # Grid 1: standard binning (boundaries at 0, 52, 104...)
    bins1 = rgb_pixels // bin_size
    indices1 = bins1[:, 0] * 25 + bins1[:, 1] * 5 + bins1[:, 2]
    counts1 = np.bincount(indices1, minlength=125)
    dominant1 = np.argmax(counts1)
    max_count1 = counts1[dominant1]

    # Grid 2: offset binning (boundaries at 26, 78, 130...)
    # Add offset before dividing, clamp to avoid overflow
    offset = bin_size // 2
    bins2 = np.minimum(rgb_pixels + offset, 255) // bin_size
    indices2 = bins2[:, 0] * 25 + bins2[:, 1] * 5 + bins2[:, 2]
    counts2 = np.bincount(indices2, minlength=125)
    dominant2 = np.argmax(counts2)
    max_count2 = counts2[dominant2]

    # Use the grid with the larger dominant cluster
    if max_count1 >= max_count2:
        mask = indices1 == dominant1
    else:
        mask = indices2 == dominant2

    dominant_pixels = rgb_pixels[mask]

    # Return median of dominant bin (robust to outliers within bin)
    return tuple(np.median(dominant_pixels, axis=0).astype(np.uint8))


def get_cell_color_skip_quantization(
    cell_pixels: np.ndarray, alpha_threshold: int = ALPHA_THRESHOLD
) -> RGBA:
    """
    Select representative RGBA color for a cell when quantization is skipped.

    Used when quantization is skipped to preserve original colors
    while handling noise/grain and filtering out background bleed-in.

    Decision logic:
    - If >=50% of pixels are transparent (alpha < threshold), return fully transparent (0,0,0,0)
    - Otherwise, return fully opaque color computed from only the opaque pixels

    The color is determined by finding the densest color region using offset binning,
    which avoids bin boundary artifacts: colors straddling one grid's boundary
    will be grouped together in the other grid.

    Args:
        cell_pixels: shape (height, width, 4), dtype=uint8 (RGBA)
        alpha_threshold: minimum alpha to consider pixel opaque (default ALPHA_THRESHOLD)

    Returns:
        RGBA tuple - either (0,0,0,0) for transparent or (R,G,B,255) for opaque
    """
    pixels = cell_pixels.reshape(-1, 4)
    total_pixels = len(pixels)

    # Edge case: empty cell
    if total_pixels == 0:
        return (0, 0, 0, 0)

    # Filter to opaque pixels only
    opaque_mask = pixels[:, 3] >= alpha_threshold
    opaque_pixels = pixels[opaque_mask]

    # If majority is transparent (>= 50%), return fully transparent
    if _is_majority_transparent(len(opaque_pixels), total_pixels):
        return (0, 0, 0, 0)

    # Get RGB of opaque pixels and find dominant color
    rgb_pixels = opaque_pixels[:, :3]
    r, g, b = _dominant_rgb_by_binning(rgb_pixels)
    return (int(r), int(g), int(b), 255)


def palette_img(
    image: Image.Image,
    num_colors: int = 16,
    quantize_method: int = Quantize.MAXCOVERAGE,
    output_dir: Path | None = None,
) -> Image.Image:
    """
    Discretizes the colors in the image to at most num_colors.
    Saves the quantized image to output_dir if not None.
    Returns the color pallete of the image.

    The maximum coverage algorithm is used by default as the quantization method.
    Emperically this algorithm proivdes the best results overall, although
    for some examples num_colors needs to be chosen very large even when the
    image has a small number of actual colors. In these instances, Quantize.FASTOCTREE
    can work instead.

    If the colors of the result don't look right, try increasing num_colors.
    """
    image_rgb = clamp_alpha(image, mode="RGB")
    quantized_img = image_rgb.quantize(
        colors=num_colors, method=quantize_method, dither=Image.Dither.NONE
    )
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
    return mode_color  # (R, G, B)


def make_background_transparent(image: Image.Image) -> Image.Image:
    """
    Make the background fully transparent by:
      1) Identifying the most common color on the image boundary
      2) Setting alpha=0 for all pixels matching that color

    Note: This sets transparency for ALL pixels matching the boundary color,
    not just boundary pixels (no flood fill).
    """
    background_color = most_common_boundary_color(image)
    image_rgba = image.convert("RGBA")
    px = list(image_rgba.getdata())

    out = []
    for r, g, b, a in px:
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
