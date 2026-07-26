"""Scalar per-cell color selection, kept as the correctness oracle for the
vectorized downsample paths in ``proper_pixel_art``.

These were the original production implementations; the shipped code now uses
vectorized ports (``downsample_quantized`` / ``downsample_binned``), and the
tests compare those against these straightforward per-cell versions.
"""

from collections import Counter

import numpy as np

from proper_pixel_art.colors import RGBA, dominant_rgb_by_binning
from proper_pixel_art.config import ColorConfig

_DEFAULTS = ColorConfig()


def _is_majority_transparent(
    opaque_count: int,
    total_count: int,
    majority_fraction: float,
) -> bool:
    """Cell is transparent if at least ``majority_fraction`` of pixels are transparent."""
    return opaque_count <= total_count * (1 - majority_fraction)


def get_opaque_cell_color(cell_pixels: np.ndarray) -> RGBA:
    """
    cell_pixels: shape (height_cell, width_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in the cell_pixels block, with 255 in fourth entry for opaque alpha.
    """
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    cell_color = Counter(flat).most_common(1)[0][0]
    return (*cell_color, 255)


def get_cell_color_with_alpha(
    cell_pixels: np.ndarray,
    cell_alpha: np.ndarray,
    alpha_threshold: int = _DEFAULTS.alpha_threshold,
    majority_fraction: float = _DEFAULTS.transparency_majority_fraction,
) -> RGBA:
    """
    Select a representative color for a quantized cell, honoring transparency.

    If at least ``majority_fraction`` of the cell's pixels are transparent
    (alpha < alpha_threshold), the cell becomes fully transparent (0,0,0,0);
    otherwise it takes the most common RGB color at full opacity (R,G,B,255).

    Args:
        cell_pixels: shape (height, width, 3), dtype=uint8 (RGB from quantized image).
        cell_alpha: shape (height, width), dtype=uint8 (alpha from the original image).
    """
    total_pixels = cell_alpha.size
    opaque_count = np.sum(cell_alpha >= alpha_threshold)

    # If enough of the cell is transparent (see majority_fraction), return fully transparent
    if _is_majority_transparent(opaque_count, total_pixels, majority_fraction):
        return (0, 0, 0, 0)

    # Otherwise return most common color with full opacity
    cell_color = get_opaque_cell_color(cell_pixels)
    return cell_color


def get_cell_color_skip_quantization(
    cell_pixels: np.ndarray,
    alpha_threshold: int = _DEFAULTS.alpha_threshold,
    majority_fraction: float = _DEFAULTS.transparency_majority_fraction,
    bin_size: int = _DEFAULTS.dominant.bin_size,
) -> RGBA:
    """
    Select a representative RGBA color for a cell when quantization is skipped.

    Preserves original colors while suppressing noise/grain and background
    bleed-in. If at least ``majority_fraction`` of the cell is transparent
    (alpha < alpha_threshold) the cell becomes fully transparent (0,0,0,0);
    otherwise the dominant color of the opaque pixels (via offset binning) is
    returned at full opacity.

    Args:
        cell_pixels: shape (height, width, 4), dtype=uint8 (RGBA).
        alpha_threshold: minimum alpha for a pixel to count as opaque.
    """
    pixels = cell_pixels.reshape(-1, 4)
    total_pixels = len(pixels)

    # Edge case: empty cell
    if total_pixels == 0:
        return (0, 0, 0, 0)

    # Filter to opaque pixels only
    opaque_mask = pixels[:, 3] >= alpha_threshold
    opaque_pixels = pixels[opaque_mask]

    # If enough of the cell is transparent (see majority_fraction), return fully transparent
    if _is_majority_transparent(len(opaque_pixels), total_pixels, majority_fraction):
        return (0, 0, 0, 0)

    # Get RGB of opaque pixels and find dominant color
    rgb_pixels = opaque_pixels[:, :3]
    r, g, b = dominant_rgb_by_binning(rgb_pixels, bin_size=bin_size)
    return (int(r), int(g), int(b), 255)
