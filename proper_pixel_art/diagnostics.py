"""Diagnostics for source images prior to pixel-art cleanup.

Reports size, opaque color count, transparency stats, and a full-resolution
noise estimate, with actionable CLI recommendations.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Neighbor difference (0-255 grayscale) counted as a "strong" transition.
_STRONG_TRANSITION = 48.0

# Noise bands, calibrated against full-resolution neighbor differences:
# clean pixel art and the repository's source assets land under the moderate
# cut, mild grain (sigma ~= 12) reads moderate, and uniform noise reads high.
_HIGH_MEAN, _HIGH_RATIO = 24.0, 0.25
_MODERATE_MEAN, _MODERATE_RATIO = 10.0, 0.08

# Rec. 601 luma weights, matching PIL's "L" conversion.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class ImageDiagnostics:
    """Summary of image characteristics relevant to pixel-art cleanup."""

    width: int
    height: int
    color_count: int
    transparent_pixels: int
    semi_transparent_pixels: int
    noise_level: str
    recommendations: tuple[str, ...]


def _count_opaque_colors(rgba: np.ndarray, opaque: np.ndarray) -> int:
    """Count distinct RGB values among pixels that carry any opacity."""
    rgb = rgba[..., :3][opaque].astype(np.uint32)

    if rgb.size == 0:
        return 0

    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    # Sort and count the value changes rather than calling np.unique, which is
    # two orders of magnitude slower on megapixel inputs with many colors.
    ordered = np.sort(packed)

    return 1 + int(np.count_nonzero(np.diff(ordered)))


def _estimate_noise_level(rgba: np.ndarray, opaque: np.ndarray) -> str:
    """Estimate high-frequency variation in an image.

    This is a lightweight heuristic, not a machine-learning classifier. It
    compares neighboring grayscale pixels at full resolution; downscaling
    first would low-pass away the very signal being measured.
    """
    # Cast to float before differencing: uint8 subtraction wraps around.
    gray = rgba[..., :3].astype(np.float32) @ _LUMA_WEIGHTS

    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))

    # A pair counts only when both pixels are opaque. Masking (rather than
    # compositing against a background) avoids counting the silhouette edge
    # of a transparent cutout as image noise.
    horizontal_valid = opaque[:, 1:] & opaque[:, :-1]
    vertical_valid = opaque[1:, :] & opaque[:-1, :]

    differences = np.concatenate(
        [horizontal[horizontal_valid], vertical[vertical_valid]]
    )

    if differences.size == 0:
        return "low"

    average_difference = float(differences.mean())
    strong_transition_ratio = float((differences >= _STRONG_TRANSITION).mean())

    if average_difference >= _HIGH_MEAN or strong_transition_ratio >= _HIGH_RATIO:
        return "high"

    if (
        average_difference >= _MODERATE_MEAN
        or strong_transition_ratio >= _MODERATE_RATIO
    ):
        return "moderate"

    return "low"


def diagnose_image(image: Image.Image) -> ImageDiagnostics:
    """Analyze an image and return pixel-art cleanup diagnostics."""
    rgba = np.asarray(image.convert("RGBA"))

    height, width = rgba.shape[:2]

    alpha = rgba[..., 3]

    # Fully transparent pixels carry no visible color, so they are excluded
    # from both the color count and the noise estimate. Semi-transparent
    # pixels are still content and count toward both.
    opaque = alpha > 0

    color_count = _count_opaque_colors(rgba, opaque)

    transparent_pixels = int(np.count_nonzero(alpha == 0))

    semi_transparent_pixels = int(np.count_nonzero((alpha > 0) & (alpha < 255)))

    noise_level = _estimate_noise_level(rgba, opaque)

    recommendations: list[str] = []

    if color_count > 32:
        recommendations.append("Try using --colors 16 or another smaller palette.")

    if semi_transparent_pixels > 0:
        recommendations.append(
            "Semi-transparent pixels detected; "
            "review transparency handling in the output."
        )

    if noise_level == "high":
        recommendations.append(
            "High-frequency variation detected; "
            "review the generated mesh and try --intermediate-dir."
        )
    elif noise_level == "moderate":
        recommendations.append(
            "Moderate image variation detected; try a few different --colors values."
        )

    if not recommendations:
        recommendations.append("No obvious cleanup recommendation.")

    return ImageDiagnostics(
        width=width,
        height=height,
        color_count=color_count,
        transparent_pixels=transparent_pixels,
        semi_transparent_pixels=semi_transparent_pixels,
        noise_level=noise_level,
        recommendations=tuple(recommendations),
    )


def format_diagnostics(
    diagnostics: ImageDiagnostics,
) -> str:
    """Format diagnostics as readable terminal output."""
    recommendations = "\n".join(
        f"- {recommendation}" for recommendation in diagnostics.recommendations
    )

    return f"""Image diagnosis
---------------
Size: {diagnostics.width}x{diagnostics.height}
Estimated colors: {diagnostics.color_count}
Transparent pixels: {diagnostics.transparent_pixels}
Semi-transparent pixels: {diagnostics.semi_transparent_pixels}
Noise level: {diagnostics.noise_level}

Recommendations:
{recommendations}
"""
