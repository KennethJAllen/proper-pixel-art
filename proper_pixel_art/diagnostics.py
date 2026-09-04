"""Diagnostics for source images prior to pixel-art cleanup.

Reports size, content color count, transparency stats, and a full-resolution
noise estimate. CLI advice is derived from the measurements at formatting
time, keeping the analysis itself presentation-free.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from proper_pixel_art.colors import ALPHA_THRESHOLD

# Minimum-neighbor difference (0-255 grayscale) that marks a pixel as
# disagreeing strongly with every one of its neighbors.
_STRONG_DEVIATION = 48.0

# Whole-image noise bands, calibrated against per-pixel minimum neighbor
# differences (see _estimate_noise_level): clean pixel art scores ~0 at any
# pixel width >= 2, the repository's source assets land under the moderate
# cut, mild grain (sigma ~= 12) reads moderate, and uniform noise reads high.
_HIGH_MEAN, _HIGH_RATIO = 10.0, 0.08
_MODERATE_MEAN, _MODERATE_RATIO = 4.0, 0.02

# Regional bands, applied to the noisiest tile rather than the whole image.
# A whole-image average dilutes grain that covers only part of the frame:
# the same sigma-20 patch reads moderate at full coverage but low below
# ~40%, which is exactly the AI-generated case --diagnose exists for. Tiles
# are scored independently so the verdict depends on how noisy the worst
# region is, not on how much of the frame it happens to occupy.
_REGION_SIZE = 32

# Tiles holding less content than this are skipped: a handful of pixels in
# the corner of a transparent cutout is too small a sample to band.
_REGION_MIN_COVERAGE = 0.25

# Set above the whole-image cuts because a maximum over many tiles drifts
# upward on large images; the headroom keeps detailed-but-clean sources
# (e.g. assets/mountain/mountain.png, noisiest tile 4.93) out of the bands.
_REGION_HIGH_MEAN, _REGION_HIGH_RATIO = 15.0, 0.12
_REGION_MODERATE_MEAN, _REGION_MODERATE_RATIO = 6.0, 0.03

# Ordered weakest to strongest; the reported level is the strongest band any
# statistic reaches.
_NOISE_LEVELS = ("low", "moderate", "high")


@dataclass(frozen=True)
class ImageDiagnostics:
    """Summary of image characteristics relevant to pixel-art cleanup."""

    width: int
    height: int
    color_count: int
    transparent_pixels: int
    semi_transparent_pixels: int
    noise_level: str


def _count_content_colors(rgba: np.ndarray, content: np.ndarray) -> int:
    """Count distinct RGB values among content pixels."""
    rgb = rgba[..., :3][content].astype(np.uint32)

    if rgb.size == 0:
        return 0

    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    # Sort and count the value changes rather than calling np.unique, which is
    # two orders of magnitude slower on megapixel inputs with many colors.
    ordered = np.sort(packed)

    return 1 + int(np.count_nonzero(np.diff(ordered)))


def _neighbor_minimum(
    gray: np.ndarray, content: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel minimum absolute grayscale difference to the 4-neighbors.

    Returns the difference map alongside a mask of the pixels that had at
    least one content neighbor to compare against; unmeasured pixels hold
    infinity in the map.
    """
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))

    # A pair counts only when both pixels are content: infinity drops the
    # pair from every minimum below. Masking (rather than compositing against
    # a background) avoids counting the silhouette edge of a transparent
    # cutout as image noise.
    horizontal = np.where(content[:, 1:] & content[:, :-1], horizontal, np.inf)
    vertical = np.where(content[1:, :] & content[:-1, :], vertical, np.inf)

    minimum = np.full(gray.shape, np.inf, dtype=np.float32)
    np.minimum(minimum[:, 1:], horizontal, out=minimum[:, 1:])
    np.minimum(minimum[:, :-1], horizontal, out=minimum[:, :-1])
    np.minimum(minimum[1:, :], vertical, out=minimum[1:, :])
    np.minimum(minimum[:-1, :], vertical, out=minimum[:-1, :])

    return minimum, np.isfinite(minimum)


def _tile_sums(values: np.ndarray) -> np.ndarray:
    """Sum ``values`` over a grid of _REGION_SIZE-square tiles.

    Zero-padded to whole tiles so a grain band along the right or bottom
    edge still lands in a tile instead of being truncated away.
    """
    height, width = values.shape
    padded = np.pad(
        values.astype(np.float64, copy=False),
        ((0, -height % _REGION_SIZE), (0, -width % _REGION_SIZE)),
    )

    return padded.reshape(
        padded.shape[0] // _REGION_SIZE,
        _REGION_SIZE,
        padded.shape[1] // _REGION_SIZE,
        _REGION_SIZE,
    ).sum(axis=(1, 3))


def _noisiest_region(
    minimum: np.ndarray, measured: np.ndarray, strong: np.ndarray
) -> tuple[float, float]:
    """Return the highest per-tile mean deviation and strong-deviation ratio.

    Both are zero when no tile holds enough content to score, which is the
    case for images smaller than a tile; the whole-image statistics cover
    those on their own.
    """
    counts = _tile_sums(measured)
    populated = counts >= _REGION_MIN_COVERAGE * _REGION_SIZE**2

    if not populated.any():
        return 0.0, 0.0

    counts = counts[populated]
    totals = _tile_sums(np.where(measured, minimum, 0.0))[populated]

    return (
        float((totals / counts).max()),
        float((_tile_sums(strong)[populated] / counts).max()),
    )


def _band(
    mean_deviation: float,
    strong_ratio: float,
    high: tuple[float, float],
    moderate: tuple[float, float],
) -> int:
    """Index into _NOISE_LEVELS for one pair of statistics."""
    if mean_deviation >= high[0] or strong_ratio >= high[1]:
        return 2

    if mean_deviation >= moderate[0] or strong_ratio >= moderate[1]:
        return 1

    return 0


def _estimate_noise_level(gray: np.ndarray, content: np.ndarray) -> str:
    """Estimate high-frequency variation in an image.

    Measures, for every content pixel, the minimum absolute grayscale
    difference to its 4-neighbors. A pixel inside a flat run has at least one
    identical neighbor and scores zero, so clean pixel art reads "low" at any
    pixel width >= 2 no matter how small the upscale, while grain and noise
    keep every pixel away from all of its neighbors. An image already at
    native 1:1 pixel resolution has no flat runs and is indistinguishable
    from noise here, but such an image needs no pixelation to begin with.

    Those differences are banded twice: once averaged over the whole image,
    and once over the noisiest tile, which is what catches grain confined to
    part of an opaque frame. The stronger of the two verdicts wins.
    """
    minimum, measured = _neighbor_minimum(gray, content)
    measured_count = int(np.count_nonzero(measured))

    if measured_count == 0:
        return "low"

    strong = measured & (minimum >= _STRONG_DEVIATION)

    average_deviation = (
        float(minimum.sum(where=measured, dtype=np.float64)) / measured_count
    )
    strong_ratio = int(np.count_nonzero(strong)) / measured_count

    region_deviation, region_strong_ratio = _noisiest_region(minimum, measured, strong)

    level = max(
        _band(
            average_deviation,
            strong_ratio,
            (_HIGH_MEAN, _HIGH_RATIO),
            (_MODERATE_MEAN, _MODERATE_RATIO),
        ),
        _band(
            region_deviation,
            region_strong_ratio,
            (_REGION_HIGH_MEAN, _REGION_HIGH_RATIO),
            (_REGION_MODERATE_MEAN, _REGION_MODERATE_RATIO),
        ),
    )

    return _NOISE_LEVELS[level]


def diagnose_image(image: Image.Image) -> ImageDiagnostics:
    """Analyze an image and return pixel-art cleanup diagnostics."""
    rgba_image = image.convert("RGBA")
    rgba = np.asarray(rgba_image)

    # PIL's "L" mode applies the Rec. 601 luma weights a manual matmul would,
    # several times faster and without full-image float temporaries.
    gray = np.asarray(rgba_image.convert("L"), dtype=np.float32)

    height, width = rgba.shape[:2]

    alpha = rgba[..., 3]

    # Content pixels are the ones pixelation will keep: at or above the same
    # alpha threshold the pipeline uses. Anything below it is rendered fully
    # transparent downstream, so it is excluded from both the color count and
    # the noise estimate.
    content = alpha >= ALPHA_THRESHOLD

    transparent_pixels = int(np.count_nonzero(alpha == 0))
    fully_opaque_pixels = int(np.count_nonzero(alpha == 255))
    semi_transparent_pixels = alpha.size - transparent_pixels - fully_opaque_pixels

    return ImageDiagnostics(
        width=width,
        height=height,
        color_count=_count_content_colors(rgba, content),
        transparent_pixels=transparent_pixels,
        semi_transparent_pixels=semi_transparent_pixels,
        noise_level=_estimate_noise_level(gray, content),
    )


# Palettes larger than this usually carry stray colors worth quantizing away.
_MANY_COLORS = 32


def build_recommendations(diagnostics: ImageDiagnostics) -> tuple[str, ...]:
    """Derive CLI advice from measured diagnostics."""
    recommendations: list[str] = []

    if diagnostics.color_count > _MANY_COLORS:
        recommendations.append("Try using --colors 16 or another smaller palette.")

    if diagnostics.semi_transparent_pixels > 0:
        recommendations.append(
            "Semi-transparent pixels detected; "
            "review transparency handling in the output."
        )

    if diagnostics.noise_level == "high":
        recommendations.append(
            "High-frequency variation detected; "
            "review the generated mesh and try --intermediate-dir."
        )
    elif diagnostics.noise_level == "moderate":
        recommendations.append(
            "Moderate image variation detected; try a few different --colors values."
        )

    if not recommendations:
        recommendations.append("No obvious cleanup recommendation.")

    return tuple(recommendations)


def format_diagnostics(
    diagnostics: ImageDiagnostics,
) -> str:
    """Format diagnostics as readable terminal output."""
    recommendations = "\n".join(
        f"- {recommendation}" for recommendation in build_recommendations(diagnostics)
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
