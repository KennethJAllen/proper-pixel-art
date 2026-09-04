"""Tests for the ``--diagnose`` image analysis helpers.

All inputs are synthetic and seeded, so noise levels and color counts are
reproducible without committing binary assets.
"""

import numpy as np
import pytest
from PIL import Image

from proper_pixel_art.colors import ALPHA_THRESHOLD
from proper_pixel_art.diagnostics import (
    build_recommendations,
    diagnose_image,
    format_diagnostics,
)


def _rgb_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def test_noise_high_for_random_noise() -> None:
    """Uniform random pixels read as high noise at full resolution."""
    rng = np.random.default_rng(0)
    image = _rgb_image(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))

    diagnostics = diagnose_image(image)

    assert diagnostics.noise_level == "high"
    assert any(
        "High-frequency variation" in recommendation
        for recommendation in build_recommendations(diagnostics)
    )


def test_noise_low_for_clean_pixel_art() -> None:
    """Nearest-neighbor upscaled pixel art is flat inside each block."""
    rng = np.random.default_rng(1)
    cells = _rgb_image(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    image = cells.resize((512, 512), Image.Resampling.NEAREST)

    assert diagnose_image(image).noise_level == "low"


@pytest.mark.parametrize("pixel_width", [2, 3, 4, 6])
def test_noise_low_for_small_pixel_widths(pixel_width: int) -> None:
    """Clean art reads low even when its pixels are only a few image pixels
    wide, where boundary transitions dominate the neighbor-pair population."""
    rng = np.random.default_rng(1)
    cells = _rgb_image(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    size = 32 * pixel_width
    image = cells.resize((size, size), Image.Resampling.NEAREST)

    assert diagnose_image(image).noise_level == "low"


def test_noise_moderate_for_mild_grain() -> None:
    """Flat gray with sigma ~= 12 grain lands between the low and high bands."""
    rng = np.random.default_rng(1234)
    grain = np.clip(128 + rng.normal(0, 12, (256, 256)), 0, 255)
    image = _rgb_image(np.repeat(grain[:, :, None], 3, axis=2))

    diagnostics = diagnose_image(image)

    assert diagnostics.noise_level == "moderate"
    assert any(
        "Moderate image variation" in recommendation
        for recommendation in build_recommendations(diagnostics)
    )


def test_noise_moderate_for_grainy_pixel_art() -> None:
    """Upscaled art with grain on top -- the AI-generated failure mode
    --diagnose exists for -- is flagged rather than passed as clean."""
    rng = np.random.default_rng(1)
    cells = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    upscaled = np.asarray(
        _rgb_image(cells).resize((256, 256), Image.Resampling.NEAREST),
        dtype=np.float64,
    )
    # Grain is shared across channels; independent per-channel noise partly
    # averages out in the luma the estimate is computed on.
    grain = rng.normal(0, 12, upscaled.shape[:2])
    noisy = np.clip(upscaled + grain[:, :, None], 0, 255)

    assert diagnose_image(_rgb_image(noisy)).noise_level != "low"


@pytest.mark.parametrize("coverage", [1.0, 0.36, 0.25, 0.04])
def test_noise_moderate_for_localized_grain(coverage: float) -> None:
    """Grain confined to part of an opaque frame reads the same at every
    coverage. Averaging over the whole image instead dilutes a small patch
    below the band, which is the common AI-generated layout: a grainy
    subject on a flat, opaque background."""
    rng = np.random.default_rng(7)
    canvas = np.full((256, 256), 128.0)
    side = int(round(256 * np.sqrt(coverage)))
    canvas[:side, :side] = np.clip(
        canvas[:side, :side] + rng.normal(0, 20, (side, side)), 0, 255
    )

    image = _rgb_image(np.repeat(canvas[:, :, None], 3, axis=2))

    assert diagnose_image(image).noise_level == "moderate"


@pytest.mark.parametrize("coverage", [1.0, 0.10])
def test_noise_high_for_localized_uniform_noise(coverage: float) -> None:
    """A uniform-noise patch reads high even at a tenth of the frame."""
    rng = np.random.default_rng(3)
    canvas = np.full((256, 256, 3), 128, dtype=np.uint8)
    side = int(round(256 * np.sqrt(coverage)))
    canvas[:side, :side] = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)

    assert diagnose_image(_rgb_image(canvas)).noise_level == "high"


def test_noise_detects_grain_along_an_edge() -> None:
    """A grain band too thin for a whole tile is padded into one rather than
    truncated away by the regional tiling."""
    rng = np.random.default_rng(9)
    canvas = np.full((200, 200), 128.0)
    canvas[160:, :] = np.clip(canvas[160:, :] + rng.normal(0, 20, (40, 200)), 0, 255)

    image = _rgb_image(np.repeat(canvas[:, :, None], 3, axis=2))

    assert diagnose_image(image).noise_level == "moderate"


def test_noise_high_for_image_smaller_than_a_region() -> None:
    """Images too small to hold a scoring tile fall back to whole-image
    statistics rather than reporting no noise."""
    rng = np.random.default_rng(11)
    image = _rgb_image(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8))

    assert diagnose_image(image).noise_level == "high"


def test_transparent_pixels_excluded() -> None:
    """A fully transparent image yields no colors, no noise, and no advice."""
    rng = np.random.default_rng(2)
    rgba = np.empty((64, 64, 4), dtype=np.uint8)
    rgba[..., :3] = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    rgba[..., 3] = 0

    diagnostics = diagnose_image(Image.fromarray(rgba, mode="RGBA"))

    assert diagnostics.noise_level == "low"
    assert diagnostics.color_count == 0
    assert diagnostics.transparent_pixels == 64 * 64
    assert diagnostics.semi_transparent_pixels == 0
    assert build_recommendations(diagnostics) == ("No obvious cleanup recommendation.",)


@pytest.mark.parametrize(
    ("num_colors", "expect_palette_recommendation"), [(4, False), (40, True)]
)
def test_color_count_ignores_transparent_colors(
    num_colors: int, expect_palette_recommendation: bool
) -> None:
    """Colors hidden under alpha == 0 do not inflate the count."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    # Opaque band: one distinct color per row, cycling through the palette.
    palette = np.arange(num_colors, dtype=np.uint8)
    rgba[:48, :, 0] = palette[np.arange(48) % num_colors][:, None]
    rgba[:48, :, 3] = 255
    # Transparent band, holding a color found nowhere in the opaque band.
    rgba[48:, :, :3] = 200

    diagnostics = diagnose_image(Image.fromarray(rgba, mode="RGBA"))

    assert diagnostics.color_count == num_colors
    has_palette_recommendation = any(
        "--colors 16" in recommendation
        for recommendation in build_recommendations(diagnostics)
    )
    assert has_palette_recommendation is expect_palette_recommendation


def test_color_count_uses_pipeline_alpha_threshold() -> None:
    """Colors below colors.ALPHA_THRESHOLD are rendered fully transparent by
    the pipeline, so they do not count as content."""
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    # A second color, hidden below the pipeline's opacity threshold.
    rgba[0, :, :3] = 200
    rgba[0, :, 3] = ALPHA_THRESHOLD - 1

    assert diagnose_image(Image.fromarray(rgba, mode="RGBA")).color_count == 1


def test_alpha_counts() -> None:
    """Alpha 0 counts as transparent; anything in (0, 255) as semi."""
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = np.array([[0, 128], [255, 7]], dtype=np.uint8)

    diagnostics = diagnose_image(Image.fromarray(rgba, mode="RGBA"))

    assert diagnostics.transparent_pixels == 1
    assert diagnostics.semi_transparent_pixels == 2
    assert any(
        "Semi-transparent pixels detected" in recommendation
        for recommendation in build_recommendations(diagnostics)
    )


def test_format_diagnostics_output() -> None:
    """The report includes the size, noise level, and bulleted advice."""
    image = _rgb_image(np.zeros((8, 4, 3), dtype=np.uint8))

    report = format_diagnostics(diagnose_image(image))

    assert "Size: 4x8" in report
    assert "Noise level: low" in report
    assert "\n- " in report
