"""Pixel width is measured in input-image pixels regardless of initial upscaling."""

import numpy as np
from PIL import Image

from proper_pixel_art import pixelate


def test_pixel_width_uses_input_pixels_independent_of_initial_upscale_factor() -> None:
    """Input-image pixel width yields the same size at every initial upscale factor."""
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(cells).resize((512, 512), Image.Resampling.NEAREST)

    # Each input cell is 16 pixels wide, independent of initial upscaling.
    cases = [
        (1, 16, (32, 32)),
        (2, 16, (32, 32)),
        (4, 16, (32, 32)),
    ]
    for initial_upscale_factor, pixel_width, expected_size in cases:
        result = pixelate(
            img,
            initial_upscale_factor=initial_upscale_factor,
            pixel_width=pixel_width,
        )
        assert result.size == expected_size, (
            f"{initial_upscale_factor=}, {pixel_width=}: {result.size=}"
        )
