"""Characterize the current pixel-width behavior, not the documented contract."""

import numpy as np
from PIL import Image

from proper_pixel_art import pixelate


def test_pixel_width_current_upscaled_coordinates_contradict_docstring() -> None:
    """Lock in the current bug: pixel_width uses upscaled image coordinates."""
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(cells).resize((512, 512), Image.Resampling.NEAREST)

    # Characterization only: the docstring describes input-image pixel width.
    # After the fix, pixel_width=16 should yield (32, 32) at every upscale
    # factor: the (2, 16) and (4, 16) expectations below must change.
    # Widths 32 and 64 currently compensate for the bug, not the input cells.
    cases = [
        (1, 16, (32, 32)),
        (2, 16, (64, 64)),
        (2, 32, (32, 32)),
        (4, 16, (128, 128)),
        (4, 64, (32, 32)),
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
