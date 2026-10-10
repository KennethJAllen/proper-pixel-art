"""Compatibility import path for :mod:`proper_pixel_art.image`.

The image implementation moved in 2.0 so the ``pixelate`` function no longer
shares its name with a module. New code should import these names from the
package root or :mod:`proper_pixel_art.image`.
"""

import warnings

warnings.warn(
    "proper_pixel_art.pixelate was renamed to proper_pixel_art.image in 2.0; "
    "import pixelate and PixelateResult from proper_pixel_art instead.",
    DeprecationWarning,
    stacklevel=2,
)

from proper_pixel_art.image import (  # noqa: E402 - warn before compatibility import
    PixelateResult,
    build_cell_map,
    downsample,
    pixelate,
)

__all__ = ["pixelate", "PixelateResult", "build_cell_map", "downsample"]
