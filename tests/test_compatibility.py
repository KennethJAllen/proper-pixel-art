"""Tests for deliberately small 2.0 migration aids."""

import importlib
import sys

import pytest


def test_legacy_pixelate_module_reexports_image_api():
    sys.modules.pop("proper_pixel_art.pixelate", None)
    with pytest.warns(DeprecationWarning, match="renamed to proper_pixel_art.image"):
        legacy = importlib.import_module("proper_pixel_art.pixelate")

    current = importlib.import_module("proper_pixel_art.image")
    assert legacy.pixelate is current.pixelate
    assert legacy.PixelateResult is current.PixelateResult
    assert legacy.build_cell_map is current.build_cell_map
    assert legacy.downsample is current.downsample
