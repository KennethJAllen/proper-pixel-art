"""Tests for the YAML/dataclass config and its integration with pixelate."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Quantize

from proper_pixel_art import pixelate
from proper_pixel_art.config import ColorConfig, MeshConfig, PixelateConfig

EXAMPLE_CONFIG = Path(__file__).parent.parent / "config.example.yaml"


def test_defaults_match_historical_values():
    """The dataclass defaults must reproduce the previously hardcoded values."""
    cfg = PixelateConfig()
    assert cfg.num_colors == 0
    assert cfg.initial_upscale_factor == 2
    assert cfg.mesh.canny_thresholds == (50, 200)
    assert cfg.mesh.closure_kernel_size == 8
    assert cfg.mesh.cluster_threshold == 4
    assert cfg.mesh.crop_border_pixels == 2
    assert cfg.mesh.trim_outlier_fraction == 0.2
    assert cfg.mesh.hough.threshold == 100
    assert cfg.mesh.hough.min_line_len == 50
    assert cfg.colors.alpha_threshold == 128
    assert cfg.colors.transparency_majority_fraction == 0.5
    assert cfg.colors.bin_size == 52
    assert cfg.colors.quantize == Quantize.MAXCOVERAGE


def test_example_yaml_matches_defaults():
    """config.example.yaml must document exactly the built-in defaults."""
    assert PixelateConfig.from_yaml(EXAMPLE_CONFIG) == PixelateConfig()


def test_from_dict_partial_deep_merge():
    """A partial config overrides only the given keys, deep-merging nested groups."""
    cfg = PixelateConfig.from_dict(
        {
            "num_colors": 16,
            "mesh": {"canny_thresholds": [10, 90], "hough": {"threshold": 42}},
            "colors": {"quantize_method": "FASTOCTREE"},
        }
    )
    # Overridden values
    assert cfg.num_colors == 16
    assert cfg.mesh.canny_thresholds == (10, 90)  # list coerced to tuple
    assert cfg.mesh.hough.threshold == 42
    assert cfg.colors.quantize == Quantize.FASTOCTREE
    # Untouched values keep their defaults
    assert cfg.mesh.closure_kernel_size == 8
    assert cfg.mesh.hough.min_line_len == 50
    assert cfg.colors.alpha_threshold == 128


def test_from_dict_does_not_mutate_input():
    """from_dict must not mutate the caller's (nested) input dict."""
    data = {"mesh": {"hough": {"threshold": 42}}}
    PixelateConfig.from_dict(data)
    assert data == {"mesh": {"hough": {"threshold": 42}}}


def test_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "num_colors: 8\nmesh:\n  closure_kernel_size: 12\n  hough:\n    rho: 2.0\n"
    )
    cfg = PixelateConfig.from_yaml(config_path)
    assert cfg.num_colors == 8
    assert cfg.mesh.closure_kernel_size == 12
    assert cfg.mesh.hough.rho == 2.0


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"not_a_real_key": 1})
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"mesh": {"bogus": 1}})


def test_unknown_quantize_method_raises():
    with pytest.raises(ValueError, match="Unknown quantize_method"):
        ColorConfig(quantize_method="NOPE").quantize


def test_bin_size_zero_raises():
    with pytest.raises(ValueError, match="bin_size must be >= 1"):
        ColorConfig(bin_size=0)


def test_background_candidates_coerced_to_tuples():
    cfg = ColorConfig(background_candidates=[[0, 255, 255]])
    assert cfg.background_candidates == [(0, 255, 255)]


def test_theta_deg_to_radians():
    assert MeshConfig().hough.theta_rad == pytest.approx(np.deg2rad(1.0))


def test_pixelate_default_config_identity(assets):
    """pixelate(img) must be pixel-identical to pixelate(img, config=PixelateConfig())."""
    img = Image.open(assets / "blob" / "blob.png")
    default_result = pixelate(img, num_colors=16)
    config_result = pixelate(img, config=PixelateConfig(num_colors=16))
    assert np.array_equal(np.array(default_result), np.array(config_result))


def test_explicit_arg_overrides_config(assets):
    """An explicit kwarg wins over the config value."""
    img = Image.open(assets / "blob" / "blob.png")
    cfg = PixelateConfig(num_colors=4)
    # Explicit num_colors=16 should override the config's 4.
    via_arg = pixelate(img, num_colors=16, config=cfg)
    via_plain = pixelate(img, num_colors=16)
    assert np.array_equal(np.array(via_arg), np.array(via_plain))


def test_explicit_zero_skips_over_config(assets):
    """num_colors=0 overrides a quantizing config, taking the skip path."""
    img = Image.open(assets / "blob" / "blob.png")
    cfg = PixelateConfig(num_colors=16)
    via_zero = pixelate(img, num_colors=0, config=cfg)
    via_skip = pixelate(img, num_colors=0)
    assert np.array_equal(np.array(via_zero), np.array(via_skip))
