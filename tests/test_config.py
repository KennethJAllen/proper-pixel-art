"""Tests for the YAML/dataclass config and its integration with pixelate."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Dither, Quantize

from proper_pixel_art import pixelate
from proper_pixel_art.config import (
    ColorConfig,
    DominantConfig,
    MeshConfig,
    PaletteConfig,
    PixelateConfig,
    VideoConfig,
)

EXAMPLE_CONFIG = Path(__file__).parent.parent / "config.example.yaml"


def test_defaults_match_historical_values():
    """The dataclass defaults must reproduce the previously hardcoded values."""
    cfg = PixelateConfig()
    assert cfg.initial_upscale_factor == 2
    assert cfg.mesh.canny_thresholds == (50, 200)
    assert cfg.mesh.closure_kernel_size == 8
    assert cfg.mesh.cluster_threshold == 4
    assert cfg.mesh.crop_border_pixels == 2
    assert cfg.mesh.trim_outlier_fraction == 0.2
    assert cfg.mesh.hough.threshold == 100
    assert cfg.mesh.hough.min_line_len == 50
    assert cfg.colors.method == "dominant"
    assert cfg.colors.alpha_threshold == 128
    assert cfg.colors.transparency_majority_fraction == 0.5
    assert cfg.colors.palette.num_colors == 16
    assert cfg.colors.palette.quantize == Quantize.MAXCOVERAGE
    assert cfg.colors.palette.dither_mode == Dither.NONE
    assert cfg.colors.palette.kmeans == 0
    assert cfg.colors.dominant.bin_size == 52
    assert cfg.colors.dominant.merge_distance == 12
    assert cfg.colors.dominant.merge_linkage == "complete"
    assert cfg.colors.dominant.max_linkage_colors == 4096


def test_example_yaml_matches_defaults():
    """config.example.yaml must document exactly the built-in defaults."""
    assert PixelateConfig.from_yaml(EXAMPLE_CONFIG) == PixelateConfig()


def test_from_dict_partial_deep_merge():
    """A partial config overrides only the given keys, deep-merging nested groups."""
    cfg = PixelateConfig.from_dict(
        {
            "mesh": {"canny_thresholds": [10, 90], "hough": {"threshold": 42}},
            "colors": {
                "method": "palette",
                "palette": {"num_colors": 16, "quantize_method": "FASTOCTREE"},
                "dominant": {"merge_distance": 6},
            },
        }
    )
    # Overridden values
    assert cfg.mesh.canny_thresholds == (10, 90)  # list coerced to tuple
    assert cfg.mesh.hough.threshold == 42
    assert cfg.colors.method == "palette"
    assert cfg.colors.palette.num_colors == 16
    assert cfg.colors.palette.quantize == Quantize.FASTOCTREE
    assert cfg.colors.dominant.merge_distance == 6
    # Untouched values keep their defaults
    assert cfg.mesh.closure_kernel_size == 8
    assert cfg.mesh.hough.min_line_len == 50
    assert cfg.colors.alpha_threshold == 128
    assert cfg.colors.palette.kmeans == 0
    assert cfg.colors.dominant.bin_size == 52


def test_width_replace_tolerance_config():
    """width_replace_tolerance defaults to 0.2 and round-trips from a dict."""
    assert MeshConfig().width_replace_tolerance == 0.2
    cfg = PixelateConfig.from_dict({"mesh": {"width_replace_tolerance": 0.5}})
    assert cfg.mesh.width_replace_tolerance == 0.5


def test_from_dict_does_not_mutate_input():
    """from_dict must not mutate the caller's (nested) input dict."""
    data = {"mesh": {"hough": {"threshold": 42}}, "colors": {"palette": {"kmeans": 1}}}
    PixelateConfig.from_dict(data)
    assert data == {
        "mesh": {"hough": {"threshold": 42}},
        "colors": {"palette": {"kmeans": 1}},
    }


def test_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "colors:\n"
        "  method: palette\n"
        "  palette:\n"
        "    num_colors: 8\n"
        "mesh:\n"
        "  closure_kernel_size: 12\n"
        "  hough:\n"
        "    rho: 2.0\n"
    )
    cfg = PixelateConfig.from_yaml(config_path)
    assert cfg.colors.method == "palette"
    assert cfg.colors.palette.num_colors == 8
    assert cfg.mesh.closure_kernel_size == 12
    assert cfg.mesh.hough.rho == 2.0


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"not_a_real_key": 1})
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"mesh": {"bogus": 1}})
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"colors": {"palette": {"bogus": 1}}})
    with pytest.raises(ValueError, match="Unknown config key"):
        PixelateConfig.from_dict({"colors": {"dominant": {"bogus": 1}}})


def test_old_flat_keys_raise_with_changelog_hint():
    """Pre-2.0 keys are gone: they fail with a pointer at the CHANGELOG."""
    for data in (
        {"num_colors": 8},
        {"colors": {"quantize_method": "FASTOCTREE"}},
        {"colors": {"bin_size": 30}},
        {"colors": {"output_color_merge_distance": 6}},
    ):
        with pytest.raises(ValueError, match="CHANGELOG"):
            PixelateConfig.from_dict(data)


def test_version_key_accepted():
    """An explicit version: 2 is accepted and stripped."""
    assert PixelateConfig.from_dict({"version": 2}) == PixelateConfig()


def test_wrong_version_raises():
    with pytest.raises(ValueError, match="Unsupported config version"):
        PixelateConfig.from_dict({"version": 1})


def test_zero_sentinels_raise():
    """The pre-2.0 numeric sentinels fail loudly with a CHANGELOG pointer."""
    with pytest.raises(ValueError, match="CHANGELOG"):
        PixelateConfig(pixel_width=0)
    with pytest.raises(ValueError, match="CHANGELOG"):
        PixelateConfig(scale_result=0)


def test_none_sentinels_are_defaults():
    cfg = PixelateConfig()
    assert cfg.scale_result is None
    assert cfg.pixel_width is None


def test_video_config_defaults_and_validation():
    cfg = PixelateConfig.from_dict({"video": {"output_format": "mp4"}})
    assert cfg.video.output_format == "mp4"
    assert cfg.video.num_sample_frames == 8
    assert cfg.video.min_vote_fraction == 0.25
    with pytest.raises(ValueError, match="Unknown output_format"):
        VideoConfig(output_format="webm")
    with pytest.raises(ValueError, match="num_sample_frames"):
        VideoConfig(num_sample_frames=0)
    with pytest.raises(ValueError, match="min_vote_fraction"):
        VideoConfig(min_vote_fraction=0)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown color method"):
        ColorConfig(method="quantize")


def test_unknown_quantize_method_raises():
    with pytest.raises(ValueError, match="Unknown quantize_method"):
        _ = PaletteConfig(quantize_method="NOPE").quantize


def test_unknown_dither_raises():
    with pytest.raises(ValueError, match="Unknown dither"):
        _ = PaletteConfig(dither="NOPE").dither_mode


def test_num_colors_out_of_range_raises():
    with pytest.raises(ValueError, match="num_colors must be in 1-256"):
        PaletteConfig(num_colors=0)
    with pytest.raises(ValueError, match="num_colors must be in 1-256"):
        PaletteConfig(num_colors=257)


def test_negative_kmeans_raises():
    with pytest.raises(ValueError, match="kmeans must be >= 0"):
        PaletteConfig(kmeans=-1)


def test_bin_size_zero_raises():
    with pytest.raises(ValueError, match="bin_size must be >= 1"):
        DominantConfig(bin_size=0)


def test_unknown_merge_linkage_raises():
    with pytest.raises(ValueError, match="Unknown merge_linkage"):
        DominantConfig(merge_linkage="ward")


def test_max_linkage_colors_too_small_raises():
    with pytest.raises(ValueError, match="max_linkage_colors must be >= 2"):
        DominantConfig(max_linkage_colors=1)


def test_background_candidates_coerced_to_tuples():
    cfg = ColorConfig(background_candidates=[[0, 255, 255]])
    assert cfg.background_candidates == [(0, 255, 255)]


def test_theta_deg_to_radians():
    assert MeshConfig().hough.theta_rad == pytest.approx(np.deg2rad(1.0))


def test_pixelate_default_config_identity(assets):
    """pixelate(img) must be pixel-identical to pixelate(img, config=PixelateConfig())."""
    img = Image.open(assets / "blob" / "blob.png")
    default_result = pixelate(img)
    config_result = pixelate(img, config=PixelateConfig())
    assert np.array_equal(np.array(default_result.image), np.array(config_result.image))
    assert default_result.pixel_width == config_result.pixel_width
    assert default_result.mesh == config_result.mesh


def test_pixelate_from_dict_matches_dataclass_config(assets):
    """A from_dict-built config behaves identically to the dataclass form."""
    img = Image.open(assets / "blob" / "blob.png")
    via_dict = pixelate(
        img,
        config=PixelateConfig.from_dict(
            {"colors": {"method": "palette", "palette": {"num_colors": 16}}}
        ),
    )
    via_dataclass = pixelate(
        img,
        config=PixelateConfig(
            colors=ColorConfig(method="palette", palette=PaletteConfig(num_colors=16))
        ),
    )
    assert np.array_equal(np.array(via_dict.image), np.array(via_dataclass.image))
