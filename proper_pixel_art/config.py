"""Configuration for the pixelate algorithm.

All tunable parameters live here as dataclasses, grouped by algorithm stage.
``PixelateConfig()`` holds the defaults; override individual fields directly or
load a (possibly partial) config from YAML with :meth:`PixelateConfig.from_yaml`.
Any key omitted from the YAML falls back to the default.
"""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import yaml
from PIL.Image import Quantize

RGB = tuple[int, int, int]


@dataclass
class HoughConfig:
    """Parameters for the probabilistic Hough line transform (grid detection)."""

    rho: float = 1.0  # accumulator distance resolution in pixels
    theta_deg: float = 1.0  # accumulator angle resolution in degrees
    threshold: int = 100  # minimum votes to accept a line
    min_line_len: int = 50  # minimum line length in pixels
    max_line_gap: int = 10  # maximum gap to join line segments

    @property
    def theta_rad(self) -> float:
        """Angle resolution in radians, as OpenCV expects."""
        return float(np.deg2rad(self.theta_deg))


@dataclass
class MeshConfig:
    """Parameters controlling pixel-grid (mesh) detection."""

    crop_border_pixels: int = 2  # border pixels trimmed before edge detection
    canny_thresholds: tuple[int, int] = (50, 200)  # (lower, upper) Canny thresholds
    closure_kernel_size: int = 8  # morphological closing kernel size
    cluster_threshold: int = 4  # max distance (px) to merge nearby grid lines
    angle_threshold_deg: float = 15  # tolerance for vertical/horizontal lines
    trim_outlier_fraction: float = 0.2  # tail fraction trimmed for pixel-width estimate
    hough: HoughConfig = field(default_factory=HoughConfig)


@dataclass
class ColorConfig:
    """Parameters controlling color selection, quantization and transparency."""

    alpha_threshold: int = 128  # alpha >= this is opaque (0-255)
    transparency_majority_fraction: float = (
        0.5  # cell transparent if >= this fraction is transparent
    )
    quantize_method: str = "MAXCOVERAGE"  # PIL.Image.Quantize member name
    bin_size: int = 52  # RGB bin size for skip-quantization dominant color
    top_colors_limit: int = 8  # common colors sampled to pick a background
    thumbnail_size: tuple[int, int] = (160, 160)  # downscale size for color analysis
    background_candidates: list[RGB] | None = None  # override background palette

    def __post_init__(self) -> None:
        # bin_size is a divisor in _dominant_rgb_by_binning (num_bins =
        # 255 // bin_size + 1); 0 would raise an opaque ZeroDivisionError later.
        if self.bin_size < 1:
            raise ValueError(f"bin_size must be >= 1, got {self.bin_size}.")
        # Normalize YAML lists-of-lists into RGB tuples. _build skips this field
        # because its default is None rather than a tuple.
        if self.background_candidates is not None:
            self.background_candidates = [tuple(c) for c in self.background_candidates]

    @property
    def quantize(self) -> int:
        """Resolve ``quantize_method`` to a ``PIL.Image.Quantize`` value."""
        try:
            return getattr(Quantize, self.quantize_method)
        except AttributeError as exc:
            valid = ", ".join(q.name for q in Quantize)
            raise ValueError(
                f"Unknown quantize_method {self.quantize_method!r}. "
                f"Valid options: {valid}."
            ) from exc


@dataclass
class PixelateConfig:
    """Top-level configuration for :func:`proper_pixel_art.pixelate.pixelate`.

    Three fields use numeric sentinels rather than ``None`` for their special
    states, leaving ``None`` free to mean "not provided" in ``pixelate`` kwargs:
    ``num_colors=0`` skips quantization, ``scale_result=1`` means no scaling, and
    ``pixel_width=0`` auto-detects the pixel width.
    """

    num_colors: int = 0
    initial_upscale_factor: int = 2
    scale_result: int = 1
    transparent_background: bool = False
    pixel_width: int = 0
    mesh: MeshConfig = field(default_factory=MeshConfig)
    colors: ColorConfig = field(default_factory=ColorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PixelateConfig":
        """Load a config from a YAML file, deep-merging over the defaults."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Config file {path} must contain a mapping at the top level."
            )
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PixelateConfig":
        """Build a config from a (possibly partial, nested) dict."""
        data = dict(data)
        # Copy nested dicts before popping so the caller's input isn't mutated.
        mesh_data = dict(data.pop("mesh", None) or {})
        colors_data = dict(data.pop("colors", None) or {})
        hough_data = dict(mesh_data.pop("hough", None) or {})

        mesh = _build(MeshConfig, mesh_data, hough=_build(HoughConfig, hough_data))
        colors = _build(ColorConfig, colors_data)
        return _build(cls, data, mesh=mesh, colors=colors)


def _build(dc_type, data: dict, **nested):
    """Return an instance of ``dc_type`` with ``data``/``nested`` overriding defaults.

    Validates that every key in ``data`` is a real field, and coerces YAML lists
    into tuples for tuple-typed fields (e.g. ``canny_thresholds``,
    ``thumbnail_size``) so they stay type-consistent whether built from defaults
    or YAML lists.
    """
    base = dc_type()
    valid = {f.name for f in fields(dc_type)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {dc_type.__name__}: {sorted(unknown)}"
        )
    overrides = dict(nested)
    for key, value in data.items():
        if isinstance(getattr(base, key), tuple) and isinstance(value, list):
            value = tuple(value)
        overrides[key] = value
    return replace(base, **overrides)
