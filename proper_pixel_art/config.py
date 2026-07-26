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
from PIL.Image import Dither, Quantize

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
    snap_lines: bool = True  # snap interpolated grid lines to gradient-profile peaks
    snap_search_window_ratio: float = 0.35  # search window as fraction of pixel width
    snap_min_search_window: int = 2  # minimum search window in pixels
    snap_strength_threshold: float = 0.5  # peak must exceed this x mean profile to snap
    anchor_snap_window: int = 2  # snap detected (anchor) lines to peaks within this px
    max_axis_width_ratio: float = (
        1.8  # per-axis width estimates further apart than this -> smaller wins
    )
    validate_width: bool = True  # score candidate widths by within-cell variance
    width_keep_tolerance: float = (
        1.5  # replace the estimated width only when it scores (1 + this) x the best
    )
    width_replace_tolerance: float = (
        0.2  # on correction, accept a replacement scoring within (1 + this) x the best
    )
    profile_width_min_gaps: int = (
        5  # fewer Hough gaps than this -> profile width estimate
    )
    split_leftover_fraction: float = (
        0.4  # leftover fraction of pixel width that earns an extra grid line
    )
    hough: HoughConfig = field(default_factory=HoughConfig)


@dataclass
class PaletteConfig:
    """Parameters for the palette-quantization method (``method: palette``).

    The image is quantized to ``num_colors`` with PIL's ``Image.quantize`` and
    each mesh cell takes its most common palette color.
    """

    num_colors: int = 16  # palette size (1-256)
    quantize_method: str = "MAXCOVERAGE"  # PIL.Image.Quantize member name
    dither: str = "NONE"  # PIL.Image.Dither member name (NONE, FLOYDSTEINBERG, ...)
    kmeans: int = 0  # PIL quantize() k-means refinement iterations; 0 disables

    def __post_init__(self) -> None:
        if not 1 <= self.num_colors <= 256:
            raise ValueError(f"num_colors must be in 1-256, got {self.num_colors}.")
        if self.kmeans < 0:
            raise ValueError(f"kmeans must be >= 0, got {self.kmeans}.")

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

    @property
    def dither_mode(self) -> int:
        """Resolve ``dither`` to a ``PIL.Image.Dither`` value."""
        try:
            return getattr(Dither, self.dither)
        except AttributeError as exc:
            valid = ", ".join(d.name for d in Dither)
            raise ValueError(
                f"Unknown dither {self.dither!r}. Valid options: {valid}."
            ) from exc


@dataclass
class DominantConfig:
    """Parameters for the dominant-color method (``method: dominant``).

    Original colors are preserved: each mesh cell picks its dominant color via
    offset binning, then near-duplicate output colors are merged by
    agglomerative clustering so flat areas collapse to a single color.
    """

    bin_size: int = 52  # RGB bin size for the per-cell dominant color
    merge_distance: int = (
        12  # merge output colors closer than this (Euclidean RGB); 0 disables
    )
    merge_linkage: str = (
        "complete"  # scipy linkage criterion: "complete", "average" or "single"
    )
    max_linkage_colors: int = (
        4096  # cap on unique colors fitted by clustering (bounds memory)
    )

    def __post_init__(self) -> None:
        # bin_size is a divisor in dominant_rgb_by_binning (num_bins =
        # 255 // bin_size + 1); 0 would raise an opaque ZeroDivisionError later.
        if self.bin_size < 1:
            raise ValueError(f"bin_size must be >= 1, got {self.bin_size}.")
        # Restrict to criteria whose cut height is in Euclidean RGB units, so
        # merge_distance keeps its meaning (ward/centroid/median cut heights
        # are not plain distances).
        if self.merge_linkage not in ("complete", "average", "single"):
            raise ValueError(
                f"Unknown merge_linkage {self.merge_linkage!r}. "
                f"Valid options: complete, average, single."
            )
        # scipy linkage needs at least 2 observations to fit anything.
        if self.max_linkage_colors < 2:
            raise ValueError(
                f"max_linkage_colors must be >= 2, got {self.max_linkage_colors}."
            )


@dataclass
class ColorConfig:
    """Parameters controlling color selection, quantization and transparency.

    ``method`` selects between the two quantization methods, whose specific
    knobs live in the ``palette`` and ``dominant`` sub-configs; the remaining
    fields apply to both.
    """

    method: str = "dominant"  # "dominant" (preserve colors) or "palette" (quantize)
    alpha_threshold: int = 128  # alpha >= this is opaque (0-255)
    transparency_majority_fraction: float = (
        0.5  # cell transparent if >= this fraction is transparent
    )
    top_colors_limit: int = 8  # common colors sampled to pick a background
    thumbnail_size: tuple[int, int] = (160, 160)  # downscale size for color analysis
    background_candidates: list[RGB] | None = None  # override background palette
    palette: "PaletteConfig" = field(default_factory=PaletteConfig)
    dominant: "DominantConfig" = field(default_factory=DominantConfig)

    def __post_init__(self) -> None:
        if self.method not in ("dominant", "palette"):
            raise ValueError(
                f"Unknown color method {self.method!r}. "
                f"Valid options: dominant, palette."
            )
        # Normalize YAML lists-of-lists into RGB tuples. _build skips this field
        # because its default is None rather than a tuple.
        if self.background_candidates is not None:
            self.background_candidates = [tuple(c) for c in self.background_candidates]


@dataclass
class PixelateConfig:
    """Top-level configuration for :func:`proper_pixel_art.pixelate.pixelate`.

    Two fields use numeric sentinels rather than ``None`` for their special
    states, leaving ``None`` free to mean "not provided" in ``pixelate`` kwargs:
    ``scale_result=1`` means no scaling and ``pixel_width=0`` auto-detects the
    pixel width. The quantization method is selected by ``colors.method``.
    """

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
        palette_data = dict(colors_data.pop("palette", None) or {})
        dominant_data = dict(colors_data.pop("dominant", None) or {})

        mesh = _build(MeshConfig, mesh_data, hough=_build(HoughConfig, hough_data))
        colors = _build(
            ColorConfig,
            colors_data,
            palette=_build(PaletteConfig, palette_data),
            dominant=_build(DominantConfig, dominant_data),
        )
        return _build(cls, data, mesh=mesh, colors=colors)


def with_num_colors(cfg: PixelateConfig, num_colors: int) -> PixelateConfig:
    """Return a copy of ``cfg`` with the ``num_colors`` shorthand applied.

    ``num_colors > 0`` selects the palette method with that palette size;
    ``num_colors == 0`` selects the dominant method. This is the mapping behind
    the CLI ``-c`` flag and the ``pixelate(num_colors=...)`` kwarg.
    """
    if num_colors:
        colors = replace(
            cfg.colors,
            method="palette",
            palette=replace(cfg.colors.palette, num_colors=num_colors),
        )
    else:
        colors = replace(cfg.colors, method="dominant")
    return replace(cfg, colors=colors)


# Keys the 1.8.0 color-method split moved out of their old dataclass, so a
# pre-1.8.0 YAML gets told where the setting went instead of a bare
# "unknown key". Keyed by the dataclass the key used to live on.
_MOVED_KEYS: dict[str, dict[str, str]] = {
    "PixelateConfig": {
        "num_colors": (
            "moved to colors.palette.num_colors "
            "(colors.method: palette selects that method)"
        ),
    },
    "ColorConfig": {
        "num_colors": "moved to colors.palette.num_colors",
        "quantize_method": "moved to colors.palette.quantize_method",
        "bin_size": "moved to colors.dominant.bin_size",
        "output_color_merge_distance": "moved to colors.dominant.merge_distance",
    },
}


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
        moved = _MOVED_KEYS.get(dc_type.__name__, {})
        hints = [f"{key!r} {moved[key]}" for key in sorted(unknown) if key in moved]
        message = f"Unknown config key(s) for {dc_type.__name__}: {sorted(unknown)}"
        if hints:
            message += ". " + "; ".join(hints)
        raise ValueError(message)
    overrides = dict(nested)
    for key, value in data.items():
        if isinstance(getattr(base, key), tuple) and isinstance(value, list):
            value = tuple(value)
        overrides[key] = value
    return replace(base, **overrides)
