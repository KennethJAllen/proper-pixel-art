"""Convert pixel-art-style images and videos to true-resolution assets."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from proper_pixel_art.config import (
    ColorConfig,
    DominantConfig,
    HoughConfig,
    MeshConfig,
    PaletteConfig,
    PixelateConfig,
    VideoConfig,
)
from proper_pixel_art.image import PixelateResult, pixelate

try:
    __version__ = _version("proper-pixel-art")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"


def __getattr__(name):
    # PEP 562: load pixelate_video lazily so importing the package doesn't pay
    # for the video pipeline (cv2 video I/O, av).
    if name == "pixelate_video":
        from proper_pixel_art.video import pixelate_video

        return pixelate_video
    raise AttributeError(name)


__all__ = [
    "pixelate",
    "pixelate_video",
    "PixelateResult",
    "PixelateConfig",
    "MeshConfig",
    "ColorConfig",
    "PaletteConfig",
    "DominantConfig",
    "HoughConfig",
    "VideoConfig",
    "__version__",
]
