from pathlib import Path

import pytest


@pytest.fixture(name="assets")
def fixture_assets() -> Path:
    assets = Path.cwd() / "assets"
    return assets


@pytest.fixture(name="pixelate_png_test_params")
def fixture_pixelate_png_test_params(assets: Path) -> dict[str, dict]:
    pixelate_png_test_params = {
        # Test case for transparent background with hole
        "anchor": {
            "num_colors": 16,
            "result_scale": 5,
            "transparent_background": True,
            "path": assets / "anchor" / "anchor.png",
        },
        "ash": {
            "num_colors": 16,
            "result_scale": 5,
            "transparent_background": False,
            "path": assets / "ash" / "ash.png",
        },
        "bat": {
            "num_colors": 16,
            "result_scale": 5,
            "transparent_background": True,
            "path": assets / "bat" / "bat.png",
        },
        "blob": {
            "num_colors": 16,
            "result_scale": 25,
            "transparent_background": False,
            "path": assets / "blob" / "blob.png",
        },
        "demon": {
            "num_colors": 64,
            "result_scale": 5,
            "transparent_background": True,
            "path": assets / "demon" / "demon.png",
        },
        "mountain": {
            "num_colors": 64,
            "result_scale": 5,
            "transparent_background": False,
            "path": assets / "mountain" / "mountain.png",
        },
        # Test case for skipping quantization
        "pumpkin": {
            "num_colors": None,
            "result_scale": 5,
            "transparent_background": False,
            "path": assets / "pumpkin" / "pumpkin.png",
        },
    }
    return pixelate_png_test_params
