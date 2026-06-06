from pathlib import Path

import pytest

from tests.cases import PIXELATE_PNG_CASES


@pytest.fixture(name="assets")
def fixture_assets() -> Path:
    assets = Path.cwd() / "assets"
    return assets


@pytest.fixture(name="pixelate_png_test_params")
def fixture_pixelate_png_test_params() -> dict[str, dict]:
    return PIXELATE_PNG_CASES
