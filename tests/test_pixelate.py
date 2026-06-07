"""Smoke test for the pixelation pipeline.

Runs the algorithm over every case in ``tests/cases.py`` and asserts it completes
without error and produces a non-empty image, guarding against execution
regressions in CI. Visual quality is validated separately by eye -- see
CONTRIBUTING.md -> Visual validation.
"""

from pathlib import Path

from PIL import Image

from proper_pixel_art import pixelate


def test_pixelate_pngs(
    pixelate_png_test_params: dict[str, dict], tmp_path: Path
) -> None:
    """Each case pixelates without error and yields a non-empty image."""
    for name, params in pixelate_png_test_params.items():
        intermediate_dir = tmp_path / name
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(params["path"])
        result = pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=intermediate_dir,
        )

        assert result.width > 0 and result.height > 0, f"Invalid dimensions for {name}"
