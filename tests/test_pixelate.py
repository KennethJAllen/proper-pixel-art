"""Visual regression tests.

Each case is pixelated and compared against its committed golden output at
``assets/{name}/result.png`` with a tolerant pixel diff. A failure means the
algorithm's output changed. If the change is intentional, regenerate the
goldens with ``uv run python scripts/gen_outputs.py``, review the resulting
image diff, and commit.
"""

from pathlib import Path

import numpy as np
import pytest
from cases import golden_path
from PIL import Image

from proper_pixel_art import pixelate

# Tolerances for the golden comparison. The algorithm is deterministic, so these
# only absorb minor cross-platform / Pillow-version quantizer drift.
MAX_MEAN_ABS_DIFF = 2.0  # mean abs per-channel difference, 0-255 scale
MAX_DIFF_FRACTION = 0.005  # fraction of channels allowed to differ by >1


def assert_images_close(actual: Image.Image, golden: Image.Image, name: str) -> None:
    """Assert two images match within tolerance (compared as RGBA arrays)."""
    actual_rgba = actual.convert("RGBA")
    golden_rgba = golden.convert("RGBA")
    assert actual_rgba.size == golden_rgba.size, (
        f"Size mismatch for {name}: {actual_rgba.size} != golden {golden_rgba.size}. "
        f"Regenerate goldens with `uv run python scripts/gen_outputs.py`."
    )

    a = np.asarray(actual_rgba, dtype=np.int16)
    g = np.asarray(golden_rgba, dtype=np.int16)
    diff = np.abs(a - g)

    mean_abs_diff = float(diff.mean())
    diff_fraction = float((diff > 1).mean())

    assert (
        mean_abs_diff <= MAX_MEAN_ABS_DIFF and diff_fraction <= MAX_DIFF_FRACTION
    ), (
        f"Output for {name} differs from golden "
        f"(mean abs diff {mean_abs_diff:.3f} > {MAX_MEAN_ABS_DIFF} or "
        f"differing fraction {diff_fraction:.4f} > {MAX_DIFF_FRACTION}). "
        f"If intentional, regenerate goldens with "
        f"`uv run python scripts/gen_outputs.py` and review the diff."
    )


def test_pixelate_pngs(pixelate_png_test_params: dict[str, dict]) -> None:
    """Pixelate each asset and compare against its committed golden output."""
    output_dir = Path.cwd() / "tests" / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    for name, params in pixelate_png_test_params.items():
        pixel_dir = output_dir / str(name)
        pixel_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(params["path"])
        result = pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=pixel_dir,
        )

        # Keep the fresh result around for inspection when a comparison fails.
        result_path = pixel_dir / "result.png"
        result.save(result_path)
        assert result.width > 0 and result.height > 0, f"Invalid dimensions for {name}"

        golden = golden_path(name)
        if not golden.exists():
            pytest.fail(
                f"Missing golden image for {name} at {golden}. "
                f"Generate it with `uv run python scripts/gen_outputs.py`."
            )

        assert_images_close(result, Image.open(golden), name)
