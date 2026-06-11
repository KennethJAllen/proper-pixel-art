"""Smoke test for the pixelation pipeline.

Runs the algorithm over every case in ``tests/cases.py`` and asserts it completes
without error and produces a non-empty image, guarding against execution
regressions in CI. Visual quality is validated separately by eye -- see
CONTRIBUTING.md -> Visual validation.
"""

from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, pixelate
from proper_pixel_art.config import ColorConfig
from proper_pixel_art.pixelate import build_cell_map, downsample
from proper_pixel_art.utils import Mesh


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


def _downsample_reference(
    image: Image.Image,
    mesh_lines: Mesh,
    skip_quantization: bool = False,
    original_alpha: np.ndarray | None = None,
    color_config: ColorConfig | None = None,
) -> Image.Image:
    """The original per-cell-loop downsample, kept as the oracle for the
    vectorized implementation."""
    color_config = color_config or ColorConfig()
    lines_x, lines_y = mesh_lines
    height_result, width_result = len(lines_y) - 1, len(lines_x) - 1

    if skip_quantization:
        img_array = np.array(image.convert("RGBA"))
    else:
        img_array = np.array(image.convert("RGB"))

    out = np.zeros((height_result, width_result, 4), dtype=np.uint8)

    for j, i in product(range(height_result), range(width_result)):
        x0, x1 = lines_x[i], lines_x[i + 1]
        y0, y1 = lines_y[j], lines_y[j + 1]
        cell = img_array[y0:y1, x0:x1]

        if skip_quantization:
            out[j, i] = colors.get_cell_color_skip_quantization(
                cell,
                alpha_threshold=color_config.alpha_threshold,
                majority_fraction=color_config.transparency_majority_fraction,
                bin_size=color_config.bin_size,
            )
        elif original_alpha is not None:
            cell_alpha = original_alpha[y0:y1, x0:x1]
            out[j, i] = colors.get_cell_color_with_alpha(
                cell,
                cell_alpha,
                alpha_threshold=color_config.alpha_threshold,
                majority_fraction=color_config.transparency_majority_fraction,
            )
        else:
            out[j, i] = colors.get_opaque_cell_color(cell)

    return Image.fromarray(out, mode="RGBA")


def _random_mesh(rng: np.random.Generator, size: int) -> list[int]:
    """Sorted line coordinates with uneven spacing, starting at 0."""
    gaps = rng.integers(3, 9, size=size)
    lines = np.concatenate(([0], np.cumsum(gaps)))
    return [int(line) for line in lines]


class TestBuildCellMap:
    def test_matches_slicing(self):
        """Every pixel's cell id agrees with the old lines[i]:lines[i+1] slices,
        and pixels past the last line get -1."""
        rng = np.random.default_rng(0)
        mesh_lines = (_random_mesh(rng, 5), _random_mesh(rng, 4))
        lines_x, lines_y = mesh_lines
        # Image extends past the last mesh line in both directions
        height, width = lines_y[-1] + 4, lines_x[-1] + 3

        cell_map = build_cell_map(mesh_lines, (height, width))
        assert cell_map.out_shape == (len(lines_y) - 1, len(lines_x) - 1)

        expected = np.full((height, width), -1, dtype=np.int32)
        n_cols = len(lines_x) - 1
        for j, i in product(range(len(lines_y) - 1), range(n_cols)):
            expected[lines_y[j] : lines_y[j + 1], lines_x[i] : lines_x[i + 1]] = (
                j * n_cols + i
            )
        np.testing.assert_array_equal(cell_map.cell_id, expected)
        np.testing.assert_array_equal(
            cell_map.cell_sizes,
            np.bincount(expected[expected >= 0], minlength=cell_map.n_cells),
        )


def _assert_equal_modulo_mode_ties(
    result: Image.Image,
    reference: Image.Image,
    rgb: np.ndarray,
    mesh_lines: Mesh,
) -> None:
    """Assert the two downsample outputs agree on every cell, allowing cells
    with an exact mode-count tie to differ as long as both picked a tied mode
    (Counter breaks ties by scan order, argmax by lowest palette index)."""
    from collections import Counter

    result_arr, reference_arr = np.array(result), np.array(reference)
    lines_x, lines_y = mesh_lines
    for j, i in np.argwhere((result_arr != reference_arr).any(axis=-1)):
        cell = rgb[lines_y[j] : lines_y[j + 1], lines_x[i] : lines_x[i + 1]]
        counts = Counter(map(tuple, cell.reshape(-1, 3))).most_common()
        top_count = counts[0][1]
        tied_modes = {color for color, count in counts if count == top_count}
        assert len(tied_modes) > 1, (
            f"cell ({j},{i}) differs without a mode tie: "
            f"{result_arr[j, i]} vs {reference_arr[j, i]}"
        )
        assert tuple(result_arr[j, i, :3]) in tied_modes
        assert tuple(reference_arr[j, i, :3]) in tied_modes
        assert result_arr[j, i, 3] == reference_arr[j, i, 3]


class TestDownsampleMatchesReference:
    """The vectorized downsample must reproduce the original per-cell loop.

    The skip-quantization path is bit-exact. The quantized path may differ
    only on exact mode-count ties (different but equally valid tie-breaking),
    which the comparison helper accounts for.
    """

    @staticmethod
    def _random_alpha(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        """Alpha with fully-opaque, fully-transparent, and mixed regions,
        including values straddling the threshold and exact 50/50 cells."""
        alpha = rng.choice(
            np.array([0, 60, 127, 128, 200, 255], dtype=np.uint8), size=shape
        )
        alpha[: shape[0] // 3] = 255
        alpha[-(shape[1] // 4) :] = 0
        return alpha

    def test_skip_quantization_path(self):
        rng = np.random.default_rng(1)
        mesh_lines = (_random_mesh(rng, 7), _random_mesh(rng, 6))
        height, width = mesh_lines[1][-1] + 2, mesh_lines[0][-1] + 2

        rgba = rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
        rgba[..., 3] = self._random_alpha(rng, (height, width))
        # Force some cells down to 0-3 opaque pixels to hit the small-cell path
        rgba[:6, :, 3] = 0
        rgba[0, :3, 3] = 255
        image = Image.fromarray(rgba, mode="RGBA")

        result = downsample(image, mesh_lines, skip_quantization=True)
        reference = _downsample_reference(image, mesh_lines, skip_quantization=True)
        np.testing.assert_array_equal(np.array(result), np.array(reference))

    def test_quantized_path_with_alpha(self):
        rng = np.random.default_rng(2)
        mesh_lines = (_random_mesh(rng, 6), _random_mesh(rng, 5))
        height, width = mesh_lines[1][-1] + 1, mesh_lines[0][-1] + 3

        palette = rng.integers(0, 256, size=(7, 3), dtype=np.uint8)
        indices = rng.integers(0, len(palette), size=(height, width))
        rgb = palette[indices]
        alpha = self._random_alpha(rng, (height, width))
        image = Image.fromarray(rgb, mode="RGB")

        result = downsample(image, mesh_lines, original_alpha=alpha)
        reference = _downsample_reference(image, mesh_lines, original_alpha=alpha)
        _assert_equal_modulo_mode_ties(result, reference, rgb, mesh_lines)

    def test_quantized_path_without_alpha(self):
        rng = np.random.default_rng(3)
        mesh_lines = (_random_mesh(rng, 6), _random_mesh(rng, 6))
        height, width = mesh_lines[1][-1], mesh_lines[0][-1]

        palette = rng.integers(0, 256, size=(5, 3), dtype=np.uint8)
        rgb = palette[rng.integers(0, len(palette), size=(height, width))]
        image = Image.fromarray(rgb, mode="RGB")

        result = downsample(image, mesh_lines)
        reference = _downsample_reference(image, mesh_lines)
        _assert_equal_modulo_mode_ties(result, reference, rgb, mesh_lines)

    def test_p_mode_image_matches_rgb(self):
        """A paletted image downsamples identically to its RGB conversion."""
        rng = np.random.default_rng(4)
        mesh_lines = (_random_mesh(rng, 5), _random_mesh(rng, 5))
        height, width = mesh_lines[1][-1], mesh_lines[0][-1]

        palette = rng.integers(0, 256, size=(6, 3), dtype=np.uint8)
        rgb = palette[rng.integers(0, len(palette), size=(height, width))]
        image_rgb = Image.fromarray(rgb, mode="RGB")
        image_p = image_rgb.quantize(colors=8, dither=Image.Dither.NONE)
        assert image_p.mode == "P"

        result_p = downsample(image_p, mesh_lines)
        result_rgb = downsample(image_p.convert("RGB"), mesh_lines)
        # P uses palette-index order and RGB uses sorted-color order for
        # tie-breaking, so ties may legitimately differ here too.
        _assert_equal_modulo_mode_ties(
            result_p, result_rgb, np.asarray(image_p.convert("RGB")), mesh_lines
        )
