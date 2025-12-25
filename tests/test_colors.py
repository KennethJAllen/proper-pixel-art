"""Tests for the colors module."""

import numpy as np

from proper_pixel_art import colors


def _make_rgba(rgb_cell: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Helper to convert RGB cell to RGBA with given alpha."""
    h, w = rgb_cell.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_cell
    rgba[:, :, 3] = alpha
    return rgba


class TestGetCellColor:
    """Tests for the mode-based color selection (for quantized images)."""

    def test_most_common_color_selected(self):
        """Returns the most frequent color in the cell."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        cell[:6, :] = [255, 0, 0]  # 60 red pixels
        cell[6:, :] = [0, 0, 255]  # 40 blue pixels

        result = colors.get_opaque_cell_color(cell)
        assert result == (255, 0, 0, 255)

    def test_single_color(self):
        """Single color cell returns that color."""
        cell = np.full((5, 5, 3), [42, 84, 126], dtype=np.uint8)
        result = colors.get_opaque_cell_color(cell)
        assert result == (42, 84, 126, 255)


class TestGetCellColorWithAlpha:
    """Tests for get_cell_color_with_alpha (quantized RGB + alpha channel)."""

    def test_majority_transparent_returns_transparent(self):
        """Cell with >=50% transparent pixels returns fully transparent."""
        # RGB cell from quantized image
        cell_pixels = np.full((10, 10, 3), [200, 100, 50], dtype=np.uint8)

        # Alpha channel from original image - 70% transparent
        cell_alpha = np.zeros((10, 10), dtype=np.uint8)
        cell_alpha[:3, :] = 255  # 30% opaque (30 pixels)
        # Remaining 70% stays at 0 (transparent)

        result = colors.get_cell_color_with_alpha(cell_pixels, cell_alpha)
        assert result == (0, 0, 0, 0)

    def test_majority_opaque_returns_most_common_color(self):
        """Cell with >50% opaque pixels returns the most common RGB color."""
        # Quantized RGB cell with multiple colors
        cell_pixels = np.zeros((10, 10, 3), dtype=np.uint8)
        cell_pixels[:7, :] = [255, 0, 0]  # 70% red
        cell_pixels[7:, :] = [0, 0, 255]  # 30% blue

        # Alpha channel from original - 70% opaque
        cell_alpha = np.zeros((10, 10), dtype=np.uint8)
        cell_alpha[:7, :] = 255  # 70% opaque
        # Remaining 30% transparent

        result = colors.get_cell_color_with_alpha(cell_pixels, cell_alpha)
        assert result == (255, 0, 0, 255)  # Most common color (red) with full opacity

    def test_exactly_50_percent_transparent_returns_transparent(self):
        """Cell with exactly 50% transparent (tie) returns transparent."""
        cell_pixels = np.full((10, 10, 3), [100, 100, 100], dtype=np.uint8)

        # Exactly 50% opaque, 50% transparent
        cell_alpha = np.zeros((10, 10), dtype=np.uint8)
        cell_alpha[:5, :] = 255  # 50% opaque
        # Remaining 50% transparent

        result = colors.get_cell_color_with_alpha(cell_pixels, cell_alpha)
        assert result == (0, 0, 0, 0)

    def test_single_color_all_opaque(self):
        """Single color cell with all opaque pixels returns that color."""
        cell_pixels = np.full((5, 5, 3), [42, 84, 126], dtype=np.uint8)
        cell_alpha = np.full((5, 5), 255, dtype=np.uint8)  # All opaque

        result = colors.get_cell_color_with_alpha(cell_pixels, cell_alpha)
        assert result == (42, 84, 126, 255)

    def test_most_common_color_selected_from_quantized(self):
        """Returns the most frequent color from quantized RGB cell."""
        # Quantized cell with clear most-common color
        cell_pixels = np.zeros((10, 10, 3), dtype=np.uint8)
        cell_pixels[:8, :] = [200, 50, 25]  # 80% this color
        cell_pixels[8:, :] = [100, 150, 75]  # 20% this color

        # All pixels opaque
        cell_alpha = np.full((10, 10), 255, dtype=np.uint8)

        result = colors.get_cell_color_with_alpha(cell_pixels, cell_alpha)
        assert result == (200, 50, 25, 255)


class TestGetCellColorSkipQuantization:
    """Tests for histogram-based color selection (when quantization is skipped).

    The function now takes RGBA input and returns RGBA output.
    For fully opaque cells (alpha=255), the RGB result matches the original algorithm.
    """

    def test_single_color_returns_that_color(self):
        """Cell with single color returns that color."""
        rgb = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        cell = _make_rgba(rgb)
        result = colors.get_cell_color_skip_quantization(cell)
        assert result == (255, 0, 0, 255)

    def test_uniform_with_outliers_filters_outliers(self):
        """Mostly uniform cell with few outliers returns the uniform color, not average."""
        rgb = np.full((10, 10, 3), [200, 100, 50], dtype=np.uint8)
        # Add a few outlier pixels (background bleed-in) - very different colors
        rgb[0, 0] = [0, 0, 255]  # Blue outlier - different bin
        rgb[0, 1] = [0, 255, 0]  # Green outlier - different bin
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Should be close to the dominant color (200, 100, 50), not skewed by outliers
        assert 190 <= r <= 210, f"Expected r near 200, got {r}"
        assert 90 <= g <= 110, f"Expected g near 100, got {g}"
        assert 40 <= b <= 60, f"Expected b near 50, got {b}"
        assert a == 255

    def test_two_color_groups_returns_dominant(self):
        """Cell with two distinct color groups returns the dominant one."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:7, :] = [255, 0, 0]  # 70% red (bin 7,0,0)
        rgb[7:, :] = [0, 0, 255]  # 30% blue (bin 0,0,7)
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Should be close to red (the dominant color)
        assert r > 200, f"Expected red-dominant result, got {result}"
        assert b < 50, f"Expected blue filtered out, got {result}"
        assert a == 255

    def test_noisy_image_returns_stable_color(self):
        """Noisy/grainy image returns a stable average, not erratic pixel."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        # Simulate film grain: many unique red-ish pixels (all in same bin ~250)
        for i in range(7):
            for j in range(10):
                rgb[i, j] = [250 + (i * j % 6), (i + j) % 5, (i - j) % 5]
        # Some blue-ish pixels with noise (minority, different bin)
        for i in range(7, 10):
            for j in range(10):
                rgb[i, j] = [(i + j) % 5, (i - j) % 5, 250 + (i * j % 6)]
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Result should be red (the dominant color bin)
        assert r > 200, f"Expected red dominant, got {result}"
        assert a == 255

    def test_gradient_returns_value_in_range(self):
        """Cell with gradient returns a value within the gradient range."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        # Create a red gradient from 50 to 200
        for i in range(10):
            for j in range(10):
                rgb[i, j, 0] = 50 + i * 15 + j  # Red varies ~50-200
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Result should be somewhere in the gradient range (mode bin + neighbors)
        assert 50 <= r <= 200, f"Red channel out of expected range, got {result}"
        assert a == 255

    def test_small_cell_still_works(self):
        """Very small cells (like 2x2) still return valid results."""
        rgb = np.array(
            [[[255, 0, 0], [255, 0, 0]], [[0, 0, 255], [0, 255, 0]]], dtype=np.uint8
        )
        cell = _make_rgba(rgb)
        result = colors.get_cell_color_skip_quantization(cell)
        assert len(result) == 4
        assert all(0 <= c <= 255 for c in result)
        assert result[3] == 255  # Alpha should be opaque

    def test_single_pixel_cell(self):
        """Single pixel cell returns that pixel's color."""
        rgb = np.array([[[128, 64, 32]]], dtype=np.uint8)
        cell = _make_rgba(rgb)
        result = colors.get_cell_color_skip_quantization(cell)
        assert result == (128, 64, 32, 255)

    def test_empty_cell(self):
        """Empty cell (0 pixels) returns transparent."""
        cell = np.zeros((0, 0, 4), dtype=np.uint8)
        result = colors.get_cell_color_skip_quantization(cell)
        assert result == (0, 0, 0, 0)

    def test_bin_boundary_handled(self):
        """Colors at bin boundary (e.g., 32) are handled via neighbor merging."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        # Values spanning boundary at 32 (bin 0/1 boundary)
        for i in range(10):
            for j in range(10):
                # Values 27-37, spanning the boundary at 32
                val = 27 + (i + j) % 11
                rgb[i, j] = [val, val, val]
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Should include all pixels via neighbor merging, average ~32
        assert 27 <= r <= 37, f"Expected value near 32, got {r}"
        assert r == g == b, f"Expected uniform gray, got {result}"
        assert a == 255

    def test_dark_to_green_gradient_returns_green_ish(self):
        """
        Gradient from dark (black) to bright green should return
        a noticeably green color, not black-ish.

        This tests the case where dark pixels dominate by count
        but green is the distinctive color.
        """
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        # Create gradient: columns go from black to bright green
        for i in range(10):
            for j in range(10):
                # R: stays low
                # G: increases from ~20 to ~150
                # B: stays low
                intensity = i * 10 + j  # 0-99, scaled
                rgb[i, j] = [
                    20 + intensity // 10,  # R: 20-29
                    20 + intensity,  # G: 20-119
                    20 + intensity // 10,  # B: 20-29
                ]
        cell = _make_rgba(rgb)

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        # Green should be meaningfully higher than red/blue
        # (showing green tint, not just dark gray)
        assert g > r + 10, f"Expected green tint, got {result}"
        assert g > b + 10, f"Expected green tint, got {result}"
        assert a == 255

    def test_majority_transparent_returns_transparent(self):
        """Cell with >=50% transparent pixels returns fully transparent."""
        cell = np.zeros((10, 10, 4), dtype=np.uint8)
        cell[:4, :, :3] = [100, 100, 100]  # 40% opaque
        cell[:4, :, 3] = 255
        cell[4:, :, 3] = 0  # 60% transparent

        result = colors.get_cell_color_skip_quantization(cell)
        assert result == (0, 0, 0, 0)

    def test_majority_opaque_returns_opaque_color(self):
        """Cell with >50% opaque pixels returns the opaque color."""
        cell = np.zeros((10, 10, 4), dtype=np.uint8)
        cell[:6, :, :3] = [100, 100, 100]  # 60% opaque
        cell[:6, :, 3] = 255
        cell[6:, :, 3] = 0  # 40% transparent

        result = colors.get_cell_color_skip_quantization(cell)
        r, g, b, a = result
        assert a == 255
        assert 90 <= r <= 110

    def test_exactly_50_percent_transparent_returns_transparent(self):
        """Cell with exactly 50% transparent (tie) returns transparent."""
        cell = np.zeros((10, 10, 4), dtype=np.uint8)
        cell[:5, :, :3] = [100, 100, 100]  # 50% opaque
        cell[:5, :, 3] = 255
        cell[5:, :, 3] = 0  # 50% transparent

        result = colors.get_cell_color_skip_quantization(cell)
        assert result == (0, 0, 0, 0)
