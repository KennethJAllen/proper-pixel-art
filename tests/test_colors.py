"""Tests for the colors module."""

import numpy as np

from proper_pixel_art.colors import get_cell_color, get_cell_color_skip_quantization


class TestGetCellColor:
    """Tests for the mode-based color selection (for quantized images)."""

    def test_most_common_color_selected(self):
        """Returns the most frequent color in the cell."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        cell[:6, :] = [255, 0, 0]  # 60 red pixels
        cell[6:, :] = [0, 0, 255]  # 40 blue pixels

        result = get_cell_color(cell)
        assert result == (255, 0, 0)

    def test_single_color(self):
        """Single color cell returns that color."""
        cell = np.full((5, 5, 3), [42, 84, 126], dtype=np.uint8)
        result = get_cell_color(cell)
        assert result == (42, 84, 126)


class TestGetCellColorSkipQuantization:
    """Tests for histogram-based color selection (when quantization is skipped)."""

    def test_single_color_returns_that_color(self):
        """Cell with single color returns that color."""
        cell = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        result = get_cell_color_skip_quantization(cell)
        assert result == (255, 0, 0)

    def test_uniform_with_outliers_filters_outliers(self):
        """Mostly uniform cell with few outliers returns the uniform color, not average."""
        cell = np.full((10, 10, 3), [200, 100, 50], dtype=np.uint8)
        # Add a few outlier pixels (background bleed-in) - very different colors
        cell[0, 0] = [0, 0, 255]  # Blue outlier - different bin
        cell[0, 1] = [0, 255, 0]  # Green outlier - different bin

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Should be close to the dominant color (200, 100, 50), not skewed by outliers
        assert 190 <= r <= 210, f"Expected r near 200, got {r}"
        assert 90 <= g <= 110, f"Expected g near 100, got {g}"
        assert 40 <= b <= 60, f"Expected b near 50, got {b}"

    def test_two_color_groups_returns_dominant(self):
        """Cell with two distinct color groups returns the dominant one."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        cell[:7, :] = [255, 0, 0]  # 70% red (bin 7,0,0)
        cell[7:, :] = [0, 0, 255]  # 30% blue (bin 0,0,7)

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Should be close to red (the dominant color)
        assert r > 200, f"Expected red-dominant result, got {result}"
        assert b < 50, f"Expected blue filtered out, got {result}"

    def test_noisy_image_returns_stable_color(self):
        """Noisy/grainy image returns a stable average, not erratic pixel."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        # Simulate film grain: many unique red-ish pixels (all in same bin ~250)
        for i in range(7):
            for j in range(10):
                cell[i, j] = [250 + (i * j % 6), (i + j) % 5, (i - j) % 5]
        # Some blue-ish pixels with noise (minority, different bin)
        for i in range(7, 10):
            for j in range(10):
                cell[i, j] = [(i + j) % 5, (i - j) % 5, 250 + (i * j % 6)]

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Result should be red (the dominant color bin)
        assert r > 200, f"Expected red dominant, got {result}"

    def test_gradient_returns_value_in_range(self):
        """Cell with gradient returns a value within the gradient range."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        # Create a red gradient from 50 to 200
        for i in range(10):
            for j in range(10):
                cell[i, j, 0] = 50 + i * 15 + j  # Red varies ~50-200

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Result should be somewhere in the gradient range (mode bin + neighbors)
        assert 50 <= r <= 200, f"Red channel out of expected range, got {result}"

    def test_small_cell_still_works(self):
        """Very small cells (like 2x2) still return valid results."""
        cell = np.array(
            [[[255, 0, 0], [255, 0, 0]], [[0, 0, 255], [0, 255, 0]]], dtype=np.uint8
        )
        result = get_cell_color_skip_quantization(cell)
        assert len(result) == 3
        assert all(0 <= c <= 255 for c in result)

    def test_single_pixel_cell(self):
        """Single pixel cell returns that pixel's color."""
        cell = np.array([[[128, 64, 32]]], dtype=np.uint8)
        result = get_cell_color_skip_quantization(cell)
        assert result == (128, 64, 32)

    def test_empty_cell(self):
        """Empty cell (0 pixels) returns black."""
        cell = np.zeros((0, 0, 3), dtype=np.uint8)
        result = get_cell_color_skip_quantization(cell)
        assert result == (0, 0, 0)

    def test_bin_boundary_handled(self):
        """Colors at bin boundary (e.g., 32) are handled via neighbor merging."""
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        # Values spanning boundary at 32 (bin 0/1 boundary)
        for i in range(10):
            for j in range(10):
                # Values 27-37, spanning the boundary at 32
                val = 27 + (i + j) % 11
                cell[i, j] = [val, val, val]

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Should include all pixels via neighbor merging, average ~32
        assert 27 <= r <= 37, f"Expected value near 32, got {r}"
        assert r == g == b, f"Expected uniform gray, got {result}"

    def test_dark_to_green_gradient_returns_green_ish(self):
        """
        Gradient from dark (black) to bright green should return
        a noticeably green color, not black-ish.
        
        This tests the case where dark pixels dominate by count
        but green is the distinctive color.
        """
        cell = np.zeros((10, 10, 3), dtype=np.uint8)
        # Create gradient: columns go from black to bright green
        for i in range(10):
            for j in range(10):
                # R: stays low
                # G: increases from ~20 to ~150
                # B: stays low
                intensity = (i * 10 + j)  # 0-99, scaled
                cell[i, j] = [
                    20 + intensity // 10,  # R: 20-29
                    20 + intensity,  # G: 20-119
                    20 + intensity // 10,  # B: 20-29
                ]

        result = get_cell_color_skip_quantization(cell)
        r, g, b = result
        # Green should be meaningfully higher than red/blue
        # (showing green tint, not just dark gray)
        assert g > r + 10, f"Expected green tint, got {result}"
        assert g > b + 10, f"Expected green tint, got {result}"
