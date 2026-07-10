from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import mesh
from proper_pixel_art.config import MeshConfig


def test_mesh():
    """
    Checks that the mesh calculated for the blob image is non-trivial.
    """
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    mesh_x, mesh_y = mesh.compute_mesh(img)
    assert (len(mesh_x)) > 2
    assert (len(mesh_y)) > 2


def _grid_image(boundaries: list[int], size: int, axis: int) -> np.ndarray:
    """Grayscale image of hard-edged stripes changing value at ``boundaries``."""
    values = np.zeros(size, dtype=np.uint8)
    fill = 0
    for start, end in zip([0] + boundaries, boundaries + [size], strict=True):
        values[start:end] = fill
        fill = 255 - fill
    line = values[np.newaxis, :] if axis == 0 else values[:, np.newaxis]
    return np.repeat(line, size, axis=axis)


def test_compute_gradient_profiles_peaks_at_boundaries():
    """Profiles of a hard-edged grid peak exactly at the stripe boundaries."""
    boundaries = [10, 20, 30, 40]
    grey = _grid_image(boundaries, size=50, axis=0)
    profile_x, profile_y = mesh.compute_gradient_profiles(grey)

    assert profile_x.shape == (50,)
    assert profile_y.shape == (50,)
    # Vertical stripes: no gradient along y.
    assert np.all(profile_y == 0)
    # A [-1, 0, 1] kernel responds at the boundary and one column before it;
    # everywhere else the profile must be zero.
    responding = {b + offset for b in boundaries for offset in (-1, 0)}
    for x in range(50):
        if x in responding:
            assert profile_x[x] == profile_x.max()
        else:
            assert profile_x[x] == 0


def test_homogenize_lines_snaps_to_drifting_grid():
    """Interpolated lines snap to profile peaks of a drifting grid."""
    # True boundaries drift off the even 10-px spacing.
    true_boundaries = [0, 11, 21, 32, 42]
    profile = np.zeros(43)
    profile[true_boundaries[1:-1]] = 100.0

    snapped = mesh.homogenize_lines(
        [0, 42], pixel_width=10, profile=profile, mesh_config=MeshConfig()
    )
    assert snapped == true_boundaries


def test_homogenize_lines_without_profile_is_unchanged():
    """profile=None (and snap_lines=False) preserve the even-spacing output."""
    lines = [0, 42]
    expected = [0, 10, 21, 31, 42]  # int(n * 42 / 4) for n in 0..3, plus end

    assert mesh.homogenize_lines(lines, pixel_width=10) == expected
    profile = np.zeros(43)
    profile[[11, 21, 32]] = 100.0
    config = MeshConfig(snap_lines=False)
    result = mesh.homogenize_lines(lines, 10, profile=profile, mesh_config=config)
    assert result == expected


def test_homogenize_lines_zero_profile_keeps_even_spacing():
    """A profile with no edge evidence falls back to rigid spacing."""
    profile = np.zeros(43)
    result = mesh.homogenize_lines([0, 42], 10, profile=profile)
    assert result == [0, 10, 21, 31, 42]


def test_homogenize_lines_splits_gaps_widened_by_snapping():
    """When two neighbors snap apart, the oversized gap gets an extra line."""
    # Targets 10/20/30 snap to peaks at 6 and 24, opening an 18-px gap
    # (1.8x pixel width); the repair pass splits it at the peak at 16.
    profile = np.zeros(41)
    profile[[6, 16, 24]] = 100.0
    result = mesh.homogenize_lines([0, 40], pixel_width=10, profile=profile)
    assert result == [0, 6, 16, 24, 30, 40]


def test_split_leftover_fraction_controls_extra_line():
    """A leftover >= split_leftover_fraction of a pixel width earns a line."""
    lines = [0, 14]  # 1.4x pixel width
    default = mesh.homogenize_lines(lines, pixel_width=10)
    assert default == [0, 7, 14]  # default fraction 0.4 splits

    config = MeshConfig(split_leftover_fraction=0.5)
    assert mesh.homogenize_lines(lines, 10, mesh_config=config) == [0, 14]


def test_estimate_width_from_profile():
    """Recovers the spacing of regular peaks; returns None on a flat profile."""
    profile = np.zeros(100)
    profile[7:98:7] = 50.0
    assert mesh.estimate_width_from_profile(profile) == 7

    assert mesh.estimate_width_from_profile(np.zeros(100)) is None
    assert mesh.estimate_width_from_profile(np.full(100, 3.0)) is None
