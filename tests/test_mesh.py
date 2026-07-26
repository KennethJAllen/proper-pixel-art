from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from proper_pixel_art import mesh
from proper_pixel_art.config import MeshConfig


def test_mesh():
    """
    Checks that the mesh calculated for the blob image is non-trivial.
    """
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    (mesh_x, mesh_y), pixel_width = mesh.compute_mesh(img)
    assert (len(mesh_x)) > 2
    assert (len(mesh_y)) > 2
    assert pixel_width >= 1


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


def test_snap_anchor_lines_snaps_to_nearby_peak():
    """An anchor 1-2px off a profile peak snaps to it; borders never move."""
    profile = np.zeros(100)
    profile[[20, 40, 60, 80]] = 100.0
    lines = [0, 21, 39, 60, 82, 99]

    snapped = mesh.snap_anchor_lines(lines, profile, MeshConfig())

    assert snapped == [0, 20, 40, 60, 80, 99]
    # Peaks further away than the window are out of reach.
    far = mesh.snap_anchor_lines([0, 50, 99], profile, MeshConfig())
    assert far == [0, 50, 99]


def test_snap_anchor_lines_flat_profile_unchanged():
    """With no gradient evidence, anchors stay where Hough put them."""
    lines = [0, 33, 66, 99]
    assert mesh.snap_anchor_lines(lines, np.zeros(100), MeshConfig()) == lines


def test_snap_anchor_lines_borders_pinned_after_clustering():
    """Anchors snapping toward a border are dropped, not merged into it."""
    profile = np.zeros(50)
    profile[2] = 100.0
    # Anchor at 4 snaps to 2, within cluster_threshold of border 0.
    lines = [0, 4, 25, 49]
    snapped = mesh.snap_anchor_lines(lines, profile, MeshConfig())
    assert snapped[0] == 0
    assert snapped[-1] == 49
    assert sorted(snapped) == snapped


def test_estimate_width_by_autocorrelation_with_missing_peaks():
    """Autocorrelation recovers the period even with 30% of peaks removed."""
    rng = np.random.default_rng(0)
    period = 8
    profile = np.zeros(400)
    peaks = np.arange(period, 400, period)
    keep = rng.random(len(peaks)) > 0.3
    profile[peaks[keep]] = 100.0

    assert mesh.estimate_width_by_autocorrelation(profile) == period


def test_estimate_width_by_autocorrelation_prefers_fundamental():
    """The first qualifying peak (fundamental) wins over the 2x harmonic."""
    period = 10
    profile = np.zeros(300)
    profile[period::period] = 100.0
    # Every second peak stronger: the 2x harmonic is also present.
    profile[2 * period :: 2 * period] = 200.0

    assert mesh.estimate_width_by_autocorrelation(profile) == period


def test_estimate_width_by_autocorrelation_degenerate_profiles():
    assert mesh.estimate_width_by_autocorrelation(np.zeros(100)) is None
    assert mesh.estimate_width_by_autocorrelation(np.full(100, 5.0)) is None
    assert mesh.estimate_width_by_autocorrelation(np.zeros(4)) is None


def test_estimate_axis_width_prefers_hough_gaps():
    """With enough gaps the Hough median wins even against a profile."""
    lines = [0, 10, 20, 30, 40, 50]  # 5 gaps of 10
    # Precomputed profile estimate would say 7; Hough gaps still win.
    assert mesh.estimate_axis_width(lines, 7, None, MeshConfig()) == 10


def test_estimate_axis_width_falls_back_to_profile():
    """Too few gaps -> profile peak estimate (autocorrelation only if no peak)."""
    lines = [0, 50]
    assert mesh.estimate_axis_width(lines, 7, 5, MeshConfig()) == 7
    assert mesh.estimate_axis_width(lines, None, 5, MeshConfig()) == 5


def test_estimate_axis_width_none_when_no_evidence():
    assert mesh.estimate_axis_width([0, 50], None, None, MeshConfig()) is None


def test_resolve_pixel_width():
    cfg = MeshConfig()
    assert mesh.resolve_pixel_width(10, 10, cfg) == 10
    assert mesh.resolve_pixel_width(10, 12, cfg) == 11  # mean when close
    assert mesh.resolve_pixel_width(10, 30, cfg) == 10  # >1.8x -> smaller wins
    assert mesh.resolve_pixel_width(10, None, cfg) == 10
    assert mesh.resolve_pixel_width(None, 12, cfg) == 12
    assert mesh.resolve_pixel_width(None, None, cfg) is None


def _checker_grey(cells: int, width: int) -> np.ndarray:
    """Checkerboard grayscale image of ``cells`` x ``cells`` hard pixels."""
    logical = np.indices((cells, cells)).sum(axis=0) % 2 * 255
    return np.repeat(np.repeat(logical, width, 0), width, 1).astype(np.uint8)


def test_score_mesh_prefers_true_grid():
    """The true-width mesh scores (near) zero; misaligned widths score higher."""
    width, cells = 10, 8
    grey = _checker_grey(cells, width)
    size = cells * width
    borders = [0, size - 1]

    def full_mesh(w):
        lines = mesh.homogenize_lines(list(borders), w)
        return lines, lines

    true_mesh = (
        list(range(0, size, width)) + [size - 1],
        list(range(0, size, width)) + [size - 1],
    )
    true_score = mesh.score_mesh(grey, (true_mesh[0], true_mesh[1]))
    off_score = mesh.score_mesh(grey, full_mesh(width + 1))
    double_score = mesh.score_mesh(grey, full_mesh(2 * width))
    assert true_score < off_score
    assert true_score < double_score


def test_select_pixel_width_confirms_correct_width():
    """On a clean hard grid the correct candidate is kept."""
    width, cells = 10, 8
    grey = _checker_grey(cells, width)
    size = cells * width
    mesh_initial = ([0, size - 1], [0, size - 1])

    chosen, scores = mesh.select_pixel_width(
        grey, mesh_initial, None, [width], MeshConfig(), debug_context=True
    )
    assert chosen == width
    assert scores[2 * width] > scores[width]


def test_select_pixel_width_corrects_double_width_lockon():
    """A 2x-harmonic estimate is corrected down to the true width."""
    width, cells = 10, 8
    grey = _checker_grey(cells, width)
    size = cells * width
    mesh_initial = ([0, size - 1], [0, size - 1])
    # Gradient profile peaking at the true boundaries, as compute_mesh
    # provides, so candidate meshes snap onto the grid they model.
    profile = np.zeros(size)
    profile[width::width] = 100.0

    # Estimate locked on to 2x the true width, with the fundamental found by
    # another estimator (as autocorrelation does). At 2x each cell spans four
    # logical pixels of the checkerboard, so its score is decisively worse.
    chosen, scores = mesh.select_pixel_width(
        grey, mesh_initial, (profile, profile), [2 * width, width], MeshConfig()
    )
    assert chosen == width
    assert scores[2 * width] > 2 * scores[width]


def test_select_pixel_width_keeps_estimate_without_backed_alternative():
    """A well-scoring width that no estimator produced is never adopted:
    on smooth content half-width cells always score better, but the estimate
    is kept when it is the only estimator-backed candidate."""
    # Smooth gradient: within-cell variance shrinks without bound as cells
    # shrink, so any expanded half-width candidate scores far "better".
    ramp = np.tile(np.arange(160, dtype=np.uint8), (160, 1))
    mesh_initial = ([0, 159], [0, 159])

    chosen, scores = mesh.select_pixel_width(
        ramp, mesh_initial, None, [40], MeshConfig(), debug_context=True
    )
    assert chosen == 40
    assert scores[20] < scores[40]  # the bias is real; the guard resists it


def test_validate_width_corrects_hough_lockon_when_enabled():
    """A Hough 2x lock-on (edges only at every other boundary) is corrected
    when validation is on and kept when it is off."""
    width, cells = 10, 16
    grey = _checker_grey(cells, width)
    size = cells * width
    # Edge evidence only at even boundaries: Hough gaps say 2x the width.
    edges = np.zeros((size, size), dtype=np.uint8)
    for pos in range(2 * width, size, 2 * width):
        edges[:, pos] = 255
        edges[pos, :] = 255
    # The gradient profile still carries the true period, so the profile and
    # autocorrelation estimators put the fundamental among the candidates.
    profiles = mesh.compute_gradient_profiles(grey)

    mesh_off, _ = mesh.compute_mesh_from_edges(
        edges,
        mesh_config=MeshConfig(validate_width=False),
        profiles=profiles,
        score_image=grey,
    )
    mesh_on, _ = mesh.compute_mesh_from_edges(
        edges,
        mesh_config=MeshConfig(validate_width=True),
        profiles=profiles,
        score_image=grey,
    )
    # Disabled: trusts the Hough gap median -> ~cells/2 cells.
    assert len(mesh_off[0]) - 1 <= cells // 2 + 1
    # Enabled: corrected to the true width -> ~cells cells.
    assert len(mesh_on[0]) - 1 >= cells - 1


def test_select_pixel_width_context_gated_by_debug_flag():
    """The +-1/double/half neighborhood is scored only for the debug output:
    without debug_context just the eligible candidates are scored."""
    width, cells = 10, 8
    grey = _checker_grey(cells, width)
    size = cells * width
    mesh_initial = ([0, size - 1], [0, size - 1])

    _, plain = mesh.select_pixel_width(grey, mesh_initial, None, [width], MeshConfig())
    assert set(plain) == {width}  # no neighbors when debug is off

    _, expanded = mesh.select_pixel_width(
        grey, mesh_initial, None, [width], MeshConfig(), debug_context=True
    )
    assert set(plain) <= set(expanded)
    # +-1, double and half neighbors are added purely for the debug output.
    assert {width - 1, width + 1, 2 * width, width // 2} <= set(expanded)


def test_width_scores_debug_output(tmp_path):
    """width_scores.txt is written when validating with an output dir."""
    width, cells = 10, 8
    grey = _checker_grey(cells, width)
    size = cells * width
    edges = np.zeros((size, size), dtype=np.uint8)
    profile = np.zeros(size)
    profile[width::width] = 100.0

    mesh.compute_mesh_from_edges(
        edges,
        intermediate_dir=tmp_path,
        mesh_config=MeshConfig(),
        profiles=(profile, profile),
        score_image=grey,
    )
    content = (tmp_path / "width_scores.txt").read_text()
    assert "<- chosen" in content


@pytest.mark.parametrize("shape", [(2, 1, 4), (2, 4)])
def test_detect_mesh_lines_handles_both_hough_shapes(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
):
    """HoughLinesP returns (N, 1, 4) on OpenCV 4 but (N, 4) on OpenCV 5.
    detect_mesh_lines must parse either without crashing (regression test for
    the ``hough_lines[:, 0]`` unpack error on OpenCV 5)."""
    # One vertical line at x=30, one horizontal line at y=50.
    lines = np.array([[30, 0, 30, 99], [0, 50, 99, 50]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(mesh.cv2, "HoughLinesP", lambda *args, **kwargs: lines)

    edges = np.zeros((100, 100), dtype=np.uint8)
    lines_x, lines_y = mesh.detect_mesh_lines(edges, MeshConfig())

    assert 30 in lines_x
    assert 50 in lines_y
