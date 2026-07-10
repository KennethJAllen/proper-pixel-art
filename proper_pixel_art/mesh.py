"""Handles mesh detection from pixel art style images"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from proper_pixel_art import colors, utils
from proper_pixel_art.config import MeshConfig
from proper_pixel_art.utils import Lines, Mesh

# Shared defaults so the helpers below don't re-declare literals; MeshConfig
# holds the canonical values.
_DEFAULTS = MeshConfig()


def close_edges(
    edges: np.ndarray, kernel_size: int = _DEFAULTS.closure_kernel_size
) -> np.ndarray:
    """
    Apply a morphological closing to fill small gaps in edge map.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


def cluster_lines(lines: Lines, threshold: int = _DEFAULTS.cluster_threshold) -> Lines:
    """Remove lines that are too close to each other by clustering near values"""
    if not lines:
        return []
    lines = sorted(lines)
    clusters = [[lines[0]]]
    for p in lines[1:]:
        if abs(p - clusters[-1][-1]) <= threshold:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    # use the median of each cluster
    return [int(np.median(cluster)) for cluster in clusters]


def detect_grid_lines(edges: np.ndarray, mesh_config: MeshConfig | None = None) -> Mesh:
    """
    - Use Hough line transformation to detect the pixel edges.
    - Only keep lines that are close to vertical or horizontal
    - Cluster the lines so they aren't too close
    Return:
    - two lists: x-coordinates (vertical lines) and y-coordinates (horizontal lines)
    """
    mesh_config = mesh_config or MeshConfig()
    hough = mesh_config.hough
    hough_lines = cv2.HoughLinesP(
        edges,
        hough.rho,
        hough.theta_rad,
        hough.threshold,
        minLineLength=hough.min_line_len,
        maxLineGap=hough.max_line_gap,
    )

    height, width = edges.shape
    # Include the sides of the image in lines since they aren't detected by the Hough transform
    lines_x, lines_y = [0, width - 1], [0, height - 1]
    if hough_lines is None:
        return lines_x, lines_y

    angle_threshold_deg = mesh_config.angle_threshold_deg
    # Loop over all detected lines, only keep the ones that are close to vertical or horizontal.
    for x1, y1, x2, y2 in hough_lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.arctan2(dy, dx))
        # vertical if angle > 90-threshold, horizontal if angle < threshold
        if angle > np.deg2rad(90 - angle_threshold_deg):
            lines_x.append(round((x1 + x2) / 2))
        elif angle < np.deg2rad(angle_threshold_deg):
            lines_y.append(round((y1 + y2) / 2))

    # Finally cluster the lines so they aren't too close to each other
    clustered_lines_x = cluster_lines(lines_x, threshold=mesh_config.cluster_threshold)
    clustered_lines_y = cluster_lines(lines_y, threshold=mesh_config.cluster_threshold)
    return clustered_lines_x, clustered_lines_y


def get_pixel_width(
    line_collection: list[Lines],
    trim_outlier_fraction: float = _DEFAULTS.trim_outlier_fraction,
) -> int:
    """
    Takes list of line coordinates, and outlier fraction.
    Returns the predicted pixel width by filtering outliers and taking the median.
    We assume the grid spacing is consistent across all sets of lines,
    then all grid spacings are concatenated.

    The resulting width does not have to be perfect because the color of the pixels
    is determined by which color is mostly in the corresponding cells.

    This method could be generalized to cases when the pixel size in the x direction
    is different from the y direction, then the width of each direction
    would have to be calculated separately.
    """
    all_gaps = []
    for lines in line_collection:
        gap = np.diff(lines)
        all_gaps.append(gap)
    gaps = np.concatenate(all_gaps)

    # Filter lower and upper percentile
    low = np.percentile(gaps, 100 * trim_outlier_fraction)
    hi = np.percentile(gaps, 100 * (1 - trim_outlier_fraction))
    middle = gaps[(gaps >= low) & (gaps <= hi)]
    if len(middle) == 0:
        # fallback to median of all gaps
        middle = gaps

    return int(np.round(np.median(middle)))


def _snap_line_to_profile(
    profile: np.ndarray,
    target: int,
    window: int,
    min_strength: float,
    low: int,
    high: int,
) -> int:
    """Snap one interpolated line to the strongest profile peak near ``target``.

    The result is confined to ``[low, high]`` so lines stay strictly increasing
    and inside their section. Falls back to the (clamped) even-spacing position
    when no peak exceeds ``min_strength``.
    """
    low = max(low, 0)
    high = min(high, profile.size - 1)
    if low > high:
        return target
    clamped_target = min(max(target, low), high)
    start = max(target - window, low)
    end = min(target + window, high)
    if start > end:
        return clamped_target
    segment = profile[start : end + 1]
    if float(segment.max()) > min_strength:
        return start + int(np.argmax(segment))
    return clamped_target


def _split_oversized_gaps(
    lines: Lines,
    pixel_width: int,
    profile: np.ndarray,
    window: int,
    min_strength: float,
    split_leftover_fraction: float,
) -> Lines:
    """Insert extra (snapped) lines into any gap left wider than
    ``(1 + split_leftover_fraction) * pixel_width``.

    Even spacing never produces such gaps, but snapping can displace two
    neighboring lines apart, leaving a cell that clearly spans multiple
    pixels. Existing lines are never moved. Repeats until no gap qualifies
    (bounded: every pass strictly adds lines).
    """
    while True:
        result: Lines = []
        for left, right in zip(lines[:-1], lines[1:], strict=True):
            result.append(left)
            gap = right - left
            num_pixels = int(gap / pixel_width + (1.0 - split_leftover_fraction))
            if num_pixels < 2:
                continue
            gap_pixel_width = gap / num_pixels
            for n in range(1, num_pixels):
                target = _snap_line_to_profile(
                    profile,
                    left + int(n * gap_pixel_width),
                    window,
                    min_strength,
                    low=result[-1] + 1,
                    high=right - 1,
                )
                if target > result[-1]:
                    result.append(target)
        result.append(lines[-1])
        if len(result) == len(lines):
            return result
        lines = result


def homogenize_lines(
    lines: Lines,
    pixel_width: int,
    profile: np.ndarray | None = None,
    mesh_config: MeshConfig | None = None,
) -> Lines:
    """
    Given sorted line coords and pixel width,
    further partition those line coordinates to approximately even spacing.

    When ``profile`` (a 1-D gradient projection profile along the same axis)
    is provided and ``mesh_config.snap_lines`` is enabled, each interpolated
    line is snapped to the strongest nearby profile peak, so the mesh tracks
    grids that drift between detected lines. Detected (anchor) lines are never
    moved, and gaps that snapping leaves wider than
    ``(1 + split_leftover_fraction)`` pixels are split with extra lines.
    With ``profile=None`` the plain even-spacing behavior applies.
    """
    mesh_config = mesh_config or MeshConfig()
    snap = profile is not None and mesh_config.snap_lines and profile.size > 0
    if snap:
        window = max(
            round(mesh_config.snap_search_window_ratio * pixel_width),
            mesh_config.snap_min_search_window,
        )
        min_strength = mesh_config.snap_strength_threshold * float(profile.mean())

    section_widths = np.diff(lines)
    complete_lines = lines[:-1]
    for index, section_width in enumerate(section_widths):
        # Get number of pixels to partition section width into: a leftover of
        # at least split_leftover_fraction of a pixel width earns an extra line
        # (0.5 is plain rounding; lower splits oversized cells more eagerly).
        num_pixels = int(
            section_width / pixel_width + (1.0 - mesh_config.split_leftover_fraction)
        )
        section_pixel_width = 0 if num_pixels == 0 else section_width / num_pixels
        line_start = lines[index]
        # n == 0 is the detected anchor line itself; only interior lines snap
        section_lines = [line_start] if num_pixels > 0 else []
        for n in range(1, num_pixels):
            target = line_start + int(n * section_pixel_width)
            if snap:
                target = _snap_line_to_profile(
                    profile,
                    target,
                    window,
                    min_strength,
                    low=section_lines[-1] + 1,
                    high=lines[index + 1] - 1,
                )
            section_lines.append(target)
        # Replace the start index in completed lines with list of new line coordinates
        # Everything will be unpacked after to maintain indexes
        complete_lines[index] = section_lines

    complete_lines = [line for sublist in complete_lines for line in sublist]
    # Add last line back in because it was excluded earlier
    complete_lines.append(lines[-1])

    if snap and len(complete_lines) >= 2:
        complete_lines = _split_oversized_gaps(
            complete_lines,
            pixel_width,
            profile,
            window,
            min_strength,
            mesh_config.split_leftover_fraction,
        )

    return complete_lines


def compute_grey(
    img: Image.Image,
    mesh_config: MeshConfig | None = None,
) -> np.ndarray:
    """
    Crop the border and return the alpha-clamped grayscale array that edge
    detection and gradient profiles are computed from, so both live in the
    same coordinate space.
    """
    mesh_config = mesh_config or MeshConfig()
    # Crop border and zero out mostly transparent pixels from alpha.
    # The grayscale clamp here only suppresses transparent-edge noise before edge
    # detection, so it intentionally uses the default alpha threshold rather than
    # color_config.alpha_threshold (which governs the final downsample step).
    cropped_img = utils.crop_border(img, num_pixels=mesh_config.crop_border_pixels)
    grey_img = colors.clamp_alpha(cropped_img, mode="L")
    return np.array(grey_img)


def compute_gradient_profiles(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1-D gradient projection profiles from a grayscale image.

    Central-difference gradient magnitude summed along each axis: the x-profile
    is the per-column sum of horizontal gradient magnitude (peaks at vertical
    pixel boundaries) and the y-profile the per-row sum of vertical gradient
    magnitude. Ends are zero-padded so profile indices match image coordinates.

    Returns:
        (profile_x, profile_y) as float64 arrays of length width and height
    """
    g = grey.astype(np.float64)
    height, width = g.shape
    profile_x = np.zeros(width)
    profile_y = np.zeros(height)
    if width >= 3:
        profile_x[1:-1] = np.abs(g[:, 2:] - g[:, :-2]).sum(axis=0)
    if height >= 3:
        profile_y[1:-1] = np.abs(g[2:, :] - g[:-2, :]).sum(axis=1)
    return profile_x, profile_y


def estimate_width_from_profile(
    profile: np.ndarray,
    peak_threshold_fraction: float = 0.2,
    min_peak_distance: int = 3,
) -> int | None:
    """
    Estimate the pixel width as the median spacing between local peaks of a
    gradient projection profile. Peaks below ``peak_threshold_fraction`` of the
    profile maximum are ignored and peaks closer than ``min_peak_distance``
    are merged. Returns None when the profile has no usable peak structure
    (e.g. a flat profile).
    """
    if profile.size < 3:
        return None
    max_val = float(profile.max())
    if max_val <= 0:
        return None
    threshold = peak_threshold_fraction * max_val
    interior = profile[1:-1]
    is_peak = (
        (interior > threshold) & (interior > profile[:-2]) & (interior > profile[2:])
    )
    peaks = np.flatnonzero(is_peak) + 1
    if len(peaks) < 2:
        return None
    merged = [peaks[0]]
    for peak in peaks[1:]:
        if peak - merged[-1] >= min_peak_distance:
            merged.append(peak)
    if len(merged) < 2:
        return None
    return int(round(float(np.median(np.diff(merged)))))


def compute_edge_map(
    img: Image.Image,
    mesh_config: MeshConfig | None = None,
) -> np.ndarray:
    """
    Compute a closed edge map from an image.
    Crops border, clamps alpha, runs Canny edge detection, and applies
    morphological closing to fill small gaps.

    Args:
        img: RGBA PIL image
        mesh_config: Tunable mesh-detection parameters (Canny, closing, ...)

    Returns:
        Binary edge map as numpy array (uint8, values 0 or 255)
    """
    mesh_config = mesh_config or MeshConfig()
    grey = compute_grey(img, mesh_config=mesh_config)
    edges = cv2.Canny(grey, *mesh_config.canny_thresholds)
    closed_edges = close_edges(edges, kernel_size=mesh_config.closure_kernel_size)
    return closed_edges


def compute_mesh_from_edges(
    closed_edges: np.ndarray,
    pixel_width: int | None = None,
    output_dir: Path | None = None,
    original_img: Image.Image | None = None,
    mesh_config: MeshConfig | None = None,
    profiles: tuple[np.ndarray, np.ndarray] | None = None,
) -> Mesh:
    """
    Compute mesh from a closed edge map using Hough transform and line homogenization.

    Args:
        closed_edges: Binary edge map (uint8, values 0 or 255)
        pixel_width: If set, skips automatic pixel width detection
        output_dir: If set, saves debug images (requires original_img)
        original_img: Original image for debug visualization overlays
        mesh_config: Tunable mesh-detection parameters (Hough, clustering, ...)
        profiles: Optional (profile_x, profile_y) 1-D projection profiles in
            edge-map coordinates. Used to snap interpolated lines to gradient
            evidence and as a pixel-width fallback when Hough finds few lines.

    Returns:
        The pixel mesh (mesh_x, mesh_y)
    """
    mesh_config = mesh_config or MeshConfig()
    mesh_initial = detect_grid_lines(closed_edges, mesh_config)

    if pixel_width is None:
        pixel_width = get_pixel_width(
            mesh_initial, trim_outlier_fraction=mesh_config.trim_outlier_fraction
        )
        # With too few Hough line gaps the median gap is unreliable; the
        # profiles aggregate all edge evidence, so prefer their estimate.
        num_gaps = sum(max(len(lines) - 1, 0) for lines in mesh_initial)
        if profiles is not None and num_gaps < mesh_config.profile_width_min_gaps:
            estimates = [
                estimate
                for estimate in map(estimate_width_from_profile, profiles)
                if estimate is not None
            ]
            if estimates:
                pixel_width = int(round(float(np.mean(estimates))))

    lines_x, lines_y = mesh_initial
    profile_x, profile_y = profiles if profiles is not None else (None, None)
    mesh_x = homogenize_lines(
        lines_x, pixel_width, profile=profile_x, mesh_config=mesh_config
    )
    mesh_y = homogenize_lines(
        lines_y, pixel_width, profile=profile_y, mesh_config=mesh_config
    )
    mesh_final = mesh_x, mesh_y

    if output_dir is not None:
        edges_img = Image.fromarray(closed_edges, mode="L")
        edges_img.save(output_dir / "closed_edges.png")

        if original_img is not None:
            img_with_lines = utils.overlay_grid_lines(original_img, mesh_initial)
            img_with_lines.save(output_dir / "lines.png")
            img_with_completed_lines = utils.overlay_grid_lines(
                original_img, mesh_final
            )
            img_with_completed_lines.save(output_dir / "mesh.png")

    return mesh_final


def compute_mesh(
    img: Image.Image,
    output_dir: Path | None = None,
    pixel_width: int | None = None,
    mesh_config: MeshConfig | None = None,
) -> Mesh:
    """
    Finds grid lines of a high resolution noisy image.
    - Uses Canny edge detector to find vertical and horizontal edges
    - Closes small gaps between edges with morphological closing
    - Uses Hough transform to detect pixel edge lines
    - Finds true width of pixels from line differences
    - Completes mesh by filling in gaps between identified lines
    inputs:
        img: The image to compute the mesh
        output_dir (optional): If set, saves images of steps in algorithm to dir
        mesh_config: Tunable mesh-detection parameters (Canny, Hough, clustering, ...)

    output:
        Returns The pixel mesh: mesh_x, mesh_y
            tuple of two lists of integer coordinates ():
        - mesh_x: Coordinates of pixel mesh on the x-axis
        - mesh_y: Coordinates of pixel mesh on the y-axis

    Note: this could even be generalized to detect grid lines that
    have been distorted via linear transformation.
    """
    mesh_config = mesh_config or MeshConfig()
    grey = compute_grey(img, mesh_config=mesh_config)
    edges = cv2.Canny(grey, *mesh_config.canny_thresholds)
    closed_edges = close_edges(edges, kernel_size=mesh_config.closure_kernel_size)
    profiles = compute_gradient_profiles(grey)

    # Save raw edges debug image (before closing) if output_dir is set
    if output_dir is not None:
        edges_img = Image.fromarray(edges, mode="L")
        edges_img.save(output_dir / "edges.png")

    return compute_mesh_from_edges(
        closed_edges,
        pixel_width=pixel_width,
        output_dir=output_dir,
        original_img=img,
        mesh_config=mesh_config,
        profiles=profiles,
    )


def compute_mesh_with_scaling(
    img: Image.Image,
    upscale_factor: int = 2,
    output_dir: Path | None = None,
    pixel_width: int | None = None,
    mesh_config: MeshConfig | None = None,
) -> tuple[Mesh, int]:
    """
    Try to compute the mesh on on the image.
    First upscale the image with a given upscale factor
    If that yields only the trivial mesh lines, try to compute the mesh on
    the original image instead.
    Returns the mesh line coordinates and the scale factor used
    """
    upscaled_img = utils.scale_img(img, upscale_factor)
    mesh_lines = compute_mesh(
        upscaled_img,
        output_dir=output_dir,
        pixel_width=pixel_width,
        mesh_config=mesh_config,
    )
    if not is_trivial_mesh(mesh_lines):
        return mesh_lines, upscale_factor

    # If no mesh is found, then use the original image instead.
    fallback_mesh_lines = compute_mesh(
        img, output_dir=output_dir, pixel_width=pixel_width, mesh_config=mesh_config
    )
    return fallback_mesh_lines, 1


def is_trivial_mesh(img_mesh: Mesh) -> bool:
    """
    Returns True if no lines have been identified when computing the mesh.
    That is, the points in mesh_x and mesh_y consist of the left, right, and top, bottom
    of the image respectively.
    """
    x_num = len(img_mesh[0])
    y_num = len(img_mesh[1])
    return x_num in (2, 3) and y_num in (2, 3)
