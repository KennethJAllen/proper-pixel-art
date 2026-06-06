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
    # Loop over all detected lines, only keep the ones that are close to vertical or horizontal
    for x1, y1, x2, y2 in hough_lines[:, 0]:
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


def homogenize_lines(lines: Lines, pixel_width: int) -> Lines:
    """
    Given sorted line coords and pixel width,
    further partition those line coordinates to approximately even spacing.
    """
    section_widths = np.diff(lines)
    complete_lines = lines[:-1]
    for index, section_width in enumerate(section_widths):
        # Get number of pixels to partition section width into
        num_pixels = int(np.round(section_width / pixel_width))
        if num_pixels == 0:
            section_pixel_width = 0
        else:
            section_pixel_width = section_width / num_pixels
        line_start = lines[index]
        section_lines = [
            line_start + int(n * section_pixel_width) for n in range(num_pixels)
        ]
        # Replace the start index in completed lines with list of new line coordinates
        # Everything will be unpacked after to maintain indexes
        complete_lines[index] = section_lines

    complete_lines = [line for sublist in complete_lines for line in sublist]
    # Add last line back in because it was excluded earlier
    complete_lines.append(lines[-1])

    return complete_lines


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
    # Crop border and zero out mostly transparent pixels from alpha.
    # The grayscale clamp here only suppresses transparent-edge noise before edge
    # detection, so it intentionally uses the default alpha threshold rather than
    # color_config.alpha_threshold (which governs the final downsample step).
    cropped_img = utils.crop_border(img, num_pixels=mesh_config.crop_border_pixels)
    grey_img = colors.clamp_alpha(cropped_img, mode="L")

    # Find edges using Canny edge detection
    edges = cv2.Canny(np.array(grey_img), *mesh_config.canny_thresholds)

    # Close small gaps in edges with morphological closing
    closed_edges = close_edges(edges, kernel_size=mesh_config.closure_kernel_size)

    # Use Hough transform to get an initial estimate for pixel lines
    mesh_initial = detect_grid_lines(closed_edges, mesh_config)

    if pixel_width is None:
        # Get the true width of the pixels if a value hasn't been provided
        pixel_width = get_pixel_width(
            mesh_initial, trim_outlier_fraction=mesh_config.trim_outlier_fraction
        )

    # Fill in the gaps between the lines to complete the grid
    lines_x, lines_y = mesh_initial
    mesh_x = homogenize_lines(lines_x, pixel_width)
    mesh_y = homogenize_lines(lines_y, pixel_width)
    mesh_final = mesh_x, mesh_y

    if output_dir is not None:
        edges_img = Image.fromarray(edges, mode="L")
        edges_img.save(output_dir / "edges.png")
        closed_edges_img = Image.fromarray(closed_edges, mode="L")
        closed_edges_img.save(output_dir / "closed_edges.png")

        img_with_lines = utils.overlay_grid_lines(img, mesh_initial)
        img_with_lines.save(output_dir / "lines.png")
        img_with_completed_lines = utils.overlay_grid_lines(img, mesh_final)
        img_with_completed_lines.save(output_dir / "mesh.png")

    return mesh_final


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
    if not _is_trivial_mesh(mesh_lines):
        return mesh_lines, upscale_factor

    # If no mesh is found, then use the original image instead.
    fallback_mesh_lines = compute_mesh(
        img, output_dir=output_dir, pixel_width=pixel_width, mesh_config=mesh_config
    )
    return fallback_mesh_lines, 1


def _is_trivial_mesh(img_mesh: Mesh) -> bool:
    """
    Returns True if no lines have been identified when computing the mesh.
    That is, the points in mesh_x and mesh_y consist of the left, right, and top, bottom
    of the image respectively.
    """
    x_num = len(img_mesh[0])
    y_num = len(img_mesh[1])
    return x_num in (2, 3) and y_num in (2, 3)
