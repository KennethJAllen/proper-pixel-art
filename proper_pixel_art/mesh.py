from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from proper_pixel_art import utils

def close_edges(edges: np.ndarray, kernel_size: int = 10) -> np.ndarray:
    """
    Apply a morphological closing to fill small gaps in edge map.
    """
    # Use a rectangular kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def cluster_lines(lines: list[int], threshold: int = 4) -> list[int]:
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

def detect_grid_lines(edges: np.ndarray,
                      hough_rho: float = 1.0,
                      hough_theta_rad: float = np.deg2rad(1),
                      hough_threshold: int = 100,
                      hough_min_line_len: int = 50,
                      hough_max_line_gap: int = 10,
                      angle_threshold_deg = 15
                     ) -> tuple[list[int], list[int]]:
    """
    - Use Hough line transformation to detect the pixel edges.
    - Only keep lines that are close to vertical or horizontal
    - Cluster the lines so they aren't too close 
    Return:
    - two lists: x-coordinates (vertical lines) and y-coordinates (horizontal lines)
    """
    hough_lines = cv2.HoughLinesP(edges,
                                  hough_rho,
                                  hough_theta_rad,
                                  hough_threshold,
                                  minLineLength=hough_min_line_len,
                                  maxLineGap=hough_max_line_gap)

    height, width = edges.shape
    # Include the sides of the image in lines since they aren't detected by the Hough transform
    lines_x, lines_y = [0, width-1], [0, height-1]
    if hough_lines is None:
        return lines_x, lines_y

    # Loop over all detected lines, only keep the ones that are close to verticle or horizontal
    for x1, y1, x2, y2 in hough_lines[:,0]:
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.arctan2(dy, dx))
        # vertical if angle > 90- threshold, horizontal if angle < threshold
        if angle > np.deg2rad(90-angle_threshold_deg):
            lines_x.append(round((x1 + x2)/2))
        elif angle < np.deg2rad(angle_threshold_deg):
            lines_y.append(round((y1 + y2)/2))

    # Finally cluster the lines so they aren't too close to each other
    clustered_lines_x = cluster_lines(lines_x)
    clustered_lines_y = cluster_lines(lines_y)
    return clustered_lines_x, clustered_lines_y

def get_pixel_width(lines_x: list[int], lines_y: list[int], trim_outlier_fraction: float = 0.2) -> int:
    """
    Takes lists of line coordinates in x and y direction, and outlier fraction.
    Returns the predicted pixel width by filtering outliers and taking the median.
    We assume that the grid spacing is equal in box x and y direction,
    which is why dx and dy are concatenated.

    The resulting width does not have to be perfect because the color of the pixels
    are detemined by which color is mostly in the corresponding cells.

    This method could be generalized to cases when the pixel size in the x direction
    is different from the y direction, then the width of each direction
    would have to be calculated separately.
    """
    dx = np.diff(lines_x)
    dy = np.diff(lines_y)
    gaps = np.concatenate((dx, dy))

    # Filter lower and upper percentile
    low = np.percentile(gaps, 100 * trim_outlier_fraction)
    hi = np.percentile(gaps, 100 * (1 - trim_outlier_fraction))
    middle = gaps[(gaps >= low) & (gaps <= hi)]
    if len(middle) == 0:
        # fallback to median of all gaps
        middle = gaps

    return np.median(middle)

def homogenize_lines(lines: list[int], pixel_width: int) -> list[int]:
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
        section_lines = [line_start + int(n*section_pixel_width) for n in range(num_pixels)]
        # Replace the start index in completed lines with list of new line coordinates
        # Everything will be unpacked after to maintain indexes
        complete_lines[index] = section_lines

    complete_lines = [line for sublist in complete_lines for line in sublist]
    # Add last line back in because it was excluded earlier
    complete_lines.append(lines[-1])

    return complete_lines

def compute_mesh(
        img: Image.Image,
        canny_thresholds: tuple[int] = (50, 200),
        closure_kernel_size: int = 8,
        output_dir: Path | None = None,
        pixel_width: int | None = None
        ) -> tuple[list[int], list[int]]:
    """
    Finds grid lines of a high resolution noisy image.
    - Uses Canny edge detector to find vertical and horizontal edges
    - Closes small gets between edges with morphological closing
    - Uses Hough transform to detect pixel edge lines
    - Finds true width of pixels from line differences
    - Completes mesh by filling in gaps between identified lines
    inputs:
        img_path: Path to image
        canny_thresholds: thresholds 1 and 2 for canny edge detection algorithm
        closure_kernel_size: Kernel size for the morphological closure
        output_dir (optional): If set, saves images of steps in algorithm to dir
    
    output:
        Returns tuple of two lists of integer coordinates:
        - mesh_x: Coordinates of pixel mesh on the x-axis
        - mesh_y: Coordinates of pixel mesh on the y-axis

    Note: this could even be generalized to detect grid lines that
    have been distorted via linear transformation.
    """
    # Crop border and zero out mostly transparent pixels from alpha
    cropped_img = utils.crop_border(img, num_pixels=2)
    grey_img = utils.clamp_alpha(cropped_img, mode='L')

    # Find edges using Canny edge detection
    edges = cv2.Canny(np.array(grey_img), *canny_thresholds)

    # Close small gaps in edges with morphological closing
    closed_edges = close_edges(edges, kernel_size=closure_kernel_size)

    # Use Hough transform to detect the pixel lines
    lines_x, lines_y = detect_grid_lines(closed_edges)

    if pixel_width is None:
        # Get the true width of the pixels
        pixel_width = get_pixel_width(lines_x, lines_y)

    # Fill in the gaps between the lines to complete the grid
    mesh_x = homogenize_lines(lines_x, pixel_width)
    mesh_y = homogenize_lines(lines_y, pixel_width)

    if output_dir is not None:
        edges_img = Image.fromarray(edges, mode="L")
        edges_img.save(output_dir / "edges.png")
        closed_edges_img = Image.fromarray(closed_edges, mode="L")
        closed_edges_img.save(output_dir / "closed_edges.png")

        img_with_lines = utils.overlay_grid_lines(img, lines_x, lines_y)
        img_with_lines.save(output_dir / "lines.png")
        img_with_completed_lines = utils.overlay_grid_lines(img, mesh_x, mesh_y)
        img_with_completed_lines.save(output_dir / "mesh.png")

    return mesh_x, mesh_y

def compute_mesh_with_scaling(
        img: Image.Image,
        upsample_factor: int,
        output_dir: Path | None = None,
        pixel_width: int | None = None
        ) -> tuple[tuple[list[int], list[int]], Image.Image]:
    """
    Try to compute the mesh on an upsampled image.
    If that yields only the trivial boundary lines, fall back to the original.
    """
    upsampled_img = utils.scale_img(img, upsample_factor)
    mesh_lines = compute_mesh(
        upsampled_img, output_dir=output_dir, pixel_width=pixel_width
    )
    if not _is_trivial_mesh(mesh_lines):
        return mesh_lines, upsampled_img

    # If no mesh is found, then use the original image instead.
    fallback_mesh_lines = compute_mesh(
        img, output_dir=output_dir, pixel_width=pixel_width
    )
    return fallback_mesh_lines, img

def _is_trivial_mesh(img_mesh: tuple[list[int], list[int]]) -> bool:
    """
    Returns True if no lines have been identified when computing the mesh.
    That is, the points in mesh_x and mesh_y conist of the left, right, and top, bottom
    of the image respectively.
    """
    return len(img_mesh[0]) == 2 and len(img_mesh[1]) == 2

def main():
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    mesh_x, mesh_y = compute_mesh(img)
    print(mesh_x, mesh_y)

if __name__ == "__main__":
    main()
