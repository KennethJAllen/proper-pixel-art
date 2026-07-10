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
    mesh_x, mesh_y = mesh.compute_mesh(img)
    assert (len(mesh_x)) > 2
    assert (len(mesh_y)) > 2


@pytest.mark.parametrize("shape", [(2, 1, 4), (2, 4)])
def test_detect_grid_lines_handles_both_hough_shapes(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
):
    """HoughLinesP returns (N, 1, 4) on OpenCV 4 but (N, 4) on OpenCV 5.
    detect_grid_lines must parse either without crashing (regression test for
    the ``hough_lines[:, 0]`` unpack error on OpenCV 5)."""
    # One vertical line at x=30, one horizontal line at y=50.
    lines = np.array([[30, 0, 30, 99], [0, 50, 99, 50]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(mesh.cv2, "HoughLinesP", lambda *args, **kwargs: lines)

    edges = np.zeros((100, 100), dtype=np.uint8)
    lines_x, lines_y = mesh.detect_grid_lines(edges, MeshConfig())

    assert 30 in lines_x
    assert 50 in lines_y
