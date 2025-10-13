from pathlib import Path
from PIL import Image
from proper_pixel_art import mesh

def test_mesh():
    """
    Checks that the mesh calculated for the blob image is non-trivial.
    """
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    mesh_x, mesh_y = mesh.compute_mesh(img)
    assert(len(mesh_x)) > 2
    assert(len(mesh_y)) > 2
