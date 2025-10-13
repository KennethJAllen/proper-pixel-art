from pathlib import Path
from PIL import Image
import numpy as np
from proper_pixel_art import colors, mesh, utils

def downsample(image: Image.Image,
               mesh_lines: tuple[list[int], list[int]],
               transparent_background: bool = False) -> Image.Image:
    """
    Downsample the image by looping over each cell in mesh and
    using the most common color as the pixel color.
    If transparent_background is True, flood fill each corner of the image with 0 alpha.
    """
    lines_x, lines_y = mesh_lines
    rgb = image.convert("RGB")
    rgb_array = np.array(rgb)
    h_new, w_new = len(lines_y) - 1, len(lines_x) - 1
    out = np.zeros((h_new, w_new, 3), dtype=np.uint8)

    for j in range(h_new):
        for i in range(w_new):
            x0, x1 = lines_x[i], lines_x[i+1]
            y0, y1 = lines_y[j], lines_y[j+1]
            cell = rgb_array[y0:y1, x0:x1]
            out[j, i] = colors.get_cell_color(cell)

    result = Image.fromarray(out, mode="RGB")
    if transparent_background:
        result = colors.make_background_transparent(result)
    return result

def pixelate(
        image: Image.Image,
        num_colors: int = 16,
        initial_upscale_factor: int = 2,
        scale_result: int | None = None,
        transparent_background: bool = False,
        intermediate_dir: Path | None = None,
        pixel_width: int | None = None
        ) -> Image.Image:
    """
    Computes the true resolution pixel art image.
    inputs:
    - image:
        A PIL image to pixelate.
    - num_colors:
        The number of colors to use when quantizing the image.
        This is an important parameter to tune,
        if it is too high, pixels that should be the same color will be different colors
        if it is too low, pixels that should be different colors will be the same color
    - scale_result:
        Upsample result by scale_result factor after algorithm is complete if not None.
    - initial_upscale_factor:
        Upsample original image by this factor. It may help detect lines.
    - transparent_background:
        If True, flood fills each corner of the result with transparent alpha.
    - intermediate_dir:
        directory to save images visualizing intermediate steps.
    - pixel_width:
        If set, skips the step to automatically identify pixel width and uses this value.

    Returns the true pixelated image.
    """
    image_rgba = image.convert("RGBA")

    # Calculate the pixel mesh lines
    mesh_lines, upscale_factor = mesh.compute_mesh_with_scaling(
        image_rgba,
        initial_upscale_factor,
        output_dir=intermediate_dir,
        pixel_width=pixel_width
    )

    # Calculate the color palette
    paletted_img = colors.palette_img(image_rgba, num_colors=num_colors, output_dir=intermediate_dir)

    # Scale the paletted image to match the dimensions for the calculated mesh
    scaled_paletted_img = utils.scale_img(paletted_img, upscale_factor)

    # Downsample the image to 1 pixel per cell in the mesh
    result = downsample(scaled_paletted_img, mesh_lines, transparent_background=transparent_background)

    # upscale the result if scale_result is set to an integer
    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result

def main():
    data_dir = Path.cwd() / "assets"

    img_paths_and_colors = [
        (data_dir / "blob" / "blob.png", 16),
        (data_dir / "bat" / "bat.png", 16),
        (data_dir / "demon" / "demon.png", 64),
        (data_dir / "ash" / "ash.png", 16),
        (data_dir / "pumpkin" / "pumpkin.png", 32),
        (data_dir / "mountain" / "mountain.png", 64),
        ]

    for img_path, num_colors in img_paths_and_colors:
        output_dir = Path.cwd() / "output" / img_path.stem
        output_dir.mkdir(exist_ok=True, parents=True)
        img = Image.open(img_path)
        result = pixelate(
            img,
            scale_result = 20,
            num_colors = num_colors,
            transparent_background = False,
            intermediate_dir = output_dir,
            )
        result.save(output_dir / "result.png")

if __name__ == "__main__":
    main()
