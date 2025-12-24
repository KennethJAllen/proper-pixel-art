"""Main functions for pixelating an image with the pixelate function"""

from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.utils import Mesh


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    transparent_background: bool = False,
    skip_quantization: bool = False,
) -> Image.Image:
    """
    Downsample the image by looping over each cell in mesh and
    selecting a representative color for each cell.

    Args:
        image: The image to downsample
        mesh_lines: Tuple of (x_lines, y_lines) defining the pixel grid
        transparent_background: If True, make background transparent
        skip_quantization: If True, preserve alpha channel and use RGBA processing
    """
    lines_x, lines_y = mesh_lines
    h_new, w_new = len(lines_y) - 1, len(lines_x) - 1

    if skip_quantization:
        # RGBA path: preserve alpha, filter transparent pixels when computing cell color
        rgba = image.convert("RGBA")
        rgba_array = np.array(rgba)
        out = np.zeros((h_new, w_new, 4), dtype=np.uint8)

        for j in range(h_new):
            for i in range(w_new):
                x0, x1 = lines_x[i], lines_x[i + 1]
                y0, y1 = lines_y[j], lines_y[j + 1]
                cell = rgba_array[y0:y1, x0:x1]
                out[j, i] = colors.get_cell_color_skip_quantization(cell)

        result = Image.fromarray(out, mode="RGBA")
    else:
        # RGB path: quantized image
        rgb = image.convert("RGB")
        rgb_array = np.array(rgb)
        out = np.zeros((h_new, w_new, 3), dtype=np.uint8)

        for j in range(h_new):
            for i in range(w_new):
                x0, x1 = lines_x[i], lines_x[i + 1]
                y0, y1 = lines_y[j], lines_y[j + 1]
                cell = rgb_array[y0:y1, x0:x1]
                out[j, i] = colors.get_cell_color(cell)

        result = Image.fromarray(out, mode="RGB")
        if transparent_background:
            result = colors.make_background_transparent(result)

    return result


def pixelate(
    image: Image.Image,
    num_colors: int | None = None,
    initial_upscale_factor: int = 2,
    scale_result: int | None = None,
    transparent_background: bool = False,
    intermediate_dir: Path | None = None,
    pixel_width: int | None = None,
) -> Image.Image:
    """
    Computes the true resolution pixel art image.
    inputs:
    - image:
        A PIL image to pixelate.
    - num_colors:
        The number of colors to use when quantizing the image.
        Use None to skip quantization and preserve all colors.
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
        pixel_width=pixel_width,
    )

    # Process colors: either quantize or preserve original (with alpha)
    skip_quantization = num_colors is None
    if skip_quantization:
        # Preserve alpha: pass RGBA directly, let downsample filter by alpha
        processed_img = image_rgba
    else:
        processed_img = colors.palette_img(
            image_rgba, num_colors=num_colors, output_dir=intermediate_dir
        )

    # Scale the processed image to match the dimensions for the calculated mesh
    scaled_img = utils.scale_img(processed_img, upscale_factor)

    # Downsample the image to 1 pixel per cell in the mesh
    result = downsample(
        scaled_img,
        mesh_lines,
        transparent_background=transparent_background,
        skip_quantization=skip_quantization,
    )

    # upscale the result if scale_result is set to an integer
    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result
