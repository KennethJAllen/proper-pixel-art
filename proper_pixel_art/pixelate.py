"""Main functions for pixelating an image with the pixelate function"""

from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.utils import Mesh


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    skip_quantization: bool = False,
    original_alpha: np.ndarray | None = None,
) -> Image.Image:
    """
    Downsample the image by looping over each cell in mesh and
    selecting a representative color for each cell.

    Transparency handling:
    - If >=50% of pixels in a cell are transparent, the entire cell becomes transparent (0,0,0,0)
    - For skip_quantization=True, uses alpha channel from image directly
    - For skip_quantization=False with original_alpha provided,
      uses alpha channel from original_alpha array

    Args:
        image: The image to downsample (RGB if quantized, RGBA if not)
        mesh_lines: Tuple of (x_lines, y_lines) defining the pixel grid
        skip_quantization: If True, use histogram-based color selection for RGBA
        original_alpha: Optional numpy array of alpha channel values from original
                       image. Used to preserve transparency through quantization.
                       Only used when skip_quantization=False.

    Returns:
        RGBA image with downsampled pixels
    """
    lines_x, lines_y = mesh_lines
    height_result, width_result = len(lines_y) - 1, len(lines_x) - 1

    # Single conversion at start
    if skip_quantization:
        img_array = np.array(image.convert("RGBA"))
    else:
        img_array = np.array(image.convert("RGB"))

    # Output is RGBA to support transparency
    out = np.zeros((height_result, width_result, 4), dtype=np.uint8)

    for j, i in product(range(height_result), range(width_result)):
        x0, x1 = lines_x[i], lines_x[i + 1]
        y0, y1 = lines_y[j], lines_y[j + 1]
        cell = img_array[y0:y1, x0:x1]

        if skip_quantization:
            # Skip quantization and use original colros
            out[j, i] = colors.get_cell_color_skip_quantization(cell)
        else:
            # Get color from quantized cell, considering alpha from original
            if original_alpha is not None:
                cell_alpha = original_alpha[y0:y1, x0:x1]
                out[j, i] = colors.get_cell_color_with_alpha(cell, cell_alpha)
            else:
                out[j, i] = colors.get_opaque_cell_color(cell)

    return Image.fromarray(out, mode="RGBA")


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
        If True, makes pixels matching the most common boundary color transparent.
        Applied after preserving original image transparency.
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

    # Extract and scale alpha channel for quantized path
    scaled_alpha_array = (
        None
        if skip_quantization
        else colors.extract_and_scale_alpha(image_rgba, upscale_factor)
    )

    # Downsample the image to 1 pixel per cell in the mesh
    result = downsample(
        scaled_img,
        mesh_lines,
        skip_quantization=skip_quantization,
        original_alpha=scaled_alpha_array,
    )

    if transparent_background:
        result = colors.make_background_transparent(result)

    if scale_result is not None:
        result = utils.scale_img(result, int(scale_result))

    return result
