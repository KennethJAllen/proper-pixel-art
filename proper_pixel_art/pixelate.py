"""Main functions for pixelating an image with the pixelate function"""

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.config import ColorConfig, PixelateConfig
from proper_pixel_art.utils import Mesh


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    skip_quantization: bool = False,
    original_alpha: np.ndarray | None = None,
    color_config: ColorConfig | None = None,
) -> Image.Image:
    """
    Collapse each mesh cell to a single representative color, returning one
    output pixel per cell as an RGBA image.

    A cell becomes transparent when enough of its pixels are transparent (see
    ``color_config.transparency_majority_fraction``). When ``skip_quantization``
    is True the alpha comes from ``image`` itself; otherwise it comes from
    ``original_alpha`` (the quantized image has no usable alpha of its own).

    Args:
        image: The image to downsample (RGB if quantized, RGBA if not).
        mesh_lines: (x_lines, y_lines) defining the pixel grid.
        skip_quantization: Preserve original colors instead of quantized ones.
        original_alpha: Alpha channel from the original image, used only when
            skip_quantization is False to carry transparency through quantization.
    """
    color_config = color_config or ColorConfig()
    lines_x, lines_y = mesh_lines
    height_result, width_result = len(lines_y) - 1, len(lines_x) - 1

    # RGBA when skipping quantization (we read alpha here); RGB otherwise.
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
            out[j, i] = colors.get_cell_color_skip_quantization(
                cell,
                alpha_threshold=color_config.alpha_threshold,
                majority_fraction=color_config.transparency_majority_fraction,
                bin_size=color_config.bin_size,
            )
        else:
            # Get color from quantized cell, considering alpha from original
            if original_alpha is not None:
                cell_alpha = original_alpha[y0:y1, x0:x1]
                out[j, i] = colors.get_cell_color_with_alpha(
                    cell,
                    cell_alpha,
                    alpha_threshold=color_config.alpha_threshold,
                    majority_fraction=color_config.transparency_majority_fraction,
                )
            else:
                out[j, i] = colors.get_opaque_cell_color(cell)

    return Image.fromarray(out, mode="RGBA")


def pixelate(
    image: Image.Image,
    num_colors: int | None = None,
    initial_upscale_factor: int | None = None,
    scale_result: int | None = None,
    transparent_background: bool | None = None,
    intermediate_dir: Path | None = None,
    pixel_width: int | None = None,
    config: PixelateConfig | None = None,
) -> Image.Image:
    """
    Computes the true resolution pixel art image.

    Every parameter below defaults to ``None``, meaning "not provided" — the
    value is taken from ``config`` (or the built-in defaults). Pass a concrete
    value to override the config.

    inputs:
    - image:
        A PIL image to pixelate.
    - num_colors:
        The number of colors to use when quantizing the image.
        Use 0 to skip quantization and preserve all colors.
        This is an important parameter to tune,
        if it is too high, pixels that should be the same color will be different colors
        if it is too low, pixels that should be different colors will be the same color
    - scale_result:
        Upsample result by scale_result factor after algorithm is complete.
        Use 1 for no scaling.
    - initial_upscale_factor:
        Upsample original image by this factor. It may help detect lines.
    - transparent_background:
        If True, makes pixels matching the most common boundary color transparent.
        Applied after preserving original image transparency.
    - intermediate_dir:
        directory to save images visualizing intermediate steps.
    - pixel_width:
        Width of the pixels in the input image. Use 0 to detect it automatically.
    - config:
        A PixelateConfig bundling every tunable parameter. Load one from YAML
        with PixelateConfig.from_yaml. Any of the explicit arguments above,
        when provided (not None), override the corresponding value in config.

    Returns the true pixelated image.
    """
    # Resolution order: explicit argument > config > built-in defaults.
    cfg = config if config is not None else PixelateConfig()
    overrides = {
        name: value
        for name, value in (
            ("num_colors", num_colors),
            ("initial_upscale_factor", initial_upscale_factor),
            ("scale_result", scale_result),
            ("transparent_background", transparent_background),
            ("pixel_width", pixel_width),
        )
        if value is not None
    }
    if overrides:
        cfg = replace(cfg, **overrides)

    image_rgba = image.convert("RGBA")

    # Calculate the pixel mesh lines
    mesh_lines, upscale_factor = mesh.compute_mesh_with_scaling(
        image_rgba,
        cfg.initial_upscale_factor,
        output_dir=intermediate_dir,
        pixel_width=cfg.pixel_width or None,  # 0 / None -> auto-detect
        mesh_config=cfg.mesh,
    )

    # Process colors: either quantize or preserve original (with alpha)
    skip_quantization = not cfg.num_colors  # 0 / None -> skip
    if skip_quantization:
        # Preserve alpha: pass RGBA directly, let downsample filter by alpha
        processed_img = image_rgba
    else:
        processed_img = colors.palette_img(
            image_rgba,
            num_colors=cfg.num_colors,
            color_config=cfg.colors,
            output_dir=intermediate_dir,
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
        color_config=cfg.colors,
    )

    if cfg.transparent_background:
        result = colors.make_background_transparent(result)

    if (cfg.scale_result or 1) > 1:
        result = utils.scale_img(result, int(cfg.scale_result))

    return result
