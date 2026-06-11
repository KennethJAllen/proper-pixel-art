"""Main functions for pixelating an image with the pixelate function"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from proper_pixel_art import colors, mesh, utils
from proper_pixel_art.config import ColorConfig, PixelateConfig
from proper_pixel_art.utils import Mesh


@dataclass
class CellMap:
    """Precomputed per-pixel cell assignment for a fixed mesh and image size.

    Building this once and reusing it across frames is what makes video
    pixelation fast: every per-frame step reduces to ``np.bincount`` over
    ``cell_id``.
    """

    cell_id: np.ndarray  # (H, W) int32; -1 marks pixels outside the mesh bounds
    n_cells: int  # n_rows * n_cols
    out_shape: tuple[int, int]  # (n_rows, n_cols) of the downsampled image
    cell_sizes: np.ndarray  # (n_cells,) pixel count per cell


def build_cell_map(mesh_lines: Mesh, image_hw: tuple[int, int]) -> CellMap:
    """
    Map every pixel of an ``image_hw``-sized image to its mesh cell index.

    Cell (j, i) covers pixels with x in [lines_x[i], lines_x[i+1]) and
    y in [lines_y[j], lines_y[j+1]); pixels outside [lines[0], lines[-1])
    get cell_id -1, matching the slicing semantics of the per-cell loop this
    replaced.
    """
    lines_x, lines_y = mesh_lines
    height, width = image_hw
    lx = np.asarray(lines_x, dtype=np.int64)
    ly = np.asarray(lines_y, dtype=np.int64)
    n_cols, n_rows = len(lines_x) - 1, len(lines_y) - 1

    coords_x = np.arange(width)
    coords_y = np.arange(height)
    xs = (np.searchsorted(lx, coords_x, side="right") - 1).astype(np.int32)
    ys = (np.searchsorted(ly, coords_y, side="right") - 1).astype(np.int32)
    xs[(coords_x < lx[0]) | (coords_x >= lx[-1])] = -1
    ys[(coords_y < ly[0]) | (coords_y >= ly[-1])] = -1

    cell_id = ys[:, None] * n_cols + xs[None, :]
    cell_id[(ys[:, None] < 0) | (xs[None, :] < 0)] = -1

    n_cells = n_rows * n_cols
    cell_sizes = np.bincount(cell_id[cell_id >= 0], minlength=n_cells)
    return CellMap(
        cell_id=cell_id,
        n_cells=n_cells,
        out_shape=(n_rows, n_cols),
        cell_sizes=cell_sizes,
    )


def _transparent_cells(
    cid: np.ndarray,
    opaque_mask: np.ndarray,
    cell_map: CellMap,
    majority_fraction: float,
) -> np.ndarray:
    """Boolean per-cell mask of cells that should be fully transparent.

    A cell is transparent when at least ``majority_fraction`` of its pixels are
    transparent (same rule as ``colors._is_majority_transparent``). Empty cells
    are transparent too.
    """
    opaque_counts = np.bincount(cid[opaque_mask], minlength=cell_map.n_cells)
    return opaque_counts <= cell_map.cell_sizes * (1 - majority_fraction)


def downsample_quantized(
    palette_idx: np.ndarray,
    alpha: np.ndarray | None,
    cell_map: CellMap,
    palette_rgb: np.ndarray,
    color_config: ColorConfig,
) -> np.ndarray:
    """
    Per-cell mode of palette indices, vectorized via a joint bincount over
    (cell, color). Returns an (n_rows, n_cols, 4) uint8 RGBA array.

    Mode ties resolve to the lowest palette index (Counter-based code resolved
    them by scan order); outputs only differ on exact count ties.
    """
    valid = cell_map.cell_id >= 0
    cid = cell_map.cell_id[valid].astype(np.int64)
    idx = palette_idx[valid].astype(np.int64)

    n_colors = len(palette_rgb)
    counts = np.bincount(
        cid * n_colors + idx, minlength=cell_map.n_cells * n_colors
    ).reshape(cell_map.n_cells, n_colors)
    mode = counts.argmax(axis=1)

    out = np.zeros((cell_map.n_cells, 4), dtype=np.uint8)
    out[:, :3] = palette_rgb[mode]
    out[:, 3] = 255

    if alpha is not None:
        transparent = _transparent_cells(
            cid,
            alpha[valid] >= color_config.alpha_threshold,
            cell_map,
            color_config.transparency_majority_fraction,
        )
        out[transparent] = 0
    out[cell_map.cell_sizes == 0] = 0
    return out.reshape(*cell_map.out_shape, 4)


def downsample_binned(
    rgba: np.ndarray,
    cell_map: CellMap,
    color_config: ColorConfig,
) -> np.ndarray:
    """
    Vectorized port of the per-cell ``colors.get_cell_color_skip_quantization``:
    per-cell dominant color via offset binning (two half-shifted bin grids, the
    grid with the larger dominant bin wins) followed by the per-channel median
    of the dominant bin. Returns an (n_rows, n_cols, 4) uint8 RGBA array.
    """
    alpha_threshold = color_config.alpha_threshold
    majority_fraction = color_config.transparency_majority_fraction
    bin_size = color_config.bin_size

    valid = cell_map.cell_id >= 0
    cid = cell_map.cell_id[valid].astype(np.int64)
    pixels = rgba[valid]

    opaque_mask = pixels[:, 3] >= alpha_threshold
    opaque_counts = np.bincount(cid[opaque_mask], minlength=cell_map.n_cells)
    transparent = _transparent_cells(cid, opaque_mask, cell_map, majority_fraction)

    out = np.zeros((cell_map.n_cells, 4), dtype=np.uint8)
    out[~transparent, 3] = 255

    # Binning only applies to cells with > 3 opaque pixels; small cells fall
    # back to colors.dominant_rgb_by_binning below to keep its <= 3 edge cases.
    small = ~transparent & (opaque_counts <= 3)
    active = opaque_mask & (~transparent & ~small)[cid]
    cid_a = cid[active]
    rgb = pixels[active, :3].astype(np.int64)

    if len(cid_a) > 0:
        num_bins = 255 // bin_size + 1
        num_bin_combos = num_bins**3
        bins1 = rgb // bin_size
        bins2 = np.minimum(rgb + bin_size // 2, 255) // bin_size
        idx1 = bins1[:, 0] * num_bins**2 + bins1[:, 1] * num_bins + bins1[:, 2]
        idx2 = bins2[:, 0] * num_bins**2 + bins2[:, 1] * num_bins + bins2[:, 2]

        counts1 = np.bincount(
            cid_a * num_bin_combos + idx1,
            minlength=cell_map.n_cells * num_bin_combos,
        ).reshape(cell_map.n_cells, num_bin_combos)
        counts2 = np.bincount(
            cid_a * num_bin_combos + idx2,
            minlength=cell_map.n_cells * num_bin_combos,
        ).reshape(cell_map.n_cells, num_bin_combos)
        dominant1, max1 = counts1.argmax(axis=1), counts1.max(axis=1)
        dominant2, max2 = counts2.argmax(axis=1), counts2.max(axis=1)
        use_grid1 = max1 >= max2

        in_dominant_bin = np.where(
            use_grid1[cid_a], idx1 == dominant1[cid_a], idx2 == dominant2[cid_a]
        )
        cid_m = cid_a[in_dominant_bin]
        dominant_rgb = rgb[in_dominant_bin].astype(np.uint16)

        # Grouped per-channel median: sort within each cell group, then average
        # the two middle elements with floor division — reproduces
        # np.median(...).astype(np.uint8) exactly.
        group_counts = np.bincount(cid_m, minlength=cell_map.n_cells)
        group_starts = np.concatenate(([0], np.cumsum(group_counts)[:-1]))
        has_pixels = group_counts > 0
        lo = group_starts[has_pixels] + (group_counts[has_pixels] - 1) // 2
        hi = group_starts[has_pixels] + group_counts[has_pixels] // 2
        for channel in range(3):
            order = np.lexsort((dominant_rgb[:, channel], cid_m))
            sorted_vals = dominant_rgb[order, channel]
            out[has_pixels, channel] = (
                (sorted_vals[lo] + sorted_vals[hi]) // 2
            ).astype(np.uint8)

    # Fix-up for cells with 1-3 opaque pixels (rare): keep the exact special
    # cases of the scalar helper.
    small_cells = np.flatnonzero(small)
    if len(small_cells) > 0:
        is_small = np.zeros(cell_map.n_cells, dtype=bool)
        is_small[small_cells] = True
        selected = opaque_mask & is_small[cid]
        cid_s = cid[selected]
        rgb_s = pixels[selected, :3]
        order = np.argsort(cid_s, kind="stable")
        cid_s, rgb_s = cid_s[order], rgb_s[order]
        boundaries = np.searchsorted(cid_s, small_cells)
        sizes = opaque_counts[small_cells]
        for cell, start, size in zip(small_cells, boundaries, sizes, strict=True):
            out[cell, :3] = colors.dominant_rgb_by_binning(
                rgb_s[start : start + size], bin_size=bin_size
            )

    return out.reshape(*cell_map.out_shape, 4)


def downsample(
    image: Image.Image,
    mesh_lines: Mesh,
    skip_quantization: bool = False,
    original_alpha: np.ndarray | None = None,
    color_config: ColorConfig | None = None,
    cell_map: CellMap | None = None,
) -> Image.Image:
    """
    Collapse each mesh cell to a single representative color, returning one
    output pixel per cell as an RGBA image.

    A cell becomes transparent when enough of its pixels are transparent (see
    ``color_config.transparency_majority_fraction``). When ``skip_quantization``
    is True the alpha comes from ``image`` itself; otherwise it comes from
    ``original_alpha`` (the quantized image has no usable alpha of its own).

    Args:
        image: The image to downsample (P or RGB if quantized, RGBA if not).
        mesh_lines: (x_lines, y_lines) defining the pixel grid.
        skip_quantization: Preserve original colors instead of quantized ones.
        original_alpha: Alpha channel from the original image, used only when
            skip_quantization is False to carry transparency through quantization.
        cell_map: Precomputed cell map for the mesh and image size; pass it when
            downsampling many same-sized images (video frames) to avoid
            rebuilding it per call.
    """
    color_config = color_config or ColorConfig()
    if cell_map is None:
        cell_map = build_cell_map(mesh_lines, (image.height, image.width))

    if skip_quantization:
        rgba = np.asarray(image.convert("RGBA"))
        out = downsample_binned(rgba, cell_map, color_config)
    else:
        if image.mode == "P":
            palette_rgb = np.array(image.getpalette(), dtype=np.uint8).reshape(-1, 3)
            palette_idx = np.asarray(image)
        else:
            arr = np.asarray(image.convert("RGB"))
            palette_rgb, inverse = np.unique(
                arr.reshape(-1, 3), axis=0, return_inverse=True
            )
            palette_idx = inverse.reshape(arr.shape[:2])
        out = downsample_quantized(
            palette_idx, original_alpha, cell_map, palette_rgb, color_config
        )

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
