from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
from PIL.Image import Quantize
import numpy as np
from proper_pixel_art import utils

def get_cell_color(cell_pixels: np.ndarray) -> tuple[int,int,int]:
    """
    cell_pixels: shape (height_cell, width_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in the cell_pixels block.
    """
    # flatten to tuple of pixel values
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    cell_color = Counter(flat).most_common(1)[0][0]
    return cell_color

def palette_img(
        img: Image.Image,
        num_colors: int = 16,
        quantize_method: int = Quantize.MAXCOVERAGE,
        output_dir: Path | None = None) -> Image.Image:
    """
    Discretizes the colors in the image img to at most num_colors.
    Saves the quantized image to output_dir if not None.
    Returns the color pallete of the image.

    The maximum coverage algorithm is used by default as the quantization method.
    Emperically this algorithm proivdes the best results overall, although
    for some examples num_colors needs to be chosen very large even when the
    image has a small number of actual colors. In these instances, Quantize.FASTOCTREE
    can work instead.

    If the colors of the result don't look right, try increasing num_colors.
    """
    img_rgb = utils.clamp_alpha(img, mode='RGB')
    quantized_img = img_rgb.quantize(colors=num_colors, method=quantize_method, dither=Image.Dither.NONE)
    if output_dir is not None:
        quantized_img.save(output_dir / "quantized_original.png")
    return quantized_img

# def apply_palette(img: Image.Image, palette: Image.Image, output_dir: Path | None = None) -> Image.Image:
#     """
#     Applies the palette from a previously quantized image.
#     """
#     img_rgb = utils.clamp_alpha(img, mode='RGB')
#     paletted_img = img_rgb.quantize(palette=palette)
#     if output_dir is not None:
#         paletted_img.save(output_dir / "quantized_scaled.png")
#     return paletted_img

def make_background_transparent(image: Image.Image) -> Image.Image:
    """
    Make image background transparent by
    flood filling each corner with full alpha.
    """
    im = image.convert("RGBA")
    fill_color = (0, 0, 0, 0) # Full alpha
    corners = [(0, 0), (im.width-1, 0), (0, im.height-1), (im.width-1, im.height-1)]
    for corner in corners:
        ImageDraw.floodfill(im, corner, fill_color)
    return im

def main():
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    paletted = palette_img(img)
    paletted.show()

if __name__ == "__main__":
    main()
