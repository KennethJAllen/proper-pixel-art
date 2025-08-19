from pathlib import Path
from PIL import Image
from proper_pixel_art import colors, mesh, utils

def pixelate(
        image: Image.Image,
        num_colors: int = 16,
        initial_upsample_factor: int = 2,
        pixel_size: int | None = None,
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
    - pixel_size:
        Upsample result by pixel_size factor after algorithm is complete if not None.
    - upsample_factor:
        Upsample original image by this factor. It may help detect lines.
    - transparent_background:
        If True, floos fills each corner of the result with transparent alpha.
    - intermediate_dir:
        directory to save images visualizing intermediate steps.
    
    Returns the result
    """
    rgba = image.convert("RGBA")

    mesh_lines, scaled_img = mesh.compute_mesh_with_scaling(
        rgba,
        initial_upsample_factor,
        output_dir=intermediate_dir,
        pixel_width=pixel_width
    )

    paletted_img = colors.palette_img(scaled_img, num_colors=num_colors)

    result = colors.downsample(paletted_img, mesh_lines, transparent_background=transparent_background)
    if pixel_size is not None:
        result = utils.scale_img(result, int(pixel_size))

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
            pixel_size = 20,
            num_colors = num_colors,
            transparent_background = False,
            intermediate_dir = output_dir,
            )
        result.save(output_dir / "result.png")

if __name__ == "__main__":
    main()
