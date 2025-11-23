"""Visual output tests"""

from pathlib import Path
from PIL import Image
from proper_pixel_art import pixelate


def test_pixelate_pngs(pixelate_png_test_params: dict[str, dict]) -> None:
    """Test that pixelate algorithm runs without error for png images and generates output."""
    output_dir = Path.cwd() / "tests" / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    for name, params in pixelate_png_test_params.items():
        pixel_dir = output_dir / str(name)
        pixel_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(params["path"])

        # Pixelate
        result = pixelate.pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=pixel_dir,
        )

        # Save the result
        result_path = pixel_dir / "result.png"
        result.save(result_path)

        # Verify the output exists and has a width and height
        assert result_path.exists(), f"Output file not created for {name}"
        assert result.width > 0 and result.height > 0, f"Invalid dimensions for {name}"

        print(f"Generated output for {name}: {result_path}")
    print(
        (
            "Successfully generated all .png test images.\n"
            f"Manually inspect the results in {output_dir} to verify pixelation quality."
        )
    )
