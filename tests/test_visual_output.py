"""Visual output tests"""
from pathlib import Path
import pytest
from PIL import Image
from proper_pixel_art import pixelate

def get_test_assets():
    """Find all test assets following the pattern assets/{name}/{name}.png"""
    assets_dir = Path("assets")
    test_assets = []

    for asset_dir in assets_dir.iterdir():
        if asset_dir.is_dir():
            expected_file = asset_dir / f"{asset_dir.name}.png"
            if expected_file.exists():
                test_assets.append((asset_dir.name, expected_file))

    return test_assets

@pytest.mark.parametrize("asset_name,asset_path", get_test_assets())
def test_pixelate_output(asset_name, asset_path):
    """Test that pixelate algorithm runs without error and generates output."""
    # Create output directory for this test asset
    output_dir = Path("tests/outputs") / asset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the image
    img = Image.open(asset_path)

    # Define test parameters
    test_params = {
        "ash": {"num_colors": 16, "result_scale": 5, "transparent_background": False},
        "bat": {"num_colors": 16, "result_scale": 5, "transparent_background": True},
        "blob": {"num_colors": 16, "result_scale": 5, "transparent_background": True},
        "demon": {"num_colors": 64, "result_scale": 5, "transparent_background": True},
        "mountain": {"num_colors": 64, "result_scale": 5, "transparent_background": False},
        "pumpkin": {"num_colors": 32, "result_scale": 5, "transparent_background": False},
        "anchor": {"num_colors": 16, "result_scale": 5, "transparent_background": True}
    }

    params = test_params.get(asset_name, {"num_colors": 16, "result_scale": 20})

    # Run the pixelate algorithm
    result = pixelate.pixelate(
        img,
        num_colors=params["num_colors"],
        scale_result=params["result_scale"],
        transparent_background=params["transparent_background"],
        intermediate_dir=output_dir,
    )

    # Save the result
    result_path = output_dir / "result.png"
    result.save(result_path)

    # Verify the output exists and has reasonable properties
    assert result_path.exists(), f"Output file not created for {asset_name}"
    assert result.width > 0 and result.height > 0, f"Invalid dimensions for {asset_name}"

    print(f"Generated output for {asset_name}: {result_path}")


def test_output_summary():
    """Print a summary of all generated outputs for manual inspection."""
    outputs_dir = Path("tests/outputs")
    if not outputs_dir.exists():
        return

    print("\n" + "="*50)
    print("VISUAL OUTPUT SUMMARY")
    print("="*50)
    print("Generated test outputs for manual inspection:")

    for asset_dir in outputs_dir.iterdir():
        if asset_dir.is_dir():
            result_file = asset_dir / "result.png"
            if result_file.exists():
                print(f"  {asset_dir.name}: {result_file}")

    print("\nPlease manually inspect these outputs to verify quality.")
    print("="*50)
