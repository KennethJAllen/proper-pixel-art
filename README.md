# Proper Pixel Art

## Summary

Converts noisy, high resolution pixel-art style images produced by generative models or sourced from low-quality web uploads to clean, true-resolution pixel-art assets.

### Challenges

Generative pixel-art style images are noisy and high resolution, often with a non-uniform grid and random artifacts. Standard downsampling techniques do not work. The current approach is to either use naive downscaling techniques or manually re-create the asset pixel by pixel.

This tool addressed both of these issues by automating the process of recovering true-resolution pixel-art assets.

## Hugging Face Spaces

Try the tool on [Hugging Face Spaces](https://huggingface.co/spaces/kennethallen/proper-pixel-art).

## Installation

### Install From PyPI

```bash
pip install proper-pixel-art  # CLI and Python API
pip install "proper-pixel-art[web]"  #  Include the local web UI
```

Or with `uv`:

```bash
uv add proper-pixel-art  # CLI and Python API
uv add proper-pixel-art --extra web  # Include the local web UI
```

### Install from source

```bash
git clone git@github.com:KennethJAllen/proper-pixel-art.git
cd proper-pixel-art
uv sync --extra web
```

## Usage

First, obtain a source pixel-art-style image (e.g. a pixel-art-style image from a generative model such as OpenAI's `gpt-image-2` or a web upload of pixel-art).

> The examples below assume you installed via `pip install` or `uv add` (commands are on your `PATH`). If you installed from source with `uv sync`, prefix each command with `uv run` (e.g. `uv run ppa ...`).

### Web Interface

Opens a browser interface where you can upload an image and adjust settings interactively.

```bash
ppa-web
# Opens http://127.0.0.1:7860
```

### CLI

```bash
ppa <input_path> -o <output_path> -c <num_colors> -s <result_scale> [-t]
```

#### Options

| Option                            | Description                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| INPUT (positional)                | Source file in pixel-art-style                                                                            |
| `-o`, `--output` `<path>`         | Output directory or file path for result. (default: '.')                                                  |
| `-c`, `--colors` `<int>`          | Number of colors for output (1-256). Use 0 to skip quantization and preserve all colors. May need to try a few different values. (default 0) |
| `-s`, `--scale-result` `<int>`    | Width/height of each "pixel" in the output. 1 = no scaling. (default: 1)                                  |
| `-t`, `--transparent` `<bool>`    | Output with transparent background. (default: off)                                                        |
| `-u`, `--initial-upscale` `<int>` | Initial image upscale factor. Increasing this may help detect pixel edges. (default 2)                    |
| `-w`, `--pixel-width` `<int>`     | Width of the pixels in the input image. Use 0 to determine it automatically. (default: 0)                 |

#### Example

```bash
ppa assets/blob/blob.png -c 16 -s 25
```

Note: `num_colors` is the parameter most likely to need tuning. Try values like 8, 16, 32, or 64 if the result doesn't look right, or use 0 to skip quantization.

### Use Without Cloning

UV offers options for users who want to run the tool without cloning

#### Web Interface (without cloning)

```bash
uvx --from "proper-pixel-art[web]" ppa-web
```

#### CLI (without cloning)

```bash
uvx --from "proper-pixel-art" ppa <input_path>
```

### Python API

For Python developers who want to integrate this tool into their own code.

```python
from PIL import Image
from proper_pixel_art import pixelate

image = Image.open('path/to/input.png')
result = pixelate(image, num_colors=16)
result.save('path/to/output.png')
```

#### Parameters

- `image` : `PIL.Image.Image`

  - A PIL image to pixelate.

- `num_colors` : `int`

  - The number of colors in result (1-256). Use 0 to skip quantization and preserve all colors.
  - May need to try a few values if the colors don't look right.
  - 8, 16, 32, or 64 typically works for quantized output.

- `initial_upscale_factor` : `int`

  - Upscale initial image. This may help detect lines.

- `scale_result` : `int`

  - Upscale result after algorithm is complete. 1 = no scaling.

- `transparent_background` : `bool`
  - If True, flood fills each corner of the result with transparent alpha.

- `intermediate_dir` : `Path | None`
  - Directory to save images visualizing intermediate steps of algorithm. Useful for development.

- `pixel_width` : `int`
  - Width of the pixels in the input image. Use 0 to determine it automatically. It may be helpful to increase this parameter if not enough pixel edges are being detected.

- `config` : `PixelateConfig | None`
  - A bundle of *every* tunable parameter, including the deeper mesh-detection (Canny, Hough, line clustering) and color (alpha/transparency thresholds, quantization method, color binning) settings that are not exposed as direct arguments. Load one from a YAML file with `PixelateConfig.from_yaml(path)`. Any explicit argument above overrides the matching value in `config`.

#### Returns

A PIL image with true pixel resolution and quantized colors.

### Configuration file

All tunable parameters can be collected in a YAML file so you can fine-tune the algorithm without changing code. See [`config.example.yaml`](config.example.yaml) for the full list of keys with their defaults. Any key you omit falls back to the default, so partial files are fine.

```python
from PIL import Image
from proper_pixel_art import pixelate
from proper_pixel_art.config import PixelateConfig

config = PixelateConfig.from_yaml('config.yaml')
result = pixelate(Image.open('input.png'), config=config)
```

From the CLI, pass `--config`. Flags given explicitly override values from the file:

```bash
ppa input.png --config config.yaml      # use the file
ppa input.png --config config.yaml -c 8 # but override num_colors to 8
```

## Examples

The algorithm is robust. It performs well for images that are already approximately aligned to a grid.

Here are a few examples. A mesh is computed, where each cell corresponds to one pixel.

### Bat

- Generated by GPT-4o.

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/bat/bat.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/bat/mesh.png" style="width:100%;" />
      <br><small>Mesh</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/bat/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

### Ash

- Screenshot from Google images of Pokemon asset.

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/ash/ash.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/ash/mesh.png" style="width:100%;" />
      <br><small>Mesh</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/ash/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

### Demon

- Original image generated by GPT-4o.

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/demon/demon.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/demon/mesh.png" style="width:100%;" />
      <br><small>Mesh</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/demon/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

### Pumpkin

- Screenshot from Google Images of Stardew Valley asset. This is an adversarial example as the source image is both low quality and the object is round.

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/pumpkin/pumpkin.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/pumpkin/mesh.png" style="width:100%;" />
      <br><small>Mesh</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/pumpkin/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

## Real Images To Pixel Art

- This tool can also be used to convert real images to pixel art by first requesting a pixelated version of the original image from GPT-4o, then using the tool to get the true pixel-resolution image.

- Consider this image of a mountain

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/real.jpg" width="50%" alt="Original mountain"/>

- Here are the results of first requesting a pixelated version of the mountain, then using the tool to get a true resolution pixel art version.

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/mountain.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/mesh.png" style="width:100%;" />
      <br><small>Mesh</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

## Algorithm

- The main algorithm solves these challenges. Here is a high level overview. We will apply it step by step on this example image of blob pixel art that was generated from GPT-4o.

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/blob.png" width="80%" alt="blob"/>

- Note that this image is high resolution and noisy.

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/zoom.png" width="80%" alt="The blob is noisy."/>

1) Trim the edges of the image and zero out pixels with more than 50% alpha.
    - This is to work around some issues with models such as GPT-4o not giving a perfectly transparent background.

2) Upscale by a factor of 2 using nearest neighbor.
    - This can help identify the correct pixel mesh.

3) Find edges of the pixel art using [Canny edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/edges.png" width="80%" alt="blob edges"/>

4) Close small gaps in edges with a [morphological closing](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/closed_edges.png" width="80%" alt="blob closed edges"/>

5) Take the [probabilistic Hough transform](https://docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html) to get the coordinates of lines in the detected edges. Only keep lines that are close to vertical or horizontal giving some grid coordinates. Cluster lines that are closeby together.

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/lines.png" width="80%" alt="blob lines"/>

6) Find the grid spacing by filtering outliers and taking the median of the spacings, then complete the mesh.

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/mesh.png" width="80%" alt="blob mesh"/>

7) Quantize the original image to a small number of colors (see the `num_colors` tuning note above).

8) In each cell specified by the mesh, choose the most common color in the cell as the color for the pixel. Recreate the original image with one pixel per cell.

    - Result upscaled by a factor of $20 \times$ using nearest neighbor.

<img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/result.png" width="80%" alt="blob pixelated"/>
