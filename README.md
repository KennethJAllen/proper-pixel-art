# Proper Pixel Art

by [Kenneth Allen](https://www.kennethallenmath.com/)

[![PyPI version](https://img.shields.io/pypi/v/proper-pixel-art.svg)](https://pypi.org/project/proper-pixel-art/)
[![Python versions](https://img.shields.io/pypi/pyversions/proper-pixel-art.svg)](https://pypi.org/project/proper-pixel-art/)
[![CI](https://github.com/KennethJAllen/proper-pixel-art/actions/workflows/ci.yml/badge.svg)](https://github.com/KennethJAllen/proper-pixel-art/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/proper-pixel-art)](https://pepy.tech/project/proper-pixel-art)
[![License: MIT](https://img.shields.io/github/license/KennethJAllen/proper-pixel-art.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kennethallen/proper-pixel-art)

<table align="center" width="100%">
  <tr>
    <td width="47%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/mountain.png" style="width:100%;" />
      <br><small>Noisy, high resolution</small>
    </td>
    <td width="6%" align="center" valign="middle">
      <h1>→</h1>
    </td>
    <td width="47%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/mountain/result.png" style="width:100%;" />
      <br><small>Clean, true-resolution pixel art</small>
    </td>
  </tr>
</table>

## Summary

Converts noisy, high-resolution pixel-art-style images (from generative models or low-quality web uploads) into clean, true-resolution assets. Such images often have a non-uniform grid and random artifacts, so standard downsampling fails — the usual alternatives are naive downscaling or redrawing the asset pixel by pixel. This tool automates the recovery instead.

## Contents

- [Installation](#installation)
  - [Install From PyPI](#install-from-pypi)
  - [Install from source](#install-from-source)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [CLI](#cli)
  - [Use Without Cloning](#use-without-cloning)
  - [Python API](#python-api)
  - [Configuration file](#configuration-file)
- [Examples](#examples)
- [Real Images To Pixel Art](#real-images-to-pixel-art)
- [Algorithm](#algorithm)

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

First, obtain a source pixel-art-style image (e.g. from a generative model such as OpenAI's `gpt-image-2`, or a web upload of pixel art).

> The examples below assume you installed via `pip install` or `uv add` (commands are on your `PATH`). If you installed from source with `uv sync`, prefix each command with `uv run` (e.g. `uv run ppa ...`).

### Web Interface

Try it live in your browser, no install required, on [Hugging Face Spaces](https://huggingface.co/spaces/kennethallen/proper-pixel-art).

To run the same interface locally:

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
| `--config` `<path>`               | YAML config file of pixelation parameters. Flags passed explicitly override values in the file. (default: none) |
| `--intermediate-dir` `<path>`     | Directory to save images visualizing intermediate algorithm steps. Useful for development. (default: none) |

#### Example

```bash
ppa assets/blob/blob.png -c 16 -s 25
```

Note: `--colors` is the parameter most likely to need tuning. See the option table above.

### Use Without Cloning

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

These mirror the CLI options above.

- `image` : `PIL.Image.Image` — the image to pixelate.
- `num_colors` : `int` — colors in result (1-256), or 0 to skip quantization. Most likely to need tuning.
- `initial_upscale_factor` : `int` — upscale the input first; may help detect lines.
- `scale_result` : `int` — upscale the result; 1 = no scaling.
- `transparent_background` : `bool` — if True, flood-fill each corner with transparent alpha.
- `intermediate_dir` : `Path | None` — save visualizations of intermediate steps (for development).
- `pixel_width` : `int` — pixel width in the input, or 0 to detect automatically.
- `config` : `PixelateConfig | None` — a bundle of *every* tunable parameter, including the deeper mesh-detection (Canny, Hough, line clustering) and color (alpha/transparency thresholds, quantization method, color binning) settings not exposed as direct arguments. Load one with `PixelateConfig.from_yaml(path)`. Explicit arguments override matching values in `config`.

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

Here's a step-by-step overview, applied to this GPT-4o-generated blob:

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
