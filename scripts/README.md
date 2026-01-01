# Pixel Art Generation Scripts

Scripts for generating pixel art using AI models.

## ppa-gen

Generate pixel art images using OpenAI's gpt-image-1.5 API and automatically pixelate them using the proper-pixel-art library.

### Setup

#### Install Dependencies

```bash
uv sync --extra scripts
```

#### Configure Environment

1. [Create a new API key](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root
3. Add your API key to `.env`:
4. `OPENAI_API_KEY=sk-your-api-key-here`

### Usage

#### Basic usage

```bash
uv run ppa-gen --prompt "A 16 bit cute pixel art cat"
```

#### With Additional Options

```bash
uv run ppa-gen \
  --prompt "A 16 bit pixel art robot character with a transparent background" \
  --scale-result 10 \
  --transparent \
  --n 2
```

### Command Line Options

#### OpenAI API Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt` | str | (required) | Text description for image generation |
| `--size` | str | 1024x1024 | Image size: '1024x1024', '1024x1536', or '1536x1024' |
| `--n` | int | 1 | Number of images to generate (1-10) |

#### Pixelation Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-c`, `--colors` | int | None | Number of colors (1-256). Omit to preserve all colors |
| `-s`, `--scale-result` | int | 1 | Width of each pixel in output image |
| `-t`, `--transparent` | flag | False | Produce transparent background |
| `-w`, `--pixel-width` | int | None | Width of pixels in input (auto-detected if omitted) |
| `-u`, `--initial-upscale` | int | 2 | Initial upscale factor for mesh detection |

#### Output Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-o`, `--output-dir` | path | '.' | Directory for generated images |

### Output Files

Generated files are named with timestamps, ending with `original.png` for original AI-generated image and `pixelated.png` for pixelated version.

### Tips for Best Results

- Ask for a "16 bit pixel art" for pixel art images that are more aligned to a grid.
- Ask for a transparent background when generating characters to make the background transparent.

### Example

```uv run ppa-gen --prompt "A 16 bit pixel art blob with a transparent background"```

<table align="center" width="100%">
  <tr>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/blob.png" style="width:100%;" />
      <br><small>Noisy, High Resolution</small>
    </td>
    <td width="33%">
      <img src="https://raw.githubusercontent.com/KennethJAllen/proper-pixel-art/main/assets/blob/result.png" style="width:100%;" />
      <br><small>True Pixel Resolution</small>
    </td>
  </tr>
</table>

See the main project README for more examples.