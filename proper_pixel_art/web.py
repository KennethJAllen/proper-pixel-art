"""Web interface for Proper Pixel Art using Gradio.

A single upload accepts either a still image or a video/GIF; the input's suffix
decides whether it is routed to :func:`pixelate` or :func:`pixelate_video`. The
common tunables plus a selection of advanced mesh/Hough/color knobs are exposed
as native controls (a collapsed "Advanced" accordion holds the deep ones); the
full parameter set is available via the CLI's ``--config`` YAML instead. The
result is previewed as a static image, an animated GIF, or a video player,
matching the produced file.

Launched via ``ppa web`` (see :mod:`proper_pixel_art.cli`).
"""

import base64
import tempfile
from dataclasses import replace
from pathlib import Path

from PIL import Image

from proper_pixel_art.cli import VIDEO_SUFFIXES
from proper_pixel_art.config import PixelateConfig
from proper_pixel_art.image import pixelate

IMG_HEIGHT = 512

# Still-image suffixes accepted by the upload alongside VIDEO_SUFFIXES.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
ACCEPTED_SUFFIXES = sorted(set(IMAGE_SUFFIXES) | set(VIDEO_SUFFIXES))

# LIBIMAGEQUANT is omitted: it raises at quantize time unless Pillow was built
# with libimagequant, so only offer the always-available methods.
QUANTIZE_METHODS = ["MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE"]

# Built-in defaults, read from the dataclasses so the UI never drifts from them.
_DEFAULTS = PixelateConfig()
_MESH = _DEFAULTS.mesh
_HOUGH = _DEFAULTS.mesh.hough
_COLOR = _DEFAULTS.colors


# UI config controls, in the positional order create_demo wires them into
# process(). build_config consumes a dict keyed by these names; tests reuse the
# tuple so their argument ordering can never silently drift from the UI's.
CONFIG_KEYS = (
    "color_method",
    "colors",
    "scale_result",
    "initial_upscale_factor",
    "pixel_width",
    "transparent_background",
    "crop_border_pixels",
    "canny_low",
    "canny_high",
    "closure_kernel_size",
    "cluster_threshold",
    "angle_threshold_deg",
    "trim_outlier_fraction",
    "rho",
    "theta_deg",
    "hough_threshold",
    "min_line_len",
    "max_line_gap",
    "alpha_threshold",
    "transparency_majority_fraction",
    "quantize_method",
    "bin_size",
    "top_colors_limit",
    "thumbnail_w",
    "thumbnail_h",
)


def build_config(values: dict) -> PixelateConfig:
    """Assemble a :class:`PixelateConfig` from the raw UI control values,
    keyed by :data:`CONFIG_KEYS`.

    Gradio sliders/numbers hand back floats; integer-typed config fields are
    cast so downstream indexing/quantization sees real ints. ``from_dict``
    validates the keys and coerces the list fields into tuples.
    """
    return PixelateConfig.from_dict(
        {
            "scale_result": int(values["scale_result"]),
            "initial_upscale_factor": int(values["initial_upscale_factor"]),
            # The UI slider uses 0 for "auto"; the config expects None.
            "pixel_width": int(values["pixel_width"]) or None,
            "transparent_background": bool(values["transparent_background"]),
            "mesh": {
                "crop_border_pixels": int(values["crop_border_pixels"]),
                "canny_thresholds": [
                    int(values["canny_low"]),
                    int(values["canny_high"]),
                ],
                "closure_kernel_size": int(values["closure_kernel_size"]),
                "cluster_threshold": int(values["cluster_threshold"]),
                "angle_threshold_deg": float(values["angle_threshold_deg"]),
                "trim_outlier_fraction": float(values["trim_outlier_fraction"]),
                "hough": {
                    "rho": float(values["rho"]),
                    "theta_deg": float(values["theta_deg"]),
                    "threshold": int(values["hough_threshold"]),
                    "min_line_len": int(values["min_line_len"]),
                    "max_line_gap": int(values["max_line_gap"]),
                },
            },
            "colors": {
                "method": values["color_method"],
                "alpha_threshold": int(values["alpha_threshold"]),
                "transparency_majority_fraction": float(
                    values["transparency_majority_fraction"]
                ),
                "top_colors_limit": int(values["top_colors_limit"]),
                "thumbnail_size": [
                    int(values["thumbnail_w"]),
                    int(values["thumbnail_h"]),
                ],
                "palette": {
                    "num_colors": int(values["colors"]),
                    "quantize_method": values["quantize_method"],
                },
                "dominant": {
                    "bin_size": int(values["bin_size"]),
                },
            },
        }
    )


def _gif_preview_html(path: Path) -> str:
    """Embed a GIF as a base64 data URI so the browser plays the animation.

    ``gr.Image`` re-encodes to a static frame, so an ``<img>`` tag is the
    reliable way to preview the animation. Nearest-neighbor rendering keeps the
    pixels crisp.
    """
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/gif;base64,{b64}" '
        'style="max-width:100%;image-rendering:pixelated;" alt="Pixelated GIF" />'
    )


def _status(message: str) -> str:
    """Format a message for the always-present status indicator."""
    return f"### {message}"


def process(file, output_format: str, sample_frames: float, *config_values):
    """Pixelate the uploaded image/video/GIF and return the four output updates.

    Returns updates for (image, gif-html, video, status) with exactly one of the
    three previews made visible, chosen by the produced file's type. The status
    line always resolves to a "Done" or error message, so a run never ends with
    the page looking unchanged and failures surface instead of vanishing.
    """
    import gradio as gr

    def outputs(img=None, gif=None, video=None, status=""):
        def show(value):
            return (
                gr.update(value=value, visible=True)
                if value is not None
                else gr.update(visible=False)
            )

        return (
            show(img),
            show(gif),
            show(video),
            gr.update(value=status, visible=bool(status)),
        )

    if not file:
        return outputs(status=_status("⚠️ Upload an image, video, or GIF first."))

    input_path = Path(file)
    try:
        config = build_config(dict(zip(CONFIG_KEYS, config_values, strict=True)))

        if input_path.suffix.lower() in VIDEO_SUFFIXES:
            # Deferred import so image runs don't pay the cv2 import cost.
            from proper_pixel_art import video

            video_overrides = {"num_sample_frames": int(sample_frames)}
            if output_format != "Auto":
                video_overrides["output_format"] = output_format
            config = replace(
                config, video=replace(config.video, **video_overrides)
            )
            out_path = video.pixelate_video(
                input_path=input_path,
                output_path=Path(tempfile.mkdtemp()),
                config=config,
            )
            if out_path.suffix.lower() == ".gif":
                return outputs(
                    gif=_gif_preview_html(out_path), status=_status("✅ Done")
                )
            return outputs(video=str(out_path), status=_status("✅ Done"))

        result = pixelate(Image.open(input_path), config=config)
        return outputs(img=result.image, status=_status("✅ Done"))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        return outputs(status=_status(f"❌ Failed: {exc}"))


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    with gr.Blocks(title="Proper Pixel Art") as demo:
        gr.Markdown(
            "# Proper Pixel Art\n"
            "Convert AI-generated pixel art (images, videos, or GIFs) to true "
            "pixel resolution"
        )

        with gr.Row():
            with gr.Column():
                input_file = gr.File(
                    label="Input (image, video, or GIF)",
                    file_types=ACCEPTED_SUFFIXES,
                    file_count="single",
                )
            with gr.Column():
                output_img = gr.Image(
                    type="pil",
                    label="Output",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                    interactive=False,
                    visible=False,
                )
                output_gif = gr.HTML(label="Output", visible=False)
                output_video = gr.Video(
                    label="Output", height=IMG_HEIGHT, interactive=False, visible=False
                )
                status = gr.Markdown(visible=False)

        with gr.Row():
            color_method = gr.Radio(
                choices=["dominant", "palette"],
                value=_COLOR.method,
                label="Color Method (dominant keeps original colors)",
            )
            num_colors = gr.Slider(
                1,
                64,
                value=_COLOR.palette.num_colors,
                step=1,
                label="Colors (palette method only)",
            )
            scale = gr.Slider(
                1, 20, value=_DEFAULTS.scale_result or 1, step=1, label="Scale Result"
            )

        with gr.Row():
            initial_upscale = gr.Slider(
                1,
                4,
                value=_DEFAULTS.initial_upscale_factor,
                step=1,
                label="Initial Upscale",
            )
            pixel_width = gr.Slider(
                0, 50, value=_DEFAULTS.pixel_width or 0, step=1, label="Pixel Width (0=auto)"
            )

        with gr.Row():
            transparent = gr.Checkbox(
                value=_DEFAULTS.transparent_background, label="Transparent Background"
            )

        with gr.Row():
            output_format = gr.Radio(
                choices=["Auto", "mp4", "gif"],
                value="Auto",
                label="Output Format (video/GIF inputs only)",
            )
            sample_frames = gr.Slider(
                2,
                32,
                value=_DEFAULTS.video.num_sample_frames,
                step=1,
                label="Sample Frames (video/GIF inputs only)",
            )

        with gr.Accordion("Advanced options", open=False):
            gr.Markdown("### Mesh detection")
            with gr.Row():
                crop_border_pixels = gr.Slider(
                    0,
                    20,
                    value=_MESH.crop_border_pixels,
                    step=1,
                    label="Crop Border Pixels",
                )
                closure_kernel_size = gr.Slider(
                    1,
                    31,
                    value=_MESH.closure_kernel_size,
                    step=1,
                    label="Closure Kernel Size",
                )
            with gr.Row():
                canny_low = gr.Number(
                    value=_MESH.canny_thresholds[0], label="Canny Threshold (low)"
                )
                canny_high = gr.Number(
                    value=_MESH.canny_thresholds[1], label="Canny Threshold (high)"
                )
            with gr.Row():
                cluster_threshold = gr.Slider(
                    1,
                    20,
                    value=_MESH.cluster_threshold,
                    step=1,
                    label="Cluster Threshold",
                )
                angle_threshold_deg = gr.Slider(
                    0,
                    45,
                    value=_MESH.angle_threshold_deg,
                    step=1,
                    label="Angle Threshold (deg)",
                )
                trim_outlier_fraction = gr.Slider(
                    0,
                    0.5,
                    value=_MESH.trim_outlier_fraction,
                    step=0.01,
                    label="Trim Outlier Fraction",
                )

            gr.Markdown("### Hough line transform")
            with gr.Row():
                rho = gr.Slider(0.1, 5, value=_HOUGH.rho, step=0.1, label="Rho")
                theta_deg = gr.Slider(
                    0.1, 5, value=_HOUGH.theta_deg, step=0.1, label="Theta (deg)"
                )
                hough_threshold = gr.Slider(
                    1, 500, value=_HOUGH.threshold, step=1, label="Threshold"
                )
            with gr.Row():
                min_line_len = gr.Slider(
                    1, 500, value=_HOUGH.min_line_len, step=1, label="Min Line Length"
                )
                max_line_gap = gr.Slider(
                    0, 100, value=_HOUGH.max_line_gap, step=1, label="Max Line Gap"
                )

            gr.Markdown("### Color")
            with gr.Row():
                alpha_threshold = gr.Slider(
                    0,
                    255,
                    value=_COLOR.alpha_threshold,
                    step=1,
                    label="Alpha Threshold",
                )
                transparency_majority_fraction = gr.Slider(
                    0,
                    1,
                    value=_COLOR.transparency_majority_fraction,
                    step=0.01,
                    label="Transparency Majority Fraction",
                )
            with gr.Row():
                quantize_method = gr.Dropdown(
                    choices=QUANTIZE_METHODS,
                    value=_COLOR.palette.quantize_method,
                    label="Quantize Method",
                )
                bin_size = gr.Slider(
                    1, 255, value=_COLOR.dominant.bin_size, step=1, label="Bin Size"
                )
                top_colors_limit = gr.Slider(
                    1,
                    32,
                    value=_COLOR.top_colors_limit,
                    step=1,
                    label="Top Colors Limit",
                )
            with gr.Row():
                thumbnail_w = gr.Number(
                    value=_COLOR.thumbnail_size[0], label="Thumbnail Width"
                )
                thumbnail_h = gr.Number(
                    value=_COLOR.thumbnail_size[1], label="Thumbnail Height"
                )

        btn = gr.Button("Pixelate", variant="primary")

        # track_tqdm=True lets the video pipeline's tqdm loop drive a live
        # browser progress bar. Declaring the progress arg here (where gradio is
        # imported) keeps the pure process() gradio-Progress-free. progress must
        # come *before* *args: gradio's special_args only scans positional
        # parameters for a Progress default and stops at the first *args, so a
        # keyword-only progress after *args is never detected (no browser bar).
        def run(progress=gr.Progress(track_tqdm=True), *args):  # noqa: B008 - gradio injects Progress via the default arg
            return process(*args)

        # Config controls keyed by CONFIG_KEYS: process() zips the received
        # positional values back with the same tuple, so the wiring cannot
        # silently fall out of order.
        config_controls = {
            "color_method": color_method,
            "colors": num_colors,
            "scale_result": scale,
            "initial_upscale_factor": initial_upscale,
            "pixel_width": pixel_width,
            "transparent_background": transparent,
            "crop_border_pixels": crop_border_pixels,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "closure_kernel_size": closure_kernel_size,
            "cluster_threshold": cluster_threshold,
            "angle_threshold_deg": angle_threshold_deg,
            "trim_outlier_fraction": trim_outlier_fraction,
            "rho": rho,
            "theta_deg": theta_deg,
            "hough_threshold": hough_threshold,
            "min_line_len": min_line_len,
            "max_line_gap": max_line_gap,
            "alpha_threshold": alpha_threshold,
            "transparency_majority_fraction": transparency_majority_fraction,
            "quantize_method": quantize_method,
            "bin_size": bin_size,
            "top_colors_limit": top_colors_limit,
            "thumbnail_w": thumbnail_w,
            "thumbnail_h": thumbnail_h,
        }
        assert tuple(config_controls) == CONFIG_KEYS

        # An instant "Processing…" flip precedes the work so the click registers
        # visibly even before the pipeline reaches its first tqdm-tracked frame.
        btn.click(
            fn=lambda: gr.update(value=_status("⏳ Processing…"), visible=True),
            inputs=None,
            outputs=status,
        ).then(
            fn=run,
            inputs=[input_file, output_format, sample_frames, *config_controls.values()],
            outputs=[output_img, output_gif, output_video, status],
        )

    return demo
