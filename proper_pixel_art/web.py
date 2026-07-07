"""Web interface for Proper Pixel Art using Gradio.

A single upload accepts either a still image or a video/GIF; the input's suffix
decides whether it is routed to :func:`pixelate` or :func:`pixelate_video`. Every
tunable parameter of :class:`PixelateConfig` is exposed as a native control: the
five common ones up front, the deeper mesh/hough/color knobs in a collapsed
"Advanced" accordion. The result is previewed as a static image, an animated GIF,
or a video player, matching the produced file.
"""

import base64
import tempfile
from pathlib import Path

from PIL import Image

from proper_pixel_art.cli import VIDEO_SUFFIXES
from proper_pixel_art.config import PixelateConfig
from proper_pixel_art.pixelate import pixelate

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


def build_config(
    num_colors: float,
    scale_result: float,
    initial_upscale_factor: float,
    pixel_width: float,
    transparent_background: bool,
    crop_border_pixels: float,
    canny_low: float,
    canny_high: float,
    closure_kernel_size: float,
    cluster_threshold: float,
    angle_threshold_deg: float,
    trim_outlier_fraction: float,
    rho: float,
    theta_deg: float,
    hough_threshold: float,
    min_line_len: float,
    max_line_gap: float,
    alpha_threshold: float,
    transparency_majority_fraction: float,
    quantize_method: str,
    bin_size: float,
    top_colors_limit: float,
    thumbnail_w: float,
    thumbnail_h: float,
) -> PixelateConfig:
    """Assemble a :class:`PixelateConfig` from the raw UI control values.

    Gradio sliders/numbers hand back floats; integer-typed config fields are
    cast so downstream indexing/quantization sees real ints. ``from_dict``
    validates the keys and coerces the list fields into tuples.
    """
    return PixelateConfig.from_dict(
        {
            "num_colors": int(num_colors),
            "scale_result": int(scale_result),
            "initial_upscale_factor": int(initial_upscale_factor),
            "pixel_width": int(pixel_width),
            "transparent_background": bool(transparent_background),
            "mesh": {
                "crop_border_pixels": int(crop_border_pixels),
                "canny_thresholds": [int(canny_low), int(canny_high)],
                "closure_kernel_size": int(closure_kernel_size),
                "cluster_threshold": int(cluster_threshold),
                "angle_threshold_deg": float(angle_threshold_deg),
                "trim_outlier_fraction": float(trim_outlier_fraction),
                "hough": {
                    "rho": float(rho),
                    "theta_deg": float(theta_deg),
                    "threshold": int(hough_threshold),
                    "min_line_len": int(min_line_len),
                    "max_line_gap": int(max_line_gap),
                },
            },
            "colors": {
                "alpha_threshold": int(alpha_threshold),
                "transparency_majority_fraction": float(transparency_majority_fraction),
                "quantize_method": quantize_method,
                "bin_size": int(bin_size),
                "top_colors_limit": int(top_colors_limit),
                "thumbnail_size": [int(thumbnail_w), int(thumbnail_h)],
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
        config = build_config(*config_values)

        if input_path.suffix.lower() in VIDEO_SUFFIXES:
            # Deferred import so image runs don't pay the cv2 import cost.
            from proper_pixel_art import video

            out_path = video.pixelate_video(
                input_path=input_path,
                output_path=Path(tempfile.mkdtemp()),
                output_format=None if output_format == "Auto" else output_format,
                num_sample_frames=int(sample_frames),
                config=config,
            )
            if out_path.suffix.lower() == ".gif":
                return outputs(
                    gif=_gif_preview_html(out_path), status=_status("✅ Done")
                )
            return outputs(video=str(out_path), status=_status("✅ Done"))

        result = pixelate(Image.open(input_path), config=config)
        return outputs(img=result, status=_status("✅ Done"))
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
            num_colors = gr.Slider(
                0,
                64,
                value=_DEFAULTS.num_colors,
                step=1,
                label="Colors (0 = skip quantization)",
            )
            scale = gr.Slider(
                1, 20, value=_DEFAULTS.scale_result, step=1, label="Scale Result"
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
                0, 50, value=_DEFAULTS.pixel_width, step=1, label="Pixel Width (0=auto)"
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
                value=8,
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
                    value=_COLOR.quantize_method,
                    label="Quantize Method",
                )
                bin_size = gr.Slider(
                    1, 255, value=_COLOR.bin_size, step=1, label="Bin Size"
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

        # An instant "Processing…" flip precedes the work so the click registers
        # visibly even before the pipeline reaches its first tqdm-tracked frame.
        btn.click(
            fn=lambda: gr.update(value=_status("⏳ Processing…"), visible=True),
            inputs=None,
            outputs=status,
        ).then(
            fn=run,
            inputs=[
                input_file,
                output_format,
                sample_frames,
                # config_values, in build_config() parameter order:
                num_colors,
                scale,
                initial_upscale,
                pixel_width,
                transparent,
                crop_border_pixels,
                canny_low,
                canny_high,
                closure_kernel_size,
                cluster_threshold,
                angle_threshold_deg,
                trim_outlier_fraction,
                rho,
                theta_deg,
                hough_threshold,
                min_line_len,
                max_line_gap,
                alpha_threshold,
                transparency_majority_fraction,
                quantize_method,
                bin_size,
                top_colors_limit,
                thumbnail_w,
                thumbnail_h,
            ],
            outputs=[output_img, output_gif, output_video, status],
        )

    return demo


def main():
    """Entry point for ppa-web command."""
    import argparse

    parser = argparse.ArgumentParser(description="Web interface for Proper Pixel Art")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address to bind the server to (e.g., 127.0.0.1 or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (e.g., 7860)",
    )
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
