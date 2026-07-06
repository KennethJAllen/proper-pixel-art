"""Web interface for Proper Pixel Art using Gradio."""

from PIL import Image

from proper_pixel_art.pixelate import pixelate

IMG_HEIGHT = 512


def process(
    image: Image.Image | None,
    num_colors: int,
    transparent: bool,
    scale: int,
    initial_upscale: int,
    pixel_width: int,
) -> Image.Image | None:
    """Process image through pixelation pipeline."""
    if image is None:
        return None
    return pixelate(
        image,
        num_colors=num_colors,
        transparent_background=transparent,
        scale_result=scale,
        initial_upscale_factor=initial_upscale,
        pixel_width=pixel_width,
    )


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    with gr.Blocks(title="Proper Pixel Art") as demo:
        gr.Markdown(
            "# Proper Pixel Art\nConvert AI-generated pixel art to true pixel resolution"
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(
                    type="pil",
                    label="Input",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                )
            with gr.Column():
                output_img = gr.Image(
                    type="pil",
                    label="Output",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                    interactive=False,
                )

        with gr.Row():
            num_colors = gr.Slider(
                0, 64, value=16, step=1, label="Colors (0 = skip quantization)"
            )
            scale = gr.Slider(1, 20, value=1, step=1, label="Scale Result")

        with gr.Row():
            initial_upscale = gr.Slider(1, 4, value=2, step=1, label="Initial Upscale")
            pixel_width = gr.Slider(
                0, 50, value=0, step=1, label="Pixel Width (0=auto)"
            )

        with gr.Row():
            transparent = gr.Checkbox(value=False, label="Transparent Background")
            btn = gr.Button("Pixelate", variant="primary")

        btn.click(
            fn=process,
            inputs=[
                input_img,
                num_colors,
                transparent,
                scale,
                initial_upscale,
                pixel_width,
            ],
            outputs=output_img,
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
