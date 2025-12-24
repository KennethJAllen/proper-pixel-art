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
        num_colors=num_colors if num_colors > 0 else None,
        transparent_background=transparent,
        scale_result=scale if scale > 1 else None,
        initial_upscale_factor=initial_upscale,
        pixel_width=pixel_width if pixel_width > 0 else None,
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
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
