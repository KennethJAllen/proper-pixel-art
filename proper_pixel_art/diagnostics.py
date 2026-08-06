from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageDiagnostics:
    """Summary of image characteristics relevant to pixel-art cleanup."""

    width: int
    height: int
    color_count: int
    transparent_pixels: int
    semi_transparent_pixels: int
    noise_level: str
    recommendations: tuple[str, ...]


def _estimate_noise_level(image: Image.Image) -> str:
    """Estimate high-frequency variation in an image.

    This is a lightweight heuristic, not a machine-learning classifier.
    It compares neighboring grayscale pixels after reducing the image size.
    """

    grayscale = image.convert("L")

    max_dimension = max(grayscale.size)

    if max_dimension > 128:
        scale = 128 / max_dimension
        resized_size = (
            max(1, round(grayscale.width * scale)),
            max(1, round(grayscale.height * scale)),
        )
        grayscale = grayscale.resize(
            resized_size,
            Image.Resampling.BILINEAR,
        )

    width, height = grayscale.size

    if width < 2 or height < 2:
        return "low"

    pixels = list(grayscale.getdata())

    total_difference = 0
    comparisons = 0
    strong_transitions = 0

    for y in range(height):
        row_start = y * width

        for x in range(width):
            current = pixels[row_start + x]

            if x + 1 < width:
                difference = abs(current - pixels[row_start + x + 1])
                total_difference += difference
                comparisons += 1

                if difference >= 48:
                    strong_transitions += 1

            if y + 1 < height:
                difference = abs(current - pixels[row_start + width + x])
                total_difference += difference
                comparisons += 1

                if difference >= 48:
                    strong_transitions += 1

    if comparisons == 0:
        return "low"

    average_difference = total_difference / comparisons
    strong_transition_ratio = strong_transitions / comparisons

    if average_difference >= 32 or strong_transition_ratio >= 0.35:
        return "high"

    if average_difference >= 16 or strong_transition_ratio >= 0.18:
        return "moderate"

    return "low"


def diagnose_image(image: Image.Image) -> ImageDiagnostics:
    """Analyze an image and return pixel-art cleanup diagnostics."""
    rgba_image = image.convert("RGBA")

    width, height = rgba_image.size

    colors = rgba_image.convert("RGB").getcolors(maxcolors=16_777_216)

    color_count = len(colors) if colors is not None else 16_777_217

    alpha_values = list(rgba_image.getchannel("A").getdata())

    transparent_pixels = sum(alpha == 0 for alpha in alpha_values)

    semi_transparent_pixels = sum(0 < alpha < 255 for alpha in alpha_values)

    noise_level = _estimate_noise_level(rgba_image)

    recommendations: list[str] = []

    if color_count > 32:
        recommendations.append("Try using --colors 16 or another smaller palette.")

    if semi_transparent_pixels > 0:
        recommendations.append(
            "Semi-transparent pixels detected; "
            "review transparency handling in the output."
        )

    if noise_level == "high":
        recommendations.append(
            "High-frequency variation detected; "
            "review the generated mesh and try --intermediate-dir."
        )
    elif noise_level == "moderate":
        recommendations.append(
            "Moderate image variation detected; try a few different --colors values."
        )

    if not recommendations:
        recommendations.append("No obvious cleanup recommendation.")

    return ImageDiagnostics(
        width=width,
        height=height,
        color_count=color_count,
        transparent_pixels=transparent_pixels,
        semi_transparent_pixels=semi_transparent_pixels,
        noise_level=noise_level,
        recommendations=tuple(recommendations),
    )


def format_diagnostics(
    diagnostics: ImageDiagnostics,
) -> str:
    """Format diagnostics as readable terminal output."""
    recommendations = "\n".join(
        f"- {recommendation}" for recommendation in diagnostics.recommendations
    )

    return f"""Image diagnosis
---------------
Size: {diagnostics.width}x{diagnostics.height}
Estimated colors: {diagnostics.color_count}
Transparent pixels: {diagnostics.transparent_pixels}
Semi-transparent pixels: {diagnostics.semi_transparent_pixels}
Noise level: {diagnostics.noise_level}

Recommendations:
{recommendations}
"""
