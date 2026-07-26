"""Canonical pixelation cases: the single source of truth shared by the smoke
test (``tests/test_pixelate.py``) and ``scripts/gen_outputs.py``.

Each case maps an asset name to its input image path and a config dict for
``PixelateConfig.from_dict``.
"""

from pathlib import Path

# Repo root: tests/cases.py -> proper-pixel-art/
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def _case(
    name: str,
    *,
    num_colors: int,
    scale_result: int,
    transparent_background: bool,
) -> dict:
    if num_colors:
        colors = {"method": "palette", "palette": {"num_colors": num_colors}}
    else:
        colors = {"method": "dominant"}
    return {
        "name": name,
        "config": {
            "scale_result": scale_result,
            "transparent_background": transparent_background,
            "colors": colors,
        },
        "path": ASSETS / name / f"{name}.png",
    }


# Ordered table of every PNG case exercised by the visual pipeline.
PIXELATE_PNG_CASES: dict[str, dict] = {
    # Transparent background with an interior hole.
    "anchor": _case(
        "anchor", num_colors=16, scale_result=5, transparent_background=True
    ),
    "ash": _case("ash", num_colors=16, scale_result=5, transparent_background=False),
    "bat": _case("bat", num_colors=16, scale_result=5, transparent_background=True),
    "blob": _case("blob", num_colors=16, scale_result=25, transparent_background=False),
    "demon": _case("demon", num_colors=64, scale_result=5, transparent_background=True),
    "mountain": _case(
        "mountain", num_colors=64, scale_result=5, transparent_background=False
    ),
    # Preserves original colors (dominant method, no quantization).
    "pumpkin": _case(
        "pumpkin", num_colors=0, scale_result=5, transparent_background=False
    ),
}
