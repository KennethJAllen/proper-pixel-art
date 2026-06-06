"""Canonical pixelation cases.

Single source of truth for the visual pipeline, shared by the smoke test
(``tests/test_pixelate.py`` via ``tests/conftest.py``) and the visual-output
regeneration script (``scripts/gen_outputs.py``). Keeping the table here ensures
the two never drift apart.

Each case maps an asset name to its pixelation parameters and input image path.
The curated example outputs under ``assets/{name}/`` (referenced by the README)
are committed and frozen -- the pipeline never overwrites them. Regenerated
outputs for visual quality validation go to a gitignored directory instead.
"""

from pathlib import Path

# Repo root: tests/cases.py -> proper-pixel-art/
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def _case(
    name: str,
    *,
    num_colors: int,
    result_scale: int,
    transparent_background: bool,
) -> dict:
    return {
        "name": name,
        "num_colors": num_colors,
        "result_scale": result_scale,
        "transparent_background": transparent_background,
        "path": ASSETS / name / f"{name}.png",
    }


# Ordered table of every PNG case exercised by the visual pipeline.
PIXELATE_PNG_CASES: dict[str, dict] = {
    # Transparent background with an interior hole.
    "anchor": _case(
        "anchor", num_colors=16, result_scale=5, transparent_background=True
    ),
    "ash": _case("ash", num_colors=16, result_scale=5, transparent_background=False),
    "bat": _case("bat", num_colors=16, result_scale=5, transparent_background=True),
    "blob": _case("blob", num_colors=16, result_scale=25, transparent_background=False),
    "demon": _case(
        "demon", num_colors=64, result_scale=5, transparent_background=True
    ),
    "mountain": _case(
        "mountain", num_colors=64, result_scale=5, transparent_background=False
    ),
    # Skips quantization (num_colors=0 preserves all colors).
    "pumpkin": _case(
        "pumpkin", num_colors=0, result_scale=5, transparent_background=False
    ),
}
