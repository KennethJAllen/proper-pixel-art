#!/usr/bin/env python3
"""Regenerate the committed golden outputs for the visual regression tests.

Run this after an *intentional* algorithm change:

    uv run python scripts/gen_outputs.py

For every case in ``tests/cases.py`` it pixelates the input asset and overwrites
``assets/{name}/result.png`` (plus the intermediate visualizations: mesh.png,
edges.png, lines.png, closed_edges.png, quantized_original.png). Review the
resulting git image diff before committing -- that diff is the visual review.
"""

import sys
from pathlib import Path

from PIL import Image

# Make the repo root importable so ``tests.cases`` resolves when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proper_pixel_art import pixelate  # noqa: E402
from tests.cases import PIXELATE_PNG_CASES, golden_path  # noqa: E402


def main() -> None:
    for name, params in PIXELATE_PNG_CASES.items():
        asset_dir = ROOT / "assets" / name
        asset_dir.mkdir(parents=True, exist_ok=True)

        print(f"Regenerating {name}...")
        img = Image.open(params["path"])
        result = pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=asset_dir,
        )

        out = golden_path(name)
        result.save(out)
        print(f"  wrote {out.relative_to(ROOT)} ({result.width}x{result.height})")

    print(
        f"\nRegenerated {len(PIXELATE_PNG_CASES)} golden(s). "
        "Review the git image diff before committing."
    )


if __name__ == "__main__":
    main()
