#!/usr/bin/env python3
"""Regenerate pixelation outputs for visual quality validation.

    uv run python scripts/gen_outputs.py

Pixelates every case in ``tests/cases.py`` into the gitignored
``tests/outputs/{name}/`` for eyeballing algorithm changes. See
CONTRIBUTING.md -> Visual validation.
"""

import sys
from pathlib import Path

from PIL import Image

# Make the repo root importable so ``tests.cases`` resolves when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proper_pixel_art import pixelate  # noqa: E402
from tests.cases import PIXELATE_PNG_CASES  # noqa: E402

OUTPUT_DIR = ROOT / "tests" / "outputs"


def main() -> None:
    for name, params in PIXELATE_PNG_CASES.items():
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Regenerating {name}...")
        img = Image.open(params["path"])
        result = pixelate(
            img,
            num_colors=params["num_colors"],
            scale_result=params["result_scale"],
            transparent_background=params["transparent_background"],
            intermediate_dir=out_dir,
        )
        result.save(out_dir / "result.png")
        print(
            f"  wrote {(out_dir / 'result.png').relative_to(ROOT)} "
            f"({result.width}x{result.height})"
        )

    print(
        f"\nRegenerated {len(PIXELATE_PNG_CASES)} case(s) in "
        f"{OUTPUT_DIR.relative_to(ROOT)}/. "
        "Compare them by eye against the committed examples in assets/."
    )


if __name__ == "__main__":
    main()
