"""Opt-in quality tests: assertions fail normally; CI runs these in an advisory job."""

import time

import pytest

from proper_pixel_art import mesh
from proper_pixel_art.config import MeshConfig
from tests.synthetic_mesh import evaluate_mesh, expected_mesh, synthetic_cases

CASES = list(synthetic_cases())
pytestmark = pytest.mark.mesh_benchmark


@pytest.mark.parametrize(
    "case_id,family,expected_width,image,source_offset",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_synthetic_mesh(case_id, family, expected_width, image, source_offset, request):
    result = dict(id=case_id, family=family, expected_width=expected_width)
    request.node.user_properties.append(("mesh_result", result))
    start = time.perf_counter()
    try:
        config = MeshConfig()
        lines, scale, width = mesh.compute_mesh_with_scaling(
            image.convert("RGBA"), mesh_config=config
        )
        expected = expected_mesh(
            image.size, expected_width, scale, config.crop_border_pixels, source_offset
        )
        result.update(
            detected_width=width / scale,
            width_passed=width / scale == expected_width,
            scale=scale,
            detector_crop=config.crop_border_pixels,
            source_offset=source_offset,
            **evaluate_mesh(lines, expected, scale),
        )
        failed = [
            metric
            for metric in ("width", "dimensions", "boundaries")
            if not result[f"{metric}_passed"]
        ]
        result["all_passed"] = not failed
        assert not failed, (
            f"Failed mesh metrics: {', '.join(failed)}; see benchmark report"
        )
    finally:
        result["seconds"] = time.perf_counter() - start
