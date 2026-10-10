"""Deterministic mesh probes and reporting; no expected results come from the detector."""

from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter

CORPUS_VERSION = 2
MIN_PIXEL_WIDTH = 4
EXPECTED_COUNTS = {
    "clean": 20,
    "blur": 20,
    "noise": 20,
    "sparse": 8,
    "stripes": 8,
    "checker": 8,
    "texture": 8,
    "lowcontrast": 8,
    "offset": 8,
}


def synthetic_cases():
    """Corpus v2: 108 cases with source pixel widths of at least four.

    Filter after generation to preserve the exact v1 images for retained IDs.
    """
    for case in _original_cases():
        if case[2] >= MIN_PIXEL_WIDTH:
            yield case


def _original_cases():
    """Replay v1's RNG stream, including cases filtered out of the active corpus.

    Keep loop order and RNG consumption stable: IDs refer to fixed images.
    Change the corpus version when changing generation, not just its seed.
    """
    for seed in range(5):
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 256, (12, 16), dtype=np.uint8)
        for width in (2, 3, 4, 6, 9, 13):
            raw = np.repeat(np.repeat(base, width, 0), width, 1)
            clean = Image.fromarray(raw)
            for family in ("clean", "blur", "noise"):
                image = clean
                if family == "blur":
                    image = clean.filter(ImageFilter.GaussianBlur(0.6))
                if family == "noise":
                    image = Image.fromarray(
                        np.clip(
                            raw.astype(float) + rng.normal(0, 8, raw.shape), 0, 255
                        ).astype(np.uint8)
                    )
                yield f"{family}-{seed}-{width}", family, width, image, (0, 0)

    for seed in range(4):
        rng = np.random.default_rng(seed)
        for width in (3, 6, 10):
            for family in (
                "sparse",
                "stripes",
                "checker",
                "texture",
                "lowcontrast",
                "offset",
            ):
                base = rng.integers(0, 256, (12, 16), dtype=np.uint8)
                if family == "sparse":
                    base = np.where(rng.random((12, 16)) < 0.08, 255, 0).astype(
                        np.uint8
                    )
                if family == "stripes":
                    base = np.tile(rng.integers(0, 256, 16, dtype=np.uint8), (12, 1))
                if family == "checker":
                    base = (np.indices((12, 16)).sum(0) % 2 * 255).astype(np.uint8)
                if family == "lowcontrast":
                    base = (100 + base // 10).astype(np.uint8)
                raw = np.repeat(np.repeat(base, width, 0), width, 1)
                if family == "texture":
                    raw = np.clip(
                        raw.astype(float)
                        + 25 * np.sin(np.indices(raw.shape)[1] * np.pi / 2),
                        0,
                        255,
                    ).astype(np.uint8)
                if family == "offset":
                    raw = raw[1:-2, 2:-1]
                yield (
                    f"{family}-{seed}-{width}",
                    family,
                    width,
                    Image.fromarray(raw),
                    (2, 1) if family == "offset" else (0, 0),
                )


# Fixed independently of observed detector results; errors are in source pixels.
BOUNDARY_TOLERANCE = 0.5
METRICS = ("width", "dimensions", "boundaries", "all")


def expected_mesh(image_size, pixel_width, scale, crop, source_offset=(0, 0)):
    """Construction grid in cropped detector coordinates, including partial cells.

    Match the detector's [0, size - 1] outer-line convention. Synthetic offset
    cropping precedes scaling; the detector's crop is measured AFTER scaling.
    The oracle is independent of detected lines, estimates, and gradient peaks.
    """
    axes = []
    for size, offset in zip(image_size, source_offset, strict=True):
        last = size * scale - 2 * crop - 1
        if last <= 0:
            raise ValueError("Image is too small for the detector crop")
        interior = [
            (boundary - offset) * scale - crop
            for boundary in range(pixel_width, offset + size, pixel_width)
            if 0 < (boundary - offset) * scale - crop < last
        ]
        axes.append([0, *interior, last])
    return axes


def match_boundaries(expected, detected, scale, tolerance=BOUNDARY_TOLERANCE):
    """Ordered one-to-one matching: maximize matches, then minimize total error.

    A line cannot explain multiple expected boundaries. Unmatched expected lines
    are misses; unmatched detected lines are extras. Displacement statistics
    cover matches only, and must be read together with misses/extras.
    """
    expected, detected = sorted(expected), sorted(detected)
    # Each state contains (matched count, total source-pixel error, index pairs).
    previous = [(0, 0.0, ()) for _ in range(len(detected) + 1)]
    for i, target in enumerate(expected):
        current = [(0, 0.0, ())]
        for j, actual in enumerate(detected):
            choices = [previous[j + 1], current[j]]
            error = abs(actual - target) / scale
            if error <= tolerance:
                count, total, pairs = previous[j]
                choices.append((count + 1, total + error, (*pairs, (i, j))))
            current.append(max(choices, key=lambda state: (state[0], -state[1])))
        previous = current
    count, total, pairs = previous[-1]
    used_expected = {i for i, _ in pairs}
    used_detected = {j for _, j in pairs}
    errors = [abs(expected[i] - detected[j]) / scale for i, j in pairs]
    return dict(
        matched=count,
        missing=[value for i, value in enumerate(expected) if i not in used_expected],
        extra=[value for j, value in enumerate(detected) if j not in used_detected],
        precision=count / len(detected) if detected else (1.0 if not expected else 0.0),
        recall=count / len(expected) if expected else 1.0,
        mean_error=sum(errors) / count if count else None,
        max_error=max(errors) if errors else None,
    )


def evaluate_mesh(lines, expected, scale, tolerance=BOUNDARY_TOLERANCE):
    """Measure dimensions, interior boundaries, and outer extent independently."""
    axes = []
    for actual, target in zip(lines, expected, strict=True):
        valid = len(actual) >= 2 and all(
            a < b for a, b in zip(actual, actual[1:], strict=False)
        )
        matches = match_boundaries(target[1:-1], actual[1:-1], scale, tolerance)
        outer_errors = (
            [abs(actual[0] - target[0]) / scale, abs(actual[-1] - target[-1]) / scale]
            if len(actual) >= 2
            else []
        )
        axes.append(
            dict(
                **matches,
                valid=valid,
                outer_errors=outer_errors,
                passed=bool(
                    valid
                    and not matches["missing"]
                    and not matches["extra"]
                    and all(error <= tolerance for error in outer_errors)
                ),
            )
        )
    cells = [max(len(axis) - 1, 0) for axis in lines]
    expected_cells = [len(axis) - 1 for axis in expected]
    return dict(
        cells=cells,
        expected_cells=expected_cells,
        dimensions_passed=cells == expected_cells,
        boundaries_passed=all(axis["passed"] for axis in axes),
        boundary_tolerance=tolerance,
        boundary_axes=axes,
        expected_lines=expected,
        detected_lines=lines,
    )


def summarize(results, weights, metric="width"):
    """Weight family pass rates; missing measurements remain in denominators."""
    groups = defaultdict(list)
    for result in results:
        groups[result["family"]].append(result)
    families = []
    for family, weight in weights.items():
        rows = groups[family]
        passed = sum(row.get(f"{metric}_passed", False) for row in rows)
        families.append(
            dict(
                family=family,
                weight=weight,
                passed=passed,
                total=len(rows),
                rate=passed / len(rows) if rows else None,
            )
        )
    complete = all(row["total"] == EXPECTED_COUNTS[row["family"]] for row in families)
    score = (
        sum(row["weight"] * row["rate"] for row in families) / sum(weights.values())
        if complete
        else None
    )
    return dict(metric=metric, families=families, weighted_pass_rate=score)


def markdown_summary(results, weights):
    summaries = {metric: summarize(results, weights, metric) for metric in METRICS}
    counts = {
        metric: sum(row.get(f"{metric}_passed", False) for row in results)
        for metric in METRICS
    }
    lines = [
        f"## Synthetic mesh benchmark (corpus v{CORPUS_VERSION}, metrics v2)",
        "",
        f"Exact source-width recovery: **{counts['width']}/{len(results)} passed**.",
        f"Mesh dimensions: **{counts['dimensions']}/{len(results)} passed**. "
        f"Boundary recovery: **{counts['boundaries']}/{len(results)} passed**. "
        f"All metrics: **{counts['all']}/{len(results)} passed**.",
        "",
        "Weights are illustrative priorities, not measured user frequencies. "
        "Boundary matching is one-to-one within 0.5 source pixels; outer extents "
        "are also checked. These metrics do not measure final visual quality.",
        "",
        "| Family | Cases | Width | Dimensions | Boundaries | All | Weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, family in enumerate(summaries["width"]["families"]):
        cells = [str(summaries[m]["families"][i]["passed"]) for m in METRICS]
        lines.append(
            f"| {family['family']} | {family['total']} | "
            + " | ".join(cells)
            + f" | {family['weight']:g} |"
        )
    lines += [
        "",
        "| Weighted pass rate | Width | Dimensions | Boundaries | All |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    scores = [summaries[m]["weighted_pass_rate"] for m in METRICS]
    lines.append(
        "| "
        + (
            "Complete corpus"
            if all(s is not None for s in scores)
            else "Incomplete corpus"
        )
        + " | "
        + " | ".join(f"{s:.1%}" if s is not None else "unavailable" for s in scores)
        + " |"
    )
    failures = [row for row in results if not row.get("all_passed", False)]
    if failures:
        lines += [
            "",
            "### Cases with failing or unavailable metrics",
            "",
            "Width and cells below are expected → detected; cells are columns × rows. "
            "Missing/extra counts cover interior boundaries on both axes. "
            "Extent/ordering failures are reported separately.",
            "",
            "| Case | Width | Cells | Missing / extra | Failed metrics |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in failures:
            axes = row.get("boundary_axes")
            counts_text = (
                f"{sum(len(a['missing']) for a in axes)} / {sum(len(a['extra']) for a in axes)}"
                if axes
                else "unavailable"
            )
            failed = [m for m in METRICS[:-1] if not row.get(f"{m}_passed", False)]
            if axes and any(
                not a["valid"]
                or any(e > row["boundary_tolerance"] for e in a["outer_errors"])
                for a in axes
            ):
                failed.append("extent/ordering")
            lines.append(
                f"| {row['id']} | {row['expected_width']} → {row.get('detected_width', '—')} | "
                f"{row.get('expected_cells', '—')} → {row.get('cells', '—')} | {counts_text} | {', '.join(failed)} |"
            )
    return "\n".join(lines) + "\n"
