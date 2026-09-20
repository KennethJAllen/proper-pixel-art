"""The advisory report must not conceal failures or depend on family sample counts."""

from collections import Counter

import pytest

from tests.synthetic_mesh import (
    EXPECTED_COUNTS,
    evaluate_mesh,
    expected_mesh,
    markdown_summary,
    match_boundaries,
    summarize,
    synthetic_cases,
)


def test_corpus_counts_and_unique_ids():
    cases = list(synthetic_cases())
    assert Counter(case[1] for case in cases) == EXPECTED_COUNTS
    assert len({case[0] for case in cases}) == 108
    assert min(case[2] for case in cases) == 4


def test_weighted_score_uses_family_rates_and_counts_failures():
    results = [
        dict(
            id=f"{family}-{index}",
            family=family,
            expected_width=6,
            status="failed" if family == "texture" else "passed",
            width_passed=family != "texture",
        )
        for family, count in EXPECTED_COUNTS.items()
        for index in range(count)
    ]
    weights = dict.fromkeys(EXPECTED_COUNTS, 1)
    weights["texture"] = 2
    summary = summarize(results, weights)
    # Texture has 8 cases, clean has 20; their influence follows weights.
    assert summary["weighted_pass_rate"] == pytest.approx(0.8)
    assert "100/108 passed" in markdown_summary(results, weights)
    texture = next(row for row in summary["families"] if row["family"] == "texture")
    assert texture["total"] == 8
    assert texture["passed"] == 0


def test_partial_run_has_no_overall_score():
    results = [dict(family=family, status="passed") for family in EXPECTED_COUNTS]
    assert (
        summarize(results, dict.fromkeys(EXPECTED_COUNTS, 1))["weighted_pass_rate"]
        is None
    )


def test_oracle_accounts_for_scale_crop_and_partial_cells():
    assert expected_mesh((18, 12), 6, scale=2, crop=2) == [
        [0, 10, 22, 31],
        [0, 10, 19],
    ]
    # Same source at fallback scale: the crop remains two detector pixels.
    assert expected_mesh((18, 12), 6, scale=1, crop=2) == [
        [0, 4, 10, 13],
        [0, 4, 7],
    ]
    # Original 18x12 image cropped [y=1:-2, x=2:-1].
    assert expected_mesh((15, 9), 6, scale=2, crop=2, source_offset=(2, 1)) == [
        [0, 6, 18, 25],
        [0, 8, 13],
    ]


def test_oracle_excludes_boundaries_on_outer_extent():
    assert expected_mesh((6, 6), 3, scale=1, crop=0) == [[0, 3, 5], [0, 3, 5]]
    # At this crop the first latent cell is removed entirely.
    assert expected_mesh((6, 6), 3, scale=2, crop=2, source_offset=(2, 1)) == [
        [0, 6, 7],
        [0, 2, 7],
    ]


def test_matching_is_one_to_one_and_minimizes_displacement():
    match = match_boundaries([10, 20], [9, 10, 20, 21], scale=2)
    assert match["matched"] == 2
    assert match["extra"] == [9, 21]
    assert match["missing"] == []
    assert match["precision"] == 0.5
    assert match["recall"] == 1.0
    assert match["max_error"] == 0
    # One detected line cannot explain both expected lines, even within tolerance.
    crowded = match_boundaries([10, 12], [11], scale=2)
    assert crowded["matched"] == 1
    assert len(crowded["missing"]) == 1


def test_missed_boundaries_are_not_hidden_by_small_matched_error():
    match = match_boundaries([10, 20, 30], [10, 20], scale=2)
    assert match["max_error"] == 0
    assert match["missing"] == [30]
    assert match["recall"] == pytest.approx(2 / 3)


def test_boundary_tolerance_is_in_source_pixels():
    assert match_boundaries([10], [11], scale=2)["matched"] == 1
    assert match_boundaries([10], [11], scale=1)["matched"] == 0
    no_match = match_boundaries([10], [12], scale=2)
    assert no_match["missing"] == [10] and no_match["extra"] == [12]
    assert no_match["max_error"] is None


def test_dimensions_and_boundaries_are_independent():
    expected = [[0, 10, 20, 30], [0, 10, 20]]
    shifted = [[0, 12, 22, 30], [0, 10, 20]]
    result = evaluate_mesh(shifted, expected, scale=2)
    assert result["dimensions_passed"]
    assert not result["boundaries_passed"]
    assert result["boundary_axes"][0]["missing"] == [10, 20]
    assert result["boundary_axes"][0]["extra"] == [12, 22]
    # Oversplitting cannot pass even when every expected boundary is recovered.
    extra = evaluate_mesh([[0, 5, 10, 20, 30], expected[1]], expected, scale=2)
    assert not extra["dimensions_passed"] and not extra["boundaries_passed"]
    assert extra["boundary_axes"][0]["extra"] == [5]


def test_extent_and_ordering_are_checked():
    expected = [[0, 10, 20], [0, 10, 20]]
    for bad in ([2, 10, 20], [0, 10, 18], [0, 10, 10, 20], []):
        assert not evaluate_mesh([bad, expected[1]], expected, 2)["boundaries_passed"]
    assert evaluate_mesh(expected, expected, 2)["boundaries_passed"]


def test_width_mismatch_does_not_fail_other_metrics_in_summary():
    expected = [[0, 10, 20], [0, 10, 20]]
    row = dict(
        id="clean-example",
        family="clean",
        expected_width=5,
        detected_width=5.5,
        width_passed=False,
        all_passed=False,
        **evaluate_mesh(expected, expected, 2),
    )
    weights = dict.fromkeys(EXPECTED_COUNTS, 1)
    assert summarize([row], weights, "width")["families"][0]["passed"] == 0
    assert summarize([row], weights, "dimensions")["families"][0]["passed"] == 1
    assert summarize([row], weights, "boundaries")["families"][0]["passed"] == 1
    report = markdown_summary([row], weights)
    assert "Boundary recovery: **1/1 passed**" in report
    assert "Incomplete corpus" in report


def test_unavailable_metrics_remain_in_denominator():
    rows = [
        dict(family="clean", width_passed=True),
        dict(family="clean", status="failed"),
    ]
    weights = dict.fromkeys(EXPECTED_COUNTS, 1)
    summary = summarize(rows, weights)["families"][0]
    assert summary["rate"] == 0.5
    assert summarize(rows, weights, "boundaries")["families"][0]["rate"] == 0
