# Synthetic mesh quality benchmark

Run the opt-in quality suite:

```sh
uv run pytest tests/test_mesh_benchmark.py --run-mesh-benchmark --mesh-report-dir=/tmp/mesh-report -q --tb=short
```

Ordinary `uv run pytest` deselects these cases. When explicitly enabled, quality
failures are normal pytest failures with a nonzero exit status. CI runs them in a
separate job with `continue-on-error: true`; they do not fail the workflow. Keep
this advisory job out of required branch-protection checks. Existing unit tests
continue to gate normally, including tests of the reporting logic.

The terminal and GitHub Actions job summary show counts and pass rates by family,
separate weighted pass rates for width, dimensions, boundaries, and all metrics,
and every failing case's expected/detected width and cell counts.
Locally, `--mesh-report-dir` saves `summary.md` and detailed `results.json`
(including weights, corpus/metrics versions, timings, expected/detected lines,
cell counts, per-axis matching statistics, and failure messages).
CI publishes the Markdown summary; detailed assertions remain in its logs.
An incomplete run has no overall weighted score. Use `-k texture` to focus on a
family, or `-k texture-1-6` to replay the width-6 texture regression.

## What the score means

Corpus v2 contains 108 deterministic images with source pixel widths of at least
**4 pixels**. Width-2 and width-3 cases are excluded. Clean, blur, and noise use
widths 4, 6, 9, and 13; the other families use widths 6 and 10. Retained images
are byte-for-byte identical to their corpus-v1 counterparts: filtering happens
after replaying the original RNG sequence. Scores across corpus versions are
not directly comparable because the evaluated population changed.
Metrics v2 adds independent geometry checks; the old width diagnostic remains
unchanged. Each case computes all three metrics before asserting, so a width
failure cannot prevent geometry measurements. A case passes only if all three
metrics pass, but CI remains advisory:

- **Width:** exact construction width in source-image pixels.
- **Dimensions:** exact column/row counts for the cropped latent grid, including
  partial cells. Counts are derived from construction geometry, not hardcoded to
  16 by 12 after cropping.
- **Boundaries:** one-to-one matching of interior detected/expected lines within
  **0.5 source pixels**, plus outer-extent and strict-ordering checks. Passing
  requires no missed or extra boundaries on either axis. This tolerance is an
  explicit localization allowance, not fitted to make current outputs pass.

The oracle transforms original grid boundaries using the synthetic crop offset
(`x=2, y=1` for the offset family), the actual detector scale (including fallback),
and the configured detector crop, which occurs after scaling. It uses the
existing detector's outer-line convention `[0, cropped_size - 1]`, excluding
interior boundaries coincident with those outer lines. That convention is not a
claim that the pipeline retains every source border pixel.

Matching maximizes the number of one-to-one matches, then minimizes total
localization error. Extra lines cannot inflate recall without reducing precision.
The JSON includes per-axis precision, recall, missing/extra coordinates, and mean
and maximum displacement of matched lines in source pixels. Coordinates in
`expected_lines`, `detected_lines`, and missing/extra lists are in the cropped,
scaled detector space. Matched displacement alone can hide misses, so always read
it alongside missing/extra counts. Outer-extent displacement is reported separately.

Different reported widths can produce identical meshes: they should then have
identical geometry scores. Conversely, the correct width can still produce a bad
mesh. The weighted width score remains separately available for comparison with
metrics v1; the new all-metrics score must not be compared directly with the old
width-only score.

Sparse/striped content can make the latent grid ambiguous, including boundaries
with no visible color transition. This benchmark measures recovery of the known
construction grid, not whether every boundary is identifiable from the image.
None of these metrics evaluates palette selection or final visual quality, or
estimates the probability of success for users.

The families are random flat cells (clean), mild blur, mild Gaussian noise,
sparse foreground cells, stripes, checkerboards, periodic within-cell texture,
low contrast, and an off-grid crop. The texture probe uses a fixed four-sample
sinusoid and is deliberately adversarial; it is not a model of all natural
textures or generated art. These cases do not cover transparency, color-only
edges, JPEG artifacts, animation, or large-image resource usage.

## Weighting relevance

`tests/mesh_weights.json` supplies illustrative relative weights:

- Clean, blur, and noise: 20 each.
- Sparse, low contrast, and off-grid crops: 10 each.
- Stripes and checkerboards: 4 each.
- Periodic texture stress: 2.

These are editable priorities, not empirical estimates of image prevalence.
The overall score is `sum(weight * family_pass_rate) / sum(weight)`; adding more
samples to a family must not grant it more influence. Zero weights exclude a
family from the weighted score but leave its failures visible. All nine keys
are required; weights must be finite, nonnegative, with positive total.

Try another profile without editing the default:

```sh
uv run pytest tests/test_mesh_benchmark.py --run-mesh-benchmark --mesh-weights=/path/to/weights.json --mesh-report-dir=/tmp/mesh-report-alternative -q
```

Calibrate weights using opt-in user examples and historical issue inputs, while
recognizing that issue reports overrepresent failures. Use representative usage
samples to estimate prevalence; treat severity/business priorities separately.
Keep a held-out real-image corpus and review visual outputs independently. Compare
results under several plausible weight profiles: an improvement that disappears
under small weight changes is weaker evidence. Always inspect per-family failures,
even when their weight is small, and keep a fixed reference profile across releases.
Do not lower a weight merely because the detector fails those cases.

To compare algorithm revisions, use the same corpus, weight profile, and dependency
environment. Changing RNG loop order changes the generated images; preserve the
original RNG stream and introduce a new corpus version for generator changes. The original known
texture regressions remain advisory here; promote agreed supported behavior to
ordinary regression tests when fixing it.
