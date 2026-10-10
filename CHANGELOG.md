# Changelog

## 2.0.0

A breaking cleanup release: the config object is now the single source of
truth, with the sentinel shorthands and pre-1.8 compatibility layers removed.

### Python API

- `pixelate()` and `pixelate_video()` are config-only: all tuning kwargs
  (`num_colors`, `scale_result`, `transparent_background`, `pixel_width`,
  `initial_upscale_factor`, `output_format`, `num_sample_frames`) are gone.
  Build a `PixelateConfig` instead (directly, via `from_dict`, or
  `from_yaml`).
- `pixelate()` returns a `PixelateResult` (`.image`, `.mesh`, `.pixel_width`,
  `.upscale_factor`) instead of a bare PIL image.
- `pixelate_video(input_path, output_path=None, config=None,
  intermediate_dir=None)`: `output_path` defaults to the current directory;
  its `.mp4`/`.gif` suffix selects the format, otherwise
  `config.video.output_format` (default `gif`).
- The module `proper_pixel_art.pixelate` was renamed to
  `proper_pixel_art.image`, so the `pixelate` function no longer shadows its
  own module. The old module path remains as a deprecated import-only shim;
  import from the package root:
  `from proper_pixel_art import pixelate, pixelate_video, PixelateConfig`.
- The package root now exports the config dataclasses, `PixelateResult`, and
  `__version__`; `pixelate_video` is imported lazily.
- Helper functions in `colors`/`mesh` take config objects instead of loose
  scalar kwargs (e.g. `ColorMerger(DominantConfig(...))`,
  `clamp_alpha(img, color_config)`); the module-level `_DEFAULTS` /
  `ALPHA_THRESHOLD` mirrors are gone. `mesh.compute_mesh*` functions also
  return the pixel width they used.
- Progress/warning output uses the `logging` module instead of `print`.

### Config schema (YAML)

- Optional top-level `version: 2` key; when present it must be `2`.
- `scale_result: null` now means "no scaling" (was `1`) and
  `pixel_width: null` means "auto-detect" (was `0`). The old `0` sentinels are
  rejected with an error.
- New `video:` section: `output_format` (`gif`/`mp4`), `num_sample_frames`
  (was CLI-only), `min_vote_fraction` (was a hardcoded constant).
- Renamed mesh keys: `width_selection_tolerance` → `width_keep_tolerance`,
  `width_replacement_tolerance` → `width_replace_tolerance`.
- The pre-1.8 moved-key hint table is gone; unknown keys simply error with a
  pointer here. Keys that moved in 1.8.0: top-level/`colors.num_colors` →
  `colors.palette.num_colors` (with `colors.method: palette`),
  `colors.quantize_method` → `colors.palette.quantize_method`,
  `colors.bin_size` → `colors.dominant.bin_size`,
  `colors.output_color_merge_distance` → `colors.dominant.merge_distance`.

### CLI

- One entry point: `ppa`. The deprecated `ppa-video` alias and the separate
  `ppa-web` script are removed; the web UI is now `ppa web [--host] [--port]`.
- `-c/--colors` is a palette size only (1-256) and implies
  `--color-method palette`; the `0 = dominant` sentinel is gone. The method is
  selected with the new `--color-method dominant|palette`.
- `-t/--transparent` gains `--no-transparent`, so a config file's
  `transparent_background: true` can be overridden off.
- The hidden `-i/--input` alias is removed; the input path is a required
  positional argument.
- `-w/--pixel-width` no longer accepts `0` for auto — omit the flag instead.
- An explicit `.mp4`/`.gif` output path now beats `-f/--format`.

### Packaging

- Requires Python >= 3.11.
- The `[scripts]` extra is removed (it installed dependencies for repo-only
  tools that are not shipped in the wheel); `scripts/` tooling now uses a uv
  dependency-group.
