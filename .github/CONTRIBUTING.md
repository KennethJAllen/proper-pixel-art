# Contributing to Proper Pixel Art

Thanks for contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/KennethJAllen/proper-pixel-art.git
cd proper-pixel-art
uv sync
```

## Before submitting a PR

```bash
uv run ruff format   # format
uv run ruff check    # lint
uv run pytest        # test
```

## Visual validation

Pixelation quality is judged **by eye**. The images under `assets/{name}/`
(inputs and the curated example outputs the README links to) are **frozen**
references — the workflow below never overwrites them.

When you change the algorithm, regenerate each case's result and intermediate
visualizations into the gitignored `tests/outputs/{name}/`:

```bash
uv run python scripts/gen_outputs.py
```

Compare `tests/outputs/{name}/` by eye against the committed example in
`assets/{name}/`. Nothing is committed — `assets/` stays as-is.

### Add a new case

1. Add the input image at `assets/{name}/{name}.png`.
2. Add an entry in [`tests/cases.py`](../tests/cases.py).
3. Run `uv run python scripts/gen_outputs.py` and eyeball `tests/outputs/{name}/`.
4. Commit the input image and the `tests/cases.py` change. (Only add example
   outputs under `assets/{name}/` if the README references them.)
