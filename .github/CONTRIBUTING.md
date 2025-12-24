# Contributing to Proper Pixel Art

Thank you for contributing! Here's how to get started.

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/KennethJAllen/proper-pixel-art.git
   cd proper-pixel-art
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Before Submitting a Pull Request

Please ensure your code passes all checks:

### 1. Format your code

```bash
uv run ruff format
```

### 2. Check for linting issues

```bash
uv run ruff check
```

### 3. Run tests

```bash
uv run pytest -s
```

- If changing the main pixelate algorithm, manualy check the results in `tests/outputs/`
- If necessary, change the number of colors in `tests/conftest.py`
