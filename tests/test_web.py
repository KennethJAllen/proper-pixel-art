"""Tests for the ``ppa-web`` interface: CLI entry point, config assembly, and
the input-type dispatch/preview routing in ``process``."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from proper_pixel_art import web

try:
    # Not just ImportError: gradio's dependency stack can raise other errors on
    # some interpreters (e.g. a pydantic/typing mismatch on pre-release Pythons),
    # and any of those should skip the web tests rather than error at collection.
    import gradio  # noqa: F401

    _HAS_GRADIO = True
except Exception:
    _HAS_GRADIO = False

# process() builds gr.update() return values, so it needs the optional web
# extra. build_config() and the CLI tests are gradio-free and always run.
requires_gradio = pytest.mark.skipif(
    not _HAS_GRADIO, reason="requires the web extra (gradio)"
)


def _config_kwargs(**overrides) -> dict:
    """The full set of ``build_config`` keyword arguments, in signature order,
    with the built-in defaults. Override any of them per test; the returned dict
    is also fed positionally to ``process`` (dict order == parameter order)."""
    kwargs = dict(
        num_colors=16,
        scale_result=1,
        initial_upscale_factor=2,
        pixel_width=0,
        transparent_background=False,
        crop_border_pixels=2,
        canny_low=50,
        canny_high=200,
        closure_kernel_size=8,
        cluster_threshold=4,
        angle_threshold_deg=15,
        trim_outlier_fraction=0.2,
        rho=1.0,
        theta_deg=1.0,
        hough_threshold=100,
        min_line_len=50,
        max_line_gap=10,
        alpha_threshold=128,
        transparency_majority_fraction=0.5,
        quantize_method="MAXCOVERAGE",
        bin_size=52,
        top_colors_limit=8,
        thumbnail_w=160,
        thumbnail_h=160,
    )
    kwargs.update(overrides)
    return kwargs


# --- ppa-web CLI ----------------------------------------------------------


def test_web_main_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that web.main() launches with default server_name and server_port as None."""
    monkeypatch.setattr(sys, "argv", ["ppa-web"])

    mock_demo = MagicMock()
    with patch("proper_pixel_art.web.create_demo", return_value=mock_demo):
        web.main()

    mock_demo.launch.assert_called_once_with(server_name=None, server_port=None)


def test_web_main_custom_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that web.main() parses --host and --port and passes them to demo.launch."""
    monkeypatch.setattr(sys, "argv", ["ppa-web", "--host", "0.0.0.0", "--port", "8080"])

    mock_demo = MagicMock()
    with patch("proper_pixel_art.web.create_demo", return_value=mock_demo):
        web.main()

    mock_demo.launch.assert_called_once_with(server_name="0.0.0.0", server_port=8080)


# --- build_config (no gradio required) ------------------------------------


def test_build_config_maps_all_ui_values() -> None:
    """Every control value lands in the right (possibly nested) config field."""
    cfg = web.build_config(
        **_config_kwargs(
            num_colors=16,
            scale_result=3,
            initial_upscale_factor=2,
            pixel_width=5,
            transparent_background=True,
            crop_border_pixels=1,
            canny_low=40,
            canny_high=210,
            closure_kernel_size=7,
            rho=1.5,
            hough_threshold=120,
            quantize_method="MEDIANCUT",
            bin_size=30,
            thumbnail_w=120,
            thumbnail_h=140,
        )
    )

    assert (cfg.num_colors, cfg.scale_result, cfg.initial_upscale_factor) == (16, 3, 2)
    assert cfg.pixel_width == 5
    assert cfg.transparent_background is True

    assert cfg.mesh.crop_border_pixels == 1
    assert cfg.mesh.canny_thresholds == (40, 210)
    assert cfg.mesh.closure_kernel_size == 7
    assert cfg.mesh.hough.rho == 1.5
    assert cfg.mesh.hough.threshold == 120

    assert cfg.colors.quantize_method == "MEDIANCUT"
    assert cfg.colors.bin_size == 30
    assert cfg.colors.thumbnail_size == (120, 140)


def test_build_config_coerces_types() -> None:
    """Gradio hands back floats; integer-typed fields must end up as real ints
    (and list fields as tuples) so downstream indexing/quantization works."""
    cfg = web.build_config(
        **_config_kwargs(
            num_colors=16.0, canny_low=40.0, canny_high=210.0, bin_size=52.0
        )
    )
    assert isinstance(cfg.num_colors, int)
    assert isinstance(cfg.mesh.canny_thresholds, tuple)
    assert all(isinstance(v, int) for v in cfg.mesh.canny_thresholds)
    assert isinstance(cfg.colors.bin_size, int)
    # quantize_method resolves to a real PIL.Image.Quantize value without error.
    assert cfg.colors.quantize is not None


# --- process() dispatch + preview routing (needs the web extra) -----------

# Small synthetic pixel art the mesh detector can resolve: random palette cells
# (so every cell boundary is an edge) upscaled by a fixed pixel width.
LOGICAL_SIZE = 16
PIXEL_WIDTH = 10
PALETTE = np.array(
    [(40, 180, 60), (200, 30, 30), (30, 30, 200), (240, 220, 80)], dtype=np.uint8
)
# build_config args in positional order, for forwarding through process().
CONFIG_ARGS = tuple(_config_kwargs().values())


def _frames(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n):
        logical = PALETTE[rng.integers(0, len(PALETTE), (LOGICAL_SIZE, LOGICAL_SIZE))]
        logical[3:7, 2 + k : 6 + k] = (200, 30, 30)  # moving square -> animation
        frames.append(np.repeat(np.repeat(logical, PIXEL_WIDTH, 0), PIXEL_WIDTH, 1))
    return frames


def _make_png(tmp_path: Path) -> Path:
    path = tmp_path / "sprite.png"
    Image.fromarray(_frames(1)[0], mode="RGB").save(path)
    return path


def _make_gif(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "anim.gif"
    images = [Image.fromarray(a, mode="RGB") for a in _frames(n)]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=[80] * n, loop=0
    )
    return path


def _make_mp4(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "anim.mp4"
    arrays = _frames(n)
    height, width = arrays[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    for arr in arrays:
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def _updates(input_path, output_format="Auto", sample_frames=4):
    img, gif, video, status = web.process(
        None if input_path is None else str(input_path),
        output_format,
        sample_frames,
        *CONFIG_ARGS,
    )
    return dict(img), dict(gif), dict(video), dict(status)


@requires_gradio
def test_process_image_shows_image_only(tmp_path: Path) -> None:
    img, gif, video, status = _updates(_make_png(tmp_path))
    assert img["visible"] is True
    assert gif["visible"] is False and video["visible"] is False
    assert isinstance(img["value"], Image.Image)
    assert status["visible"] is True and "Done" in status["value"]


@requires_gradio
def test_process_gif_shows_animated_gif_preview(tmp_path: Path) -> None:
    img, gif, video, status = _updates(_make_gif(tmp_path))
    assert gif["visible"] is True
    assert img["visible"] is False and video["visible"] is False
    assert gif["value"].startswith('<img src="data:image/gif;base64,')
    assert "Done" in status["value"]


@requires_gradio
def test_process_mp4_shows_video_preview(tmp_path: Path) -> None:
    img, gif, video, status = _updates(_make_mp4(tmp_path))
    assert video["visible"] is True
    assert img["visible"] is False and gif["visible"] is False
    out = Path(video["value"])
    assert out.suffix == ".mp4" and out.exists()
    assert "Done" in status["value"]


@requires_gradio
def test_process_format_override_routes_mp4_to_gif(tmp_path: Path) -> None:
    """Choosing format=gif for a video input produces the GIF preview."""
    img, gif, video, status = _updates(_make_mp4(tmp_path), output_format="gif")
    assert gif["visible"] is True
    assert img["visible"] is False and video["visible"] is False


@requires_gradio
def test_process_no_file_shows_prompt_and_hides_previews() -> None:
    img, gif, video, status = _updates(None)
    assert not any(u["visible"] for u in (img, gif, video))
    assert status["visible"] is True and status["value"]


@requires_gradio
def test_process_failure_surfaces_in_status(tmp_path: Path) -> None:
    """A pipeline error resolves to a visible '❌ Failed' status, not a blank page."""
    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"not a real png")
    img, gif, video, status = _updates(bad)
    assert not any(u["visible"] for u in (img, gif, video))
    assert status["visible"] is True and "Failed" in status["value"]
