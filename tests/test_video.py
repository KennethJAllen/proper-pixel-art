"""Tests for video/GIF pixelation.

All fixtures are generated programmatically: a small logical pixel-art
animation is upscaled with nearest-neighbor, gets per-pixel noise, and is
saved as GIF/MP4 — no binary assets are committed.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from proper_pixel_art import video, video_io
from proper_pixel_art.config import PixelateConfig

LOGICAL_SIZE = 16
PIXEL_WIDTH = 10
PALETTE = np.array(
    [(40, 180, 60), (200, 30, 30), (30, 30, 200), (240, 220, 80)], dtype=np.uint8
)
BACKGROUND = (40, 180, 60)


def _logical_frames(
    n_frames: int, rng: np.random.Generator, uniform_border: bool = False
) -> list[np.ndarray]:
    """Logical frames of random palette cells with a moving red square.

    Random cells give the grid detector full-length edges (like real pixel-art
    video); the square makes the animation non-static.
    """
    frames = []
    for k in range(n_frames):
        logical = PALETTE[
            rng.integers(0, len(PALETTE), size=(LOGICAL_SIZE, LOGICAL_SIZE))
        ]
        if uniform_border:
            logical[0], logical[-1] = BACKGROUND, BACKGROUND
            logical[:, 0], logical[:, -1] = BACKGROUND, BACKGROUND
        x = 2 + k
        logical[3:7, x : x + 4] = (200, 30, 30)
        frames.append(logical)
    return frames


def _noisy_upscale(logical: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    big = np.repeat(np.repeat(logical, PIXEL_WIDTH, 0), PIXEL_WIDTH, 1).astype(np.int16)
    big += rng.integers(-3, 4, big.shape)
    return np.clip(big, 0, 255).astype(np.uint8)


def _make_noisy_arrays(n_frames: int, seed: int = 42, **kwargs) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        _noisy_upscale(frame, rng) for frame in _logical_frames(n_frames, rng, **kwargs)
    ]


def _save_gif(arrays: list[np.ndarray], path: Path, durations: list[int]) -> None:
    images = [Image.fromarray(arr, mode="RGB") for arr in arrays]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )


def _save_mp4(arrays: list[np.ndarray], path: Path, fps: float = 10.0) -> None:
    height, width = arrays[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for arr in arrays:
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()


def _parse_gif_frames(path: Path) -> list[dict]:
    """Minimal GIF block parser returning per-frame descriptor info, used to
    assert on-disk structure (frame offsets/sizes, local color tables)."""
    data = path.read_bytes()
    packed = data[10]
    pos = 13 + (3 * 2 ** ((packed & 0x07) + 1) if packed & 0x80 else 0)
    frames = []
    while pos < len(data):
        block = data[pos]
        if block == 0x3B:  # trailer
            break
        if block == 0x21:  # extension
            pos += 2
            while data[pos] != 0:
                pos += 1 + data[pos]
            pos += 1
        elif block == 0x2C:  # image descriptor
            frame = {
                "x": int.from_bytes(data[pos + 1 : pos + 3], "little"),
                "y": int.from_bytes(data[pos + 3 : pos + 5], "little"),
                "width": int.from_bytes(data[pos + 5 : pos + 7], "little"),
                "height": int.from_bytes(data[pos + 7 : pos + 9], "little"),
                "local_color_table": bool(data[pos + 9] & 0x80),
            }
            frames.append(frame)
            pos += 10
            if frame["local_color_table"]:
                pos += 3 * 2 ** ((data[pos - 1] & 0x07) + 1)
            pos += 1  # LZW minimum code size
            while data[pos] != 0:
                pos += 1 + data[pos]
            pos += 1
        else:
            raise ValueError(f"Unexpected GIF block {block:#x} at {pos}")
    return frames


def _assert_colors_near_palette(frame: Image.Image, tolerance: int = 40) -> None:
    """Every output pixel should sit near one of the logical palette colors."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    distances = np.abs(rgb[:, :, None, :] - PALETTE[None, None, :, :].astype(np.int16))
    nearest = distances.max(axis=-1).min(axis=-1)
    assert nearest.max() <= tolerance, (
        f"Output colors stray {nearest.max()} from the logical palette"
    )


class TestGifRoundtrip:
    def test_gif_to_gif(self, tmp_path: Path):
        durations = [100, 40, 40, 120, 80, 60]
        arrays = _make_noisy_arrays(len(durations))
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, durations)

        output_path = video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=8
        )

        with Image.open(output_path) as result:
            assert result.n_frames == len(durations)
            assert result.size == (LOGICAL_SIZE, LOGICAL_SIZE)
            out_durations = []
            for index in range(result.n_frames):
                result.seek(index)
                out_durations.append(result.info["duration"])
                _assert_colors_near_palette(result)
        assert out_durations == durations

        # One global palette: no frame carries a local color table
        frames = _parse_gif_frames(output_path)
        assert len(frames) == len(durations)
        assert not any(frame["local_color_table"] for frame in frames)

    def test_gif_with_scale_result(self, tmp_path: Path):
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50, 50, 50])

        output_path = video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=8, scale_result=4
        )
        with Image.open(output_path) as result:
            assert result.size == (LOGICAL_SIZE * 4, LOGICAL_SIZE * 4)


class TestMp4Output:
    def test_mp4_to_mp4(self, tmp_path: Path):
        arrays = _make_noisy_arrays(6)
        input_path = tmp_path / "anim.mp4"
        _save_mp4(arrays, input_path)

        output_path = video.pixelate_video(
            input_path, tmp_path / "out.mp4", num_colors=8, scale_result=4
        )

        cap = cv2.VideoCapture(str(output_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        assert abs(n_frames - 6) <= 1  # codecs may report off-by-one counts
        assert size == (LOGICAL_SIZE * 4, LOGICAL_SIZE * 4)

    def test_mp4_fidelity(self, tmp_path: Path):
        """MP4 output must be near-lossless: solid palette blocks survive the
        encode within a few units per channel (regression test for the old
        cv2.VideoWriter smearing)."""
        rng = np.random.default_rng(3)
        arrays = [
            np.repeat(np.repeat(logical, PIXEL_WIDTH, 0), PIXEL_WIDTH, 1)
            for logical in _logical_frames(4, rng)
        ]
        frames = (Image.fromarray(arr, mode="RGB") for arr in arrays)
        output_path = tmp_path / "out.mp4"
        height, width = arrays[0].shape[:2]

        video._write_mp4(frames, output_path, fps=10.0, frame_size=(width, height))

        decoded = [
            np.asarray(f.convert("RGB")) for f, _ in video_io.iter_frames(output_path)
        ]
        assert len(decoded) == len(arrays)
        for original, roundtripped in zip(arrays, decoded, strict=True):
            np.testing.assert_allclose(
                roundtripped.astype(np.int16), original.astype(np.int16), atol=6
            )

    def test_mp4_odd_dimensions_padded_to_even(self, tmp_path: Path):
        """Odd frame sizes are padded to even (yuv420p requirement) by
        replicating the edge row/column."""
        rng = np.random.default_rng(5)
        arr = rng.integers(0, 256, size=(15, 17, 3), dtype=np.uint8)
        frames = (Image.fromarray(arr, mode="RGB") for _ in range(3))
        output_path = tmp_path / "odd.mp4"

        video._write_mp4(frames, output_path, fps=10.0, frame_size=(17, 15))

        decoded = [f for f, _ in video_io.iter_frames(output_path)]
        assert len(decoded) == 3
        assert decoded[0].size == (18, 16)

    def test_gif_to_mp4(self, tmp_path: Path):
        arrays = _make_noisy_arrays(4)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50, 50, 50, 50])

        output_path = video.pixelate_video(
            input_path, tmp_path / "out", num_colors=8, output_format="mp4"
        )
        assert output_path.suffix == ".mp4"
        assert output_path.exists()


class TestOutputFormat:
    def test_unsupported_output_extension_raises(self, tmp_path: Path):
        arrays = _make_noisy_arrays(2)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50, 50])

        with pytest.raises(ValueError, match="Unsupported output format"):
            video.pixelate_video(input_path, tmp_path / "out.webm", num_colors=8)

    @pytest.mark.parametrize("input_ext", [".gif", ".mp4", ".webm"])
    def test_directory_output_defaults_to_gif(self, tmp_path: Path, input_ext: str):
        """Without an output extension the format is GIF, whatever the input is."""
        arrays = _make_noisy_arrays(2)
        gif_path = tmp_path / "anim.gif"
        _save_gif(arrays, gif_path, [50, 50])
        input_path = (
            gif_path
            if input_ext == ".gif"
            else gif_path.rename(tmp_path / f"anim{input_ext}")
        )

        output_path = video.pixelate_video(input_path, tmp_path, num_colors=8)
        assert output_path.suffix == ".gif"
        assert output_path.exists()


class TestOutputNaming:
    def test_directory_output_gets_size_suffix(self, tmp_path: Path):
        """A directory output gets the same '<stem>_<W>x<H>.<ext>' default
        name as the image pipeline."""
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 3)
        out_dir = tmp_path / "out"

        output_path = video.pixelate_video(
            input_path, out_dir, num_colors=8, scale_result=2
        )

        size = LOGICAL_SIZE * 2
        assert output_path == out_dir / f"anim_{size}x{size}.gif"
        assert output_path.is_file()


class TestVariableFrameSizeGif:
    def test_iter_frames_composites_delta_frames(self, tmp_path: Path):
        """GIF writers crop unchanged regions away, so stored frames have
        varying sizes/offsets (the issue commenter's case). iter_frames must
        composite them back to full logical-screen RGBA frames."""
        rng = np.random.default_rng(7)
        background = PALETTE[
            rng.integers(0, len(PALETTE), size=(LOGICAL_SIZE, LOGICAL_SIZE))
        ]
        arrays = []
        for k in range(4):
            logical = background.copy()
            logical[3:7, 2 + k : 6 + k] = (200, 30, 30)
            arrays.append(np.repeat(np.repeat(logical, PIXEL_WIDTH, 0), PIXEL_WIDTH, 1))
        input_path = tmp_path / "delta.gif"
        _save_gif(arrays, input_path, [50] * 4)

        # Precondition: the file really stores cropped delta frames
        stored = _parse_gif_frames(input_path)
        full = (LOGICAL_SIZE * PIXEL_WIDTH, LOGICAL_SIZE * PIXEL_WIDTH)
        assert any((f["width"], f["height"]) != full for f in stored), (
            "Fixture GIF should contain cropped delta frames"
        )

        decoded = list(video_io.iter_frames(input_path))
        assert len(decoded) == 4
        for frame, duration in decoded:
            assert frame.size == full
            assert frame.mode == "RGBA"
            assert duration == 50
        # Compositing correctness: static background region is preserved
        first = np.asarray(decoded[0][0].convert("RGB"))
        last = np.asarray(decoded[-1][0].convert("RGB"))
        np.testing.assert_array_equal(first[-10:], last[-10:])


class TestTransparentBackground:
    def test_transparency_consistent_across_frames(self, tmp_path: Path):
        durations = [50] * 4
        arrays = _make_noisy_arrays(len(durations), uniform_border=True)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, durations)

        output_path = video.pixelate_video(
            input_path,
            tmp_path / "out.gif",
            num_colors=8,
            transparent_background=True,
        )

        with Image.open(output_path) as result:
            for index in range(result.n_frames):
                result.seek(index)
                alpha = np.asarray(result.convert("RGBA"))[..., 3]
                # The uniform border ring becomes transparent in every frame
                assert (alpha[0] == 0).all() and (alpha[-1] == 0).all()
                assert (alpha[:, 0] == 0).all() and (alpha[:, -1] == 0).all()
                # The moving red square stays opaque
                assert (alpha[3:7, 2:6] == 255).any()


class TestConfigSupport:
    def test_config_values_apply(self, tmp_path: Path):
        """Scalar values from a config take effect (scale_result is observable)."""
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 3)
        cfg = PixelateConfig.from_dict({"num_colors": 8, "scale_result": 4})

        output_path = video.pixelate_video(input_path, tmp_path / "out.gif", config=cfg)

        with Image.open(output_path) as result:
            assert result.size == (LOGICAL_SIZE * 4, LOGICAL_SIZE * 4)
            _assert_colors_near_palette(result)

    def test_explicit_args_override_config(self, tmp_path: Path):
        """Explicit scalar kwargs beat the corresponding config values."""
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 3)
        cfg = PixelateConfig.from_dict({"num_colors": 8, "scale_result": 4})

        output_path = video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=2, scale_result=2, config=cfg
        )

        with Image.open(output_path) as result:
            # Explicit scale_result=2 beats config's 4
            assert result.size == (LOGICAL_SIZE * 2, LOGICAL_SIZE * 2)
            # Explicit num_colors=2 beats config's 8: the 4-color fixture
            # collapses to at most 2 colors
            rgb = np.asarray(result.convert("RGB")).reshape(-1, 3)
            assert len(np.unique(rgb, axis=0)) <= 2

    def test_config_reaches_mesh_and_color_stages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """cfg.mesh / cfg.colors are threaded into the internal stages."""
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 3)
        cfg = PixelateConfig.from_dict(
            {
                "num_colors": 8,
                "mesh": {"closure_kernel_size": 6},
                "colors": {"top_colors_limit": 4},
            }
        )
        captured = {}

        original_mesh = video.compute_video_mesh
        original_palette = video.build_global_palette

        def spy_mesh(*args, **kwargs):
            captured["mesh_config"] = kwargs.get("mesh_config")
            return original_mesh(*args, **kwargs)

        def spy_palette(*args, **kwargs):
            captured["color_config"] = kwargs.get("color_config")
            return original_palette(*args, **kwargs)

        monkeypatch.setattr(video, "compute_video_mesh", spy_mesh)
        monkeypatch.setattr(video, "build_global_palette", spy_palette)

        video.pixelate_video(input_path, tmp_path / "out.gif", config=cfg)

        assert captured["mesh_config"] is cfg.mesh
        assert captured["color_config"] is cfg.colors


class TestIntermediateDir:
    # pixelate_video writes fixed-named debug images straight into the dir it is
    # given; the per-input subdirectory (foo/<stem>/) is a CLI-layer convention
    # (see tests/test_cli.py), not a pixelate_video behavior.
    def test_writes_debug_visualizations(self, tmp_path: Path):
        """intermediate_dir gets the mesh overlays and palette preview, matching
        the still-image path — not just a redundant edge dump."""
        arrays = _make_noisy_arrays(4)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 4)
        inter = tmp_path / "inter"

        video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=8, intermediate_dir=inter
        )

        expected = {
            "closed_edges.png",
            "lines.png",
            "mesh.png",
            "quantized_original.png",
        }
        written = {p.name for p in inter.iterdir()}
        assert expected <= written
        # The pre-fix redundant duplicate of closed_edges.png is gone.
        assert "aggregated_edges.png" not in written
        # The edge map and the palette preview capture different stages, so must
        # not be byte-identical (guards against the old same-array-twice bug).
        assert (inter / "closed_edges.png").read_bytes() != (
            inter / "quantized_original.png"
        ).read_bytes()

    def test_skip_quantization_omits_palette_preview(self, tmp_path: Path):
        """With quantization skipped (num_colors=0) there is no palette to
        preview, but the mesh overlays are still written."""
        arrays = _make_noisy_arrays(4)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 4)
        inter = tmp_path / "inter"

        video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=0, intermediate_dir=inter
        )

        written = {p.name for p in inter.iterdir()}
        assert {"closed_edges.png", "lines.png", "mesh.png"} <= written
        assert "quantized_original.png" not in written


class TestVideoIo:
    def test_probe_gif(self, tmp_path: Path):
        durations = [100, 40, 40, 120]
        arrays = _make_noisy_arrays(len(durations))
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, durations)

        info = video_io.probe(input_path)
        assert info.exact_count
        assert info.n_frames == len(durations)
        assert info.size == (LOGICAL_SIZE * PIXEL_WIDTH, LOGICAL_SIZE * PIXEL_WIDTH)
        assert info.fps == 1000.0 / np.mean(durations)

    def test_read_sample_frames_subset(self, tmp_path: Path):
        arrays = _make_noisy_arrays(10)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 10)

        samples = video_io.read_sample_frames(input_path, 4)
        assert len(samples) == 4
        assert all(frame.mode == "RGBA" for frame in samples)
        # The first sample is frame 0 (modulo the GIF writer's 256-color
        # quantization of the noisy fixture)
        decoded = np.asarray(samples[0].convert("RGB"), dtype=np.int16)
        assert np.abs(decoded - arrays[0].astype(np.int16)).max() <= 8

    def test_read_sample_frames_more_than_available(self, tmp_path: Path):
        arrays = _make_noisy_arrays(3)
        input_path = tmp_path / "anim.gif"
        _save_gif(arrays, input_path, [50] * 3)

        samples = video_io.read_sample_frames(input_path, 99)
        assert len(samples) == 3


class TestColorMergeConsistency:
    def test_merged_palette_identical_across_frames(self, tmp_path: Path):
        """In skip-quantization mode the fitted ColorMerger gives every frame
        the same output palette for logically identical colors (no flicker)."""
        rng = np.random.default_rng(7)
        # Static background of near-identical greens (speckle source) with a
        # moving red square, so frames differ but share the same true colors.
        frames = []
        for k in range(4):
            logical = np.tile(
                np.array(BACKGROUND, dtype=np.uint8), (LOGICAL_SIZE, LOGICAL_SIZE, 1)
            )
            logical[3:7, 2 + k : 6 + k] = (200, 30, 30)
            frames.append(_noisy_upscale(logical, rng))
        input_path = tmp_path / "anim.gif"
        _save_gif(frames, input_path, [50] * 4)

        output_path = video.pixelate_video(
            input_path, tmp_path / "out.gif", num_colors=0
        )

        with Image.open(output_path) as result:
            palettes = []
            for frame_idx in range(result.n_frames):
                result.seek(frame_idx)
                rgba = np.asarray(result.convert("RGBA"))
                opaque = rgba[rgba[..., 3] >= 128][:, :3]
                palettes.append({tuple(c) for c in opaque})
        shared = set.intersection(*palettes)
        # The static background must map to one shared color in every frame.
        union = set.union(*palettes)
        background_like = {
            c for c in union if abs(c[0] - 40) < 30 and c[1] > 150 and abs(c[2] - 60) < 30
        }
        assert len(background_like) == 1
        assert background_like <= shared
