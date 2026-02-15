"""Tests for video pixelation."""

from pathlib import Path

import cv2

from proper_pixel_art import mesh, video

VIDEO_PATH = Path.cwd() / "assets" / "video" / "warrior.mp4"
OUTPUT_DIR = Path.cwd() / "tests" / "outputs" / "video"


def test_compute_video_mesh():
    """Verify aggregated mesh detection produces a non-trivial mesh."""
    mesh_lines, upscale_factor = video.compute_video_mesh(
        VIDEO_PATH, num_samples=8, output_dir=OUTPUT_DIR / "mesh_debug"
    )
    mesh_x, mesh_y = mesh_lines
    assert len(mesh_x) > 2, f"Trivial mesh x: only {len(mesh_x)} lines"
    assert len(mesh_y) > 2, f"Trivial mesh y: only {len(mesh_y)} lines"
    assert not mesh._is_trivial_mesh(mesh_lines)
    print(f"Video mesh: {len(mesh_x)} x-lines, {len(mesh_y)} y-lines, upscale={upscale_factor}")


def test_pixelate_video_mp4():
    """Full end-to-end: pixelate and save as MP4."""
    output_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_pixelated.mp4"
    result_path = video.pixelate_video(
        input_path=VIDEO_PATH,
        output_path=output_path,
        num_colors=16,
        scale_result=5,
        output_format="mp4",
        num_sample_frames=8,
    )
    assert result_path.exists(), "Output MP4 not created"
    cap = cv2.VideoCapture(str(result_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0, "Output MP4 has no frames"
    print(f"MP4 output: {result_path} ({frame_count} frames)")


def test_pixelate_video_gif():
    """Full end-to-end: pixelate and save as GIF."""
    output_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_pixelated.gif"
    result_path = video.pixelate_video(
        input_path=VIDEO_PATH,
        output_path=output_path,
        num_colors=16,
        scale_result=5,
        output_format="gif",
        num_sample_frames=8,
    )
    assert result_path.exists(), "Output GIF not created"
    assert result_path.stat().st_size > 0, "Output GIF is empty"
    print(f"GIF output: {result_path} ({result_path.stat().st_size} bytes)")
