"""Integration test for ``countervision.discover`` against a synthetic mp4.

CI does not have access to the real (gitignored) footage, so we synthesize a
tiny mp4 with cv2.VideoWriter using the canonical ``YYYYMMDDHHMMSSmmm``
filename pattern and assert the full discover → probe → timeparse path
returns the expected metadata.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.discover import (  # noqa: E402
    discover_all,
    discover_camera_dirs,
    list_camera_videos,
    load_config,
    probe_video,
)


def _write_synthetic_mp4(path: Path, *, frames: int = 10, fps: int = 25,
                          width: int = 160, height: int = 120) -> None:
    """Write a minimal mp4 with mp4v codec — works on Linux CI + macOS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"could not open VideoWriter for {path}"
    try:
        for i in range(frames):
            frame = np.full((height, width, 3), fill_value=(i * 25) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    assert path.exists() and path.stat().st_size > 0


@pytest.fixture
def synthetic_videos_root(tmp_path: Path) -> Path:
    """Build a `videos/` tree with one mp4 per camera, named per spec."""
    root = tmp_path / "videos"
    _write_synthetic_mp4(root / "camera-1" / "20260607205350587.mp4")
    _write_synthetic_mp4(root / "camera-3" / "20260608003129784.mp4")
    _write_synthetic_mp4(
        root
        / "camera-5"
        / "20260608044448561_AH8174419_Barkerend - Kingz_16_video.mp4"
    )
    return root


def test_discover_camera_dirs(synthetic_videos_root: Path) -> None:
    dirs = discover_camera_dirs(synthetic_videos_root)
    assert [p.name for p in dirs] == ["camera-1", "camera-3", "camera-5"]


def test_list_camera_videos_filters_extensions(synthetic_videos_root: Path) -> None:
    extra = synthetic_videos_root / "camera-1" / "notes.txt"
    extra.write_text("hi")
    vids = list_camera_videos(synthetic_videos_root / "camera-1", (".mp4", ".mov"))
    assert [v.name for v in vids] == ["20260607205350587.mp4"]


def test_probe_video_returns_real_metadata(synthetic_videos_root: Path) -> None:
    target = synthetic_videos_root / "camera-1" / "20260607205350587.mp4"
    probe = probe_video(target, fps_fallback=15.0)
    assert probe.width == 160
    assert probe.height == 120
    assert probe.frame_count == 10
    assert probe.fps == pytest.approx(25.0, abs=0.5)
    assert probe.recording_start.is_from_filename
    assert probe.recording_start.timestamp == datetime(2026, 6, 7, 20, 53, 50, 587_000)


def test_discover_all_end_to_end(synthetic_videos_root: Path) -> None:
    config = load_config(_PIPELINE_DIR / "config.yaml")
    cameras = discover_all(config, videos_root=synthetic_videos_root)

    assert [c.config.camera_id for c in cameras] == ["camera-1", "camera-3", "camera-5"]
    by_id = {c.config.camera_id: c for c in cameras}

    assert by_id["camera-1"].config.area == "Cosmetics & Skincare"
    assert by_id["camera-3"].config.area == "Customer Seating / Try-on Lounge"
    assert by_id["camera-5"].config.area == "Service & Consultation Desk"

    for cam in cameras:
        assert len(cam.videos) == 1, cam.config.camera_id
        v = cam.videos[0]
        assert v.recording_start.is_from_filename
        assert v.fps > 0
        assert v.width == 160 and v.height == 120
        assert v.frame_count == 10


def test_discover_unmapped_camera_is_labelled(tmp_path: Path) -> None:
    """A camera folder on disk with no config entry is labelled `(unmapped)`."""
    root = tmp_path / "videos"
    _write_synthetic_mp4(root / "camera-1" / "20260607205350587.mp4")
    _write_synthetic_mp4(root / "camera-99" / "20260607205350587.mp4")

    config = load_config(_PIPELINE_DIR / "config.yaml")
    cameras = discover_all(config, videos_root=root)
    by_id = {c.config.camera_id: c for c in cameras}
    assert "(unmapped)" in by_id["camera-99"].config.area
