"""Camera + video-file discovery, config loading, and ffprobe-free media probing.

Auto-discovers cameras by listing ``videos/*/`` and matches every video file
whose extension is in ``video_extensions`` (config.yaml). Probes each file with
OpenCV so the dry-run has no hard dep on ffprobe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from .timeparse import RecordingStart, resolve_recording_start

log = logging.getLogger(__name__)

DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    area: str
    zones: list[Any]
    entry_line: Any | None


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    recording_start: RecordingStart

    @property
    def frame_size(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class CameraVideos:
    config: CameraConfig
    videos: list[VideoProbe]


@dataclass(frozen=True)
class ProcessingWindow:
    """Time-window slice of a clip to process (in seconds)."""

    start_seconds: float
    duration_seconds: float | None  # None == process to end of clip

    def to_frame_range(self, fps: float, total_frames: int) -> tuple[int, int]:
        """Resolve to ``[start_frame, end_frame_exclusive]`` clipped to clip length."""
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        start = max(0, int(round(self.start_seconds * fps)))
        if self.duration_seconds is None:
            end = total_frames
        else:
            end = start + int(round(self.duration_seconds * fps))
        return start, min(end, total_frames)


@dataclass(frozen=True)
class PipelineConfig:
    store_name: str
    fps_fallback: float
    video_extensions: tuple[str, ...]
    cameras: dict[str, CameraConfig]
    identity: dict[str, Any]
    behaviour: dict[str, Any]
    processing_window: ProcessingWindow
    detect: dict[str, Any]
    raw: dict[str, Any]


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate ``pipeline/config.yaml`` (or a custom path)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "pipeline" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cameras_raw = raw.get("cameras") or {}
    cameras = {
        cam_id: CameraConfig(
            camera_id=cam_id,
            area=str(spec.get("area", cam_id)),
            zones=list(spec.get("zones") or []),
            entry_line=spec.get("entry_line"),
        )
        for cam_id, spec in cameras_raw.items()
    }

    ext_raw = raw.get("video_extensions") or list(DEFAULT_VIDEO_EXTENSIONS)
    extensions = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_raw)

    pw_raw = raw.get("processing_window") or {}
    window = ProcessingWindow(
        start_seconds=float(pw_raw.get("start_seconds", 0.0)),
        duration_seconds=(
            None
            if pw_raw.get("duration_seconds") in (None, "full", -1)
            else float(pw_raw.get("duration_seconds", 180.0))
        ),
    )

    return PipelineConfig(
        store_name=str(raw.get("store_name", "CounterVision Demo Store")),
        fps_fallback=float(raw.get("fps_fallback", 25)),
        video_extensions=extensions,
        cameras=cameras,
        identity=dict(raw.get("identity") or {}),
        behaviour=dict(raw.get("behaviour") or {}),
        processing_window=window,
        detect=dict(raw.get("detect") or {}),
        raw=raw,
    )


def discover_camera_dirs(videos_root: str | Path | None = None) -> list[Path]:
    """Return every immediate sub-directory under ``videos/`` in sorted order."""
    root = Path(videos_root) if videos_root else PROJECT_ROOT / "videos"
    if not root.is_dir():
        raise FileNotFoundError(f"videos root not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def list_camera_videos(camera_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return every video file in ``camera_dir`` whose extension is allowed, sorted."""
    return sorted(
        p
        for p in camera_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")
    )


def probe_video(path: Path, fps_fallback: float) -> VideoProbe:
    """Probe a video with OpenCV: fps, frame size, frame count, duration."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")
    try:
        fps_raw = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    fps = float(fps_raw) if fps_raw and fps_raw > 0 else float(fps_fallback)
    if not fps_raw:
        log.warning("FPS unknown for %s — using fallback %.2f", path.name, fps)
    duration = frame_count / fps if fps > 0 else 0.0

    return VideoProbe(
        path=path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration,
        recording_start=resolve_recording_start(path),
    )


def discover_all(
    config: PipelineConfig,
    videos_root: str | Path | None = None,
) -> list[CameraVideos]:
    """Walk ``videos/*/``, probe every matching video, and pair with config."""
    out: list[CameraVideos] = []
    for cam_dir in discover_camera_dirs(videos_root):
        cam_id = cam_dir.name
        cam_cfg = config.cameras.get(cam_id) or CameraConfig(
            camera_id=cam_id, area=f"(unmapped) {cam_id}", zones=[], entry_line=None
        )
        if cam_id not in config.cameras:
            log.warning(
                "Camera %r found on disk but missing from config.yaml — labelling as unmapped.",
                cam_id,
            )
        videos = [
            probe_video(v, config.fps_fallback)
            for v in list_camera_videos(cam_dir, config.video_extensions)
        ]
        out.append(CameraVideos(config=cam_cfg, videos=videos))
    return out
