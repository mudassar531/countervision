"""CounterVision pipeline entrypoint.

Phase 0 supports ``--dry-run``: discover cameras, parse recording-start
timestamps from filenames, probe each video, and print a summary. Later
phases will plug in detection, tracking, identity, journey, aggregation,
and rendering.

Usage:
    python pipeline/main.py --dry-run
    python pipeline/main.py --dry-run --config pipeline/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the in-tree ``countervision`` package importable when running this file
# directly (``python pipeline/main.py``) without installing it.
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.discover import (  # noqa: E402
    CameraVideos,
    PipelineConfig,
    discover_all,
    load_config,
)
from countervision.logging_setup import configure_logging  # noqa: E402

PROJECT_ROOT = _PIPELINE_DIR.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="countervision",
        description="CounterVision offline retail-analytics pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "pipeline" / "config.yaml",
        help="Path to config.yaml (default: pipeline/config.yaml).",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=PROJECT_ROOT / "videos",
        help="Root directory containing one sub-folder per camera (default: ./videos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover cameras + probe videos + print a summary. No model inference.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the dry-run summary as JSON to PATH (for CI / downstream tools).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level override (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args(argv)


def _format_report(store_name: str, cameras: list[CameraVideos]) -> str:
    total_files = sum(len(c.videos) for c in cameras)
    lines: list[str] = []
    bar = "=" * max(len(store_name), 32)
    lines.append(store_name)
    lines.append(bar)
    lines.append(f"Discovered {len(cameras)} cameras / {total_files} video files")
    lines.append("")

    for cam in cameras:
        cfg = cam.config
        lines.append(f"[{cfg.camera_id}]  area: {cfg.area}")
        if not cam.videos:
            lines.append("    (no video files matched configured extensions)")
            lines.append("")
            continue
        for v in cam.videos:
            try:
                rel = v.path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = v.path
            source = (
                "parsed-from-filename"
                if v.recording_start.is_from_filename
                else "FALLBACK: file mtime"
            )
            lines.append(f"  {rel}")
            lines.append(
                f"    recording_start: {v.recording_start.isoformat()}  ({source})"
            )
            lines.append(
                f"    fps: {v.fps:.2f}  frame_size: {v.frame_size}  "
                f"duration: {v.duration_seconds:.2f}s  frames: {v.frame_count}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _to_json(store_name: str, cameras: list[CameraVideos]) -> dict:
    return {
        "store_name": store_name,
        "camera_count": len(cameras),
        "video_count": sum(len(c.videos) for c in cameras),
        "cameras": [
            {
                "camera_id": c.config.camera_id,
                "area": c.config.area,
                "videos": [
                    {
                        "path": str(v.path),
                        "fps": v.fps,
                        "width": v.width,
                        "height": v.height,
                        "frame_count": v.frame_count,
                        "duration_seconds": v.duration_seconds,
                        "recording_start": v.recording_start.isoformat(),
                        "recording_start_source": v.recording_start.source,
                    }
                    for v in c.videos
                ],
            }
            for c in cameras
        ],
    }


def run_dry_run(config: PipelineConfig, videos_root: Path, json_out: Path | None) -> int:
    cameras = discover_all(config, videos_root=videos_root)
    print(_format_report(config.store_name, cameras))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(_to_json(config.store_name, cameras), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to {json_out}", file=sys.stderr)

    if not cameras:
        print("ERROR: no cameras discovered under videos/.", file=sys.stderr)
        return 2
    if all(not c.videos for c in cameras):
        print("ERROR: cameras discovered but no video files matched.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log = configure_logging(args.log_level)
    log.debug("args=%s", args)

    config = load_config(args.config)

    if args.dry_run:
        return run_dry_run(config, args.videos_root, args.json_out)

    print(
        "Nothing to do. Phase 0 only supports --dry-run; later phases will add "
        "detect/track/identity/aggregate.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
