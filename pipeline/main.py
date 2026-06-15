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
    ProcessingWindow,
    discover_all,
    load_config,
)
from countervision.logging_setup import configure_logging  # noqa: E402

PROJECT_ROOT = _PIPELINE_DIR.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"


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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write pipeline artifacts (default: data/output).",
    )

    # Mode flags — mutually independent but at most one runs per invocation.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover cameras + probe videos + print a summary. No model inference.",
    )
    parser.add_argument(
        "--run-detect-track",
        action="store_true",
        help="Phase 1: run YOLO26 + BoT-SORT per camera over the processing window.",
    )
    parser.add_argument(
        "--run-zones",
        action="store_true",
        help="Phase 2: read tracks/<cam>.jsonl, write zones/<cam>.json + heatmaps/<cam>.png.",
    )
    parser.add_argument(
        "--draw-zones",
        metavar="CAMERA_ID",
        default=None,
        help="Phase 2: open the interactive cv2 zone editor for one camera (writes config.yaml).",
    )
    parser.add_argument(
        "--draw-zones-default",
        action="store_true",
        help="Phase 2: non-interactively populate empty zones / entry_line with sane defaults.",
    )
    parser.add_argument(
        "--overwrite-zones",
        action="store_true",
        help="Used with --draw-zones-default: also overwrite zones that are already set.",
    )

    # Processing-window overrides (Phase 1+).
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help="Override processing_window.start_seconds.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override processing_window.duration_seconds.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process each clip end-to-end (overrides --duration-seconds).",
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


def _effective_window(args: argparse.Namespace, base: ProcessingWindow) -> ProcessingWindow:
    """Apply CLI overrides on top of the config-default processing window."""
    start = args.start_seconds if args.start_seconds is not None else base.start_seconds
    if args.full:
        duration = None
    elif args.duration_seconds is not None:
        duration = args.duration_seconds
    else:
        duration = base.duration_seconds
    return ProcessingWindow(start_seconds=float(start), duration_seconds=duration)


def run_detect_track_mode(
    config: PipelineConfig,
    videos_root: Path,
    output_root: Path,
    window: ProcessingWindow,
) -> int:
    from countervision.detect_track import run_detect_track, summarize_results

    cameras = discover_all(config, videos_root=videos_root)
    cameras_with_video = [c for c in cameras if c.videos]
    if not cameras_with_video:
        print(
            "ERROR: no videos found under videos/ — Phase 1 needs real footage.",
            file=sys.stderr,
        )
        return 2

    results = run_detect_track(
        config=config,
        cameras=cameras_with_video,
        window=window,
        out_root=output_root,
    )
    summary_txt = summarize_results(results)
    print(summary_txt)

    summary_json_path = output_root / "phase1_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(
            {
                "store_name": config.store_name,
                "window": {
                    "start_seconds": window.start_seconds,
                    "duration_seconds": window.duration_seconds,
                },
                "cameras": [
                    {
                        "camera_id": r.camera_id,
                        "area": r.area,
                        "video": str(r.video_path.name),
                        "start_frame": r.start_frame,
                        "end_frame": r.end_frame,
                        "frames_processed": r.frames_processed,
                        "detections_total": r.detections_total,
                        "unique_track_ids": r.unique_track_ids,
                        "id_switch_count": r.id_switch_count,
                        "id_switches": r.id_switches[:50],
                        "fps_processing": round(r.fps_processing, 2),
                        "elapsed_seconds": round(r.elapsed_seconds, 2),
                        "annotated_path": str(r.annotated_path.relative_to(PROJECT_ROOT)),
                        "frame_jpg_path": str(r.frame_jpg_path.relative_to(PROJECT_ROOT)),
                        "tracks_jsonl_path": str(
                            r.tracks_jsonl_path.relative_to(PROJECT_ROOT)
                        ),
                    }
                    for r in results
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary JSON: {summary_json_path}", file=sys.stderr)
    return 0


def run_draw_zones_default_mode(
    config_path: Path,
    output_root: Path,
    overwrite: bool,
) -> int:
    from tools.draw_zones import populate_defaults  # noqa: I001

    summary = populate_defaults(
        config_path=config_path,
        frames_dir=output_root / "frames",
        overwrite=overwrite,
    )
    print("draw-zones-default — wrote config.yaml")
    for cam_id, info in summary.items():
        print(f"  [{cam_id}] {info}")
    return 0


def run_draw_zones_interactive_mode(
    config_path: Path,
    output_root: Path,
    camera_id: str,
) -> int:
    from tools.draw_zones import interactive_draw  # noqa: I001

    frame_path = output_root / "frames" / f"{camera_id}.jpg"
    if not frame_path.exists():
        print(
            f"ERROR: no frame jpg for {camera_id} at {frame_path}. "
            "Run --run-detect-track first.",
            file=sys.stderr,
        )
        return 2
    written = interactive_draw(
        camera_id=camera_id,
        frame_path=frame_path,
        config_path=config_path,
    )
    if not written:
        print(f"[{camera_id}] cancelled (no config changes written)")
        return 1
    print(f"[{camera_id}] saved {len(written.get('zones') or [])} zone(s) "
          f"and {'an' if written.get('entry_line') else 'no'} entry line to {config_path}")
    return 0


def run_zones_mode(
    config: PipelineConfig,
    videos_root: Path,
    output_root: Path,
) -> int:
    from countervision.zones import run_zone_analytics, summarize_results

    tracks_dir = output_root / "tracks"
    frames_dir = output_root / "frames"

    cameras = discover_all(config, videos_root=videos_root)
    results = []
    for cam in cameras:
        cam_id = cam.config.camera_id
        tracks_path = tracks_dir / f"{cam_id}.jsonl"
        frame_path = frames_dir / f"{cam_id}.jpg"
        if not tracks_path.exists() or not frame_path.exists():
            print(
                f"[{cam_id}] skipping — missing tracks ({tracks_path.exists()}) "
                f"or frame ({frame_path.exists()}). Run --run-detect-track first.",
                file=sys.stderr,
            )
            continue
        # The raw config block is the source of truth for zone polygons /
        # entry-line coords; the typed CameraConfig only carries area + the
        # serialized blob.
        block = config.raw.get("cameras", {}).get(cam_id, {})
        expected_videos = [v.path.name for v in cam.videos]
        result = run_zone_analytics(
            camera_id=cam_id,
            area=cam.config.area,
            tracks_jsonl=tracks_path,
            fps=config.fps_fallback,
            frame_jpg=frame_path,
            out_root=output_root,
            zones_cfg=block.get("zones"),
            entry_line_cfg=block.get("entry_line"),
            expected_videos=expected_videos,
        )
        results.append(result)

    if not results:
        print("ERROR: no cameras had tracks + frame artefacts to consume.", file=sys.stderr)
        return 2

    print(summarize_results(results, PROJECT_ROOT))
    summary_path = output_root / "phase2_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "store_name": config.store_name,
                "cameras": [
                    {
                        "camera_id": r.camera_id,
                        "area": r.area,
                        "frames_processed": r.frames_processed,
                        "person_tracks_count": r.person_tracks_count,
                        "footfall_in": r.line_crossing.in_count if r.line_crossing else 0,
                        "footfall_out": r.line_crossing.out_count if r.line_crossing else 0,
                        "zone_peak_occupancy": {
                            z.name: z.occupancy_peak for z in r.zones
                        },
                        "videos_considered": r.videos_considered,
                        "videos_skipped": r.videos_skipped,
                        "json_path": str(r.json_path.relative_to(PROJECT_ROOT)),
                        "heatmap_path": str(r.heatmap_path.relative_to(PROJECT_ROOT)),
                    }
                    for r in results
                ],
                "guardrails": {
                    "unique_visitors_locked": True,
                    "unique_visitors_note": results[0].note_unique_visitors_locked,
                    "dwell_provisional_note": results[0].note_provisional_dwell,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Phase 2 summary JSON: {summary_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log = configure_logging(args.log_level)
    log.debug("args=%s", args)

    config = load_config(args.config)
    window = _effective_window(args, config.processing_window)

    if args.dry_run:
        return run_dry_run(config, args.videos_root, args.json_out)
    if args.run_detect_track:
        return run_detect_track_mode(config, args.videos_root, args.output_root, window)
    if args.draw_zones_default:
        return run_draw_zones_default_mode(args.config, args.output_root, args.overwrite_zones)
    if args.draw_zones is not None:
        return run_draw_zones_interactive_mode(args.config, args.output_root, args.draw_zones)
    if args.run_zones:
        return run_zones_mode(config, args.videos_root, args.output_root)

    print(
        "Nothing to do. Modes: --dry-run (Phase 0) / --run-detect-track (Phase 1) / "
        "--draw-zones-default / --draw-zones CAM / --run-zones (Phase 2).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
