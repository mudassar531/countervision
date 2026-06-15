"""Phase 1 — detect + track.

Runs YOLO26 + BoT-SORT (with ReID + enlarged track buffer) over a
configurable processing window of each camera's video. Produces:

* ``data/output/annotated/<camera>.mp4`` — boxes, IDs and short traces,
  colored by tracker_id so visual ID stability is obvious to an
  evaluator.
* ``data/output/frames/<camera>.jpg`` — the clean first decoded frame of
  the processing window (used as the heatmap backdrop in Phase 2).
* ``data/output/tracks/<camera>.jsonl`` — one record per detection
  (camera, frame_idx, wall-clock, tracker_id, xyxy, conf). Phase 2+
  consumers (zones, identity, journey, aggregate) read this so they
  never re-run the model.

We don't have ground-truth IDs, so the "ID-switch count" we report is a
spatial-overlap **proxy** (see :class:`IdSwitchCounter`): when a brand-new
tracker_id appears whose bounding box overlaps a recently-disappeared
tracker_id (IoU ≥ ``id_switch_iou`` within ``id_switch_lookback_frames``),
that is counted as one likely switch. It's a useful churn metric, not a
MOTA score.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from .discover import (
    PROJECT_ROOT,
    CameraVideos,
    PipelineConfig,
    ProcessingWindow,
    VideoProbe,
)
from .timeparse import wall_clock_for_frame

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ID-switch proxy metric
# --------------------------------------------------------------------------- #


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class IdSwitchCounter:
    """Spatial-overlap proxy for ID switches when no ground truth is available.

    Whenever a previously-unseen ``tracker_id`` appears, check if any other
    ``tracker_id`` was seen in the last ``lookback_frames`` frames whose last
    bounding box overlaps the new one with IoU ≥ ``iou_threshold``. If yes,
    record one likely switch (lost_id → new_id) at the current frame.
    """

    iou_threshold: float = 0.30
    lookback_frames: int = 30

    last_seen: dict[int, tuple[np.ndarray, int]] = field(default_factory=dict)
    seen_ids: set[int] = field(default_factory=set)
    switches: list[dict[str, Any]] = field(default_factory=list)

    def update(self, frame_idx: int, ids: np.ndarray, boxes: np.ndarray) -> None:
        for tid_raw, box in zip(ids, boxes, strict=True):
            tid = int(tid_raw)
            if tid not in self.seen_ids:
                self.seen_ids.add(tid)
                best: tuple[int, float] | None = None
                for prev_id, (prev_box, prev_frame) in self.last_seen.items():
                    if prev_id == tid:
                        continue
                    if frame_idx - prev_frame > self.lookback_frames:
                        continue
                    iou = _iou_xyxy(prev_box, box)
                    if iou >= self.iou_threshold and (best is None or iou > best[1]):
                        best = (prev_id, iou)
                if best is not None:
                    self.switches.append(
                        {
                            "frame_idx": frame_idx,
                            "lost_id": best[0],
                            "new_id": tid,
                            "iou": round(best[1], 4),
                        }
                    )
            self.last_seen[tid] = (box.copy(), frame_idx)

    @property
    def count(self) -> int:
        return len(self.switches)


# --------------------------------------------------------------------------- #
# Per-camera tracking
# --------------------------------------------------------------------------- #


@dataclass
class CameraTrackResult:
    camera_id: str
    area: str
    video_path: Path
    annotated_path: Path
    frame_jpg_path: Path
    tracks_jsonl_path: Path
    start_frame: int
    end_frame: int
    frames_processed: int
    detections_total: int
    unique_track_ids: int
    id_switch_count: int
    id_switches: list[dict[str, Any]]
    fps_processing: float
    elapsed_seconds: float


def _resolve_paths(camera_id: str, out_root: Path) -> tuple[Path, Path, Path]:
    annotated = out_root / "annotated" / f"{camera_id}.mp4"
    frame_jpg = out_root / "frames" / f"{camera_id}.jpg"
    tracks_jl = out_root / "tracks" / f"{camera_id}.jsonl"
    for p in (annotated, frame_jpg, tracks_jl):
        p.parent.mkdir(parents=True, exist_ok=True)
    return annotated, frame_jpg, tracks_jl


def _build_annotators(detect_cfg: dict[str, Any]) -> tuple[
    sv.BoxAnnotator, sv.LabelAnnotator, sv.TraceAnnotator
]:
    """Annotators colored by tracker_id (so ID stability is visually obvious)."""
    box = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        thickness=int(detect_cfg.get("box_thickness", 2)),
    )
    label = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        text_color=sv.Color.WHITE,
        text_scale=float(detect_cfg.get("label_text_scale", 0.5)),
        text_padding=4,
    )
    trace = sv.TraceAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        trace_length=int(detect_cfg.get("trace_length", 90)),
        thickness=int(detect_cfg.get("box_thickness", 2)),
    )
    return box, label, trace


def _draw_hud(
    frame: np.ndarray,
    *,
    camera_id: str,
    area: str,
    wall_clock_iso: str,
    frame_idx: int,
    total_frames: int,
    live_count: int,
) -> np.ndarray:
    """Lightweight HUD overlay so the validator can read context at a glance."""
    h = 64
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (10, 19, 71), -1)  # navy
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    line1 = f"{camera_id}  -  {area}"
    line2 = f"frame {frame_idx + 1}/{total_frames}   t={wall_clock_iso}   in_frame={live_count}"
    cv2.putText(frame, line1, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, line2, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 255), 1)
    return frame


def _resolve_tracker_yaml(detect_cfg: dict[str, Any]) -> str:
    raw = detect_cfg.get("tracker_yaml", "countervision/botsort_demo.yaml")
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / "pipeline" / raw).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Tracker YAML not found: {p}")
    return str(p)


def track_camera(
    *,
    config: PipelineConfig,
    camera_id: str,
    area: str,
    video: VideoProbe,
    window: ProcessingWindow,
    out_root: Path,
    model: Any,
) -> CameraTrackResult:
    """Run detection + BoT-SORT on one camera's video for the given window."""
    detect_cfg = config.detect
    start_frame, end_frame = window.to_frame_range(video.fps, video.frame_count)
    if end_frame <= start_frame:
        raise ValueError(
            f"Empty processing window for {video.path.name}: "
            f"start={start_frame} end={end_frame} total={video.frame_count}"
        )
    total_window_frames = end_frame - start_frame

    annotated_path, frame_jpg_path, tracks_jsonl_path = _resolve_paths(camera_id, out_root)

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video.path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    video_info = sv.VideoInfo(width=video.width, height=video.height, fps=int(round(video.fps)))
    box_ann, label_ann, trace_ann = _build_annotators(detect_cfg)
    id_switches = IdSwitchCounter(
        iou_threshold=float(detect_cfg.get("id_switch_iou", 0.30)),
        lookback_frames=int(detect_cfg.get("id_switch_lookback_frames", 30)),
    )

    tracker_yaml = _resolve_tracker_yaml(detect_cfg)
    classes = list(detect_cfg.get("classes", [0]))
    imgsz = int(detect_cfg.get("imgsz", 960))
    conf = float(detect_cfg.get("conf", 0.30))
    iou = float(detect_cfg.get("iou", 0.55))
    device = str(detect_cfg.get("device", "mps"))

    seen_ids: set[int] = set()
    detections_total = 0
    frames_processed = 0
    start_t = time.perf_counter()

    log.info(
        "[%s] %s — window frames %d..%d (%d frames, ~%.1fs at %.2f fps)",
        camera_id,
        video.path.name,
        start_frame,
        end_frame,
        total_window_frames,
        total_window_frames / video.fps,
        video.fps,
    )

    sink_ctx = sv.VideoSink(target_path=str(annotated_path), video_info=video_info)
    tracks_fh = tracks_jsonl_path.open("w", encoding="utf-8")

    try:
        with sink_ctx as sink:
            for offset in range(total_window_frames):
                ok, frame = cap.read()
                if not ok:
                    log.warning(
                        "[%s] decode ended early at offset %d (requested %d)",
                        camera_id,
                        offset,
                        total_window_frames,
                    )
                    break

                if offset == 0:
                    cv2.imwrite(str(frame_jpg_path), frame)

                results = model.track(
                    source=[frame],
                    persist=True,
                    tracker=tracker_yaml,
                    classes=classes,
                    device=device,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )
                result = results[0]
                detections = sv.Detections.from_ultralytics(result)

                # Drop detections without a tracker_id — they're useless for
                # cross-frame analytics (counts, dwell, journey) and would
                # also break the colour-by-track annotators downstream.
                if detections.tracker_id is None:
                    detections = detections[:0]
                else:
                    keep = ~np.isnan(detections.tracker_id.astype(float))
                    detections = detections[keep]

                frame_idx_global = start_frame + offset
                wall_clock_iso = wall_clock_for_frame(
                    video.recording_start, frame_idx_global, video.fps
                ).isoformat(timespec="milliseconds")

                tids = (
                    detections.tracker_id.astype(int)
                    if detections.tracker_id is not None and len(detections) > 0
                    else np.zeros((0,), dtype=int)
                )
                boxes = (
                    detections.xyxy.astype(float)
                    if len(detections) > 0
                    else np.zeros((0, 4), dtype=float)
                )

                if len(detections) > 0:
                    id_switches.update(frame_idx_global, tids, boxes)
                    seen_ids.update(int(t) for t in tids)
                    detections_total += len(detections)
                    confs = (
                        detections.confidence
                        if detections.confidence is not None
                        else np.full(len(detections), float("nan"))
                    )
                    for tid, box, c in zip(tids, boxes, confs, strict=True):
                        tracks_fh.write(
                            json.dumps(
                                {
                                    "camera_id": camera_id,
                                    "video": video.path.name,
                                    "frame_idx": int(frame_idx_global),
                                    "frame_offset": int(offset),
                                    "wall_clock": wall_clock_iso,
                                    "tracker_id": int(tid),
                                    "xyxy": [round(float(x), 2) for x in box.tolist()],
                                    "conf": (
                                        round(float(c), 4) if not np.isnan(c) else None
                                    ),
                                }
                            )
                            + "\n"
                        )

                labels = [f"#{int(t)}" for t in tids]
                annotated = frame.copy()
                # Annotators require a tracker_id field; skip them entirely
                # when no person was detected this frame.
                if len(detections) > 0:
                    annotated = trace_ann.annotate(scene=annotated, detections=detections)
                    annotated = box_ann.annotate(scene=annotated, detections=detections)
                    annotated = label_ann.annotate(
                        scene=annotated, detections=detections, labels=labels
                    )
                annotated = _draw_hud(
                    annotated,
                    camera_id=camera_id,
                    area=area,
                    wall_clock_iso=wall_clock_iso,
                    frame_idx=offset,
                    total_frames=total_window_frames,
                    live_count=len(detections),
                )
                sink.write_frame(frame=annotated)
                frames_processed += 1

                if frames_processed % 250 == 0:
                    elapsed = time.perf_counter() - start_t
                    rate = frames_processed / elapsed if elapsed > 0 else 0.0
                    log.info(
                        "[%s] %d/%d frames (%.1f fps, %d unique IDs, %d switches)",
                        camera_id,
                        frames_processed,
                        total_window_frames,
                        rate,
                        len(seen_ids),
                        id_switches.count,
                    )
    finally:
        tracks_fh.close()
        cap.release()

    elapsed = time.perf_counter() - start_t
    rate = frames_processed / elapsed if elapsed > 0 else 0.0
    log.info(
        "[%s] DONE — %d frames in %.1fs (%.1f fps), %d unique IDs, %d switches",
        camera_id,
        frames_processed,
        elapsed,
        rate,
        len(seen_ids),
        id_switches.count,
    )

    return CameraTrackResult(
        camera_id=camera_id,
        area=area,
        video_path=video.path,
        annotated_path=annotated_path,
        frame_jpg_path=frame_jpg_path,
        tracks_jsonl_path=tracks_jsonl_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frames_processed=frames_processed,
        detections_total=detections_total,
        unique_track_ids=len(seen_ids),
        id_switch_count=id_switches.count,
        id_switches=id_switches.switches,
        fps_processing=rate,
        elapsed_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# Orchestrator over every discovered camera
# --------------------------------------------------------------------------- #


def _load_yolo_model(detect_cfg: dict[str, Any]):
    """Lazy import so Phase 0 dry-run + lint don't need torch / ultralytics."""
    from ultralytics import YOLO  # type: ignore

    name = str(detect_cfg.get("model", "yolo26s.pt"))
    log.info("Loading YOLO model %s ...", name)
    return YOLO(name)


def _reset_tracker_if_present(model: Any) -> None:
    """Reset the internal BoT-SORT/ByteTrack state so each camera starts fresh.

    Without this, ``persist=True`` keeps lost-track state across cameras and
    new IDs start where the previous camera left off (e.g. camera-3 first ID
    is 10 because camera-1 burned through 9). Reset is a no-op on the first
    call (no predictor / trackers attached yet).
    """
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return
    for t in trackers:
        if hasattr(t, "reset"):
            t.reset()
        elif hasattr(t, "reset_id"):
            t.reset_id()


def run_detect_track(
    config: PipelineConfig,
    cameras: list[CameraVideos],
    window: ProcessingWindow,
    out_root: Path,
) -> list[CameraTrackResult]:
    """Run detect+track on the first video of each discovered camera."""
    model = _load_yolo_model(config.detect)
    results: list[CameraTrackResult] = []
    for cam in cameras:
        if not cam.videos:
            log.warning("[%s] no videos discovered — skipping.", cam.config.camera_id)
            continue
        video = cam.videos[0]  # one representative clip per camera for the demo run
        _reset_tracker_if_present(model)
        result = track_camera(
            config=config,
            camera_id=cam.config.camera_id,
            area=cam.config.area,
            video=video,
            window=window,
            out_root=out_root,
            model=model,
        )
        results.append(result)
    return results


def summarize_results(results: list[CameraTrackResult]) -> str:
    """Compact stdout summary of a detect+track run (for the CLI + logs)."""
    if not results:
        return "(no cameras processed)\n"
    lines = ["", "Phase 1 detect+track summary", "============================"]
    total_frames = total_det = total_ids = total_switches = 0
    for r in results:
        lines.append(
            f"[{r.camera_id}]  area: {r.area}"
        )
        lines.append(
            f"  video         : {r.video_path.name}"
        )
        lines.append(
            f"  window        : frames {r.start_frame}..{r.end_frame}  "
            f"({r.frames_processed} processed)"
        )
        lines.append(
            f"  detections    : {r.detections_total} total  "
            f"({r.detections_total / max(r.frames_processed, 1):.2f}/frame)"
        )
        lines.append(
            f"  unique IDs    : {r.unique_track_ids}"
        )
        lines.append(
            f"  ID switches   : {r.id_switch_count}  (spatial-overlap proxy; see README)"
        )
        lines.append(
            f"  processing    : {r.elapsed_seconds:.1f}s  ({r.fps_processing:.1f} fps)"
        )
        lines.append(
            f"  annotated mp4 : {r.annotated_path.relative_to(PROJECT_ROOT)}"
        )
        lines.append(
            f"  first frame   : {r.frame_jpg_path.relative_to(PROJECT_ROOT)}"
        )
        lines.append(
            f"  tracks jsonl  : {r.tracks_jsonl_path.relative_to(PROJECT_ROOT)}"
        )
        lines.append("")
        total_frames += r.frames_processed
        total_det += r.detections_total
        total_ids += r.unique_track_ids
        total_switches += r.id_switch_count
    lines.append(
        f"TOTAL  {len(results)} cameras  |  {total_frames} frames  |  "
        f"{total_det} detections  |  {total_ids} unique IDs  |  "
        f"{total_switches} ID switches (proxy)"
    )
    return "\n".join(lines) + "\n"


# Silence "imported but unused" for the optional sink_ctx context manager.
_ = nullcontext
