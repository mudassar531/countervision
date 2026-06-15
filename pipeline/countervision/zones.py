"""Phase 2 — zones, footfall, dwell, heatmap, occupancy.

Consumes ``data/output/tracks/<camera>.jsonl`` (written by Phase 1) and
emits, per camera:

* ``data/output/zones/<camera>.json`` — structured analytics
  (footfall in/out, per-zone presence / dwell / occupancy peak,
  occupancy time-series, per-track dwell *provisionally*).
* ``data/output/heatmaps/<camera>.png`` — heatmap overlay on the
  Phase-1 first-frame jpg.

**Guardrail — no "unique visitors".** Tracker IDs are not visitor
identities (one person can fragment into several IDs, one ID can drift).
We always speak about "person tracks" or "active tracks". The
authoritative unique-visitor count is set in Phase 3 from face embeddings.

The primitives (line-crossing, polygon-presence, heatmap accumulation)
are pure NumPy + OpenCV so the unit tests don't pull in supervision /
torch and stay CI-friendly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Geometry primitives — pure NumPy/OpenCV (no supervision import)
# --------------------------------------------------------------------------- #


def bottom_center_xy(xyxy: np.ndarray) -> tuple[float, float]:
    """Bottom-center anchor of an [x1,y1,x2,y2] box, in pixel coords."""
    x = (float(xyxy[0]) + float(xyxy[2])) / 2.0
    y = float(xyxy[3])
    return x, y


def _line_side(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed cross-product (which side of line A→B is point P on)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def point_in_polygon(p: tuple[float, float], polygon: np.ndarray) -> bool:
    """OpenCV ray-casting test. ``polygon`` is an ``(N, 2)`` float ndarray."""
    return cv2.pointPolygonTest(polygon.astype(np.float32), p, False) >= 0


@dataclass
class LineCrossing:
    """Count tracker_ids crossing a directed line A→B.

    A new bottom-center anchor on the **negative** side becomes "outside" and a
    transition to the positive side counts as **in**; the reverse counts as
    **out**. Direction is set by the order of (start, end) — re-draw the line
    the other way to flip in/out.
    """

    start: tuple[float, float]
    end: tuple[float, float]

    in_count: int = 0
    out_count: int = 0
    last_side: dict[int, int] = field(default_factory=dict)  # tid → -1 / 0 / +1
    events: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        frame_idx: int,
        wall_clock_iso: str,
        ids: np.ndarray,
        boxes: np.ndarray,
    ) -> None:
        """Update counts for this frame's detections.

        Convention: stand at ``start`` looking toward ``end``. Bottom-center
        anchors on your **left** are "inside" (``in``); to your **right**
        are "outside" (``out``). With image y growing down, that means a
        horizontal line drawn left→right counts an "in" when the person
        crosses upward (smaller y) — i.e. walking *toward* the camera /
        deeper into the store. Re-draw the line right→left to swap.
        """
        for tid_raw, box in zip(ids, boxes, strict=True):
            tid = int(tid_raw)
            p = bottom_center_xy(box)
            s = _line_side(p, self.start, self.end)
            side = 1 if s > 0 else (-1 if s < 0 else 0)
            prev = self.last_side.get(tid, 0)
            if prev != 0 and side != 0 and prev != side:
                direction = "in" if side < 0 else "out"
                if direction == "in":
                    self.in_count += 1
                else:
                    self.out_count += 1
                self.events.append(
                    {
                        "frame_idx": int(frame_idx),
                        "wall_clock": wall_clock_iso,
                        "tracker_id": tid,
                        "direction": direction,
                    }
                )
            if side != 0:
                self.last_side[tid] = side


@dataclass
class PolygonZone:
    """Count and dwell-time per tracker_id inside an arbitrary polygon."""

    name: str
    polygon: np.ndarray  # (N, 2) float32
    color: str = "#0A1347"  # navy default for the dashboard

    frames_in_zone_by_track: dict[int, int] = field(default_factory=dict)
    occupancy_peak: int = 0
    _current_active: set[int] = field(default_factory=set)

    def update(self, frame_idx: int, ids: np.ndarray, boxes: np.ndarray) -> None:
        self._current_active.clear()
        for tid_raw, box in zip(ids, boxes, strict=True):
            tid = int(tid_raw)
            if point_in_polygon(bottom_center_xy(box), self.polygon):
                self._current_active.add(tid)
                self.frames_in_zone_by_track[tid] = (
                    self.frames_in_zone_by_track.get(tid, 0) + 1
                )
        self.occupancy_peak = max(self.occupancy_peak, len(self._current_active))

    def dwell_seconds_by_track(self, fps: float) -> dict[int, float]:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        return {tid: round(n / fps, 2) for tid, n in self.frames_in_zone_by_track.items()}


@dataclass
class HeatmapAccumulator:
    """Per-frame gaussian density accumulator over box bottom-center points."""

    width: int
    height: int
    radius: int = 28          # gaussian sigma in pixels
    weight: float = 1.0
    accumulator: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height must be positive, got {self.width}x{self.height}")
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def add(self, boxes: np.ndarray) -> None:
        for box in boxes:
            x, y = bottom_center_xy(box)
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < self.width and 0 <= cy < self.height:
                self.accumulator[cy, cx] += self.weight

    def render(
        self,
        *,
        base_frame: np.ndarray | None = None,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Return an 8-bit BGR PNG: either the heat alone or composited on ``base_frame``.

        Short-circuits to an all-zeros image when the accumulator is empty so
        the PNG honestly says "no data" instead of "deep blue everywhere"
        (the JET colormap maps 0 to dark blue).
        """
        sigma = max(1.0, float(self.radius))
        ksize = int(round(sigma * 6.0)) | 1  # 6σ kernel, force odd
        blurred = cv2.GaussianBlur(self.accumulator, (ksize, ksize), sigma)
        peak = float(blurred.max())
        if peak <= 0:
            out_h, out_w = (
                (base_frame.shape[0], base_frame.shape[1])
                if base_frame is not None
                else (self.height, self.width)
            )
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        heat = np.clip(blurred * (255.0 / peak), 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat, colormap)
        if base_frame is None:
            return colored
        if base_frame.shape[:2] != colored.shape[:2]:
            base_frame = cv2.resize(base_frame, (colored.shape[1], colored.shape[0]))
        mask = (heat > 8).astype(np.float32)[..., None]
        mixed = base_frame.astype(np.float32) * (1.0 - mask * alpha) + colored.astype(np.float32) * (
            mask * alpha
        )
        return np.clip(mixed, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Track loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrameDetections:
    frame_idx: int
    wall_clock_iso: str
    ids: np.ndarray   # (N,)
    boxes: np.ndarray  # (N, 4) xyxy
    confs: np.ndarray  # (N,)


def load_tracks_jsonl(path: Path) -> tuple[list[FrameDetections], dict[str, Any]]:
    """Load tracks JSONL grouped by frame. Returns ``(frames_sorted, meta)``."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    videos_seen: set[str] = set()
    cameras_seen: set[str] = set()
    wall_clock_first: str | None = None
    wall_clock_last: str | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            by_frame[int(rec["frame_idx"])].append(rec)
            videos_seen.add(rec.get("video", ""))
            cameras_seen.add(rec.get("camera_id", ""))
            wc = rec.get("wall_clock")
            if wc:
                if wall_clock_first is None or wc < wall_clock_first:
                    wall_clock_first = wc
                if wall_clock_last is None or wc > wall_clock_last:
                    wall_clock_last = wc

    frames: list[FrameDetections] = []
    for f_idx in sorted(by_frame):
        recs = by_frame[f_idx]
        ids = np.array([r["tracker_id"] for r in recs], dtype=int)
        boxes = np.array([r["xyxy"] for r in recs], dtype=float)
        confs = np.array(
            [r["conf"] if r.get("conf") is not None else float("nan") for r in recs],
            dtype=float,
        )
        wall = recs[0].get("wall_clock", "")
        frames.append(FrameDetections(f_idx, wall, ids, boxes, confs))

    meta = {
        "cameras_in_file": sorted(cameras_seen),
        "videos_in_file": sorted(v for v in videos_seen if v),
        "frame_range": (frames[0].frame_idx, frames[-1].frame_idx) if frames else (0, 0),
        "wall_clock_first": wall_clock_first,
        "wall_clock_last": wall_clock_last,
    }
    return frames, meta


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


@dataclass
class ZoneAnalyticsResult:
    camera_id: str
    area: str
    fps: float
    frames_processed: int
    person_tracks_count: int          # NOT "unique_visitors"
    line_crossing: LineCrossing | None
    zones: list[PolygonZone]
    occupancy_timeseries: list[dict[str, Any]]
    footfall_by_hour: list[dict[str, Any]]
    heatmap_path: Path
    json_path: Path
    videos_considered: list[str]
    videos_skipped: list[str]
    note_provisional_dwell: str
    note_unique_visitors_locked: str


def _parse_line(line_cfg: dict[str, Any] | list[list[float]] | None) -> tuple[
    tuple[float, float], tuple[float, float]
] | None:
    """Accept either ``{start: [x,y], end:[x,y]}`` or ``[[x,y],[x,y]]``."""
    if line_cfg is None:
        return None
    if isinstance(line_cfg, dict):
        start = tuple(float(v) for v in line_cfg["start"])
        end = tuple(float(v) for v in line_cfg["end"])
    else:
        start = tuple(float(v) for v in line_cfg[0])
        end = tuple(float(v) for v in line_cfg[1])
    return start, end


def _parse_zones(zones_cfg: list[dict[str, Any]] | None) -> list[PolygonZone]:
    out: list[PolygonZone] = []
    for z in zones_cfg or []:
        polygon = np.array(z["polygon"], dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(f"Zone {z.get('name')!r} polygon must be (N≥3, 2): got {polygon.shape}")
        out.append(
            PolygonZone(
                name=str(z.get("name", "zone")),
                polygon=polygon,
                color=str(z.get("color", "#0A1347")),
            )
        )
    return out


def _hour_bucket(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%H:00")
    except (ValueError, TypeError):
        return ""


def run_zone_analytics(
    *,
    camera_id: str,
    area: str,
    tracks_jsonl: Path,
    fps: float,
    frame_jpg: Path,
    out_root: Path,
    zones_cfg: list[dict[str, Any]] | None,
    entry_line_cfg: dict[str, Any] | list[list[float]] | None,
    expected_videos: list[str],
) -> ZoneAnalyticsResult:
    frames, meta = load_tracks_jsonl(tracks_jsonl)
    if not frames:
        raise ValueError(f"No tracks in {tracks_jsonl}")

    frame_bgr = cv2.imread(str(frame_jpg))
    if frame_bgr is None:
        raise RuntimeError(f"Could not read base frame {frame_jpg}")
    h, w = frame_bgr.shape[:2]

    zones = _parse_zones(zones_cfg)
    line_pts = _parse_line(entry_line_cfg)
    line_crossing = LineCrossing(start=line_pts[0], end=line_pts[1]) if line_pts else None
    heatmap = HeatmapAccumulator(width=w, height=h)

    seen_tracker_ids: set[int] = set()
    occupancy: list[dict[str, Any]] = []
    last_second_emitted: int | None = None

    for fd in frames:
        seen_tracker_ids.update(int(t) for t in fd.ids)
        heatmap.add(fd.boxes)
        if line_crossing is not None:
            line_crossing.update(fd.frame_idx, fd.wall_clock_iso, fd.ids, fd.boxes)
        for z in zones:
            z.update(fd.frame_idx, fd.ids, fd.boxes)

        sec = int(fd.frame_idx / fps)
        if sec != last_second_emitted:
            occupancy.append(
                {
                    "t": fd.wall_clock_iso,
                    "frame_idx": int(fd.frame_idx),
                    "second_bucket": sec,
                    "active_tracks": int(len(fd.ids)),
                }
            )
            last_second_emitted = sec

    footfall_by_hour: list[dict[str, Any]] = []
    if line_crossing is not None and line_crossing.events:
        buckets: dict[str, dict[str, int]] = {}
        for ev in line_crossing.events:
            h_bucket = _hour_bucket(ev["wall_clock"])
            if not h_bucket:
                continue
            b = buckets.setdefault(h_bucket, {"in": 0, "out": 0})
            b[ev["direction"]] += 1
        footfall_by_hour = [
            {"hour": k, "in": v["in"], "out": v["out"], "total": v["in"] + v["out"]}
            for k, v in sorted(buckets.items())
        ]

    heatmap_path = out_root / "heatmaps" / f"{camera_id}.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = heatmap.render(base_frame=frame_bgr)
    if not cv2.imwrite(str(heatmap_path), rendered):
        raise RuntimeError(f"Failed to write heatmap PNG: {heatmap_path}")

    note_provisional_dwell = (
        "Per-tracker dwell is PROVISIONAL. One person may produce multiple "
        "tracker IDs across the window (occlusion / fragmentation). "
        "Phase 3 face-linking replaces this with per-person dwell."
    )
    note_unique_visitors_locked = (
        "tracker_id is not a visitor identity. Authoritative unique-visitor "
        "counts come from face-based identity in Phase 3."
    )

    videos_in_tracks = meta["videos_in_file"]
    videos_skipped = sorted(set(expected_videos) - set(videos_in_tracks))

    payload = {
        "version": 1,
        "camera_id": camera_id,
        "area": area,
        "fps": fps,
        "frame_jpg": str(frame_jpg),
        "heatmap_png": str(heatmap_path),
        "window": {
            "frame_first": meta["frame_range"][0],
            "frame_last": meta["frame_range"][1],
            "wall_clock_first": meta["wall_clock_first"],
            "wall_clock_last": meta["wall_clock_last"],
            "frames_processed": len(frames),
        },
        "videos_considered": videos_in_tracks,
        "videos_skipped": videos_skipped,
        "person_tracks": {
            "count": len(seen_tracker_ids),
            "note": (
                "Count of unique tracker_ids. NOT 'unique visitors' — see "
                "unique_visitors_locked / Phase 3."
            ),
        },
        "unique_visitors_locked": True,
        "unique_visitors_note": note_unique_visitors_locked,
        "footfall": (
            None
            if line_crossing is None
            else {
                "entry_line": {
                    "start": list(line_crossing.start),
                    "end": list(line_crossing.end),
                    "anchor": "bottom_center",
                },
                "in_count": line_crossing.in_count,
                "out_count": line_crossing.out_count,
                "total_crossings": line_crossing.in_count + line_crossing.out_count,
                "events": line_crossing.events,
            }
        ),
        "footfall_by_hour": footfall_by_hour,
        "zones": [
            {
                "name": z.name,
                "color": z.color,
                "polygon": z.polygon.tolist(),
                "occupancy_peak": z.occupancy_peak,
                "active_tracker_ids": sorted(z.frames_in_zone_by_track),
                "dwell_seconds_by_track_provisional": {
                    str(k): v for k, v in z.dwell_seconds_by_track(fps).items()
                },
                "avg_dwell_seconds_provisional": (
                    round(
                        sum(z.dwell_seconds_by_track(fps).values())
                        / max(1, len(z.frames_in_zone_by_track)),
                        2,
                    )
                ),
                "provisional_note": note_provisional_dwell,
            }
            for z in zones
        ],
        "occupancy_timeseries": occupancy,
    }

    json_path = out_root / "zones" / f"{camera_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return ZoneAnalyticsResult(
        camera_id=camera_id,
        area=area,
        fps=fps,
        frames_processed=len(frames),
        person_tracks_count=len(seen_tracker_ids),
        line_crossing=line_crossing,
        zones=zones,
        occupancy_timeseries=occupancy,
        footfall_by_hour=footfall_by_hour,
        heatmap_path=heatmap_path,
        json_path=json_path,
        videos_considered=videos_in_tracks,
        videos_skipped=videos_skipped,
        note_provisional_dwell=note_provisional_dwell,
        note_unique_visitors_locked=note_unique_visitors_locked,
    )


def summarize_results(results: list[ZoneAnalyticsResult], project_root: Path) -> str:
    """Compact stdout summary for the --run-zones CLI."""
    if not results:
        return "(no cameras processed)\n"
    out: list[str] = ["", "Phase 2 zone-analytics summary", "==============================", ""]
    for r in results:
        in_c = r.line_crossing.in_count if r.line_crossing else 0
        out_c = r.line_crossing.out_count if r.line_crossing else 0
        peaks = ", ".join(f"{z.name}:{z.occupancy_peak}" for z in r.zones) or "(no zones)"
        out.append(f"[{r.camera_id}]  area: {r.area}")
        out.append(f"  frames consumed   : {r.frames_processed}")
        out.append(f"  person tracks     : {r.person_tracks_count}  "
                   "(NOT unique visitors — see unique_visitors_locked)")
        out.append(f"  footfall (in/out) : {in_c} / {out_c}  (total {in_c + out_c})")
        out.append(f"  zone peak occ.    : {peaks}")
        out.append(f"  videos in tracks  : {', '.join(r.videos_considered) or '(none)'}")
        if r.videos_skipped:
            out.append(f"  videos NOT tracked: {', '.join(r.videos_skipped)}")
        out.append(f"  heatmap           : {r.heatmap_path.relative_to(project_root)}")
        out.append(f"  zones JSON        : {r.json_path.relative_to(project_root)}")
        out.append("")
    return "\n".join(out) + "\n"
