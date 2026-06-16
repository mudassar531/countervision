"""Phase 3 — face-based identity (buffalo_l: SCRFD + ArcFace, CPU on M2).

Replaces Phase 2's tracker-id-based "person tracks" with **authoritative
unique visitors**, merges fragmented Phase-1 tracker IDs that belong to
the same face, recomputes per-person dwell, and emits non-accusatory
review-prompt alerts for repeat visitors and watchlist hits.

Inputs (read-only):

* ``data/output/tracks/<camera>.jsonl`` — Phase-1 tracker boxes per
  frame; we only consume frames that fall on the
  ``identity.sample_every_n_frames`` cadence.
* ``videos/<camera>/<file>`` — re-decoded for the same processing
  window as Phase 1 to access the actual pixels for face detection.
* ``watchlist/*.jpg|*.png`` — operator-provided reference faces.

Outputs:

* ``data/output/identity/<camera>.json`` — authoritative
  ``unique_visitors_count``, ``persons[]`` with ``linked_tracker_ids``
  + ``dwell_seconds`` (authoritative), ``alerts[]`` (review prompts),
  ``unique_visitors_locked: false`` (UNLOCKING Phase 2's sentinel).
* ``data/output/persons/<camera>/<Pxxx>.jpg`` — the highest-quality
  face crop per person (used as thumbnail in the dashboard).
* ``data/output/alerts/<id>.jpg`` — full-frame screenshot for each
  alert (the dashboard "Alerts feed" panel renders these).

Guardrails enforced:

* Embeddings are computed only on faces that pass
  ``det_score >= identity.quality_min``.
* Cosine cutoff for clustering / repeat / watchlist is
  ``identity.cosine_match`` (start 0.30–0.45; tuned in PROGRESS.md).
* Alert ``copy`` is always a non-accusatory review prompt with
  similarity score so the operator can verify before acting.
* When ``identity.enabled: false`` the orchestrator no-ops.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .discover import (
    PROJECT_ROOT,
    CameraVideos,
    PipelineConfig,
    ProcessingWindow,
    VideoProbe,
)
from .id_switch import iou_xyxy
from .timeparse import wall_clock_for_frame

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure helpers (no insightface dependency — for unit tests + reuse)
# --------------------------------------------------------------------------- #


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Treats the inputs as already-L2-normalized."""
    return float(np.dot(a, b))


def link_face_to_tracker(
    face_bbox: np.ndarray,
    person_boxes_by_tid: dict[int, np.ndarray],
) -> tuple[int | None, float]:
    """Return ``(tracker_id, score)`` for the person box best containing this face.

    Heuristic: face center must lie inside the person bbox; among such
    boxes pick the one whose top quarter (the head region) overlaps most
    with the face bbox. Returns ``(None, 0.0)`` if no person bbox
    contains the face.
    """
    fx = (face_bbox[0] + face_bbox[2]) / 2.0
    fy = (face_bbox[1] + face_bbox[3]) / 2.0
    best_tid: int | None = None
    best_score = 0.0
    for tid, pbox in person_boxes_by_tid.items():
        if not (pbox[0] <= fx <= pbox[2] and pbox[1] <= fy <= pbox[3]):
            continue
        # Head region = top quarter of the person box.
        head = np.array([pbox[0], pbox[1], pbox[2], pbox[1] + (pbox[3] - pbox[1]) / 4.0])
        score = iou_xyxy(face_bbox.astype(float), head)
        if score > best_score:
            best_score = score
            best_tid = int(tid)
    if best_tid is None:
        # Face center is not inside any person box (e.g. wall poster, reflection)
        return None, 0.0
    return best_tid, best_score


@dataclass
class _PersonState:
    """Internal accumulator for one face-cluster while we process frames."""

    person_id: str
    embedding_sum: np.ndarray              # (512,) running sum (not normalized)
    embedding_count: int = 0
    centroid: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    linked_tracker_ids: set[int] = field(default_factory=set)
    appearance_frames: list[int] = field(default_factory=list)
    appearance_iso: list[str] = field(default_factory=list)
    best_det_score: float = 0.0
    best_face_bbox: np.ndarray | None = None
    best_frame_idx: int = -1
    best_thumbnail: np.ndarray | None = None  # BGR crop
    best_full_frame: np.ndarray | None = None  # BGR full frame at best moment

    def __post_init__(self) -> None:
        if self.centroid is None and self.embedding_count > 0:
            self.centroid = self.embedding_sum / np.linalg.norm(self.embedding_sum)


class PersonCluster:
    """Greedy online cosine-clustering of L2-normalized 512-d embeddings.

    The centroid is the L2-normalized running sum of all embeddings
    assigned to a cluster; new embeddings join the cluster whose
    centroid has the highest cosine similarity above ``cosine_match``.
    Greedy is order-sensitive but works well in practice for retail
    sessions; the alternative (offline agglomerative) gains very
    little for our cluster sizes (≤ ~30 people).
    """

    def __init__(self, cosine_match: float):
        self.cosine_match = cosine_match
        self.persons: list[_PersonState] = []

    @property
    def count(self) -> int:
        return len(self.persons)

    def assign(self, embedding: np.ndarray) -> _PersonState:
        if not self.persons:
            return self._spawn(embedding)
        sims = np.array([float(np.dot(embedding, p.centroid)) for p in self.persons])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self.cosine_match:
            p = self.persons[best_idx]
            p.embedding_sum = p.embedding_sum + embedding
            p.embedding_count += 1
            p.centroid = p.embedding_sum / np.linalg.norm(p.embedding_sum)
            return p
        return self._spawn(embedding)

    def _spawn(self, embedding: np.ndarray) -> _PersonState:
        person_id = f"P{len(self.persons) + 1:03d}"
        p = _PersonState(
            person_id=person_id,
            embedding_sum=embedding.copy(),
            embedding_count=1,
            centroid=embedding.copy(),
        )
        self.persons.append(p)
        return p


def compute_visit_count(frame_indices: list[int], gap_seconds: float, fps: float) -> int:
    """Number of "visit segments" — runs of presence separated by ``gap_seconds`` of absence."""
    if not frame_indices:
        return 0
    fids = sorted(frame_indices)
    visits = 1
    # zip pairs adjacent elements — fids[1:] is intentionally one shorter.
    for prev, curr in zip(fids, fids[1:], strict=False):
        if (curr - prev) / fps >= gap_seconds:
            visits += 1
    return visits


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #


@dataclass
class WatchlistEntry:
    label: str
    image_path: Path
    embedding: np.ndarray
    det_score: float


class WatchlistMatcher:
    """Pre-embed the operator's watchlist and match per-person centroids."""

    def __init__(
        self,
        watchlist_dir: Path,
        face_app: Any | None,
        threshold: float,
        quality_min: float,
    ) -> None:
        self.entries: list[WatchlistEntry] = []
        self.threshold = threshold
        self.skipped: list[dict[str, Any]] = []

        if face_app is None or not watchlist_dir.is_dir():
            return
        for img_path in sorted(watchlist_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                self.skipped.append({"path": str(img_path), "reason": "could not read image"})
                continue
            faces = face_app.get(img)
            if not faces:
                self.skipped.append({"path": str(img_path), "reason": "no face detected"})
                continue
            best = max(faces, key=lambda f: float(f.det_score))
            if float(best.det_score) < quality_min:
                self.skipped.append(
                    {"path": str(img_path), "reason": f"det_score {best.det_score:.3f} < {quality_min}"}
                )
                continue
            self.entries.append(
                WatchlistEntry(
                    label=img_path.stem,
                    image_path=img_path,
                    embedding=best.normed_embedding.astype(np.float32),
                    det_score=float(best.det_score),
                )
            )

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return ``(label, similarity)`` if best ≥ threshold, else ``(None, best_sim)``."""
        if not self.entries:
            return None, 0.0
        sims = np.array([float(np.dot(embedding, e.embedding)) for e in self.entries])
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        if best_sim >= self.threshold:
            return self.entries[idx].label, best_sim
        return None, best_sim


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_face_app(det_size: tuple[int, int]):
    """Lazy import so unit tests + Phase 0 dry-run don't pay the cost."""
    from insightface.app import FaceAnalysis  # type: ignore

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


def _load_tracks_by_frame(tracks_jsonl: Path) -> dict[int, dict[int, np.ndarray]]:
    """Map ``frame_idx → {tracker_id: xyxy_box (np.ndarray of 4 floats)}``."""
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with tracks_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_frame[int(r["frame_idx"])][int(r["tracker_id"])] = np.array(
                r["xyxy"], dtype=float
            )
    return by_frame


def _crop_thumbnail(frame: np.ndarray, bbox: np.ndarray, *, pad: float = 0.30) -> np.ndarray:
    """Square-padded crop around the face for a clean thumbnail."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int).tolist()
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    x1c = max(0, x1 - px)
    y1c = max(0, y1 - py)
    x2c = min(w, x2 + px)
    y2c = min(h, y2 + py)
    return frame[y1c:y2c, x1c:x2c].copy()


def _alert_copy(kind: str, label: str | None, similarity: float, visit_count: int = 0) -> str:
    """Non-accusatory review prompts — never assert identity, always invite verification."""
    if kind == "watchlist":
        return (
            f"Possible match with watchlist entry '{label}' "
            f"(face similarity {similarity:.2f}). Please verify before acting."
        )
    if kind == "repeat_visitor":
        return (
            f"Possibly a returning visitor — face seen across "
            f"{visit_count} separate visits in this window. Please verify."
        )
    return f"Review prompt: {kind}. Please verify."


def _alert_severity(kind: str, similarity: float) -> str:
    if kind == "watchlist":
        if similarity >= 0.60:
            return "high"
        if similarity >= 0.45:
            return "warn"
        return "info"
    return "info"


def _next_alert_id() -> str:
    return f"A-{uuid.uuid4().hex[:10]}"


def _save_thumbnails_and_alerts(
    persons: list[_PersonState],
    *,
    out_root: Path,
    camera_id: str,
    fps: float,
    cosine_match: float,
    quality_min: float,
    watchlist: WatchlistMatcher,
    repeat_visit_threshold: int = 2,
    visit_gap_seconds: float = 30.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Persist per-person thumbnails and emit non-accusatory alerts."""
    persons_dir = out_root / "persons" / camera_id
    persons_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = out_root / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)

    person_records: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []

    for p in persons:
        if p.best_thumbnail is None:
            continue
        thumb_path = persons_dir / f"{p.person_id}.jpg"
        cv2.imwrite(str(thumb_path), p.best_thumbnail)
        # Also write the full frame at the best moment so the watchlist
        # seeder has enough surrounding context for SCRFD to redetect
        # the face on a fresh load. Tight 100-pixel thumbnails alone
        # often fail re-detection.
        full_path = persons_dir / f"{p.person_id}_full.jpg"
        if p.best_full_frame is not None:
            cv2.imwrite(str(full_path), p.best_full_frame)

        visits = compute_visit_count(p.appearance_frames, visit_gap_seconds, fps)
        is_repeat = visits >= repeat_visit_threshold
        wl_label, wl_sim = watchlist.match(p.centroid)

        unique_frames = sorted(set(p.appearance_frames))
        face_dwell = (len(unique_frames) / fps) if unique_frames else 0.0

        first_iso = p.appearance_iso[0] if p.appearance_iso else None
        last_iso = p.appearance_iso[-1] if p.appearance_iso else None
        first_frame_idx = unique_frames[0] if unique_frames else None
        last_frame_idx = unique_frames[-1] if unique_frames else None

        person_records.append(
            {
                "person_id": p.person_id,
                "first_seen_frame": first_frame_idx,
                "last_seen_frame": last_frame_idx,
                "first_seen": first_iso,
                "last_seen": last_iso,
                "face_appearances": len(unique_frames),
                "face_dwell_seconds_authoritative": round(face_dwell, 2),
                "embedding_count": p.embedding_count,
                "best_det_score": round(p.best_det_score, 4),
                "linked_tracker_ids": sorted(p.linked_tracker_ids),
                "visit_count": visits,
                "is_repeat": is_repeat,
                "watchlist_match": wl_label,
                "watchlist_similarity": round(wl_sim, 4),
                "thumbnail": str(thumb_path.relative_to(PROJECT_ROOT)),
            }
        )

        # --- emit alerts ---
        if wl_label is not None:
            alert_id = _next_alert_id()
            alert_jpg = alerts_dir / f"{alert_id}.jpg"
            if p.best_full_frame is not None:
                cv2.imwrite(str(alert_jpg), p.best_full_frame)
            alerts.append(
                {
                    "id": alert_id,
                    "type": "watchlist",
                    "camera_id": camera_id,
                    "person_id": p.person_id,
                    "watchlist_label": wl_label,
                    "similarity": round(wl_sim, 4),
                    "timestamp": last_iso,
                    "thumbnail": str(thumb_path.relative_to(PROJECT_ROOT)),
                    "frame_jpg": str(alert_jpg.relative_to(PROJECT_ROOT))
                    if p.best_full_frame is not None
                    else None,
                    "severity": _alert_severity("watchlist", wl_sim),
                    "copy": _alert_copy("watchlist", wl_label, wl_sim),
                }
            )
        if is_repeat:
            alert_id = _next_alert_id()
            alert_jpg = alerts_dir / f"{alert_id}.jpg"
            if p.best_full_frame is not None:
                cv2.imwrite(str(alert_jpg), p.best_full_frame)
            alerts.append(
                {
                    "id": alert_id,
                    "type": "repeat_visitor",
                    "camera_id": camera_id,
                    "person_id": p.person_id,
                    "visit_count": visits,
                    "timestamp": last_iso,
                    "thumbnail": str(thumb_path.relative_to(PROJECT_ROOT)),
                    "frame_jpg": str(alert_jpg.relative_to(PROJECT_ROOT))
                    if p.best_full_frame is not None
                    else None,
                    "severity": _alert_severity("repeat_visitor", 0.0),
                    "copy": _alert_copy("repeat_visitor", None, 0.0, visit_count=visits),
                }
            )

    return person_records, alerts


@dataclass
class CameraIdentityResult:
    camera_id: str
    area: str
    fps: float
    quality_min: float
    cosine_match: float
    sample_every_n_frames: int
    frames_processed: int
    faces_seen: int
    faces_quality_gated: int
    unique_visitors_count: int
    person_records: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    watchlist_entries: list[dict[str, Any]]
    json_path: Path
    elapsed_seconds: float


def _compute_per_person_dwell_from_tracks(
    tracks_by_frame: dict[int, dict[int, np.ndarray]],
    person_records: list[dict[str, Any]],
    fps: float,
) -> dict[str, float]:
    """Authoritative whole-scene dwell: union of frames in which any linked tracker_id is alive."""
    dwell: dict[str, float] = {}
    if not tracks_by_frame:
        return dwell
    # invert: frame_idx → set of tracker_ids alive
    by_frame_tids: dict[int, set[int]] = {
        f: set(d.keys()) for f, d in tracks_by_frame.items()
    }
    for rec in person_records:
        linked = set(rec["linked_tracker_ids"])
        if not linked:
            dwell[rec["person_id"]] = 0.0
            continue
        active = sum(1 for tids in by_frame_tids.values() if linked.intersection(tids))
        dwell[rec["person_id"]] = round(active / fps, 2)
    return dwell


def run_identity_for_camera(
    *,
    config: PipelineConfig,
    cam: CameraVideos,
    video: VideoProbe,
    window: ProcessingWindow,
    out_root: Path,
    face_app: Any,
    watchlist: WatchlistMatcher,
) -> CameraIdentityResult:
    identity_cfg = config.identity
    quality_min = float(identity_cfg.get("quality_min", 0.55))
    cosine_match = float(identity_cfg.get("cosine_match", 0.38))
    sample_every_n = max(1, int(identity_cfg.get("sample_every_n_frames", 5)))

    camera_id = cam.config.camera_id
    area = cam.config.area
    tracks_jsonl = out_root / "tracks" / f"{camera_id}.jsonl"
    if not tracks_jsonl.exists():
        raise RuntimeError(f"missing tracks JSONL for {camera_id}: {tracks_jsonl}")
    tracks_by_frame = _load_tracks_by_frame(tracks_jsonl)
    if not tracks_by_frame:
        raise RuntimeError(f"empty tracks JSONL for {camera_id}: {tracks_jsonl}")

    start_frame, end_frame = window.to_frame_range(video.fps, video.frame_count)
    total_window = end_frame - start_frame
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video.path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    cluster = PersonCluster(cosine_match=cosine_match)
    track_to_person_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    faces_seen = 0
    faces_quality_gated = 0
    sampled_frames = 0

    log.info(
        "[%s] identity — window %d..%d (%d frames, sampling every %d, q_min=%.2f, cos=%.2f)",
        camera_id, start_frame, end_frame, total_window,
        sample_every_n, quality_min, cosine_match,
    )

    t0 = time.perf_counter()
    for offset in range(total_window):
        ok, frame = cap.read()
        if not ok:
            log.warning("[%s] decode ended early at offset %d", camera_id, offset)
            break
        if offset % sample_every_n != 0:
            continue
        sampled_frames += 1
        frame_idx_global = start_frame + offset
        person_boxes = tracks_by_frame.get(frame_idx_global, {})

        faces = face_app.get(frame)
        faces_seen += len(faces)
        for f in faces:
            ds = float(f.det_score)
            if ds < quality_min:
                continue
            faces_quality_gated += 1
            tid, link_score = link_face_to_tracker(
                np.asarray(f.bbox, dtype=float), person_boxes
            )
            embedding = np.asarray(f.normed_embedding, dtype=np.float32)
            person = cluster.assign(embedding)
            if tid is not None:
                person.linked_tracker_ids.add(tid)
                track_to_person_votes[tid][person.person_id] += 1
            person.appearance_frames.append(frame_idx_global)
            wall_iso = wall_clock_for_frame(
                video.recording_start, frame_idx_global, video.fps
            ).isoformat(timespec="milliseconds")
            person.appearance_iso.append(wall_iso)
            if ds > person.best_det_score:
                person.best_det_score = ds
                person.best_face_bbox = np.asarray(f.bbox, dtype=float)
                person.best_frame_idx = frame_idx_global
                person.best_thumbnail = _crop_thumbnail(frame, np.asarray(f.bbox, dtype=float))
                person.best_full_frame = frame.copy()

        if sampled_frames % 50 == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "[%s] sampled %d frames (%.1f fps eff), %d faces seen, %d gated, %d persons",
                camera_id, sampled_frames,
                sampled_frames / max(elapsed, 1e-6),
                faces_seen, faces_quality_gated, cluster.count,
            )

    cap.release()

    # Pick the dominant person per tracker_id by majority vote
    dominant: dict[int, str] = {}
    for tid, votes in track_to_person_votes.items():
        if not votes:
            continue
        dominant[tid] = max(votes, key=votes.get)

    person_records, alerts = _save_thumbnails_and_alerts(
        cluster.persons,
        out_root=out_root,
        camera_id=camera_id,
        fps=video.fps,
        cosine_match=cosine_match,
        quality_min=quality_min,
        watchlist=watchlist,
    )
    # Replace each person's linked_tracker_ids with the dominant set
    # (i.e. tracker_ids whose majority-vote person == this person) so a
    # transient face misassignment doesn't bind a tracker_id forever.
    for rec in person_records:
        pid = rec["person_id"]
        rec["linked_tracker_ids"] = sorted(
            tid for tid, dom in dominant.items() if dom == pid
        )

    track_dwell = _compute_per_person_dwell_from_tracks(tracks_by_frame, person_records, video.fps)
    for rec in person_records:
        rec["track_dwell_seconds_authoritative"] = round(track_dwell.get(rec["person_id"], 0.0), 2)

    elapsed = time.perf_counter() - t0
    payload = {
        "version": 1,
        "camera_id": camera_id,
        "area": area,
        "fps": video.fps,
        "quality_min": quality_min,
        "cosine_match": cosine_match,
        "sample_every_n_frames": sample_every_n,
        "window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_processed": total_window,
            "frames_sampled": sampled_frames,
        },
        "faces_seen": faces_seen,
        "faces_quality_gated": faces_quality_gated,
        "unique_visitors_count": len(person_records),
        "unique_visitors_locked": False,
        "unique_visitors_note": (
            "Authoritative count from face-based identity (Phase 3). "
            "Tracker IDs (Phase 1/2 person_tracks) are no longer the source of truth."
        ),
        "persons": person_records,
        "watchlist": [
            {
                "label": e.label,
                "image_path": str(e.image_path.relative_to(PROJECT_ROOT))
                if str(e.image_path).startswith(str(PROJECT_ROOT))
                else str(e.image_path),
                "det_score": round(e.det_score, 4),
            }
            for e in watchlist.entries
        ],
        "watchlist_skipped": watchlist.skipped,
        "alerts": alerts,
        "elapsed_seconds": round(elapsed, 2),
    }

    json_path = out_root / "identity" / f"{camera_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return CameraIdentityResult(
        camera_id=camera_id,
        area=area,
        fps=video.fps,
        quality_min=quality_min,
        cosine_match=cosine_match,
        sample_every_n_frames=sample_every_n,
        frames_processed=total_window,
        faces_seen=faces_seen,
        faces_quality_gated=faces_quality_gated,
        unique_visitors_count=len(person_records),
        person_records=person_records,
        alerts=alerts,
        watchlist_entries=[
            {"label": e.label, "det_score": round(e.det_score, 4)} for e in watchlist.entries
        ],
        json_path=json_path,
        elapsed_seconds=elapsed,
    )


def run_identity(
    config: PipelineConfig,
    cameras: list[CameraVideos],
    window: ProcessingWindow,
    out_root: Path,
    *,
    watchlist_dir: Path,
    det_size: tuple[int, int] = (640, 640),
) -> list[CameraIdentityResult]:
    if not config.identity.get("enabled", True):
        log.warning("identity.enabled=false in config — skipping Phase 3 entirely.")
        return []
    face_app = _load_face_app(det_size)
    watchlist = WatchlistMatcher(
        watchlist_dir=watchlist_dir,
        face_app=face_app,
        threshold=float(config.identity.get("cosine_match", 0.38)),
        quality_min=float(config.identity.get("quality_min", 0.55)),
    )
    log.info(
        "watchlist: %d entries embedded, %d skipped (%s)",
        len(watchlist.entries), len(watchlist.skipped), watchlist_dir,
    )
    results: list[CameraIdentityResult] = []
    for cam in cameras:
        if not cam.videos:
            log.warning("[%s] no videos — skipping.", cam.config.camera_id)
            continue
        result = run_identity_for_camera(
            config=config,
            cam=cam,
            video=cam.videos[0],
            window=window,
            out_root=out_root,
            face_app=face_app,
            watchlist=watchlist,
        )
        results.append(result)
    return results


def summarize_results(results: list[CameraIdentityResult]) -> str:
    if not results:
        return "(no cameras processed)\n"
    lines = ["", "Phase 3 identity summary", "========================", ""]
    for r in results:
        repeats = sum(1 for p in r.person_records if p["is_repeat"])
        wl = sum(1 for p in r.person_records if p["watchlist_match"])
        gated_pct = (
            100.0 * r.faces_quality_gated / max(r.faces_seen, 1)
        )
        lines.append(f"[{r.camera_id}]  area: {r.area}")
        lines.append(
            f"  thresholds       : quality_min={r.quality_min:.2f}  "
            f"cosine_match={r.cosine_match:.2f}  sample_every_n={r.sample_every_n_frames}"
        )
        lines.append(
            f"  faces            : {r.faces_seen} seen / {r.faces_quality_gated} quality-gated "
            f"({gated_pct:.1f}%)"
        )
        lines.append(
            f"  unique visitors  : {r.unique_visitors_count}  "
            "(AUTHORITATIVE — unique_visitors_locked: false)"
        )
        lines.append(
            f"  repeats / watch  : {repeats} repeat / {wl} watchlist hit(s)"
        )
        lines.append(
            f"  alerts emitted   : {len(r.alerts)}"
        )
        lines.append(
            f"  watchlist entries: {len(r.watchlist_entries)}"
        )
        lines.append(
            f"  processing       : {r.elapsed_seconds:.1f}s"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def seed_watchlist_from_person(
    *,
    out_root: Path,
    watchlist_dir: Path,
    camera_id: str,
    person_id: str,
    label: str | None = None,
) -> Path:
    """Copy a person's best-quality face crop into ./watchlist/ for demo purposes.

    Useful for showing the watchlist alert flow without supplying real reference
    photos. The operator simply runs ``--seed-watchlist CAM PID [LABEL]`` after
    a Phase 3 run, then re-runs ``--run-identity`` to see the alert fire.

    We copy the FULL FRAME (``Pxxx_full.jpg``) rather than the tight
    thumbnail because SCRFD often fails to redetect a face on a tightly
    cropped image — it needs surrounding context.
    """
    candidates = [
        out_root / "persons" / camera_id / f"{person_id}_full.jpg",
        out_root / "persons" / camera_id / f"{person_id}.jpg",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"no thumbnail or full frame for {camera_id}/{person_id}: looked at {candidates}"
        )
    label = label or f"seeded_{camera_id}_{person_id}"
    dst = watchlist_dir / f"{label}.jpg"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


_ = (datetime, timedelta)  # silence unused-import warnings — used in callers
