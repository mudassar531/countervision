"""Tests for the Phase 5 aggregator (no model deps)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.aggregate import (  # noqa: E402
    SCHEMA_VERSION,
    _generate_insights,
    aggregate,
)

# --------------------------------------------------------------------------- #
# Synthetic Phase 1-4 inputs
# --------------------------------------------------------------------------- #


def _make_inputs(
    tmp_path: Path,
    *,
    include_cross_camera: bool = True,
    cross_camera_links_count: int = 1,
) -> Path:
    """Build a minimal but realistic data/output/ tree under ``tmp_path``."""
    out = tmp_path / "out"
    (out / "zones").mkdir(parents=True)
    (out / "identity").mkdir(parents=True)
    (out / "frames").mkdir(parents=True)
    (out / "heatmaps").mkdir(parents=True)

    # Two cameras, each with a zone JSON and an identity JSON.
    for cam_id, area, visitors_count, peak_occ in [
        ("camera-A", "Cosmetics & Skincare", 2, 3),
        ("camera-B", "Customer Seating", 8, 2),
    ]:
        (out / "zones" / f"{cam_id}.json").write_text(json.dumps({
            "version": 1, "camera_id": cam_id, "area": area, "fps": 25,
            "frame_jpg": f"frames/{cam_id}.jpg",
            "heatmap_png": f"heatmaps/{cam_id}.png",
            "footfall": {
                "entry_line": {"start": [0, 100], "end": [100, 100], "anchor": "bottom_center"},
                "in_count": 0, "out_count": 0,
                "events": [],
            },
            "footfall_by_hour": [],
            "zones": [{
                "name": "Main floor", "color": "#0A1347",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "occupancy_peak": peak_occ,
                "dwell_seconds_by_track_provisional": {},
                "avg_dwell_seconds_provisional": 0.0,
                "active_tracker_ids": [],
                "provisional_note": "...",
            }],
            "occupancy_timeseries": [
                {"t": "2026-06-08T00:31:30.000", "frame_idx": 0, "second_bucket": 0, "active_tracks": 1}
            ],
        }))
        persons = [
            {
                "person_id": f"P{i + 1:03d}",
                "first_seen": "2026-06-08T00:31:30.000",
                "last_seen":  "2026-06-08T00:34:30.000",
                "face_appearances": 20,
                "face_dwell_seconds_authoritative": 0.8 * (i + 1),
                "embedding_count": 20,
                "best_det_score": 0.85,
                "linked_tracker_ids": [i + 1],
                "visit_count": 2 if i == 0 else 1,
                "is_repeat": i == 0,
                "watchlist_match": None,
                "watchlist_similarity": 0.0,
                "thumbnail": f"persons/{cam_id}/P{i + 1:03d}.jpg",
                "embedding_centroid": [0.0] * 512,
                "track_dwell_seconds_authoritative": 100 + i * 10,
            }
            for i in range(visitors_count)
        ]
        # one alert in camera-B
        alerts: list[dict[str, Any]] = []
        if cam_id == "camera-B":
            alerts.append({
                "id": "A-test01", "type": "watchlist", "camera_id": cam_id,
                "person_id": "P001", "watchlist_label": "demo_watchlist",
                "similarity": 0.55, "timestamp": "2026-06-08T00:34:10.000",
                "thumbnail": "...", "frame_jpg": "...",
                "severity": "warn", "copy": "Possible match ... Please verify before acting.",
            })
        (out / "identity" / f"{cam_id}.json").write_text(json.dumps({
            "version": 1, "camera_id": cam_id, "area": area, "fps": 25,
            "quality_min": 0.55, "cosine_match": 0.32, "sample_every_n_frames": 5,
            "window": {"start_frame": 0, "end_frame": 4500, "frames_processed": 4500, "frames_sampled": 900},
            "faces_seen": visitors_count * 20, "faces_quality_gated": visitors_count * 20,
            "unique_visitors_count": visitors_count, "unique_visitors_locked": False,
            "unique_visitors_note": "...",
            "persons": persons,
            "watchlist": [{"label": "demo_watchlist", "image_path": "watchlist/demo_watchlist.jpg", "det_score": 0.81}],
            "watchlist_skipped": [],
            "alerts": alerts,
            "elapsed_seconds": 100.0,
        }))

    if include_cross_camera:
        links = [
            {
                "from": {"camera_id": "camera-A", "person_id": "P001",
                         "area": "Cosmetics & Skincare",
                         "first_seen": "2026-06-08T00:31:30.000",
                         "last_seen":  "2026-06-08T00:34:30.000"},
                "to":   {"camera_id": "camera-B", "person_id": "P001",
                         "area": "Customer Seating",
                         "first_seen": "2026-06-08T01:00:00.000",
                         "last_seen":  "2026-06-08T01:03:00.000"},
                "similarity": 0.60, "time_gap": "26 m",
                "presence_note": "Same face appears in ... repeat presence ...",
            }
        ][:cross_camera_links_count]
        no_reliable = len(links) == 0
        (out / "cross_camera.json").write_text(json.dumps({
            "version": 1,
            "headline": {
                "store_wide_unique_visitors": 9 if not no_reliable else 10,
                "naive_total_per_camera_sum": 10,
                "saved_by_cross_camera_dedup": 1 if not no_reliable else 0,
                "cross_camera_links_count": len(links),
                "no_reliable_cross_camera_matches": no_reliable,
                "headline_message": "...",
            },
            "thresholds": {
                "in_camera_cluster": 0.32, "cross_camera_match": 0.50,
                "min_face_appearances_for_cross_camera": 3,
            },
            "per_camera_unique": {"camera-A": 2, "camera-B": 8},
            "persons_considered": 10, "persons_skipped": [],
            "cross_camera_links": links,
            "store_wide_persons": [],
        }))
    return out


# --------------------------------------------------------------------------- #
# aggregate() — end-to-end on synthetic inputs
# --------------------------------------------------------------------------- #


class TestAggregate:
    def test_schema_basics(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path)
        res = aggregate(output_root=out, store_name="Demo Store")
        d = json.loads(res.analytics_path.read_text())
        assert d["version"] == SCHEMA_VERSION
        assert d["store"]["name"] == "Demo Store"
        assert sorted(d["store"]["cameras"]) == ["camera-A", "camera-B"]
        # Areas carry authoritative dwell
        a = next(a for a in d["areas"] if a["camera_id"] == "camera-B")
        assert a["unique_visitors"] == 8
        assert a["avg_dwell_seconds"] > 0
        assert a["max_dwell_seconds"] >= a["avg_dwell_seconds"]

    def test_locked_fields_are_emitted(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path)
        res = aggregate(output_root=out, store_name="Demo Store")
        d = json.loads(res.analytics_path.read_text())
        for key in ("conversion_rate", "revenue_uplift", "weather",
                    "staffing_recommendations_quantified"):
            block = d["kpis"][key]
            assert block["value"] is None
            assert block["locked"] is True
            assert isinstance(block["reason"], str) and block["reason"]

    def test_cross_camera_carries_hedging(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path, cross_camera_links_count=1)
        res = aggregate(output_root=out, store_name="Demo Store")
        d = json.loads(res.analytics_path.read_text())
        assert d["cross_camera"] is not None
        assert "render_hint" in d["cross_camera"]
        assert "do not render these as continuous" in d["cross_camera"]["render_hint"]
        # store-wide KPI carries hedging
        block = d["kpis"]["store_wide_unique_visitors"]
        assert block["confidence"] in ("medium", "low")
        assert "saved_by_dedup" in block
        assert "method" in block

    def test_no_reliable_cross_camera_falls_back(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path, cross_camera_links_count=0)
        res = aggregate(output_root=out, store_name="Demo Store")
        d = json.loads(res.analytics_path.read_text())
        block = d["kpis"]["store_wide_unique_visitors"]
        assert block["confidence"] == "low"
        assert block["no_reliable_cross_camera_matches"] is True

    def test_alerts_get_confidence_note(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path)
        res = aggregate(output_root=out, store_name="Demo Store")
        d = json.loads(res.analytics_path.read_text())
        wl_alerts = [a for a in d["alerts"] if a.get("type") == "watchlist"]
        assert len(wl_alerts) == 1
        assert "verification prompt" in wl_alerts[0]["confidence_note"]
        assert wl_alerts[0]["confidence_level"] in ("low", "medium")

    def test_sqlite_mirror_populated(self, tmp_path: Path) -> None:
        out = _make_inputs(tmp_path)
        res = aggregate(output_root=out, store_name="Demo Store")
        conn = sqlite3.connect(str(res.sqlite_path))
        try:
            assert conn.execute("SELECT COUNT(*) FROM areas").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0] == 10
            assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM cross_camera_links").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0] >= 1
        finally:
            conn.close()


# --------------------------------------------------------------------------- #
# _generate_insights — fires only on reliable numbers
# --------------------------------------------------------------------------- #


def _area(name: str, *, unique: int, avg_dwell: float = 0.0,
          max_dwell: float = 0.0, peak: int = 0) -> dict[str, Any]:
    return {
        "camera_id": f"cam-{name}", "area": name,
        "unique_visitors": unique, "footfall_in": 0, "footfall_out": 0,
        "footfall_total": 0, "avg_dwell_seconds": avg_dwell,
        "max_dwell_seconds": max_dwell, "occupancy_peak": peak,
        "occupancy_timeseries": [],
    }


class TestInsights:
    def test_no_areas_returns_no_insights(self) -> None:
        assert _generate_insights(areas=[], visitors=[], store_wide_block={}) == []

    def test_highest_dwell_insight(self) -> None:
        areas = [
            _area("A", unique=3, avg_dwell=30, max_dwell=30, peak=1),
            _area("B", unique=3, avg_dwell=80, max_dwell=120, peak=1),
        ]
        insights = _generate_insights(areas=areas, visitors=[], store_wide_block={})
        ids = {i["id"] for i in insights}
        assert "highest_dwell_area" in ids
        top = next(i for i in insights if i["id"] == "highest_dwell_area")
        assert "B" in top["title"]

    def test_imbalance_only_when_clearly_imbalanced(self) -> None:
        balanced = [_area("A", unique=4, avg_dwell=10), _area("B", unique=4, avg_dwell=10)]
        insights = _generate_insights(areas=balanced, visitors=[], store_wide_block={})
        assert "area_engagement_imbalance" not in {i["id"] for i in insights}

        imbalanced = [_area("A", unique=8, avg_dwell=10), _area("B", unique=2, avg_dwell=10)]
        insights = _generate_insights(areas=imbalanced, visitors=[], store_wide_block={})
        assert "area_engagement_imbalance" in {i["id"] for i in insights}

    def test_demo_headline_needs_5_visitors(self) -> None:
        areas = [_area("A", unique=3, avg_dwell=10, max_dwell=20)]
        ids = {i["id"] for i in _generate_insights(areas=areas, visitors=[], store_wide_block={})}
        assert "demo_headline_framing" not in ids

        areas = [_area("A", unique=10, avg_dwell=10, max_dwell=180)]
        ids = {i["id"] for i in _generate_insights(areas=areas, visitors=[], store_wide_block={})}
        assert "demo_headline_framing" in ids

    def test_repeat_visitor_insight_requires_two_repeats(self) -> None:
        areas = [_area("A", unique=5, avg_dwell=10, max_dwell=20)]
        visitors_one = [{"is_repeat": True}]
        ids = {i["id"] for i in _generate_insights(areas=areas, visitors=visitors_one, store_wide_block={})}
        assert "repeat_visitor_opportunity" not in ids

        visitors_two = [{"is_repeat": True}, {"is_repeat": True}, {"is_repeat": False}]
        ids = {i["id"] for i in _generate_insights(areas=areas, visitors=visitors_two, store_wide_block={})}
        assert "repeat_visitor_opportunity" in ids

    def test_no_insight_built_on_cross_camera(self) -> None:
        """Insights ignore the cross-camera block entirely — by design."""
        areas = [_area("A", unique=10, avg_dwell=50, max_dwell=200, peak=4)]
        block = {"value": 99, "confidence": "low", "no_reliable_cross_camera_matches": True}
        insights = _generate_insights(areas=areas, visitors=[], store_wide_block=block)
        # None of the insight bodies should mention store-wide / cross-camera
        # framing — we want them tied to per-area numbers only.
        for i in insights:
            assert "cross-camera" not in i["detail"].lower()
            assert "store-wide" not in i["detail"].lower()


@pytest.mark.parametrize(
    "value,confidence",
    [(0, "low"), (1, "low"), (4, "low"), (5, "medium"), (100, "medium")],
)
def test_footfall_confidence_threshold(
    tmp_path: Path, value: int, confidence: str
) -> None:
    """Footfall confidence flips from 'low' to 'medium' at value >= 5."""
    out = _make_inputs(tmp_path)
    # Patch zones so footfall_total equals `value` (split between cameras).
    z_path = out / "zones" / "camera-A.json"
    z = json.loads(z_path.read_text())
    z["footfall"]["in_count"] = value
    z_path.write_text(json.dumps(z))
    res = aggregate(output_root=out, store_name="Demo")
    d = json.loads(res.analytics_path.read_text())
    assert d["kpis"]["footfall_total"]["value"] == value
    assert d["kpis"]["footfall_total"]["confidence"] == confidence
