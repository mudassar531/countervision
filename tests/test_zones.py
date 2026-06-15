"""Unit tests for the pure-numpy zone primitives (no torch / no supervision)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.zones import (  # noqa: E402
    HeatmapAccumulator,
    LineCrossing,
    PolygonZone,
    bottom_center_xy,
    load_tracks_jsonl,
    point_in_polygon,
    run_zone_analytics,
)
from tools.draw_zones import (  # noqa: E402
    default_entry_line_for_frame,
    default_zones_for_frame,
)


def _box(x: float, y: float, w: float = 40, h: float = 80) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=float)


class TestBottomCenter:
    def test_simple(self) -> None:
        bx, by = bottom_center_xy(_box(100, 200))
        assert bx == 120.0
        assert by == 280.0


class TestPointInPolygon:
    def test_inside(self) -> None:
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        assert point_in_polygon((5.0, 5.0), poly)

    def test_outside(self) -> None:
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        assert not point_in_polygon((50.0, 50.0), poly)

    def test_boundary_inclusive(self) -> None:
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        # cv2.pointPolygonTest returns 0 on the edge, we treat that as inside.
        assert point_in_polygon((10.0, 5.0), poly)


class TestLineCrossing:
    @pytest.fixture
    def horizontal_line(self) -> LineCrossing:
        # left→right line at y=100 → "above" the line is positive cross product.
        return LineCrossing(start=(0.0, 100.0), end=(1920.0, 100.0))

    def test_in_then_out(self, horizontal_line: LineCrossing) -> None:
        # frame 0: tid 1 is below (y_bottom=180) the line at y=100 → side −1
        horizontal_line.update(0, "t0", np.array([1]), np.array([_box(500, 110)]))  # y_bot=190
        assert horizontal_line.in_count == 0 and horizontal_line.out_count == 0
        # frame 1: tid 1 moved up so its bottom is now at y=80 → above → side +1 → "in"
        horizontal_line.update(1, "t1", np.array([1]), np.array([_box(500, 0)]))
        assert horizontal_line.in_count == 1 and horizontal_line.out_count == 0
        # frame 2: moves back down → side −1 → "out"
        horizontal_line.update(2, "t2", np.array([1]), np.array([_box(500, 110)]))
        assert horizontal_line.in_count == 1 and horizontal_line.out_count == 1
        assert len(horizontal_line.events) == 2

    def test_separate_ids_counted_independently(self, horizontal_line: LineCrossing) -> None:
        horizontal_line.update(0, "t0", np.array([1, 2]), np.array([_box(500, 200), _box(800, 200)]))
        # Both cross upwards → 2 "in"s
        horizontal_line.update(
            1, "t1", np.array([1, 2]), np.array([_box(500, 0), _box(800, 0)])
        )
        assert horizontal_line.in_count == 2
        assert horizontal_line.out_count == 0

    def test_no_crossing_when_only_one_side_seen(self, horizontal_line: LineCrossing) -> None:
        horizontal_line.update(0, "t0", np.array([1]), np.array([_box(500, 200)]))
        horizontal_line.update(1, "t1", np.array([1]), np.array([_box(520, 220)]))
        assert horizontal_line.in_count == 0
        assert horizontal_line.out_count == 0


class TestPolygonZone:
    def test_dwell_accumulates(self) -> None:
        zone = PolygonZone(
            name="z", polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        )
        # tid 1 inside the zone for 3 frames
        for frame in range(3):
            zone.update(frame, np.array([1]), np.array([_box(40, 0)]))
        # tid 1 leaves
        zone.update(3, np.array([1]), np.array([_box(500, 500)]))
        assert zone.frames_in_zone_by_track == {1: 3}
        dwell = zone.dwell_seconds_by_track(fps=25)
        assert dwell[1] == pytest.approx(3 / 25, abs=1e-2)

    def test_occupancy_peak(self) -> None:
        zone = PolygonZone(
            name="z", polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        )
        zone.update(
            0,
            np.array([1, 2, 3]),
            np.array([_box(10, 0), _box(40, 0), _box(70, 0)]),
        )
        zone.update(1, np.array([1]), np.array([_box(40, 0)]))
        assert zone.occupancy_peak == 3


class TestHeatmap:
    def test_render_returns_uint8_bgr_of_correct_size(self) -> None:
        hm = HeatmapAccumulator(width=200, height=120, radius=8)
        hm.add(np.array([_box(100, 0), _box(120, 0)]))
        out = hm.render()
        assert out.dtype == np.uint8
        assert out.shape == (120, 200, 3)

    def test_empty_accumulator_renders_black(self) -> None:
        hm = HeatmapAccumulator(width=80, height=80, radius=4)
        out = hm.render()
        assert out.sum() == 0

    def test_composited_onto_base_blends_in(self) -> None:
        hm = HeatmapAccumulator(width=80, height=80, radius=4)
        # box bottom must land inside the 80-tall frame for the
        # accumulator to register the contribution.
        hm.add(np.array([_box(40, 0, w=40, h=40)]))  # bottom_center = (60, 40)
        base = np.full((80, 80, 3), 200, dtype=np.uint8)
        out = hm.render(base_frame=base)
        # The hot pixel area should differ from a clean copy of base.
        assert (out != base).any()


class TestDefaultZones:
    def test_default_polygon_is_central_60(self) -> None:
        z = default_zones_for_frame(1920, 1080)
        assert len(z) == 1
        polygon = z[0]["polygon"]
        # margin = 20% so the polygon should span 20%..80% horizontally / vertically.
        assert polygon[0] == [384, 216]
        assert polygon[2] == [1536, 864]

    def test_default_entry_line_is_at_75pct_height(self) -> None:
        line = default_entry_line_for_frame(1920, 1080)
        assert line["start"][1] == 810  # 0.75 * 1080
        assert line["end"][1] == 810


class TestRunZoneAnalyticsEndToEnd:
    def test_full_flow_on_synthetic_tracks(self, tmp_path: Path) -> None:
        # Synthesize a small tracks JSONL with 2 IDs crossing a line at y=100.
        records = []
        for frame_idx in range(10):
            t = f"2026-06-07T20:53:{50 + frame_idx:02d}.000"
            # tid 1 walks down (top → bottom of frame)
            y1 = 0 + frame_idx * 20
            # tid 2 walks up (bottom → top)
            y2 = 200 - frame_idx * 20
            for tid, y in [(1, y1), (2, y2)]:
                records.append(
                    {
                        "camera_id": "camera-1",
                        "video": "synth.mp4",
                        "frame_idx": frame_idx,
                        "frame_offset": frame_idx,
                        "wall_clock": t,
                        "tracker_id": tid,
                        "xyxy": [50 if tid == 1 else 150, y, 90 if tid == 1 else 190, y + 60],
                        "conf": 0.9,
                    }
                )
        tracks_path = tmp_path / "tracks" / "camera-1.jsonl"
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        tracks_path.write_text("\n".join(json.dumps(r) for r in records))

        # Synthetic 320x240 base frame.
        import cv2

        frame_path = tmp_path / "frames" / "camera-1.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), np.full((240, 320, 3), 240, dtype=np.uint8))

        result = run_zone_analytics(
            camera_id="camera-1",
            area="Cosmetics & Skincare",
            tracks_jsonl=tracks_path,
            fps=25.0,
            frame_jpg=frame_path,
            out_root=tmp_path / "out",
            zones_cfg=[
                {
                    "name": "Main floor",
                    "polygon": [[60, 60], [260, 60], [260, 180], [60, 180]],
                }
            ],
            entry_line_cfg={"start": [0, 100], "end": [320, 100]},
            expected_videos=["synth.mp4", "another.mp4"],
        )

        assert result.frames_processed == 10
        assert result.person_tracks_count == 2
        # Both ids cross y=100 once → either 2 ins or one in + one out depending on direction.
        crossings = result.line_crossing.in_count + result.line_crossing.out_count
        assert crossings == 2
        # synth.mp4 is in tracks; another.mp4 is expected but missing → skipped.
        assert result.videos_considered == ["synth.mp4"]
        assert result.videos_skipped == ["another.mp4"]
        # JSON written; "unique_visitors_locked": true sentinel present.
        payload = json.loads(result.json_path.read_text())
        assert payload["unique_visitors_locked"] is True
        assert "unique_visitors_note" in payload
        assert payload["person_tracks"]["count"] == 2
        # Heatmap PNG exists + non-empty.
        assert result.heatmap_path.exists() and result.heatmap_path.stat().st_size > 0


class TestLoadTracksJsonl:
    def test_groups_by_frame_sorted(self, tmp_path: Path) -> None:
        recs = [
            {"camera_id": "c", "video": "v.mp4", "frame_idx": 2, "tracker_id": 1,
             "xyxy": [0, 0, 1, 1], "conf": 0.5, "wall_clock": "2026-06-07T20:53:55.000"},
            {"camera_id": "c", "video": "v.mp4", "frame_idx": 0, "tracker_id": 1,
             "xyxy": [0, 0, 1, 1], "conf": 0.5, "wall_clock": "2026-06-07T20:53:50.000"},
            {"camera_id": "c", "video": "v.mp4", "frame_idx": 0, "tracker_id": 2,
             "xyxy": [0, 0, 1, 1], "conf": 0.5, "wall_clock": "2026-06-07T20:53:50.000"},
        ]
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs))
        frames, meta = load_tracks_jsonl(p)
        assert [f.frame_idx for f in frames] == [0, 2]
        assert len(frames[0].ids) == 2
        assert meta["frame_range"] == (0, 2)
        assert meta["videos_in_file"] == ["v.mp4"]
