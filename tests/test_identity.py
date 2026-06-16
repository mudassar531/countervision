"""Unit tests for the pure-numpy identity primitives.

No insightface / onnxruntime needed — we exercise the cluster +
linker + watchlist matcher logic with synthetic L2-normalized
embeddings so CI stays light.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.identity import (  # noqa: E402
    PersonCluster,
    WatchlistMatcher,
    compute_visit_count,
    cosine_similarity,
    link_face_to_tracker,
)


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        v = _unit([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_are_zero(self) -> None:
        a = _unit([1.0, 0.0, 0.0])
        b = _unit([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_are_minus_one(self) -> None:
        a = _unit([1.0, 1.0, 1.0])
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)


class TestPersonCluster:
    def test_two_similar_embeddings_share_a_cluster(self) -> None:
        c = PersonCluster(cosine_match=0.40)
        e1 = _unit([1.0, 0.05, 0.0, 0.0])
        e2 = _unit([1.0, 0.10, 0.0, 0.0])  # cosine ≈ 0.9999
        p1 = c.assign(e1)
        p2 = c.assign(e2)
        assert p1.person_id == p2.person_id == "P001"
        assert c.count == 1
        assert p1.embedding_count == 2

    def test_orthogonal_embeddings_split(self) -> None:
        c = PersonCluster(cosine_match=0.40)
        c.assign(_unit([1.0, 0.0, 0.0]))
        c.assign(_unit([0.0, 1.0, 0.0]))
        c.assign(_unit([0.0, 0.0, 1.0]))
        assert c.count == 3
        assert [p.person_id for p in c.persons] == ["P001", "P002", "P003"]

    def test_below_threshold_spawns_new_cluster(self) -> None:
        c = PersonCluster(cosine_match=0.50)
        # Make two embeddings whose cosine is well under 0.50.
        c.assign(_unit([1.0, 0.1, 0.0, 0.0]))
        c.assign(_unit([0.1, 1.0, 0.0, 0.0]))  # cosine ≈ 0.20
        assert c.count == 2

    def test_centroid_drifts_toward_majority(self) -> None:
        c = PersonCluster(cosine_match=0.40)
        # Three nearly-identical embeddings; centroid should remain unit-norm.
        for _ in range(3):
            c.assign(_unit([1.0, 0.05, 0.05, 0.05]))
        p = c.persons[0]
        assert np.linalg.norm(p.centroid) == pytest.approx(1.0, abs=1e-5)


class TestLinkFaceToTracker:
    def test_face_inside_one_person_box(self) -> None:
        face = np.array([100, 100, 140, 140], dtype=float)
        person_boxes = {7: np.array([80, 60, 200, 400], dtype=float)}
        tid, score = link_face_to_tracker(face, person_boxes)
        assert tid == 7 and score > 0

    def test_face_with_no_matching_person_box(self) -> None:
        face = np.array([1000, 1000, 1040, 1040], dtype=float)
        person_boxes = {1: np.array([0, 0, 100, 200], dtype=float)}
        tid, score = link_face_to_tracker(face, person_boxes)
        assert tid is None and score == 0.0

    def test_picks_box_whose_head_region_overlaps_most(self) -> None:
        # Two person boxes both contain the face center; the one whose top
        # quarter (the head region) overlaps the face the most should win.
        face = np.array([200, 100, 250, 160], dtype=float)
        person_boxes = {
            1: np.array([180, 80, 270, 600], dtype=float),  # tall person, head region 80–210
            2: np.array([100, 0, 400, 500], dtype=float),   # very tall, head region 0–125
        }
        tid, _ = link_face_to_tracker(face, person_boxes)
        assert tid == 1


class TestVisitCount:
    def test_continuous_appearances_count_as_one_visit(self) -> None:
        assert compute_visit_count([0, 5, 10, 15], gap_seconds=30, fps=25) == 1

    def test_gap_above_threshold_counts_extra_visits(self) -> None:
        # 25 fps → 30 s gap = 750 frames apart
        assert compute_visit_count([0, 100, 1000], gap_seconds=30, fps=25) == 2

    def test_empty_returns_zero(self) -> None:
        assert compute_visit_count([], gap_seconds=30, fps=25) == 0


class _StubFace:
    def __init__(self, det_score: float, emb: np.ndarray) -> None:
        self.det_score = det_score
        self.normed_embedding = emb


class _StubFaceApp:
    """Returns one face per image whose embedding is keyed by filename."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping
        self._counter = 0

    def get(self, _img: np.ndarray) -> list[Any]:
        # Watchlist test calls .get once per image in directory order; we use
        # call ordering to map to embeddings since cv2.imread can't carry
        # filename into here.
        keys = list(self._mapping.keys())
        if self._counter >= len(keys):
            return []
        emb = self._mapping[keys[self._counter]]
        self._counter += 1
        return [_StubFace(det_score=0.9, emb=emb)]


class TestWatchlistMatcher:
    def test_self_match_passes(self, tmp_path: Path) -> None:
        import cv2

        emb_a = _unit([1.0, 0.1, 0.0])
        # Real (decodable) image — content doesn't matter; the stub keys
        # by call ordering, but cv2.imread must succeed for the entry to
        # be considered.
        cv2.imwrite(
            str(tmp_path / "alice.jpg"),
            np.full((32, 32, 3), 200, dtype=np.uint8),
        )
        wl = WatchlistMatcher(
            watchlist_dir=tmp_path,
            face_app=_StubFaceApp({"alice.jpg": emb_a}),
            threshold=0.40,
            quality_min=0.55,
        )
        assert len(wl.entries) == 1
        label, sim = wl.match(emb_a)
        assert label == "alice"
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_query_returns_no_match(self, tmp_path: Path) -> None:
        import cv2

        cv2.imwrite(
            str(tmp_path / "alice.jpg"),
            np.full((32, 32, 3), 200, dtype=np.uint8),
        )
        wl = WatchlistMatcher(
            watchlist_dir=tmp_path,
            face_app=_StubFaceApp({"alice.jpg": _unit([1.0, 0.0, 0.0])}),
            threshold=0.40,
            quality_min=0.55,
        )
        label, sim = wl.match(_unit([0.0, 1.0, 0.0]))
        assert label is None
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_empty_watchlist_returns_zero(self, tmp_path: Path) -> None:
        wl = WatchlistMatcher(
            watchlist_dir=tmp_path,
            face_app=_StubFaceApp({}),
            threshold=0.40,
            quality_min=0.55,
        )
        label, sim = wl.match(_unit([1.0, 0.0, 0.0]))
        assert label is None and sim == 0.0
