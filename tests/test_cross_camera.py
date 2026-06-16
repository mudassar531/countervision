"""Unit tests for cross-camera identity (Phase 4) — pure-numpy primitives."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from countervision.cross_camera import (  # noqa: E402
    UnionFind,
    build_store_wide_persons,
    find_cross_camera_links,
    load_identity_persons,
    run_cross_camera,
)


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _write_identity_json(
    path: Path,
    *,
    camera_id: str,
    area: str,
    persons: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "camera_id": camera_id,
        "area": area,
        "unique_visitors_count": len(persons),
        "persons": persons,
    }
    path.write_text(json.dumps(payload))


# --------------------------------------------------------------------------- #
# UnionFind
# --------------------------------------------------------------------------- #


class TestUnionFind:
    def test_singleton_components(self) -> None:
        uf = UnionFind()
        for x in ("a", "b", "c"):
            uf.add(x)
        comps = sorted(sorted(c) for c in uf.components())
        assert comps == [["a"], ["b"], ["c"]]

    def test_union_merges_two(self) -> None:
        uf = UnionFind()
        for x in ("a", "b", "c"):
            uf.add(x)
        uf.union("a", "b")
        comps = sorted(sorted(c) for c in uf.components())
        assert comps == [["a", "b"], ["c"]]

    def test_transitive_merge(self) -> None:
        uf = UnionFind()
        for x in ("a", "b", "c", "d"):
            uf.add(x)
        uf.union("a", "b")
        uf.union("b", "c")
        comps = sorted(sorted(c) for c in uf.components())
        assert comps == [["a", "b", "c"], ["d"]]


# --------------------------------------------------------------------------- #
# load_identity_persons
# --------------------------------------------------------------------------- #


class TestLoadIdentityPersons:
    def test_filters_low_appearance_persons(self, tmp_path: Path) -> None:
        emb = _unit([1.0, 0.0, 0.0])
        _write_identity_json(
            tmp_path / "camera-1.json",
            camera_id="camera-1",
            area="A",
            persons=[
                {"person_id": "P001", "face_appearances": 10,
                 "embedding_centroid": emb.tolist(),
                 "first_seen": "2026-06-07T20:53:50.000",
                 "last_seen":  "2026-06-07T20:54:50.000"},
                {"person_id": "P002", "face_appearances": 1,  # below gate of 3
                 "embedding_centroid": emb.tolist(),
                 "first_seen": "2026-06-07T20:55:00.000",
                 "last_seen":  "2026-06-07T20:55:01.000"},
            ],
        )
        persons, skipped = load_identity_persons(tmp_path, min_face_appearances=3)
        assert [p.person_id for p in persons] == ["P001"]
        assert len(skipped) == 1
        assert "face_appearances=1" in skipped[0]["reason"]

    def test_skips_persons_without_centroid(self, tmp_path: Path) -> None:
        _write_identity_json(
            tmp_path / "camera-1.json",
            camera_id="camera-1",
            area="A",
            persons=[
                {"person_id": "P001", "face_appearances": 10,
                 "embedding_centroid": None,
                 "first_seen": "2026-06-07T20:53:50.000",
                 "last_seen":  "2026-06-07T20:54:50.000"},
            ],
        )
        persons, skipped = load_identity_persons(tmp_path, min_face_appearances=3)
        assert persons == []
        assert "no embedding_centroid" in skipped[0]["reason"]


# --------------------------------------------------------------------------- #
# find_cross_camera_links + build_store_wide_persons
# --------------------------------------------------------------------------- #


def _camperson(cam: str, pid: str, embedding: np.ndarray, first: str, last: str,
               area: str = "Area") -> Any:
    from countervision.cross_camera import _CamPerson  # type: ignore

    return _CamPerson(
        camera_id=cam, person_id=pid, area=area,
        centroid=embedding, face_appearances=10,
        first_seen=first, last_seen=last,
    )


class TestFindCrossCameraLinks:
    def test_no_match_below_threshold(self) -> None:
        people = [
            _camperson("camera-1", "P001", _unit([1.0, 0.0, 0.0]),
                       "2026-06-07T20:53:50.000", "2026-06-07T20:54:50.000"),
            _camperson("camera-3", "P001", _unit([0.0, 1.0, 0.0]),
                       "2026-06-08T00:31:30.000", "2026-06-08T00:32:30.000"),
        ]
        links = find_cross_camera_links(people, threshold=0.50)
        assert links == []

    def test_one_match_above_threshold(self) -> None:
        emb = _unit([1.0, 0.05, 0.0])
        people = [
            _camperson("camera-1", "P001", emb,
                       "2026-06-07T20:53:50.000", "2026-06-07T20:54:50.000"),
            _camperson("camera-3", "P004", emb,
                       "2026-06-08T00:31:30.000", "2026-06-08T00:32:30.000"),
        ]
        links = find_cross_camera_links(people, threshold=0.50)
        assert len(links) == 1
        link = links[0]
        # Earliest first_seen is camera-1 → that should be `from`.
        assert link.from_camera == "camera-1"
        assert link.to_camera == "camera-3"
        assert link.similarity == pytest.approx(1.0, abs=1e-5)
        assert "h" in link.time_gap or "m" in link.time_gap
        assert "repeat presence" in link.presence_note

    def test_same_camera_pair_ignored(self) -> None:
        emb = _unit([1.0, 0.0, 0.0])
        people = [
            _camperson("camera-1", "P001", emb,
                       "2026-06-07T20:53:50.000", "2026-06-07T20:54:50.000"),
            _camperson("camera-1", "P002", emb,
                       "2026-06-07T20:55:00.000", "2026-06-07T20:55:01.000"),
        ]
        links = find_cross_camera_links(people, threshold=0.50)
        assert links == []


class TestBuildStoreWidePersons:
    def test_no_links_means_each_person_is_own_component(self) -> None:
        people = [
            _camperson("camera-1", "P001", _unit([1.0, 0.0, 0.0]),
                       "2026-06-07T20:53:50.000", "2026-06-07T20:54:50.000"),
            _camperson("camera-3", "P001", _unit([0.0, 1.0, 0.0]),
                       "2026-06-08T00:31:30.000", "2026-06-08T00:32:30.000"),
        ]
        store_wide = build_store_wide_persons(people, links=[])
        assert len(store_wide) == 2
        # IDs are renumbered by earliest first-seen.
        assert store_wide[0].store_person_id == "S001"
        assert store_wide[0].members[0]["camera_id"] == "camera-1"

    def test_transitive_merge_across_three_cameras(self) -> None:
        emb = _unit([1.0, 0.05, 0.0])
        people = [
            _camperson("camera-1", "P001", emb,
                       "2026-06-07T20:53:50.000", "2026-06-07T20:54:50.000"),
            _camperson("camera-3", "P004", emb,
                       "2026-06-08T00:31:30.000", "2026-06-08T00:32:30.000"),
            _camperson("camera-5", "P006", emb,
                       "2026-06-08T04:44:50.000", "2026-06-08T04:45:50.000"),
        ]
        links = find_cross_camera_links(people, threshold=0.50)
        store_wide = build_store_wide_persons(people, links=links)
        # All three people are the same → one store-wide visitor with
        # members across all three cameras.
        assert len(store_wide) == 1
        assert {m["camera_id"] for m in store_wide[0].members} == {
            "camera-1", "camera-3", "camera-5"
        }
        assert store_wide[0].areas_visited  # populated


# --------------------------------------------------------------------------- #
# run_cross_camera (full driver)
# --------------------------------------------------------------------------- #


class TestRunCrossCamera:
    def test_no_reliable_match_path(self, tmp_path: Path) -> None:
        idir = tmp_path / "identity"
        _write_identity_json(
            idir / "camera-1.json",
            camera_id="camera-1",
            area="A",
            persons=[{
                "person_id": "P001", "face_appearances": 10,
                "embedding_centroid": _unit([1.0, 0.0, 0.0]).tolist(),
                "first_seen": "2026-06-07T20:53:50.000",
                "last_seen":  "2026-06-07T20:54:50.000",
            }],
        )
        _write_identity_json(
            idir / "camera-3.json",
            camera_id="camera-3",
            area="B",
            persons=[{
                "person_id": "P001", "face_appearances": 10,
                "embedding_centroid": _unit([0.0, 1.0, 0.0]).tolist(),
                "first_seen": "2026-06-08T00:31:30.000",
                "last_seen":  "2026-06-08T00:32:30.000",
            }],
        )
        result = run_cross_camera(
            identity_dir=idir,
            out_root=tmp_path,
            cross_camera_match=0.50,
            in_camera_cluster=0.32,
            min_face_appearances=3,
        )
        assert result.no_reliable_cross_camera_matches is True
        assert result.store_wide_unique_visitors == 2  # = naive sum
        assert result.saved_by_cross_camera_dedup == 0
        payload = json.loads(result.json_path.read_text())
        assert payload["headline"]["no_reliable_cross_camera_matches"] is True
        assert "No reliable" in payload["headline"]["headline_message"]

    def test_match_dedups(self, tmp_path: Path) -> None:
        idir = tmp_path / "identity"
        emb = _unit([1.0, 0.05, 0.0]).tolist()
        _write_identity_json(
            idir / "camera-3.json",
            camera_id="camera-3",
            area="Customer Seating",
            persons=[{
                "person_id": "P004", "face_appearances": 30,
                "embedding_centroid": emb,
                "first_seen": "2026-06-08T00:31:30.000",
                "last_seen":  "2026-06-08T00:34:30.000",
            }],
        )
        _write_identity_json(
            idir / "camera-5.json",
            camera_id="camera-5",
            area="Service Desk",
            persons=[{
                "person_id": "P006", "face_appearances": 30,
                "embedding_centroid": emb,
                "first_seen": "2026-06-08T04:44:50.000",
                "last_seen":  "2026-06-08T04:48:50.000",
            }],
        )
        result = run_cross_camera(
            identity_dir=idir,
            out_root=tmp_path,
            cross_camera_match=0.50,
            in_camera_cluster=0.32,
            min_face_appearances=3,
        )
        assert result.no_reliable_cross_camera_matches is False
        assert len(result.cross_camera_links) == 1
        assert result.cross_camera_links[0].similarity == pytest.approx(1.0, abs=1e-5)
        assert result.naive_total_per_camera_sum == 2
        assert result.store_wide_unique_visitors == 1
        assert result.saved_by_cross_camera_dedup == 1
        # Time gap should mention hours since the cameras are 4+ hours apart.
        assert "h" in result.cross_camera_links[0].time_gap

    def test_low_appearance_persons_count_as_independent_visitors(
        self, tmp_path: Path
    ) -> None:
        """Persons skipped from the matching pool still count toward the headline."""
        idir = tmp_path / "identity"
        emb_match = _unit([1.0, 0.05, 0.0]).tolist()
        _write_identity_json(
            idir / "camera-3.json",
            camera_id="camera-3",
            area="C3",
            persons=[
                {"person_id": "P001", "face_appearances": 30,
                 "embedding_centroid": emb_match,
                 "first_seen": "2026-06-08T00:31:30.000",
                 "last_seen":  "2026-06-08T00:34:30.000"},
                # Low appearances → skipped from matching, but still counted
                # as a visitor (we don't pretend it doesn't exist).
                {"person_id": "P099", "face_appearances": 1,
                 "embedding_centroid": _unit([1.0, 0.0, 0.0]).tolist(),
                 "first_seen": "2026-06-08T00:32:30.000",
                 "last_seen":  "2026-06-08T00:32:31.000"},
            ],
        )
        _write_identity_json(
            idir / "camera-5.json",
            camera_id="camera-5",
            area="C5",
            persons=[{
                "person_id": "P006", "face_appearances": 30,
                "embedding_centroid": emb_match,
                "first_seen": "2026-06-08T04:44:50.000",
                "last_seen":  "2026-06-08T04:48:50.000",
            }],
        )
        result = run_cross_camera(
            identity_dir=idir,
            out_root=tmp_path,
            cross_camera_match=0.50,
            in_camera_cluster=0.32,
            min_face_appearances=3,
        )
        # naive sum = 3 (2 from cam-3 + 1 from cam-5); the matched pair
        # collapses 2 → 1; P099 stays as its own visitor; total = 2.
        assert result.naive_total_per_camera_sum == 3
        assert result.store_wide_unique_visitors == 2
