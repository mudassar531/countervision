"""Phase 4 — cross-camera identity (de-dup, not journey).

Reads each camera's ``identity/<camera>.json`` (Phase 3 output) and
matches the saved L2-normalized centroids across cameras at a
**deliberately higher cosine threshold** than the in-camera clustering
cutoff. Cross-camera precision matters more than recall for the
store-wide unique-visitor number — a false-merge across cameras is a
much worse demo failure than missing a true match.

Honest framing for these specific videos: the per-camera recordings
**do not overlap in time** (camera-1 at 20:53 on 2026-06-07, camera-3
at 00:31 / 00:54 on 2026-06-08, camera-5 at 04:44 on 2026-06-08 — gaps
of hours between cameras). A high-confidence cross-camera face match
is therefore "**same face seen in these areas across the captured
period**" — repeat presence — *not* a single continuous trip. Each
emitted link makes that explicit.

Headline metric: ``store_wide_unique_visitors`` = number of
connected components in the cross-camera match graph. If no pair
clears the high threshold we honestly emit
``no_reliable_cross_camera_matches: true`` and the store-wide count
falls back to the sum of the per-camera counts (no fabricated links).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .discover import PROJECT_ROOT

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Union-Find (pure, testable, no external deps)
# --------------------------------------------------------------------------- #


class UnionFind:
    """Minimal disjoint-set union with union-by-rank + path compression."""

    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}
        self.rank: dict[Any, int] = {}

    def add(self, x: Any) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Any) -> Any:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Any, y: Any) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> list[list[Any]]:
        groups: dict[Any, list[Any]] = {}
        for node in self.parent:
            groups.setdefault(self.find(node), []).append(node)
        return list(groups.values())


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CamPerson:
    camera_id: str
    person_id: str
    area: str
    centroid: np.ndarray            # (512,), L2-normalized
    face_appearances: int
    first_seen: str | None
    last_seen: str | None


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _format_gap(a: datetime | None, b: datetime | None) -> str:
    if a is None or b is None:
        return "unknown gap"
    delta = abs(b - a)
    total_seconds = int(delta.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if hours > 0:
        return f"{hours} h {minutes} m"
    return f"{minutes} m"


def load_identity_persons(
    identity_dir: Path,
    *,
    min_face_appearances: int,
) -> tuple[list[_CamPerson], list[dict[str, Any]]]:
    """Return ``(persons, skipped)`` for cross-camera matching.

    Each ``identity/<camera>.json`` from Phase 3 is read; persons whose
    ``face_appearances < min_face_appearances`` are excluded from the
    matching pool (their centroids are too noisy to trust across
    cameras), but the skip is recorded in the returned ``skipped`` list
    so we can report it honestly.
    """
    persons: list[_CamPerson] = []
    skipped: list[dict[str, Any]] = []
    for json_path in sorted(identity_dir.glob("*.json")):
        if json_path.name.startswith("phase"):
            continue  # skip phase3_summary.json
        data = json.loads(json_path.read_text())
        camera_id = data.get("camera_id") or json_path.stem
        area = data.get("area", camera_id)
        for p in data.get("persons", []):
            centroid_list = p.get("embedding_centroid")
            if not centroid_list:
                skipped.append(
                    {
                        "camera_id": camera_id,
                        "person_id": p["person_id"],
                        "reason": "no embedding_centroid in identity JSON",
                    }
                )
                continue
            face_appearances = int(p.get("face_appearances", 0))
            if face_appearances < min_face_appearances:
                skipped.append(
                    {
                        "camera_id": camera_id,
                        "person_id": p["person_id"],
                        "reason": (
                            f"face_appearances={face_appearances} < "
                            f"{min_face_appearances} (centroid too noisy)"
                        ),
                    }
                )
                continue
            persons.append(
                _CamPerson(
                    camera_id=camera_id,
                    person_id=p["person_id"],
                    area=area,
                    centroid=np.asarray(centroid_list, dtype=np.float32),
                    face_appearances=face_appearances,
                    first_seen=p.get("first_seen"),
                    last_seen=p.get("last_seen"),
                )
            )
    return persons, skipped


# --------------------------------------------------------------------------- #
# Cross-camera matching + presence framing
# --------------------------------------------------------------------------- #


@dataclass
class CrossCameraLink:
    from_camera: str
    from_person: str
    from_area: str
    from_first_seen: str | None
    from_last_seen: str | None
    to_camera: str
    to_person: str
    to_area: str
    to_first_seen: str | None
    to_last_seen: str | None
    similarity: float
    time_gap: str
    presence_note: str


def _presence_note(
    a: _CamPerson, b: _CamPerson, similarity: float, time_gap: str
) -> str:
    """Phrase the link as repeat presence across non-overlapping windows."""
    return (
        f"Same face appears in '{a.area}' ({a.camera_id}, last seen "
        f"{a.last_seen or '?'}) and in '{b.area}' ({b.camera_id}, first seen "
        f"{b.first_seen or '?'}). Recording windows do not overlap "
        f"(gap ≈ {time_gap}), so this represents the same person being seen "
        "in these areas across the captured period — repeat presence, not a "
        f"single continuous trip. Cosine similarity {similarity:.2f}."
    )


def find_cross_camera_links(
    persons: list[_CamPerson],
    threshold: float,
) -> list[CrossCameraLink]:
    """Pairwise cross-camera cosine matching above ``threshold``."""
    links: list[CrossCameraLink] = []
    for i, a in enumerate(persons):
        for b in persons[i + 1:]:
            if a.camera_id == b.camera_id:
                continue
            sim = float(np.dot(a.centroid, b.centroid))
            if sim < threshold:
                continue
            # Order each link earliest → latest by first_seen so the
            # presence note reads chronologically.
            ta, tb = _parse_iso(a.first_seen), _parse_iso(b.first_seen)
            if ta and tb and tb < ta:
                a, b = b, a
            gap = _format_gap(_parse_iso(a.last_seen), _parse_iso(b.first_seen))
            links.append(
                CrossCameraLink(
                    from_camera=a.camera_id,
                    from_person=a.person_id,
                    from_area=a.area,
                    from_first_seen=a.first_seen,
                    from_last_seen=a.last_seen,
                    to_camera=b.camera_id,
                    to_person=b.person_id,
                    to_area=b.area,
                    to_first_seen=b.first_seen,
                    to_last_seen=b.last_seen,
                    similarity=sim,
                    time_gap=gap,
                    presence_note=_presence_note(a, b, sim, gap),
                )
            )
    links.sort(key=lambda link: link.similarity, reverse=True)
    return links


@dataclass
class StoreWidePerson:
    store_person_id: str
    members: list[dict[str, Any]] = field(default_factory=list)
    areas_visited: list[str] = field(default_factory=list)
    first_seen_overall: str | None = None
    last_seen_overall: str | None = None


def build_store_wide_persons(
    persons: list[_CamPerson],
    links: list[CrossCameraLink],
) -> list[StoreWidePerson]:
    """Connected components → one store-wide visitor each."""
    uf: UnionFind = UnionFind()
    for p in persons:
        uf.add((p.camera_id, p.person_id))
    for link in links:
        uf.union((link.from_camera, link.from_person), (link.to_camera, link.to_person))

    by_key: dict[tuple[str, str], _CamPerson] = {(p.camera_id, p.person_id): p for p in persons}

    out: list[StoreWidePerson] = []
    for i, comp in enumerate(uf.components(), start=1):
        members_p: list[_CamPerson] = [by_key[k] for k in comp]
        members_p.sort(
            key=lambda m: _parse_iso(m.first_seen) or datetime.max
        )
        first_seens = [_parse_iso(m.first_seen) for m in members_p if _parse_iso(m.first_seen)]
        last_seens = [_parse_iso(m.last_seen) for m in members_p if _parse_iso(m.last_seen)]
        store_person = StoreWidePerson(
            store_person_id=f"S{i:03d}",
            members=[
                {
                    "camera_id": m.camera_id,
                    "person_id": m.person_id,
                    "area": m.area,
                    "first_seen": m.first_seen,
                    "last_seen": m.last_seen,
                    "face_appearances": m.face_appearances,
                }
                for m in members_p
            ],
            areas_visited=sorted({m.area for m in members_p}),
            first_seen_overall=min(first_seens).isoformat(timespec="milliseconds")
            if first_seens
            else None,
            last_seen_overall=max(last_seens).isoformat(timespec="milliseconds")
            if last_seens
            else None,
        )
        out.append(store_person)
    out.sort(key=lambda s: s.first_seen_overall or "")
    # Renumber S001.. so order matches first-seen
    for i, s in enumerate(out, start=1):
        s.store_person_id = f"S{i:03d}"
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


@dataclass
class CrossCameraResult:
    threshold: float
    in_camera_cluster_threshold: float
    min_face_appearances: int
    persons_considered: int
    persons_skipped: list[dict[str, Any]]
    per_camera_unique: dict[str, int]
    naive_total_per_camera_sum: int
    store_wide_unique_visitors: int
    saved_by_cross_camera_dedup: int
    cross_camera_links: list[CrossCameraLink]
    store_wide_persons: list[StoreWidePerson]
    no_reliable_cross_camera_matches: bool
    json_path: Path


def run_cross_camera(
    *,
    identity_dir: Path,
    out_root: Path,
    cross_camera_match: float,
    in_camera_cluster: float,
    min_face_appearances: int,
) -> CrossCameraResult:
    persons, skipped = load_identity_persons(
        identity_dir, min_face_appearances=min_face_appearances
    )
    per_camera_counts = _per_camera_counts_from_jsons(identity_dir)

    if not persons:
        log.warning("Phase 4: no persons with embedding_centroid loaded.")
    links = find_cross_camera_links(persons, cross_camera_match)
    store_wide = build_store_wide_persons(persons, links)

    naive_total = sum(per_camera_counts.values())
    store_wide_count = (
        # Persons with >= min_appearances form components; everyone with too
        # few appearances counts as their own visitor (we couldn't reliably
        # match them, so we don't dedupe them).
        len(store_wide)
        + (naive_total - sum(len(s.members) for s in store_wide))
    )
    saved = max(0, naive_total - store_wide_count)
    no_reliable = len(links) == 0

    payload = {
        "version": 1,
        "headline": {
            "store_wide_unique_visitors": store_wide_count,
            "naive_total_per_camera_sum": naive_total,
            "saved_by_cross_camera_dedup": saved,
            "cross_camera_links_count": len(links),
            "no_reliable_cross_camera_matches": no_reliable,
            "headline_message": (
                (
                    f"No reliable cross-camera matches in this window — "
                    f"store-wide unique = sum of per-camera unique. "
                    f"(Threshold: cosine ≥ {cross_camera_match:.2f}; "
                    "lower would risk false merges.)"
                )
                if no_reliable
                else (
                    f"Identified {len(links)} cross-camera face match(es) above "
                    f"cosine ≥ {cross_camera_match:.2f}. Recording windows do not "
                    "overlap in time — these are repeat-presence events across "
                    "the captured period, not single continuous trips."
                )
            ),
        },
        "thresholds": {
            "in_camera_cluster": in_camera_cluster,
            "cross_camera_match": cross_camera_match,
            "min_face_appearances_for_cross_camera": min_face_appearances,
        },
        "per_camera_unique": per_camera_counts,
        "persons_considered": len(persons),
        "persons_skipped": skipped,
        "cross_camera_links": [_link_to_dict(link) for link in links],
        "store_wide_persons": [_swp_to_dict(s) for s in store_wide],
    }

    json_path = out_root / "cross_camera.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return CrossCameraResult(
        threshold=cross_camera_match,
        in_camera_cluster_threshold=in_camera_cluster,
        min_face_appearances=min_face_appearances,
        persons_considered=len(persons),
        persons_skipped=skipped,
        per_camera_unique=per_camera_counts,
        naive_total_per_camera_sum=naive_total,
        store_wide_unique_visitors=store_wide_count,
        saved_by_cross_camera_dedup=saved,
        cross_camera_links=links,
        store_wide_persons=store_wide,
        no_reliable_cross_camera_matches=no_reliable,
        json_path=json_path,
    )


def _per_camera_counts_from_jsons(identity_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for json_path in sorted(identity_dir.glob("*.json")):
        if json_path.name.startswith("phase"):
            continue
        data = json.loads(json_path.read_text())
        cam = data.get("camera_id") or json_path.stem
        counts[cam] = int(data.get("unique_visitors_count", 0))
    return counts


def _link_to_dict(link: CrossCameraLink) -> dict[str, Any]:
    return {
        "from": {
            "camera_id": link.from_camera,
            "person_id": link.from_person,
            "area": link.from_area,
            "first_seen": link.from_first_seen,
            "last_seen": link.from_last_seen,
        },
        "to": {
            "camera_id": link.to_camera,
            "person_id": link.to_person,
            "area": link.to_area,
            "first_seen": link.to_first_seen,
            "last_seen": link.to_last_seen,
        },
        "similarity": round(link.similarity, 4),
        "time_gap": link.time_gap,
        "presence_note": link.presence_note,
    }


def _swp_to_dict(s: StoreWidePerson) -> dict[str, Any]:
    return {
        "store_person_id": s.store_person_id,
        "members": s.members,
        "areas_visited": s.areas_visited,
        "first_seen_overall": s.first_seen_overall,
        "last_seen_overall": s.last_seen_overall,
    }


def summarize_results(r: CrossCameraResult) -> str:
    lines = [
        "",
        "Phase 4 cross-camera summary",
        "============================",
        "",
        f"  thresholds        : in_camera_cluster={r.in_camera_cluster_threshold:.2f}  "
        f"cross_camera_match={r.threshold:.2f}  "
        f"min_appearances={r.min_face_appearances}",
        f"  persons considered: {r.persons_considered}  "
        f"(skipped {len(r.persons_skipped)} below appearance gate)",
        f"  per-camera unique : {r.per_camera_unique}",
        f"  naive total       : {r.naive_total_per_camera_sum}",
        f"  cross-cam links   : {len(r.cross_camera_links)} above threshold",
        f"  saved by dedup    : {r.saved_by_cross_camera_dedup}",
        f"  STORE-WIDE UNIQUE : {r.store_wide_unique_visitors}  ← headline",
        "",
    ]
    if r.no_reliable_cross_camera_matches:
        lines.append(
            "  ⚠ no reliable cross-camera matches — falling back to per-camera sum."
        )
        lines.append(
            "    (lower the cross_camera_match threshold to recover more links, "
            "at the cost of false merges.)"
        )
    else:
        lines.append("  Cross-camera links (highest similarity first):")
        for link in r.cross_camera_links:
            lines.append(
                f"    {link.from_camera}/{link.from_person} ↔ "
                f"{link.to_camera}/{link.to_person}   "
                f"sim={link.similarity:.2f}   gap≈{link.time_gap}"
            )
        lines.append("")
        lines.append("  Store-wide persons (S001 = earliest first-seen):")
        for s in r.store_wide_persons:
            members = ", ".join(
                f"{m['camera_id']}/{m['person_id']}" for m in s.members
            )
            lines.append(
                f"    {s.store_person_id}  [{members}]  areas={s.areas_visited}"
            )
    lines.append("")
    try:
        rel = r.json_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = r.json_path
    lines.append(f"  written to: {rel}")
    return "\n".join(lines) + "\n"
