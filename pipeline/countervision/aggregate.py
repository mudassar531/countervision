"""Phase 5 — aggregate everything into the dashboard contract.

Reads the per-phase outputs and emits the single ``analytics.json``
file the dashboard consumes (plus a small ``analytics.db`` sqlite
mirror so downstream tools can SQL it). Generates plain-English retail
**insights** tied only to the *reliable* numbers — never to a
cross-camera link or a near-zero footfall count, both of which are
flagged through with hedging copy so the dashboard can render them
with appropriate caveats.

Locked fields (uncomputable from this footage — never fabricated):

* ``conversion_rate`` — no POS data
* ``revenue_uplift`` — no POS data
* ``weather`` — no external feed
* ``staffing_recommendations_quantified`` — needs payroll integration

Each is emitted as ``{"value": null, "locked": true, "reason": "..."}``
so the dashboard can render a "data not available" badge instead of a
made-up number.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _maybe_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _peak_hour(footfall_by_hour: list[dict[str, Any]]) -> str | None:
    """Return the ``HH:00`` bucket with the highest combined in+out."""
    if not footfall_by_hour:
        return None
    return max(footfall_by_hour, key=lambda b: b.get("total", 0)).get("hour")


def _locked(reason: str) -> dict[str, Any]:
    return {"value": None, "locked": True, "reason": reason}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class AggregateResult:
    analytics_path: Path
    sqlite_path: Path
    schema_version: int
    cameras_count: int
    areas_count: int
    persons_count: int
    alerts_count: int
    insights_count: int
    store_wide_unique_visitors: int
    store_wide_locked: bool


def aggregate(
    *,
    output_root: Path,
    store_name: str,
) -> AggregateResult:
    zones_dir = output_root / "zones"
    identity_dir = output_root / "identity"
    cross_camera_path = output_root / "cross_camera.json"

    if not zones_dir.is_dir():
        raise RuntimeError(f"missing {zones_dir} — run --run-zones first")
    if not identity_dir.is_dir():
        raise RuntimeError(f"missing {identity_dir} — run --run-identity first")

    zone_jsons = {
        p.stem: _read_json(p)
        for p in sorted(zones_dir.glob("*.json"))
        if not p.name.startswith("phase")
    }
    identity_jsons = {
        p.stem: _read_json(p)
        for p in sorted(identity_dir.glob("*.json"))
        if not p.name.startswith("phase")
    }
    cross_camera = _read_json(cross_camera_path) if cross_camera_path.exists() else None

    cameras = sorted(set(zone_jsons) | set(identity_jsons))
    if not cameras:
        raise RuntimeError("no per-camera artefacts found — run earlier phases first")

    # ----------- per-area aggregation -------------------------------------- #
    areas: list[dict[str, Any]] = []
    footfall_by_hour_global: dict[str, dict[str, int]] = {}
    occupancy_by_camera: dict[str, list[dict[str, Any]]] = {}
    persons_by_camera: dict[str, list[dict[str, Any]]] = {}
    naive_per_camera_unique = 0
    total_footfall_in = 0
    total_footfall_out = 0
    window_start: datetime | None = None
    window_end: datetime | None = None

    for cam_id in cameras:
        z = zone_jsons.get(cam_id, {})
        i = identity_jsons.get(cam_id, {})
        area_name = i.get("area") or z.get("area") or cam_id
        zone_block = (z.get("zones") or [{}])[0]
        unique_visitors = int(i.get("unique_visitors_count", 0))
        naive_per_camera_unique += unique_visitors
        footfall = z.get("footfall") or {}
        in_count = int(footfall.get("in_count", 0))
        out_count = int(footfall.get("out_count", 0))
        total_footfall_in += in_count
        total_footfall_out += out_count

        # Authoritative per-area dwell from Phase 3 face-linked persons
        person_records = i.get("persons", [])
        persons_by_camera[cam_id] = person_records
        dwell_seconds_list = [
            float(p.get("track_dwell_seconds_authoritative", 0.0))
            for p in person_records
            if float(p.get("track_dwell_seconds_authoritative", 0.0)) > 0
        ]
        avg_dwell = round(sum(dwell_seconds_list) / len(dwell_seconds_list), 1) if dwell_seconds_list else 0.0
        max_dwell = round(max(dwell_seconds_list), 1) if dwell_seconds_list else 0.0

        # Hourly footfall accumulator (global)
        for bucket in z.get("footfall_by_hour", []) or []:
            slot = footfall_by_hour_global.setdefault(
                bucket["hour"], {"in": 0, "out": 0, "total": 0}
            )
            slot["in"] += int(bucket.get("in", 0))
            slot["out"] += int(bucket.get("out", 0))
            slot["total"] += int(bucket.get("total", 0))

        occupancy_by_camera[cam_id] = z.get("occupancy_timeseries", []) or []

        # Window envelope (from identity which carries first/last seen)
        for ts_key in ("first_seen", "last_seen"):
            for p in person_records:
                t = _maybe_iso(p.get(ts_key))
                if t is None:
                    continue
                if window_start is None or t < window_start:
                    window_start = t
                if window_end is None or t > window_end:
                    window_end = t

        areas.append(
            {
                "camera_id": cam_id,
                "area": area_name,
                "unique_visitors": unique_visitors,
                "footfall_in": in_count,
                "footfall_out": out_count,
                "footfall_total": in_count + out_count,
                "avg_dwell_seconds": avg_dwell,
                "max_dwell_seconds": max_dwell,
                "occupancy_peak": int(zone_block.get("occupancy_peak", 0)),
                "frame_jpg": z.get("frame_jpg") or i.get("frame_jpg"),
                "heatmap_png": z.get("heatmap_png"),
                "zone_polygon": zone_block.get("polygon"),
                "entry_line": footfall.get("entry_line"),
                "occupancy_timeseries": occupancy_by_camera[cam_id],
                "person_tracks_note": (
                    "Authoritative per-area unique-visitor count comes from "
                    "face-based identity (Phase 3). Phase 2's tracker-id count "
                    "is no longer the source of truth."
                ),
            }
        )

    # ----------- visitors (flat list per camera-person) -------------------- #
    visitors: list[dict[str, Any]] = []
    for cam_id, person_records in persons_by_camera.items():
        area_name = next((a["area"] for a in areas if a["camera_id"] == cam_id), cam_id)
        for p in person_records:
            visitors.append(
                {
                    "camera_id": cam_id,
                    "area": area_name,
                    "person_id": p["person_id"],
                    "first_seen": p.get("first_seen"),
                    "last_seen": p.get("last_seen"),
                    "face_appearances": p.get("face_appearances"),
                    "track_dwell_seconds": p.get("track_dwell_seconds_authoritative"),
                    "face_dwell_seconds": p.get("face_dwell_seconds_authoritative"),
                    "linked_tracker_ids": p.get("linked_tracker_ids", []),
                    "visit_count": p.get("visit_count", 1),
                    "is_repeat": bool(p.get("is_repeat", False)),
                    "watchlist_match": p.get("watchlist_match"),
                    "watchlist_similarity": p.get("watchlist_similarity"),
                    "thumbnail": p.get("thumbnail"),
                }
            )

    # ----------- alerts (collated from identity JSONs) -------------------- #
    alerts: list[dict[str, Any]] = []
    for cam_id, ij in identity_jsons.items():
        for a in ij.get("alerts", []) or []:
            alert = dict(a)
            alert.setdefault("area", next(
                (x["area"] for x in areas if x["camera_id"] == cam_id), cam_id
            ))
            if alert.get("type") == "watchlist":
                alert["confidence_note"] = (
                    "Possible match — face similarity is below 0.6; treat as a "
                    "verification prompt, not an identification."
                )
                alert["confidence_level"] = "low" if (alert.get("similarity") or 0) < 0.6 else "medium"
            alerts.append(alert)

    # ----------- KPIs (reliable headline + hedged store-wide) ------------- #
    store_wide_reliable = (
        cross_camera is not None
        and not cross_camera["headline"]["no_reliable_cross_camera_matches"]
    )
    if cross_camera is not None:
        store_wide_unique = int(cross_camera["headline"]["store_wide_unique_visitors"])
        store_wide_block = {
            "value": store_wide_unique,
            "locked": False,
            "confidence": "medium" if store_wide_reliable else "low",
            "method": (
                "Face-based de-dup across cameras at cosine >= "
                f"{cross_camera['thresholds']['cross_camera_match']}. "
                "Recording windows do not overlap by hours, so any cross-camera "
                "match is repeat presence across the captured period, not a "
                "single continuous trip."
                if store_wide_reliable
                else "No cross-camera face matches cleared the high-precision "
                "threshold in this window — store-wide unique falls back to the "
                "sum of per-camera unique visitors. Result may double-count."
            ),
            "naive_per_camera_sum": naive_per_camera_unique,
            "saved_by_dedup": int(cross_camera["headline"]["saved_by_cross_camera_dedup"]),
            "cross_camera_links_count": int(cross_camera["headline"]["cross_camera_links_count"]),
            "no_reliable_cross_camera_matches": bool(
                cross_camera["headline"]["no_reliable_cross_camera_matches"]
            ),
        }
    else:
        store_wide_unique = naive_per_camera_unique
        store_wide_block = {
            "value": store_wide_unique,
            "locked": False,
            "confidence": "low",
            "method": "Phase 4 not run — store-wide unique is the naive sum of "
                      "per-camera unique visitors and may double-count.",
            "naive_per_camera_sum": naive_per_camera_unique,
            "saved_by_dedup": 0,
            "cross_camera_links_count": 0,
            "no_reliable_cross_camera_matches": True,
        }

    # Average dwell across the store, weighted by visitor count per area
    per_area_dwell = [(a["avg_dwell_seconds"], a["unique_visitors"]) for a in areas]
    weighted_total = sum(d * n for d, n in per_area_dwell)
    visitor_total = sum(n for _, n in per_area_dwell)
    avg_dwell_store = round(weighted_total / visitor_total, 1) if visitor_total > 0 else 0.0

    peak_hour_global = _peak_hour(
        [{"hour": h, **v} for h, v in sorted(footfall_by_hour_global.items())]
    )
    footfall_total = total_footfall_in + total_footfall_out

    kpis = {
        "store_wide_unique_visitors": store_wide_block,
        "per_camera_unique_visitors_sum": naive_per_camera_unique,
        "footfall_total": {
            "value": footfall_total,
            "locked": False,
            "confidence": "low" if footfall_total < 5 else "medium",
            "note": (
                "Line-crossing count from auto-generated entry lines (placed at "
                "75% frame height). For demo quality, an operator should redraw "
                "the line at each scene's true entrance via "
                "`python main.py --draw-zones CAM`. Headline counts here may "
                "be lower than the actual store footfall."
            ),
            "in_count": total_footfall_in,
            "out_count": total_footfall_out,
        },
        "watchlist_hits": {
            "value": sum(1 for a in alerts if a.get("type") == "watchlist"),
            "locked": False,
            "confidence": "low",
            "note": "Each watchlist alert is a verification prompt, not an "
                    "identification. Similarity scores are attached to each event.",
        },
        "repeat_visitors_per_area": {
            "value": sum(1 for v in visitors if v["is_repeat"]),
            "locked": False,
            "confidence": "medium",
            "note": "A 'repeat' here means the same face was seen across >=2 "
                    "separate visit segments within a single camera's window.",
        },
        "avg_dwell_seconds_store": {
            "value": avg_dwell_store,
            "locked": False,
            "confidence": "high",
            "note": "Weighted average across areas. Per-person dwell comes from "
                    "the union of frames where any linked tracker_id is alive — "
                    "merges Phase-1 ID fragmentation.",
        },
        "peak_hour": peak_hour_global,
        "active_alerts": len(alerts),
        # ---------- LOCKED fields ---------- #
        "conversion_rate": _locked("No POS data — cannot compute conversion."),
        "revenue_uplift": _locked("No POS data — cannot compute revenue impact."),
        "weather": _locked("No external weather feed integrated."),
        "staffing_recommendations_quantified": _locked(
            "Quantified staffing recommendations require payroll/scheduling "
            "integration; insights are qualitative only."
        ),
    }

    # ----------- insights (only from reliable numbers) -------------------- #
    insights = _generate_insights(areas=areas, visitors=visitors, store_wide_block=store_wide_block)

    # ----------- assemble + persist --------------------------------------- #
    payload: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="milliseconds"),
        "store": {
            "name": store_name,
            "cameras": cameras,
            "window": {
                "start": window_start.isoformat(timespec="milliseconds") if window_start else None,
                "end": window_end.isoformat(timespec="milliseconds") if window_end else None,
            },
        },
        "kpis": kpis,
        "footfall_by_hour": [
            {"hour": h, **v} for h, v in sorted(footfall_by_hour_global.items())
        ],
        "areas": areas,
        "visitors": visitors,
        "alerts": alerts,
        "cross_camera": (
            {
                "thresholds": cross_camera["thresholds"],
                "headline": cross_camera["headline"],
                "links": cross_camera["cross_camera_links"],
                "store_wide_persons": cross_camera["store_wide_persons"],
                "persons_skipped": cross_camera["persons_skipped"],
                "render_hint": (
                    "Dashboard MUST render cross-camera links with hedging copy "
                    "from `presence_note`. Recording windows do not overlap; do not "
                    "render these as continuous 'journeys'."
                ),
            }
            if cross_camera
            else None
        ),
        "insights": insights,
        "locked_fields_note": (
            "Fields marked `locked: true` are uncomputable from the current "
            "footage. They are never fabricated; the dashboard should render a "
            "'data not available' badge with the supplied reason."
        ),
    }

    analytics_path = output_root / "analytics.json"
    analytics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    sqlite_path = output_root / "analytics.db"
    _write_sqlite_mirror(sqlite_path, payload)

    return AggregateResult(
        analytics_path=analytics_path,
        sqlite_path=sqlite_path,
        schema_version=SCHEMA_VERSION,
        cameras_count=len(cameras),
        areas_count=len(areas),
        persons_count=len(visitors),
        alerts_count=len(alerts),
        insights_count=len(insights),
        store_wide_unique_visitors=store_wide_unique,
        store_wide_locked=False,
    )


# --------------------------------------------------------------------------- #
# Insights — only generated from reliable numbers (no cross-camera, no
# near-zero footfall).
# --------------------------------------------------------------------------- #


def _generate_insights(
    *,
    areas: list[dict[str, Any]],
    visitors: list[dict[str, Any]],
    store_wide_block: dict[str, Any],
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    if not areas:
        return insights

    # 1) Highest-dwell area — always reliable when we have any dwell.
    by_dwell = sorted(
        (a for a in areas if a["avg_dwell_seconds"] > 0),
        key=lambda a: a["avg_dwell_seconds"],
        reverse=True,
    )
    if by_dwell:
        top = by_dwell[0]
        insights.append(
            {
                "id": "highest_dwell_area",
                "title": f"Longest average dwell in {top['area']}",
                "detail": (
                    f"Visitors spend an average of {top['avg_dwell_seconds']:.0f}s in "
                    f"{top['area']} — the longest of any area. Consider staffing a "
                    "consult / assisted-sale role here, or testing higher-margin "
                    "displays in the same eye-line."
                ),
                "evidence": {
                    "area": top["area"],
                    "avg_dwell_seconds": top["avg_dwell_seconds"],
                    "unique_visitors": top["unique_visitors"],
                },
                "confidence": "high",
            }
        )

    # 2) Biggest crowd zone (peak occupancy).
    by_peak = sorted(areas, key=lambda a: a["occupancy_peak"], reverse=True)
    if by_peak and by_peak[0]["occupancy_peak"] >= 2:
        top = by_peak[0]
        insights.append(
            {
                "id": "peak_occupancy_zone",
                "title": f"Peak crowding in {top['area']}",
                "detail": (
                    f"{top['area']} peaked at {top['occupancy_peak']} people present "
                    "at the same time during the captured window. Watch for queue "
                    "build-up here at busier times; consider an extra staff "
                    "presence or self-serve options."
                ),
                "evidence": {
                    "area": top["area"],
                    "occupancy_peak": top["occupancy_peak"],
                },
                "confidence": "high",
            }
        )

    # 3) Imbalance between areas — unique-visitor split signals which area
    # is the engagement leader.
    if len(areas) >= 2:
        ranked = sorted(areas, key=lambda a: a["unique_visitors"], reverse=True)
        top, bottom = ranked[0], ranked[-1]
        if top["unique_visitors"] >= 3 and top["unique_visitors"] > 2 * max(
            bottom["unique_visitors"], 1
        ):
            insights.append(
                {
                    "id": "area_engagement_imbalance",
                    "title": f"Most engagement is happening in {top['area']}",
                    "detail": (
                        f"{top['area']} drew {top['unique_visitors']} unique "
                        f"faces vs only {bottom['unique_visitors']} in "
                        f"{bottom['area']}. Worth investigating whether the "
                        "store layout pushes traffic toward the busy area, "
                        "or whether the quieter area needs a clearer visual cue."
                    ),
                    "evidence": {
                        "leader_area": top["area"],
                        "leader_unique": top["unique_visitors"],
                        "trailing_area": bottom["area"],
                        "trailing_unique": bottom["unique_visitors"],
                    },
                    "confidence": "medium",
                }
            )

    # 4) Repeat visitors — only meaningful at >=2 repeats so we don't
    # over-interpret a single noisy ID.
    repeat_count = sum(1 for v in visitors if v["is_repeat"])
    if repeat_count >= 2:
        insights.append(
            {
                "id": "repeat_visitor_opportunity",
                "title": f"{repeat_count} repeat visitors identified within this window",
                "detail": (
                    f"{repeat_count} face(s) appeared across >=2 separate visit "
                    "segments inside a single camera. This is a small sample, "
                    "but at scale it would be a loyalty-program signal — "
                    "consider tagging frequent returners for a personalised "
                    "consultation."
                ),
                "evidence": {"repeat_visitors_per_area": repeat_count},
                "confidence": "medium",
            }
        )

    # 5) Headline framing for the demo — uses dwell + visitor depth, not
    # cross-camera link or footfall.
    total_visitors = sum(a["unique_visitors"] for a in areas)
    if total_visitors >= 5:
        deepest = max(areas, key=lambda a: a["max_dwell_seconds"])
        insights.append(
            {
                "id": "demo_headline_framing",
                "title": "Areas with the deepest engagement are the staffing priority",
                "detail": (
                    f"Across the captured window we identified {total_visitors} "
                    f"unique faces. The deepest single-person dwell was "
                    f"{deepest['max_dwell_seconds']:.0f}s in {deepest['area']}. "
                    "Staffing decisions should follow dwell-by-area, not raw "
                    "footfall — high dwell is where conversion happens."
                ),
                "evidence": {
                    "total_unique_visitors": total_visitors,
                    "deepest_area": deepest["area"],
                    "deepest_dwell_seconds": deepest["max_dwell_seconds"],
                },
                "confidence": "high",
            }
        )

    return insights


# --------------------------------------------------------------------------- #
# sqlite mirror (faithful, not a query layer)
# --------------------------------------------------------------------------- #


def _write_sqlite_mirror(db_path: Path, payload: dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE areas (
                camera_id TEXT PRIMARY KEY,
                area TEXT,
                unique_visitors INTEGER,
                footfall_in INTEGER,
                footfall_out INTEGER,
                footfall_total INTEGER,
                avg_dwell_seconds REAL,
                max_dwell_seconds REAL,
                occupancy_peak INTEGER,
                frame_jpg TEXT,
                heatmap_png TEXT
            );

            CREATE TABLE visitors (
                camera_id TEXT,
                person_id TEXT,
                area TEXT,
                first_seen TEXT,
                last_seen TEXT,
                face_appearances INTEGER,
                track_dwell_seconds REAL,
                face_dwell_seconds REAL,
                visit_count INTEGER,
                is_repeat INTEGER,
                watchlist_match TEXT,
                watchlist_similarity REAL,
                thumbnail TEXT,
                PRIMARY KEY (camera_id, person_id)
            );

            CREATE TABLE alerts (
                id TEXT PRIMARY KEY,
                type TEXT,
                camera_id TEXT,
                area TEXT,
                person_id TEXT,
                timestamp TEXT,
                severity TEXT,
                confidence_level TEXT,
                similarity REAL,
                copy TEXT,
                thumbnail TEXT,
                frame_jpg TEXT
            );

            CREATE TABLE footfall_by_hour (
                hour TEXT PRIMARY KEY,
                in_count INTEGER,
                out_count INTEGER,
                total INTEGER
            );

            CREATE TABLE occupancy_timeseries (
                camera_id TEXT,
                second_bucket INTEGER,
                t TEXT,
                active_tracks INTEGER,
                PRIMARY KEY (camera_id, second_bucket)
            );

            CREATE TABLE cross_camera_links (
                from_camera TEXT,
                from_person TEXT,
                to_camera TEXT,
                to_person TEXT,
                similarity REAL,
                time_gap TEXT,
                presence_note TEXT
            );

            CREATE TABLE insights (
                id TEXT PRIMARY KEY,
                title TEXT,
                detail TEXT,
                confidence TEXT
            );

            CREATE TABLE kpis (
                key TEXT PRIMARY KEY,
                value_json TEXT
            );
            """
        )

        conn.executemany(
            "INSERT INTO areas VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    a["camera_id"], a["area"], a["unique_visitors"],
                    a["footfall_in"], a["footfall_out"], a["footfall_total"],
                    a["avg_dwell_seconds"], a["max_dwell_seconds"], a["occupancy_peak"],
                    a.get("frame_jpg"), a.get("heatmap_png"),
                )
                for a in payload["areas"]
            ],
        )
        conn.executemany(
            "INSERT INTO visitors VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    v["camera_id"], v["person_id"], v["area"],
                    v.get("first_seen"), v.get("last_seen"),
                    v.get("face_appearances"),
                    v.get("track_dwell_seconds"),
                    v.get("face_dwell_seconds"),
                    v.get("visit_count"),
                    1 if v.get("is_repeat") else 0,
                    v.get("watchlist_match"),
                    v.get("watchlist_similarity"),
                    v.get("thumbnail"),
                )
                for v in payload["visitors"]
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    a.get("id"), a.get("type"), a.get("camera_id"),
                    a.get("area"), a.get("person_id"), a.get("timestamp"),
                    a.get("severity"), a.get("confidence_level"),
                    a.get("similarity"), a.get("copy"),
                    a.get("thumbnail"), a.get("frame_jpg"),
                )
                for a in payload["alerts"]
            ],
        )
        conn.executemany(
            "INSERT INTO footfall_by_hour VALUES (?,?,?,?)",
            [
                (b["hour"], b.get("in", 0), b.get("out", 0), b.get("total", 0))
                for b in payload["footfall_by_hour"]
            ],
        )
        conn.executemany(
            "INSERT INTO occupancy_timeseries VALUES (?,?,?,?)",
            [
                (a["camera_id"], pt["second_bucket"], pt.get("t"), pt.get("active_tracks", 0))
                for a in payload["areas"]
                for pt in a.get("occupancy_timeseries", [])
            ],
        )
        if payload.get("cross_camera"):
            conn.executemany(
                "INSERT INTO cross_camera_links VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        link["from"]["camera_id"], link["from"]["person_id"],
                        link["to"]["camera_id"], link["to"]["person_id"],
                        link["similarity"], link["time_gap"], link["presence_note"],
                    )
                    for link in payload["cross_camera"]["links"]
                ],
            )
        conn.executemany(
            "INSERT INTO insights VALUES (?,?,?,?)",
            [
                (i["id"], i["title"], i["detail"], i.get("confidence", "medium"))
                for i in payload["insights"]
            ],
        )
        conn.executemany(
            "INSERT INTO kpis VALUES (?,?)",
            [(k, json.dumps(v, default=str)) for k, v in payload["kpis"].items()],
        )
        conn.commit()
    finally:
        conn.close()


def summarize_result(result: AggregateResult, project_root: Path) -> str:
    try:
        rel_json = result.analytics_path.relative_to(project_root)
        rel_db = result.sqlite_path.relative_to(project_root)
    except ValueError:
        rel_json = result.analytics_path
        rel_db = result.sqlite_path
    return (
        "\nPhase 5 aggregate summary"
        "\n========================="
        f"\n  schema version           : {result.schema_version}"
        f"\n  cameras / areas          : {result.cameras_count} / {result.areas_count}"
        f"\n  visitors (camera-person) : {result.persons_count}"
        f"\n  alerts                   : {result.alerts_count}"
        f"\n  insights generated       : {result.insights_count}"
        f"\n  store-wide unique        : {result.store_wide_unique_visitors}"
        f"\n  written analytics.json   : {rel_json}"
        f"\n  written sqlite mirror    : {rel_db}"
        "\n"
    )
