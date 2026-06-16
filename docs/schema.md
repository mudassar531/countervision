# `analytics.json` — schema reference

> Version `1`. Produced by `pipeline/countervision/aggregate.py`.
> The dashboard reads **only** this file (plus the referenced static
> images / heatmaps). Nothing in the live demo path runs a model.

## Honesty conventions

The pipeline produces three kinds of numbers, and the schema labels them
explicitly so the dashboard can render with appropriate hedging:

* **Reliable headlines** — `confidence: "high" | "medium"`. Tied to
  Phase 3 face-based dwell, occupancy and area-level unique counts.
  Safe to show as the big number.
* **Hedged / low-confidence** — `confidence: "low"`. Cross-camera
  links, footfall (when entry lines weren't redrawn for the scene),
  and watchlist hits. The schema includes a `note` / `method`
  explaining the caveat; render with a footnote, not as a hard fact.
* **Locked** — `{"value": null, "locked": true, "reason": "..."}`.
  Anything the footage cannot compute (POS conversion, weather,
  quantified staffing). The dashboard should render a "data not
  available" badge, **never** a made-up number.

The `locked_fields_note` at the root of the JSON spells this out for
implementors.

## Top-level shape

```jsonc
{
  "version": 1,
  "generated_at": "2026-06-16T19:30:00.000",
  "store": { "name": "...", "cameras": ["camera-1", ...],
             "window": {"start": "ISO", "end": "ISO"} },
  "kpis": { ... see below ... },
  "footfall_by_hour": [{"hour": "00:00", "in": 0, "out": 1, "total": 1}, ...],
  "areas": [{...}, ...],
  "visitors": [{...}, ...],
  "alerts": [{...}, ...],
  "cross_camera": { ... or null if Phase 4 not run },
  "insights": [{...}, ...],
  "locked_fields_note": "..."
}
```

## `kpis`

```jsonc
{
  "store_wide_unique_visitors": {
    "value": 13,
    "locked": false,
    "confidence": "medium",          // or "low" if no reliable links
    "method": "Face-based de-dup ... repeat presence ...",
    "naive_per_camera_sum": 16,
    "saved_by_dedup": 3,
    "cross_camera_links_count": 3,
    "no_reliable_cross_camera_matches": false
  },
  "per_camera_unique_visitors_sum": 16,   // honest fallback if dashboard needs it
  "footfall_total": {
    "value": 3,
    "confidence": "low",   // jumps to "medium" once value >= 5
    "in_count": 2,
    "out_count": 1,
    "note": "Auto-generated entry lines at 75% frame height. Operator should redraw..."
  },
  "watchlist_hits": {
    "value": 4,
    "confidence": "low",
    "note": "Each watchlist alert is a verification prompt, not an identification."
  },
  "repeat_visitors_per_area": {
    "value": 5,
    "confidence": "medium",
    "note": "Same face across >=2 separate visit segments within a single camera."
  },
  "avg_dwell_seconds_store": {
    "value": 73.9,
    "confidence": "high",
    "note": "Per-person dwell from union of frames where any linked tracker_id is alive."
  },
  "peak_hour": "04:00",
  "active_alerts": 9,

  // --- LOCKED (uncomputable from this footage; never fabricated) --- //
  "conversion_rate":     { "value": null, "locked": true, "reason": "No POS data ..." },
  "revenue_uplift":      { "value": null, "locked": true, "reason": "No POS data ..." },
  "weather":             { "value": null, "locked": true, "reason": "No external weather feed ..." },
  "staffing_recommendations_quantified": { "value": null, "locked": true, "reason": "..." }
}
```

## `areas[]`

One element per camera. Authoritative dwell + occupancy live here.

```jsonc
{
  "camera_id": "camera-3",
  "area": "Customer Seating / Try-on Lounge",
  "unique_visitors": 8,             // from Phase 3 face identity (authoritative)
  "footfall_in": 0,
  "footfall_out": 1,
  "footfall_total": 1,
  "avg_dwell_seconds": 96.0,        // mean of track_dwell_seconds_authoritative
  "max_dwell_seconds": 180.0,
  "occupancy_peak": 3,
  "frame_jpg": "data/output/frames/camera-3.jpg",
  "heatmap_png": "data/output/heatmaps/camera-3.png",
  "zone_polygon": [[384, 216], ...],
  "entry_line": {"start": [288, 810], "end": [1632, 810], "anchor": "bottom_center"},
  "occupancy_timeseries": [{"t": "ISO", "frame_idx": 0, "second_bucket": 0, "active_tracks": 2}, ...],
  "person_tracks_note": "Authoritative per-area unique-visitor count comes from face-based identity (Phase 3)."
}
```

## `visitors[]`

Flat list — one element per `(camera_id, person_id)` pair from Phase 3.
Cross-camera de-dup happens in `cross_camera.store_wide_persons[]`, NOT
here. This list is the raw building block; the dashboard composes
store-wide views from it.

```jsonc
{
  "camera_id": "camera-5",
  "area": "Service & Consultation Desk",
  "person_id": "P006",
  "first_seen": "2026-06-08T04:47:13.000",
  "last_seen":  "2026-06-08T04:50:00.000",
  "face_appearances": 58,
  "track_dwell_seconds": 31.6,      // AUTHORITATIVE — union of linked_tracker_ids
  "face_dwell_seconds": 2.3,
  "linked_tracker_ids": [37, 50, 53, 60, 66, 68, 79],   // merged Phase-1 fragments
  "visit_count": 1,
  "is_repeat": false,
  "watchlist_match": "staff_lead_demo",
  "watchlist_similarity": 0.5206,
  "thumbnail": "data/output/persons/camera-5/P006.jpg"
}
```

## `alerts[]`

Non-accusatory review prompts. Watchlist alerts also carry a
`confidence_level` and `confidence_note` because each one is a
verification prompt, not an identification.

```jsonc
{
  "id": "A-...",
  "type": "watchlist" | "repeat_visitor",
  "camera_id": "camera-5",
  "area": "Service & Consultation Desk",
  "person_id": "P006",
  "timestamp": "ISO",
  "severity": "info" | "warn" | "high",
  "similarity": 0.52,                // watchlist only
  "confidence_level": "low" | "medium",
  "confidence_note": "Possible match — face similarity is below 0.6 ...",
  "thumbnail": "data/output/persons/camera-5/P006.jpg",
  "frame_jpg": "data/output/alerts/A-xxx.jpg",
  "copy": "Possible match with watchlist entry 'X' ... Please verify before acting."
}
```

## `cross_camera` (or `null`)

Hedged, never a hard fact. The `render_hint` is intentional — implementors
should respect it.

```jsonc
{
  "thresholds": {
    "in_camera_cluster": 0.32,
    "cross_camera_match": 0.50,    // distinct, higher bar
    "min_face_appearances_for_cross_camera": 3
  },
  "headline": {
    "store_wide_unique_visitors": 13,
    "naive_total_per_camera_sum": 16,
    "saved_by_cross_camera_dedup": 3,
    "cross_camera_links_count": 3,
    "no_reliable_cross_camera_matches": false,
    "headline_message": "Identified 3 cross-camera face match(es) ..."
  },
  "links": [
    {
      "from": {"camera_id": "camera-3", "person_id": "P003", "area": "...", "first_seen": "ISO", "last_seen": "ISO"},
      "to":   {"camera_id": "camera-5", "person_id": "P004", "area": "...", "first_seen": "ISO", "last_seen": "ISO"},
      "similarity": 0.60,
      "time_gap": "4 h 13 m",
      "presence_note": "Same face appears in '...' and in '...'. Recording windows do not overlap (gap ~ 4h 13m), so this represents the same person being seen in these areas across the captured period — repeat presence, not a single continuous trip. Cosine similarity 0.60."
    }
  ],
  "store_wide_persons": [
    {
      "store_person_id": "S003",
      "members": [{"camera_id": "...", "person_id": "...", "area": "...", "first_seen": "ISO", "last_seen": "ISO", "face_appearances": 30}, ...],
      "areas_visited": ["Customer Seating / Try-on Lounge", "Service & Consultation Desk"],
      "first_seen_overall": "ISO",
      "last_seen_overall": "ISO"
    }
  ],
  "persons_skipped": [{"camera_id": "camera-1", "person_id": "P001", "reason": "face_appearances=1 < 3 (centroid too noisy)"}],
  "render_hint": "Dashboard MUST render cross-camera links with hedging copy from `presence_note`. Recording windows do not overlap; do not render these as continuous 'journeys'."
}
```

## `insights[]`

3–5 plain-English retail recommendations. **Always tied to a reliable
number** — never to a cross-camera link or near-zero footfall. Each
insight carries its own `confidence` so the dashboard can sort or
filter.

```jsonc
{
  "id": "highest_dwell_area",
  "title": "Longest average dwell in Customer Seating / Try-on Lounge",
  "detail": "Visitors spend an average of 96s in ... Consider staffing a consult ...",
  "evidence": {"area": "Customer Seating / Try-on Lounge", "avg_dwell_seconds": 96.0, "unique_visitors": 8},
  "confidence": "high"
}
```

The set of `id`s currently emitted by the generator:

| id                          | trigger                                                       |
|-----------------------------|---------------------------------------------------------------|
| `highest_dwell_area`        | any area has avg dwell > 0                                    |
| `peak_occupancy_zone`       | the leading area's peak occupancy ≥ 2                         |
| `area_engagement_imbalance` | leader has ≥ 3 visitors **and** ≥ 2× trailer                  |
| `repeat_visitor_opportunity`| ≥ 2 repeat visitors across the window                         |
| `demo_headline_framing`     | ≥ 5 total unique visitors                                     |

Adding insights is a one-function change in
`aggregate._generate_insights`. The constraint is the same as elsewhere:
**only fire when the underlying number is reliable** (e.g. never gate an
insight on the cross-camera count alone, and never on a footfall total
below 5 unless the entry line has been redrawn for the scene).

## sqlite mirror — `analytics.db`

A faithful, read-only mirror of the JSON above. Tables:

| table                  | rows                          | use                                       |
|------------------------|-------------------------------|-------------------------------------------|
| `areas`                | 1 / camera                    | KPI cards, heatmap backdrop                |
| `visitors`             | 1 / (camera, person)          | per-area visitor list                      |
| `alerts`               | all alerts                    | Alerts feed panel                          |
| `footfall_by_hour`     | 1 / hour bucket               | Recharts area chart                        |
| `occupancy_timeseries` | 1 / (camera, second)          | per-area occupancy line                    |
| `cross_camera_links`   | each above-threshold pair     | journey panel (with hedging copy)          |
| `insights`             | 3–5 rows                      | Insights panel                             |
| `kpis`                 | 1 / KPI key, value as JSON    | top-level KPI cards                        |

Both files live under `data/output/` and are gitignored.
