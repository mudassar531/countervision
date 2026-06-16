/**
 * TypeScript view of `analytics.json` — kept in sync with `docs/schema.md`.
 *
 * The schema deliberately exposes hedging on every value the pipeline
 * produced with low confidence. The dashboard renders these with caveats;
 * never as hard facts.
 */

export type Confidence = "high" | "medium" | "low";

export interface LockedKpi {
  value: null;
  locked: true;
  reason: string;
}

export interface HedgedKpi<T> {
  value: T;
  locked: false;
  confidence: Confidence;
  note?: string;
  method?: string;
  [key: string]: unknown;
}

export interface StoreWideKpi {
  value: number;
  locked: false;
  confidence: Confidence;
  method: string;
  naive_per_camera_sum: number;
  saved_by_dedup: number;
  cross_camera_links_count: number;
  no_reliable_cross_camera_matches: boolean;
}

export interface FootfallKpi {
  value: number;
  locked: false;
  confidence: Confidence;
  in_count: number;
  out_count: number;
  note: string;
}

export interface Kpis {
  store_wide_unique_visitors: StoreWideKpi;
  per_camera_unique_visitors_sum: number;
  footfall_total: FootfallKpi;
  watchlist_hits: HedgedKpi<number>;
  repeat_visitors_per_area: HedgedKpi<number>;
  avg_dwell_seconds_store: HedgedKpi<number>;
  peak_hour: string | null;
  active_alerts: number;
  conversion_rate: LockedKpi;
  revenue_uplift: LockedKpi;
  weather: LockedKpi;
  staffing_recommendations_quantified: LockedKpi;
}

export interface OccupancyPoint {
  t: string;
  frame_idx: number;
  second_bucket: number;
  active_tracks: number;
}

export interface Area {
  camera_id: string;
  area: string;
  unique_visitors: number;
  footfall_in: number;
  footfall_out: number;
  footfall_total: number;
  avg_dwell_seconds: number;
  max_dwell_seconds: number;
  occupancy_peak: number;
  frame_jpg: string | null;
  heatmap_png: string | null;
  zone_polygon: number[][] | null;
  entry_line: { start: number[]; end: number[]; anchor?: string } | null;
  occupancy_timeseries: OccupancyPoint[];
  person_tracks_note: string;
}

export interface Visitor {
  camera_id: string;
  area: string;
  person_id: string;
  first_seen: string | null;
  last_seen: string | null;
  face_appearances: number | null;
  track_dwell_seconds: number | null;
  face_dwell_seconds: number | null;
  linked_tracker_ids: number[];
  visit_count: number;
  is_repeat: boolean;
  watchlist_match: string | null;
  watchlist_similarity: number | null;
  thumbnail: string | null;
}

export interface Alert {
  id: string;
  type: "watchlist" | "repeat_visitor";
  camera_id: string;
  area?: string;
  person_id?: string;
  timestamp?: string;
  severity?: "info" | "warn" | "high";
  similarity?: number;
  visit_count?: number;
  confidence_level?: "low" | "medium";
  confidence_note?: string;
  thumbnail?: string;
  frame_jpg?: string | null;
  copy: string;
  watchlist_label?: string;
}

export interface CrossCameraLink {
  from: {
    camera_id: string;
    person_id: string;
    area: string;
    first_seen: string | null;
    last_seen: string | null;
  };
  to: {
    camera_id: string;
    person_id: string;
    area: string;
    first_seen: string | null;
    last_seen: string | null;
  };
  similarity: number;
  time_gap: string;
  presence_note: string;
}

export interface CrossCamera {
  thresholds: {
    in_camera_cluster: number;
    cross_camera_match: number;
    min_face_appearances_for_cross_camera: number;
  };
  headline: {
    store_wide_unique_visitors: number;
    naive_total_per_camera_sum: number;
    saved_by_cross_camera_dedup: number;
    cross_camera_links_count: number;
    no_reliable_cross_camera_matches: boolean;
    headline_message: string;
  };
  links: CrossCameraLink[];
  store_wide_persons: Array<{
    store_person_id: string;
    members: Array<{
      camera_id: string;
      person_id: string;
      area: string;
      first_seen: string | null;
      last_seen: string | null;
      face_appearances: number;
    }>;
    areas_visited: string[];
    first_seen_overall: string | null;
    last_seen_overall: string | null;
  }>;
  persons_skipped: Array<{ camera_id: string; person_id: string; reason: string }>;
  render_hint: string;
}

export interface Insight {
  id: string;
  title: string;
  detail: string;
  evidence?: Record<string, unknown>;
  confidence: Confidence;
}

export interface FootfallHourBucket {
  hour: string;
  in: number;
  out: number;
  total: number;
}

export interface Analytics {
  version: number;
  generated_at: string;
  store: {
    name: string;
    cameras: string[];
    window: { start: string | null; end: string | null };
  };
  kpis: Kpis;
  footfall_by_hour: FootfallHourBucket[];
  areas: Area[];
  visitors: Visitor[];
  alerts: Alert[];
  cross_camera: CrossCamera | null;
  insights: Insight[];
  locked_fields_note: string;
}

/** Map any `data/output/...` path the JSON may carry to a public URL. */
export function asPublicUrl(relPath: string | null | undefined): string | null {
  if (!relPath) return null;
  // analytics.json may carry either a relative path ("data/output/frames/x.jpg")
  // or an absolute one ("/Users/.../data/output/frames/x.jpg"). Pull out the
  // portion after the first `data/output/` either way, then prepend `/data/`.
  const marker = "data/output/";
  const idx = relPath.indexOf(marker);
  const tail = idx >= 0 ? relPath.slice(idx + marker.length) : relPath.replace(/^\/+/, "");
  return `/data/${tail}`;
}
