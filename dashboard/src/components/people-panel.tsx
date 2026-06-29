"use client";

import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ConfidencePill } from "@/components/kpi";
import { asPublicUrl, type Confidence, type CrossCamera, type Visitor } from "@/lib/analytics";

/**
 * People panel — the core per-person story, surfaced from the already-computed
 * `visitors[]` records in analytics.json. One row per unique face identified in
 * the window. Nothing is recomputed here; this is a presentation of Phase 3's
 * authoritative numbers (dwell merged across fragmented tracker ids).
 *
 * Cross-camera linking is deliberately de-prioritised — the footage is not
 * time-synced, so each row's area is its own camera's area, with at most a
 * subtle, hedged "also face-matched in" chip for the few records Phase 4
 * linked across cameras (never framed as a continuous journey).
 */

type SortKey = "dwell" | "visits" | "faces" | "person";
type SortDir = "asc" | "desc";

/**
 * Identity-evidence tier for the confidence pill. Derived purely from the
 * number of quality-gated face detections behind this person record — more
 * face evidence ⇒ a more robust identity. Thresholds are a presentation
 * classification (documented here, surfaced in the row's title text), not a
 * fabricated metric: the underlying count is `face_appearances` from Phase 3.
 */
function evidenceLevel(faceAppearances: number | null): Confidence {
  const f = faceAppearances ?? 0;
  if (f >= 50) return "high";
  if (f >= 10) return "medium";
  return "low";
}

/**
 * Label for the evidence pill. Deliberately worded as "evidence" (not
 * "confidence") so the colour can never be read as endorsing the watchlist
 * flag or the dwell number on the same row — it speaks only to how much face
 * evidence backs this identity. Reuses the dashboard's existing pill colours.
 */
const EVIDENCE_LABEL: Record<Confidence, string> = {
  high: "Strong evidence",
  medium: "Moderate evidence",
  low: "Limited evidence",
};

/** Format authoritative track-dwell seconds for display. */
function formatDwell(seconds: number | null): string {
  if (seconds == null || seconds < 0.5) return "—";
  const total = Math.round(seconds); // round first so 119.6s → 2:00, not 1:60
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** "1 face" / "N faces". */
function faceLabel(n: number): string {
  return `${n} ${n === 1 ? "face" : "faces"}`;
}

/**
 * Map "<camera_id>/<person_id>" → other areas the same face was matched in,
 * from Phase 4's store-wide groups. Used only for the subtle hedged chip.
 */
function buildExtraAreas(crossCamera: CrossCamera | null): Map<string, string[]> {
  const map = new Map<string, string[]>();
  if (!crossCamera) return map;
  for (const group of crossCamera.store_wide_persons ?? []) {
    const areas = group.areas_visited ?? [];
    if (areas.length < 2) continue;
    for (const m of group.members ?? []) {
      const others = areas.filter((a) => a !== m.area);
      if (others.length) map.set(`${m.camera_id}/${m.person_id}`, others);
    }
  }
  return map;
}

export function PeoplePanel({
  visitors,
  crossCamera,
}: {
  visitors: Visitor[];
  crossCamera?: CrossCamera | null;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("dwell");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const extraAreas = useMemo(() => buildExtraAreas(crossCamera ?? null), [crossCamera]);

  const rows = useMemo(() => {
    const val = (v: Visitor): number | string => {
      switch (sortKey) {
        case "dwell":
          return v.track_dwell_seconds ?? 0;
        case "visits":
          return v.visit_count ?? 0;
        case "faces":
          return v.face_appearances ?? 0;
        case "person":
          return `${v.camera_id}/${v.person_id}`;
      }
    };
    return [...visitors].sort((a, b) => {
      const av = val(a);
      const bv = val(b);
      let cmp: number;
      if (typeof av === "number" && typeof bv === "number") cmp = av - bv;
      else cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [visitors, sortKey, sortDir]);

  const summary = useMemo(() => {
    const longest = visitors.reduce<Visitor | null>(
      (best, v) =>
        (v.track_dwell_seconds ?? 0) > (best?.track_dwell_seconds ?? -1) ? v : best,
      null,
    );
    return {
      total: visitors.length,
      repeats: visitors.filter((v) => v.is_repeat).length,
      watchlist: visitors.filter((v) => v.watchlist_match).length,
      longest,
      storeWide: crossCamera?.headline?.store_wide_unique_visitors ?? null,
    };
  }, [visitors, crossCamera]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Numeric columns are most useful highest-first; the id column ascending.
      setSortDir(key === "person" ? "asc" : "desc");
    }
  }

  if (!visitors.length) {
    return (
      <Card className="border-border/60">
        <CardHeader>
          <CardTitle className="text-lg">People in store</CardTitle>
          <CardDescription>
            No face records in this window — run the identity phase first.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle className="text-lg">People in store — time on camera</CardTitle>
        <CardDescription className="max-w-prose">
          Every unique face identified in this window, with authoritative
          time-on-camera. Dwell is the union of frames where any linked tracker
          id was alive, so it captures the full visit even when a track
          fragments through occlusion. Sorted longest-first — click a column to
          re-sort.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0 space-y-5">
        {/* Headline summary strip */}
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <SummaryStat
            label="People identified"
            value={`${summary.total}`}
            sub={
              summary.storeWide != null
                ? `${summary.storeWide} store-wide after face de-dup`
                : "unique faces this window"
            }
          />
          <SummaryStat
            label="Longest on camera"
            value={formatDwell(summary.longest?.track_dwell_seconds ?? null)}
            sub={
              summary.longest
                ? `${summary.longest.person_id} · ${summary.longest.area}`
                : undefined
            }
          />
          <SummaryStat
            label="Repeat visitors"
            value={`${summary.repeats}`}
            sub="seen across ≥2 visit segments"
          />
          <SummaryStat
            label="Watchlist reviews"
            value={`${summary.watchlist}`}
            sub="verification prompts, not IDs"
          />
        </div>

        <div className="rounded-xl border border-border/60 overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-secondary/60 hover:bg-secondary/60">
                <TableHead className="pl-3">Person</TableHead>
                <SortableHead
                  label="Time on camera"
                  active={sortKey === "dwell"}
                  dir={sortDir}
                  onClick={() => toggleSort("dwell")}
                  className="text-right"
                />
                <TableHead>Area seen in</TableHead>
                <SortableHead
                  label="Visits"
                  active={sortKey === "visits"}
                  dir={sortDir}
                  onClick={() => toggleSort("visits")}
                  className="text-right"
                />
                <TableHead>Flags</TableHead>
                <SortableHead
                  label="Identity evidence"
                  active={sortKey === "faces"}
                  dir={sortDir}
                  onClick={() => toggleSort("faces")}
                  title="Sort by face-detection evidence"
                />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((v) => {
                const thumb = asPublicUrl(v.thumbnail);
                const dwell = v.track_dwell_seconds ?? 0;
                const unlinked = dwell < 0.5; // face seen but no sustained body track to merge
                const level = evidenceLevel(v.face_appearances);
                const others = extraAreas.get(`${v.camera_id}/${v.person_id}`) ?? [];
                return (
                  <TableRow key={`${v.camera_id}-${v.person_id}`}>
                    {/* Person: thumbnail + id + camera */}
                    <TableCell className="pl-3">
                      <div className="flex items-center gap-3">
                        <div className="size-10 shrink-0 overflow-hidden rounded-full border border-border bg-secondary">
                          {thumb ? (
                            /* eslint-disable-next-line @next/next/no-img-element */
                            <img
                              src={encodeURI(thumb)}
                              alt={`face thumbnail for ${v.person_id}`}
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="grid h-full w-full place-items-center text-[10px] text-muted-foreground">
                              n/a
                            </div>
                          )}
                        </div>
                        <div className="min-w-0">
                          <code className="font-mono text-sm font-medium text-foreground">
                            {v.person_id}
                          </code>
                          <div className="text-[11px] text-muted-foreground">
                            {v.camera_id}
                          </div>
                        </div>
                      </div>
                    </TableCell>

                    {/* Time on camera — the headline metric */}
                    <TableCell className="text-right align-middle">
                      {unlinked ? (
                        <span
                          className="text-sm text-muted-foreground"
                          title={`Face detected ${v.face_appearances ?? 0}× but not linked to a sustained body track, so a merged time-on-camera could not be computed.`}
                        >
                          —
                        </span>
                      ) : (
                        <span className="text-base font-semibold tabular-nums text-foreground">
                          {formatDwell(dwell)}
                        </span>
                      )}
                      <div className="text-[11px] tabular-nums text-muted-foreground">
                        {faceLabel(v.face_appearances ?? 0)}
                        {v.linked_tracker_ids.length > 1
                          ? ` · ${v.linked_tracker_ids.length} merged ids`
                          : ""}
                      </div>
                    </TableCell>

                    {/* Area seen in (+ subtle hedged cross-camera chip) */}
                    <TableCell className="max-w-[16rem] whitespace-normal">
                      <span className="text-sm text-foreground/90">{v.area}</span>
                      {others.length > 0 && (
                        <span
                          className="ml-2 inline-flex items-center rounded-full border border-border bg-secondary/70 px-2 py-0.5 text-[10px] font-medium text-muted-foreground align-middle"
                          title={`Same face also matched in ${others.join(
                            ", ",
                          )} (Phase 4 cross-camera face match). Recording windows don't overlap — repeat presence, not a continuous journey.`}
                        >
                          ↔ also matched in {others.length} area
                          {others.length > 1 ? "s" : ""}
                        </span>
                      )}
                    </TableCell>

                    {/* Visits */}
                    <TableCell className="text-right tabular-nums">
                      <span className="text-sm font-medium text-foreground/90">
                        {v.visit_count ?? 1}
                      </span>
                    </TableCell>

                    {/* Flags */}
                    <TableCell>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {v.is_repeat && (
                          <Badge
                            variant="secondary"
                            className="text-[10px] py-0"
                            title={`Seen across ${v.visit_count ?? 2} separate visit segments in this camera.`}
                          >
                            repeat visitor
                          </Badge>
                        )}
                        {v.watchlist_match && (
                          <Badge
                            variant="outline"
                            className="border-amber-300 bg-amber-50 text-amber-900 text-[10px] py-0"
                            title={`Possible match with watchlist entry '${
                              v.watchlist_match
                            }'${
                              typeof v.watchlist_similarity === "number"
                                ? ` (similarity ${v.watchlist_similarity.toFixed(2)})`
                                : ""
                            } — a review prompt, not an identification. Please verify before acting.`}
                          >
                            watchlist review
                          </Badge>
                        )}
                        {!v.is_repeat && !v.watchlist_match && (
                          <span className="text-[11px] text-muted-foreground">—</span>
                        )}
                      </div>
                    </TableCell>

                    {/* Identity evidence — how much face evidence backs this id */}
                    <TableCell>
                      <ConfidencePill
                        level={level}
                        label={EVIDENCE_LABEL[level]}
                        className="whitespace-nowrap"
                      />
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>

        <p className="text-[11px] italic leading-snug text-muted-foreground/90">
          Identity evidence is the number of quality-gated face detections
          behind each record (strong ≥ 50, moderate ≥ 10, otherwise limited) —
          it speaks only to how robust the face match is, and never endorses the
          watchlist flag or the time value on the same row. “—” time-on-camera
          means a face was detected but never linked to a sustained body track,
          so no merged dwell could be computed. Watchlist flags are
          non-accusatory verification prompts.
        </p>
      </CardContent>
    </Card>
  );
}

function SortableHead({
  label,
  active,
  dir,
  onClick,
  className,
  title,
}: {
  label: string;
  active: boolean;
  dir: SortDir;
  onClick: () => void;
  className?: string;
  title?: string;
}) {
  return (
    <TableHead
      className={className}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={onClick}
        title={title}
        className={`inline-flex items-center gap-1 font-medium transition-colors hover:text-foreground ${
          active ? "text-foreground" : "text-muted-foreground"
        }`}
      >
        {label}
        <span className="text-[10px] leading-none">
          {active ? (dir === "asc" ? "↑" : "↓") : "↕"}
        </span>
      </button>
    </TableHead>
  );
}

function SummaryStat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-card p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-foreground">
        {value}
      </div>
      {sub && <div className="mt-0.5 text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}
