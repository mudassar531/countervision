import { promises as fs } from "node:fs";
import path from "node:path";

import { AlertsFeed } from "@/components/alerts-feed";
import { AreasAndVisitorsPanel } from "@/components/area-detail";
import { DwellByAreaChart, FootfallByHourChart } from "@/components/charts";
import { AreaHeatmapHero } from "@/components/heatmap-hero";
import { CrossCameraPanel, InsightsPanel } from "@/components/insights-and-cross";
import { ConfidencePill, KpiCard, LockedKpiCard } from "@/components/kpi";
import { PeoplePanel } from "@/components/people-panel";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { AnnotatedVideoPanel, type CameraVideo } from "@/components/video-panel";
import type { Analytics, Confidence } from "@/lib/analytics";
import { asPublicUrl } from "@/lib/analytics";

async function loadAnalytics(): Promise<Analytics | null> {
  const file = path.join(process.cwd(), "public", "data", "analytics.json");
  try {
    const raw = await fs.readFile(file, "utf-8");
    return JSON.parse(raw) as Analytics;
  } catch (err) {
    console.error("[dashboard] analytics.json not found at", file, err);
    return null;
  }
}

function formatRange(start?: string | null, end?: string | null) {
  if (!start || !end) return "no window data";
  try {
    const s = new Date(start);
    const e = new Date(end);
    const sameDay = s.toDateString() === e.toDateString();
    const sStr = s.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
    if (sameDay) {
      const eTime = e.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return `${sStr} – ${eTime}`;
    }
    const eStr = e.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
    return `${sStr} – ${eStr}`;
  } catch {
    return `${start} – ${end}`;
  }
}

function asConfidence(c: string | undefined): Confidence {
  return c === "high" || c === "medium" || c === "low" ? c : "low";
}

export default async function Home() {
  const a = await loadAnalytics();
  if (!a) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-24">
        <h1 className="text-2xl font-semibold">analytics.json not found</h1>
        <p className="text-muted-foreground mt-2">
          Run <code className="font-mono">python pipeline/main.py --run-aggregate</code>
          {" "}and then <code className="font-mono">npm run dev</code> again.
        </p>
      </main>
    );
  }

  const cameraIds = a.store.cameras;
  const areaByCamera = new Map(a.areas.map((x) => [x.camera_id, x.area]));
  const frameByCamera = new Map(a.areas.map((x) => [x.camera_id, x.frame_jpg]));
  const videoCameras: CameraVideo[] = cameraIds.map((cam) => ({
    camera_id: cam,
    area: areaByCamera.get(cam) ?? cam,
    src: asPublicUrl(`data/output/annotated/${cam}.mp4`),
    poster: asPublicUrl(frameByCamera.get(cam) ?? null),
  }));
  const k = a.kpis;
  const storeWideConf: Confidence = k.store_wide_unique_visitors.confidence ?? "low";

  const dwellChartData = a.areas.map((x) => ({
    area: x.area,
    avg: Math.round(x.avg_dwell_seconds),
    max: Math.round(x.max_dwell_seconds),
  }));

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* -------------------- 1. Branded Overview -------------------- */}
      <header className="relative overflow-hidden border-b border-border bg-[var(--navy)] text-[var(--navy-fg)]">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-50"
          style={{
            background:
              "radial-gradient(60% 70% at 80% 0%, oklch(0.55 0.18 270 / 0.45), transparent 70%)," +
              "radial-gradient(40% 50% at 0% 100%, oklch(0.35 0.14 230 / 0.45), transparent 70%)",
          }}
        />
        <div className="relative mx-auto max-w-7xl px-6 py-10 flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <div className="space-y-3 max-w-2xl">
            <div className="flex items-center gap-2">
              <div className="flex size-8 items-center justify-center rounded-md bg-[var(--navy-fg)] text-[var(--navy)] text-sm font-bold">
                CV
              </div>
              <span className="text-xs uppercase tracking-[0.2em] text-[var(--navy-fg)]/75">
                CounterVision · Agents Limited
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-semibold leading-tight">
              {a.store.name}
            </h1>
            <p className="text-[var(--navy-fg)]/80 text-sm md:text-base max-w-prose">
              Multi-camera CCTV turned into actionable retail analytics —
              footfall, dwell, occupancy, identity and watchlist, all from
              standard camera feeds with no extra hardware.
            </p>
          </div>
          <div className="rounded-xl border border-[var(--navy-fg)]/15 bg-white/5 px-4 py-3 text-xs space-y-1 min-w-fit backdrop-blur">
            <div className="text-[var(--navy-fg)]/70 uppercase tracking-wider">
              Captured window
            </div>
            <div className="font-medium text-[var(--navy-fg)]">
              {formatRange(a.store.window.start, a.store.window.end)}
            </div>
            <Separator className="bg-[var(--navy-fg)]/15 my-1.5" />
            <div className="flex items-center gap-3 text-[var(--navy-fg)]/85">
              <span>{cameraIds.length} cameras</span>
              <span>·</span>
              <span>{a.areas.length} areas</span>
              <span>·</span>
              <span>{a.visitors.length} face records</span>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-8 space-y-10">
        {/* -------------------- 2. KPI cards -------------------- */}
        <section aria-label="Key performance indicators" className="space-y-4">
          <SectionTitle
            eyebrow="Headline KPIs"
            title="The reliable numbers, up front"
            description="Lead with the metrics the pipeline can stand behind: per-area unique faces, authoritative dwell, occupancy. Hedged or locked KPIs sit alongside, labelled."
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard
              label="Avg dwell — store"
              value={k.avg_dwell_seconds_store.value.toFixed(1)}
              unit="s"
              level="high"
              footnote={k.avg_dwell_seconds_store.note}
              accent
            />
            <KpiCard
              label="Store-wide unique visitors"
              value={k.store_wide_unique_visitors.value}
              level={storeWideConf}
              hint={`naive sum ${k.store_wide_unique_visitors.naive_per_camera_sum} · saved ${k.store_wide_unique_visitors.saved_by_dedup} via face dedup`}
              footnote={k.store_wide_unique_visitors.method}
              accent
            />
            <KpiCard
              label="Repeat visitors"
              value={k.repeat_visitors_per_area.value}
              level={asConfidence(k.repeat_visitors_per_area.confidence)}
              footnote={k.repeat_visitors_per_area.note}
            />
            <KpiCard
              label="Peak hour"
              value={k.peak_hour ?? "—"}
              level="medium"
              hint={`${k.active_alerts} alerts in this window`}
            />
            <KpiCard
              label="Footfall (in / out)"
              value={`${k.footfall_total.in_count} / ${k.footfall_total.out_count}`}
              level={asConfidence(k.footfall_total.confidence)}
              footnote={k.footfall_total.note}
            />
            <KpiCard
              label="Watchlist hits"
              value={k.watchlist_hits.value}
              level={asConfidence(k.watchlist_hits.confidence)}
              footnote={k.watchlist_hits.note}
            />
            <LockedKpiCard
              label="Conversion rate"
              reason={k.conversion_rate.reason}
              cta="Unlock with POS integration"
            />
            <LockedKpiCard
              label="Revenue uplift"
              reason={k.revenue_uplift.reason}
              cta="Unlock with POS integration"
            />
          </div>
        </section>

        {/* -------------------- People (core per-person story) -------------------- */}
        <section aria-label="People in store" className="space-y-4">
          <SectionTitle
            eyebrow="The core story"
            title="Who came in — and how long they stayed"
            description="Each face the pipeline identified, with its authoritative time on camera, the area it was seen in, repeat-visit count and watchlist review status. Sortable; longest time on camera first."
          />
          <PeoplePanel visitors={a.visitors} crossCamera={a.cross_camera} />
        </section>

        {/* -------------------- 3. Per-area heatmap (hero) -------------------- */}
        <section aria-label="Per-area heatmaps">
          <AreaHeatmapHero areas={a.areas} />
        </section>

        {/* -------------------- 4 & 5. Footfall + Dwell -------------------- */}
        <section className="grid gap-4 lg:grid-cols-2">
          <Card className="border-border/60">
            <CardHeader>
              <div className="flex items-start justify-between gap-2">
                <div className="space-y-1">
                  <CardTitle className="text-lg">Footfall by hour</CardTitle>
                  <CardDescription className="max-w-prose">
                    Auto-generated entry lines (75% height). Redraw per scene
                    via <code className="font-mono text-xs">--draw-zones CAM</code>
                    {" "}for production-grade counts.
                  </CardDescription>
                </div>
                <ConfidencePill level={asConfidence(k.footfall_total.confidence)} />
              </div>
            </CardHeader>
            <CardContent>
              <FootfallByHourChart data={a.footfall_by_hour} />
            </CardContent>
          </Card>
          <Card className="border-border/60">
            <CardHeader>
              <div className="flex items-start justify-between gap-2">
                <div className="space-y-1">
                  <CardTitle className="text-lg">Dwell by area</CardTitle>
                  <CardDescription className="max-w-prose">
                    Average and max per-person dwell. Per-person dwell merges
                    Phase-1 tracker-id fragmentation via face identity.
                  </CardDescription>
                </div>
                <ConfidencePill level="high" />
              </div>
            </CardHeader>
            <CardContent>
              <DwellByAreaChart data={dwellChartData} />
            </CardContent>
          </Card>
        </section>

        {/* -------------------- 6. Per-area occupancy + visitors -------------------- */}
        <section aria-label="Per-area detail">
          <AreasAndVisitorsPanel areas={a.areas} visitors={a.visitors} />
        </section>

        {/* -------------------- 7. Cross-camera (hedged, no journey paths) -------------------- */}
        <section aria-label="Cross-camera presence">
          <CrossCameraPanel data={a.cross_camera} />
        </section>

        {/* -------------------- 8. Annotated video player -------------------- */}
        <section aria-label="Annotated video">
          <AnnotatedVideoPanel cameras={videoCameras} />
        </section>

        {/* -------------------- 9. Alerts feed -------------------- */}
        <section aria-label="Alerts feed">
          <AlertsFeed alerts={a.alerts} />
        </section>

        {/* -------------------- 10. Insights -------------------- */}
        <section aria-label="Insights">
          <InsightsPanel insights={a.insights} />
        </section>

        <footer className="pt-6 pb-12 text-center text-xs text-muted-foreground">
          analytics.json schema v{a.version} · generated{" "}
          {new Date(a.generated_at).toLocaleString()} · {a.locked_fields_note}
        </footer>
      </div>
    </main>
  );
}

function SectionTitle({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="space-y-1">
      <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
        {eyebrow}
      </p>
      <h2 className="text-xl font-semibold text-foreground">{title}</h2>
      <p className="text-sm text-muted-foreground max-w-3xl">{description}</p>
    </div>
  );
}
