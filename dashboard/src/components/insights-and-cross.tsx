import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ConfidencePill } from "@/components/kpi";
import type { Insight, CrossCamera } from "@/lib/analytics";

export function InsightsPanel({ insights }: { insights: Insight[] }) {
  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle className="text-lg">Plain-English insights</CardTitle>
        <CardDescription className="max-w-prose">
          Generated only from reliable per-area numbers (dwell, occupancy,
          unique faces). Never built on a cross-camera link or a near-zero
          footfall reading.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {insights.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Not enough reliable data in this window to surface an insight.
          </p>
        ) : (
          insights.map((i) => (
            <div
              key={i.id}
              className="rounded-lg border border-border/60 bg-card p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <h3 className="text-sm font-semibold text-foreground">{i.title}</h3>
                <ConfidencePill level={i.confidence} />
              </div>
              <p className="mt-1.5 text-sm leading-relaxed text-foreground/85">
                {i.detail}
              </p>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

export function CrossCameraPanel({ data }: { data: CrossCamera | null }) {
  if (!data) {
    return (
      <Card className="border-dashed border-border bg-card/60">
        <CardHeader>
          <CardTitle className="text-lg">Cross-camera presence</CardTitle>
          <CardDescription>
            Phase 4 hasn&apos;t been run yet — store-wide de-dup unavailable.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const { headline, links, store_wide_persons: storeWide, thresholds } = data;
  const noReliable = headline.no_reliable_cross_camera_matches;

  return (
    <Card className="border-border/60">
      <CardHeader>
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="space-y-1">
            <CardTitle className="text-lg">
              Cross-camera presence — repeat appearances
            </CardTitle>
            <CardDescription className="max-w-3xl">
              <strong className="font-medium text-foreground">{data.render_hint}</strong>
            </CardDescription>
          </div>
          <ConfidencePill level={noReliable ? "low" : "medium"} />
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        <div className="grid sm:grid-cols-3 gap-3 text-sm">
          <Stat
            label="Store-wide unique"
            value={`${headline.store_wide_unique_visitors}`}
            sub={`from naive sum ${headline.naive_total_per_camera_sum}`}
          />
          <Stat
            label="Cross-camera links"
            value={`${headline.cross_camera_links_count}`}
            sub={`cosine ≥ ${thresholds.cross_camera_match}`}
          />
          <Stat
            label="Visitors deduped"
            value={`${headline.saved_by_cross_camera_dedup}`}
            sub={noReliable ? "no reliable matches" : "saved via face dedup"}
          />
        </div>
        {noReliable ? (
          <p className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            {headline.headline_message}
          </p>
        ) : (
          <div className="space-y-2">
            <h4 className="text-xs uppercase tracking-wider text-muted-foreground">
              Repeat-presence pairs
            </h4>
            {links.map((l, idx) => (
              <div
                key={`${l.from.camera_id}-${l.from.person_id}-${l.to.camera_id}-${l.to.person_id}-${idx}`}
                className="rounded-lg border border-border/60 bg-card p-3 text-sm"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-semibold text-foreground">
                    {l.from.camera_id}/{l.from.person_id}
                  </span>
                  <span className="text-muted-foreground">{l.from.area}</span>
                  <span className="text-muted-foreground">↔</span>
                  <span className="font-semibold text-foreground">
                    {l.to.camera_id}/{l.to.person_id}
                  </span>
                  <span className="text-muted-foreground">{l.to.area}</span>
                  <span className="ml-auto rounded-full bg-[var(--navy)] px-2 py-0.5 text-[11px] font-medium text-[var(--navy-fg)] tabular-nums">
                    sim {l.similarity.toFixed(2)}
                  </span>
                  <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium tabular-nums">
                    gap {l.time_gap}
                  </span>
                </div>
                <p className="mt-2 text-xs leading-snug text-muted-foreground italic">
                  {l.presence_note}
                </p>
              </div>
            ))}
            {storeWide.length > 0 && (
              <div className="pt-2">
                <h4 className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                  Store-wide visitors (connected components)
                </h4>
                <ul className="text-xs space-y-1">
                  {storeWide.map((sw) => (
                    <li
                      key={sw.store_person_id}
                      className="flex items-center gap-2 flex-wrap"
                    >
                      <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-foreground/80">
                        {sw.store_person_id}
                      </code>
                      <span className="text-muted-foreground">
                        {sw.members
                          .map((m) => `${m.camera_id}/${m.person_id}`)
                          .join(" + ")}
                      </span>
                      <span className="ml-auto text-muted-foreground">
                        {sw.areas_visited.join(" + ")}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border/60 bg-card p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-foreground tabular-nums">
        {value}
      </div>
      {sub && (
        <div className="text-[11px] text-muted-foreground mt-0.5">{sub}</div>
      )}
    </div>
  );
}
