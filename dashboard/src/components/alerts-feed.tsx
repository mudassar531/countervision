import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { asPublicUrl, type Alert } from "@/lib/analytics";

const SEVERITY_TONE: Record<NonNullable<Alert["severity"]>, string> = {
  info: "bg-sky-100 text-sky-900 border-sky-200",
  warn: "bg-amber-100 text-amber-900 border-amber-200",
  high: "bg-rose-100 text-rose-900 border-rose-200",
};

const TYPE_LABEL: Record<Alert["type"], string> = {
  watchlist: "Watchlist match",
  repeat_visitor: "Repeat visitor",
};

function formatTime(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function AlertsFeed({ alerts }: { alerts: Alert[] }) {
  if (!alerts.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Alerts</CardTitle>
          <CardDescription>
            No alerts in this window — every alert in the feed is a review
            prompt, not an identification.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const sorted = [...alerts].sort((a, b) => {
    const sa = a.severity === "high" ? 3 : a.severity === "warn" ? 2 : 1;
    const sb = b.severity === "high" ? 3 : b.severity === "warn" ? 2 : 1;
    return sb - sa;
  });

  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle className="text-lg">Alerts</CardTitle>
        <CardDescription className="max-w-prose">
          Each alert is a non-accusatory review prompt with the cosine similarity
          attached. Treat them as verification cues, not identifications.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0 space-y-2">
        {sorted.map((a) => {
          const sev = a.severity ?? "info";
          const thumb = asPublicUrl(a.thumbnail ?? null);
          return (
            <div
              key={a.id}
              className="flex items-start gap-3 rounded-lg border border-border/60 bg-card p-3"
            >
              <div className="size-12 shrink-0 overflow-hidden rounded-full border border-border bg-secondary">
                {thumb ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={encodeURI(thumb)}
                    alt={`thumbnail for ${a.person_id ?? a.id}`}
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="grid h-full w-full place-items-center text-xs text-muted-foreground">
                    n/a
                  </div>
                )}
              </div>
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge
                    variant="outline"
                    className={`border px-2 py-0.5 text-[10px] uppercase tracking-wider ${SEVERITY_TONE[sev]}`}
                  >
                    {sev}
                  </Badge>
                  <span className="text-xs font-medium text-foreground/80">
                    {TYPE_LABEL[a.type]}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {a.camera_id}
                    {a.area ? ` · ${a.area}` : ""}
                  </span>
                  <span className="ml-auto text-xs tabular-nums text-muted-foreground">
                    {formatTime(a.timestamp)}
                  </span>
                </div>
                <p className="text-sm leading-snug text-foreground/90">
                  {a.copy}
                </p>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  {a.person_id && (
                    <span>person <code className="font-mono text-foreground/80">{a.person_id}</code></span>
                  )}
                  {typeof a.similarity === "number" && (
                    <span>similarity <span className="tabular-nums font-medium">{a.similarity.toFixed(2)}</span></span>
                  )}
                  {a.confidence_note && (
                    <span className="italic">{a.confidence_note}</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
