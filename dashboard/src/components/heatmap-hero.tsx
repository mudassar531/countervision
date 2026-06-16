"use client";

import { useState } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { asPublicUrl, type Area } from "@/lib/analytics";

type ImageKind = "heatmap" | "frame";

export function AreaHeatmapHero({ areas }: { areas: Area[] }) {
  const [tab, setTab] = useState(areas[0]?.camera_id ?? "");
  const [kind, setKind] = useState<ImageKind>("heatmap");

  if (!areas.length) {
    return (
      <Card>
        <CardContent className="p-8 text-sm text-muted-foreground">
          No areas to render — run the pipeline first.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden border-border/60">
      <CardHeader className="flex-row items-center justify-between gap-4">
        <div className="space-y-1">
          <CardTitle className="text-lg">Where people spend time</CardTitle>
          <CardDescription className="max-w-prose">
            Per-area heatmap from gaussian density over the bottom-center of every
            tracked person. Hot spots are where people actually stand and dwell.
          </CardDescription>
        </div>
        <div className="flex items-center gap-1 rounded-full border border-border/60 bg-secondary p-1 text-xs font-medium">
          <button
            type="button"
            onClick={() => setKind("heatmap")}
            className={`rounded-full px-3 py-1 transition ${
              kind === "heatmap"
                ? "bg-[var(--navy)] text-[var(--navy-fg)]"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Heatmap
          </button>
          <button
            type="button"
            onClick={() => setKind("frame")}
            className={`rounded-full px-3 py-1 transition ${
              kind === "frame"
                ? "bg-[var(--navy)] text-[var(--navy-fg)]"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Clean frame
          </button>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="bg-secondary rounded-full p-1">
            {areas.map((a) => (
              <TabsTrigger
                key={a.camera_id}
                value={a.camera_id}
                className="rounded-full px-3 py-1.5 text-xs font-medium data-[state=active]:bg-[var(--navy)] data-[state=active]:text-[var(--navy-fg)]"
              >
                <span className="font-semibold">{a.camera_id}</span>
                <span className="ml-2 hidden sm:inline opacity-70">{a.area}</span>
              </TabsTrigger>
            ))}
          </TabsList>
          {areas.map((a) => {
            const heat = asPublicUrl(a.heatmap_png);
            const frame = asPublicUrl(a.frame_jpg);
            const src = kind === "heatmap" ? heat ?? frame : frame ?? heat;
            return (
              <TabsContent key={a.camera_id} value={a.camera_id} className="mt-4">
                <div className="overflow-hidden rounded-xl border border-border/60 bg-black">
                  {src ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={encodeURI(src)}
                      alt={`${kind} for ${a.area}`}
                      className="aspect-video w-full object-cover"
                    />
                  ) : (
                    <div className="aspect-video w-full grid place-items-center text-sm text-white/60">
                      no image
                    </div>
                  )}
                </div>
                <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <Stat label="Area" value={a.area} />
                  <Stat label="Unique faces" value={`${a.unique_visitors}`} />
                  <Stat
                    label="Avg dwell"
                    value={`${a.avg_dwell_seconds.toFixed(0)}s`}
                  />
                  <Stat label="Peak occupancy" value={`${a.occupancy_peak}`} />
                </div>
              </TabsContent>
            );
          })}
        </Tabs>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/60 bg-card px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="text-sm font-semibold text-foreground">{value}</div>
    </div>
  );
}
