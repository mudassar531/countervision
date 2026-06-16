"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { OccupancyLineChart } from "@/components/charts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { asPublicUrl, type Area, type Visitor } from "@/lib/analytics";

export function AreasAndVisitorsPanel({
  areas,
  visitors,
}: {
  areas: Area[];
  visitors: Visitor[];
}) {
  const [tab, setTab] = useState(areas[0]?.camera_id ?? "");
  if (!areas.length) return null;

  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle className="text-lg">Per-area detail</CardTitle>
        <CardDescription className="max-w-prose">
          Occupancy timeline and the unique faces identified inside each area —
          dwell here is the union of frames where any linked Phase-1 tracker id
          was alive, so it captures the full visit length even when an id
          fragments through occlusion.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="bg-secondary rounded-full p-1 flex-wrap h-auto">
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
            const occupancy = (a.occupancy_timeseries ?? []).map((pt) => ({
              ...pt,
              area: a.area,
            }));
            const areaVisitors = visitors.filter((v) => v.camera_id === a.camera_id);
            return (
              <TabsContent key={a.camera_id} value={a.camera_id} className="mt-4 space-y-5">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold text-foreground">
                      Occupancy timeline
                    </h4>
                    <span className="text-xs text-muted-foreground">
                      peak {a.occupancy_peak} simultaneous
                    </span>
                  </div>
                  <OccupancyLineChart data={occupancy} />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold text-foreground">
                      Visitors seen in this area
                    </h4>
                    <span className="text-xs text-muted-foreground">
                      {areaVisitors.length} unique face(s)
                    </span>
                  </div>
                  <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {areaVisitors.map((v) => (
                      <VisitorCard key={`${v.camera_id}-${v.person_id}`} v={v} />
                    ))}
                  </div>
                </div>
              </TabsContent>
            );
          })}
        </Tabs>
      </CardContent>
    </Card>
  );
}

function VisitorCard({ v }: { v: Visitor }) {
  const thumb = asPublicUrl(v.thumbnail);
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border/60 bg-card p-3">
      <div className="size-12 shrink-0 overflow-hidden rounded-full border border-border bg-secondary">
        {thumb ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={encodeURI(thumb)}
            alt={`thumbnail for ${v.person_id}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="grid h-full w-full place-items-center text-[11px] text-muted-foreground">
            n/a
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-xs">
          <code className="font-mono text-foreground/85 font-medium">
            {v.person_id}
          </code>
          {v.is_repeat && (
            <Badge variant="secondary" className="text-[10px] py-0">
              repeat
            </Badge>
          )}
          {v.watchlist_match && (
            <Badge variant="outline" className="text-[10px] py-0 border-amber-300 text-amber-900 bg-amber-50">
              watchlist
            </Badge>
          )}
        </div>
        <div className="text-[11px] text-muted-foreground mt-0.5">
          dwell{" "}
          <span className="tabular-nums font-medium text-foreground/85">
            {Math.round(v.track_dwell_seconds ?? 0)}s
          </span>{" "}
          · faces{" "}
          <span className="tabular-nums">{v.face_appearances ?? 0}</span>
          {v.linked_tracker_ids.length > 1 && (
            <>
              {" "}
              · {v.linked_tracker_ids.length} merged ids
            </>
          )}
        </div>
      </div>
    </div>
  );
}
