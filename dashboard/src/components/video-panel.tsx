"use client";

import { useState } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type CameraVideo = {
  camera_id: string;
  area: string;
  src: string | null;
  poster: string | null;
};

export function AnnotatedVideoPanel({ cameras }: { cameras: CameraVideo[] }) {
  const [tab, setTab] = useState(cameras[0]?.camera_id ?? "");
  if (!cameras.length) return null;

  return (
    <Card className="overflow-hidden border-border/60">
      <CardHeader>
        <CardTitle className="text-lg">Annotated walkthrough</CardTitle>
        <CardDescription className="max-w-prose">
          The same window the analytics were computed on — boxes, IDs and short
          motion traces are burned in by the pipeline so this plays without any
          live inference.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="bg-secondary rounded-full p-1">
            {cameras.map((c) => (
              <TabsTrigger
                key={c.camera_id}
                value={c.camera_id}
                className="rounded-full px-3 py-1.5 text-xs font-medium data-[state=active]:bg-[var(--navy)] data-[state=active]:text-[var(--navy-fg)]"
              >
                <span className="font-semibold">{c.camera_id}</span>
                <span className="ml-2 hidden sm:inline opacity-70">{c.area}</span>
              </TabsTrigger>
            ))}
          </TabsList>
          {cameras.map((c) => (
            <TabsContent key={c.camera_id} value={c.camera_id} className="mt-4">
              <div className="overflow-hidden rounded-xl border border-border/60 bg-black aspect-video">
                {c.src ? (
                  <video
                    key={c.src}
                    src={encodeURI(c.src)}
                    poster={c.poster ? encodeURI(c.poster) : undefined}
                    controls
                    playsInline
                    preload="metadata"
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="grid h-full w-full place-items-center text-sm text-white/60">
                    no annotated mp4 found for {c.camera_id}
                  </div>
                )}
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}
