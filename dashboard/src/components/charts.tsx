"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const NAVY = "#0a1347";
const NAVY_TINT = "#3b46a8";
const AXIS = "#64748b";
const GRID = "#e2e8f0";

const tooltipStyle = {
  backgroundColor: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  fontSize: 12,
  color: NAVY,
  boxShadow: "0 6px 24px rgba(10,19,71,0.08)",
};

const legendStyle = {
  paddingTop: 6,
  fontSize: 11,
  color: AXIS,
};

export function FootfallByHourChart({
  data,
}: {
  data: Array<{ hour: string; in: number; out: number; total: number }>;
}) {
  if (!data.length) {
    return (
      <div className="flex h-full min-h-[260px] flex-col items-center justify-center text-center text-sm text-muted-foreground">
        <p>No line-crossing events in this window.</p>
        <p className="text-xs mt-1">
          Redraw the entry line per scene to capture footfall accurately.
        </p>
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="hour" stroke={AXIS} tickLine={false} axisLine={false} fontSize={12} />
        <YAxis
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          fontSize={12}
          width={30}
          allowDecimals={false}
        />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: NAVY, fillOpacity: 0.05 }} />
        <Legend wrapperStyle={legendStyle} iconType="circle" />
        <Bar dataKey="in" name="In" fill={NAVY} radius={[6, 6, 0, 0]} />
        <Bar dataKey="out" name="Out" fill={NAVY_TINT} radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DwellByAreaChart({
  data,
}: {
  data: Array<{ area: string; avg: number; max: number }>;
}) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="area"
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          fontSize={11}
          interval={0}
          tickFormatter={(v: string) => (v.length > 18 ? v.slice(0, 17) + "…" : v)}
        />
        <YAxis stroke={AXIS} tickLine={false} axisLine={false} fontSize={12} width={36} unit="s" />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: NAVY, fillOpacity: 0.05 }} />
        <Legend wrapperStyle={legendStyle} iconType="circle" />
        <Bar dataKey="avg" name="Avg dwell (s)" fill={NAVY} radius={[6, 6, 0, 0]} />
        <Bar dataKey="max" name="Max dwell (s)" fill={NAVY_TINT} radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function OccupancyLineChart({
  data,
}: {
  data: Array<{ second_bucket: number; t: string; active_tracks: number; area: string }>;
}) {
  if (!data.length) {
    return (
      <div className="flex h-[260px] items-center justify-center text-sm text-muted-foreground">
        No occupancy data.
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="second_bucket"
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          fontSize={12}
          unit="s"
        />
        <YAxis
          stroke={AXIS}
          tickLine={false}
          axisLine={false}
          fontSize={12}
          width={30}
          allowDecimals={false}
        />
        <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: NAVY, strokeOpacity: 0.15 }} />
        <Line
          type="monotone"
          dataKey="active_tracks"
          name="People in frame"
          stroke={NAVY}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: NAVY }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
