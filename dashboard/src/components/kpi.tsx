import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Confidence } from "@/lib/analytics";

type ConfidenceLevel = Confidence | "locked";

const TONE: Record<ConfidenceLevel, string> = {
  high: "bg-emerald-100 text-emerald-900 border-emerald-200/80",
  medium: "bg-amber-100 text-amber-900 border-amber-200/80",
  low: "bg-rose-100 text-rose-900 border-rose-200/80",
  locked: "bg-slate-200 text-slate-700 border-slate-300/80",
};

const LABEL: Record<ConfidenceLevel, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Hedged",
  locked: "Locked — needs integration",
};

export function ConfidencePill({
  level,
  className,
  label,
}: {
  level: ConfidenceLevel;
  className?: string;
  /** Override the default tier label while keeping the tier's colour/shape. */
  label?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wide",
        TONE[level],
        className,
      )}
    >
      <span className="size-1.5 rounded-full bg-current opacity-80" />
      {label ?? LABEL[level]}
    </span>
  );
}

export function KpiCard({
  label,
  value,
  unit,
  hint,
  level,
  footnote,
  accent,
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  level: ConfidenceLevel;
  footnote?: string;
  accent?: boolean;
}) {
  return (
    <Card
      className={cn(
        "relative overflow-hidden border-border/60 transition-shadow hover:shadow-md",
        accent && "bg-[var(--navy)] text-[var(--navy-fg)] border-[var(--navy)]",
      )}
    >
      {accent && (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-30 mix-blend-screen"
          style={{
            background:
              "radial-gradient(circle at 100% 0%, oklch(0.65 0.18 270 / 0.6), transparent 60%)",
          }}
        />
      )}
      <CardHeader className="pb-2 relative">
        <div className="flex items-center justify-between gap-2">
          <CardDescription
            className={cn(
              "text-xs uppercase tracking-wider font-medium",
              accent ? "text-[var(--navy-fg)]/80" : "text-muted-foreground",
            )}
          >
            {label}
          </CardDescription>
          <ConfidencePill level={level} />
        </div>
      </CardHeader>
      <CardContent className="relative pt-0 space-y-1">
        <div className="flex items-baseline gap-1.5">
          <CardTitle
            className={cn(
              "text-4xl font-semibold tabular-nums leading-none",
              accent && "text-[var(--navy-fg)]",
            )}
          >
            {value}
          </CardTitle>
          {unit && (
            <span
              className={cn(
                "text-base font-medium",
                accent ? "text-[var(--navy-fg)]/80" : "text-muted-foreground",
              )}
            >
              {unit}
            </span>
          )}
        </div>
        {hint && (
          <p
            className={cn(
              "text-xs",
              accent ? "text-[var(--navy-fg)]/75" : "text-muted-foreground",
            )}
          >
            {hint}
          </p>
        )}
        {footnote && (
          <p className="text-[11px] italic leading-snug pt-2 text-muted-foreground/90">
            {footnote}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function LockedKpiCard({
  label,
  reason,
  cta = "Unlock with integration",
}: {
  label: string;
  reason: string;
  cta?: string;
}) {
  return (
    <Card className="border-dashed border-border bg-card/60">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardDescription className="text-xs uppercase tracking-wider font-medium text-muted-foreground">
            {label}
          </CardDescription>
          <ConfidencePill level="locked" />
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-1.5">
        <p className="text-sm text-muted-foreground leading-snug">{reason}</p>
        <Badge
          variant="secondary"
          className="font-medium text-[var(--navy)] bg-[var(--navy-soft)] border border-[var(--navy)]/15"
        >
          {cta}
        </Badge>
      </CardContent>
    </Card>
  );
}
